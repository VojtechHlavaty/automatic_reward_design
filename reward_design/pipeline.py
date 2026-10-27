"""Curriculum reward design with LLM agents on IsaacLab.

Each iteration:
1. Retrieval: a query agent writes a search query, the vector database returns similar earlier curricula, and a
   selector agent picks the most useful one (or none).
2. Planning: a planner agent samples ``num_candidates`` curricula (1-5 stages with objectives, transition flags and
   iteration budgets).
3. Stage generation: for every stage, a stage agent writes a YAML (reward terms from the registry, PPO settings,
   randomization). The YAML is validated against the schema; errors go back to the agent (up to
   ``stage_max_retries`` retries).
4. Training: the candidates train in parallel, stage after stage; each stage continues from the previous stage's
   checkpoint.
5. Evaluation: task score, original-reward score and task-flag rates from the training logs, and (with --visual) a
   vision-language model scores the stages from rollout videos of the final policy.
6. Feedback: a feedback agent analyzes every candidate; human feedback can be added in
   ``iter<i>/human_feedback.txt``. Everything is stored in the vector database for later retrieval.

Every artifact is written to a file, so an interrupted run continues where it stopped (``resume=<run folder>``).
"""

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import yaml
from omegaconf import OmegaConf

import review
from core import (
    CONFIG_FILE,
    DUMMY_FAILURE,
    ROOT_DIR,
    LLMClient,
    TrainingLauncher,
    evaluate_run,
    make_vlm_client,
    parse_training_log,
    reduce_metric,
    resume_config,
    terminate_stale_runs,
    training_finished,
)
from utils.extract_task_code import file_to_string
from utils.file_utils import load_tensorboard_logs
from rd_lib import agents
from rd_lib.derived import HELPERS_DOC
from rd_lib.derived_checker import make_checker
from rd_lib.quantities import get_adapter
from rd_lib.reward_registry import write_reward_file
from rd_lib.vdb import Embedder, VectorDB

STATE_FILE = "pipeline_state.json"


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=4)
    os.replace(tmp, path)


def resume_args(checkpoint):
    """IsaacLab train.py arguments that load a checkpoint from the run's log root."""
    return [
        "--resume",
        "--load_run", re.escape(os.path.basename(os.path.dirname(checkpoint))),
        "--checkpoint", re.escape(os.path.basename(checkpoint)),
    ]


# ---------------------------------------------------------------------------
# Training of curricula
# ---------------------------------------------------------------------------


def install_init_checkpoint(cfg):
    """Copy ``cfg.init_checkpoint`` into this run's RSL-RL log root, where IsaacLab's ``--resume`` finds it.

    Returns the path of the copy, or None when no initial policy is configured.
    """
    if not cfg.get("init_checkpoint"):
        return None
    source = Path(str(cfg.init_checkpoint)).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"init_checkpoint not found: {source}")
    experiment = cfg.env.get("experiment_name")
    if not experiment:
        raise ValueError("init_checkpoint needs env.experiment_name (the task's RSL-RL experiment name)")
    target = Path("logs") / "rsl_rl" / experiment / "init_policy" / "model_0.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    logging.info(f"Initial policy for every curriculum: {source} (copied to {target})")
    return str(target.resolve())


def stage_extra_args(cfg, spec, randomization_options):
    training = spec["training"]
    args = [f"agent.algorithm.entropy_coef={training.get('entropy_coef', cfg.default_entropy_coef)}"]
    if "learning_rate" in training:
        args.append(f"agent.algorithm.learning_rate={training['learning_rate']}")
    for key in ("gamma", "lam"):
        if key in training:
            args.append(f"agent.algorithm.{key}={training[key]}")
    if training.get("steps_per_env"):
        # longer rollouts with proportionally fewer environments: the same transitions per iteration
        steps = int(training["steps_per_env"])
        args += [f"agent.num_steps_per_env={steps}", "--num_envs", str(max(int(cfg.num_envs * 24 / steps), 64))]
    for key, value in (spec.get("randomization") or {}).items():
        args.append(f"{randomization_options[key]['override']}={value}")
    # settings of the training wrapper (isaaclab_backend/train.py)
    job = {}
    exploration = dict(training.get("exploration") or {})
    if exploration:
        names = list(cfg.env.get("action_names") or [])
        per_action = exploration.pop("per_action", None) or {}
        job["exploration"] = dict(exploration, fixed_std_per_action={
            str(names.index(name)): value for name, value in per_action.items() if name in names})
    if training.get("reset_critic"):
        job["reset_critic"] = True
    if "gamma" in training:
        job["gamma"] = training["gamma"]
    if job:
        args += ["--stage_job", json.dumps(job)]
    return args


def head_mean(values):
    """Mean over the first 10% of a logged series."""
    values = list(values)
    head = values[:max(len(values) // 10, 1)]
    return float(sum(head) / len(head))


def progress_key(cfg, outcome):
    """How far a policy got in the task: the rates of the task's progress flags, the last (hardest) flag first.

    Compared lexicographically, so a policy that reaches a later task step ranks higher than one that only does
    earlier steps more often. Rates are rounded to 0.02 so that noise in a late flag does not decide alone.
    """
    if not outcome or "error" in outcome:
        return tuple(-1.0 for _ in progress_flags(cfg))
    rates = dict(outcome["flag_rates"])
    # a success ends the episode, so its flag is true for one step only: use the episode success rate instead
    success_flag = cfg.env.get("success_flag", "succeeded")
    if success_flag in rates:
        rates[success_flag] = outcome.get("eval_success_rate", outcome.get("task_score", 0.0))
    return tuple(round(rates.get(flag, 0.0) / 0.02) * 0.02 for flag in reversed(progress_flags(cfg)))


def progress_flags(cfg):
    """Task flags from the first to the last step of the task (env config ``progress_flags``)."""
    return list(cfg.env.get("progress_flags") or [])


def best_stage(cfg, stage_results):
    """Index of the stage whose final policy got furthest in the task (later stage on ties), or None."""
    valid = [s for s, r in enumerate(stage_results) if "error" not in r and r.get("checkpoint")]
    if not valid:
        return None
    return max(valid, key=lambda s: (progress_key(cfg, stage_results[s]), s))


def furthest_level(cfg, rates, threshold=0.05):
    """Index of the last progress flag (env config ``progress_flags``) with a rate of at least ``threshold``, or -1."""
    level = -1
    for i, flag in enumerate(progress_flags(cfg)):
        if (rates or {}).get(flag, 0.0) >= threshold:
            level = i
    return level


def eval_progressed(new, old, margin=0.05):
    """Whether checkpoint evaluation ``new`` got clearly further in the task than ``old``: compared from the success
    rate, then from the last success condition back to the first; the first difference larger than ``margin``
    decides."""
    a = [new["success_rate"], *reversed(list(new["conditions"].values()))]
    b = [old["success_rate"], *reversed([old["conditions"].get(k, 0.0) for k in new["conditions"]])]
    for x, y in zip(a, b):
        if abs(x - y) > margin:
            return x > y
    # no clear difference in the steps reached: partial progress on a declared scalar (e.g. the distance still to go
    # in the next step) decides, by a relative improvement of at least 15%
    return scalar_progress(new, old) > 0.15


def scalar_progress(new, old):
    """Mean relative improvement of evaluation ``new`` over ``old`` on the declared progress scalars (0 if none)."""
    gains = []
    for name, entry in (new.get("progress_scalars") or {}).items():
        other = (old.get("progress_scalars") or {}).get(name)
        if not other or entry["median_best"] is None or other["median_best"] is None:
            continue
        a, b = entry["median_best"], other["median_best"]
        scale = max(abs(b), 1e-6)
        gains.append((b - a) / scale if entry["better"] == "lower" else (a - b) / scale)
    return sum(gains) / len(gains) if gains else 0.0


def warm_start_ok(cfg, state):
    """Whether the next curricula should start from the best policy so far instead of the initial policy.

    Only a policy that got clearly further in the task than the initial policy is inherited (a policy that learned an
    early step in a form the next step cannot build on would pass the dead end on to every later curriculum). Both are
    compared by their deterministic checkpoint evaluations; the initial policy is evaluated before the first
    iteration.
    """
    best = state.get("best")
    if not (cfg.warm_start_across_iterations and best and best.get("checkpoint")):
        return False
    if best.get("eval") and state.get("init_eval"):
        return eval_progressed(best["eval"], state["init_eval"])
    if best.get("flag_rates") is None or state.get("init_level") is None:
        return True  # no information (older state files, or no checkpoint evaluation)
    return furthest_level(cfg, best["flag_rates"]) > state["init_level"]


def best_flag_rates(cfg, stage_results, stage):
    """Final flag rates of a stage, with the success flag replaced by the evaluated success rate."""
    try:
        result = stage_results[stage]
    except (IndexError, TypeError, KeyError):
        return None
    if not result or "error" in result:
        return None
    rates = dict(result.get("flag_rates") or {})
    success_flag = cfg.env.get("success_flag", "succeeded")
    if success_flag in rates:
        rates[success_flag] = result.get("eval_success_rate", result.get("task_score", 0.0))
    return rates


def annotate_quantities(cfg, quantities, checker, adapter):
    """Append to each scalar quantity's description its measured values on states of the checker's policy (the
    policy the curricula start from), by task progress level, so that the LLM sets ranges and thresholds from real
    values (e.g. how far the arm is from its start pose after placing) instead of guessing."""
    base = annotate_quantities.base.setdefault(id(quantities), {n: q.description for n, q in quantities.items()})
    for n, q in quantities.items():
        q.description = base[n]
    if checker is None or not cfg.get("quantity_profile", True):
        return
    data = checker.profile(adapter, progress_flags(cfg))
    for n, levels in data.items():
        if n not in quantities:
            continue
        text = "; ".join(f"{level}: median {v[1]:g} (5-95%: {v[0]:g} to {v[2]:g})" for level, v in levels.items())
        quantities[n].description = (base[n] + f" [measured on states of the policy the curricula start from, by the "
                                     f"last task step reached: {text}]")
    logging.info(f"Quantity ranges on the starting policy's states: {json.dumps(data)[:3000]}")


annotate_quantities.base = {}


def stage_outcome(cfg, log_dir, spec, flags):
    """Training statistics of one stage: scores, final task-flag rates, transition, action noise."""
    score, secondary, correlation, _, _ = evaluate_run(cfg, log_dir)
    logs = load_tensorboard_logs(log_dir)
    rates = {
        flag: reduce_metric(logs[f"Generated/metric_{flag}"], "tail")
        for flag in flags if f"Generated/metric_{flag}" in logs
    }
    std = logs.get("Policy/mean_std", [float("nan")])[-1]
    # final mean of every derived quantity of the stage (flags: rate)
    derived = {
        key[len("Generated/derived_"):]: reduce_metric(values, "tail")
        for key, values in logs.items() if key.startswith("Generated/derived_")
    }
    # the same statistics over the first 10% of the stage (to detect skills that the stage destroyed) and the mean
    # contribution of every reward term at the start and the end (to see what the policy optimized)
    rates_start = {
        flag: head_mean(logs[f"Generated/metric_{flag}"])
        for flag in flags if f"Generated/metric_{flag}" in logs
    }
    terms = {}
    for term in spec["reward"]["terms"]:
        values = logs.get(f"Generated/{term['name']}")
        if values:
            terms[term["name"]] = {"start": head_mean(values), "end": reduce_metric(values, "tail")}
    reward = logs.get("Generated/gpt_reward")
    transition = spec["stage"]["transition"]
    transition_rate = rates.get(transition["flag"], 0.0)
    # a flag that ends the episode (the success flag, and env episode_end_flags) is true for one step only, so its
    # share of the steps says nothing: such a stage is judged by the episode success rate instead
    episode_flags = set(cfg.env.get("episode_end_flags") or []) | {cfg.env.get("success_flag")}
    episode_series = None
    if transition["flag"] in episode_flags and logs.get(cfg.env.success_metric):
        episode_series = logs[cfg.env.success_metric]
        transition_rate = reduce_metric(episode_series, "tail")
    # exploration per action (rd_lib/exploration.py): noise and share of steps with the output beyond the clip range
    names = list(cfg.env.get("action_names") or [])
    exploration, exploration_warnings = {}, []
    i = 0
    while f"Exploration/std_{i}" in logs:
        name = names[i] if i < len(names) else f"action {i}"
        noise = reduce_metric(logs[f"Exploration/std_{i}"], "tail")
        saturated = reduce_metric(logs[f"Exploration/saturated_{i}"], "tail")
        exploration[name] = {"noise": round(noise, 3), "saturated": round(saturated, 3)}
        if saturated > 0.9:
            exploration_warnings.append(
                f"{name}: output beyond the clip range in {100 * saturated:.0f}% of steps, so the other end of its range "
                "is rarely explored")
        elif saturated > 0.5 and noise < 0.2:
            exploration_warnings.append(
                f"{name}: output held at one end of its range ({100 * saturated:.0f}% of steps beyond the clip range) "
                f"with little noise ({noise:.2f}), so the policy almost never tries the other end; if a step needs it "
                "(e.g. opening a gripper that holds an object), keep that action's noise higher in this stage")
        i += 1
    # shaping terms that stopped improving part way: the policy found a cheap state that partly satisfies them (e.g.
    # tipping an object raises its center a little), so they show where the reward stops guiding
    stalled_terms = []
    for term in spec["reward"]["terms"]:
        fill_series = logs.get(f"Generated/fill_{term['name']}") or []
        gate_series = logs.get(f"Generated/gate_{term['name']}") or []
        if gate_series and len(gate_series) == len(fill_series):
            # the term's level where its gates are open (phase-conditioned), not diluted by the steps it does not apply
            fill_series = [f / g if g > 0.01 else 0.0 for f, g in zip(fill_series, gate_series)]
        if len(fill_series) < 20:
            continue
        n = len(fill_series)
        end = sum(fill_series[-n // 5:]) / len(fill_series[-n // 5:])
        middle = fill_series[n // 2:n // 2 + n // 5]
        middle = sum(middle) / len(middle)
        if end < 0.5 and abs(end - middle) < 0.03:
            gates = term.get("require") or []
            gate_text = f" (gated on {', '.join(gates)})" if gates else ""
            if end < 0.01:
                stalled_terms.append(f"{term['name']}{gate_text}: never earned")
            else:
                occupancy = (f", its gates open in {100 * sum(gate_series[-len(gate_series) // 5:]) / len(gate_series[-len(gate_series) // 5:]):.0f}% of the steps"
                             if gate_series else "")
                stalled_terms.append(f"{term['name']}{gate_text}: flat at {100 * end:.0f}% of its maximum where it applies "
                                     f"since the middle of the stage{occupancy}")
    series = episode_series or logs.get(f"Generated/metric_{transition['flag']}") or []
    window = max(len(series) // 10, 1)
    still_rising = (len(series) >= 4 * window and
                    sum(series[-window:]) / window > sum(series[-3 * window:-2 * window]) / window + 0.03)
    return {
        "exploration": exploration,
        "exploration_warnings": exploration_warnings,
        "still_rising": bool(still_rising),
        "stalled_terms": stalled_terms,
        "log_dir": log_dir,
        "checkpoint": review.latest_checkpoint(log_dir),
        "flag_rates_start": rates_start,
        "term_means": terms,
        "reward_start_end": [head_mean(reward), reduce_metric(reward, "tail")] if reward else None,
        "task_score": score,
        "gt_score": secondary,
        "reward_correlation": correlation,
        "flag_rates": rates,
        "derived_means": derived,
        "action_std": std,
        "transition_flag": transition["flag"],
        "transition_rate": transition_rate,
        "transition_reached": transition_rate >= transition["threshold"],
    }


def train_curricula(cfg, launcher, jobs, randomization_options, flags, seed=None, writing=None):
    """Train several curricula in parallel, each one stage after stage.

    A curriculum starts its next stage as soon as its own previous stage has finished, without waiting for the
    other curricula. Stages that already finished (in a resumed run) are reused.

    Args:
        jobs: {key: [stage job dict]}, a stage job has: spec, reward_path, log_file, run_name (and optionally
            seed, init_checkpoint).
        writing: {key: bool} while stages are still being written: their jobs are appended to jobs[key] as they are
            accepted (from other threads) and start as soon as the curriculum is ready for them.
    Returns: {key: [stage outcome dict or {"error": message}]} (a failed stage ends its curriculum).
    """
    results = {key: [] for key in jobs}
    next_stage = {key: 0 for key in jobs}  # None after a failed stage
    active = {}  # key -> (process, stage index)

    def finish(key, s):
        job = jobs[key][s]
        log_dir, error = parse_training_log(job["log_file"], job["reward_path"])
        if error or log_dir is None or review.latest_checkpoint(log_dir) is None:
            results[key].append({"error": error or "no checkpoint was written"})
            next_stage[key] = None
            logging.info(f"{job['run_name']}: stage failed: {(error or '').strip()[-300:]}")
            return
        outcome = stage_outcome(cfg, log_dir, job["spec"], flags)
        outcome["start_from_stage"] = job.get("start_from_stage")
        outcome["hacking_warnings"] = hacking_warnings(cfg, outcome)
        outcome["yaml_index"] = job.get("yaml_index", s)
        outcome["stage_name"] = job["spec"]["stage"]["name"] + (" (extension)" if job.get("continue_from") is not None else "")
        outcome["threshold"] = job["spec"]["stage"]["transition"]["threshold"]
        results[key].append(outcome)
        next_stage[key] = s + 1
        # extend a stage that missed its target while its key metric was still rising: same reward, continued from
        # its own checkpoint (at most stage_extensions per curriculum, each stage_extension_share of the budget)
        used = sum(1 for j in jobs[key] if j.get("continue_from") is not None)
        if (not outcome["transition_reached"] and outcome["still_rising"] and used < cfg.get("stage_extensions", 0)):
            extension = dict(job, continue_from=s, yaml_index=job.get("yaml_index", s))
            spec = json.loads(json.dumps(job["spec"]))
            spec["training"]["max_iterations"] = max(
                cfg.min_stage_iterations, int(cfg.curriculum_iterations * cfg.get("stage_extension_share", 0.25)))
            extension["spec"] = spec
            extension["log_file"] = job["log_file"].replace(".txt", f"_ext{used + 1}.txt")
            extension["run_name"] = f"{job['run_name']}_ext{used + 1}"
            jobs[key].insert(s + 1, extension)
            logging.info(f"{job['run_name']}: {outcome['transition_flag']} still rising "
                         f"({outcome['transition_rate']:.3f} < {outcome['threshold']}); extending the stage by "
                         f"{spec['training']['max_iterations']} iterations")
        logging.info(
            f"{job['run_name']}: task score {outcome['task_score']:.3g}, gt {outcome['gt_score']:.3g}, "
            f"{outcome['transition_flag']} rate {outcome['transition_rate']:.3f} "
            f"({'reached' if outcome['transition_reached'] else 'not reached'}), std {outcome['action_std']:.2f}"
        )

    def ready(key):
        s = next_stage[key]
        return key not in active and s is not None and s < len(jobs[key])

    def start(key):
        """Launch the next stage of a curriculum; a stage that already finished is only evaluated."""
        s = next_stage[key]
        job = jobs[key][s]
        if training_finished(job["log_file"]):
            finish(key, s)
            return
        extra = stage_extra_args(cfg, job["spec"], randomization_options)
        if job.get("continue_from") is not None:
            source = job["continue_from"]
            job["start_from_stage"] = source
            extra += resume_args(results[key][source]["checkpoint"])
        elif s > 0:
            source = s - 1
            if cfg.get("stage_rollback", True):
                # a stage that destroyed an earlier skill is not continued: start from the best stage so far
                source = best_stage(cfg, results[key]) if best_stage(cfg, results[key]) is not None else s - 1
                if source != s - 1:
                    logging.info(f"{job['run_name']}: starting from stage {source} (it got further in the task than "
                                 f"stage {s - 1})")
            job["start_from_stage"] = source
            extra += resume_args(results[key][source]["checkpoint"])
        elif job.get("init_checkpoint"):
            extra += resume_args(job["init_checkpoint"])
        process = launcher.launch(
            reward_file=job["reward_path"],
            log_file=job["log_file"],
            run_name=job["run_name"],
            max_iterations=job["spec"]["training"]["max_iterations"],
            seed=job.get("seed", seed),
            extra_args=extra,
        )
        active[key] = (process, s)

    def still_writing():
        return writing is not None and any(writing.values())

    while True:
        # launch every curriculum whose next stage can start (reused stages may make the next one ready at once)
        while any(ready(key) for key in jobs):
            for key in sorted(jobs):
                if ready(key):
                    start(key)
        if not active:
            if still_writing():
                time.sleep(5)  # nothing to run yet: wait for the next accepted stage
                continue
            break
        # wait until one of the running stages finishes (or a newly written stage can start), then evaluate it
        while all(process.poll() is None for process, _ in active.values()) and not any(ready(key) for key in jobs):
            launcher._poll()
            time.sleep(5)
        launcher._poll()  # also closes the log files of finished processes
        for key, (process, s) in list(active.items()):
            if process.poll() is not None:
                del active[key]
                finish(key, s)
    return results


def evaluate_checkpoint(cfg, task, adapter, checkpoint, gpu, cache_file):
    """Deterministic evaluation of a checkpoint (isaaclab_backend.evaluate_checkpoint): episode success rate and the
    share of episodes reaching each success condition. Cached in ``cache_file``; None if it fails."""
    cached = read_json(cache_file)
    if cached is not None and cached.get("checkpoint") == checkpoint:
        return cached
    python = cfg.isaaclab_python or sys.executable
    cmd = [python, "-m", "isaaclab_backend.evaluate_checkpoint", "--task", task, "--checkpoint", checkpoint,
           "--adapter", adapter, "--num_envs", str(cfg.get("eval_num_envs", 256)),
           "--episodes_per_env", str(cfg.get("eval_episodes_per_env", 3)), "--headless"]
    if cfg.env.get("progress_scalars"):
        cmd += ["--progress_scalars", json.dumps(OmegaConf.to_container(cfg.env.progress_scalars, resolve=True))]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        out = subprocess.run(cmd, cwd=ROOT_DIR, env=env, capture_output=True, text=True, timeout=1800).stdout
    except subprocess.TimeoutExpired:
        logging.info(f"Checkpoint evaluation timed out: {checkpoint}")
        return None
    line = next((l for l in out.splitlines() if l.startswith("@@RD_EVAL@@ ")), None)
    if line is None:
        logging.info(f"Checkpoint evaluation failed: {checkpoint}: {out.strip()[-500:]}")
        return None
    result = dict(json.loads(line[len("@@RD_EVAL@@ "):]), checkpoint=checkpoint)
    write_json(cache_file, result)
    return result


def apply_iteration_schedule(cfg, limits, it):
    """Set this iteration's number of candidates and training budget from the optional schedules
    (num_candidates_schedule, curriculum_iterations_schedule: one value per iteration, the last one repeats).
    Returns whether this iteration is a wide round (more candidates than the narrow value num_candidates_narrow)."""
    def pick(schedule):
        values = list(schedule)
        return values[min(it, len(values) - 1)]

    if cfg.get("num_candidates_schedule"):
        cfg.num_candidates = int(pick(cfg.num_candidates_schedule))
    if cfg.get("curriculum_iterations_schedule"):
        cfg.curriculum_iterations = int(pick(cfg.curriculum_iterations_schedule))
        limits["max_iterations"] = cfg.curriculum_iterations
    wide = bool(cfg.get("num_candidates_schedule")) and cfg.num_candidates > int(cfg.get("num_candidates_narrow", 2))
    logging.info(f"Iteration {it}: {cfg.num_candidates} candidates, {cfg.curriculum_iterations} PPO iterations each"
                 + (" (wide search)" if wide else ""))
    return wide


def llm_usage_text(path):
    """Requests, prompt tokens (of them cached), completion tokens and time per role, from the usage log."""
    totals = {}
    try:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                t = totals.setdefault(entry["role"], [0, 0, 0, 0, 0.0])
                t[0] += 1
                t[1] += entry.get("prompt_tokens") or 0
                t[2] += entry.get("cached_prompt_tokens") or 0
                t[3] += entry.get("completion_tokens") or 0
                t[4] += entry.get("seconds") or 0.0
    except (OSError, ValueError):
        return "(no usage log)"
    return "; ".join(f"{role}: {n} requests, {p / 1e3:.0f}k prompt tokens ({c / 1e3:.0f}k cached), {o / 1e3:.0f}k "
                     f"completion tokens, {sec / 60:.0f} min" for role, (n, p, c, o, sec) in sorted(totals.items()))


def funnel_text(evaluation):
    """One line: share of episodes reaching each success condition (deterministic policy)."""
    if not evaluation:
        return ""
    parts = ", ".join(f"{name} {100 * rate:.0f}%" for name, rate in evaluation["conditions"].items())
    text = (f"Checkpoint evaluation ({evaluation['episodes']} episodes, deterministic policy): success "
            f"{100 * evaluation['success_rate']:.1f}%; episodes reaching each success condition: {parts}")
    shares = evaluation.get("time_share") or {}
    if shares:
        text += (". Share of the time in which each state holds now (latched flags such as a 'once' step only say that "
                 "the step happened at least once): " + ", ".join(
                     f"{name} {100 * share:.0f}%" for name, share in sorted(shares.items(), key=lambda x: -x[1])
                     if share >= 0.005))
    limits = evaluation.get("joints_at_limit") or {}
    if limits:
        text += (". Joints at a position limit (share of the time; a policy that parks a joint against its limit "
                 "often avoids the task): " + ", ".join(
                     f"{name} {100 * share:.0f}%" for name, share in sorted(limits.items(), key=lambda x: -x[1])))
    hold = evaluation.get("success_hold")
    if hold and hold["episodes_started"] > 0:
        broken = ", ".join(f"{name} {100 * share:.0f}%" for name, share in hold["broken_by"].items() if share > 0)
        text += (f". Success hold ({hold['required_s']} s of all success conditions at once): started in "
                 f"{100 * hold['episodes_started']:.0f}% of the episodes, median longest hold "
                 f"{hold['median_longest_s']:.2f} s")
        if hold["breaks"]:
            text += (f"; it ended early {hold['breaks']} times, and the condition that was false when it ended "
                     f"(share of the ends): {broken or 'none recorded'}")
        longest = hold.get("median_longest_s") or 0.0
        if (longest >= 0.6 * hold["required_s"] and hold["breaks"] > 2 * max(evaluation["episodes"], 1)
                and evaluation["success_rate"] < 0.5 * hold["episodes_started"]):
            text += (". WARNING: the hold repeatedly reaches most of the required time and then ends, many times per "
                     "episode: the policy is probably ending the hold on purpose just before success to keep collecting "
                     "a reward for holding (finishing ends the episode and its reward stream). Make finishing pay more "
                     "than holding: cap or drop the per-step hold credit and give the success event more than the hold "
                     "credit the rest of an episode could collect")
    return text


def hacking_warnings(cfg, result):
    """Signs of reward hacking in a trained stage: the reward rose while a task step that the policy could do at
    the start of the stage got much worse."""
    if "error" in result or not result.get("reward_start_end"):
        return []
    warnings = []
    r0, r1 = result["reward_start_end"]
    rose = r1 > r0 + 0.1 * abs(r0) + 1e-4
    order = progress_flags(cfg)
    for i, flag in enumerate(order):
        start = result["flag_rates_start"].get(flag)
        end = result["flag_rates"].get(flag)
        if start is None or end is None:
            continue
        # giving up an earlier step is fine when a later step got better (e.g. releasing a grasp over the target)
        later_improved = any(
            result["flag_rates"].get(later, 0.0) >= result["flag_rates_start"].get(later, 0.0) + 0.1
            for later in order[i + 1:])
        if start >= 0.2 and end < start - max(0.15, 0.3 * start) and not later_improved:
            warnings.append(
                f"'{flag}' fell from {start:.2f} to {end:.2f} during the stage while the reward "
                f"{'rose' if rose else 'changed'} from {r0:.4g} to {r1:.4g}")
    # a command-based flag that rose without its physical counterpart (env config physical_evidence: {flag: physical})
    for flag, physical in (cfg.env.get("physical_evidence") or {}).items():
        rate = result["flag_rates"].get(flag)
        phys = result["flag_rates"].get(physical)
        if rate is None or phys is None:
            continue
        rose = rate - result["flag_rates_start"].get(flag, 0.0)
        if rate > 0.05 and rose > 0.03 and phys < 0.25 * rate:
            warnings.append(f"'{flag}' rose to {rate:.2f} but its physical evidence '{physical}' is only {phys:.2f}: the "
                            "flag may be triggered without doing the step")
    return warnings


def stage_statistics_text(result):
    """Flag rates and reward-term means at the start and the end of a stage."""
    lines = [f"reward per step: {result['reward_start_end'][0]:.4g} -> {result['reward_start_end'][1]:.4g}"]
    lines += [f"flag {k}: {result['flag_rates_start'].get(k, float('nan')):.3f} -> {v:.3f}"
              for k, v in result["flag_rates"].items()]
    lines += [f"term {k}: {v['start']:.4g} -> {v['end']:.4g}" for k, v in result["term_means"].items()]
    lines += [f"derived {k}: {v:.4g} (end)" for k, v in (result.get("derived_means") or {}).items()]
    if result.get("eval"):
        lines.append(funnel_text(result["eval"]))
    return "\n".join(lines)


def outcomes_text(plan, stage_results):
    lines = []
    for s, result in enumerate(stage_results):
        name = result.get("stage_name") or plan["stages"][min(s, len(plan["stages"]) - 1)]["name"]
        if "error" in result:
            lines.append(f"Stage {s + 1} ({name}): training failed: {result['error'].strip()[-800:]}")
            continue
        rates = ", ".join(f"{k} {v:.3f}" for k, v in result["flag_rates"].items())
        lines.append(
            f"Stage {s + 1} ({name}): task score {result['task_score']:.3g}, original-reward score "
            f"{result['gt_score']:.3g}, transition {result['transition_flag']} rate {result['transition_rate']:.3f} "
            f"(target {result.get('threshold', plan['stages'][min(s, len(plan['stages']) - 1)]['transition']['threshold'])}, "
            f"{'reached' if result['transition_reached'] else 'not reached'}), final action std "
            f"{result['action_std']:.2f}. Final flag rates: {rates}"
            + (". Derived quantities (final mean; rate for flags): "
               + ", ".join(f"{k} {v:.3g}" for k, v in result["derived_means"].items())
               if result.get("derived_means") else "")
            + (f". Started from the policy of stage {result['start_from_stage'] + 1}"
               if result.get("start_from_stage") is not None and result["start_from_stage"] != s - 1 else "")
        )
        if result.get("eval"):
            lines.append("  " + funnel_text(result["eval"]))
        if result.get("exploration"):
            lines.append("  Action noise at the end of the stage (noise std; share of steps with the output beyond the "
                         "clip range): " + ", ".join(f"{name} {v['noise']:.2f} ({100 * v['saturated']:.0f}%)"
                                                     for name, v in result["exploration"].items()))
        if result.get("exploration_warnings"):
            lines.append(f"  Exploration warnings: {'; '.join(result['exploration_warnings'])}")
        if result.get("stalled_terms"):
            lines.append(
                "  Stalled shaping terms (the policy settled in a state that pays only part of the term; find that state "
                "in the visual evaluation and exclude it with a stricter quantity or gate, or make the next step pay "
                f"more): {'; '.join(result['stalled_terms'])}")
        if result.get("hacking_warnings"):
            lines.append(f"  Reward-hacking warnings: {'; '.join(result['hacking_warnings'])}")
        audit = result.get("hacking_audit")
        if audit:
            lines.append(
                f"  Reward-hacking audit: {'HACKING' if audit.get('hacking') else 'no hacking found'}: "
                f"{audit.get('diagnosis', '')} Exploited terms: {', '.join(audit.get('exploited_terms', [])) or '-'}. "
                f"Fix: {audit.get('fix', '')}")
    return "\n".join(lines)


def outcome_summary(scores):
    best = scores.get("best_stage")
    return (
        f"task score {scores['task_score']:.3g}, visual score {scores['visual_score']:.2f}, "
        f"original-reward score {scores['gt_score']:.3g}, stages completed {scores['stages_trained']}"
        + (f", best policy after stage {best + 1}" if best is not None else "")
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


@hydra.main(config_path=f"{ROOT_DIR}/cfg", config_name="pipeline", version_base="1.3")
def main(cfg):
    if cfg.resume:
        cfg = resume_config(cfg)
        logging.info(f"Resuming the run in {Path.cwd()}")
    elif not os.path.exists(CONFIG_FILE):
        OmegaConf.save(cfg, CONFIG_FILE)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    signal.signal(signal.SIGHUP, lambda signum, frame: sys.exit(128 + signum))
    signal.signal(signal.SIGINT, signal.default_int_handler)
    launcher = TrainingLauncher(cfg)
    try:
        run_pipeline(cfg, launcher)
    finally:
        launcher.terminate_all()


def run_pipeline(cfg, launcher):
    run_dir = Path.cwd()
    logging.info(f"Workspace: {run_dir}")
    task = cfg.env.task
    task_text = cfg.env.description
    init_checkpoint = install_init_checkpoint(cfg)
    if init_checkpoint and cfg.env.get("init_policy_description"):
        task_text = f"{task_text}\n{cfg.env.init_policy_description}"
    elif not init_checkpoint and cfg.env.get("scratch_policy_description"):
        task_text = f"{task_text}\n{cfg.env.scratch_policy_description}"
    adapter = cfg.env.adapter
    quantities = get_adapter(adapter)
    flags = [n for n, q in quantities.items() if q.kind == "flag"]
    randomization_options = OmegaConf.to_container(cfg.env.get("stage_randomization") or {}, resolve=True)
    limits = {
        "min_iterations": cfg.min_stage_iterations,
        "max_iterations": cfg.curriculum_iterations,
        "entropy_coef": tuple(cfg.entropy_coef_range),
        "learning_rate": tuple(cfg.learning_rate_range),
        "max_abs_weight": cfg.max_abs_weight,
        "action_names": list(cfg.env.get("action_names") or []),
    }
    prompts = agents.load_prompts(f"{ROOT_DIR}/prompts")
    review_prompts = review.load_prompts(f"{ROOT_DIR}/prompts")

    # exploration settings of the policy (noise floor/ceiling, output squashing) for every training, checker and
    # evaluation process of this run (rd_lib/exploration.py reads them from the environment)
    exploration = cfg.env.get("exploration")
    if exploration:
        os.environ["RD_EXPLORATION"] = json.dumps(OmegaConf.to_container(exploration, resolve=True))
        logging.info(f"Exploration settings: {os.environ['RD_EXPLORATION']}")
    if cfg.get("fast_logging", True):
        # the same statistics without a GPU synchronization per value and step (environment logs stay tensors,
        # the reward statistics are batched): the logged numbers do not change
        os.environ["RD_TENSOR_LOGS"] = "1"
        os.environ["RD_BATCHED_STATS"] = "1"
    else:
        os.environ["RD_BATCHED_STATS"] = "0"
    # hard-case starts (rd_lib/hard_starts.py): a share of the training episodes restart from start states in which
    # the policy failed; replayed episodes are left out of the training statistics
    hard = cfg.get("hard_starts")
    if hard and float(hard.get("prob", 0.0)) > 0.0:
        settings = dict(OmegaConf.to_container(hard, resolve=True))
        settings.update(adapter=cfg.env.adapter, success_flag=cfg.env.get("success_flag") or "succeeded")
        os.environ["RD_HARD_STARTS"] = json.dumps(settings)
        logging.info(f"Hard-case starts: {os.environ['RD_HARD_STARTS']}")
    else:
        os.environ.pop("RD_HARD_STARTS", None)
    # optional: the task's progress flags (except the success flag, which ends the episode) as extra observation
    # inputs of the policy, in every process of this run (rd_lib/obs_extension.py)
    if cfg.env.get("obs_progress_flags"):
        obs_flags = [f for f in progress_flags(cfg) if f != cfg.env.get("success_flag")]
        os.environ["RD_OBS_FLAGS"] = json.dumps({"adapter": cfg.env.adapter, "flags": obs_flags})
        logging.info(f"Progress flags appended to the policy observations: {obs_flags}")

    # model roles: thinking only where it pays (planning, reward-hacking review and audit); stages are written
    # without thinking; simple calls (retrieval query, example selection, format fixes) go to the light model
    # token use per role, one JSON line per request (summed per iteration in pipeline.log)
    LLMClient.usage_file = str(run_dir / "llm_usage.jsonl")
    llm = LLMClient(cfg)
    llms = {
        "strong": llm.with_options(thinking=True, max_tokens=cfg.get("reasoning_max_tokens"), role="strong"),
        "writer": llm.with_options(thinking=False, role="writer"),
    }
    if cfg.get("llm_light") and cfg.llm_light.get("model"):
        llms["light"] = LLMClient(OmegaConf.merge(cfg, cfg.llm_light)).with_options(thinking=False, role="light")
    else:
        llms["light"] = llm.with_options(thinking=False, max_tokens=cfg.get("light_max_tokens", 4096), role="light")
    vlm = make_vlm_client(cfg, llm).with_options(role="vlm")
    terminate_stale_runs(run_dir)

    # derived quantities: the LLM may define new quantities in the stage YAMLs; a simulator worker checks them
    checker, env_code = None, "(not available)"
    if cfg.get("derived_quantities", True):
        context_file = Path(ROOT_DIR) / "env_context" / f"{cfg.env.env_name}_obs.py"
        if cfg.env.get("env_reference"):
            env_code = cfg.env.env_reference  # compact attribute reference (meanings and units), not the source
        elif context_file.exists():
            env_code = context_file.read_text()
        if cfg.env.get("action_names"):
            env_code += ("\nAction names, in order (for training.exploration.per_action): "
                         + ", ".join(cfg.env.action_names))
        checker = make_checker(cfg, ROOT_DIR, task, init_checkpoint, str(run_dir / "derived_checker.log"))
        checker.audit = {"success_flag": cfg.env.get("success_flag"), "gamma": 0.99}
        logging.info("Starting the derived-quantity checker (a small simulation of the task)")
        if not checker.start():
            checker = None
    annotate_quantities(cfg, quantities, checker, adapter)
    try:
        run_iterations(cfg, launcher, run_dir, task, task_text, adapter, quantities, flags, randomization_options,
                       limits, prompts, review_prompts, llms, vlm, init_checkpoint, checker, env_code)
    finally:
        if checker is not None:
            checker.close()


def run_iterations(cfg, launcher, run_dir, task, task_text, adapter, quantities, flags, randomization_options,
                   limits, prompts, review_prompts, llms, vlm, init_checkpoint, checker, env_code):
    llm, writer, light = llms["strong"], llms["writer"], llms["light"]
    visual = bool(cfg.visual)
    if visual:
        visual, error = review.probe_vision(vlm)
        if not visual:
            logging.warning(f"{vlm.model} does not accept image input; continuing without visual evaluation. Error: {error}")

    os.environ.setdefault("HF_HOME", cfg.hf_home)
    logging.info(f"Loading the retrieval database {cfg.vdb_path} (embeddings: {cfg.embedding_model})")
    vdb = VectorDB(cfg.vdb_path, Embedder(cfg.embedding_model))
    logging.info(f"The retrieval database has {len(vdb)} records")

    state = read_json(STATE_FILE, {"completed_iterations": 0, "iterations": [], "best": None})
    run_tag = run_dir.name

    # the initial policy's deterministic evaluation: the reference for warm starts (a best policy is inherited only if
    # it got clearly further in the task)
    if (init_checkpoint and cfg.get("checkpoint_evaluation", True) and state.get("init_eval") is None
            and state["completed_iterations"] == 0):
        gpus = list(cfg.gpus) if cfg.gpus else [None]
        logging.info(f"Evaluating the initial policy {init_checkpoint}")
        evaluation = evaluate_checkpoint(cfg, task, adapter, init_checkpoint, gpus[-1], "init_eval.json")
        if evaluation is not None:
            state["init_eval"] = {k: v for k, v in evaluation.items() if k != "checkpoint"}
            logging.info(f"Initial policy: {funnel_text(evaluation)}")
            write_json(STATE_FILE, state)

    failed_attempts = 0
    it = state["completed_iterations"]
    while it < cfg.iteration:
        it_dir = Path(f"iter{it}")
        it_dir.mkdir(exist_ok=True)
        wide = apply_iteration_schedule(cfg, limits, it)

        # the live check of derived quantities visits the states of the policy the curricula start from
        if (checker is not None and warm_start_ok(cfg, state)
                and checker.init_checkpoint != state["best"]["checkpoint"]):
            checker.close()
            checker.init_checkpoint = state["best"]["checkpoint"]
            logging.info(f"Restarting the derived-quantity checker with the policy {checker.init_checkpoint}")
            if not checker.start():
                checker = None
            annotate_quantities(cfg, quantities, checker, adapter)

        # ---- human feedback on the previous iteration (optional) and feedback text for the planner
        feedback_text = ""
        if it > 0:
            previous = state["iterations"][-1]
            human_file = Path(f"iter{it - 1}") / "human_feedback.txt"
            human = human_file.read_text().strip() if human_file.exists() else ""
            if human and not previous.get("human_feedback"):
                previous["human_feedback"] = human
                for record_id in previous["record_ids"]:
                    record = next((r for r in vdb.records if r["id"] == record_id), None)
                    if record is not None:
                        vdb.update(record_id, human_feedback=human, text=record["text"] + f"\nHuman feedback: {human}")
                write_json(STATE_FILE, state)
                logging.info(f"Iteration {it}: added the human feedback on iteration {it - 1} to the database")
            feedback_text = previous["feedback_text"]
            if previous.get("human_feedback"):
                feedback_text += f"\nHuman feedback: {previous['human_feedback']}"

        # ---- 1. retrieval
        retrieval_file = str(it_dir / "retrieval.json")
        retrieval = read_json(retrieval_file)
        if retrieval is None:
            query = agents.query_agent(light, prompts, task_text, feedback_text)
            hits = vdb.search(query, cfg.retrieval_top_k)
            choice, reason = agents.selector_agent(light, prompts, task_text, hits)
            example = hits[choice][1] if choice is not None else None
            retrieval = {
                "query": query,
                "hits": [{"id": r["id"], "score": s} for s, r in hits],
                "choice": example["id"] if example else None,
                "reason": reason,
                "example_text": example_text(example) if example else "",
            }
            write_json(retrieval_file, retrieval)
        logging.info(
            f"Iteration {it}: retrieval query '{retrieval['query']}', {len(retrieval['hits'])} hits, "
            f"selected {retrieval['choice']}: {retrieval['reason']}"
        )

        # ---- 2. curriculum plans
        planner_feedback = feedback_text
        if wide:
            # a wide round: many short curricula from the current best policy, each a different attempt at
            # the next task step it does not do yet; the best is kept (warm start) and the next rounds narrow down
            planner_feedback = (feedback_text or "") + (
                f"\n\nThis iteration is a wide search: {cfg.num_candidates} short curricula of {cfg.curriculum_iterations} "
                "PPO iterations each are trained in parallel from the same starting policy, and only the best one is kept. "
                "Plan each curriculum as one stage (two at most) that targets the first task step the starting policy does "
                "not do reliably yet, and make the curricula differ from each other in how they reward that step (different "
                "quantities, gates, shaping and noise settings), so that the search covers clearly different ideas.")
        target = float(cfg.get("min_transition_threshold", 0.0))
        if target > 0:
            planner_feedback = (planner_feedback or "") + (
                f"\n\nTarget: the task must be solved reliably. Every stage's transition threshold must be at least "
                f"{target:g} (e.g. {target:g}-{min(target + 0.1, 1.0):g} of the episodes); plans with lower thresholds are "
                "rejected. A stage that does not reach its threshold is extended while it improves.")
        context = agents.planner_context(
            task_text, quantities, cfg.curriculum_iterations, limits, randomization_options,
            retrieval["example_text"], planner_feedback,
        )
        context.update(
            helpers=HELPERS_DOC,
            env_interface=checker.interface_text() if checker is not None else "(not available)",
            env_code=env_code,
        )
        plans_file = str(it_dir / "plans.json")
        plans = read_json(plans_file)
        if plans is None:
            plans = []
            for attempt in range(3):
                missing = cfg.num_candidates - len(plans)
                if missing <= 0:
                    break
                logging.info(f"Iteration {it}: planning {missing} curricula with {llm.model}")
                new_plans, _ = agents.planner_agent(
                    llm, prompts, context, missing, cfg.curriculum_iterations, cfg.min_stage_iterations, flags,
                    float(cfg.get("min_transition_threshold", 0.0)),
                )
                plans += [p for p in new_plans if p is not None]
            plans = plans[:cfg.num_candidates]
            write_json(plans_file, plans)
        for c, plan in enumerate(plans):
            logging.info(
                f"Iteration {it}: candidate {c} curriculum: "
                + " -> ".join(f"{s['name']} ({s['iterations']} it, {s['transition']['flag']} >= {s['transition']['threshold']})"
                              for s in plan["stages"])
            )

        # ---- 3. stage YAMLs (validated), in parallel across candidates, and 4. staged training: with stream_training
        # a candidate's stage starts training as soon as it is accepted (and its curriculum is ready for it), while its
        # later stages and the other candidates are still being written; the writing itself is unchanged
        stream = bool(cfg.get("stream_training", True))
        start_checkpoint = state["best"]["checkpoint"] if warm_start_ok(cfg, state) else init_checkpoint
        jobs = {c: [] for c in range(len(plans))}
        writing = {c: True for c in range(len(plans))}

        def add_job(c, s, spec):
            job = {
                "spec": spec,
                "reward_path": str((it_dir / f"cand{c}" / f"stage{s}_reward.py").resolve()),
                "log_file": str(it_dir / f"cand{c}" / f"stage{s}.txt"),
                "run_name": f"{cfg.env.env_name}_iter{it}_cand{c}_stage{s}",
                "yaml_index": s,
            }
            if s == 0 and start_checkpoint:
                job["init_checkpoint"] = start_checkpoint
            jobs[c].append(job)

        def generate_and_queue(c):
            try:
                return generate(c)
            finally:
                writing[c] = False

        def generate(c):
            cand_dir = it_dir / f"cand{c}"
            cand_dir.mkdir(exist_ok=True)
            plan = plans[c]
            specs, previous = [], []
            for s in range(len(plan["stages"])):
                yaml_path = cand_dir / f"stage{s}.yaml"
                if yaml_path.exists():
                    spec = yaml.safe_load(yaml_path.read_text())
                else:
                    spec, attempts = agents.stage_agent(
                        writer, prompts, context, plan, s, previous, quantities, limits, randomization_options,
                        adapter, cfg.stage_max_retries, checker,
                        cfg.get("hacking_review_rounds", 2) if cfg.get("hacking_review", True) else 0,
                        review_client=llm, light_client=light,
                        progress_flags=progress_flags(cfg) if cfg.get("reward_ordering_check", True) else None,
                        review_measured=bool(cfg.get("review_measured", False)),
                        review_incremental=bool(cfg.get("review_incremental", False)),
                    )
                    write_json(str(cand_dir / f"stage{s}_attempts.json"), attempts)
                    if spec is None:
                        logging.info(f"Iteration {it}: candidate {c} stage {s} failed validation {len(attempts)} times")
                        if not specs:
                            return c, None
                        # the stages accepted so far are still a curriculum: train them instead of dropping the
                        # candidate (its later stages are planned again in the next iteration)
                        logging.info(f"Iteration {it}: candidate {c} trains its first {len(specs)} stage(s) only")
                        plan["stages"] = plan["stages"][:len(specs)]
                        return c, specs
                    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False))
                    logging.info(f"Iteration {it}: candidate {c} stage {s} valid after {len(attempts)} attempt(s)")
                write_reward_file(str(yaml_path.resolve()), str((cand_dir / f"stage{s}_reward.py").resolve()), ROOT_DIR)
                previous.append(yaml_path.read_text())
                specs.append(spec)
                if stream:
                    add_job(c, s, spec)
            return c, specs

        pool = ThreadPoolExecutor(max_workers=max(len(plans), 1))
        futures = [pool.submit(generate_and_queue, c) for c in range(len(plans))]
        streamed = None
        if stream:
            streamed = train_curricula(cfg, launcher, jobs, randomization_options, flags, writing=writing)
        generated = dict(f.result() for f in futures)
        pool.shutdown()
        candidates = [c for c in range(len(plans)) if generated.get(c)]
        if not candidates:
            failed_attempts += 1
            if failed_attempts >= 3:
                raise RuntimeError(f"Iteration {it}: no candidate curriculum passed validation in 3 attempts")
            logging.info(f"Iteration {it}: no candidate curriculum passed validation, planning again")
            os.remove(plans_file)
            for stage_file in it_dir.glob("cand*/stage*"):
                stage_file.unlink()
            continue  # the same iteration again

        # ---- 4. staged training (already done above with stream_training)
        if stream:
            jobs = {c: jobs[c] for c in candidates}
            results = {c: streamed[c] for c in candidates}
        else:
            for c in candidates:
                for s, spec in enumerate(generated[c]):
                    add_job(c, s, spec)
            jobs = {c: jobs[c] for c in candidates}
            results = train_curricula(cfg, launcher, jobs, randomization_options, flags)

        # ---- 5. evaluation: training statistics and visual evaluation of the best policy of every curriculum
        # A curriculum is judged by the stage whose policy got furthest in the task (not only its last stage, which
        # may have destroyed an earlier skill); that stage's checkpoint is recorded, evaluated and warm-started from.
        scores = {}
        for c in candidates:
            trained = sum(1 for r in results[c] if "error" not in r)
            b = best_stage(cfg, results[c])
            chosen = results[c][b] if b is not None else {}
            scores[c] = {
                "task_score": chosen.get("task_score", DUMMY_FAILURE),
                "gt_score": chosen.get("gt_score", DUMMY_FAILURE),
                "visual_score": DUMMY_FAILURE,
                "progress": list(progress_key(cfg, chosen)) if chosen else None,
                "best_stage": b,
                "stages_trained": f"{trained}/{len(jobs[c])}",
                "checkpoint": chosen.get("checkpoint"),
            }
        # deterministic evaluation of each candidate's best checkpoint (in parallel on the training GPUs): the episode
        # success rate ranks the candidates, the condition breakdown goes to the feedback
        evaluated = [c for c in candidates if scores[c]["checkpoint"]]
        if cfg.get("checkpoint_evaluation", True) and evaluated:
            gpus = list(cfg.gpus) if cfg.gpus else [None]

            def run_eval(i_c):
                i, c = i_c
                return c, evaluate_checkpoint(cfg, task, adapter, scores[c]["checkpoint"], gpus[i % len(gpus)],
                                              str(it_dir / f"cand{c}" / "checkpoint_eval.json"))

            with ThreadPoolExecutor(max_workers=len(evaluated)) as pool:
                for c, evaluation in pool.map(run_eval, enumerate(evaluated)):
                    if evaluation is None:
                        continue
                    b = scores[c]["best_stage"]
                    results[c][b]["eval"] = evaluation
                    results[c][b]["eval_success_rate"] = evaluation["success_rate"]
                    scores[c]["task_score"] = evaluation["success_rate"]
                    scores[c]["progress"] = list(progress_key(cfg, results[c][b]))
                    scores[c]["eval"] = {k: v for k, v in evaluation.items() if k != "checkpoint"}
                    logging.info(f"Iteration {it}: candidate {c} stage {b}: {funnel_text(evaluation)}")
        # what the numbers say about the video's policy, so that the VLM explains how the numbers come about (e.g. how
        # a 'grasped' flag is reached) instead of describing a different behavior than the measured one
        for c in evaluated:
            b = scores[c]["best_stage"]
            r = results[c][b] if b is not None else {}
            facts = []
            if r.get("eval"):
                facts.append(funnel_text(r["eval"]))
            if r.get("flag_rates"):
                facts.append("Task flag rates during training: "
                             + ", ".join(f"{k} {v:.2f}" for k, v in r["flag_rates"].items()))
            if r.get("stalled_terms"):
                facts.append("Reward terms that stopped improving part way: " + "; ".join(r["stalled_terms"]))
            scores[c]["measured_facts"] = "\n".join(facts)
        reports = {}
        if visual and evaluated:
            reports = visual_evaluation(cfg, launcher, vlm, review_prompts, it_dir, evaluated, scores, plans, task_text)
            for c in evaluated:
                scores[c]["visual_score"] = review.visual_score(reports[c])

        reference = state.get("init_eval") or {}

        def rank(c):
            progress = tuple(scores[c]["progress"] or progress_key(cfg, None))
            partial = round(scalar_progress(scores[c].get("eval") or {}, reference), 2) if reference else 0.0
            return (scores[c]["task_score"], *progress, partial, scores[c]["visual_score"], scores[c]["gt_score"])

        best = max(candidates, key=rank)
        logging.info(
            f"Iteration {it}: best candidate {best}: "
            + "; ".join(f"cand {c}: {outcome_summary(scores[c])}" for c in candidates)
        )
        logging.info(f"LLM token use so far, per role: {llm_usage_text(run_dir / 'llm_usage.jsonl')}")

        # ---- 6. feedback and database
        def audit(c):
            """LLM audit of every stage with hacking warnings (or of every stage with hacking_audit: all)."""
            for s, result in enumerate(results[c]):
                if "error" in result or not cfg.get("hacking_audit", True):
                    continue
                if not result.get("hacking_warnings") and cfg.get("hacking_audit") != "all":
                    continue
                path = str(it_dir / f"cand{c}" / f"stage{s}_hacking_audit.json")
                yaml_index = result.get("yaml_index", s)
                found = read_json(path)
                if found is None:
                    found = agents.hacking_audit_agent(
                        llm, prompts, task_text, file_to_string(str(it_dir / f"cand{c}" / f"stage{yaml_index}.yaml")),
                        stage_statistics_text(result), "; ".join(result.get("hacking_warnings", [])),
                    )
                    write_json(path, found)
                result["hacking_audit"] = found
                logging.info(f"Iteration {it}: candidate {c} stage {s} reward-hacking audit: "
                             f"{'hacking' if found.get('hacking') else 'no hacking'}: {found.get('diagnosis', '')[:300]}")

        def analyze(c):
            audit(c)
            path = str(it_dir / f"cand{c}" / "feedback.json")
            feedback = read_json(path)
            if feedback is None:
                curriculum = "\n\n".join(file_to_string(str(it_dir / f"cand{c}" / f"stage{s}.yaml")) for s in range(len(plans[c]["stages"])))
                visual_text = review.format_report(reports[c]) if c in reports else ""
                feedback = agents.feedback_agent(
                    writer, prompts, task_text, curriculum, outcomes_text(plans[c], results[c]), visual_text, ""
                )
                write_json(path, feedback)
            return c, feedback

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            feedbacks = dict(pool.map(analyze, candidates))

        record_ids = []
        for c in candidates:
            record_id = f"{run_tag}-iter{it}-cand{c}"
            record_ids.append(record_id)
            if vdb.has(record_id):
                continue
            stage_yamls = [file_to_string(str(it_dir / f"cand{c}" / f"stage{s}.yaml")) for s in range(len(plans[c]["stages"]))]
            record = {
                "id": record_id,
                "task": task_text,
                "env_task": task,
                "adapter": adapter,
                "plan": plans[c],
                "stage_yamls": stage_yamls,
                "outcomes": outcomes_text(plans[c], results[c]),
                "outcome_summary": outcome_summary(scores[c]),
                "scores": {k: v for k, v in scores[c].items() if k != "checkpoint"},
                "visual_report": review.format_report(reports[c]) if c in reports else "",
                "feedback": f"{feedbacks[c]['analysis']} Recommendations: {feedbacks[c]['recommendations']}",
                "human_feedback": "",
            }
            text = (
                f"Task: {task_text}\nRobot task: {task}\nCurriculum: "
                + " -> ".join(f"{s['name']}: {s['objective']}" for s in plans[c]["stages"])
                + f"\nOutcome: {record['outcome_summary']}\nFeedback: {record['feedback']}"
            )
            vdb.add(record, text)

        # ---- overall best and the state
        if state["best"] is None or rank(best) > tuple(state["best"]["rank"]):
            state["best"] = {
                "iteration": it,
                "candidate": best,
                "rank": list(rank(best)),
                "checkpoint": scores[best]["checkpoint"],
                "best_stage": scores[best]["best_stage"],
                "flag_rates": best_flag_rates(cfg, results[best], scores[best]["best_stage"]),
                "eval": scores[best].get("eval"),
                "stage_yamls": [str((it_dir / f"cand{best}" / f"stage{s}.yaml").resolve()) for s in range(len(plans[best]["stages"]))],
            }
        feedback_text = (
            f"Best curriculum of iteration {it} (candidate {best}): "
            + " -> ".join(f"{s['name']} ({s['iterations']} it)" for s in plans[best]["stages"])
            + f"\nIts stage YAMLs:\n" + "\n\n".join(
                file_to_string(str(it_dir / f"cand{best}" / f"stage{s}.yaml")) for s in range(len(plans[best]["stages"])))
            + f"\nOutcomes:\n{outcomes_text(plans[best], results[best])}"
            + (f"\nVisual evaluation:\n{review.format_report(reports[best])}" if best in reports else "")
            + f"\nAnalysis: {feedbacks[best]['analysis']}\nRecommendations: {feedbacks[best]['recommendations']}"
            + "\nOther candidates: " + "; ".join(
                f"candidate {c}: {outcome_summary(scores[c])}, recommendations: {feedbacks[c]['recommendations'][:500]}"
                for c in candidates if c != best)
        )
        if cfg.warm_start_across_iterations and state["best"] and not warm_start_ok(cfg, state):
            init_eval = state.get("init_eval")
            feedback_text += (
                "\nThe next curricula start again from the initial policy: the best policy so far got no clearly "
                "further in the task than the initial policy"
                + (f" (initial policy: {funnel_text(dict(init_eval, episodes=init_eval.get('episodes', 0)))})"
                   if init_eval else "")
                + ", so the way it does its steps is not inherited.")
        if warm_start_ok(cfg, state):
            b = state["best"]
            b_result = None
            try:
                b_result = (results[b["candidate"]][b["best_stage"]] if b["iteration"] == it else None)
            except (KeyError, IndexError, TypeError):
                b_result = None
            feedback_text += (
                f"\nThe next curricula start from the best policy so far (iteration {b['iteration']}, candidate "
                f"{b['candidate']}, after stage {(b.get('best_stage') or 0) + 1})"
                + (f", whose final flag rates were: "
                   + ", ".join(f"{k} {v:.2f}" for k, v in b_result["flag_rates"].items()) if b_result else "")
                + ". Its skills do not need to be learned again; plan the stages from there."
            )
        write_review(it_dir, it, candidates, plans, scores, results, reports, feedbacks, best)
        state["iterations"].append({
            "iteration": it,
            "best": best,
            "scores": {str(c): {k: v for k, v in scores[c].items() if k != "checkpoint"} for c in candidates},
            "record_ids": record_ids,
            "feedback_text": feedback_text,
            "human_feedback": "",
        })
        state["completed_iterations"] = it + 1
        write_json(STATE_FILE, state)

        # ---- optional wait for human feedback before the next iteration
        if cfg.human_feedback_wait > 0 and it + 1 < cfg.iteration:
            human_file = it_dir / "human_feedback.txt"
            logging.info(f"Iteration {it}: waiting up to {cfg.human_feedback_wait} s for {human_file.resolve()}")
            deadline = time.time() + cfg.human_feedback_wait
            while time.time() < deadline and not human_file.exists():
                time.sleep(10)
        failed_attempts = 0
        it += 1

    # ---- final evaluation: the best curriculum, trained again from scratch with other seeds
    if state["best"] is None:
        logging.info("No curriculum was trained successfully")
        return
    best = state["best"]
    logging.info(f"Best curriculum: iteration {best['iteration']}, candidate {best['candidate']}, rank {best['rank']}")
    if cfg.num_eval <= 0:
        return
    specs = [yaml.safe_load(Path(p).read_text()) for p in best["stage_yamls"]]
    eval_dir = Path("final_eval")
    eval_dir.mkdir(exist_ok=True)
    jobs = {}
    for k in range(cfg.num_eval):
        jobs[k] = [{
            "spec": spec,
            "reward_path": str(Path(p).with_name(Path(p).stem + "_reward.py")),
            "log_file": str(eval_dir / f"eval{k}_stage{s}.txt"),
            "run_name": f"{cfg.env.env_name}_eval{k}_stage{s}",
        } for s, (spec, p) in enumerate(zip(specs, best["stage_yamls"]))]
        for job in jobs[k]:
            job["seed"] = 100 + k  # a different seed for every evaluation run
        if cfg.get("final_eval_from_best", False) and best.get("checkpoint"):
            jobs[k][0]["init_checkpoint"] = best["checkpoint"]  # continue from the best policy of the search
        elif init_checkpoint:
            jobs[k][0]["init_checkpoint"] = init_checkpoint  # "from scratch" = from the configured initial policy
    results = train_curricula(cfg, launcher, jobs, randomization_options, flags)
    summary = {}
    for k, stage_results in results.items():
        b = best_stage(cfg, stage_results)
        final = stage_results[b] if b is not None else {"error": "no stage"}
        summary[k] = {key: final.get(key) for key in ("task_score", "gt_score", "flag_rates", "action_std")} \
            if "error" not in final else {"error": final["error"][-500:]}
        logging.info(f"Final evaluation run {k}: {summary[k]}")
    write_json(str(eval_dir / "summary.json"), summary)


def example_text(record, max_chars=7000):
    """The retrieved record as context for the planner and the stage agent."""
    parts = [
        f"Task: {record.get('task', '')}",
        "Curriculum: " + " -> ".join(
            f"{s['name']} ({s['iterations']} it): {s['objective']}" for s in record.get("plan", {}).get("stages", [])),
        "Stage YAMLs:\n" + "\n\n".join(record.get("stage_yamls", [])),
        f"Outcomes:\n{record.get('outcomes', '')}",
        f"Visual evaluation:\n{record.get('visual_report', '')}" if record.get("visual_report") else "",
        f"Automatic feedback: {record.get('feedback', '')}",
        f"Human feedback: {record['human_feedback']}" if record.get("human_feedback") else "",
    ]
    return "\n".join(p for p in parts if p)[:max_chars]


def visual_evaluation(cfg, launcher, vlm, review_prompts, it_dir, candidates, scores, plans, task_text):
    """Record rollout videos of the final policies and let the VLM score the curriculum stages.

    With ``visual_failures`` > 0 the VLM sees failed episodes instead of one random rollout: frames of the most common
    ways of failing, dense close-ups around the moment progress stopped, and the timeline of the success conditions
    (isaaclab_backend/record_failures.py).
    """
    if int(cfg.get("visual_failures", 0) or 0) > 0:
        return failure_evaluation(cfg, launcher, vlm, review_prompts, it_dir, candidates, scores, plans, task_text)
    launched = {}
    for c in candidates:
        if (it_dir / f"cand{c}" / "video.mp4").exists():
            continue
        checkpoint = scores[c]["checkpoint"]
        launched[c] = (time.time(), os.path.dirname(checkpoint))
        launcher.launch_command(
            review.build_video_command(cfg, launcher.python, checkpoint),
            str(it_dir / f"cand{c}" / "video.txt"),
            f"video {it_dir.name} cand{c} ({os.path.basename(checkpoint)})",
        )
    launcher.wait_all()
    for c, (start, log_dir) in launched.items():
        video = review.find_new_video(log_dir, start)
        if video is not None:
            os.replace(video, it_dir / f"cand{c}" / "video.mp4")

    def evaluate(c):
        path = str(it_dir / f"cand{c}" / "visual.json")
        report = read_json(path)
        if report is None:
            video = it_dir / f"cand{c}" / "video.mp4"
            if not video.exists():
                report = {"error": "no rollout video could be recorded"}
            else:
                subtasks = [{"name": s["name"], "description": s["objective"], "success_criterion": ""}
                            for s in plans[c]["stages"]]
                rollouts = review.extract_rollout_frames(str(video), cfg.visual_rollouts, cfg.visual_num_frames,
                                                      cfg.visual_frame_width, crop=cfg.env.get("visual_crop"))
                try:
                    report = review.evaluate_rollouts(vlm, review_prompts, task_text, subtasks, rollouts,
                                                   facts=scores[c].get("measured_facts", ""))
                except Exception as e:
                    report = {"error": f"VLM request failed: {e}"}
            write_json(path, report)
        return c, report

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        reports = dict(pool.map(evaluate, candidates))
    for c, report in reports.items():
        logging.info(f"{it_dir.name} candidate {c}: visual score {review.visual_score(report):.2f}\n{review.format_report(report)}")
    return reports


def failure_evaluation(cfg, launcher, vlm, review_prompts, it_dir, candidates, scores, plans, task_text):
    """Record frames of failed episodes of the final policies and let the VLM explain the failures."""
    for c in candidates:
        out_dir = it_dir / f"cand{c}" / "failures"
        if (out_dir / "failures.json").exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = scores[c]["checkpoint"]
        launcher.launch_command(
            review.build_failures_command(cfg, launcher.python, checkpoint, str(out_dir)),
            str(it_dir / f"cand{c}" / "failures.txt"),
            f"failure frames {it_dir.name} cand{c} ({os.path.basename(checkpoint)})",
        )
    launcher.wait_all()

    def evaluate(c):
        path = str(it_dir / f"cand{c}" / "visual.json")
        report = read_json(path)
        if report is None:
            data = review.load_failure_frames(str(it_dir / f"cand{c}" / "failures"), cfg.visual_frame_width)
            if data is None:
                report = {"error": "no failure frames could be recorded"}
            else:
                subtasks = [{"name": s["name"], "description": s["objective"], "success_criterion": ""}
                            for s in plans[c]["stages"]]
                try:
                    report = review.evaluate_failures(vlm, review_prompts, task_text, subtasks, data,
                                                   facts=scores[c].get("measured_facts", ""))
                except Exception as e:
                    report = {"error": f"VLM request failed: {e}"}
            write_json(path, report)
        return c, report

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        reports = dict(pool.map(evaluate, candidates))
    for c, report in reports.items():
        logging.info(f"{it_dir.name} candidate {c}: visual score {review.visual_score(report):.2f} (failed episodes)\n"
                     f"{review.format_report(report)}")
    return reports


def write_review(it_dir, it, candidates, plans, scores, results, reports, feedbacks, best):
    """A Markdown summary of the iteration for the human reviewer."""
    lines = [f"# Iteration {it}\n", f"Best candidate: {best}\n",
             "Write optional feedback (2-4 sentences) into `human_feedback.txt` in this folder; it is used in the "
             "next iteration and stored in the retrieval database.\n"]
    for c in candidates:
        lines.append(f"## Candidate {c}\n")
        lines.append(f"Scores: {outcome_summary(scores[c])}\n")
        lines.append("Curriculum: " + " -> ".join(s["name"] for s in plans[c]["stages"]) + "\n")
        lines.append("```\n" + outcomes_text(plans[c], results[c]) + "\n```\n")
        if c in reports:
            media = "`cand{c}/failures/` (frames and clips of failed episodes)" if (
                it_dir / f"cand{c}" / "failures").is_dir() else f"`cand{c}/video.mp4`"
            lines.append(f"Video: {media}\n\n```\n{review.format_report(reports[c])}```\n")
        lines.append(f"Analysis: {feedbacks[c]['analysis']}\n\nRecommendations: {feedbacks[c]['recommendations']}\n")
    (it_dir / "REVIEW.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.argv = ["visual=True" if arg == "--visual" else arg for arg in sys.argv]
    main()
