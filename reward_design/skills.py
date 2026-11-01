"""Learn a task as a chain of skills, one pipeline run per skill, and put the skills together.

The task's env config lists the skills in task order (``skills:``). For each skill, in order:

1. A skill env config is written (cfg/env/skill_<run>_<k>.yaml): the task's config with a section that tells the
   planner and the stage writer which skill this run trains, where its episodes start and when they end; the
   progress flags of this skill only; the observation flags of the whole chain (so that all skill policies take the
   same observation).
2. The reward-design pipeline (pipeline.py) runs on it, with RD_SKILL set for all its processes
   (rd_lib/skill_starts.py): episodes end with success when the skill's end condition holds, and every episode of a
   skill after the first starts from a state in which the previous skill ended.
3. The best policy of that run is recorded on its own episodes (isaaclab_backend/record_skill_starts.py): the states in
   which it completes the skill are the start bank of the next skill.

A skill whose best policy completes it in less than ``--min_success`` of its episodes stops the chain. Finally the
skill policies are evaluated together on the full task (isaaclab_backend/evaluate_skills.py): each environment runs
the policy of its current skill and switches to the next one when the current skill's end condition holds.

    python skills.py --env g1_dex1_two_tables --llm openrouter_haiku5.5 --gpus 0,1 --init_checkpoint <model.pt>

Rerun with ``--run_dir`` of an earlier run to continue it (finished skills are kept). Progress: <run_dir>/skills.log,
the state in <run_dir>/skills_state.json, the summary in <run_dir>/summary.md.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parent
ENV_DIR = ROOT_DIR / "cfg" / "env"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def wait_for_empty_gpus(gpus):
    """Wait until no process (of any user) computes on these GPUs."""
    waited = False
    while True:
        out = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, gpus)), "--query-compute-apps=pid",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        if not out:
            if waited:
                logging.info(f"GPUs {gpus} are empty")
            return
        if not waited:
            logging.info(f"Waiting for GPUs {gpus} to be empty (processes: {out.split()})")
            waited = True
        time.sleep(60)


def marked_json(text, mark):
    for line in text.splitlines():
        if line.startswith(mark):
            return json.loads(line[len(mark):])
    return None


def skill_settings(base, skill, bank):
    return {"adapter": base.adapter, "end_flags": list(skill.get("end_flags") or []),
            "end_hold_s": float(skill.get("end_hold_s", 0.0)), "start_bank": bank,
            "start_prob": float(skill.get("start_prob", 1.0)), "match": list(base.get("skill_match") or [])}


def chain_text(skills):
    return "; ".join(f"{k + 1}. {s.name}: {s.objective}" for k, s in enumerate(skills))


def write_skill_env(base, skills, k, name, init_text):
    """The env config of skill k's pipeline run: the task's config with this skill's section."""
    skill = skills[k]
    cfg = OmegaConf.to_container(base, resolve=False)
    cfg.pop("skills", None)
    hold = float(skill.get("end_hold_s", 0.0))
    if skill.get("end_flags"):
        end = (f"An episode ends with success as soon as {skill.end_text} (all of the flags "
               f"{', '.join(skill.end_flags)} hold) continuously for {hold} s: the state in which skill {k + 2} starts. "
               f"The success flag 'succeeded' of this run is this end condition, not the task's placement.")
    else:
        end = "An episode ends with the task's own success (the cube placed in the tray, as described above)."
    cfg["description"] = (
        f"{base.description}\n"
        f"THIS RUN TRAINS ONE SKILL OF THE TASK. The task is learned as a chain of skills, each trained by its own "
        f"run and then put together (each skill's policy runs until its end condition holds, then the next skill's "
        f"policy takes over): {chain_text(skills)}. This run trains skill {k + 1} of {len(skills)}: {skill.name}. "
        f"Episodes of this run start {skill.start_text}. {end} Train only this skill: reward the progress towards its "
        f"end condition (the left wrist camera's aim at the right hand and the cube is part of every skill); the steps "
        f"after it are trained by the next skills and cannot be reached here, because the episode ends first.")
    flags = list(base.progress_flags)
    cfg["progress_flags"] = list(skill.progress_flags) + ["succeeded"]
    cfg["obs_flags"] = [f for f in flags if f != base.success_flag]
    if skill.get("start", "task") != "task":
        cfg["stage_randomization"] = {}  # the layout comes with the start states
    cfg["init_policy_description"] = init_text
    path = ENV_DIR / f"{name}.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    return path


def run_logged(cmd, log_path, env):
    logging.info(f"Running: {' '.join(cmd)[:400]} (log: {log_path})")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                              cwd=ROOT_DIR)
    return proc.returncode, Path(log_path).read_text(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="env config (cfg/env/<name>.yaml) with a skills: list")
    parser.add_argument("--llm", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--init_checkpoint", default=None, help="initial policy of the skills with init: initial")
    parser.add_argument("--init_description", default="", help="what the initial policy can do (for the planner)")
    parser.add_argument("--iterations", type=int, default=2, help="pipeline iterations per skill")
    parser.add_argument("--min_success", type=float, default=0.3, help="a skill below this success stops the chain")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--pipeline_args", nargs="*", default=[], help="extra Hydra overrides of every pipeline run")
    args = parser.parse_args()

    base = OmegaConf.load(ENV_DIR / f"{args.env}.yaml")
    skills = list(base.skills)
    gpus = [int(g) for g in args.gpus.split(",")]
    run_dir = Path(args.run_dir or ROOT_DIR / "outputs" / "skills" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(run_dir / "skills.log"), logging.StreamHandler(sys.stdout)])
    state_path = run_dir / "skills_state.json"
    state = read_json(state_path, {"env": args.env, "skills": [], "composed": None})
    python = sys.executable
    obs_flags = [f for f in base.progress_flags if f != base.success_flag]
    common_env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    common_env["RD_OBS_FLAGS"] = json.dumps({"adapter": base.adapter, "flags": obs_flags})
    if base.get("exploration"):
        common_env["RD_EXPLORATION"] = json.dumps(OmegaConf.to_container(base.exploration))
    for key in ("RD_SKILL", "RD_HARD_STARTS"):
        common_env.pop(key, None)
    logging.info(f"Skill chain of {args.env}: {chain_text(skills)} (run folder {run_dir})")

    bank = None
    for k, skill in enumerate(skills):
        done = state["skills"][k] if k < len(state["skills"]) else None
        if done and done.get("bank_done"):
            bank = done.get("bank")
            continue
        name = f"skill_{run_dir.name.replace('-', '_')}_{k}"
        skill_dir = run_dir / f"skill{k}_{skill.name}"
        record = done or {"name": skill.name, "dir": str(skill_dir)}
        if k >= len(state["skills"]):
            state["skills"].append(record)
        # initial policy: the configured one, or the final policy of the previous skill
        if skill.get("init", "initial") == "previous" and k > 0:
            prev = state["skills"][k - 1]
            init = prev["checkpoint"]
            init_text = (f"The initial policy is the final policy of the previous skill of this chain, "
                         f"{prev['name']} (it completes that skill in {round(100 * prev['success'])} percent of its "
                         f"episodes, without exploration noise); every episode of this run starts where that skill "
                         f"ended, so the policy already handles that state.")
        else:
            init = args.init_checkpoint
            init_text = args.init_description
        settings = skill_settings(base, skill, bank)
        env_vars = dict(common_env, RD_SKILL=json.dumps(settings))

        # 1-2. the pipeline run of this skill
        state_file = skill_dir / "pipeline_state.json"
        finished = read_json(state_file, {}).get("completed_iterations", 0) >= args.iterations
        if not finished:
            env_path = write_skill_env(base, skills, k, name, init_text)
            logging.info(f"Skill {k + 1}/{len(skills)} {skill.name}: env config {env_path}, start bank {bank}, "
                         f"initial policy {init}")
            cmd = [python, "pipeline.py", f"env={name}", f"llm={args.llm}", f"gpus=[{','.join(map(str, gpus))}]",
                   f"visual={args.visual}", f"iteration={args.iterations}", "num_eval=0", f"hydra.run.dir={skill_dir}"]
            if skill_dir.exists() and state_file.exists():
                cmd.append(f"resume={skill_dir}")
            if init:
                cmd.append(f"init_checkpoint={init}")
            cmd += list(args.pipeline_args)
            wait_for_empty_gpus(gpus)
            code, _ = run_logged(cmd, run_dir / f"skill{k}_pipeline.log", env_vars)
            if code != 0:
                logging.info(f"Skill {skill.name}: the pipeline exited with code {code}; see skill{k}_pipeline.log")
        best = (read_json(state_file, {}) or {}).get("best")
        if not best or not best.get("checkpoint"):
            logging.info(f"Skill {skill.name}: no trained policy; the chain stops here")
            break
        checkpoint = str((skill_dir / best["checkpoint"]).resolve()) if not os.path.isabs(best["checkpoint"]) \
            else best["checkpoint"]
        success = float((best.get("eval") or {}).get("success_rate", 0.0))
        record.update(checkpoint=checkpoint, success=success, eval=best.get("eval"))
        write_json(state_path, state)
        logging.info(f"Skill {skill.name}: best policy {checkpoint}, completes the skill in {success:.1%} of its "
                     f"episodes (deterministic)")
        if success < args.min_success:
            logging.info(f"Skill {skill.name} is below {args.min_success:.0%}: the chain stops here")
            break

        # 3. start bank of the next skill
        if k < len(skills) - 1:
            bank_path = run_dir / f"bank{k}_{skill.name}.pt"
            cmd = [python, "-m", "isaaclab_backend.record_skill_starts", "--task", base.task, "--checkpoint",
                   checkpoint, "--out", str(bank_path), "--headless"]
            wait_for_empty_gpus(gpus)
            code, text = run_logged(cmd, run_dir / f"bank{k}.log",
                                    dict(env_vars, CUDA_VISIBLE_DEVICES=str(gpus[0])))
            meta = marked_json(text, "@@RD_SKILL_BANK@@ ")
            if code != 0 or not meta or not meta.get("states"):
                logging.info(f"Skill {skill.name}: no start states recorded (see bank{k}.log); the chain stops here")
                break
            record.update(bank=str(bank_path), bank_meta=meta)
            bank = str(bank_path)
            logging.info(f"Skill {skill.name}: start bank of the next skill: {meta['states']} states "
                         f"({meta['success_rate']:.1%} of {meta['episodes']} recorded episodes completed the skill)")
        record["bank_done"] = True
        write_json(state_path, state)

    # 4. the skills put together on the full task
    trained = [s for s in state["skills"] if s.get("checkpoint")]
    if trained:
        chain = [{"name": s["name"], "checkpoint": s["checkpoint"],
                  "end_flags": list(skills[k].get("end_flags") or []),
                  "end_hold_s": float(skills[k].get("end_hold_s", 0.0))} for k, s in enumerate(trained)]
        chain_path = run_dir / "chain.json"
        write_json(chain_path, chain)
        cmd = [python, "-m", "isaaclab_backend.evaluate_skills", "--task", base.task, "--skills", str(chain_path),
               "--adapter", base.adapter, "--headless"]
        wait_for_empty_gpus(gpus)
        code, text = run_logged(cmd, run_dir / "composed_eval.log", dict(common_env, CUDA_VISIBLE_DEVICES=str(gpus[0])))
        state["composed"] = marked_json(text, "@@RD_SKILLS_EVAL@@ ")
        write_json(state_path, state)
        composed = state["composed"]
        if composed:
            logging.info(f"Skills together on the full task: success {composed['success_rate']:.1%}; "
                         + "; ".join(f"{s['name']}: reached {s['reached']:.0%}, completed {s['completed']:.0%}"
                                     for s in composed["skills"]))
    write_summary(run_dir, state, skills)


def write_summary(run_dir, state, skills):
    lines = [f"# Skill chain of {state['env']}", ""]
    for k, skill in enumerate(skills):
        s = state["skills"][k] if k < len(state["skills"]) else {}
        rate = f"{s['success']:.1%}" if "success" in s else "not trained"
        bank = f", start bank of the next skill: {s['bank_meta']['states']} states" if s.get("bank_meta") else ""
        lines.append(f"- {k + 1}. {skill.name}: completes the skill in {rate} of its episodes{bank}")
    composed = state.get("composed")
    if composed:
        lines += ["", f"Together on the full task: success {composed['success_rate']:.1%}", ""]
        for s in composed["skills"]:
            lines.append(f"- {s['name']}: reached in {s['reached']:.0%} of the episodes, completed in "
                         f"{s['completed']:.0%}")
        lines += ["", "Task conditions:"] + [f"- {name}: {v:.0%}" for name, v in composed["conditions"].items()]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
