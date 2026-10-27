"""Live check of LLM-written derived quantities on real environment states (an IsaacLab worker for the pipeline).

The worker builds the task with a few environments, reports the environment interface (robot bodies and joints,
assets, tensor attributes) and then answers check requests: for every request it resets the environments, runs
the initial policy (or random actions) for a short rollout and evaluates the derived quantities on the visited
states. It reports errors (exceptions, wrong type or shape, NaN, constant values, values outside the declared
range) and value statistics. Protocol: JSON lines on stdin; answers are stdout lines starting with ``@@CHECK@@``.

    python -m isaaclab_backend.derived_check_server --task <gym id> --headless [--init_checkpoint model.pt]
"""

import argparse
import json
import os
import sys
import traceback

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rollout_steps", type=int, default=240, help="control steps per check (4 s)")
parser.add_argument("--sample_every", type=int, default=8)
parser.add_argument("--init_checkpoint", default=None, help="RSL-RL checkpoint whose policy visits the states")
add_launcher_args(parser)
args_cli, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_backend.tasks_import import register_tasks  # noqa: E402

register_tasks()  # the tasks of this repository (tasks/)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from rd_lib import helpers  # noqa: E402
from rd_lib.derived import check_value, compile_derived, static_errors  # noqa: E402
from rd_lib.quantities import get_adapter  # noqa: E402
from rd_lib.reward_registry import FUNCTIONS, build_reward_fn_from_spec  # noqa: E402
if os.environ.get("RD_EXPLORATION"):
    from rd_lib import exploration  # noqa: F401,E402  (the same output squashing as in training)
if os.environ.get("RD_OBS_FLAGS"):
    from rd_lib import obs_extension  # noqa: F401,E402  (progress flags in the observations)

ORDER_MIN_SAMPLES = 40  # samples a progress level needs before the reward ordering is judged
ORDER_QUANTILE = 0.75  # levels are compared by this quantile of the reward per second: the states in which the step
# is done well (latched flags such as a valid lift stay true after a later drop, which lowers a level's mean)
ORDER_MIN_GAIN = 0.3  # a later level must earn at least this share more per second than the level before
NO_PAY_LEVEL = 0.01  # a shaping or potential term below this share of its maximum in 95% of the states where its
# gates are open gives the policy no direction there
BALANCE_SECONDS = 2.0  # a goal (potential) term must be able to pay at least this many seconds of the per-second reward
# the policy already earns where the term applies (else it cannot pull the policy anywhere)
BALANCE_MIN_RATE = 0.5  # ... judged only where that per-second reward is at least this large
GAP_MIN_TRANSITIONS = 8  # observed step changes needed before the reward across a step change is judged
GAP_TOLERANCE = 0.2  # share of the earlier reward per second that may be lost at a step change
GAP_MIN_LOSS = 1.0  # a loss (reward over the planning horizon) smaller than this is not judged: noise of the samples

MARK = "@@CHECK@@ "


def send(obj):
    sys.stdout.write(MARK + json.dumps(obj) + "\n")
    sys.stdout.flush()


def env_interface(u):
    """Bodies, joints, assets and tensor attributes of the environment, for the LLM prompt."""
    assets = {}
    for name, value in vars(u).items():
        data = getattr(value, "data", None)
        if data is None or not hasattr(data, "root_pos_w"):
            continue
        if hasattr(value, "joint_names") and hasattr(value, "body_names"):
            assets[name] = {"type": "articulation", "bodies": list(value.body_names), "joints": list(value.joint_names)}
        else:
            assets[name] = {"type": "rigid object"}
    tensors = {}
    for name, value in vars(u).items():
        if isinstance(value, torch.Tensor) and name.startswith("_") and value.dim() >= 1 and value.shape[0] == u.num_envs:
            tensors[name] = f"{str(value.dtype).replace('torch.', '')} {tuple(value.shape[1:]) or 'per env'}"
    return {"assets": assets, "tensors": tensors, "num_envs_in_check": u.num_envs}


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        policy = None
        if args_cli.init_checkpoint:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(args_cli.init_checkpoint)
            actor = runner.alg.actor
            actor.eval()
            policy = lambda obs: actor(obs, stochastic_output=True)  # noqa: E731
        send({"ready": True, "interface": env_interface(u)})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            try:
                if request.get("profile"):
                    send({"id": request["id"], **profile(env, u, policy, request)})
                else:
                    send({"id": request["id"], **check(env, u, policy, request)})
            except Exception:  # noqa: BLE001
                send({"id": request["id"], "errors": ["internal checker error: " + traceback.format_exc()[-1500:]],
                      "stats": {}})
        env.close()


def profile(env, u, policy, request):
    """Values of every registered scalar quantity on the states of the policy, by task progress level (the last
    progress flag that is true): {quantity: {level: [p5, median, p95, samples]}} (levels with too few samples are
    left out). Tells the LLM the real ranges, e.g. how far a joint is from its target after an earlier step."""
    quantities = get_adapter(request["adapter"])
    flags = list(request.get("progress_flags") or [])
    names = ["(no progress flag)"] + flags
    scalars = [n for n, q in quantities.items() if q.kind == "scalar"]
    values = {n: [[] for _ in names] for n in scalars}
    with torch.inference_mode():
        for _ in range(int(request.get("episodes", 2))):
            env.reset()
            obs = env.get_observations() if policy is not None else None
            for step in range(int(u.max_episode_length)):
                if policy is not None:
                    obs, _, _, _ = env.step(policy(obs))
                else:
                    env.step(2.0 * torch.rand(u.num_envs, u.single_action_space.shape[0], device=u.device) - 1.0)
                if step % args_cli.sample_every:
                    continue
                level = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
                for i, flag in enumerate(flags):
                    level = torch.where(quantities[flag].fn(u).bool(), torch.full_like(level, i + 1), level)
                for n in scalars:
                    try:
                        v = quantities[n].fn(u).float()
                    except Exception:  # noqa: BLE001
                        continue
                    for i in range(len(names)):
                        sel = v[level == i]
                        if sel.numel():
                            values[n][i].append(sel.cpu())
    result = {}
    for n in scalars:
        per_level = {}
        for i, name in enumerate(names):
            if not values[n][i]:
                continue
            v = torch.cat(values[n][i])
            if v.numel() < ORDER_MIN_SAMPLES:
                continue
            q = torch.quantile(v, torch.tensor([0.05, 0.5, 0.95]))
            per_level[name] = [round(float(q[0]), 3), round(float(q[1]), 3), round(float(q[2]), 3), int(v.numel())]
        if per_level:
            result[n] = per_level
    return {"profile": result, "errors": [], "stats": {}}


def check(env, u, policy, request):
    items = request["derived"]
    quantities = get_adapter(request["adapter"])
    errors, compiled = [], {}
    for item in items:
        e = static_errors(item)
        if e:
            errors += e
            continue
        try:
            compiled[item["name"]] = compile_derived(item, helpers)
        except Exception as ex:  # noqa: BLE001
            errors.append(f"derived quantity '{item['name']}': the code does not load: {type(ex).__name__}: {ex}")
    if errors:
        return {"errors": errors, "stats": {}}
    # reward ordering over the task's progress levels (level = the last progress flag that is true)
    flags = list(request.get("progress_flags") or [])
    reward_fn, event_terms, potential_terms = None, [], []
    if flags and request.get("stage"):
        try:
            reward_fn = build_reward_fn_from_spec(dict(request["stage"], adapter=request["adapter"]))
            event_terms = [t["name"] for t in request["stage"]["reward"]["terms"] if t["kind"] == "event"]
            potential_terms = [t["name"] for t in request["stage"]["reward"]["terms"] if t["kind"] == "potential"]
        except Exception as ex:  # noqa: BLE001
            return {"errors": [f"the stage reward does not build: {type(ex).__name__}: {ex}"], "stats": {}}
    level_sum = torch.zeros(len(flags) + 1, device=u.device)
    level_count = torch.zeros(len(flags) + 1, device=u.device)
    level_values = [[] for _ in range(len(flags) + 1)]
    # first step at which each environment reached each level, and rising edges of the event terms' flags
    first_reach = torch.full((u.num_envs, len(flags) + 1), -1, dtype=torch.long, device=u.device)
    event_edges = {}
    event_prev = {}
    samples = {item["name"]: [] for item in items}
    failed = set()
    # terms that should give the policy a direction: shaping and potential terms with a positive weight on a scalar
    # quantity; their level (fill) and quantity where their gates are open
    terms = request["stage"]["reward"]["terms"] if (reward_fn is not None and request.get("stage")) else []
    graded = [t for t in terms if t["kind"] in ("shaping", "potential") and float(t["weight"]) > 0
              and t.get("fn") in FUNCTIONS and FUNCTIONS[t["fn"]][3] != "flag"]
    fill_values = {t["name"]: [] for t in graded}
    time_pay_where = {t["name"]: [] for t in graded}  # per-second reward (time-based terms) where the term applies
    graded_quantity = {t["name"]: [] for t in graded}
    # reward per second around step changes (a progress level rising between two samples): the rate two samples
    # before the change (still at the earlier level) and the lowest rate over the change
    prev_level = torch.full((u.num_envs,), -1, dtype=torch.long, device=u.device)
    prev_rates = [torch.zeros(u.num_envs, device=u.device), torch.zeros(u.num_envs, device=u.device)]
    transitions = {}
    # per-term reward per second at each progress level (for the reward-hacking review), on request
    term_table = bool(request.get("term_table"))
    table_terms = [t["name"] for t in terms if t["kind"] != "event"] if term_table else []
    term_level_sum = {name: torch.zeros(len(flags) + 1, device=u.device) for name in table_terms}
    with torch.inference_mode():
        env.reset()  # fresh episodes for every check
        obs = env.get_observations() if policy is not None else None
        for step in range(args_cli.rollout_steps):
            if policy is not None:
                obs, _, _, _ = env.step(policy(obs))
            else:
                actions = 2.0 * torch.rand(u.num_envs, u.single_action_space.shape[0], device=u.device) - 1.0
                env.step(actions)
            if reward_fn is not None:
                for term in request["stage"]["reward"]["terms"]:
                    if term["kind"] != "event" or term["quantity"] not in quantities:
                        continue
                    flag_now = quantities[term["quantity"]].fn(u).bool()
                    prev = event_prev.get(term["name"], torch.zeros_like(flag_now))
                    edges = event_edges.setdefault(term["name"], torch.zeros(u.num_envs, device=u.device))
                    edges += (flag_now & ~prev).float()
                    event_prev[term["name"]] = flag_now
            # the reward is evaluated in every step (potential terms pay the change since the previous call), and
            # recorded in the sampled steps
            step_reward = None
            if reward_fn is not None:
                try:
                    step_reward = reward_fn(u)
                except Exception as ex:  # noqa: BLE001
                    errors.append(f"the stage reward raised {type(ex).__name__} on live states: {ex}")
                    reward_fn = None
            if step % args_cli.sample_every:
                continue
            cache = {}

            def q(name):
                if name in cache:
                    return cache[name]
                if name in compiled:
                    value = compiled[name](u, q)
                elif name in quantities:
                    value = quantities[name].fn(u)
                else:
                    raise KeyError(f"unknown quantity '{name}' in q(); registered: {sorted(quantities)}")
                cache[name] = value
                return value

            for item in items:
                name = item["name"]
                if name in failed:
                    continue
                try:
                    value = q(name)
                except Exception as ex:  # noqa: BLE001
                    tb = traceback.extract_tb(ex.__traceback__)
                    where = next((f" (line {f.lineno})" for f in reversed(tb) if f.filename.startswith("<derived")), "")
                    errors.append(f"derived quantity '{name}' raised {type(ex).__name__}{where}: {ex}")
                    failed.add(name)
                    continue
                message = check_value(item, value, u.num_envs)
                if message:
                    errors.append(message)
                    failed.add(name)
                    continue
                samples[name].append(value.reshape(-1).float().cpu())
            if step_reward is not None:
                total, components = step_reward
                zero = torch.zeros_like(total, dtype=torch.float)
                # without the events: what the policy earns per step; without the potential terms as well: what it
                # earns per second for being at a level (potentials pay progress, not time, and are left out of the
                # comparison of the levels)
                paid = total.float() - sum((components[n].float() for n in event_terms if n in components), zero)
                continuous = paid - sum((components[n].float() for n in potential_terms if n in components), zero)
                level = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
                for i, flag in enumerate(flags):
                    level = torch.where(quantities[flag].fn(u).bool(), torch.full_like(level, i + 1), level)
                rate = paid / u.step_dt
                fresh = getattr(u, "episode_length_buf", None)
                steady = (fresh >= 3 * args_cli.sample_every) if fresh is not None else torch.ones_like(level, dtype=torch.bool)
                moved = (prev_level >= 0) & (level > prev_level) & steady
                for env_id in torch.nonzero(moved).flatten().tolist():
                    key = (int(prev_level[env_id]), int(level[env_id]))
                    entry = transitions.setdefault(key, ([], [], []))
                    entry[0].append(float(prev_rates[0][env_id]))
                    entry[1].append(min(float(prev_rates[1][env_id]), float(rate[env_id])))
                    entry[2].append(step)
                prev_level = level.clone()
                prev_rates = [prev_rates[1], rate.clone()]
                for t in graded:
                    fill, gate = components.get("fill_" + t["name"]), components.get("gate_" + t["name"])
                    if fill is None or gate is None:
                        continue
                    gate = gate.bool()
                    if bool(gate.any()):
                        fill_values[t["name"]].append(fill.float()[gate].cpu())
                        time_pay_where[t["name"]].append((continuous / u.step_dt).float()[gate].cpu())
                        try:
                            graded_quantity[t["name"]].append(q(t["quantity"]).float()[gate].cpu())
                        except Exception:  # noqa: BLE001
                            pass
                rows = torch.arange(u.num_envs, device=u.device)
                unset = first_reach[rows, level] < 0
                first_reach[rows[unset], level[unset]] = step
                level_sum.index_add_(0, level, continuous / u.step_dt)
                for name in table_terms:
                    if name in components:
                        term_level_sum[name].index_add_(0, level, components[name].float() / u.step_dt)
                level_count.index_add_(0, level, torch.ones_like(continuous))
                for i in range(len(flags) + 1):
                    level_values[i].append((continuous / u.step_dt)[level == i].float().cpu())
    stats = {}
    if flags and level_count.sum() > 0:
        names = ["(no progress flag)"] + flags
        good = [float(torch.quantile(torch.cat(v), ORDER_QUANTILE)) if level_count[i] > 0 else 0.0
                for i, v in enumerate(level_values)]
        per_level = {names[i]: {"reward_per_s": round(float(level_sum[i] / level_count[i]), 4),
                                f"reward_per_s_p{round(100 * ORDER_QUANTILE)}": round(good[i], 4),
                                "samples": int(level_count[i])}
                     for i in range(len(names)) if level_count[i] > 0}
        stats["reward_by_progress_level"] = per_level
        if term_table:
            table = {}
            for i in range(len(names)):
                if level_count[i] < ORDER_MIN_SAMPLES:
                    continue
                row = {name: round(float(term_level_sum[name][i] / level_count[i]), 3) for name in table_terms}
                table[names[i]] = {k: v for k, v in row.items() if abs(v) >= 0.005}
            stats["terms_by_progress_level"] = table
        reached = [i for i in range(len(names)) if level_count[i] >= ORDER_MIN_SAMPLES]
        # a reward that pays mainly through events and potential terms (progress, not time) has small per-second rates
        # at every level; comparing those rates says nothing about progress, so the ordering is not judged then
        terms = request["stage"]["reward"]["terms"] if request.get("stage") else []
        progress_pay = sum(abs(float(t["weight"])) for t in terms if t["kind"] in ("event", "potential"))
        episode_s = float(u.max_episode_length) * u.step_dt
        if progress_pay > 0 and max((abs(good[i]) for i in reached), default=0.0) < 0.05 * progress_pay / episode_s:
            reached = []
        for a, b in zip(reached, reached[1:]):
            ra, rb = good[a], good[b]
            if rb < ra + ORDER_MIN_GAIN * abs(ra) - 1e-3:
                errors.append(
                    f"reward ordering: on live states, the policy earns {rb:.3g}/s at progress level '{names[b]}' but "
                    f"{ra:.3g}/s at the earlier level '{names[a]}' ({round(100 * ORDER_QUANTILE)}th percentile of the "
                    f"reward per second, event bonuses excluded). Reaching '{names[b]}' must pay at least "
                    f"{round(100 * ORDER_MIN_GAIN)}% more than staying at '{names[a]}': keep paying the earlier steps' "
                    "terms, pay the new step more, and stop the credit for holding the earlier state from growing")
    # finishing must pay: when success ends the episode, staying just before success must not earn more (discounted,
    # until the episode ends) than the success event pays
    audit = request.get("audit") or {}
    success_flag = audit.get("success_flag")
    terminates = bool(getattr(u.cfg, "terminate_on_success", audit.get("terminates", True)))
    if flags and level_count.sum() > 0 and success_flag and terminates and request.get("stage"):
        gamma = float(request["stage"].get("training", {}).get("gamma") or audit.get("gamma", 0.99))
        candidates = [i for i in range(len(names)) if level_count[i] >= ORDER_MIN_SAMPLES
                      and names[i] != success_flag]
        if candidates:
            pre = max(candidates)
            rate = good[pre]
            reached = first_reach[:, pre]
            reached = reached[reached >= 0]
            start = float(reached.float().median()) if reached.numel() else 0.0
            n = max(int(u.max_episode_length - start), 1)
            stream = rate * u.step_dt * (1.0 - gamma ** n) / (1.0 - gamma)
            bonus = sum(float(t["weight"]) for t in request["stage"]["reward"]["terms"]
                        if t["kind"] == "event" and t["quantity"] == success_flag)
            # only for stages that pay for success: an intermediate stage (no success event) does not end its
            # episodes at its own goal, so there is no finishing to avoid
            if bonus > 0 and stream > bonus + 1e-3:
                errors.append(
                    f"reward ordering (finishing): success ends the episode, but staying at the furthest level the "
                    f"policy reaches ('{names[pre]}', about {rate:.3g}/s) until the episode ends is worth about "
                    f"{stream:.3g} (discounted, from step {int(start)} of {int(u.max_episode_length)}), more than the "
                    f"success event pays ({bonus:.3g}). The policy then avoids finishing. Make the success event larger "
                    "than that, or pay the progress towards success with potential terms instead of per-second credit "
                    "that continues while waiting")
    # goal terms too weak to matter: a potential term pays at most weight * (1 - phi) for the progress still ahead;
    # if that is worth less than a few seconds of the per-second reward the policy earns anyway where the term
    # applies, reaching its goal is not worth leaving that state (and costs or noise on the way dominate)
    for t in graded:
        if t["kind"] != "potential" or not fill_values[t["name"]] or not time_pay_where[t["name"]]:
            continue
        phi = torch.cat(fill_values[t["name"]])
        pay = torch.cat(time_pay_where[t["name"]])
        if phi.numel() < ORDER_MIN_SAMPLES:
            continue
        remaining = float(t["weight"]) * (1.0 - float(phi.median()))
        rate = float(torch.quantile(pay, 0.5))
        stats.setdefault("goal_terms", {})[t["name"]] = {
            "level_where_it_applies": round(float(phi.median()), 4), "max_payout_ahead": round(remaining, 3),
            "per_second_reward_there": round(rate, 3), "samples": int(phi.numel())}
        if rate >= BALANCE_MIN_RATE and remaining < BALANCE_SECONDS * rate:
            errors.append(
                f"reward ordering (weak goal term): potential term '{t['name']}' ({t['fn']} of {t['quantity']}, weight "
                f"{t['weight']}) can pay at most about {remaining:.3g} in total for the progress still ahead (its level is "
                f"{float(phi.median()):.3g} of 1 where it applies), but the policy already earns about {rate:.3g}/s there "
                f"from per-second terms, so the whole goal is worth less than {BALANCE_SECONDS:g} s of staying put; "
                "penalties and noise on the way dominate and the policy does not move. Give it a weight of at least several "
                "seconds of that per-second reward, choose a function that changes over the values where it applies, "
                "and make sure no penalty grows along the way to its goal")
    # terms that pay nothing where they apply: flat at 0 over the states the policy visits there
    for t in graded:
        if not fill_values[t["name"]]:
            continue
        v = torch.cat(fill_values[t["name"]])
        if v.numel() < ORDER_MIN_SAMPLES or float(torch.quantile(v, 0.95)) >= NO_PAY_LEVEL:
            continue
        where = ""
        if graded_quantity[t["name"]]:
            x = torch.cat(graded_quantity[t["name"]])
            p = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))
            where = (f"; there {t['quantity']} is {float(p[0]):.3g} to {float(p[2]):.3g} (5th to 95th percentile, median "
                     f"{float(p[1]):.3g})")
        errors.append(
            f"reward ordering (no pay): term '{t['name']}' ({t['kind']}, {t['fn']} of {t['quantity']}) pays almost "
            f"nothing where it applies: its gates are open in {v.numel()} checked states, but it is below "
            f"{round(100 * NO_PAY_LEVEL)}% of its maximum in 95% of them{where}. A term that is flat at 0 gives the policy "
            "no direction: choose the function parameters so that it changes over the values the policy actually "
            "visits there; a narrow bonus near the goal can be kept in addition to such a term")
    # reward gaps at step changes: the reward drops when a step is done, because the credit of the earlier step ends
    # before the credit of the next step starts (event bonuses of the new step count against the loss)
    if transitions and request.get("stage"):
        names = ["(no progress flag)"] + flags
        gamma = float(request["stage"].get("training", {}).get("gamma") or (request.get("audit") or {}).get("gamma", 0.99))
        for (a, b), (before, during, steps) in sorted(transitions.items()):
            if len(before) < GAP_MIN_TRANSITIONS:
                continue
            rb = sum(before) / len(before)
            rd = sum(during) / len(during)
            loss_rate = rb - rd - GAP_TOLERANCE * abs(rb)
            if loss_rate <= 1e-3:
                continue
            # how long the lower rate lasts is unknown: count it for the time a policy plans ahead (1 / (1 - gamma) steps,
            # at most the rest of the episode), against the events that pay for reaching the new level
            remaining = max(int(u.max_episode_length) - int(sum(steps) / len(steps)), 1)
            horizon = min(1.0 / (1.0 - gamma), remaining)
            loss = loss_rate * u.step_dt * horizon
            bonus = sum(float(t["weight"]) for t in request["stage"]["reward"]["terms"]
                        if t["kind"] == "event" and t["quantity"] in names[a + 1:b + 1] and float(t["weight"]) > 0)
            if loss > bonus + GAP_MIN_LOSS:
                errors.append(
                    f"reward ordering (gap at a step change): when the policy gets from '{names[a]}' to '{names[b]}' "
                    f"({len(before)} times on live states), the reward per second drops from {rb:.3g} shortly before to "
                    f"{rd:.3g} during the change (event bonuses excluded); over the next {int(horizon)} steps that loses "
                    f"about {loss:.3g}, more than the events for the new step pay ({bonus:.3g}). The credit of the "
                    "earlier step ends before the credit of the next step starts, so the policy learns to avoid the "
                    "step: keep paying the earlier step (or give continuity credit) until the next step's credit has "
                    "taken over, and pay the intermediate states of the change at least as much as the state before")
        stats["step_changes_observed"] = {f"{names[a]} -> {names[b]}": len(v[0]) for (a, b), v in sorted(transitions.items())}
    for name, edges in event_edges.items():
        fired = edges[edges > 0]
        if fired.numel() >= 4 and float(fired.mean()) > 1.5:
            errors.append(
                f"reward ordering (repeated event): event '{name}' fired {float(fired.mean()):.1f} times per episode on "
                "average where it fired at all: its flag can be switched off and on again, so the bonus can be "
                "collected repeatedly. Use a latched flag or a different event")
    for item in items:
        name = item["name"]
        if name in failed or not samples[name]:
            continue
        values = torch.cat(samples[name])
        if item["kind"] == "flag":
            stats[name] = {"rate": round(float(values.mean()), 4)}
            continue
        p = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95]))
        stats[name] = {"min": round(float(values.min()), 4), "p5": round(float(p[0]), 4), "median": round(float(p[1]), 4),
                       "p95": round(float(p[2]), 4), "max": round(float(values.max()), 4)}
        if float(values.std()) < 1e-6:
            # not an error: a quantity about a later event (e.g. a drop after a lift) can be constant on the
            # states the current policy reaches; the note goes back to the LLM with the stats
            stats[name]["note"] = (
                "constant on all checked states (the current policy may never reach the situation it measures; if "
                "it should already vary, check the bodies, frames and formulas)")
        elif "range" in item:
            lo, hi = item["range"]
            tol = 0.01 * (hi - lo)
            outside = float(((values < lo - tol) | (values > hi + tol)).float().mean())
            if outside > 0.05:
                errors.append(
                    f"derived quantity '{name}': {100 * outside:.0f}% of the checked values are outside the declared "
                    f"range [{lo}, {hi}] (observed min {stats[name]['min']}, 5% {stats[name]['p5']}, median "
                    f"{stats[name]['median']}, 95% {stats[name]['p95']}, max {stats[name]['max']}); fix the formula "
                    "or the range")
    return {"errors": errors, "stats": stats}


if __name__ == "__main__":
    main()
