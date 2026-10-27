"""Evaluate an RSL-RL checkpoint of a task: episode success rate and a breakdown by success condition.

Runs the deterministic policy (or the stochastic one) in many environments for a few episodes each and records every
episode's outcome when it ends, before the reset clears the task state (a success may end the episode, so the flag
is never visible after a step). For each success condition of the task (rd_lib.quantities.get_funnel) it reports
the share of episodes in which the condition was ever true. Prints one line starting with ``@@RD_EVAL@@`` (JSON).

    python -m isaaclab_backend.evaluate_checkpoint --task <gym id> --checkpoint model.pt --adapter <name> --headless
"""

import argparse
import importlib.metadata as metadata
import json
import os
import sys

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes_per_env", type=int, default=3)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--progress_scalars", default=None,
                    help="JSON list of {quantity, when, better}: the best value of each scalar quantity while the flag "
                         "'when' holds, per episode (partial progress towards a later step)")
add_launcher_args(parser)
args_cli, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_backend.tasks_import import register_tasks  # noqa: E402

register_tasks()  # the tasks of this repository (tasks/)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
if os.environ.get("RD_EXPLORATION"):
    from rd_lib import exploration  # noqa: F401,E402  (the same output squashing as in training)
if os.environ.get("RD_OBS_FLAGS"):
    from rd_lib import obs_extension  # noqa: F401,E402  (progress flags in the observations)
from rd_lib.quantities import get_adapter, get_funnel, get_hold  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

MARK = "@@RD_EVAL@@ "


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        actor = runner.alg.actor
        actor.eval()
        u = env.unwrapped
        funnel = get_funnel(args_cli.adapter)
        ever = torch.zeros(u.num_envs, len(funnel), dtype=torch.bool, device=u.device)
        # success hold (e.g. conditions that must hold continuously for 0.25 s): in how many episodes it starts, its
        # longest duration, and which condition ends it when it ends before success
        hold = get_hold(args_cli.adapter)
        hold_prev = torch.zeros(u.num_envs, device=u.device)
        hold_started = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
        hold_longest = torch.zeros(u.num_envs, device=u.device)
        hold_totals = {"episodes_started": 0, "longest": [], "breaks": 0,
                       "broken_by": [0] * (len(hold[2]) if hold else 0)}
        totals = {"episodes": 0, "success": 0, "conditions": [0] * len(funnel), "success_steps": []}

        original_reset = u._reset_idx

        def reset_idx(env_ids):
            if env_ids is not None and u.common_step_counter > 0:
                ids = torch.as_tensor(env_ids, device=u.device)
                totals["episodes"] += len(ids)
                succeeded = u._succeeded[ids] if hasattr(u, "_succeeded") else torch.zeros_like(ids, dtype=torch.bool)
                totals["success"] += int(succeeded.sum())
                totals["success_steps"] += u.episode_length_buf[ids[succeeded]].tolist()
                for k in range(len(funnel)):
                    totals["conditions"][k] += int(ever[ids, k].sum())
                ever[ids] = False
                if hold:
                    hold_totals["episodes_started"] += int(hold_started[ids].sum())
                    hold_totals["longest"] += hold_longest[ids][hold_started[ids]].tolist()
                    hold_started[ids] = False
                    hold_longest[ids] = 0.0
                    hold_prev[ids] = 0.0
            return original_reset(env_ids)

        u._reset_idx = reset_idx
        # partial progress: per episode, the best value of each declared scalar while its flag holds
        scalars = json.loads(args_cli.progress_scalars) if args_cli.progress_scalars else []
        adapter_quantities = get_adapter(args_cli.adapter) if scalars else {}
        best_now = [torch.full((u.num_envs,), float("nan"), device=u.device) for _ in scalars]
        scalar_results = [[] for _ in scalars]
        reset_inner = u._reset_idx

        def reset_with_scalars(env_ids):
            if env_ids is not None and scalars:
                ids = torch.as_tensor(env_ids, device=u.device)
                for k in range(len(scalars)):
                    values = best_now[k][ids]
                    scalar_results[k] += values[~torch.isnan(values)].tolist()
                    best_now[k][ids] = float("nan")
            return reset_inner(env_ids)

        u._reset_idx = reset_with_scalars
        # where the policy spends its time: the share of steps in which each flag that is not latched holds now (a
        # latched flag only says that a step happened once), and the share of steps in which a joint is at a position
        # limit (e.g. a policy that parks an arm or the torso against a limit)
        all_quantities = get_adapter(args_cli.adapter)
        state_flags = [(name, q.fn) for name, q in all_quantities.items() if q.kind == "flag" and not q.latched]
        state_counts = torch.zeros(len(state_flags), device=u.device)
        limit_counts = {}
        limit_steps = [0]
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args_cli.episodes_per_env * int(u.max_episode_length)):
                obs, _, _, _ = env.step(actor(obs, stochastic_output=args_cli.stochastic))
                for k, (_, fn) in enumerate(funnel):
                    ever[:, k] |= fn(u).bool()
                for k, (_, fn) in enumerate(state_flags):
                    try:
                        state_counts[k] += fn(u).float().mean()
                    except Exception:  # noqa: BLE001
                        pass
                limit_steps[0] += 1
                for art_name, art in u.scene.articulations.items():
                    limits = art.data.soft_joint_pos_limits.torch
                    pos = art.data.joint_pos.torch
                    span = (limits[..., 1] - limits[..., 0]).clamp_min(0.0)
                    tol = torch.clamp(0.02 * span, min=0.01)
                    at_limit = ((pos - limits[..., 0] < tol) | (limits[..., 1] - pos < tol)) & (span > 1e-3)
                    share = at_limit.float().mean(dim=0)
                    counts = limit_counts.setdefault(art_name, torch.zeros_like(share))
                    counts += share
                    low, high = limit_counts.setdefault(art_name + "/range", [pos.min(dim=0).values.clone(),
                                                                              pos.max(dim=0).values.clone()])
                    torch.minimum(low, pos.min(dim=0).values, out=low)
                    torch.maximum(high, pos.max(dim=0).values, out=high)
                for k, spec in enumerate(scalars):
                    when = adapter_quantities[spec["when"]].fn(u).bool() if spec.get("when") else torch.ones(
                        u.num_envs, dtype=torch.bool, device=u.device)
                    value = adapter_quantities[spec["quantity"]].fn(u).float()
                    current = best_now[k]
                    better = torch.minimum(current, value) if spec.get("better", "lower") == "lower" else torch.maximum(
                        current, value)
                    best_now[k] = torch.where(when, torch.where(torch.isnan(current), value, better), current)
                if hold:
                    timer = hold[0](u).float()
                    hold_started |= timer > 0
                    hold_longest = torch.maximum(hold_longest, timer)
                    broke = (hold_prev > 0) & (timer == 0)
                    if bool(broke.any()):
                        hold_totals["breaks"] += int(broke.sum())
                        for k, (_, fn) in enumerate(hold[2]):
                            hold_totals["broken_by"][k] += int((broke & ~fn(u).bool()).sum())
                    hold_prev = timer
        n = max(totals["episodes"], 1)
        steps = sorted(totals["success_steps"])
        result = {
            "episodes": totals["episodes"],
            "success_rate": totals["success"] / n,
            "conditions": {name: totals["conditions"][k] / n for k, (name, _) in enumerate(funnel)},
            "median_time_to_success_s": steps[len(steps) // 2] * u.step_dt if steps else None,
            "policy": "stochastic" if args_cli.stochastic else "deterministic",
        }
        steps_run = max(limit_steps[0], 1)
        result["time_share"] = {name: round(float(state_counts[k]) / steps_run, 4)
                                for k, (name, _) in enumerate(state_flags)}
        result["joints_at_limit"] = {}
        for art_name, counts in limit_counts.items():
            if art_name.endswith("/range"):
                continue
            names = u.scene.articulations[art_name].joint_names
            low, high = limit_counts[art_name + "/range"]
            moved = ((high - low) > 0.02).tolist()  # joints the policy never moves (fixed or uncontrolled) are left out
            for j, share in enumerate((counts / steps_run).tolist()):
                if share >= 0.05 and moved[j]:
                    result["joints_at_limit"][names[j]] = round(share, 3)
        if scalars:
            result["progress_scalars"] = {}
            for k, spec in enumerate(scalars):
                values = sorted(scalar_results[k])
                result["progress_scalars"][spec["quantity"]] = {
                    "when": spec.get("when"), "better": spec.get("better", "lower"),
                    "median_best": round(values[len(values) // 2], 4) if values else None,
                    "episodes": len(values)}
        if hold:
            longest = sorted(hold_totals["longest"])
            b = max(hold_totals["breaks"], 1)
            result["success_hold"] = {
                "required_s": hold[1],
                "episodes_started": hold_totals["episodes_started"] / n,
                "median_longest_s": longest[len(longest) // 2] if longest else None,
                "breaks": hold_totals["breaks"],
                "broken_by": {name: hold_totals["broken_by"][k] / b for k, (name, _) in enumerate(hold[2])},
            }
        sys.stdout.write(MARK + json.dumps(result) + "\n")
        sys.stdout.flush()
        env.close()


if __name__ == "__main__":
    main()
