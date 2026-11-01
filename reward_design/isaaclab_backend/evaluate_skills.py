"""Evaluate a chain of skill policies put together on the full task (skills.py).

Every environment runs the policy of its current skill. It starts with the first skill and moves on to the next one
when the current skill's end condition holds (all its end flags continuously for its end_hold_s), exactly the moment
at which that skill's training episodes ended and the next skill's start states were recorded. The last skill ends
with the task's own success. All skill policies take the same observation (the task's observation with the same
progress flags), so they can be switched in every step.

Reports the full task's success rate and condition breakdown (as isaaclab_backend/evaluate_checkpoint.py) and, per
skill, the share of episodes that reached it and the share of those that completed it (the handoff rates). Prints
one line starting with ``@@RD_SKILLS_EVAL@@`` (JSON).

    python -m isaaclab_backend.evaluate_skills --task <gym id> --skills skills.json --adapter <name> --headless

skills.json: [{"name": ..., "checkpoint": ..., "end_flags": [...], "end_hold_s": ...}, ...] in task order.
"""

import argparse
import importlib.metadata as metadata
import json
import os
import sys

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--skills", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes_per_env", type=int, default=3)
add_launcher_args(parser)
args_cli, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_backend.tasks_import import register_tasks  # noqa: E402

register_tasks()  # the tasks of this repository (tasks/)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

if os.environ.get("RD_SKILL"):
    raise SystemExit("unset RD_SKILL: the chain runs on the full task")
if os.environ.get("RD_EXPLORATION"):
    from rd_lib import exploration  # noqa: F401,E402  (the same output squashing as in training)
if os.environ.get("RD_OBS_FLAGS"):
    from rd_lib import obs_extension  # noqa: F401,E402  (progress flags in the observations)
from rd_lib.quantities import get_adapter, get_funnel  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

MARK = "@@RD_SKILLS_EVAL@@ "


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    skills = json.load(open(args_cli.skills))
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        actors = []
        for skill in skills:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(skill["checkpoint"])
            runner.alg.actor.eval()
            actors.append(runner.alg.actor)
        u = env.unwrapped
        n, dev, k_last = u.num_envs, u.device, len(skills) - 1
        quantities = get_adapter(args_cli.adapter)
        end_fns = [[quantities[f].fn for f in s.get("end_flags") or []] for s in skills]
        holds = [float(s.get("end_hold_s", 0.0)) for s in skills]
        funnel = get_funnel(args_cli.adapter)
        current = torch.zeros(n, dtype=torch.long, device=dev)  # skill index per environment
        end_time = torch.zeros(n, device=dev)
        furthest = torch.zeros(n, dtype=torch.long, device=dev)
        ever = torch.zeros(n, len(funnel), dtype=torch.bool, device=dev)
        steps_in = torch.zeros(n, len(skills), device=dev)
        totals = {"episodes": 0, "success": 0, "conditions": [0] * len(funnel),
                  "reached": [0] * len(skills), "completed": [0] * len(skills), "steps": [0.0] * len(skills)}
        original_reset = u._reset_idx

        def reset_idx(env_ids):
            if env_ids is not None and u.common_step_counter > 0:
                ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
                succeeded = u._succeeded[ids].bool()
                totals["episodes"] += len(ids)
                totals["success"] += int(succeeded.sum())
                for k in range(len(funnel)):
                    totals["conditions"][k] += int(ever[ids, k].sum())
                reach = torch.where(succeeded, torch.full_like(furthest[ids], k_last + 1), furthest[ids])
                for k in range(len(skills)):
                    totals["reached"][k] += int((reach >= k).sum())
                    totals["completed"][k] += int((reach >= k + 1).sum())
                    totals["steps"][k] += float(steps_in[ids, k].sum())
                ever[ids] = False
                current[ids] = 0
                end_time[ids] = 0.0
                furthest[ids] = 0
                steps_in[ids] = 0.0
            return original_reset(env_ids)

        u._reset_idx = reset_idx
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args_cli.episodes_per_env * int(u.max_episode_length)):
                actions = torch.zeros(n, env.num_actions, device=dev)
                for k, actor in enumerate(actors):
                    mask = current == k
                    if bool(mask.any()):
                        actions[mask] = actor(obs, stochastic_output=False)[mask]
                steps_in[torch.arange(n, device=dev), current] += 1.0
                obs, _, _, _ = env.step(actions)
                for k, (_, fn) in enumerate(funnel):
                    ever[:, k] |= fn(u).bool()
                # handoffs: the current skill's end condition (environments that were just reset start at skill 0)
                done = torch.zeros(n, dtype=torch.bool, device=dev)
                for k in range(k_last):
                    mask = current == k
                    if not bool(mask.any()) or not end_fns[k]:
                        continue
                    cond = torch.ones(n, dtype=torch.bool, device=dev)
                    for fn in end_fns[k]:
                        cond &= fn(u).bool()
                    t = torch.where(cond, end_time + u.step_dt, torch.zeros_like(end_time))
                    end_time = torch.where(mask, t, end_time)
                    done |= mask & (end_time >= holds[k] - 1e-6)
                current = torch.where(done, current + 1, current)
                end_time = torch.where(done, torch.zeros_like(end_time), end_time)
                furthest = torch.maximum(furthest, current)
        e = max(totals["episodes"], 1)
        per_skill = []
        for k, skill in enumerate(skills):
            reached = totals["reached"][k]
            per_skill.append({
                "name": skill["name"],
                "reached": reached / e,
                "completed": totals["completed"][k] / e,
                "completed_when_reached": totals["completed"][k] / reached if reached else None,
                "mean_time_s": totals["steps"][k] * u.step_dt / reached if reached else None,
            })
        result = {
            "episodes": totals["episodes"],
            "success_rate": totals["success"] / e,
            "conditions": {name: totals["conditions"][k] / e for k, (name, _) in enumerate(funnel)},
            "skills": per_skill,
        }
        sys.stdout.write(MARK + json.dumps(result) + "\n")
        sys.stdout.flush()
        env.close()


if __name__ == "__main__":
    main()
