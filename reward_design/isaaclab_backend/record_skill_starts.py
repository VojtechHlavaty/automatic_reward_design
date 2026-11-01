"""Record the start bank of the next skill: the states in which this skill's policy ends its episodes successfully.

Runs a skill's final policy (deterministic) in the skill's own episodes (RD_SKILL: its start bank and end condition,
rd_lib/skill_starts.py) and keeps the full scene state of every environment at the moment its episode succeeds,
before the reset clears it. Saves {"state": {key: [M, ...]}, "signature": [M, k], "meta": {...}} with torch.save and
prints one line starting with ``@@RD_SKILL_BANK@@`` (JSON: episodes, success rate, states kept).

    python -m isaaclab_backend.record_skill_starts --task <gym id> --checkpoint model.pt --out bank.pt --headless
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
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_states", type=int, default=8192)
parser.add_argument("--max_episodes_per_env", type=int, default=8)
parser.add_argument("--stochastic", action="store_true")
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

if os.environ.get("RD_EXPLORATION"):
    from rd_lib import exploration  # noqa: F401,E402  (the same output squashing as in training)
if os.environ.get("RD_OBS_FLAGS"):
    from rd_lib import obs_extension  # noqa: F401,E402  (progress flags in the observations)
from rd_lib import skill_starts  # noqa: E402  (needs RD_SKILL: the skill's start bank and end condition)
from rd_lib.hard_starts import SceneState  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

MARK = "@@RD_SKILL_BANK@@ "


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if not skill_starts.SETTINGS:
        raise SystemExit("RD_SKILL is not set: the skill's end condition is needed")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        actor = runner.alg.actor
        actor.eval()
        u = env.unwrapped
        io = SceneState(u)
        match = list(skill_starts.SETTINGS.get("match") or [])
        kept = {"states": [], "signatures": [], "count": 0}
        totals = {"episodes": 0, "success": 0}
        inner_reset = u._reset_idx

        def reset_idx(env_ids):
            if env_ids is not None and u.common_step_counter > 0:
                ids = torch.as_tensor(env_ids, device=u.device, dtype=torch.long)
                succeeded = u._succeeded[ids].bool()
                totals["episodes"] += len(ids)
                totals["success"] += int(succeeded.sum())
                s_ids = ids[succeeded]
                room = args_cli.max_states - kept["count"]
                if len(s_ids) and room > 0:
                    s_ids = s_ids[:room]
                    if io.tensor_names is None:
                        io._find_tensors()
                    keys = set(skill_starts.bank_keys(io))
                    state = {k: v for k, v in io._read(s_ids, local=True).items()
                             if not k.startswith("env.") or k in keys}
                    kept["states"].append({k: v.cpu() for k, v in state.items()})
                    kept["signatures"].append(skill_starts.signature(io, s_ids, match).cpu())
                    kept["count"] += len(s_ids)
            return inner_reset(env_ids)

        u._reset_idx = reset_idx
        obs = env.get_observations()
        steps = args_cli.max_episodes_per_env * int(u.max_episode_length)
        with torch.inference_mode():
            for _ in range(steps):
                obs, _, _, _ = env.step(actor(obs, stochastic_output=args_cli.stochastic))
                if kept["count"] >= args_cli.max_states:
                    break
        result = {"episodes": totals["episodes"], "success_rate": totals["success"] / max(totals["episodes"], 1),
                  "states": kept["count"], "checkpoint": args_cli.checkpoint, "skill": skill_starts.SETTINGS}
        if kept["count"]:
            state = {k: torch.cat([s[k] for s in kept["states"]]) for k in kept["states"][0]}
            torch.save({"state": state, "signature": torch.cat(kept["signatures"]), "meta": result}, args_cli.out)
        sys.stdout.write(MARK + json.dumps(result) + "\n")
        sys.stdout.flush()
        env.close()


if __name__ == "__main__":
    main()
