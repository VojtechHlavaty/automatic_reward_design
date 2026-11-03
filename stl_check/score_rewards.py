"""Replay one suite file and compute the return of each LLM-generated reward on every episode.

The environment is built as when recording (same task, number of environments, seed and contact sensor), reset with
the recorded seed, and driven with the recorded actions; the task is deterministic for a fixed seed and fixed
actions, so the replay reaches the recorded states (the script reports the largest deviation of the cube position
as a check). Every reward is computed at the point where the task computes its own reward (inside
``reward_manager.compute``, before the resets of the step), as in training (reward_design/isaaclab_backend/
reward_patch.py); the return of an episode is the sum over its valid steps.

    python stl_check/score_rewards.py --suite suite/hover.pt --rewards <stage0.yaml> <stage1.yaml> \\
        --out suite/hover_returns.json --headless

A reward is a pipeline stage YAML (compiled with rd_lib.reward_registry) or a Python file defining
compute_reward(env).
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--suite", required=True)
parser.add_argument("--rewards", nargs="+", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--reward_design", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "reward_design"), help="the reward-design pipeline (for rd_lib)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, args_cli.reward_design)
os.environ.setdefault("RD_SUCCESS_FLAG", "succeeded")  # as the pipeline sets it for this task (env.success_flag)

from stl_check.predicates import read_state  # noqa: E402
from stl_check.env_setup import make_env  # noqa: E402


def load_reward(path):
    if path.endswith((".yaml", ".yml")):
        from rd_lib.reward_registry import build_reward_fn

        return build_reward_fn(path)
    from isaaclab_backend.reward_patch import load_reward_fn

    return load_reward_fn(path)


def main():
    suite = torch.load(args_cli.suite)
    env = make_env(suite["task"], suite["num_envs"], suite["seed"], args_cli.device)
    u = env.unwrapped
    from isaaclab_backend.tasks_import import register_tasks

    register_tasks()  # the task adapters (tasks/franka_lift) for the stage YAMLs
    fns = [load_reward(p) for p in args_cli.rewards]
    per_step = []  # this step's reward of each reward function
    original_compute = u.reward_manager.compute

    def compute(dt):
        out = original_compute(dt=dt)
        values = []
        for fn in fns:
            r = fn(u)
            values.append((r[0] if isinstance(r, tuple) else r).detach().float())
        per_step.append(torch.stack(values))  # [rewards, E]
        return out

    u.reward_manager.compute = compute
    env.reset(seed=suite["seed"])
    actions = suite["actions"].to(u.device)  # [E, T, 8]
    valid = suite["valid"].to(u.device)  # [E, T]
    saved_cube = suite["traj"]["cube_pos"].to(u.device)
    deviation = 0.0
    with torch.inference_mode():
        for k in range(actions.shape[1]):
            cube = read_state(u)["cube_pos"]
            dev = (cube - saved_cube[:, k]).norm(dim=-1)[valid[:, k]]
            if dev.numel():
                deviation = max(deviation, float(dev.max()))
            env.step(actions[:, k])
    rewards = torch.stack(per_step, dim=-1)  # [rewards, E, T]
    returns = (rewards * valid.unsqueeze(0)).sum(dim=-1)  # [rewards, E]
    result = {"suite": os.path.abspath(args_cli.suite), "kind": suite["kind"],
              "replay_max_cube_deviation_m": deviation,
              "returns": {path: returns[i].tolist() for i, path in enumerate(args_cli.rewards)}}
    with open(args_cli.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"@@STL_RETURNS {suite['kind']}: {len(fns)} rewards, replay deviation {deviation:.2e} m -> {args_cli.out}",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
