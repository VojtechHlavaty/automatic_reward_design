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
compute_reward(env). A stage YAML of a run with level pay gets the run's level pay settings; such runs need the
reward-design code that has rd_lib/level_pay.py (--reward_design).
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
parser.add_argument("--train_task", default="Isaac-Lift-Cube-Franka-v0",
                    help="the task the rewards were trained on (its PPO gamma discounts the returns); the scripted "
                         "episodes run in the IK-Abs variant, which has no PPO settings")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, args_cli.reward_design)
os.environ.setdefault("RD_SUCCESS_FLAG", "succeeded")  # as the pipeline sets it for this task (env.success_flag)

from stl_check.predicates import read_state  # noqa: E402
from stl_check.env_setup import make_env  # noqa: E402


def level_pay_of(path):
    """The level pay settings (JSON) of the pipeline run a stage YAML comes from (its "Level pay:" line in
    <run>/pipeline.log), or None. With level pay on, the training reward of a stage is the LLM's terms plus the
    progress levels the framework pays (rd_lib/level_pay.py), so the replay pays them too."""
    log = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "pipeline.log")
    if not os.path.exists(log):
        return None
    for line in open(log, errors="replace"):
        if "Level pay: " in line:
            return line.split("Level pay: ", 1)[1].strip()
    return None


def stage_gamma(path):
    """The PPO discount a stage YAML sets for its training (training.gamma), or None (the task's default)."""
    if not path.endswith((".yaml", ".yml")):
        return None
    import yaml

    gamma = ((yaml.safe_load(open(path)) or {}).get("training") or {}).get("gamma")
    return None if gamma is None else float(gamma)


def load_reward(path):
    if path.endswith((".yaml", ".yml")):
        from rd_lib.reward_registry import build_reward_fn

        level_pay = level_pay_of(path)  # read when the reward function is built
        if level_pay:
            os.environ["RD_LEVEL_PAY"] = level_pay
        else:
            os.environ.pop("RD_LEVEL_PAY", None)
        # the discount of the potential terms, as in training: the stage's own gamma, else rd_lib's default (0.99)
        gamma = stage_gamma(path)
        if gamma is not None:
            os.environ["RD_GAMMA"] = str(gamma)
        else:
            os.environ.pop("RD_GAMMA", None)

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
    if suite.get("checkpoint"):
        env.reset()  # the recording's RSL-RL wrapper (RslRlVecEnvWrapper) resets once more when it is created
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
    # the discounted return from the start of the episode, which PPO maximizes: the stage's gamma, else the PPO
    # default of the training task. Potential terms pay gamma * phi' - phi per step, so the plain sum charges (1 - gamma) * phi for
    # every step spent at a high potential; discounted, they add up to gamma^T phi_T - phi_0 as intended.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task_gamma = float(load_cfg_from_registry(args_cli.train_task, "rsl_rl_cfg_entry_point").algorithm.gamma)
    gammas = [stage_gamma(p) or task_gamma for p in args_cli.rewards]
    steps = torch.arange(rewards.shape[-1], device=rewards.device, dtype=torch.float32)
    discount = torch.stack([g ** steps for g in gammas]).unsqueeze(1)  # [rewards, 1, T]
    discounted = (rewards * valid.unsqueeze(0) * discount).sum(dim=-1)
    result = {"suite": os.path.abspath(args_cli.suite), "kind": suite["kind"],
              "replay_max_cube_deviation_m": deviation,
              "returns": {path: returns[i].tolist() for i, path in enumerate(args_cli.rewards)},
              "gamma": dict(zip(args_cli.rewards, gammas)),
              "discounted_returns": {path: discounted[i].tolist() for i, path in enumerate(args_cli.rewards)}}
    with open(args_cli.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"@@STL_RETURNS {suite['kind']}: {len(fns)} rewards, replay deviation {deviation:.2e} m -> {args_cli.out}",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
