"""Train an IsaacLab task with a generated reward function.

This script registers ``Generated-<task>`` and then runs IsaacLab's own RSL-RL training script, so
all IsaacLab training options (``--num_envs``, ``--max_iterations``, ``--seed``, ``--run_name``,
Hydra overrides, ...) are available unchanged.

Example::

    python train.py --task <gym-id> --reward_file reward.py --isaaclab_root ~/IsaacLab \
        --num_envs 4096 --max_iterations 500
"""

import argparse
import json
import os
import runpy
import sys
import traceback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACEBACK_BEGIN = "=== TRAINING TRACEBACK BEGIN ==="
TRACEBACK_END = "=== TRAINING TRACEBACK END ==="


def main():
    parser = argparse.ArgumentParser(description="Train an IsaacLab task with a generated reward.")
    parser.add_argument("--task", required=True, help="Original IsaacLab gym task ID.")
    parser.add_argument("--reward_file", required=True, help="Python file that defines compute_reward(env).")
    parser.add_argument("--isaaclab_root", default=os.environ.get("ISAACLAB_PATH"), help="IsaacLab repository.")
    parser.add_argument("--stage_job", default=None,
                        help="JSON with the stage settings: exploration (overrides of RD_EXPLORATION), "
                             "reset_critic (bool), gamma")
    args, remaining = parser.parse_known_args()
    job = json.loads(args.stage_job) if args.stage_job else {}
    if job.get("exploration") and os.environ.get("RD_EXPLORATION"):
        settings = json.loads(os.environ["RD_EXPLORATION"])
        override = dict(job["exploration"])
        settings.update(override)
        os.environ["RD_EXPLORATION"] = json.dumps(settings)
        print(f"Exploration of this stage: {os.environ['RD_EXPLORATION']}", flush=True)
    if job.get("reset_critic"):
        os.environ["RD_RESET_CRITIC"] = "1"
    if job.get("gamma"):
        os.environ["RD_GAMMA"] = str(job["gamma"])  # discount of potential-based shaping terms
    if not args.isaaclab_root:
        parser.error("--isaaclab_root is required when ISAACLAB_PATH is not set.")

    train_script = os.path.join(args.isaaclab_root, "scripts", "reinforcement_learning", "rsl_rl", "train.py")
    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"IsaacLab RSL-RL training script not found: {train_script}")

    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.dirname(train_script))
    from isaaclab_backend import reward_patch

    if os.environ.get("RD_EXPLORATION"):
        # noise floor/ceiling and output squashing of the policy (see rd_lib/exploration.py); the floor can drop over
        # the run (min_std -> min_std_final), so it needs the number of rollout steps of the run
        if "--max_iterations" in remaining:
            os.environ["RD_EXPLORATION_ITERATIONS"] = remaining[remaining.index("--max_iterations") + 1]
        steps_override = [a for a in remaining if a.startswith("agent.num_steps_per_env=")]
        if steps_override:
            os.environ["RD_EXPLORATION_STEPS_PER_ITERATION"] = steps_override[-1].split("=", 1)[1]
        from rd_lib import exploration  # noqa: F401
    if os.environ.get("RD_OBS_FLAGS"):
        from rd_lib import obs_extension  # noqa: F401  (progress flags in the observations)

    if os.environ.get("RD_RESET_CRITIC"):
        # the reward changed: keep the actor (its skills) but start the critic and the optimizer fresh, so the value
        # function is not that of the previous stage's reward
        from rsl_rl.algorithms import ppo as ppo_module

        original_load = ppo_module.PPO.load

        def load_actor_only(self, loaded_dict, load_cfg, strict):
            cfg = dict(load_cfg or {"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": True})
            cfg.update(critic=False, optimizer=False)
            print("Loading the actor only (fresh critic and optimizer)", flush=True)
            return original_load(self, loaded_dict, cfg, strict)

        ppo_module.PPO.load = load_actor_only

    os.environ[reward_patch.REWARD_FILE_ENV] = os.path.abspath(args.reward_file)
    # Surface errors raised inside IsaacLab's Hydra wrapper with a full traceback.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    import isaaclab_tasks  # noqa: F401
    from isaaclab_backend.tasks_import import register_tasks

    register_tasks()  # the tasks of this repository (tasks/)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    steps_override = [a for a in remaining if a.startswith("agent.num_steps_per_env=")]
    os.environ[reward_patch.STEPS_PER_ITER_ENV] = (steps_override[-1].split("=", 1)[1] if steps_override
                                                   else str(agent_cfg.num_steps_per_env))
    task_id = reward_patch.register_generated_task(args.task)

    sys.argv = [train_script, "--task", task_id, *remaining]
    runpy.run_path(train_script, run_name="__main__")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        # Print a marked traceback for the pipeline, then exit hard: a crashed simulator can hang on shutdown.
        print(TRACEBACK_BEGIN, flush=True)
        traceback.print_exc(file=sys.stdout)
        print(TRACEBACK_END, flush=True)
        sys.stdout.flush()
        os._exit(1)
