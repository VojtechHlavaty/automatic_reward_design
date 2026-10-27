"""Runtime reward patching for IsaacLab environments.

This module runs inside the IsaacLab training process. It registers an ``Generated-<task>`` gym ID
that builds the original environment and replaces its reward with the function found in the
file named by the ``RD_REWARD_FILE`` environment variable.

The original (human-engineered) reward is still computed every step, so that task state updated
inside it keeps working and so that it can be logged as ``Generated/gt_reward``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import os
import sys

import gymnasium as gym
import torch

REWARD_FILE_ENV = "RD_REWARD_FILE"
STEPS_PER_ITER_ENV = "RD_STEPS_PER_ITER"
TASK_PREFIX = "Generated-"
REWARD_FN_NAME = "compute_reward"


def generated_task_id(task_id: str) -> str:
    return TASK_PREFIX + task_id


def register_generated_task(task_id: str) -> str:
    """Register ``Generated-<task_id>`` with the same config entry points as ``task_id``."""
    spec = gym.spec(task_id)
    new_id = generated_task_id(task_id)
    if new_id in gym.registry:
        return new_id
    kwargs = dict(spec.kwargs)
    kwargs["base_entry_point"] = spec.entry_point
    gym.register(
        id=new_id,
        entry_point=f"{__name__}:make_env",
        disable_env_checker=True,
        kwargs=kwargs,
    )
    return new_id


def load_reward_fn(path: str):
    """Load the reward function from a Python file.

    The function named ``compute_reward`` is used. When it is missing, the first function defined
    in the file is used.
    """
    spec = importlib.util.spec_from_file_location("generated_reward", path)
    module = importlib.util.module_from_spec(spec)
    # common names are available without imports in the generated reward files
    module.__dict__.update({"math": math, "torch": torch, "Tensor": torch.Tensor})
    spec.loader.exec_module(module)
    fn = getattr(module, REWARD_FN_NAME, None)
    if fn is None:
        functions = [
            f for _, f in inspect.getmembers(module, inspect.isfunction) if f.__module__ == module.__name__
        ]
        if not functions:
            raise ValueError(f"No reward function found in '{path}'.")
        fn = min(functions, key=lambda f: f.__code__.co_firstlineno)
    return fn


def _resolve_entry_point(entry_point):
    if callable(entry_point):
        return entry_point
    module_name, attr_name = entry_point.split(":")
    return getattr(importlib.import_module(module_name), attr_name)


class GeneratedRewardHook:
    """Computes the generated reward and logs its statistics once per RSL-RL iteration."""

    def __init__(self, env, reward_fn, steps_per_iter: int, log_dir: str | None):
        self.env = env
        self.reward_fn = reward_fn
        self.steps_per_iter = max(int(steps_per_iter), 1)
        self.writer = None
        # distributed training: only the first process writes the statistics (each process simulates its own share)
        if log_dir and int(os.environ.get("RANK", "0")) == 0:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(os.path.join(log_dir, "reward_terms"))
        self._sums: dict[str, torch.Tensor] = {}
        self._count = 0
        self._iteration = 0
        # batched statistics (default): one stacked mean per step instead of one mean per component, accumulated on
        # the GPU; the NaN/Inf check is accumulated too and raised at the end of the rollout window (before PPO uses it)
        self._batched = os.environ.get("RD_BATCHED_STATS", "1") != "0"
        self._keys: list[str] | None = None
        self._vector_sum: torch.Tensor | None = None
        self._bad = None
        self._fresh_rows = None

    def __call__(self, gt_reward: torch.Tensor) -> torch.Tensor:
        num_envs = self.env.num_envs
        output = self.reward_fn(self.env)
        if not (isinstance(output, tuple) and len(output) == 2):
            raise TypeError(
                "The reward function must return a tuple (total_reward, reward_components_dict), got"
                f" {type(output).__name__}."
            )
        reward, components = output
        if not isinstance(reward, torch.Tensor):
            raise TypeError(f"The total reward must be a torch.Tensor, got {type(reward).__name__}.")
        if reward.dim() == 2 and reward.shape[1] == 1:
            reward = reward.squeeze(1)
        if reward.shape != (num_envs,):
            raise ValueError(
                f"The total reward must have shape ({num_envs},) (one value per environment), got"
                f" {tuple(reward.shape)}."
            )
        if not isinstance(components, dict):
            raise TypeError(f"The reward components must be a dict, got {type(components).__name__}.")
        reward = reward.float()
        if self._batched:
            self._add_batched(reward, gt_reward, components)
            return reward
        if not torch.isfinite(reward).all():
            raise FloatingPointError("The total reward contains NaN or Inf values.")

        self._add("Generated/gpt_reward", reward)
        self._add("Generated/gt_reward", gt_reward)
        for name, value in components.items():
            if isinstance(value, torch.Tensor):
                self._add(f"Generated/{name}", value.float())
            elif isinstance(value, (int, float)):
                self._add(f"Generated/{name}", torch.tensor(float(value), device=reward.device))
        self._count += 1
        if self._count >= self.steps_per_iter:
            self._flush()
        return reward

    def _add_batched(self, reward, gt_reward, components):
        keys = ["Generated/gpt_reward", "Generated/gt_reward"]
        values = [reward, gt_reward.float()]
        for name, value in components.items():
            if isinstance(value, torch.Tensor):
                keys.append(f"Generated/{name}")
                values.append(value.detach().float())
            elif isinstance(value, (int, float)):
                keys.append(f"Generated/{name}")
                values.append(torch.tensor(float(value), device=reward.device))
        n = reward.shape[0]
        if all(v.shape == (n,) for v in values):
            stacked = torch.stack(values)
            means = stacked.mean(dim=1)
            hard = sys.modules.get("rd_lib.hard_starts")
            fresh = hard.fresh_mask() if hard is not None else None
            if fresh is not None:
                # with hard-case starts, the task's flag and quantity rates count only normal episodes
                if self._fresh_rows is None or self._fresh_rows[0] != keys:
                    rows = [i for i, k in enumerate(keys) if k.startswith(("Generated/metric_", "Generated/derived_"))]
                    self._fresh_rows = (keys, torch.tensor(rows, dtype=torch.long, device=reward.device))
                rows = self._fresh_rows[1]
                if len(rows):
                    w = fresh.float()
                    means[rows] = (stacked[rows] * w).sum(dim=1) / w.sum().clamp_min(1.0)
        else:
            means = torch.stack([v.mean() for v in values])
        bad = ~torch.isfinite(reward).all()
        self._bad = bad if self._bad is None else (self._bad | bad)
        if self._keys != keys:
            if self._vector_sum is not None:  # the component set changed: keep what was summed so far
                self._fold_vector()
            self._keys = keys
            self._vector_sum = means.clone()
        else:
            self._vector_sum += means
        self._count += 1
        if self._count >= self.steps_per_iter:
            self._flush()

    def _fold_vector(self):
        for key, total in zip(self._keys, self._vector_sum):
            self._sums[key] = self._sums[key] + total if key in self._sums else total.clone()
        self._keys, self._vector_sum = None, None

    def _add(self, key: str, value: torch.Tensor):
        value = value.detach().mean()
        if key in self._sums:
            self._sums[key] += value
        else:
            self._sums[key] = value.clone()

    def _flush(self):
        if self._batched:
            if self._bad is not None and bool(self._bad):
                raise FloatingPointError("The total reward contains NaN or Inf values.")
            self._bad = None
            if self._vector_sum is not None:
                self._fold_vector()
            if self.writer is not None and self._sums:
                keys = list(self._sums)
                values = (torch.stack([self._sums[k] for k in keys]) / self._count).tolist()  # one transfer
                for key, value in zip(keys, values):
                    self.writer.add_scalar(key, value, self._iteration)
                exploration = sys.modules.get("rd_lib.exploration")
                if exploration is not None:
                    for key, value in exploration.pop_stats().items():
                        self.writer.add_scalar(key, value, self._iteration)
                hard = sys.modules.get("rd_lib.hard_starts")
                if hard is not None and hard._active is not None:
                    for key, value in hard._active.pop_stats().items():
                        self.writer.add_scalar(key, value, self._iteration)
                self.writer.flush()
            self._sums = {}
            self._count = 0
            self._iteration += 1
            return
        if self.writer is not None:
            for key, total in self._sums.items():
                self.writer.add_scalar(key, (total / self._count).item(), self._iteration)
            exploration = sys.modules.get("rd_lib.exploration")
            if exploration is not None:  # action noise and saturation per action (pipeline exploration settings)
                for key, value in exploration.pop_stats().items():
                    self.writer.add_scalar(key, value, self._iteration)
            self.writer.flush()
        self._sums = {}
        self._count = 0
        self._iteration += 1


def install_reward_hook(env, hook: GeneratedRewardHook):
    """Replace the reward computation of a manager-based or direct IsaacLab environment."""
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

    if isinstance(env, ManagerBasedRLEnv):
        original_compute = env.reward_manager.compute

        def compute(dt: float) -> torch.Tensor:
            return hook(original_compute(dt=dt))

        env.reward_manager.compute = compute
    elif isinstance(env, DirectRLEnv):
        original_get_rewards = env._get_rewards

        def get_rewards() -> torch.Tensor:
            return hook(original_get_rewards())

        env._get_rewards = get_rewards
    else:
        raise TypeError(f"Unsupported environment type: {type(env).__name__}.")


def make_env(cfg, render_mode: str | None = None, base_entry_point=None, **kwargs):
    """Gym entry point: build the original environment and patch its reward."""
    env_cls = _resolve_entry_point(base_entry_point)
    if os.environ.get("RD_ROBOT_SPEED"):
        from rd_lib import robot_speed

        cfg = robot_speed.apply_cfg(cfg)
    env = env_cls(cfg=cfg, render_mode=render_mode, **kwargs)

    reward_file = os.environ[REWARD_FILE_ENV]
    reward_fn = load_reward_fn(reward_file)
    log_dir = getattr(cfg, "log_dir", None)
    steps_per_iter = int(os.environ.get(STEPS_PER_ITER_ENV, "24"))
    hook = GeneratedRewardHook(env, reward_fn, steps_per_iter, log_dir)
    install_reward_hook(env, hook)
    if os.environ.get("RD_ROBOT_SPEED"):
        robot_speed.patch_env(env)  # after the reward hook: scales the patched per-step reward
    if os.environ.get("RD_HARD_STARTS"):
        from rd_lib import hard_starts

        hard_starts.install(env)
    if os.environ.get("RD_PROFILE"):
        _install_profiler(env, hook)

    print(f"Generated reward file: {reward_file}", flush=True)
    print(f"Tensorboard Directory: {log_dir}", flush=True)
    return env


def _install_profiler(env, hook):
    """Profiling only (RD_PROFILE=1): GPU-synchronized timing of the parts of each environment step: the whole step,
    the original reward (with its logging), the generated reward function and the statistics. Synchronizing adds a little
    time itself; the shares are what matters. Prints a summary every 240 steps."""
    import time

    times = {"step": 0.0, "original reward": 0.0, "generated reward fn": 0.0, "statistics": 0.0}
    count = [0]

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    original_step = env.step

    def step(action):
        t0 = sync()
        out = original_step(action)
        times["step"] += sync() - t0
        count[0] += 1
        if count[0] % 240 == 0:
            n = count[0]
            total = times["step"]
            print("[RD_PROFILE] " + ", ".join(f"{k} {1000 * v / n:.2f} ms ({100 * v / total:.0f}%)"
                                                 for k, v in times.items()), flush=True)
        return out

    env.step = step
    original_get_rewards = env._get_rewards.__closure__  # (the hook wrapper)
    base = type(env)._get_rewards
    reward_fn = hook.reward_fn
    add = hook._add

    def timed_reward_fn(e):
        t0 = sync()
        out = reward_fn(e)
        times["generated reward fn"] += sync() - t0
        return out

    def timed_add(key, value):
        t0 = sync()
        add(key, value)
        times["statistics"] += sync() - t0

    hook.reward_fn = timed_reward_fn
    hook._add = timed_add
    add_batched = hook._add_batched

    def timed_add_batched(*args):
        t0 = sync()
        add_batched(*args)
        times["statistics"] += sync() - t0

    hook._add_batched = timed_add_batched

    def get_rewards():
        t0 = sync()
        gt = base(env)
        times["original reward"] += sync() - t0
        return hook(gt)

    env._get_rewards = get_rewards
