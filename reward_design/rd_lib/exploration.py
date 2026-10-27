"""Exploration settings of RSL-RL's Gaussian policy, applied by the pipeline to every training run of a task.

With actions clipped to [-1, 1] two things go wrong without them:

* an entropy bonus grows the action noise without bound, and without any bonus the noise collapses before a skill is
  discovered; so the noise is clamped per action to [min_std, max_std];
* a policy can drive an output far beyond the clip range (e.g. a gripper held closed); the opposite action is then
  never sampled and the clipping gives no gradient towards it, so selected outputs are squashed smoothly to
  +-limit (limit * tanh(x / limit)), which keeps them explorable.

The settings come from the environment variable RD_EXPLORATION (JSON), set by the pipeline from the env config::

    {"min_std": 0.3, "max_std": 1.0, "min_std_per_action": {"8": 0.8}, "squash": {"8": 1.5}}

Importing this module installs the settings (in the training, evaluation and checker processes alike) and keeps
statistics per action (mean noise, share of steps in which the squashed mean is outside [-1, 1], i.e. saturated)
that the reward hook logs as Exploration/std_<i> and Exploration/saturated_<i>.
"""

import json
import os

import torch
from torch.distributions import Normal

import rsl_rl.modules.distribution as distribution

ENV_VAR = "RD_EXPLORATION"
SETTINGS = json.loads(os.environ.get(ENV_VAR) or "{}")
MIN_STD = float(SETTINGS.get("min_std", 0.0))
MAX_STD = float(SETTINGS.get("max_std", 1.0e9))
MIN_STD_PER_ACTION = {int(k): float(v) for k, v in (SETTINGS.get("min_std_per_action") or {}).items()}
SQUASH = {int(k): float(v) for k, v in (SETTINGS.get("squash") or {}).items()}
# noise fixed per action (floor and ceiling at once), e.g. a stage that needs a quiet gripper
FIXED_STD_PER_ACTION = {int(k): float(v) for k, v in (SETTINGS.get("fixed_std_per_action") or {}).items()}

# optional schedule: the floor min_std drops linearly to min_std_final over the training run (actions with their own
# floor in min_std_per_action keep it); late in a run, the policy can then hold still and be precise, which conditions
# such as "the arm is still at its target" need. The number of PPO iterations comes from the training process.
MIN_STD_FINAL = SETTINGS.get("min_std_final")
STEPS_PER_ITERATION = int(os.environ.get("RD_EXPLORATION_STEPS_PER_ITERATION", 24))
TOTAL_ROLLOUT_STEPS = int(os.environ.get("RD_EXPLORATION_ITERATIONS", 0)) * STEPS_PER_ITERATION

_stats = {"std": None, "saturated": None, "count": 0}
_rollout_steps = {"n": 0}


def current_min_std():
    if MIN_STD_FINAL is None or TOTAL_ROLLOUT_STEPS <= 0:
        return MIN_STD
    progress = min(_rollout_steps["n"] / TOTAL_ROLLOUT_STEPS, 1.0)
    return MIN_STD + (float(MIN_STD_FINAL) - MIN_STD) * progress


def _squash(mlp_output: torch.Tensor) -> torch.Tensor:
    if not SQUASH:
        return mlp_output
    mean = mlp_output.clone()
    for index, limit in SQUASH.items():
        mean[..., index] = limit * torch.tanh(mlp_output[..., index] / limit)
    return mean


def _std(self, mean: torch.Tensor) -> torch.Tensor:
    raw = self.std_param if self.std_type == "scalar" else torch.exp(self.log_std_param)
    low = torch.full_like(raw, current_min_std())
    std = torch.maximum(raw, low).clamp(max=MAX_STD)
    # the task's per-action floors (e.g. a gripper that must keep trying to open while it holds an object) apply after
    # the stage's cap: a stage that lowers the noise for precision does not stop that action from exploring (a stage
    # can still fix an action's noise with per_action)
    if MIN_STD_PER_ACTION:
        floors = torch.zeros_like(std).detach()
        for index, value in MIN_STD_PER_ACTION.items():
            if index < floors.shape[-1]:
                floors[index] = value
        std = torch.maximum(std, floors)  # out of place: std is part of the autograd graph
    if FIXED_STD_PER_ACTION:
        std = std.clone()
        for index, value in FIXED_STD_PER_ACTION.items():
            if index < std.shape[-1]:
                std[index] = value
    return std.expand_as(mean)


def _record(mean: torch.Tensor, std: torch.Tensor):
    with torch.no_grad():
        flat_mean = mean.reshape(-1, mean.shape[-1])
        saturated = (flat_mean.abs() > 1.0).float().mean(0)
        noise = std.reshape(-1, std.shape[-1])[0]
        if _stats["count"] == 0:
            _stats["std"], _stats["saturated"] = noise.clone(), saturated.clone()
        else:
            _stats["std"] += noise
            _stats["saturated"] += saturated
        _stats["count"] += 1


def _update(self, mlp_output: torch.Tensor) -> None:
    mean = _squash(mlp_output)
    std = _std(self, mean)
    if not torch.is_grad_enabled():  # rollouts (not the PPO update): record what the policy explores
        _record(mean, std)
        _rollout_steps["n"] += 1
    self._distribution = Normal(mean, std)


_deterministic = distribution.GaussianDistribution.deterministic_output


def _deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    return _deterministic(self, _squash(mlp_output))


def pop_stats():
    """Mean noise and saturated share per action since the last call ({} if none), then reset."""
    if _stats["count"] == 0:
        return {}
    out = {}
    for i, (s, sat) in enumerate(zip((_stats["std"] / _stats["count"]).tolist(),
                                     (_stats["saturated"] / _stats["count"]).tolist())):
        out[f"Exploration/std_{i}"] = s
        out[f"Exploration/saturated_{i}"] = sat
    _stats.update(std=None, saturated=None, count=0)
    return out


if SETTINGS:
    distribution.GaussianDistribution.update = _update
    distribution.GaussianDistribution.deterministic_output = _deterministic_output
