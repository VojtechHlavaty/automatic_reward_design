"""Append the task's progress flags to the policy observation (env option ``obs_progress_flags``).

A policy that has to behave differently before and after a task step (e.g. carry an object, then go home after
placing it) often cannot tell the phases apart from its observation alone. With this option every IsaacLab direct
environment gets the task's latched progress flags (from the task adapter) appended to its observation groups,
as 0/1 values, in every process that builds the environment (training, evaluation, checker, videos).

Checkpoints trained without the flags still load: the input layer and the observation normalizer are widened with
zero weights (mean 0, variance 1) for the new inputs, so the loaded policy behaves exactly as before until training
teaches it to use them. (The optimizer state of such a checkpoint is not loaded.)

The settings come from the environment variable RD_OBS_FLAGS (JSON: {"adapter": name, "flags": [flag, ...]}).
"""

import json
import os

import torch

ENV_VAR = "RD_OBS_FLAGS"
SETTINGS = json.loads(os.environ.get(ENV_VAR) or "{}")


def _install_env_patch(flag_fns):
    """Wrap the observations of every environment built with gymnasium.make (after the simulator has started)."""
    import gymnasium as gym

    original_make = gym.make

    def make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        base = env.unwrapped
        if not getattr(base, "_rd_obs_flags", False) and hasattr(base, "_get_observations"):
            get_obs = base._get_observations

            def get_observations():
                obs = get_obs()
                flags = torch.stack([fn(base).float() for fn in flag_fns], dim=-1)
                return {key: torch.cat((value, flags), dim=-1)
                        if isinstance(value, torch.Tensor) and value.dim() == 2 else value
                        for key, value in obs.items()}

            base._get_observations = get_observations
            base._rd_obs_flags = True
            print(f"{len(flag_fns)} progress flags appended to the observations", flush=True)
        return env

    gym.make = make


def widen_state_dict(loaded: dict, current: dict) -> bool:
    """Pad tensors of ``loaded`` whose last dimension is smaller than in ``current`` (new observation inputs):
    weights and means with zeros, variances and standard deviations with ones. Returns whether anything changed."""
    changed = False
    for key, target in current.items():
        value = loaded.get(key)
        if value is None or value.shape == target.shape or value.dim() != target.dim() or value.dim() == 0:
            continue
        if value.shape[:-1] != target.shape[:-1] or value.shape[-1] > target.shape[-1]:
            continue
        fill = 1.0 if key.endswith(("_var", "_std")) else 0.0
        pad = torch.full((*value.shape[:-1], target.shape[-1] - value.shape[-1]), fill, dtype=value.dtype,
                         device=value.device)
        loaded[key] = torch.cat((value, pad), dim=-1)
        changed = True
    return changed


def _install_load_patch():
    from rsl_rl.algorithms import ppo as ppo_module

    original_load = ppo_module.PPO.load

    def load(self, loaded_dict, load_cfg, strict):
        changed = False
        for key, model in (("actor_state_dict", self.actor), ("critic_state_dict", self.critic)):
            if key in loaded_dict:
                changed |= widen_state_dict(loaded_dict[key], model.state_dict())
        if changed:
            print("The checkpoint has fewer observation inputs; widened with zero weights for the progress "
                  "flags (optimizer state not loaded)", flush=True)
            load_cfg = dict(load_cfg or {"actor": True, "critic": True, "optimizer": True, "iteration": True,
                                         "rnd": True})
            load_cfg["optimizer"] = False
        return original_load(self, loaded_dict, load_cfg, strict)

    ppo_module.PPO.load = load


if SETTINGS.get("flags"):
    from rd_lib.quantities import get_adapter

    _quantities = get_adapter(SETTINGS["adapter"])
    _install_env_patch([_quantities[name].fn for name in SETTINGS["flags"]])
    _install_load_patch()
