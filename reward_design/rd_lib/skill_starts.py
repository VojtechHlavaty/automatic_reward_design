"""Skill episodes: one skill of a task chain (skills.py), trained and evaluated on its own.

A task such as "walk to a table, pick a cube, carry it to another table, place it" is split into skills that are
learned one after another, each by its own pipeline run with its own reward. In the processes of a skill's run
(training, evaluation, checker, videos) this module changes the task's episodes in two ways:

* End condition: the episode succeeds (``env._succeeded``, which ends it) when all of the skill's end flags (task
  quantities of the adapter) hold continuously for ``end_hold_s``. Without end flags the task's own success stays (the
  last skill).
* Start states: every reset restarts the environment, with probability ``start_prob``, from a state in which the
  previous skill ended (a start bank recorded by isaaclab_backend/record_skill_starts.py with that skill's final
  policy). The bank holds full scene states (rd_lib/hard_starts.SceneState) relative to the environment origin, so a
  state can go into any environment. Properties fixed per environment at spawn time (e.g. the cube's size and the
  tray's shape: ``match`` tensors) are not written; a state is taken from the bank entries whose match tensors equal
  the environment's own (falling back to entries that match the first match tensor, then to any entry).

Settings come from the environment variable RD_SKILL (JSON)::

    {"adapter": "<task adapter>", "end_flags": ["lifted_once", ...], "end_hold_s": 0.25,
     "start_bank": "<path of bank.pt>" or null, "start_prob": 1.0, "match": ["_item_size", "_tray_inner"]}

On import (when RD_SKILL is set) gymnasium.make is wrapped so that every environment built in the process gets it;
the training entry point (isaaclab_backend/reward_patch.py) installs it itself, before hard-case starts.
"""

from __future__ import annotations

import json
import os

import torch

ENV_VAR = "RD_SKILL"
SETTINGS = json.loads(os.environ.get(ENV_VAR) or "{}")
# helper objects of the pipeline itself (reward hook, statistics): their state belongs to a run, not to a scene state
_OWN_MODULES = ("isaaclab_backend", "rd_lib")

_active = None


def bank_keys(io):
    """State keys a start bank stores: the scene and the task's tensors, without the pipeline's own helper objects."""
    keys = []
    for path in io.tensor_names:
        owner = io.env if len(path) == 1 else getattr(io.env, path[0])
        if len(path) > 1 and type(owner).__module__.split(".")[0] in _OWN_MODULES:
            continue
        keys.append("env." + ".".join(path))
    return keys


def signature(io, ids, match):
    """Float tensor [len(ids), k]: the spawn-time properties of these environments (concatenated match tensors)."""
    parts = []
    for name in match:
        value = getattr(io.env, name, None)
        if isinstance(value, torch.Tensor):
            parts.append(value[ids].reshape(len(ids), -1).float())
    if not parts:
        return torch.zeros(len(ids), 0, device=io.device)
    return torch.cat(parts, dim=-1)


class SkillEpisodes:
    def __init__(self, env, settings):
        from rd_lib.hard_starts import SceneState
        from rd_lib.quantities import get_adapter

        self.env = env
        self.n = env.num_envs
        self.device = env.device
        quantities = get_adapter(settings["adapter"])
        self.end_flags = list(settings.get("end_flags") or [])
        self.end_fns = [quantities[f].fn for f in self.end_flags]
        self.end_hold = float(settings.get("end_hold_s", 0.0))
        self.start_prob = float(settings.get("start_prob", 1.0))
        self.match = list(settings.get("match") or [])
        self.end_time = torch.zeros(self.n, device=self.device)
        self.io = SceneState(env)
        self.bank = None
        path = settings.get("start_bank")
        if path:
            data = torch.load(path, map_location=self.device)
            self.bank = data["state"]
            self.bank_signature = data["signature"].to(self.device).float()
            self.bank_size = int(self.bank_signature.shape[0])
            print(f"Skill start bank: {path} ({self.bank_size} states)", flush=True)
        self.counts = {"starts": 0, "exact": 0, "partial": 0, "any": 0}
        self._choices = None  # per environment: bank entries that fit it (exact, first match tensor, any)

    # -- end condition (after the task's own progress update, before the terminations) ---------------------------
    def update_progress(self, original):
        original()
        if not self.end_fns:
            return
        holds = torch.ones(self.n, dtype=torch.bool, device=self.device)
        for fn in self.end_fns:
            holds &= fn(self.env).bool()
        self.end_time = torch.where(holds, self.end_time + self.env.step_dt, torch.zeros_like(self.end_time))
        self.env._succeeded = self.end_time >= self.end_hold - 1e-6

    # -- start states ---------------------------------------------------------------------------------------------
    def _fit(self):
        """For every environment the bank entries it can start from: same spawn properties, else same first property
        (e.g. the cube size), else any."""
        sig = signature(self.io, torch.arange(self.n, device=self.device), self.match)
        bank = self.bank_signature
        exact = (sig.unsqueeze(1) - bank.unsqueeze(0)).abs().amax(dim=-1) < 1e-5 if sig.shape[1] else None
        first = None
        if self.match and sig.shape[1]:
            width = getattr(self.env, self.match[0]).reshape(self.n, -1).shape[1]
            first = (sig[:, :width].unsqueeze(1) - bank[:, :width].unsqueeze(0)).abs().amax(dim=-1) < 1e-5
        self._choices = (exact, first)

    def _pick(self, ids):
        """A random fitting bank entry per environment, and the counts of each kind of fit."""
        exact, first = self._choices
        scores = torch.rand(len(ids), self.bank_size, device=self.device)
        kind = torch.full((len(ids),), 2, device=self.device, dtype=torch.long)  # 2 = any entry
        if first is not None:
            has_first = first[ids].any(dim=-1)
            kind[has_first] = 1
            scores = torch.where(first[ids] | ~has_first.unsqueeze(-1), scores, scores - 2.0)
        if exact is not None:
            has_exact = exact[ids].any(dim=-1)
            kind[has_exact] = 0
            scores = torch.where(exact[ids] | ~has_exact.unsqueeze(-1), scores, scores - 4.0)
        return scores.argmax(dim=-1), kind

    def reset(self, original, env_ids):
        out = original(env_ids)
        ids = (torch.arange(self.n, device=self.device) if env_ids is None
               else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        self.end_time[ids] = 0.0
        if self.bank is None or not len(ids):
            return out
        if self.io.tensor_names is None:
            self.io._find_tensors()
            self._fit()
        go = ids[torch.rand(len(ids), device=self.device) < self.start_prob]
        if len(go):
            entry, kind = self._pick(go)
            state = {k: v[entry] for k, v in self.bank.items()}
            self.io._write(go, state, local=True, skip={"env." + m for m in self.match})
            self.end_time[go] = 0.0
            if isinstance(getattr(self.env, "_succeeded", None), torch.Tensor):
                self.env._succeeded[go] = False  # the state was recorded at the previous skill's success
            self.counts["starts"] += len(go)
            for k, name in enumerate(("exact", "partial", "any")):
                self.counts[name] += int((kind == k).sum())
            if not getattr(self, "_reported", False):
                self._reported = True
                share = [round(float((kind == k).float().mean()), 3) for k in range(3)]
                print(f"Skill starts from the bank: {len(go)} environments, fit exact / first property / any = "
                      f"{share}", flush=True)
        return out

    def pop_stats(self):
        c = self.counts
        stats = {}
        if c["starts"]:
            stats["SkillStarts/share_exact_fit"] = c["exact"] / c["starts"]
        self.counts = {k: 0 for k in c}
        return stats


def install(env):
    """Install the skill's episodes on a DirectRLEnv when RD_SKILL is set (once per env). Returns the instance."""
    global _active
    if not SETTINGS or getattr(env, "_rd_skill", None) is not None:
        return getattr(env, "_rd_skill", None)
    skill = SkillEpisodes(env, SETTINGS)
    original_progress = env._update_task_progress

    def update_task_progress():
        skill.update_progress(original_progress)

    env._update_task_progress = update_task_progress
    original_reset = env._reset_idx

    def reset_idx(env_ids):
        return skill.reset(original_reset, env_ids)

    env._reset_idx = reset_idx
    env._rd_skill = skill
    _active = skill
    print(f"Skill episodes: {SETTINGS}", flush=True)
    return skill


def _install_make_patch():
    import gymnasium as gym

    original_make = gym.make

    def make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        if hasattr(env.unwrapped, "_reset_idx"):
            install(env.unwrapped)
        return env

    gym.make = make


if SETTINGS:
    _install_make_patch()
