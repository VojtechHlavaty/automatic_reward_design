"""Hard-case starts: more training on the start conditions in which the policy fails.

A plain PPO run spends most of its episodes on start conditions the policy already handles. This module keeps, for
every environment, the start states of its recent failed episodes and restarts a share of the episodes from one of
them, so the policy practises the cases it fails.

How it works (task-independent; it only uses the IsaacLab scene and the env's own tensors):

* Right after every reset, the start state of the new episode is snapshotted: the root pose and velocity of every
  articulation and rigid object of the scene, the joint positions and velocities of every articulation, and every
  per-environment tensor attribute of the env (first dimension = number of environments, e.g. sampled goal poses,
  object sizes, timers), so that the task's internal state matches the restored scene.
* When an episode ends without the success flag, its start state is stored in that environment's slots (a ring of
  ``slots`` entries). Start states stay in the environment they came from, so per-environment properties fixed at
  spawn time (e.g. the object's size) always match.
* At a reset, with probability ``prob``, an environment with stored start states restarts from a random one of them.
  A stored start is dropped when the policy succeeds from it or after ``max_tries`` failed replays.
* Replayed episodes are left out of the training statistics: the env's episode logs (e.g. the success rate) only
  count episodes that started normally, and the pipeline's per-step flag rates (``metric_*``, ``derived_*``) are averaged
  over environments in a normal episode. The stage scores therefore keep meaning the same thing.

Settings come from the environment variable RD_HARD_STARTS (JSON)::

    {"prob": 0.25, "slots": 4, "max_tries": 3, "adapter": "<task adapter>", "success_flag": "succeeded"}
"""

from __future__ import annotations

import json
import os

import torch

ENV_VAR = "RD_HARD_STARTS"
# DirectRLEnv buffers that a reset sets itself (restoring them would only copy the same fresh values)
_SKIP = {"reset_buf", "reset_terminated", "reset_time_outs", "reward_buf", "episode_length_buf"}

_active = None  # the installed HardStarts instance (read by the reward statistics)


def fresh_mask():
    """Bool tensor [num_envs]: environments in an episode that started normally, or None when not installed."""
    return None if _active is None else ~_active.replay_active


def _torch(x):
    return x.torch if hasattr(x, "torch") else x


class HardStarts:
    def __init__(self, env, settings):
        from rd_lib.quantities import get_adapter

        self.env = env
        self.n = env.num_envs
        self.device = env.device
        self.prob = float(settings.get("prob", 0.25))
        self.slots = int(settings.get("slots", 4))
        self.max_tries = int(settings.get("max_tries", 3))
        self.success_fn = get_adapter(settings["adapter"])[settings.get("success_flag", "succeeded")].fn
        scene = env.scene
        self.articulations = list(scene.articulations.values())
        self.rigid_objects = list(scene.rigid_objects.values())
        self.tensor_names = None  # per-environment env attributes, found at the first reset (after the env's init)
        self.stored = None  # name -> tensor [slots, num_envs, ...] of failed start states
        self.current = None  # name -> tensor [num_envs, ...]: start state of the running episode
        self.valid = torch.zeros(self.slots, self.n, dtype=torch.bool, device=self.device)
        self.tries = torch.zeros(self.slots, self.n, dtype=torch.long, device=self.device)
        self.next_slot = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.replay_slot = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        self.replay_active = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.counts = {"episodes": 0, "replays": 0, "replay_successes": 0, "stored": 0}

    # -- state -------------------------------------------------------------------------------------------------
    def _find_tensors(self):
        names = []
        for name, value in vars(self.env).items():
            if name in _SKIP or not isinstance(value, torch.Tensor) or value.dim() == 0:
                continue
            if value.shape[0] == self.n and not value.requires_grad and value.numel() // self.n <= 4096:
                names.append(name)
        self.tensor_names = names

    def _read(self, ids):
        """Start state of these environments: name -> tensor [len(ids), ...]."""
        state = {f"env.{name}": getattr(self.env, name)[ids].clone() for name in self.tensor_names}
        for k, art in enumerate(self.articulations):
            d = art.data
            state[f"art{k}.pose"] = _torch(d.root_link_pose_w)[ids].clone()
            state[f"art{k}.vel"] = _torch(d.root_com_vel_w)[ids].clone()
            state[f"art{k}.qpos"] = _torch(d.joint_pos)[ids].clone()
            state[f"art{k}.qvel"] = _torch(d.joint_vel)[ids].clone()
        for k, obj in enumerate(self.rigid_objects):
            d = obj.data
            state[f"obj{k}.pose"] = _torch(d.root_link_pose_w)[ids].clone()
            state[f"obj{k}.vel"] = _torch(d.root_com_vel_w)[ids].clone()
        return state

    def _write(self, ids, state):
        for name in self.tensor_names:
            getattr(self.env, name)[ids] = state[f"env.{name}"]
        for k, art in enumerate(self.articulations):
            art.write_root_pose_to_sim_index(root_pose=state[f"art{k}.pose"], env_ids=ids)
            art.write_root_velocity_to_sim_index(root_velocity=state[f"art{k}.vel"], env_ids=ids)
            art.write_joint_state_to_sim_index(position=state[f"art{k}.qpos"], velocity=state[f"art{k}.qvel"],
                                               env_ids=ids)
            art.set_joint_position_target_index(target=state[f"art{k}.qpos"], env_ids=ids)
        for k, obj in enumerate(self.rigid_objects):
            obj.write_root_pose_to_sim_index(root_pose=state[f"obj{k}.pose"], env_ids=ids)
            try:
                obj.write_root_velocity_to_sim_index(root_velocity=state[f"obj{k}.vel"], env_ids=ids)
            except Exception:  # noqa: BLE001  (kinematic bodies take no velocity)
                pass

    # -- reset -------------------------------------------------------------------------------------------------
    def reset(self, original, env_ids):
        env = self.env
        if env_ids is None or env.common_step_counter == 0 or self.current is None:
            out = original(env_ids)
            if self.tensor_names is None:
                self._find_tensors()
            ids = torch.arange(self.n, device=self.device) if env_ids is None else torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long)
            if self.current is None:
                full = self._read(torch.arange(self.n, device=self.device))
                self.current = full
                self.stored = {k: torch.zeros((self.slots, *v.shape), dtype=v.dtype, device=self.device)
                               for k, v in full.items()}
            else:
                for k, v in self._read(ids).items():
                    self.current[k][ids] = v
            self.replay_active[ids] = False
            return out
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # 1. outcome of the episodes that end now (before the reset clears the task state)
        success = self.success_fn(env).bool()[ids]
        was_replay = self.replay_active[ids]
        slot = self.replay_slot[ids]
        # replayed starts: drop when solved or tried too often
        r_ids, r_slot, r_ok = ids[was_replay], slot[was_replay], success[was_replay]
        if len(r_ids):
            self.tries[r_slot, r_ids] += 1
            drop = r_ok | (self.tries[r_slot, r_ids] >= self.max_tries)
            self.valid[r_slot[drop], r_ids[drop]] = False
        # normal episodes that failed: store their start state
        f_ids = ids[~was_replay & ~success]
        if len(f_ids):
            s = self.next_slot[f_ids]
            for k in self.stored:
                self.stored[k][s, f_ids] = self.current[k][f_ids]
            self.valid[s, f_ids] = True
            self.tries[s, f_ids] = 0
            self.next_slot[f_ids] = (s + 1) % self.slots
        self.counts["episodes"] += len(ids)
        self.counts["replays"] += len(r_ids)
        self.counts["replay_successes"] += int(r_ok.sum()) if len(r_ids) else 0
        self.counts["stored"] += len(f_ids)

        # 2. reset: episodes that were replays first, so the env's episode logs end up describing normal episodes
        log = env.extras.get("log")
        saved = dict(log) if isinstance(log, dict) else None
        normal = ids[~was_replay]
        out = None
        if len(r_ids):
            out = original(r_ids)
        if len(normal):
            out = original(normal)
        elif saved is not None:  # only replays ended: keep the previous episode logs
            env.extras["log"].clear()
            env.extras["log"].update(saved)

        # 3. new start states: some environments restart from a stored failed start
        for k, v in self._read(ids).items():
            self.current[k][ids] = v
        has = self.valid[:, ids].any(dim=0)
        pick = has & (torch.rand(len(ids), device=self.device) < self.prob)
        p_ids = ids[pick]
        self.replay_active[ids] = False
        self.replay_slot[ids] = -1
        if len(p_ids):
            # a random valid slot per environment
            scores = torch.rand(self.slots, len(p_ids), device=self.device) * self.valid[:, p_ids]
            chosen = scores.argmax(dim=0)
            state = {k: v[chosen, p_ids] for k, v in self.stored.items()}
            self._write(p_ids, state)
            for k, v in state.items():
                self.current[k][p_ids] = v
            self.replay_active[p_ids] = True
            self.replay_slot[p_ids] = chosen
        return out

    def pop_stats(self):
        c = self.counts
        stats = {
            "HardStarts/replay_share": c["replays"] / max(c["episodes"], 1),
            "HardStarts/replay_success": c["replay_successes"] / max(c["replays"], 1),
            "HardStarts/stored_per_iteration": float(c["stored"]),
            "HardStarts/envs_with_hard_starts": float(self.valid.any(dim=0).float().mean()),
        }
        self.counts = {k: 0 for k in c}
        return stats


def install(env):
    """Install hard-case starts on a DirectRLEnv when RD_HARD_STARTS is set. Returns the instance or None."""
    global _active
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    settings = json.loads(raw)
    if float(settings.get("prob", 0.0)) <= 0.0:
        return None
    hard = HardStarts(env, settings)
    original = env._reset_idx

    def reset_idx(env_ids):
        return hard.reset(original, env_ids)

    env._reset_idx = reset_idx
    _active = hard
    print(f"Hard-case starts: {settings}", flush=True)
    return hard
