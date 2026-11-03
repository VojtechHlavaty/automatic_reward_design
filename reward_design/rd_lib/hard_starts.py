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

Progress starts (``progress_prob`` > 0) use the same snapshots in the middle of episodes, so that the later steps of a
task are practised before the policy gets there often on its own:

* In every step, the task's progress flags (``progress_flags``, in task order, without the flags that end the episode)
  are evaluated. When a flag first becomes true in an episode, the state is snapshotted; if the episode goes on for
  ``progress_commit_steps`` more steps (it did not end right after, e.g. by a fall), the snapshot is kept as that
  environment's state for that flag (the newest one replaces the older one). A further flag reached before then
  replaces the waiting snapshot.
* At a reset that is not a hard-case replay, with probability ``progress_prob``, an environment with kept states
  restarts from the state of the furthest flag it has. The episode gets its full length again. These episodes count
  as replays (left out of the statistics) and can themselves reach and keep further flags.
* Event reward terms see the restored flags as already set on the first step (rd_lib/reward_registry.py), so a
  restored state pays no event, and potential terms start from the restored state.

The snapshot covers the per-environment tensors of the env and of the env's own helper objects one level down (e.g.
a walking controller's observation history), not those of IsaacLab or other libraries.

Settings come from the environment variable RD_HARD_STARTS (JSON)::

    {"prob": 0.25, "slots": 4, "max_tries": 3, "adapter": "<task adapter>", "success_flag": "succeeded",
     "progress_prob": 0.3, "progress_commit_steps": 25, "progress_flags": ["at_table", "grasped", ...]}
"""

from __future__ import annotations

import json
import os

import torch

ENV_VAR = "RD_HARD_STARTS"
# DirectRLEnv buffers that a reset sets itself (restoring them would only copy the same fresh values)
_SKIP = {"reset_buf", "reset_terminated", "reset_time_outs", "reward_buf", "episode_length_buf"}
# helper objects of these libraries are not searched for per-environment tensors (the scene's state is read directly)
_LIBRARIES = ("isaaclab", "isaacsim", "omni", "pxr", "torch", "gymnasium", "numpy", "warp", "carb", "builtins",
              "collections", "rsl_rl")

_active = None  # the installed HardStarts instance (read by the reward statistics)


def fresh_mask():
    """Bool tensor [num_envs]: environments in an episode that started normally, or None when not installed."""
    return None if _active is None else ~_active.replay_active


def _torch(x):
    return x.torch if hasattr(x, "torch") else x


class SceneState:
    """Reads and writes the full state of chosen environments: the root pose and velocity of every articulation and
    rigid object of the scene, the joint positions and velocities of every articulation, and every per-environment
    tensor attribute of the env and of its own helper objects one level down (found at the first reset, after the env's
    init). With ``local`` the poses are relative to each environment's origin, so a state can be written into another
    environment than the one it was read from (rd_lib/skill_starts.py)."""

    def __init__(self, env):
        self.env = env
        self.n = env.num_envs
        self.device = env.device
        scene = env.scene
        self.articulations = list(scene.articulations.values())
        self.rigid_objects = list(scene.rigid_objects.values())
        self.tensor_names = None

    def _per_env(self, value):
        return (isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == self.n
                and not value.requires_grad and value.numel() // self.n <= 4096)

    def _find_tensors(self):
        """Per-environment tensors of the env and of its own helper objects one level down: (attribute path)."""
        names = []
        for name, value in vars(self.env).items():
            if name in _SKIP:
                continue
            if self._per_env(value):
                names.append((name,))
            elif (hasattr(value, "__dict__") and not isinstance(value, type)
                  and type(value).__module__.split(".")[0] not in _LIBRARIES):
                for sub, sub_value in vars(value).items():
                    if self._per_env(sub_value):
                        names.append((name, sub))
        self.tensor_names = names

    def _tensor(self, path):
        value = self.env
        for part in path:
            value = getattr(value, part)
        return value

    def _read(self, ids, local=False):
        """State of these environments: name -> tensor [len(ids), ...]."""
        state = {"env." + ".".join(path): self._tensor(path)[ids].clone() for path in self.tensor_names}
        origin = self.env.scene.env_origins[ids] if local else None
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
        if local:
            for key in state:
                if key.endswith(".pose"):
                    state[key][:, :3] -= origin
        return state

    def _write(self, ids, state, local=False, skip=()):
        for path in self.tensor_names:
            key = "env." + ".".join(path)
            if key in state and key not in skip:
                self._tensor(path)[ids] = state[key]
        if local:
            origin = self.env.scene.env_origins[ids]
            state = {k: (torch.cat((v[:, :3] + origin, v[:, 3:]), dim=-1) if k.endswith(".pose") else v)
                     for k, v in state.items()}
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


class HardStarts(SceneState):
    def __init__(self, env, settings):
        from rd_lib.quantities import get_adapter

        super().__init__(env)
        self.prob = float(settings.get("prob", 0.25))
        self.slots = int(settings.get("slots", 4))
        self.max_tries = int(settings.get("max_tries", 3))
        quantities = get_adapter(settings["adapter"])
        self.success_fn = quantities[settings.get("success_flag", "succeeded")].fn
        self.progress_prob = float(settings.get("progress_prob", 0.0))
        self.commit_steps = int(settings.get("progress_commit_steps", 25))
        self.progress_names = [f for f in settings.get("progress_flags") or [] if f in quantities]
        if self.progress_prob <= 0.0:
            self.progress_names = []
        self.progress_fns = [quantities[f].fn for f in self.progress_names]
        nf = len(self.progress_names)
        self.seen = torch.zeros(self.n, max(nf, 1), dtype=torch.bool, device=self.device)
        self.bank = None  # name -> tensor [flags, num_envs, ...]: kept state per flag
        self.bank_valid = torch.zeros(max(nf, 1), self.n, dtype=torch.bool, device=self.device)
        self.pending = None  # name -> tensor [num_envs, ...]: snapshot waiting for progress_commit_steps
        self.pending_flag = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        self.pending_step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.progress_active = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.stored = None  # name -> tensor [slots, num_envs, ...] of failed start states
        self.current = None  # name -> tensor [num_envs, ...]: start state of the running episode
        self.valid = torch.zeros(self.slots, self.n, dtype=torch.bool, device=self.device)
        self.tries = torch.zeros(self.slots, self.n, dtype=torch.long, device=self.device)
        self.next_slot = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.replay_slot = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        self.replay_active = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.counts = {"episodes": 0, "replays": 0, "replay_successes": 0, "stored": 0, "progress_starts": 0,
                       "progress_successes": 0, "progress_ended": 0, "progress_kept": 0}

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
        hard = was_replay & (slot >= 0)
        was_progress = self.progress_active[ids]
        self.counts["progress_ended"] += int(was_progress.sum())
        self.counts["progress_successes"] += int((was_progress & success).sum())
        # replayed starts: drop when solved or tried too often
        r_ids = ids[was_replay]
        h_ids, h_slot, h_ok = ids[hard], slot[hard], success[hard]
        r_ok = success[was_replay]
        if len(h_ids):
            self.tries[h_slot, h_ids] += 1
            drop = h_ok | (self.tries[h_slot, h_ids] >= self.max_tries)
            self.valid[h_slot[drop], h_ids[drop]] = False
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
        self.counts["replays"] += len(h_ids)
        self.counts["replay_successes"] += int(h_ok.sum()) if len(h_ids) else 0
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
        self.progress_active[ids] = False
        self.seen[ids] = False
        self.pending_flag[ids] = -1
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
        # progress starts: the other environments may restart from the kept state of their furthest flag
        if self.bank is not None:
            rest = ids[~pick]
            kept = self.bank_valid[:, rest]
            go = kept.any(dim=0) & (torch.rand(len(rest), device=self.device) < self.progress_prob)
            g_ids = rest[go]
            if len(g_ids):
                order = torch.arange(1, kept.shape[0] + 1, device=self.device).unsqueeze(-1)
                flag = (kept[:, go] * order).argmax(dim=0)  # the furthest kept flag
                state = {k: v[flag, g_ids] for k, v in self.bank.items()}
                self.seen[g_ids] = state.pop("seen")
                self._write(g_ids, state)
                for k, v in state.items():
                    self.current[k][g_ids] = v
                self.replay_active[g_ids] = True
                self.progress_active[g_ids] = True
                self.counts["progress_starts"] += len(g_ids)
        return out

    # -- progress snapshots (every step, after the env computed its terminations) --------------------------------
    def step(self, dones):
        """Track the first time each progress flag becomes true in an episode, snapshot it, keep it if the episode
        goes on for progress_commit_steps more steps."""
        if not self.progress_fns or self.tensor_names is None:
            return
        env = self.env
        steps = env.episode_length_buf
        alive = ~dones.bool()
        # keep the waiting snapshots whose episode went on long enough
        ready = (self.pending_flag >= 0) & (steps - self.pending_step >= self.commit_steps) & alive
        c_ids = ready.nonzero().squeeze(-1)
        if len(c_ids):
            if self.bank is None:
                self.bank = {k: torch.zeros((len(self.progress_fns), *v.shape), dtype=v.dtype, device=self.device)
                             for k, v in self.pending.items()}
            flag = self.pending_flag[c_ids]
            for k, v in self.pending.items():
                self.bank[k][flag, c_ids] = v[c_ids]
            self.bank_valid[flag, c_ids] = True
            self.pending_flag[c_ids] = -1
            self.counts["progress_kept"] += len(c_ids)
        self.pending_flag[~alive] = -1
        # flags reached for the first time in this episode: snapshot the furthest one
        flags = torch.stack([fn(env).bool() for fn in self.progress_fns], dim=-1)
        first = flags & ~self.seen
        self.seen |= flags
        order = torch.arange(1, flags.shape[1] + 1, device=self.device)
        furthest = (first * order).max(dim=-1).values - 1  # -1: none
        new = alive & (furthest >= 0) & (furthest > self.pending_flag)
        n_ids = new.nonzero().squeeze(-1)
        if len(n_ids):
            state = self._read(n_ids)
            state["seen"] = self.seen[n_ids].clone()
            if self.pending is None:
                self.pending = {k: torch.zeros((self.n, *v.shape[1:]), dtype=v.dtype, device=self.device)
                                for k, v in state.items()}
            for k, v in state.items():
                self.pending[k][n_ids] = v
            self.pending_flag[n_ids] = furthest[n_ids]
            self.pending_step[n_ids] = steps[n_ids]

    def pop_stats(self):
        c = self.counts
        stats = {
            "HardStarts/replay_share": c["replays"] / max(c["episodes"], 1),
            "HardStarts/replay_success": c["replay_successes"] / max(c["replays"], 1),
            "HardStarts/stored_per_iteration": float(c["stored"]),
            "HardStarts/envs_with_hard_starts": float(self.valid.any(dim=0).float().mean()),
        }
        if self.progress_fns:
            stats["ProgressStarts/share"] = c["progress_starts"] / max(c["episodes"], 1)
            stats["ProgressStarts/success"] = c["progress_successes"] / max(c["progress_ended"], 1)
            stats["ProgressStarts/kept_per_iteration"] = float(c["progress_kept"])
            for k, name in enumerate(self.progress_names):
                stats[f"ProgressStarts/envs_with_{name}"] = float(self.bank_valid[k].float().mean())
        self.counts = {k: 0 for k in c}
        return stats


def install(env):
    """Install hard-case starts on a DirectRLEnv when RD_HARD_STARTS is set. Returns the instance or None."""
    global _active
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    settings = json.loads(raw)
    if float(settings.get("prob", 0.0)) <= 0.0 and float(settings.get("progress_prob", 0.0)) <= 0.0:
        return None
    from isaaclab.envs import DirectRLEnv

    if not isinstance(env, DirectRLEnv):  # the snapshots and the done hook need a DirectRLEnv's buffers
        print(f"Hard-case starts: not available for {type(env).__name__} (only DirectRLEnv tasks); off", flush=True)
        return None
    hard = HardStarts(env, settings)
    original = env._reset_idx

    def reset_idx(env_ids):
        return hard.reset(original, env_ids)

    env._reset_idx = reset_idx
    if hard.progress_fns:
        original_dones = env._get_dones

        def get_dones():
            terminated, truncated = original_dones()
            hard.step(terminated | truncated)
            return terminated, truncated

        env._get_dones = get_dones
    _active = hard
    print(f"Hard-case starts: {settings}", flush=True)
    return hard
