"""Record frames of failed episodes of a checkpoint, focused on the moment the episode stopped making progress.

The policy runs (deterministic) in ``--num_envs`` environments for one episode each. The state of every articulation
and rigid object is recorded at every step together with the task's success conditions (rd_lib.quantities funnel).
An environment whose episode has ended is not reset again, so its scene (e.g. randomized colors) stays the one of its
recorded episode. Then a few failed episodes are chosen (the most common ways of failing first, by the last success
condition they reached) and replayed kinematically for the camera:

* context frames: the start, the middle of the episode and its end;
* dense frames around the critical moment: when the last reached condition was lost again (e.g. the object left the
  target after a release), or else when it was reached (the policy made no further progress after it).

With a ``visual_crop`` (fractions x0, y0, x1, y1 of the frame) the dense frames are close-ups of the work area. A
timeline of every success condition (when it became true, when it was lost) is written with the frames. Optionally,
a short clip of each critical window is saved for the human reviewer. Writes ``failures.json`` into ``--out_dir``.

    python -m isaaclab_backend.record_failures --task <gym id> --checkpoint model.pt --adapter <name> \
        --out_dir <dir> --headless --enable_cameras [hydra overrides, e.g. env.viewer.eye=[1,1,1]]
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
parser.add_argument("--adapter", required=True)
parser.add_argument("--success_flag", default="succeeded")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_failures", type=int, default=3)
parser.add_argument("--dense_frames", type=int, default=6)
parser.add_argument("--window_before", type=float, default=0.3, help="seconds before the critical moment")
parser.add_argument("--window_after", type=float, default=1.2, help="seconds after the critical moment")
parser.add_argument("--crop", default=None, help="JSON [x0, y0, x1, y1]: close-up of the work area (fractions)")
parser.add_argument("--clip", action="store_true", help="also save an MP4 clip of each critical window")
add_launcher_args(parser)
args_cli, remaining = setup_preset_cli(parser)
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + remaining

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_backend.tasks_import import register_tasks  # noqa: E402

register_tasks()  # the tasks of this repository (tasks/)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
if os.environ.get("RD_EXPLORATION"):
    from rd_lib import exploration  # noqa: F401,E402  (the same output squashing as in training)
if os.environ.get("RD_OBS_FLAGS"):
    from rd_lib import obs_extension  # noqa: F401,E402  (progress flags in the observations)
from rd_lib.quantities import get_adapter, get_funnel  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def choose_failures(furthest, failed, k):
    """Up to ``k`` failed environments, one per way of failing (last reached condition), most common ways first."""
    groups = {}
    for e in failed:
        groups.setdefault(furthest[e], []).append(e)
    order = sorted(groups.values(), key=len, reverse=True)
    chosen = []
    while len(chosen) < k and any(order):
        for group in order:
            if group and len(chosen) < k:
                chosen.append(group.pop(0))
    return chosen


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    import imageio

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array"),
                                 clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        actor = runner.alg.actor
        actor.eval()
        u = env.unwrapped
        n, dev = u.num_envs, u.device
        funnel = get_funnel(args_cli.adapter)
        success_fn = get_adapter(args_cli.adapter)[args_cli.success_flag].fn
        arts = list(u.scene.articulations.values())
        objs = list(u.scene.rigid_objects.values())
        os.makedirs(args_cli.out_dir, exist_ok=True)

        # ---- 1. one episode per environment, states and success conditions at every step
        done_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        success = torch.zeros(n, dtype=torch.bool, device=dev)
        original_reset = u._reset_idx

        def reset_idx(env_ids):
            if env_ids is None or u.common_step_counter == 0:
                return original_reset(env_ids)
            ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
            new = ids[done_step[ids] < 0]
            if len(new):
                success[new] = success_fn(u).bool()[new]
                done_step[new] = len(states) - 1  # the last recorded state of the episode (before its final step)
            return None  # no reset: the scene of the recorded episode stays

        u._reset_idx = reset_idx

        def snapshot():
            s = {}
            for k, a in enumerate(arts):
                s[f"a{k}p"] = _t(a.data.root_link_pose_w).clone()
                s[f"a{k}q"] = _t(a.data.joint_pos).clone()
            for k, o in enumerate(objs):
                s[f"o{k}p"] = _t(o.data.root_link_pose_w).clone()
            return s

        states, flags = [], []
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(int(u.max_episode_length)):
                states.append(snapshot())
                flags.append(torch.stack([fn(u).bool() for _, fn in funnel], dim=-1))
                obs, _, _, _ = env.step(actor(obs, stochastic_output=False))
                if bool((done_step >= 0).all()):
                    break
        steps = len(states)
        done_step[done_step < 0] = steps - 1
        done_step.clamp_(max=steps - 1)
        F = torch.stack(flags)  # [steps, envs, conditions]
        dt = u.step_dt

        # ---- 2. failed episodes and their critical moments
        ever = F.any(dim=0)  # [envs, conditions]
        reached = [[k for k in range(len(funnel)) if bool(ever[e, k])] for e in range(n)]
        furthest = [r[-1] if r else -1 for r in reached]
        failed = [e for e in range(n) if not bool(success[e])]
        chosen = choose_failures(furthest, list(failed), args_cli.max_failures)
        note = ""
        if not chosen:
            chosen = [int(torch.argmax(done_step.float()))] if n else []
            note = "No failed episode in this sample; the frames show a successful episode."

        def timeline(e):
            end = int(done_step[e])
            lines = []
            for k, (name, _) in enumerate(funnel):
                series = F[: end + 1, e, k].tolist()
                if not any(series):
                    lines.append(f"- {name}: never")
                    continue
                events, prev = [], False
                for i, v in enumerate(series):
                    if v != prev:
                        events.append(("true" if v else "false") + f" at {i * dt:.2f} s")
                        prev = v
                more = len(events) > 6
                lines.append(f"- {name}: " + ", ".join(events[:6]) + (", ..." if more else ""))
            return "\n".join(lines)

        def critical(e):
            end = int(done_step[e])
            k = furthest[e]
            if k < 0:
                return end // 2, "no success condition was reached; frames are spread over the episode"
            series = F[: end + 1, e, k].tolist()
            first = series.index(True)
            lost = next((i for i in range(first, len(series)) if not series[i]), None)
            name = funnel[k][0]
            if lost is not None:
                return lost, f"'{name}' (the last condition reached) became true at {first * dt:.2f} s and was lost " \
                             f"again at {lost * dt:.2f} s"
            return first, f"'{name}' (the last condition reached) became true at {first * dt:.2f} s; the episode " \
                          f"made no further progress until it ended at {end * dt:.2f} s"

        # ---- 3. kinematic replay for the camera
        from isaacsim.core.rendering_manager import ViewportManager

        origins = u.scene.env_origins
        eye0 = list(u.cfg.viewer.eye)
        look0 = list(u.cfg.viewer.lookat)
        env0 = int(getattr(u.cfg.viewer, "env_index", 0))
        local = getattr(u.cfg.viewer, "origin_type", "env") == "env"
        crop = json.loads(args_cli.crop) if args_cli.crop else None
        u.render()  # creates the recording camera

        def place_camera(e):
            # viewer eye/lookat are relative to the viewed env's origin (origin_type "env") or world positions near
            # env_index's origin: either way, shift them to environment e
            shift = (origins[e] if local else origins[e] - origins[env0]).tolist()
            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[a + b for a, b in zip(eye0, shift)],
                                            target=[a + b for a, b in zip(look0, shift)])

        def show(e, i, settle=3):
            ids = torch.tensor([e], device=dev)
            s = states[i]
            for k, a in enumerate(arts):
                a.write_root_pose_to_sim_index(root_pose=s[f"a{k}p"][ids], env_ids=ids)
                a.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
                q = s[f"a{k}q"][ids]
                a.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=ids)
                a.set_joint_position_target_index(target=q, env_ids=ids)
            for k, o in enumerate(objs):
                o.write_root_pose_to_sim_index(root_pose=s[f"o{k}p"][ids], env_ids=ids)
                try:
                    o.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
                except Exception:  # noqa: BLE001  (kinematic bodies)
                    pass
            u.scene.write_data_to_sim()
            u.sim.forward()
            frame = None
            for _ in range(settle):
                frame = u.render()
            return frame

        def close_up(frame):
            if not crop:
                return frame
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = (round(f * m) for f, m in zip(crop, (w, h, w, h)))
            return frame[y0:y1, x0:x1]

        result = {"episodes": n, "success_rate": float(success.float().mean()), "note": note,
                  "failure_counts": {}, "failures": []}
        for e in failed:
            key = funnel[furthest[e]][0] if furthest[e] >= 0 else "(none)"
            result["failure_counts"][key] = result["failure_counts"].get(key, 0) + 1
        for r, e in enumerate(chosen):
            place_camera(e)
            end = int(done_step[e])
            center, why = critical(e)
            lo = max(0, center - round(args_cli.window_before / dt))
            hi = min(end, center + round(args_cli.window_after / dt))
            dense = sorted({round(lo + j * (hi - lo) / max(args_cli.dense_frames - 1, 1))
                            for j in range(args_cli.dense_frames)})
            frames = []
            for i, kind in [(0, "start"), (min(lo, end) // 2, "context"), *[(i, "critical") for i in dense],
                            (end, "end")]:
                image = show(e, i)
                name = f"failure{r}_{i:04d}_{kind}.jpg"
                imageio.imwrite(os.path.join(args_cli.out_dir, name),
                                close_up(image) if kind == "critical" else image, quality=88)
                frame = {"t": round(i * dt, 2), "kind": kind, "file": name}
                if kind != "critical" and crop:
                    close_name = name.replace(".jpg", "_close.jpg")
                    imageio.imwrite(os.path.join(args_cli.out_dir, close_name), close_up(image), quality=90)
                    frame["close_file"] = close_name
                frames.append(frame)
            clip = None
            if args_cli.clip:
                clip = f"failure{r}.mp4"
                writer = imageio.get_writer(os.path.join(args_cli.out_dir, clip), fps=max(int(round(0.5 / dt)), 1),
                                            codec="libx264", quality=7)
                for i in range(max(lo - round(1.0 / dt), 0), min(hi + round(0.5 / dt), end) + 1, 2):
                    writer.append_data(show(e, i, settle=2))
                writer.close()
            result["failures"].append({
                "env": e, "success": bool(success[e]), "end_s": round(end * dt, 2),
                "last_condition": funnel[furthest[e]][0] if furthest[e] >= 0 else None,
                "critical_moment": why, "timeline": timeline(e), "frames": frames, "clip": clip})
        with open(os.path.join(args_cli.out_dir, "failures.json"), "w") as f:
            json.dump(result, f, indent=1)
        print("@@RD_FAILURES@@ " + json.dumps({k: result[k] for k in ("episodes", "success_rate", "failure_counts")}),
              flush=True)
        env.close()


if __name__ == "__main__":
    main()
