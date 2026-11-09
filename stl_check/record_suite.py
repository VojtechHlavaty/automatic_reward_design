"""Record one kind of episode of the Franka lift-cube task for the trajectory suite.

Every environment runs one episode (no second episode after its end: its steps are marked invalid). Each control
step, the raw state (predicates.read_state) and the action are saved. A contact sensor on the cube is added (the
task has none; it only reports forces and does not change the physics); the scoring script builds the environment
the same way, so that the recorded actions replay to the same states.

Kinds:
  success    scripted: above the cube, down, close, lift to the goal and hold there
  hover      scripted: the open gripper parks next to the cube (a reward that pays for closeness)
  push       scripted: closed fingers sweep the cube along the table (moved or tipped without a grasp)
  lift_drop  scripted: as success, but the gripper opens shortly after the lift (dropped before the hold ends)
  random     uniform random actions in [-1, 1]
  policy     a trained RSL-RL checkpoint (--checkpoint), deterministic

    python stl_check/record_suite.py --kind success --num_envs 16 --seed 1 --out suite/success.pt --headless

The scripted kinds run in IsaacLab's IK-Abs variant of the task (Isaac-Lift-Cube-Franka-IK-Abs-v0: the same scene,
cube, goal, rewards and success, but a stiff-gain arm whose action is the target end-effector pose). In the
joint-position task the arm's soft gains let it lag and sag under gravity, so a scripted position controller
misses the cube by several cm. Random and policy episodes use the joint-position task the pipeline trains on. Each
suite file records its task, and score_rewards.py replays it in that task.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--kind", required=True, choices=["success", "hover", "push", "lift_drop", "random", "policy"])
parser.add_argument("--task", default=None, help="default: the IK-Abs variant for the scripted kinds, else "
                    "Isaac-Lift-Cube-Franka-v0 (the task the pipeline trains on)")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--checkpoint", default=None, help="RSL-RL checkpoint for --kind policy")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl_check.env_setup import TASK, make_env  # noqa: E402
from stl_check.predicates import read_state  # noqa: E402

SCRIPTED_TASK = "Isaac-Lift-Cube-Franka-IK-Abs-v0"  # the scripted kinds: the same task, a stiff arm and pose actions
HAND = "panda_hand"


class ScriptedPose:
    """Phase targets of the end effector (env-local) -> absolute pose actions of the IK-Abs task, per environment.

    The IK-Abs task (the lift task with FRANKA_PANDA_HIGH_PD_CFG and a DifferentialInverseKinematicsAction) takes as
    action the pose of panda_hand plus 0.107 m along its z axis (about the fingertip center), in the robot's root
    frame, and the gripper command. The orientation stays that of the default pose (gripper pointing down). A
    setpoint moves towards each phase target at MAX_SPEED, and a phase ends when the end effector has arrived (within
    ARRIVED of the phase target) or after TIMEOUT.
    """

    MAX_SPEED = 0.4  # m/s
    ARRIVED = 0.02  # m (the IK arm settles up to about 1.4 cm short of targets far to the side; at 0.012 it waited out
    # the timeouts there and lifted too late to hold the cube at the goal)
    TIMEOUT = 1.5  # s per approach phase
    CLOSE_TIME = 0.4  # s with the fingers closing before the lift

    def __init__(self, env, kind):
        from isaaclab.utils.math import subtract_frame_transforms

        self.u = env.unwrapped
        self.kind = kind
        robot = self.u.scene["robot"]
        n, dev = self.u.num_envs, self.u.device
        hand = robot.data.body_pose_w.torch[:, robot.find_bodies(HAND)[0][0]]
        root = robot.data.root_pose_w.torch
        _, self.quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7])
        self.root = root.clone()
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase_t = torch.zeros(n, device=dev)  # time in the current phase [s]
        self.setpoint = None
        self.drop_after = 0.3 + 0.4 * torch.rand(n, device=dev)  # lift_drop: open this long after the lift starts
        self.sweep_to = None
        self.hold_offset = None  # success / lift_drop: fingertip center minus cube center when the lift starts

    def _arrived(self, ee, target):
        return (torch.linalg.norm(ee - target, dim=-1) < self.ARRIVED) | (self.phase_t > self.TIMEOUT)

    def plan(self, state):
        """Phase target (env-local) and gripper action (-1 close, +1 open) of every environment.

        success / lift_drop: 0 above the cube, 1 down to it, 2 close, 3 lift to the goal (lift_drop opens after
        drop_after in phase 3). hover: 0 above a point 7 cm beside the cube, 1 down to it, then stay. push: 0 above a
        point 8 cm behind the cube, 1 down to it, 2 sweep 33 cm along y through the cube; the fingers stay closed.
        """
        cube, goal, ee = state["cube_pos"], state["goal_pos"], state["ee_pos"]
        dev = cube.device
        up = torch.tensor([0.0, 0.0, 0.12], device=dev)
        grip = torch.ones(cube.shape[0], device=dev)
        p = self.phase
        none = torch.zeros_like(p, dtype=torch.bool)
        if self.kind in ("success", "lift_drop"):
            target = torch.where((p == 0).unsqueeze(-1), cube + up, cube)
            # lift: bring the cube (not the fingertip center) to the goal, with the grasp offset at the start of the lift
            if self.hold_offset is None:
                self.hold_offset = torch.zeros_like(cube)
            self.hold_offset = torch.where((p < 3).unsqueeze(-1), ee - cube, self.hold_offset)
            target = torch.where((p >= 3).unsqueeze(-1), goal + self.hold_offset, target)
            grip = torch.where(p >= 2, -torch.ones_like(grip), grip)
            if self.kind == "lift_drop":
                grip = torch.where((p >= 3) & (self.phase_t > self.drop_after), torch.ones_like(grip), grip)
            nxt = ((p == 0) & self._arrived(ee, cube + up)) | ((p == 1) & self._arrived(ee, cube))
            nxt |= (p == 2) & (self.phase_t > self.CLOSE_TIME)
        elif self.kind == "hover":
            spot = cube + torch.tensor([0.0, 0.07, 0.0], device=dev)
            target = torch.where((p == 0).unsqueeze(-1), spot + up, spot)
            nxt = (p == 0) & self._arrived(ee, spot + up)
        else:  # push
            grip[:] = -1.0
            spot = cube + torch.tensor([0.0, -0.08, 0.0], device=dev)
            if self.sweep_to is None:
                self.sweep_to = spot.clone()
            self.sweep_to = torch.where((p <= 1).unsqueeze(-1), spot + torch.tensor([0.0, 0.33, 0.0], device=dev),
                                        self.sweep_to)  # fixed once the sweep starts
            target = torch.where((p == 0).unsqueeze(-1), spot + up, spot)
            target = torch.where((p >= 2).unsqueeze(-1), self.sweep_to, target)
            nxt = ((p == 0) & self._arrived(ee, spot + up)) | ((p == 1) & self._arrived(ee, spot))
        nxt = nxt | none
        self.phase = self.phase + nxt.long()
        self.phase_t = torch.where(nxt, torch.zeros_like(self.phase_t), self.phase_t + self.u.step_dt)
        return target, grip

    def actions(self, t, state):
        from isaaclab.utils.math import subtract_frame_transforms

        target, grip = self.plan(state)
        if self.setpoint is None:
            self.setpoint = state["ee_pos"].clone()
        to_go = target - self.setpoint
        dist = torch.linalg.norm(to_go, dim=-1, keepdim=True).clamp_min(1e-9)
        self.setpoint = self.setpoint + to_go * (self.MAX_SPEED * self.u.step_dt / dist).clamp(max=1.0)
        world = self.setpoint + self.u.scene.env_origins
        pos_b, _ = subtract_frame_transforms(self.root[:, :3], self.root[:, 3:7], world)
        if os.environ.get("STL_DEBUG") and round(t / self.u.step_dt) % 20 == 0:
            print(f"t {t:.2f} phase {int(self.phase[0])} setpoint {[round(float(x), 3) for x in self.setpoint[0]]} "
                  f"ee now {[round(float(x), 3) for x in state['ee_pos'][0]]} cube "
                  f"{[round(float(x), 3) for x in state['cube_pos'][0]]} width "
                  f"{float(state['finger_pos'][0].sum()):.3f}", flush=True)
        return torch.cat((pos_b, self.quat_b, grip.unsqueeze(-1)), dim=-1)


def main():
    scripted = args_cli.kind not in ("random", "policy")
    task = args_cli.task or (SCRIPTED_TASK if scripted else TASK)
    env = make_env(task, args_cli.num_envs, args_cli.seed, args_cli.device)
    u = env.unwrapped
    n, dev, dt = u.num_envs, u.device, u.step_dt
    obs, _ = env.reset(seed=args_cli.seed)
    controller, actor = None, None
    if args_cli.kind == "policy":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        from importlib import metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        if os.environ.get("RD_EXPLORATION"):  # a pipeline policy: the same output squashing as in its training
            sys.path.insert(0, os.path.expanduser("~/automatic_reward_design/reward_design"))
            from rd_lib import exploration  # noqa: F401

        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=dev)
        runner.load(args_cli.checkpoint)
        actor = runner.alg.actor
        actor.eval()
        obs = wrapped.get_observations()
    elif args_cli.kind != "random":
        controller = ScriptedPose(env, args_cli.kind)
    generator = torch.Generator(device=dev).manual_seed(args_cli.seed)

    ended = torch.zeros(n, dtype=torch.bool, device=dev)
    states, actions, valid = [], [], []
    steps = int(u.max_episode_length)
    with torch.inference_mode():
        for k in range(steps):
            state = read_state(u)
            if args_cli.kind == "random":
                a = torch.rand(n, 8, device=dev, generator=generator) * 2 - 1
            elif actor is not None:
                a = actor(obs, stochastic_output=False)
            else:
                a = controller.actions(k * dt, state)
            states.append(state)
            actions.append(a.clone())
            valid.append(~ended)
            if actor is not None:
                obs, _, dones, _ = wrapped.step(a)
            else:
                obs, _, terminated, truncated, _ = env.step(a)
                dones = terminated | truncated
            ended |= dones.bool()  # this env's episode is over; later steps belong to its next episode
    traj = {key: torch.stack([s[key] for s in states], dim=1).cpu() for key in states[0]}  # [E, T, ...]
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    torch.save({"kind": args_cli.kind, "task": task, "seed": args_cli.seed, "num_envs": n, "dt": dt,
                "checkpoint": args_cli.checkpoint, "traj": traj,
                "actions": torch.stack(actions, dim=1).cpu(), "valid": torch.stack(valid, dim=1).cpu()},
               args_cli.out)
    print(f"@@STL_SUITE {args_cli.kind}: {n} episodes x {steps} steps -> {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
