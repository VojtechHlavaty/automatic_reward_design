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

The scripted kinds use differential IK (DLS) on the end effector and convert the joint targets to the task's
joint-position actions: action = (q_target - q_default) / 0.5.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--kind", required=True, choices=["success", "hover", "push", "lift_drop", "random", "policy"])
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")  # stl_check.env_setup.TASK
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--checkpoint", default=None, help="RSL-RL checkpoint for --kind policy")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl_check.env_setup import make_env  # noqa: E402
from stl_check.predicates import read_state  # noqa: E402

ACTION_SCALE = 0.5  # the task's JointPositionAction scale (target = default + 0.5 * action)
HAND = "panda_hand"
HAND_TO_EE = 0.1034  # the end effector (fingertip center) is this far past panda_hand along its z axis [m]


class ScriptedIK:
    """Waypoints of the end effector (env-local) -> joint-position actions, per environment."""

    def __init__(self, env, kind):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.u = env.unwrapped
        self.kind = kind
        robot = self.u.scene["robot"]
        self.robot = robot
        self.arm_ids = robot.find_joints("panda_joint.*")[0]
        self.hand_id = robot.find_bodies(HAND)[0][0]
        self.jac_id = self.hand_id - 1 if robot.is_fixed_base else self.hand_id
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.u.num_envs, device=self.u.device)
        n = self.u.num_envs
        # gripper pointing down (as in the default pose): keep the hand's start orientation
        self.down = robot.data.body_quat_w.torch[:, self.hand_id].clone()
        self.default = robot.data.default_joint_pos.torch[:, self.arm_ids].clone()
        self.drop_after = 0.3 + 0.4 * torch.rand(n, device=self.u.device)  # lift_drop: open this long after the lift

    def plan(self, t, state):
        """Target end-effector position (env-local) and gripper action (-1 close, +1 open) for time t [s]."""
        cube, goal = state["cube_pos"], state["goal_pos"]
        n = cube.shape[0]
        target = cube.clone()
        grip = torch.ones(n, device=cube.device)
        above = cube + torch.tensor([0.0, 0.0, 0.12], device=cube.device)
        if self.kind in ("success", "lift_drop"):
            target = above if t < 0.8 else cube.clone()  # above it, then down to its center
            if t >= 1.1:
                grip[:] = -1.0
            if t >= 1.4:
                target = goal.clone()
            if self.kind == "lift_drop" and t >= 1.4:
                lifted_for = t - 1.4
                grip = torch.where(lifted_for > self.drop_after, torch.ones_like(grip), grip)
        elif self.kind == "hover":
            side = torch.tensor([0.0, 0.07, 0.0], device=cube.device)  # beside the cube, open gripper
            target = above + side if t < 0.8 else cube + side
        elif self.kind == "push":
            grip[:] = -1.0  # closed fingers as a paddle
            behind = cube + torch.tensor([0.0, -0.08, 0.0], device=cube.device)
            through = cube + torch.tensor([0.0, 0.25, 0.0], device=cube.device)
            if t < 0.8:
                target = behind + torch.tensor([0.0, 0.0, 0.12], device=cube.device)
            elif t < 1.4:
                target = behind
            else:
                target = through
        return target, grip

    def actions(self, t, state):
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms

        robot = self.robot
        ee_target, grip = self.plan(t, state)
        # hand target: the fingertip center sits HAND_TO_EE along the hand's z axis
        offset = quat_apply(self.down, torch.tensor([0.0, 0.0, HAND_TO_EE], device=self.u.device).expand(
            self.u.num_envs, 3))
        hand_target_w = ee_target + self.u.scene.env_origins - offset
        root = robot.data.root_pose_w.torch
        pos_b, quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand_target_w, self.down)
        self.ik.set_command(torch.cat((pos_b, quat_b), dim=-1))
        hand_w = robot.data.body_pose_w.torch[:, self.hand_id]
        hand_pos_b, hand_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand_w[:, :3], hand_w[:, 3:7])
        jac_cols = [j + robot.num_base_dofs for j in self.arm_ids]
        jacobian = robot.data.body_link_jacobian_w.torch[:, self.jac_id, :, jac_cols]
        q = robot.data.joint_pos.torch[:, self.arm_ids]
        q_target = self.ik.compute(hand_pos_b, hand_quat_b, jacobian, q)
        arm = (q_target - self.default) / ACTION_SCALE
        return torch.cat((arm, grip.unsqueeze(-1)), dim=-1)


def main():
    env = make_env(args_cli.task, args_cli.num_envs, args_cli.seed, args_cli.device)
    u = env.unwrapped
    n, dev, dt = u.num_envs, u.device, u.step_dt
    obs, _ = env.reset(seed=args_cli.seed)
    controller, actor = None, None
    if args_cli.kind == "policy":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=dev)
        runner.load(args_cli.checkpoint)
        actor = runner.alg.actor
        actor.eval()
        obs = wrapped.get_observations()
    elif args_cli.kind != "random":
        controller = ScriptedIK(env, args_cli.kind)
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
    torch.save({"kind": args_cli.kind, "task": args_cli.task, "seed": args_cli.seed, "num_envs": n, "dt": dt,
                "checkpoint": args_cli.checkpoint, "traj": traj,
                "actions": torch.stack(actions, dim=1).cpu(), "valid": torch.stack(valid, dim=1).cpu()},
               args_cli.out)
    print(f"@@STL_SUITE {args_cli.kind}: {n} episodes x {steps} steps -> {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
