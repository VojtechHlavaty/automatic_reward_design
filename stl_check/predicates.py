"""Predicate signals of the Franka lift-cube task (``Isaac-Lift-Cube-Franka-v0``), batched in PyTorch.

Two layers:

* ``read_state(env)`` reads the raw state of every environment in one step (positions env-local): the recorder
  calls it every control step and stacks the results into a trajectory of shape [episodes, steps, ...];
* ``signals(traj)`` turns a recorded trajectory into named float signals of shape [episodes, steps], the inputs of
  the STL predicates (``robustness.Pred``). Each is a pure function of the recorded state, so a spec can be
  evaluated offline, long after the simulator has closed.

Units: metres, m/s, newtons, the raw gripper action. ``grasp`` is a combined signal (see ``grasp_margin``),
normalized so that 0 is the edge of a grasp and about +-1 is far from it.
"""

from __future__ import annotations

import torch

REACH = 0.045  # fingertip center this close to the cube center counts as at the cube [m]. Was 0.02: a firm grasp
# near the cube's edge (0.048 m cube) holds the fingertip center 3-4 cm off its center (suite16, 5 of 16 scripted lifts)
SHUT = 0.01  # fingers closer than this have closed on nothing [m]
FINGER_OPEN = 0.04  # each finger joint: 0.04 m open, 0 closed


# ---------------------------------------------------------------------------------------------------------------
# raw state, read from a live IsaacLab environment
# ---------------------------------------------------------------------------------------------------------------


def read_state(env) -> dict:
    """The raw state of every environment now ({name: [num_envs, ...] tensor}, positions env-local)."""
    from isaaclab.utils.math import combine_frame_transforms

    robot, cube = env.scene["robot"], env.scene["object"]
    origins = env.scene.env_origins
    finger_ids = getattr(env, "_stl_finger_ids", None)
    if finger_ids is None:
        finger_ids = robot.find_joints("panda_finger.*")[0]
        env._stl_finger_ids = finger_ids
    command = env.command_manager.get_command("object_pose")
    goal_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    state = {
        "ee_pos": env.scene["ee_frame"].data.target_pos_w.torch[..., 0, :] - origins,
        "cube_pos": cube.data.root_pos_w.torch - origins,
        "cube_quat": cube.data.root_quat_w.torch.clone(),  # (x, y, z, w)
        "cube_lin_vel": cube.data.root_lin_vel_w.torch.clone(),
        "goal_pos": goal_w - origins,
        "finger_pos": robot.data.joint_pos.torch[:, finger_ids].clone(),
        "gripper_action": env.action_manager.action[:, 7].clone(),
        "arm_joint_pos": robot.data.joint_pos.torch[:, :7].clone(),
    }
    sensors = getattr(env.scene, "sensors", {})
    if "cube_contact" in sensors:  # added by record_suite.py (the task itself has no contact sensor)
        state["cube_force"] = torch.linalg.norm(sensors["cube_contact"].data.net_forces_w.torch, dim=-1).amax(dim=-1)
    return {k: v.detach().float() for k, v in state.items()}


# ---------------------------------------------------------------------------------------------------------------
# predicates: pure functions of a recorded trajectory ({name: [E, T, ...]}) -> [E, T] float
# ---------------------------------------------------------------------------------------------------------------


def dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between two point trajectories [E, T, 3] -> [E, T] [m]."""
    return torch.linalg.norm(a - b, dim=-1)


def height(pos: torch.Tensor) -> torch.Tensor:
    """Height above the table top (z = 0 at the robot base) [m]."""
    return pos[..., 2]


def gripper_width(finger_pos: torch.Tensor) -> torch.Tensor:
    """Opening between the fingers [m]: 0.08 fully open, 0 fully closed (it stops at the cube's width on a grasp)."""
    return finger_pos.sum(dim=-1)


def gripper_open(finger_pos: torch.Tensor) -> torch.Tensor:
    """How open the gripper is, as a share of fully open (0 closed .. 1 open)."""
    return gripper_width(finger_pos) / (2 * FINGER_OPEN)


def speed(vel: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(vel, dim=-1)


def tilt(quat: torch.Tensor) -> torch.Tensor:
    """Angle between the cube's up axis and the world's [rad] (0 upright; a tipped cube is near pi/2)."""
    x, y = quat[..., 0], quat[..., 1]
    up_z = 1.0 - 2.0 * (x * x + y * y)  # z component of the cube's rotated z axis
    return torch.arccos(up_z.clamp(-1.0, 1.0))


def grasp_margin(ee_pos, cube_pos, finger_pos, gripper_action) -> torch.Tensor:
    """Signed grasp margin (> 0: grasped): the minimum of three normalized margins, all of which must hold:

    * the fingertip center is within REACH of the cube center: (REACH - d) / REACH;
    * the gripper is commanded closed (the binary gripper action is negative): -action, clipped to [-1, 1];
    * the fingers have stopped on something, not shut: (width - SHUT) / SHUT, clipped to [-1, 1].

    There is no contact sensor on the fingers, so a cube that is only touched while the fingers close counts too.
    """
    near = (REACH - dist(ee_pos, cube_pos)) / REACH
    closing = (-gripper_action).clamp(-1.0, 1.0)
    not_shut = ((gripper_width(finger_pos) - SHUT) / SHUT).clamp(-1.0, 1.0)
    return torch.minimum(torch.minimum(near.clamp(-1.0, 1.0), closing), not_shut)


def contact_force(traj: dict) -> torch.Tensor:
    """Largest net contact force on the cube [N] (table and fingers together; 0 if the recording had no sensor)."""
    force = traj.get("cube_force")
    if force is None:
        return torch.zeros(traj["cube_pos"].shape[:2], device=traj["cube_pos"].device)
    return force


def signals(traj: dict) -> dict:
    """All predicate signals of a trajectory: {name: [E, T] float}."""
    return {
        "ee_cube_dist": dist(traj["ee_pos"], traj["cube_pos"]),
        "cube_goal_dist": dist(traj["cube_pos"], traj["goal_pos"]),
        "cube_height": height(traj["cube_pos"]),
        "gripper_open": gripper_open(traj["finger_pos"]),
        "grasp": grasp_margin(traj["ee_pos"], traj["cube_pos"], traj["finger_pos"], traj["gripper_action"]),
        "cube_speed": speed(traj["cube_lin_vel"]),
        "cube_tilt": tilt(traj["cube_quat"]),
        "contact_force": contact_force(traj),
    }
