"""Task quantities of IsaacLab's Franka lift-cube task (see ``reward_design/rd_lib/quantities.py``).

``Isaac-Lift-Cube-Franka-v0`` is a manager-based task: the robot, the cube and the end-effector frame are scene
entities (``env.scene["robot"]``, ``env.scene["object"]``, ``env.scene["ee_frame"]``), and the goal is the
``object_pose`` command (a point in the robot's root frame, drawn once per 5 s episode). The task's own success is
the sticky flag of its ``object_goal_tracking`` reward term: at some step of the episode the cube was lifted (center
above z = 0.04 m) and within 0.05 m of the goal.

The pipeline finds this file through ``ADAPTERS`` and ``FUNNELS``; it must not import the task's environment module.
"""

import torch

from isaaclab.utils.math import combine_frame_transforms

from rd_lib.quantities import Quantity, mark_latched

LIFT_HEIGHT = 0.04  # cube center height above which the task counts the cube as lifted [m] (object_is_lifted)
GOAL_RADIUS = 0.05  # the task's success threshold: cube within this distance of the goal while lifted [m]
REACH_RADIUS = 0.03  # end effector (fingertip center) this close to the cube center counts as at the cube [m]
FINGER_OPEN = 0.04  # each finger joint: 0.04 m open, 0 closed


def _robot(env):
    return env.scene["robot"]


def _cube_pos(env):
    """Cube center, world frame [num_envs, 3]."""
    return env.scene["object"].data.root_pos_w.torch


def _ee_pos(env):
    """End effector (midpoint between the fingertips, 0.1034 m past panda_hand), world frame [num_envs, 3]."""
    return env.scene["ee_frame"].data.target_pos_w.torch[..., 0, :]


def _goal_pos(env):
    """The commanded goal point of the cube, world frame [num_envs, 3]."""
    robot = _robot(env)
    command = env.command_manager.get_command("object_pose")
    goal, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    return goal


def _finger_ids(env):
    ids = getattr(env, "_rd_finger_ids", None)
    if ids is None:
        ids = _robot(env).find_joints("panda_finger.*")[0]
        env._rd_finger_ids = ids
    return ids


def _gripper_width(env):
    """Opening between the fingers [m]: the sum of the two finger joints (0.08 open, 0 closed)."""
    return _robot(env).data.joint_pos.torch[:, _finger_ids(env)].sum(dim=-1)


def _gripper_cmd_closed(env):
    """The gripper action asks to close (IsaacLab's binary gripper action: a negative action closes)."""
    return env.action_manager.action[:, 7] < 0


def _cube_height(env):
    return _cube_pos(env)[:, 2] - env.scene.env_origins[:, 2]


def _ee_to_cube(env):
    return torch.norm(_cube_pos(env) - _ee_pos(env), dim=-1)


def _cube_to_goal(env):
    return torch.norm(_goal_pos(env) - _cube_pos(env), dim=-1)


def _grasped(env):
    # no contact sensors: the fingers are closing, the fingertip center is at the cube, and the fingers have not
    # closed all the way (something is between them)
    return _gripper_cmd_closed(env) & (_ee_to_cube(env) < REACH_RADIUS) & (_gripper_width(env) > 0.01)


def _lifted(env):
    return _cube_height(env) > LIFT_HEIGHT


def _succeeded(env):
    """The task's own sticky success flag (object_goal_tracking term), reset at the start of every episode."""
    term = env.reward_manager.get_term_cfg("object_goal_tracking").func
    return term._succeeded.clone()


def _franka_lift():
    q = {
        "ee_to_cube_dist": Quantity(
            _ee_to_cube, "scalar",
            "distance from the end effector (the midpoint between the two fingertips) to the cube center [m]"),
        "cube_height": Quantity(
            _cube_height, "scalar",
            f"height of the cube center above the table top (z = 0 at the robot base) [m]; the task counts the cube "
            f"as lifted above {LIFT_HEIGHT}"),
        "cube_to_goal_dist": Quantity(
            _cube_to_goal, "scalar",
            f"distance from the cube center to the commanded goal point [m] (success: below {GOAL_RADIUS} while "
            f"lifted)"),
        "ee_to_goal_dist": Quantity(
            lambda env: torch.norm(_goal_pos(env) - _ee_pos(env), dim=-1), "scalar",
            "distance from the end effector to the commanded goal point [m]"),
        "cube_speed": Quantity(
            lambda env: torch.norm(env.scene["object"].data.root_lin_vel_w.torch, dim=-1), "scalar",
            "linear speed of the cube [m/s]"),
        "gripper_width": Quantity(
            _gripper_width, "scalar",
            "opening between the fingers [m] (0.08 fully open, 0 fully closed; it stops at the cube's width when the "
            "fingers close on it)"),
        "action_rate": Quantity(
            lambda env: torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1),
            "scalar", "sum of squared changes of the 8 actions between control steps"),
        "arm_joint_vel_sq": Quantity(
            lambda env: torch.sum(torch.square(_robot(env).data.joint_vel.torch[:, :7]), dim=-1), "scalar",
            "sum of squared arm joint velocities (the 7 arm joints) [rad^2/s^2]"),
        "gripper_closing": Quantity(
            _gripper_cmd_closed, "flag", "the gripper action asks to close (a negative gripper action)"),
        "at_cube": Quantity(
            lambda env: _ee_to_cube(env) < REACH_RADIUS, "flag",
            f"the end effector is within {REACH_RADIUS} m of the cube center"),
        "grasped": Quantity(
            _grasped, "flag",
            f"the gripper is closing, the end effector is within {REACH_RADIUS} m of the cube center and the fingers "
            f"have not closed all the way (there are no contact sensors, so a cube that is only touched while the "
            f"fingers close also counts)"),
        "lifted": Quantity(
            _lifted, "flag", f"the cube center is above z = {LIFT_HEIGHT} m (the task's lift test)"),
        "held_lifted": Quantity(
            lambda env: _lifted(env) & _grasped(env), "flag", "the cube is lifted and grasped"),
        "at_goal": Quantity(
            lambda env: _lifted(env) & (_cube_to_goal(env) < GOAL_RADIUS), "flag",
            f"the cube is lifted and within {GOAL_RADIUS} m of the goal (the task's success test, now)"),
        "succeeded": Quantity(
            _succeeded, "flag",
            "the task's success: at some step of this episode the cube was at the goal (latched; the episode does "
            "not end at success, it lasts 5 s)"),
    }
    return mark_latched(q, "succeeded")


def _franka_lift_funnel():
    """The task's success conditions in order (the share of episodes reaching each is reported after evaluations)."""
    return [
        ("end effector at the cube", lambda env: _ee_to_cube(env) < REACH_RADIUS),
        ("grasped", _grasped),
        (f"cube lifted above {LIFT_HEIGHT} m", _lifted),
        ("lifted while grasped", lambda env: _lifted(env) & _grasped(env)),
        (f"cube within {GOAL_RADIUS} m of the goal while lifted (success)",
         lambda env: _lifted(env) & (_cube_to_goal(env) < GOAL_RADIUS)),
    ]


FUNNELS = {"franka_lift": _franka_lift_funnel}
ADAPTERS = {"franka_lift": _franka_lift}
