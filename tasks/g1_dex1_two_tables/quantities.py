"""Task quantities of the G1-Dex1 two-table task (see ``reward_design/rd_lib/quantities.py``).

The grasp, lift, carry, release and placement quantities are the table pick-and-place task's (adapter
``g1_dex1_pick_place`` of ``tasks/g1_dex1_table``), with the descriptions and the measures that depend on the layout
redone for two tables: the cube starts on the cube table, the tray stands on the tray table, and both tables, the
tray and the cube vary. Walking quantities are added. The pipeline finds this file through ``ADAPTERS``,
``FUNNELS`` and ``HOLDS``; it must not import the task's environment module.
"""

import torch

from isaaclab.utils import math as math_utils

from rd_lib.quantities import Quantity, get_adapter, get_funnel, get_hold, mark_latched

# base-to-object horizontal distances from which the right gripper can work [m]. Reach check (standing walker pose,
# jaws horizontal, hand above the table): the feet stop the pelvis about 0.17 m from the table edge, the grasp point
# reaches about 0.29-0.38 m ahead of the pelvis on tables of 0.75-0.90 m (only about 0.17 m on a 0.60 m table)
REACH = 0.38  # to the cube
TRAY_REACH = 0.45  # to the tray center (the tray's center is at most about 0.21 m in from the table edge)
# right jaw joint position at which the jaws would touch a cube of zero size [m]: the fingertip gap is 0.050 - 2 x joint
# (measured: 0.090 m fully open at -0.020, 0.002 m closed at 0.024); the pads stop about 2 mm per jaw earlier
JAW_TOUCH_AT_ZERO = 0.023


def _base_xy(env):
    return env._robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]


def _base_yaw(env):
    x, y, z, w = env._robot.data.root_quat_w.torch.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _heading_error(env, target_xy):
    to_target = target_xy - _base_xy(env)
    angle = torch.atan2(to_target[:, 1], to_target[:, 0]) - _base_yaw(env)
    return torch.atan2(torch.sin(angle), torch.cos(angle)).abs()


def _base_to_cube(env):
    return torch.norm(env._item_pos[:, :2] - _base_xy(env), dim=-1)


def _base_to_tray(env):
    return torch.norm(env._box_center_local[:, :2] - _base_xy(env), dim=-1)


def _left_cam_view(env, target):
    """Where an env-local point is in the left wrist camera's view: (angle from the optical axis [rad], horizontal and
    vertical angle [rad] (positive right / down), distance [m])."""
    d = target - env._left_cam_pos
    x, y, z = ((d * env._left_cam_axes[:, k]).sum(dim=-1) for k in range(3))
    dist = torch.norm(d, dim=-1)
    return torch.acos((z / dist.clamp_min(1e-6)).clamp(-1.0, 1.0)), torch.atan2(x, z), torch.atan2(y, z), dist


def _in_left_view(env, target):
    """The point is inside the left wrist camera's image (with a 0.05 rad margin) and 0.08-1.2 m in front of it."""
    angle, h, v, dist = _left_cam_view(env, target)
    half_h, half_v = env.cfg.left_camera_half_fov
    return (h.abs() < half_h - 0.05) & (v.abs() < half_v - 0.05) & (angle < torch.pi / 2) & (dist > 0.08) & (dist < 1.2)


def _g1_dex1_two_tables():
    """G1 walking with Unitree's walking policy, two Dex1 grippers: walk to the cube table, pick the cube with the right
    gripper, carry it to the tray on the tray table and place it there, while the left arm aims its wrist camera at the
    right hand and the cube."""
    q = get_adapter("g1_dex1_pick_place")

    # the right gripper: the pick-and-place quantities that read its joints or fingertips are redone with the
    # environment's values of it (the joint and body indices differ in this robot setup)
    palm_dist = lambda env: torch.norm(env._palm_to_item, dim=-1)  # noqa: E731
    jaw = lambda env: env._right_jaw_pos  # noqa: E731
    # jaw joint position at which the jaws touch this environment's cube (each jaw closes half the gap; on a 0.05 m
    # cube they stop near -0.002; a scripted grasp of every size stops within 1 mm of it): stopped by the cube = within
    # 3 mm short of it to 6 mm past it (still open does not count), open = 5 mm wider than the cube
    on_cube = lambda env: JAW_TOUCH_AT_ZERO - env._item_size / 2  # noqa: E731
    stopped = lambda env: (jaw(env) > on_cube(env) - 0.003) & (jaw(env) < on_cube(env) + 0.006)  # noqa: E731
    opened = lambda env: jaw(env) < on_cube(env) - 0.005  # noqa: E731

    def secure_grasp(env):
        return (env._hand_cmd[:, 0] > 0.5) & stopped(env) & (palm_dist(env) < 0.04)

    def grasp_face_error(env):
        close_dir = env._right_tips[:, 1] - env._right_tips[:, 0]
        close_dir = close_dir / close_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        r = math_utils.matrix_from_quat(env._item_quat_w)
        cos = (r * close_dir.unsqueeze(-1)).sum(dim=1).abs().max(dim=-1).values
        return torch.acos(cos.clamp(max=1.0))

    def cube_hand_slip(env):
        return torch.norm(env._item.data.root_lin_vel_w.torch - env._right_tip_vel, dim=-1)

    def in_tray_area(env):
        margin = env._item_size / 2 if env.cfg.placement_anywhere_in_tray else env._item_radius
        half = env._tray_inner / 2 - margin.unsqueeze(-1)
        return (env._tray_xy(env._item_pos).abs() < half).all(dim=-1)

    q["secure_grasp"] = Quantity(
        secure_grasp, "flag",
        "a grasp that can lift the cube: the right gripper commanded closed, its jaws stopped by the cube and its grasp "
        "point (between the fingertips) within 0.04 m of the cube center, i.e. the cube sits deep between the jaws; a "
        "pinch on an edge or a corner, or a cube pressed or tipped against the table, does not count")
    q["jaws_on_cube"] = Quantity(
        lambda env: (env._hand_cmd[:, 0] > 0.5) & stopped(env) & (palm_dist(env) < 0.05), "flag",
        "the right gripper is commanded closed and its jaws are stopped at the cube's width (not still open, not fully "
        "closed) with the grasp point within 0.05 m of the cube center")
    q["grasp_face_error"] = Quantity(
        grasp_face_error, "scalar",
        "angle between the right gripper's closing line and the nearest cube face normal [rad] (0 = the jaws close on "
        "two opposite faces, about 0.79 = on two edges, 0.96 = on two corners)")
    q["cube_hand_slip"] = Quantity(
        cube_hand_slip, "scalar",
        "speed of the cube relative to the right gripper's fingertips [m/s] (near 0 while the cube moves with the hand)")
    q["jaws_open"] = Quantity(
        opened, "flag",
        "the right gripper's jaws are physically open, at least 5 mm per jaw wider than the cube (each jaw joint stops "
        "at about 0.023 - cube_size/2 m on the cube, fully open is -0.020, so a 0.07 m cube leaves only about 8 mm per "
        "jaw); the jaws follow the command with a delay of several steps")
    q["gripper_open"] = Quantity(
        lambda env: (env._hand_cmd[:, 0] < 0.3) & opened(env), "flag",
        "the right gripper is open both by command (below 0.3, as the task's release and success require) and "
        "physically (jaws_open)")
    q["cube_released"] = Quantity(
        lambda env: opened(env) & (env._hand_cmd[:, 0] < 0.3) & (palm_dist(env) > 0.04) & env._above_once
        & in_tray_area(env) & (env._item_pos[:, 2] < env._table_h + env._tray_wall_h), "flag",
        "the cube has been carried over the tray (above_once) and is now physically out of the right gripper (jaws open, "
        "gripper commanded open, grasp point more than 0.04 m from the cube) and inside the tray (center within the "
        "tray's inner area and below the rim); a cube dropped elsewhere, or thrown so that it bounces out, does not "
        "count")
    q["palm_to_cube_dist"].description = (
        "distance from the right gripper's grasp point (midpoint between its fingertips) to the cube center [m]")
    q["hand_closure"].description = "right gripper command, 0 = open, 1 = closed (closed counts above 0.5, open below 0.3)"
    q["hand_closed"].description = "the right gripper command is above 0.5 (closed)"
    q["grasped"].description = (
        "the right gripper's grasp point within 0.10 m of the cube, the gripper closed and the cube more than 0.01 m "
        "above its rest height (position estimate, no contact sensing; pressing or tipping the cube also counts); use "
        "secure_grasp to reward holding the cube")
    q["hand_away"].description = "the right gripper's grasp point is more than 0.10 m from the cube"

    # the left arm: its wrist camera watches the right hand and the cube; the left gripper stays open
    right_palm = lambda env: env._palm_pos  # noqa: E731
    cube = lambda env: env._item_pos  # noqa: E731
    q["left_cam_angle_to_right_gripper"] = Quantity(
        lambda env: _left_cam_view(env, right_palm(env))[0], "scalar",
        "angle between the left wrist camera's optical axis and the direction from it to the right gripper's grasp "
        "point [rad] (0 = the right gripper in the image center; the image spans +-0.82 rad horizontally, +-0.67 "
        "vertically)")
    q["left_cam_angle_to_cube"] = Quantity(
        lambda env: _left_cam_view(env, cube(env))[0], "scalar",
        "angle between the left wrist camera's optical axis and the direction from it to the cube center [rad]")
    q["left_cam_dist_to_right_gripper"] = Quantity(
        lambda env: _left_cam_view(env, right_palm(env))[3], "scalar",
        "distance from the left wrist camera to the right gripper's grasp point [m] (about 0.2-0.5 gives a useful close "
        "view)")
    q["right_gripper_in_left_view"] = Quantity(
        lambda env: _in_left_view(env, right_palm(env)), "flag",
        "the right gripper's grasp point is inside the left wrist camera's image (0.05 rad from its edges) and 0.08-1.2 "
        "m in front of the camera")
    q["cube_in_left_view"] = Quantity(
        lambda env: _in_left_view(env, cube(env)), "flag",
        "the cube center is inside the left wrist camera's image and 0.08-1.2 m in front of the camera")
    q["work_in_left_view"] = Quantity(
        lambda env: _in_left_view(env, right_palm(env)) & _in_left_view(env, cube(env)), "flag",
        "both the right gripper's grasp point and the cube are inside the left wrist camera's image (the left camera "
        "watches the grasp)")
    q["hands_dist"] = Quantity(
        lambda env: torch.norm(env._palms[:, 1] - env._palms[:, 0], dim=-1), "scalar",
        "distance between the two grippers' grasp points [m]; the simulation does not model collisions between the "
        "robot's own links, so the arms can pass through each other here but not on the real robot: keep it above "
        "about 0.12")
    q["left_hand_to_cube_dist"] = Quantity(
        lambda env: torch.norm(env._item_pos - env._palms[:, 1], dim=-1), "scalar",
        "distance from the left gripper's grasp point to the cube center [m] (the left hand must not touch or push the "
        "cube: keep it above about 0.08)")

    # layout-dependent descriptions and measures
    q["action_rate"].description = "sum of squared changes of the 18 actions between control steps"
    q["arm_joint_vel_sq"].description = "sum of squared arm joint velocities (both arms, 14 joints) [rad^2/s^2]"
    q["cube_height"].description = (
        "height of the cube center above its rest height on the cube table [m] (0 on the table, 0.12 = valid-lift "
        "height); measured at the cube center, which also rises when the cube is only tipped; use cube_bottom_height "
        "to reward lifting")
    q["cube_xy_from_pickup"].description = (
        "horizontal distance of the cube from its start position on the cube table [m] (a valid lift needs < 0.10)")
    q["near_pickup"].description = "the cube is within 0.10 m (horizontally) of its start position"
    q["cube_to_target_xy_dist"].description = (
        "horizontal distance from the cube to the tray center [m] (the release needs the cube over the opening: both "
        "offsets along the tray's walls below half the tray's inner size minus half the cube edge)")
    q["clear_rim"].description = (
        "the cube bottom is above the tray rim (tray table height + tray wall height) plus 0.015 m")
    q["lifted_once"].description = (
        "valid lift reached in this episode (latched): grasped, clear of the tray rim and within 0.10 m of the cube's "
        "start position for 0.1 s with the cube at least 0.12 m above its rest height on the cube table; if the tray "
        "table is much higher than the cube table, the lift must reach above its rim")

    def cube_bottom_height(env):
        # lowest cube corner above the cube table top
        r = math_utils.matrix_from_quat(env._item_quat_w)
        return env._item_pos[:, 2] - env._item_size / 2 * r[:, 2, :].abs().sum(dim=-1) - env._cube_table_top[:, 2]

    q["cube_bottom_height"] = Quantity(
        cube_bottom_height, "scalar",
        "height of the lowest cube corner above the cube table top [m] (0 while the cube lies on the cube table, also "
        "when tipped; about 0.12 at the valid-lift height with a flat cube); the measure of lifting")
    q["cube_controlled"] = Quantity(
        lambda env: secure_grasp(env) & (cube_bottom_height(env) > 0.005) & (cube_hand_slip(env) < 0.25), "flag",
        "the cube is held off the cube table and moves with the hand: secure_grasp, lowest corner more than 0.005 m "
        "above the cube table, and slip speed below 0.25 m/s (a firmly held cube reads up to about 0.2 from the robot's sway)")
    q["cube_released"].description = q["cube_released"].description.replace(
        "below the rim, z < 0.78", "below the rim")

    def over_opening(env):
        half = env._tray_inner / 2 - (env._item_size / 2).unsqueeze(-1)
        return (env._tray_xy(env._item_pos).abs() < half).all(dim=-1) & env._clear_rim

    q["cube_over_opening"] = Quantity(
        over_opening, "flag",
        "the cube is above the tray opening now and clear of the rim, where a release can start (not latched, unlike "
        "above_once, which stays true after the cube has passed over the tray once)")

    # layout
    q["cube_size"] = Quantity(lambda env: env._item_size, "scalar", "edge length of this environment's cube [m] (0.025 to 0.07)")
    q["cube_table_height"] = Quantity(
        lambda env: env._cube_table_top[:, 2], "scalar", "height of the cube table top [m] (0.72 to 0.90)")
    q["tray_table_height"] = Quantity(
        lambda env: env._tray_table_top[:, 2], "scalar", "height of the tray table top [m] (0.60 to 0.90)")
    q["tray_wall_height"] = Quantity(
        lambda env: env._tray_wall_h, "scalar", "height of this environment's tray walls [m] (0.04 to 0.12)")

    # walking
    q["base_to_cube_dist"] = Quantity(
        _base_to_cube, "scalar",
        f"horizontal distance from the robot base (pelvis) to the cube [m] (the gripper can work on it below about "
        f"{REACH})")
    q["base_to_tray_dist"] = Quantity(
        _base_to_tray, "scalar",
        f"horizontal distance from the robot base (pelvis) to the tray center [m] (the gripper can place below about "
        f"{TRAY_REACH})")
    q["heading_to_cube_error"] = Quantity(
        lambda env: _heading_error(env, env._item_pos[:, :2]), "scalar",
        "angle between the robot's heading and the horizontal direction from its base to the cube [rad] (0 = facing it)")
    q["heading_to_tray_error"] = Quantity(
        lambda env: _heading_error(env, env._box_center_local[:, :2]), "scalar",
        "angle between the robot's heading and the horizontal direction from its base to the tray center [rad]")
    q["base_speed"] = Quantity(
        lambda env: torch.norm(env._robot.data.root_lin_vel_w.torch[:, :2], dim=-1), "scalar",
        "horizontal speed of the robot base [m/s] (the walking policy walks at up to about 0.8)")
    q["base_turn_rate"] = Quantity(
        lambda env: env._robot.data.root_ang_vel_w.torch[:, 2].abs(), "scalar",
        "turn rate of the robot base about the vertical [rad/s]")
    q["walk_command"] = Quantity(
        lambda env: torch.norm(env._walk_cmd[:, :2], dim=-1), "scalar",
        "commanded walking speed (forward and sideways) given to the walking policy [m/s] (it ignores small commands: "
        "about < 0.5 m/s forward, < 0.3-0.5 m/s sideways, < 1 rad/s turning)")
    q["base_height"] = Quantity(
        lambda env: env._robot.data.root_pos_w.torch[:, 2], "scalar",
        "height of the robot base (pelvis) [m] (about 0.78 standing; the robot counts as fallen below 0.45)")
    q["fell"] = Quantity(
        lambda env: env._fell, "flag",
        "the robot fell (base below 0.45 m or body tilted more than 50 deg): ends the episode; true only in that one "
        "step, so a penalty on it is paid once (weight x 0.02 s when scaled by the time step); potential terms give back "
        "everything they paid when the robot falls")
    q["at_cube_table"] = Quantity(
        lambda env: _base_to_cube(env) < REACH, "flag",
        f"the robot base is within {REACH} m (horizontally) of the cube, close enough for the gripper to reach it")
    q["at_tray_table"] = Quantity(
        lambda env: _base_to_tray(env) < TRAY_REACH, "flag",
        f"the robot base is within {TRAY_REACH} m (horizontally) of the tray center, close enough to place the cube")
    controlled = q["cube_controlled"].fn
    q["carried_to_tray_table"] = Quantity(
        lambda env: env._lifted_once & controlled(env) & (_base_to_tray(env) < TRAY_REACH), "flag",
        f"after a valid lift, the robot holds the cube (cube_controlled) with its base within {TRAY_REACH} m of the tray "
        "center")
    q["before_lift"] = Quantity(
        lambda env: ~env._lifted_once, "flag", "no valid lift yet in this episode (the walk to the cube table)")
    q["succeeded"].description = (
        "task success: the released cube rests still in the tray for 0.25 s with the gripper open and away (ends the "
        "episode)")
    return mark_latched(q, "lifted_once", "above_once", "succeeded")


def _g1_dex1_two_tables_funnel():
    """Success conditions of the task, in order, for the checkpoint evaluation (each: ever true in an episode)."""
    q = _g1_dex1_two_tables()
    # the pick-and-place funnel with this task's grasp and release measures (the right gripper; the cube table)
    own = {"secure grasp": "secure_grasp", "cube held off the table and moving with the hand": "cube_controlled",
           "cube physically released into the tray": "cube_released"}
    funnel = [(name, q[own[name]].fn if name in own else fn) for name, fn in get_funnel("g1_dex1_pick_place")]
    carried = next(i for i, (name, _) in enumerate(funnel) if name == "carried over the tray")
    return (
        [("at the cube table (base within reach of the cube)", q["at_cube_table"].fn)]
        + funnel[:carried]
        + [("carried to the tray table (cube held, base within reach of the tray)", q["carried_to_tray_table"].fn)]
        + funnel[carried:]
    )


def _g1_dex1_two_tables_hold():
    """The success hold: the placement conditions of the pick-and-place task."""
    return get_hold("g1_dex1_pick_place")


FUNNELS = {"g1_dex1_two_tables": _g1_dex1_two_tables_funnel}
HOLDS = {"g1_dex1_two_tables": _g1_dex1_two_tables_hold}
ADAPTERS = {"g1_dex1_two_tables": _g1_dex1_two_tables}
