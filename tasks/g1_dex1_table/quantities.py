"""Task quantities of the G1-Dex1 table pick-and-place tasks (see ``reward_design/rd_lib/quantities.py``).

The pipeline finds this file through ``ADAPTERS``, ``FUNNELS`` and ``HOLDS``; it must not import the task's
environment module.
"""

import torch

from isaaclab.utils import math as math_utils

from rd_lib.quantities import Quantity, mark_latched


# ---------------------------------------------------------------------------
# G1 table pick-and-place: quantities shared by the tasks below
# ---------------------------------------------------------------------------


def _table_pick_place_base():
    rest_z = 0.725  # cube center height at rest on the table [m]
    pickup_xy = (0.30, -0.15)  # nominal pickup point [m]

    def cube_xy_from_pickup(env):
        pickup = getattr(env, "_pickup_xy", None)  # per-environment pickup point (moving table tasks)
        if pickup is None:
            pickup = torch.tensor(pickup_xy, device=env.device)
        return torch.norm(env._item_pos[:, :2] - pickup, dim=-1)

    q = {
        "palm_to_cube_dist": Quantity(
            lambda env: torch.norm(env._palm_to_item, dim=-1), "scalar",
            "distance from the right palm to the cube center [m]"),
        "cube_height": Quantity(
            lambda env: env._item_pos[:, 2] - getattr(env, "_item_rest_z", rest_z), "scalar",
            "height of the cube center above its rest height on the table [m] (0 on the table, 0.12 = valid-lift height)"),
        "cube_xy_from_pickup": Quantity(
            cube_xy_from_pickup, "scalar",
            "horizontal distance of the cube from the nominal pickup point (0.30, -0.15) [m] (a valid lift needs < 0.10)"),
        "cube_to_target_dist": Quantity(
            lambda env: torch.norm(env._item_to_target, dim=-1), "scalar",
            "distance from the cube to the carry target 0.18 m above the tray center [m]"),
        "cube_to_target_xy_dist": Quantity(
            lambda env: torch.norm(env._item_to_target[:, :2], dim=-1), "scalar",
            "horizontal distance from the cube to the tray center [m] (the release needs both x and y offsets < 0.037)"),
        "cube_speed": Quantity(
            lambda env: torch.norm(env._item.data.root_lin_vel_w.torch, dim=-1), "scalar",
            "linear speed of the cube [m/s]"),
        "hand_closure": Quantity(
            lambda env: env._hand_cmd[:, 0], "scalar",
            "hand command, 0 = open, 1 = closed (closed counts above 0.5, open below 0.3)"),
        "action_rate": Quantity(
            lambda env: torch.sum(torch.square(env.actions - env._prev_actions), dim=-1), "scalar",
            "sum of squared changes of the 8 actions between control steps"),
        "arm_joint_vel_sq": Quantity(
            lambda env: torch.sum(torch.square(env._robot.data.joint_vel.torch[:, env._arm_ids]), dim=-1), "scalar",
            "sum of squared right-arm joint velocities [rad^2/s^2]"),
        "hand_closed": Quantity(
            lambda env: env._hand_cmd[:, 0] > 0.5, "flag",
            "the hand command is above 0.5 (closed)"),
        "grasped": Quantity(
            lambda env: env._grasped, "flag",
            "palm within 0.10 m of the cube, hand closed and cube more than 0.01 m above its rest height "
            "(position estimate, no contact sensing)"),
        "clear_rim": Quantity(
            lambda env: env._clear_rim, "flag",
            "the cube bottom is above the tray rim plus 0.015 m (cube center about 0.11 m above rest)"),
        "near_pickup": Quantity(
            lambda env: cube_xy_from_pickup(env) < 0.10, "flag",
            "the cube is within 0.10 m (horizontally) of the nominal pickup point"),
        "lifted_once": Quantity(
            lambda env: env._lifted_once, "flag",
            "valid lift reached in this episode (latched): grasped, clear of the rim and near the pickup point for "
            "0.1 s with the cube at least 0.12 m above rest"),
        "above_once": Quantity(
            lambda env: env._above_once, "flag",
            "after a valid lift, the grasped cube was above the tray opening (latched)"),
        "released_above": Quantity(
            lambda env: env._released_above, "flag",
            "the cube was released above the tray opening and has not left it (latched until it leaves)"),
        "wall_contact": Quantity(
            lambda env: env._wall_contact, "flag",
            "the cube overlaps the tray walls below the rim"),
        "cube_in_tray": Quantity(
            lambda env: env._item_in_box(), "flag",
            "the cube lies inside the tray on its floor"),
        "succeeded": Quantity(
            lambda env: env._succeeded, "flag",
            "task success: the released cube rests still in the tray for 0.25 s with the hand open and away "
            "(ends the episode)"),
    }
    return mark_latched(q, "lifted_once", "above_once", "succeeded")


def _g1_dex1_pick_place():
    """G1 with Dex1 two-finger grippers and a turning waist: pick the cube and place it into the tray."""
    q = _table_pick_place_base()
    q["action_rate"].description = "sum of squared changes of the 9 actions between control steps"
    q["arm_joint_vel_sq"].description = "sum of squared waist-yaw and right-arm joint velocities [rad^2/s^2]"
    q["palm_to_cube_dist"].description = "distance from the right gripper's grasp point (midpoint between the fingertips) to the cube center [m]"
    q["hand_closure"].description = "gripper command, 0 = open, 1 = closed (closed counts above 0.5, open below 0.3)"
    q["hand_closed"].description = "the gripper command is above 0.5 (closed)"
    q["grasped"].description = (
        "grasp point (midpoint between the two fingertips) within 0.10 m of the cube, gripper closed and cube more "
        "than 0.01 m above its rest height (position estimate, no contact sensing; pressing or tipping the cube also "
        "counts)")
    q["released_above"].description += (
        " (detected from the gripper command, not from contact: a one-step command flick while still holding the cube "
        "also sets it; use cube_released for a physical release)")
    q["cube_height"].description += (
        "; measured at the cube center, which also rises when the cube is only tipped (up to 0.018 m when it stands on "
        "a corner); use cube_bottom_height to reward lifting")
    q["grasped"].description += "; use secure_grasp to reward holding the cube"

    # grasp quality and whole-object state (a pinch on an edge or a tipped cube must not count as grasping or lifting)
    tips = {}

    def tip_ids(env):
        if "ids" not in tips:
            tips["ids"] = env._robot.find_bodies(["right_hand_Link1_3", "right_hand_Link2_3"], preserve_order=True)[0]
        return tips["ids"]

    def cube_tilt(env):
        # angle between the world z axis and the nearest cube face normal (cube symmetric: 0 = flat, up to 0.96 rad)
        r = math_utils.matrix_from_quat(env._item_quat_w)  # columns: the cube's local axes in world coordinates
        return torch.acos(r[:, 2, :].abs().max(dim=-1).values.clamp(max=1.0))

    def cube_bottom_height(env):
        # height of the lowest cube corner above the table top (0 when the cube lies on the table, also when tipped)
        r = math_utils.matrix_from_quat(env._item_quat_w)
        half = env._item_size / 2 if hasattr(env, "_item_size") else 0.05 / 2
        return env._item_pos[:, 2] - half * r[:, 2, :].abs().sum(dim=-1) - getattr(env, "_table_h", 0.70)

    def grasp_face_error(env):
        # angle between the jaws' closing line and the nearest cube face normal, in 3D (a cube turned about the
        # closing line between two faces stays at 0)
        tip = env._robot.data.body_pos_w.torch[:, tip_ids(env)]
        close_dir = tip[:, 1] - tip[:, 0]
        close_dir = close_dir / close_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        r = math_utils.matrix_from_quat(env._item_quat_w)  # columns: the cube's face normals in world coordinates
        cos = (r * close_dir.unsqueeze(-1)).sum(dim=1).abs().max(dim=-1).values
        return torch.acos(cos.clamp(max=1.0))

    def secure_grasp(env):
        # the grasp point (between the fingertips) close to the cube center: a pinch on an edge or a corner leaves it
        # more than 0.04 m away (checked: a policy that lifts the cube 96% of the time has it in 98% of the
        # episodes, a policy that pinches an edge and tips the cube in 0%)
        return (env._hand_cmd[:, 0] > 0.5) & (jaw_position(env) < 0.012) & (palm_dist(env) < 0.04)

    def cube_hand_slip(env):
        tip_vel = env._robot.data.body_lin_vel_w.torch[:, tip_ids(env)].mean(dim=1)
        return torch.norm(env._item.data.root_lin_vel_w.torch - tip_vel, dim=-1)

    q["cube_tilt"] = Quantity(
        cube_tilt, "scalar",
        "tilt of the cube [rad]: angle between the vertical and the nearest cube face normal (0 = lying flat on a "
        "face, 0.79 = standing on an edge, 0.96 = standing on a corner)")
    q["cube_bottom_height"] = Quantity(
        cube_bottom_height, "scalar",
        "height of the lowest cube corner above the table top [m] (0 while any part of the cube touches the table, "
        "also when it is tipped; about 0.12 at the valid-lift height with a flat cube); the measure of lifting")
    q["grasp_face_error"] = Quantity(
        grasp_face_error, "scalar",
        "angle between the jaws' closing line and the nearest cube face normal [rad] (0 = the jaws close on two "
        "opposite faces, about 0.79 = on two edges, 0.96 = on two corners)")
    q["jaws_on_cube"] = Quantity(
        lambda env: (env._hand_cmd[:, 0] > 0.5) & (jaw_position(env) < 0.012) & (palm_dist(env) < 0.05), "flag",
        "the gripper is commanded closed and the jaws are stopped by something (not fully closed) with the grasp "
        "point within 0.05 m of the cube center")
    q["secure_grasp"] = Quantity(
        secure_grasp, "flag",
        "a grasp that can lift the cube: gripper commanded closed, jaws stopped by the cube and the grasp point "
        "(between the fingertips) within 0.04 m of the cube center, i.e. the cube sits deep between the jaws; a pinch "
        "on an edge or a corner, or a cube pressed or tipped against the table, does not count (grasp_face_error "
        "measures the alignment of the jaws with the faces in addition)")
    q["cube_hand_slip"] = Quantity(
        cube_hand_slip, "scalar",
        "speed of the cube relative to the fingertips [m/s] (near 0 while the cube moves with the hand)")
    q["cube_controlled"] = Quantity(
        lambda env: secure_grasp(env) & (cube_bottom_height(env) > 0.005) & (cube_hand_slip(env) < 0.15), "flag",
        "the cube is held off the table and moves with the hand: secure_grasp, lowest corner more than 0.005 m above "
        "the table, and slip speed below 0.15 m/s")

    # physical conditions (from the jaw joints and the cube state, not from commands)
    def jaw_position(env):
        return env._robot.data.joint_pos.torch[:, env._hand_ids].mean(dim=-1)

    def cube_speed(env):
        return torch.norm(env._item.data.root_lin_vel_w.torch, dim=-1)

    def cube_spin(env):
        return torch.norm(env._item.data.root_ang_vel_w.torch, dim=-1)

    palm_dist = lambda env: torch.norm(env._palm_to_item, dim=-1)  # noqa: E731
    q["jaws_open"] = Quantity(
        lambda env: jaw_position(env) < -0.010, "flag",
        "the gripper jaws are physically open (jaw joints below -0.010 m; on the cube they stop near -0.002, fully "
        "open is -0.020); the jaws follow the command with a delay of several steps, so a brief open command keeps this "
        "true for a while after the command is closed again; the task's release and success use the command: use "
        "gripper_open")
    q["gripper_open"] = Quantity(
        lambda env: (env._hand_cmd[:, 0] < 0.3) & (jaw_position(env) < -0.010), "flag",
        "the gripper is open both by command (below 0.3, as the task's release and success require) and physically "
        "(jaws_open)")

    # the task's success conditions, exactly as the environment checks them every step; success needs them all
    # continuously for success_hold_s (0.25 s)
    def success_conditions(env):
        return (env._released_above & env._item_in_box() & (palm_dist(env) > env.cfg.grasp_distance)
                & (env._hand_cmd[:, 0] < 0.3)
                & (torch.norm(env._item.data.root_lin_vel_w.torch, dim=-1) < env.cfg.settle_speed)
                & (torch.norm(env._item.data.root_ang_vel_w.torch, dim=-1) < env.cfg.settle_angular_speed))

    q["success_conditions"] = Quantity(
        success_conditions, "flag",
        "all success conditions of the task hold in this step, exactly as the environment checks them: released_above, "
        "cube_in_tray, grasp point more than 0.10 m from the cube, gripper command below 0.3, cube speed below 0.10 m/s "
        "and angular speed below 1.0 rad/s; success needs them all continuously for 0.25 s (any condition false for one "
        "step restarts the count)")
    q["success_hold_s"] = Quantity(
        lambda env: env._settle_time, "scalar",
        "time for which success_conditions has held continuously [s] (success at 0.25)")
    def over_tray(env):
        # cube center inside the tray's inner area (in the tray frame; the tray can be turned)
        inner = getattr(env, "_tray_inner", None)  # per-environment tray inside size (x, y), when the tray varies
        if inner is None:
            inner = torch.full((env.num_envs, 2), 0.20 - 0.01, device=env.device)
        margin = env._item_size / 2 if getattr(env.cfg, "placement_anywhere_in_tray", False) else env._item_radius
        tray_half = inner / 2 - margin.unsqueeze(-1)
        return (env._tray_xy(env._item_pos).abs() < tray_half).all(dim=-1)

    q["cube_released"] = Quantity(
        lambda env: (jaw_position(env) < -0.010) & (env._hand_cmd[:, 0] < 0.3) & (palm_dist(env) > 0.04)
        & env._above_once & over_tray(env)
        & (env._item_pos[:, 2] < getattr(env, "_table_h", 0.70) + getattr(env, "_tray_wall_h", 0.08)), "flag",
        "the cube has been carried over the tray (above_once) and is now physically out of the gripper (jaws open, "
        "gripper command open, cube more than 0.04 m from the grasp point) and inside the tray (center within the "
        "tray's inner area and below the rim, z < 0.78); a cube dropped elsewhere, or thrown from high up so that it "
        "bounces out, does not count")
    q["cube_at_rest"] = Quantity(
        lambda env: (cube_speed(env) < 0.05) & (cube_spin(env) < 0.5), "flag",
        "the cube is at rest (speed below 0.05 m/s, angular speed below 0.5 rad/s)")
    def over_opening(env):
        # the cube above the tray opening now (in the tray frame) and clear of the rim: where a release can start
        half = (0.20 - 0.01) / 2 - env._item_radius - 0.015
        return (env._tray_xy(env._item_pos).abs() < half.unsqueeze(-1)).all(dim=-1) & env._clear_rim

    q["cube_over_opening"] = Quantity(
        over_opening, "flag",
        "the cube is above the tray opening now and clear of the rim (not latched, unlike above_once, which stays "
        "true after the cube has passed over the tray once)")
    q["hand_away"] = Quantity(
        lambda env: palm_dist(env) > 0.10, "flag",
        "the grasp point is more than 0.10 m from the cube (the hand has moved away)")
    return q


def _g1_dex1_pick_place_funnel():
    """Success conditions of the task, in order, for the checkpoint evaluation (each: ever true in an episode)."""
    q = _g1_dex1_pick_place()
    in_tray = lambda env: q["cube_in_tray"].fn(env) & env._above_once  # noqa: E731
    return [
        ("grasped", q["grasped"].fn),
        ("secure grasp", q["secure_grasp"].fn),
        ("cube held off the table and moving with the hand", q["cube_controlled"].fn),
        ("valid lift", lambda env: env._lifted_once),
        ("carried over the tray", lambda env: env._above_once),
        ("released (task flag)", lambda env: env._released_above),
        ("cube physically released into the tray", q["cube_released"].fn),
        ("cube in tray", in_tray),
        ("cube in tray and released (task flag)", lambda env: in_tray(env) & env._released_above),
        ("... and hand away", lambda env: in_tray(env) & env._released_above & q["hand_away"].fn(env)),
        ("... and gripper open", lambda env: in_tray(env) & env._released_above & q["hand_away"].fn(env)
         & (env._hand_cmd[:, 0] < 0.3)),
        ("... and cube at rest", lambda env: in_tray(env) & env._released_above & q["hand_away"].fn(env)
         & (env._hand_cmd[:, 0] < 0.3) & q["cube_at_rest"].fn(env)),
        ("all success conditions at once (the 0.25 s hold starts)", q["success_conditions"].fn),
    ]


def _g1_dex1_pick_place_hold():
    """The success hold of the task: (timer [s], required time [s], [(condition name, fn)])."""
    palm_dist = lambda env: torch.norm(env._palm_to_item, dim=-1)  # noqa: E731
    return (lambda env: env._settle_time, 0.25, [
        ("release over the opening (released_above)", lambda env: env._released_above),
        ("cube in the tray", lambda env: env._item_in_box()),
        ("hand more than 0.10 m away", lambda env: palm_dist(env) > env.cfg.grasp_distance),
        ("gripper command below 0.3", lambda env: env._hand_cmd[:, 0] < 0.3),
        ("cube speed below 0.10 m/s", lambda env: torch.norm(env._item.data.root_lin_vel_w.torch, dim=-1)
         < env.cfg.settle_speed),
        ("cube angular speed below 1.0 rad/s", lambda env: torch.norm(env._item.data.root_ang_vel_w.torch, dim=-1)
         < env.cfg.settle_angular_speed),
    ])


# ---------------------------------------------------------------------------
# G1 Dex1 pick, place and return (G1-Dex1-Pick-Place-Return-v0)
# ---------------------------------------------------------------------------


def _arm_error(env):
    joint_pos = env._robot.data.joint_pos.torch[:, env._arm_ids]
    return torch.norm(joint_pos - env._robot.data.default_joint_pos.torch[:, env._arm_ids], dim=-1)


def _arm_speed(env):
    return torch.norm(env._robot.data.joint_vel.torch[:, env._arm_ids], dim=-1)


def _return_conditions(env):
    return (env._placed & env._item_in_box()
            & (torch.norm(env._item.data.root_lin_vel_w.torch, dim=-1) < env.cfg.settle_speed)
            & (env._hand_cmd[:, 0] < 0.3) & (_arm_error(env) < env.cfg.return_joint_tolerance)
            & (_arm_speed(env) < env.cfg.return_max_joint_speed))


def _g1_dex1_pick_place_return():
    q = _g1_dex1_pick_place()
    q["success_conditions"].description = q["success_conditions"].description.replace(
        "all success conditions of the task", "all placement conditions").replace(
        "success needs them all continuously for 0.25 s", "the placement (placed) needs them all continuously for 0.25 s")
    q["success_hold_s"].description = "time for which success_conditions (the placement conditions) has held [s] (placed at 0.25)"
    q["placed"] = Quantity(
        lambda env: env._placed, "flag",
        "the cube has been placed (latched): released over the opening, resting in the tray with the gripper open and the "
        "hand away for 0.25 s; this is the first half of success, the arm must then return to its start pose")
    q["arm_error_from_start"] = Quantity(
        _arm_error, "scalar",
        "norm of the 8 arm joint (waist yaw and right arm) differences from the start pose [rad] (the episode starts "
        "within about 0.1-0.28 of it; the return needs < 0.3)")
    q["arm_speed"] = Quantity(
        _arm_speed, "scalar", "norm of the 8 arm joint speeds [rad/s] (the return needs < 0.5)")
    q["return_conditions"] = Quantity(
        _return_conditions, "flag",
        "all success conditions of the task hold in this step, exactly as the environment checks them: placed, cube "
        "still in the tray and slower than 0.10 m/s, gripper command below 0.3, arm_error_from_start below 0.3 and "
        "arm_speed below 0.5; success needs them all continuously for 0.25 s")
    q["return_hold_s"] = Quantity(
        lambda env: env._return_time, "scalar",
        "time for which return_conditions has held continuously [s] (success at 0.25)")
    q["succeeded"].description = (
        "task success: the cube is placed (placed) and then the arm is back at its start pose, still, with the gripper "
        "open and the cube still in the tray, for 0.25 s (return_conditions; ends the episode)")
    return mark_latched(q, "placed")


def _g1_dex1_pick_place_return_funnel():
    funnel = [item for item in _g1_dex1_pick_place_funnel()
              if not item[0].startswith("all success conditions")]
    q = _g1_dex1_pick_place_return()
    return funnel + [
        ("placement conditions at once (the 0.25 s placement hold starts)", q["success_conditions"].fn),
        ("placed (placement held 0.25 s)", q["placed"].fn),
        ("... and arm within 0.3 rad of the start pose", lambda env: env._placed & (_arm_error(env) < 0.3)),
        ("all success conditions at once (the 0.25 s return hold starts)", q["return_conditions"].fn),
    ]


def _g1_dex1_pick_place_return_hold():
    return (lambda env: env._return_time, 0.25, [
        ("placed", lambda env: env._placed),
        ("cube in the tray", lambda env: env._item_in_box()),
        ("cube speed below 0.10 m/s", lambda env: torch.norm(env._item.data.root_lin_vel_w.torch, dim=-1)
         < env.cfg.settle_speed),
        ("gripper command below 0.3", lambda env: env._hand_cmd[:, 0] < 0.3),
        ("arm within 0.3 rad of the start pose", lambda env: _arm_error(env) < env.cfg.return_joint_tolerance),
        ("arm speed below 0.5 rad/s", lambda env: _arm_speed(env) < env.cfg.return_max_joint_speed),
    ])


def _g1_dex1_pick_place_return_random():
    """Pick, place and return with a varied layout (G1-Dex1-Pick-Place-Return-Random-v0): the cube
    edge is 0.04-0.06 m (fixed per environment), the tray center and yaw change at every reset."""
    q = _g1_dex1_pick_place_return()
    q["cube_height"].description = (
        "height of the cube center above its rest height on the table [m] (0 on the table, 0.12 = valid-lift height); "
        "measured at the cube center, which also rises when the cube is only tipped; use cube_bottom_height to reward "
        "lifting")
    q["cube_to_target_xy_dist"].description = (
        "horizontal distance from the cube to the tray center [m] (the release needs the cube within the opening: "
        "both offsets along the tray's walls below 0.095 - 0.015 - the cube's bounding radius, i.e. about 0.045 for a "
        "4 cm cube and 0.028 for a 6 cm cube)")
    q["cube_size"] = Quantity(
        lambda env: env._item_size, "scalar", "edge length of this environment's cube [m] (0.04 to 0.06)")
    q["tray_yaw"] = Quantity(
        lambda env: env._box_yaw, "scalar", "yaw of the tray about its center [rad] (-0.79 to 0.79)")
    return q


FUNNELS = {
    "g1_dex1_pick_place": _g1_dex1_pick_place_funnel,
    "g1_dex1_pick_place_return": _g1_dex1_pick_place_return_funnel,
    "g1_dex1_pick_place_return_random": _g1_dex1_pick_place_return_funnel,
}


HOLDS = {
    "g1_dex1_pick_place": _g1_dex1_pick_place_hold,
    "g1_dex1_pick_place_return": _g1_dex1_pick_place_return_hold,
    "g1_dex1_pick_place_return_random": _g1_dex1_pick_place_return_hold,
}


ADAPTERS = {
    "g1_dex1_pick_place": _g1_dex1_pick_place,
    "g1_dex1_pick_place_return": _g1_dex1_pick_place_return,
    "g1_dex1_pick_place_return_random": _g1_dex1_pick_place_return_random,
}
