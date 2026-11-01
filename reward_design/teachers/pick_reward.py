"""Hand-written reward of the picking teacher (task G1-Dex1-Two-Tables-Pick-v0): grasp the cube with the right gripper
and lift it. Used with isaaclab_backend/train.py --reward_file; values are per second (multiplied by the step time).

Ordered so that each further step pays more per second than the one before (reach < aligned and close < secure grasp <
lift), the gripper is paid for staying open away from the cube and for closing only around it, and the cube pushed away
from where it lay costs reward.
"""

import math

import torch


def compute_reward(env):
    dt = env.step_dt
    palm = torch.norm(env._palm_to_item, dim=-1)
    cmd = env._hand_cmd[:, 0]
    jaw = env._right_jaw_pos
    on_cube = 0.023 - env._item_size / 2  # jaw joint position at which the jaws touch this cube
    stopped = (jaw > on_cube - 0.003) & (jaw < on_cube + 0.006)
    secure = (cmd > 0.5) & stopped & (palm < 0.04)
    height = env._item_pos[:, 2] - env._item_rest_z
    # lift target: past the task's success height (0.15 m, valid lift at 0.12 m) and clear of the tray rim on a higher
    # tray table, with 0.02 m to spare
    rim_need = env._table_h + env._tray_wall_h + env._item_radius + env.cfg.wall_clearance - env._item_rest_z
    target = torch.clamp(rim_need, min=env.cfg.pick_lift_height) + 0.02
    pushed = torch.norm(env._item_pos[:, :2] - env._pickup_xy, dim=-1)
    # the cube off the table ends the episode (as the task's termination): without a cost, ending early escapes the
    # per-second penalties, so knocking the cube off would pay
    lower_top = torch.minimum(env._cube_table_top[:, 2], env._tray_table_top[:, 2])
    dropped = env._item_pos[:, 2] < lower_top - env.cfg.fall_height

    # jaws closing across two opposite faces: angle between the closing line and the nearest face normal
    close_dir = env._right_tips[:, 1] - env._right_tips[:, 0]
    close_dir = close_dir / close_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    import isaaclab.utils.math as math_utils

    r = math_utils.matrix_from_quat(env._item_quat_w)
    face = torch.acos((r * close_dir.unsqueeze(-1)).sum(dim=1).abs().max(dim=-1).values.clamp(max=1.0))

    reach = 1.0 - torch.tanh(palm / 0.10)  # up to 1
    near = palm < 0.06
    align = near.float() * (1.0 - face / (math.pi / 4)).clamp(min=0.0)  # up to 1 near the cube
    open_far = (palm > 0.06).float() * (1.0 - cmd)  # keep the jaws open on the way
    close_near = (palm < 0.03).float() * cmd  # close with the cube between the jaws
    grasp = secure.float()
    lift = secure.float() * (height / target).clamp(0.0, 1.0)
    held_high = secure & (height > target - 0.02) & env._clear_rim

    # penalties on states, not on raw actions (those would mostly punish the exploration noise): the base drifting
    # from where it stood, the left arm leaving its start pose
    drift = torch.norm(env._base_pos[:, :2] - env._start_base_xy, dim=-1)
    left_dev = ((env._arm_targets[:, 7:] - env._arm_start_targets[:, 7:]) ** 2).sum(dim=-1)

    terms = {
        "reach": 2.0 * reach,
        "align": 1.0 * align,
        "open_far": 0.3 * open_far,
        "close_near": 1.0 * close_near,
        "secure_grasp": 3.0 * grasp,
        "lift": 6.0 * lift,
        "held_high": 4.0 * held_high.float(),
        "cube_pushed": -3.0 * (pushed * (~secure).float()).clamp(max=0.3),
        "base_drift": -2.0 * drift.clamp(max=0.5),
        "left_arm_pose": -0.2 * left_dev.clamp(max=4.0),
        "fell": -50.0 * env._fell.float() / dt,
        "cube_dropped": -30.0 * dropped.float() / dt,
        "success": 50.0 * env._succeeded.float() / dt,
    }
    terms = {k: v * dt for k, v in terms.items()}
    return sum(terms.values()), terms
