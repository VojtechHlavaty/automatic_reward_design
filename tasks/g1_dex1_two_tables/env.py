from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis, sample_uniform, subtract_frame_transforms

from g1_dex1_table.env import G1Dex1TablePickPlaceEnv, _log_value

from .env_cfg import TABLE_BLOCK_H, TABLE_TOP, G1Dex1TwoTablesEnvCfg
from .walker import ARM_JOINTS, UnitreeWalker


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
    """Yaw [rad] of (x, y, z, w) quaternions."""
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _near_edge(depth: torch.Tensor, room: torch.Tensor, far_end: float) -> torch.Tensor:
    """Offsets [n, 2] in a table's frame (x across its 0.6 m depth, y along its 1.0 m width) of points ``depth`` in from
    one of its free edges: the long sides x = +-0.3 or the far end y = far_end * 0.5 (the other end faces the other
    table). The edge is drawn in proportion to its length; along the edge the point is drawn within +- ``room`` [n, 2]."""
    n, dev = len(depth), depth.device
    full = torch.tensor(TABLE_TOP, device=dev) / 2
    edge = torch.multinomial(torch.tensor([TABLE_TOP[1], TABLE_TOP[1], TABLE_TOP[0]], device=dev), n, replacement=True)
    along = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * room
    x = torch.where(edge == 0, full[0] - depth, torch.where(edge == 1, depth - full[0], along[:, 0]))
    y = torch.where(edge == 2, far_end * (full[1] - depth), along[:, 1])
    return torch.stack((x, y), dim=-1)


def _rotate_xy(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]), dim=-1)


class G1Dex1TwoTablesEnv(G1Dex1TablePickPlaceEnv):
    """The G1 walks to the cube table, picks the cube with its right gripper and places it into the tray on the other
    table, while its left arm aims the left wrist camera at the right hand and the cube.

    Unitree's walking policy drives the legs (and holds the waist); the task policy commands the walk and moves both
    arms and the right gripper; the left gripper stays open. The task state (grasp, lift, carry, release, placement)
    is computed for the right gripper as in the table pick-and-place task, in the environment-local frame: the lift is
    measured from the cube table, the rim and the placement from the tray table. The left wrist camera's pose (Adam
    Basl's calibration of the real G1-D) is computed every step, for the quantities that measure what it sees. The
    robot starts at the environment origin; the layout is new at every reset.

    Actions (18, each in [-1, 1]): walking command (forward speed, sideways speed, turn rate), the 7 right-arm joint
    target speeds, the right gripper command (-1 open, +1 closed), the 7 left-arm joint target speeds.
    """

    cfg: G1Dex1TwoTablesEnvCfg

    def __init__(self, cfg: G1Dex1TwoTablesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        joint_names = list(self._robot.joint_names)
        # the walker's standing pose (its observations are relative to it; the robot's own start pose differs: the arms
        # start raised as in the pick-and-place task)
        default = self._robot.data.default_joint_pos.torch.clone()
        for pattern, value in cfg.walker_default_joint_pos.items():
            ids, _ = self._robot.find_joints(pattern)
            default[:, ids] = value
        self._walker = UnitreeWalker(
            cfg.walker_policy, joint_names, default, self.num_envs, self.device, body_height=cfg.walker_body_height
        )
        self._leg_ids = self._walker.leg_ids.tolist()
        self._waist_ids = self._walker.waist_ids.tolist()
        self._arm_in_walker = [ARM_JOINTS.index(joint_names[i]) for i in self._arm_ids]
        # grippers (0 right, 1 left): the gripper of each jaw joint, the fingertip bodies; the left gripper stays open
        self._hand_side = torch.tensor([0 if n.startswith("right") else 1 for n in cfg.hand_joint_names], device=self.device)
        self._tip_ids = torch.tensor(
            [self._robot.find_bodies(list(tips), preserve_order=True)[0] for tips in cfg.hand_tip_body_names],
            device=self.device,
        )
        # the left wrist camera (calibrated pose in the left wrist yaw link; axes as OpenCV: x right, y down, z forward)
        import json

        from .env_cfg import CAMERA_CALIBRATION, DEX1_BASE_IN_WRIST_YAW

        with open(CAMERA_CALIBRATION) as f:
            calibration = json.load(f)["cameras"]
        pose = torch.tensor(calibration["left_wrist"]["pose"]["T_parent_cam"], device=self.device)
        self._left_cam_offset = pose[:3, 3] + torch.tensor(DEX1_BASE_IN_WRIST_YAW, device=self.device)
        self._left_cam_rot = pose[:3, :3]
        self._left_wrist_id = self._robot.find_bodies("left_wrist_yaw_link")[0][0]
        self._walk_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        # table top centers (z = table height) and yaws, env-local
        self._cube_table_top = torch.zeros(self.num_envs, 3, device=self.device)
        self._cube_table_yaw = torch.zeros(self.num_envs, device=self.device)
        self._tray_table_top = torch.zeros(self.num_envs, 3, device=self.device)
        self._tray_table_yaw = torch.zeros(self.num_envs, device=self.device)
        self._start_yaw = torch.zeros(self.num_envs, device=self.device)
        self._fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pick_time = torch.zeros(self.num_envs, device=self.device)
        # where each episode started: base position (env frame) and arm joint targets (right arm, then left)
        self._start_base_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._arm_start_targets = torch.zeros(self.num_envs, len(self._arm_ids), device=self.device)
        # calibrated cameras: the lens warp of each (render -> real lens)
        self._lens = {}
        if cfg.cameras:
            import json

            from .cameras.camera_lens import LensWarp
            from .env_cfg import CAMERA_CALIBRATION

            with open(CAMERA_CALIBRATION) as f:
                calibration = json.load(f)["cameras"]
            for name in cfg.cameras:
                c = calibration[name]
                self._lens[name] = LensWarp({**c["lens"], "size": c["image_size"]})

    def camera_images(self) -> dict[str, torch.Tensor]:
        """The calibrated cameras' current images as the real cameras see them (real lens: focal lengths, principal
        point and distortion): {"head" | "left_wrist" | "right_wrist": uint8 [num_envs, 480, 640, 3]}. The head image is
        the left view of the head's stereo camera."""
        return {
            name: warp(self.scene[f"{name}_camera"].data.output["rgb"][..., :3]) for name, warp in self._lens.items()
        }

    def review_camera_view(self, env_id: int) -> tuple[list[float], list[float]]:
        """Camera for videos and the pipeline's failure replay (world positions eye, target): frames the robot and what it
        works towards (the cube until it has been lifted off the cube table, then the tray), from the robot's right side
        so that the right hand is in view, closer as the robot gets closer."""
        origin = self.scene.env_origins[env_id]
        base = self._robot.data.root_pos_w.torch[env_id] - origin
        cube = self._item.data.root_pos_w.torch[env_id] - origin
        lifted = (cube[2] > self._item_rest_z[env_id] + 0.05) | (
            torch.norm(cube[:2] - self._pickup_xy[env_id]) > 0.15)
        focus = torch.where(lifted, self._box_center_local[env_id], cube)
        to_focus = focus[:2] - base[:2]
        span = float(torch.norm(to_focus).clamp_min(0.4))
        u = to_focus / span
        right = torch.stack((u[1], -u[0]))
        mid = 0.7 * base[:2] + 0.3 * focus[:2]  # nearer the robot, so that it stays well inside the frame
        eye_xy = mid + right * (1.5 + 0.6 * span) - u * (0.8 + 0.3 * span)
        eye = torch.cat((eye_xy, torch.tensor([1.5 + 0.3 * span], device=self.device))) + origin
        target = torch.cat((mid, torch.tensor([0.7], device=self.device))) + origin
        return eye.tolist(), target.tolist()

    def render(self, recompute: bool = False):
        # videos (render_mode rgb_array): the camera follows the robot of the viewed environment
        if self.render_mode == "rgb_array" and self.cfg.review_camera_follow:
            try:
                from isaacsim.core.rendering_manager import ViewportManager

                eye, target = self.review_camera_view(int(self.cfg.viewer.env_index))
                ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye, target=target)
            except ImportError:  # no Kit renderer (e.g. a Newton visualizer)
                pass
        return super().render(recompute)

    # actions

    def _walker_arm_targets(self) -> torch.Tensor:
        """Targets of the 14 arm joints in the walker's order."""
        targets = torch.empty_like(self._arm_targets)
        targets[:, self._arm_in_walker] = self._arm_targets
        return targets

    def _pre_physics_step(self, actions: torch.Tensor):
        cfg = self.cfg
        self._prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        forward = self.actions[:, 0]
        self._walk_cmd[:, 0] = torch.where(forward >= 0, forward, 0.5 * forward) * cfg.max_forward_speed
        self._walk_cmd[:, 1] = self.actions[:, 1] * cfg.max_side_speed
        self._walk_cmd[:, 2] = self.actions[:, 2] * cfg.max_turn_rate
        arm_actions = torch.cat((self.actions[:, 3:10], self.actions[:, 11:18]), dim=-1)  # right, left (_arm_ids order)
        delta = arm_actions * cfg.arm_action_scale * self.step_dt
        self._arm_targets[:] = torch.clamp(self._arm_targets + delta, self._arm_lower, self._arm_upper)
        self._hand_cmd[:] = (self.actions[:, 10:11] + 1.0) / 2.0
        data = self._robot.data
        self._walker.step(
            self._walk_cmd,
            data.root_ang_vel_b.torch,
            data.projected_gravity_b.torch,
            data.joint_pos.torch,
            data.joint_vel.torch,
            self._walker_arm_targets(),
        )

    def _apply_action(self):
        default = self._robot.data.default_joint_pos.torch
        hand_cmd = torch.where(self._hand_side == 0, self._hand_cmd, 0.0)  # the left gripper stays open
        hand_targets = self._hand_open + hand_cmd * (self._hand_closed - self._hand_open)
        self._robot.set_joint_position_target_index(target=self._walker.leg_targets, joint_ids=self._leg_ids)
        self._robot.set_joint_position_target_index(target=default[:, self._waist_ids], joint_ids=self._waist_ids)
        self._robot.set_joint_position_target_index(target=self._arm_targets, joint_ids=self._arm_ids)
        self._robot.set_joint_position_target_index(target=hand_targets, joint_ids=self._hand_ids)

    # observations

    def _compute_task_values(self):
        data = self._robot.data
        origins = self.scene.env_origins
        n = self.num_envs
        self._item_pos = self._item.data.root_pos_w.torch - origins
        self._item_quat_w = self._item.data.root_quat_w.torch
        self._item_to_target = self._carry_target_local - self._item_pos
        # both grippers (0 right, 1 left): fingertips, grasp points (midpoint between the fingertips); the task uses the
        # right one: its grasp point, jaw position (mean of its two jaw joints), fingertips and their mean velocity
        tips = data.body_pos_w.torch[:, self._tip_ids.flatten()].view(n, 2, 2, 3) - origins[:, None, None]
        self._palms = tips.mean(dim=2)
        self._palm_pos = self._palms[:, 0]
        self._palm_to_item = self._item_pos - self._palm_pos
        self._right_tips = tips[:, 0]
        self._right_tip_vel = data.body_lin_vel_w.torch[:, self._tip_ids[0]].mean(dim=1)
        self._right_jaw_pos = data.joint_pos.torch[:, self._hand_ids[:2]].mean(dim=-1)
        # the left wrist camera: position (env-local) and axes (world) [N, 3, 3]: [:, 0] right, [:, 1] down, [:, 2] the
        # optical axis
        wrist_pos = data.body_pos_w.torch[:, self._left_wrist_id]
        wrist_quat = data.body_quat_w.torch[:, self._left_wrist_id]
        self._left_cam_pos = wrist_pos + quat_apply(wrist_quat, self._left_cam_offset.expand(n, 3)) - origins
        self._left_cam_axes = torch.stack(
            [quat_apply(wrist_quat, self._left_cam_rot[:, k].expand(n, 3)) for k in range(3)], dim=1
        )
        self._base_pos = data.root_pos_w.torch - origins
        self._base_yaw = _yaw_of(data.root_quat_w.torch)

    def _get_observations(self) -> dict:
        self._compute_task_values()
        data = self._robot.data
        root_pos = data.root_pos_w.torch
        root_quat = data.root_quat_w.torch
        origins = self.scene.env_origins

        def in_base(points_local):
            return subtract_frame_transforms(root_pos, root_quat, points_local + origins)[0]

        def yaw_obs(yaw):
            rel = yaw - self._base_yaw
            return torch.stack((torch.sin(rel), torch.cos(rel)), dim=-1)

        item_b, item_quat_b = subtract_frame_transforms(root_pos, root_quat, self._item_pos + origins, self._item_quat_w)
        joint_pos = data.joint_pos.torch
        obs = torch.cat(
            (
                data.root_lin_vel_b.torch,  # 3
                data.root_ang_vel_b.torch,  # 3
                data.projected_gravity_b.torch,  # 3
                joint_pos[:, self._arm_ids] - data.default_joint_pos.torch[:, self._arm_ids],  # 14 (right, left)
                data.joint_vel.torch[:, self._arm_ids] * 0.1,  # 14
                joint_pos[:, self._hand_ids[:2]],  # 2 right jaws
                in_base(self._palm_pos),  # 3 right grasp point
                in_base(self._left_cam_pos),  # 3 left wrist camera position
                quat_apply_inverse(root_quat, self._left_cam_axes[:, 2]),  # 3 its optical axis
                item_b,  # 3
                item_quat_b,  # 4
                quat_apply_inverse(root_quat, self._palm_to_item),  # 3
                quat_apply_inverse(root_quat, self._item_to_target),  # 3
                in_base(self._box_center_local),  # 3
                yaw_obs(self._box_yaw),  # 2
                in_base(self._cube_table_top),  # 3
                yaw_obs(self._cube_table_yaw),  # 2
                in_base(self._tray_table_top),  # 3
                yaw_obs(self._tray_table_yaw),  # 2
                self._item_size.unsqueeze(-1),  # 1
                self.actions,  # 18
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _place_at_cube_table(self, env_ids: torch.Tensor):
        """Picking variant: the robot stands outside the cube's table edge, facing it (+- stance_yaw_noise), with the
        cube cube_ahead ahead of and cube_right to the right of its pelvis (drawn uniformly)."""
        cfg = self.cfg
        dev = self.device
        n = len(env_ids)
        top = self._cube_table_top[env_ids]
        yaw = self._cube_table_yaw[env_ids]
        cube = self._pickup_xy[env_ids]
        local = _rotate_xy(cube - top[:, :2], -yaw)  # the cube in the table frame
        half = torch.tensor(TABLE_TOP, device=dev) / 2
        # the nearest free edge: a long side (x = +-0.3) or the far end (y = +0.5)
        side = (half[0] - local[:, 0].abs()) < (half[1] - local[:, 1])
        normal = torch.where(side.unsqueeze(-1), torch.stack((torch.sign(local[:, 0]), torch.zeros(n, device=dev)), -1),
                             torch.tensor([0.0, 1.0], device=dev).expand(n, 2))
        forward = -_rotate_xy(normal, yaw)  # facing the table, env frame
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading = heading + sample_uniform(-cfg.stance_yaw_noise, cfg.stance_yaw_noise, (n,), dev)
        ahead = sample_uniform(cfg.cube_ahead[0], cfg.cube_ahead[1], (n,), dev)
        right = sample_uniform(cfg.cube_right[0], cfg.cube_right[1], (n,), dev)
        f = torch.stack((torch.cos(heading), torch.sin(heading)), -1)
        left = torch.stack((-f[:, 1], f[:, 0]), -1)
        pelvis = cube - f * ahead.unsqueeze(-1) + left * right.unsqueeze(-1)
        root = torch.tensor(cfg.scene.robot.init_state.pos, device=dev).repeat(n, 1)
        root[:, :2] = pelvis
        self._start_base_xy[env_ids] = pelvis
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        self._robot.write_root_pose_to_sim_index(
            root_pose=torch.cat((root + self.scene.env_origins[env_ids], quat_from_angle_axis(heading, z_axis)), -1),
            env_ids=env_ids)
        self._start_yaw[env_ids] = heading

    def _update_task_progress(self):
        super()._update_task_progress()
        if self.cfg.success_on_lift:
            # picking variant: after the full task's valid lift (lifted_once: above 0.12 m and clear of the tray rim),
            # the cube held in the right gripper (commanded closed, jaws stopped at the cube's width, grasp point within
            # 0.04 m) at least pick_lift_height above its rest height and clear of the rim, continuously for
            # pick_hold_time; the full task's carry starts from there
            on_cube = 0.023 - self._item_size / 2
            jaw = self._right_jaw_pos
            held = (self._lifted_once & self._clear_rim
                    & (self._hand_cmd[:, 0] > 0.5) & (jaw > on_cube - 0.003) & (jaw < on_cube + 0.006)
                    & (torch.norm(self._palm_to_item, dim=-1) < 0.04)
                    & (self._item_pos[:, 2] - self._item_rest_z > self.cfg.pick_lift_height))
            self._pick_time = torch.where(held, self._pick_time + self.step_dt, 0.0)
            self._succeeded = self._pick_time >= self.cfg.pick_hold_time

    # rewards (the pipeline replaces them)

    def _get_rewards(self) -> torch.Tensor:
        return super()._get_rewards() + self.cfg.rew_fall * self._fell.float()

    # dones

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_task_values()
        self._update_task_progress()
        # the cube fell: below the lower table top (it can only rest on one of the tables or in the tray)
        lower_top = torch.minimum(self._cube_table_top[:, 2], self._tray_table_top[:, 2])
        item_fell = self._item_pos[:, 2] < lower_top - self.cfg.fall_height
        gravity = self._robot.data.projected_gravity_b.torch
        tilt = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))
        self._fell = (self._base_pos[:, 2] < self.cfg.fall_base_height) | (tilt > self.cfg.fall_tilt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = item_fell | self._fell
        if self.cfg.terminate_on_success:
            terminated |= self._succeeded
        return terminated, time_out

    # reset

    def _sample_layout(self, n: int):
        """Table and robot layout in the line frame, and a random turn of the whole layout.

        Line frame: the cube table center at the origin, the tables end to end along +x (joined by their short sides:
        each table's long side along the line), the tray table at +x. The robot starts at a distance from the cube
        table drawn from [robot_distance_min, robot_distance_max], anywhere on the C-shaped band around the cube table
        that opens towards the tray table (in front of it, behind it or beyond its far end, never past its end towards
        the tray table), facing the cube table center.

        Returns:
            Cube table center and yaw, tray table center and yaw, robot position and heading (line frame), the turn.
        """
        cfg = self.cfg
        dev = self.device
        # table yaw 90 deg in the line frame: the table's 1.0 m width runs along the line, its 0.6 m depth across it
        along, across = TABLE_TOP[1] / 2, TABLE_TOP[0] / 2
        table_yaw = torch.full((n,), torch.pi / 2, device=dev)
        gap = sample_uniform(cfg.table_gap_min, cfg.table_gap_max, (n,), dev)
        cube_center = torch.zeros(n, 2, device=dev)
        tray_center = torch.stack((2 * along + gap, torch.zeros(n, device=dev)), dim=-1)

        # robot start: a point drawn uniformly along the C (the curve at this distance from the cube table, without the
        # part beyond the end towards the tray table): the long side behind the table, the arc around its far corner
        # behind, the far end, the arc around its far corner in front, the long side in front
        d = sample_uniform(cfg.robot_distance_min, max(cfg.robot_distance_max, cfg.robot_distance_min), (n,), dev)
        arc = torch.pi / 2 * d
        lengths = torch.stack((torch.full_like(d, 2 * along), arc, torch.full_like(d, 2 * across), arc,
                               torch.full_like(d, 2 * along)), dim=-1)
        ends = lengths.cumsum(dim=-1)
        s = torch.rand(n, device=dev) * ends[:, -1]
        piece = (s.unsqueeze(-1) >= ends).sum(dim=-1)  # which of the five pieces s falls on
        u = s - torch.cat((torch.zeros(n, 1, device=dev), ends[:, :-1]), dim=-1).gather(1, piece.unsqueeze(-1)).squeeze(-1)
        phi_back = torch.pi / 2 + u / d  # angle on the arc around the far corner behind (center (-along, across))
        phi_front = torch.pi + u / d  # on the arc around the far corner in front (center (-along, -across))
        x = torch.stack((along - u, -along + d * torch.cos(phi_back), torch.full_like(d, -along) - d,
                         -along + d * torch.cos(phi_front), -along + u), dim=-1)
        y = torch.stack((across + d, across + d * torch.sin(phi_back), across - u,
                         -across + d * torch.sin(phi_front), -across - d), dim=-1)
        robot = torch.stack((x.gather(1, piece.unsqueeze(-1)), y.gather(1, piece.unsqueeze(-1))), dim=-1).squeeze(1)
        heading = torch.atan2(-robot[:, 1], -robot[:, 0])  # facing the cube table center
        heading = heading + sample_uniform(-cfg.heading_noise, cfg.heading_noise, (n,), dev)
        turn = sample_uniform(-torch.pi, torch.pi, (n,), dev)
        return cube_center, table_yaw, tray_center, table_yaw.clone(), robot, heading, turn

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        log = self.extras.setdefault("log", {})
        log["Metrics/fall_rate"] = _log_value(self._fell[env_ids].float().mean())
        super()._reset_idx(env_ids)  # task state, robot joints (right arm with noise)
        cfg = self.cfg
        dev = self.device
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)

        # layout: the robot starts at the environment origin, the line frame is moved and turned around it
        cube_c, cube_yaw, tray_c, tray_yaw, robot, heading, turn = self._sample_layout(n)

        def to_local(xy):
            return _rotate_xy(xy - robot, turn)

        cube_h = sample_uniform(cfg.cube_table_height[0], cfg.cube_table_height[1], (n,), dev)
        tray_h = sample_uniform(cfg.tray_table_height[0], cfg.tray_table_height[1], (n,), dev)
        cube_yaw = _wrap(cube_yaw + turn)
        tray_yaw = _wrap(tray_yaw + turn)
        self._cube_table_top[env_ids] = torch.cat((to_local(cube_c), cube_h.unsqueeze(-1)), -1)
        self._tray_table_top[env_ids] = torch.cat((to_local(tray_c), tray_h.unsqueeze(-1)), -1)
        self._cube_table_yaw[env_ids] = cube_yaw
        self._tray_table_yaw[env_ids] = tray_yaw
        for name, top, yaw in (
            ("cube_table", self._cube_table_top[env_ids], cube_yaw),
            ("tray_table", self._tray_table_top[env_ids], tray_yaw),
        ):
            pos = top.clone()
            pos[:, 2] -= TABLE_BLOCK_H / 2  # block center: the top is at the table height
            self.scene[name].write_root_pose_to_sim_index(
                root_pose=torch.cat((pos + origins, quat_from_angle_axis(yaw, z_axis)), -1), env_ids=env_ids
            )

        # the base task's table height is the tray table's (rim, placement); the cube rests on the cube table
        self._table_h[env_ids] = tray_h
        self._item_rest_z[env_ids] = cube_h + self._item_size[env_ids] / 2
        half_top = torch.tensor(TABLE_TOP, device=dev) / 2 - cfg.table_margin

        # tray: near a free edge of the tray table (the long sides and the far end y = -0.5 of the table frame; y = +0.5
        # faces the cube table), its center its outer radius plus a small gap in from that edge (any yaw fits), so
        # that the robot standing at the edge can reach into it
        tray_radius = torch.norm(self._tray_inner[env_ids] / 2 + self._tray_t, dim=-1, keepdim=True)
        room = (half_top - tray_radius).clamp_min(0.0)
        depth = tray_radius.squeeze(-1) + sample_uniform(cfg.tray_edge_gap[0], cfg.tray_edge_gap[1], (n,), dev)
        offset = _near_edge(depth, room, far_end=-1.0)
        center = torch.zeros(n, 3, device=dev)
        center[:, :2] = self._tray_table_top[env_ids, :2] + _rotate_xy(offset, tray_yaw)
        center[:, 2] = tray_h
        box_yaw = sample_uniform(-torch.pi, torch.pi, (n,), dev)
        self._box_center_local[env_ids] = center
        self._box_yaw[env_ids] = box_yaw
        self._carry_target_local[env_ids] = center + torch.tensor([0.0, 0.0, cfg.carry_height], device=dev)
        tray_quat = quat_from_angle_axis(box_yaw, z_axis)
        for part, part_offset in self._tray_parts:
            part_pos = center.clone()
            part_pos[:, :2] += _rotate_xy(part_offset[:2].repeat(n, 1), box_yaw)
            part_pos[:, 2] += part_offset[2]
            part.write_root_pose_to_sim_index(root_pose=torch.cat((part_pos + origins, tray_quat), -1), env_ids=env_ids)

        # cube: near a free edge of the cube table (the long sides x = +-0.3 and the far end y = +0.5 of the table
        # frame; y = -0.5 faces the tray table), the edge drawn in proportion to its length, any yaw
        depth = sample_uniform(cfg.cube_edge_depth[0], cfg.cube_edge_depth[1], (n,), dev)
        offset = _near_edge(depth, half_top.repeat(n, 1), far_end=1.0)
        pos = torch.zeros(n, 3, device=dev)
        pos[:, :2] = self._cube_table_top[env_ids, :2] + _rotate_xy(offset, cube_yaw)
        pos[:, 2] = self._item_rest_z[env_ids]
        self._pickup_xy[env_ids] = pos[:, :2]
        item_yaw = sample_uniform(-torch.pi, torch.pi, (n,), dev)
        self._item.write_root_pose_to_sim_index(
            root_pose=torch.cat((pos + origins, quat_from_angle_axis(item_yaw, z_axis)), -1), env_ids=env_ids
        )
        self._item.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)

        # robot: standing at the environment origin, facing the cube table
        start_yaw = _wrap(heading + turn)
        start = torch.tensor(cfg.scene.robot.init_state.pos, device=dev).repeat(n, 1)
        self._start_yaw[env_ids] = start_yaw
        self._robot.write_root_pose_to_sim_index(
            root_pose=torch.cat((start + origins, quat_from_angle_axis(start_yaw, z_axis)), -1), env_ids=env_ids
        )
        self._robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        self._start_base_xy[env_ids] = start[:, :2]
        if cfg.start_at_cube_table:
            self._place_at_cube_table(env_ids)
        self._arm_start_targets[env_ids] = self._arm_targets[env_ids]
        self._pick_time[env_ids] = 0.0
        self._walk_cmd[env_ids] = 0.0
        self._fell[env_ids] = False
        self._walker.reset(env_ids, self._walker_arm_targets()[env_ids])
