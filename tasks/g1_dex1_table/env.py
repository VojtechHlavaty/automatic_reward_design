from __future__ import annotations

from collections.abc import Sequence

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_from_angle_axis,
    sample_uniform,
    subtract_frame_transforms,
)

# RD_TENSOR_LOGS=1: the logged values stay GPU tensors (RSL-RL averages them once per iteration) instead of
# synchronizing with the GPU for each value in every step; the logged numbers are the same
_TENSOR_LOGS = os.environ.get("RD_TENSOR_LOGS") == "1"


def _log_value(t: torch.Tensor):
    return t if _TENSOR_LOGS else t.item()


from .env_cfg import (
    BOX_IN,
    BOX_POS,
    BOX_T,
    BOX_WALL_H,
    ITEM_POS,
    TABLE_H,
    TABLE_POS,
    G1Dex1TablePickPlaceEnvCfg,
)


class G1Dex1TablePickPlaceEnv(DirectRLEnv):
    """G1 (fixed base) with Dex1-1 grippers picks the item from the table with its right gripper and places it into the box.

    Actions: joint target deltas of the controlled joints (waist yaw and the 7 right-arm joints) and 1 gripper
    command (-1 open, +1 closed).
    """

    cfg: G1Dex1TablePickPlaceEnvCfg

    def __init__(self, cfg: G1Dex1TablePickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint / body indices
        self._arm_ids, _ = self._robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self._hand_ids, hand_names = self._robot.find_joints(self.cfg.hand_joint_names, preserve_order=True)
        self._palm_ids = self._robot.find_bodies(self.cfg.palm_body_names, preserve_order=True)[0]

        limits = self._robot.data.soft_joint_pos_limits.torch[0]
        self._arm_lower = limits[self._arm_ids, 0].clone()
        self._arm_upper = limits[self._arm_ids, 1].clone()

        # hand open / closed joint targets
        self._hand_open = torch.tensor(
            [next(v for k, v in self.cfg.hand_open_pos.items() if k in n) for n in hand_names],
            device=self.device,
        )
        self._hand_closed = torch.tensor(
            [next(v for k, v in self.cfg.hand_closed_pos.items() if k in n) for n in hand_names],
            device=self.device,
        )

        # buffers
        self._arm_targets = self._robot.data.default_joint_pos.torch[:, self._arm_ids].clone()
        self._hand_cmd = torch.zeros(self.num_envs, 1, device=self.device)  # 0 open .. 1 closed
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self.actions)
        self._succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasped = torch.zeros_like(self._succeeded)
        self._lifted_once = torch.zeros_like(self._succeeded)
        self._above_once = torch.zeros_like(self._succeeded)
        self._released_above = torch.zeros_like(self._succeeded)
        self._release_rewarded = torch.zeros_like(self._succeeded)
        self._release_armed = torch.zeros_like(self._succeeded)
        self._release_seen = torch.zeros_like(self._succeeded)  # release condition met at a physics substep
        self._grasp_time = torch.zeros(self.num_envs, device=self.device)
        self._settle_time = torch.zeros_like(self._grasp_time)
        # with success_requires_return: the placement (the original success) is latched and success needs the arm back
        # at its start pose afterwards
        self._placed = torch.zeros_like(self._succeeded)
        self._return_time = torch.zeros_like(self._grasp_time)
        self._best_lift = torch.zeros_like(self._grasp_time)
        self._best_transport = torch.zeros_like(self._grasp_time)
        self._best_transport_fine = torch.zeros_like(self._grasp_time)
        self._stage_bonus = torch.zeros_like(self._grasp_time)

        # task layout in the environment-local frame, per environment: the cube size (fixed per environment), the
        # tray center and yaw (new at every reset when randomized)
        self._item_size = self._item_sizes_per_env()
        self._item_radius = 3**0.5 * self._item_size / 2  # circumscribed sphere covers the cube for any orientation
        # table top height (new at every reset when table_height_range > 0) and tray shape (fixed per environment):
        # inner size along the tray's x and y, wall height, wall thickness
        self._table_h = torch.full((self.num_envs,), TABLE_H, device=self.device)
        self._tray_inner, self._tray_wall_h = self._tray_shapes_per_env()
        self._tray_t = BOX_T
        self._item_rest_z = self._table_h + self._item_size / 2
        self._box_center_local = torch.tensor(BOX_POS, device=self.device).repeat(self.num_envs, 1)
        self._box_yaw = torch.zeros(self.num_envs, device=self.device)
        # nominal pickup point (x, y) of each environment: the lift must start near it (layouts that move set it per reset)
        self._pickup_xy = torch.tensor(ITEM_POS[:2], device=self.device).repeat(self.num_envs, 1)
        self._carry_target_local = self._box_center_local + torch.tensor(
            [0.0, 0.0, self.cfg.carry_height], device=self.device
        )
        # tray parts and their offsets from the tray center (bottom center) in the tray frame (one asset when the
        # tray varies: its root is the bottom center)
        if self.cfg.tray_variants > 0:
            self._tray_parts = [(self.scene["tray"], torch.zeros(3, device=self.device))]
        else:
            self._tray_parts = [
                (self.scene[name], torch.tensor(offset, device=self.device))
                for name, offset in (
                    ("box_floor", (0.0, 0.0, BOX_T / 2)),
                    ("box_wall_px", (BOX_IN / 2, 0.0, BOX_WALL_H / 2)),
                    ("box_wall_nx", (-BOX_IN / 2, 0.0, BOX_WALL_H / 2)),
                    ("box_wall_py", (0.0, BOX_IN / 2, BOX_WALL_H / 2)),
                    ("box_wall_ny", (0.0, -BOX_IN / 2, BOX_WALL_H / 2)),
                )
            ]

    def _item_sizes_per_env(self) -> torch.Tensor:
        """Cube edge length of each environment: with several sizes, the scene's clone plan says which cube each
        environment got."""
        sizes = torch.tensor(self.cfg.item_sizes, device=self.device, dtype=torch.float)
        if len(sizes) == 1:
            return sizes.repeat(self.num_envs)
        plan = self.scene.clone_plan
        rows = [i for i, source in enumerate(plan.sources) if source.endswith("/Item")]
        if len(rows) != len(sizes):
            raise RuntimeError(f"expected {len(sizes)} cube variants in the clone plan, found {len(rows)}")
        mask = torch.as_tensor(plan.clone_mask, device=self.device)[rows]  # (variants, envs)
        if not bool((mask.sum(dim=0) == 1).all()):
            raise RuntimeError("each environment must get exactly one cube variant")
        return sizes[mask.to(torch.int).argmax(dim=0)]

    def _tray_shapes_per_env(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner size (x, y) [m] and wall height [m] of each environment's tray: the fixed square tray, or the varied
        tray each environment got (read from the scene's clone plan, like the cube size)."""
        if self.cfg.tray_variants <= 0:
            inner = torch.full((self.num_envs, 2), BOX_IN - BOX_T, device=self.device)
            return inner, torch.full((self.num_envs,), BOX_WALL_H, device=self.device)
        from .tray_spawner import tray_variants

        shapes = torch.tensor(
            tray_variants(self.cfg.tray_variants, self.cfg.tray_inner_range, self.cfg.tray_wall_height_range,
                          self.cfg.tray_variant_seed),
            device=self.device,
        )
        plan = self.scene.clone_plan
        rows = [i for i, source in enumerate(plan.sources) if source.endswith("/tray")]
        if len(rows) != len(shapes):
            raise RuntimeError(f"expected {len(shapes)} tray variants in the clone plan, found {len(rows)}")
        mask = torch.as_tensor(plan.clone_mask, device=self.device)[rows]
        if not bool((mask.sum(dim=0) == 1).all()):
            raise RuntimeError("each environment must get exactly one tray variant")
        chosen = shapes[mask.to(torch.int).argmax(dim=0)]
        return chosen[:, :2].contiguous(), chosen[:, 2].contiguous()

    def _tray_xy(self, pos: torch.Tensor) -> torch.Tensor:
        """x and y of environment-local positions in the tray frame (origin at the tray center, axes along the walls)."""
        rel = pos[:, :2] - self._box_center_local[:, :2]
        c, s = torch.cos(self._box_yaw), torch.sin(self._box_yaw)
        return torch.stack((c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]), dim=-1)

    def _setup_scene(self):
        # robot, table, box and item are created from the scene config; just keep handles
        self._robot: Articulation = self.scene["robot"]
        self._item: RigidObject = self.scene["item"]
        # ground and light are global, spawned directly rather than as scene entries
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=self.cfg.light_intensity)
        light_cfg.func("/World/light", light_cfg)

    # pre-physics step

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # integrate arm joint targets
        num_joints = len(self._arm_ids)
        delta = self.actions[:, :num_joints] * self.cfg.arm_action_scale * self.step_dt
        self._arm_targets[:] = torch.clamp(self._arm_targets + delta, self._arm_lower, self._arm_upper)
        # hand command in [0, 1]
        self._hand_cmd[:] = (self.actions[:, num_joints : num_joints + 1] + 1.0) / 2.0

    def _substep_release_check(self):
        """The release condition (armed, gripper commanded open, cube over the opening and not yet supported) checked at
        a physics substep. With long control steps (slow robot speed) a released cube can land in the tray within one
        step, so the once-per-step check in _update_task_progress would never see it falling."""
        cfg = self.cfg
        item_pos = self._item.data.root_pos_w.torch - self.scene.env_origins
        radius = self._item_radius
        rel_xy = self._tray_xy(item_pos).abs()
        half_in = self._tray_inner / 2
        in_outer = (rel_xy < half_in + self._tray_t + radius.unsqueeze(-1)).all(dim=-1)
        in_inner = (rel_xy < half_in - radius.unsqueeze(-1)).all(dim=-1)
        below_rim = item_pos[:, 2] - radius < self._table_h + self._tray_wall_h
        wall = in_outer & ~in_inner & below_rim
        unsupported = (item_pos[:, 2] - self._item_rest_z > cfg.grasp_min_lift) & ~wall
        if cfg.placement_anywhere_in_tray:
            opening_half = half_in - (self._item_size / 2).unsqueeze(-1)
        else:
            opening_half = half_in - (radius + cfg.wall_clearance).unsqueeze(-1)
        over_opening = (rel_xy < opening_half).all(dim=-1)
        self._release_seen |= self._release_armed & over_opening & unsupported & (self._hand_cmd[:, 0] < 0.3)

    def _apply_action(self):
        if self.cfg.substep_release_check:
            self._substep_release_check()
        hand_targets = self._hand_open + self._hand_cmd * (self._hand_closed - self._hand_open)
        self._robot.set_joint_position_target_index(target=self._arm_targets, joint_ids=self._arm_ids)
        self._robot.set_joint_position_target_index(target=hand_targets, joint_ids=self._hand_ids)

    # post-physics step

    def _compute_task_values(self):
        origins = self.scene.env_origins
        # grasp point: the midpoint between the fingertips
        self._palm_pos = self._robot.data.body_pos_w.torch[:, self._palm_ids].mean(dim=1) - origins
        self._item_pos = self._item.data.root_pos_w.torch - origins
        self._item_quat_w = self._item.data.root_quat_w.torch
        self._palm_to_item = self._item_pos - self._palm_pos
        self._item_to_target = self._carry_target_local - self._item_pos

    def _get_observations(self) -> dict:
        self._compute_task_values()
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch
        palm_b, _ = subtract_frame_transforms(root_pos, root_quat, self._palm_pos + self.scene.env_origins)
        item_b, item_quat_b = subtract_frame_transforms(
            root_pos,
            root_quat,
            self._item_pos + self.scene.env_origins,
            self._item_quat_w,
        )
        joint_pos = self._robot.data.joint_pos.torch
        layout = ()
        if self.cfg.observe_layout:
            tray_b, _ = subtract_frame_transforms(root_pos, root_quat, self._box_center_local + self.scene.env_origins)
            layout = (
                tray_b,  # 3
                torch.stack((torch.sin(self._box_yaw), torch.cos(self._box_yaw)), dim=-1),  # 2
                self._item_size.unsqueeze(-1),  # 1
            )
        obs = torch.cat(
            (
                joint_pos[:, self._arm_ids] - self._robot.data.default_joint_pos.torch[:, self._arm_ids],  # 8
                self._robot.data.joint_vel.torch[:, self._arm_ids] * 0.1,  # 8
                joint_pos[:, self._hand_ids],  # 2 (gripper jaws)
                palm_b,  # 3
                item_b,  # 3
                item_quat_b,  # 4
                self._palm_to_item,  # 3
                self._item_to_target,  # 3
                self.actions,  # 9
                *layout,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        reach_dist = torch.norm(self._palm_to_item, dim=-1)
        reach = (1.0 - torch.tanh(reach_dist / 0.1)) * (~self._released_above)
        # Reward improvements only: holding a pose or repeating a lift cannot farm progress.
        lift = ((self._item_pos[:, 2] - self._item_rest_z) / cfg.lift_height).clamp(0.0, 1.0)
        lift *= self._grasped & ~self._wall_contact
        lift_progress = (lift - self._best_lift).clamp_min(0.0)
        self._best_lift = torch.maximum(self._best_lift, lift)
        target_dist = torch.norm(self._item_to_target, dim=-1)
        carrying = self._lifted_once & self._grasped & self._clear_rim & ~self._wall_contact
        transport = carrying * (1.0 - torch.tanh(target_dist / 0.3))
        transport_fine = carrying * (1.0 - torch.tanh(target_dist / 0.05))
        transport_progress = (transport - self._best_transport).clamp_min(0.0)
        fine_progress = (transport_fine - self._best_transport_fine).clamp_min(0.0)
        self._best_transport = torch.maximum(self._best_transport, transport)
        self._best_transport_fine = torch.maximum(self._best_transport_fine, transport_fine)
        # Horizontal sliding before a valid lift is undesirable even without wall contact.
        push = torch.norm(self._item.data.root_lin_vel_w.torch[:, :2], dim=-1).clamp(max=1.0)
        push *= ~self._lifted_once & ~self._clear_rim

        # regularization
        action_rate = torch.sum(torch.square(self.actions - self._prev_actions), dim=-1)
        joint_vel = torch.sum(torch.square(self._robot.data.joint_vel.torch[:, self._arm_ids]), dim=-1)

        terms = {
            "reach": cfg.rew_reach * reach,
            "grasp": cfg.rew_grasp * (self._grasped & ~self._released_above).float(),
            "wall_contact": cfg.rew_wall_contact * self._wall_contact.float(),
            "push": cfg.rew_push * push,
            "action_rate": cfg.rew_action_rate * action_rate,
            "joint_vel": cfg.rew_joint_vel * joint_vel,
        }
        log = self.extras.setdefault("log", {})
        for name, value in terms.items():
            log[f"Reward/{name}"] = _log_value(value.mean())
        progress = (
            cfg.rew_lift * lift_progress
            + cfg.rew_transport * transport_progress
            + cfg.rew_transport_fine * fine_progress
            + self._stage_bonus
        )
        log["Reward/progress"] = _log_value(progress.mean())
        return torch.stack(list(terms.values()), dim=0).sum(dim=0) * self.step_dt + progress

    def _update_task_progress(self):
        # Called exactly once per control step, before rewards and resets.
        cfg = self.cfg
        # No contact sensors (not available on the real robot): grasp, support and wall contact
        # are estimated from the item and palm positions.
        radius = self._item_radius
        palm_dist = torch.norm(self._palm_to_item, dim=-1)
        near_hand = palm_dist < cfg.grasp_distance
        hand_closed = self._hand_cmd[:, 0] > cfg.grasp_closed_cmd
        unsupported = self._item_pos[:, 2] - self._item_rest_z > cfg.grasp_min_lift
        # item overlaps the band occupied by the tray walls, below the rim
        rel_xy = self._tray_xy(self._item_pos).abs()
        half_in = self._tray_inner / 2
        in_outer = (rel_xy < half_in + self._tray_t + radius.unsqueeze(-1)).all(dim=-1)
        in_inner = (rel_xy < half_in - radius.unsqueeze(-1)).all(dim=-1)
        rim_z = self._table_h + self._tray_wall_h
        below_rim = self._item_pos[:, 2] - radius < rim_z
        self._wall_contact = in_outer & ~in_inner & below_rim
        unsupported &= ~self._wall_contact
        self._grasped = near_hand & hand_closed & unsupported
        self._clear_rim = self._item_pos[:, 2] - radius > rim_z + cfg.wall_clearance
        held_clear = self._grasped & self._clear_rim
        # Establish the lift near the pickup location, before using the tray as support.
        near_pickup = torch.norm(self._item_pos[:, :2] - self._pickup_xy, dim=-1) < cfg.pickup_radius
        self._grasp_time = torch.where(held_clear & near_pickup, self._grasp_time + self.step_dt, 0.0)
        lifted = (self._grasp_time >= cfg.grasp_hold_time) & (
            self._item_pos[:, 2] - self._item_rest_z >= cfg.lift_height
        )
        new_lift = lifted & ~self._lifted_once
        self._lifted_once |= lifted
        if cfg.placement_anywhere_in_tray:
            # the release must start with the cube over the inside of the tray (a flat cube fits between the walls)
            opening_half = half_in - (self._item_size / 2).unsqueeze(-1)
        else:
            # the cube's bounding sphere and a clearance away from every wall
            opening_half = half_in - (radius + cfg.wall_clearance).unsqueeze(-1)
        over_opening = (self._tray_xy(self._item_pos).abs() < opening_half).all(dim=-1)
        above = self._lifted_once & held_clear & over_opening
        new_above = above & ~self._above_once
        self._above_once |= above
        self._release_armed = (self._release_armed | above) & over_opening & self._clear_rim & ~self._wall_contact
        hand_free = ~near_hand
        # Release must start over the opening after an observed grasped carry. Without contact
        # sensing, the opened hand marks the release; the palm cannot clear the item before it drops.
        released = self._release_armed & over_opening & unsupported & (self._hand_cmd[:, 0] < 0.3)
        if cfg.substep_release_check:
            # _release_seen already required the armed state at that substep (the armed state recomputed above is off
            # once the cube has dropped below the rim, so it must not be required again here)
            released |= self._release_seen & (self._hand_cmd[:, 0] < 0.3)
            self._release_seen[:] = False
        self._release_armed &= ~released
        new_release = released & ~self._release_rewarded
        self._release_rewarded |= released
        if cfg.placement_anywhere_in_tray:
            # the release must start over the opening; afterwards the cube may end anywhere in the tray (also against a
            # wall): the release only becomes invalid if the cube leaves the tray
            in_tray_xy = (self._tray_xy(self._item_pos).abs() < half_in + self._tray_t / 2).all(dim=-1)
            self._released_above = (self._released_above | released) & in_tray_xy
        else:
            # Wall-assisted placement or leaving the opening invalidates that release attempt.
            self._released_above = (self._released_above | released) & over_opening & ~self._wall_contact
        settled = (
            self._released_above
            & self._item_in_box()
            & hand_free
            & (self._hand_cmd[:, 0] < 0.3)
            & (torch.norm(self._item.data.root_lin_vel_w.torch, dim=-1) < cfg.settle_speed)
            & (torch.norm(self._item.data.root_ang_vel_w.torch, dim=-1) < cfg.settle_angular_speed)
        )
        self._settle_time = torch.where(settled, self._settle_time + self.step_dt, 0.0)
        success = self._settle_time >= cfg.success_hold_time
        new_placed = torch.zeros_like(success)
        if cfg.success_requires_return:
            # placed (the original success, latched), then the arm back at its start pose, still, with the gripper
            # open and the cube still in the tray, continuously for return_hold_time
            new_placed = success & ~self._placed
            self._placed |= success
            joint_pos = self._robot.data.joint_pos.torch[:, self._arm_ids]
            default = self._robot.data.default_joint_pos.torch[:, self._arm_ids]
            arm_error = torch.norm(joint_pos - default, dim=-1)
            arm_speed = torch.norm(self._robot.data.joint_vel.torch[:, self._arm_ids], dim=-1)
            home = (
                self._placed
                & self._item_in_box()
                & (torch.norm(self._item.data.root_lin_vel_w.torch, dim=-1) < cfg.settle_speed)
                & (self._hand_cmd[:, 0] < 0.3)
                & (arm_error < cfg.return_joint_tolerance)
                & (arm_speed < cfg.return_max_joint_speed)
                & self._return_extra_conditions()
            )
            self._return_time = torch.where(home, self._return_time + self.step_dt, 0.0)
            success = self._return_time >= cfg.return_hold_time
        else:
            self._placed |= success  # the placement is the success here (the placed flag marks it as well)
        self._stage_bonus = (
            cfg.rew_lift * new_lift.float()
            + cfg.rew_in_box * new_above.float()
            + cfg.rew_released * new_release.float()
            + cfg.rew_placed * new_placed.float()
            + cfg.rew_success * (success & ~self._succeeded).float()
        )
        self._succeeded |= success

    def _return_extra_conditions(self) -> torch.Tensor:
        """Further conditions of the return (all true here; tasks with a moving robot add e.g. being back at the start)."""
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _item_in_box(self) -> torch.Tensor:
        rel = self._tray_xy(self._item_pos)
        if self.cfg.placement_anywhere_in_tray:
            # cube resting on the tray floor anywhere inside the walls: a flat cube touching a wall has its center half
            # an edge from it (5 mm tolerance)
            half = self._tray_inner / 2 - (self._item_size / 2 - 0.005).unsqueeze(-1)
        else:
            half = self._tray_inner / 2 - self._item_radius.unsqueeze(-1)  # clear of the walls for any orientation
        rest_z = self._table_h + self._tray_t + self._item_size / 2
        return (
            (rel[:, 0].abs() < half[:, 0])
            & (rel[:, 1].abs() < half[:, 1])
            & (self._item_pos[:, 2] < rest_z + 0.015)
            & (self._item_pos[:, 2] > rest_z - 0.005)
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_task_values()
        self._update_task_progress()
        item_fell = self._item_pos[:, 2] < self._table_h - self.cfg.fall_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # with terminate_on_success False the episode continues after success (e.g. to return to the start pose)
        terminated = item_fell | self._succeeded if self.cfg.terminate_on_success else item_fell
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = _log_value(self._succeeded[env_ids].float().mean())
        if self.cfg.success_requires_return:
            log["Metrics/placed_rate"] = _log_value(self._placed[env_ids].float().mean())
        self._placed[env_ids] = False
        self._return_time[env_ids] = 0.0
        self._succeeded[env_ids] = False
        for buffer in (
            self._grasped,
            self._lifted_once,
            self._above_once,
            self._released_above,
            self._release_rewarded,
            self._release_armed,
            self._release_seen,
            self._grasp_time,
            self._settle_time,
            self._best_lift,
            self._best_transport,
            self._best_transport_fine,
            self._stage_bonus,
        ):
            buffer[env_ids] = 0

        super()._reset_idx(env_ids)
        n = len(env_ids)

        # robot: default pose with noise on the arm
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._arm_ids] += sample_uniform(
            -self.cfg.arm_joint_pos_noise,
            self.cfg.arm_joint_pos_noise,
            (n, len(self._arm_ids)),
            self.device,
        )
        joint_pos[:, self._arm_ids] = torch.clamp(joint_pos[:, self._arm_ids], self._arm_lower, self._arm_upper)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self._arm_targets[env_ids] = joint_pos[:, self._arm_ids]
        self._hand_cmd[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # table: random height (the table top moves; the tray and the cube stand on it)
        if self.cfg.table_height_range > 0.0:
            r = self.cfg.table_height_range
            self._table_h[env_ids] = self.cfg.table_height_center + sample_uniform(-r, r, (n,), self.device)
            table_pos = torch.tensor(TABLE_POS, device=self.device).repeat(n, 1)
            table_pos[:, 2] += self._table_h[env_ids] - TABLE_H
            table_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(n, 1)
            self.scene["table"].write_root_pose_to_sim_index(
                root_pose=torch.cat((table_pos + self.scene.env_origins[env_ids], table_quat), dim=-1), env_ids=env_ids
            )
            self._item_rest_z[env_ids] = self._table_h[env_ids] + self._item_size[env_ids] / 2

        # tray: random center and yaw (every part moves with it)
        center = torch.tensor(BOX_POS, device=self.device).repeat(n, 1)
        center[:, 2] = self._table_h[env_ids]
        center[:, :2] += sample_uniform(-self.cfg.tray_pos_noise, self.cfg.tray_pos_noise, (n, 2), self.device)
        tray_yaw = sample_uniform(-self.cfg.tray_yaw_range, self.cfg.tray_yaw_range, (n,), self.device)
        self._box_center_local[env_ids] = center
        self._box_yaw[env_ids] = tray_yaw
        self._carry_target_local[env_ids] = center + torch.tensor([0.0, 0.0, self.cfg.carry_height], device=self.device)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(n, 1)
        tray_quat = quat_from_angle_axis(tray_yaw, z_axis)
        c, s = torch.cos(tray_yaw), torch.sin(tray_yaw)
        for part, offset in self._tray_parts:
            part_pos = center.clone()
            part_pos[:, 0] += c * offset[0] - s * offset[1]
            part_pos[:, 1] += s * offset[0] + c * offset[1]
            part_pos[:, 2] += offset[2]
            part.write_root_pose_to_sim_index(
                root_pose=torch.cat((part_pos + self.scene.env_origins[env_ids], tray_quat), dim=-1), env_ids=env_ids
            )

        # item: random xy on the table and random yaw, resting on the table
        pos = torch.tensor(ITEM_POS, device=self.device).repeat(n, 1)
        pos[:, 2] = self._item_rest_z[env_ids]
        pos[:, :2] += sample_uniform(-self.cfg.item_pos_noise, self.cfg.item_pos_noise, (n, 2), self.device)
        pos += self.scene.env_origins[env_ids]
        yaw = sample_uniform(-torch.pi, torch.pi, (n,), self.device)
        quat = quat_from_angle_axis(yaw, z_axis)
        self._item.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        self._item.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )
