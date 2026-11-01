"""Unitree's G1 walking policy (from ``unitree_sim_isaaclab``) as a frozen low-level controller of the legs.

The policy is the one Unitree's simulator uses for its "Wholebody" tasks (``assets/model/policy.onnx``). It is a small
MLP (910 -> 512 -> 256 -> 128 -> 12, ELU) that reads a 10-step history of the robot state and a walking command and
outputs the 12 leg joint targets; the waist is held at its default pose and the arms are free (their positions and
targets are part of its input). It runs at 50 Hz (physics dt 0.005 s, 4 substeps), like in Unitree's simulator.

The ONNX weights are copied into a torch module, so the walker runs batched on the GPU for all environments.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# joint order of the policy's leg outputs and of the leg part of its joint observations
LEG_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# joint order of the "previous action" observation (the 29 body joints in the order of Unitree's training setup)
ACTION_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip
HISTORY = 10
FRAME = 3 + 3 + 4 + 26 + 26 + 29  # angular velocity, gravity, command, joint positions, joint velocities, last action
ACTION_SCALE = 0.25  # leg target = output * scale + default position [rad]


def load_walker_mlp(path: str, device: str) -> nn.Module:
    """The walking policy's MLP with the weights of the ONNX file.

    Args:
        path: The ONNX file of the policy.
        device: Torch device of the module.

    Returns:
        The policy network (input: 910 observation values, output: 12 leg actions).
    """
    import onnx
    from onnx import numpy_helper

    weights = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in onnx.load(path).graph.initializer}
    layers = []
    for i in (0, 2, 4, 6):
        weight, bias = weights[f"actor.{i}.weight"], weights[f"actor.{i}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if i != 6:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).to(device).eval()


class UnitreeWalker:
    """Batched walking controller: walking command -> leg joint targets, with the observation history of the policy.

    Args:
        policy_path: The ONNX file of Unitree's walking policy.
        joint_names: All joint names of the robot articulation (in articulation order).
        default_joint_pos: Default joint positions of the robot [rad or m], shape [num_envs, num_joints].
        num_envs: Number of environments.
        device: Torch device.
        body_height: Body height command of the policy [m] (0.8 in Unitree's simulator).
    """

    def __init__(self, policy_path, joint_names, default_joint_pos, num_envs, device, body_height=0.8,
                 history_layout="frame", ang_vel_scale=1.0, joint_vel_scale=1.0):
        index = {name: i for i, name in enumerate(joint_names)}
        self.leg_ids = torch.tensor([index[n] for n in LEG_JOINTS], device=device)
        self.waist_ids = torch.tensor([index[n] for n in WAIST_JOINTS], device=device)
        self.arm_ids = torch.tensor([index[n] for n in ARM_JOINTS], device=device)
        self.obs_ids = torch.cat((self.leg_ids, self.arm_ids))
        self.action_ids = torch.tensor([index[n] for n in ACTION_JOINTS], device=device)
        self.policy = load_walker_mlp(policy_path, device)
        self.default = default_joint_pos.clone()
        self.body_height = body_height
        self.num_envs = num_envs
        self.device = device
        self.history = torch.zeros(num_envs, HISTORY, FRAME, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)  # the next frame fills the whole history
        # the last action of the policy's observation: leg outputs (unscaled), waist and arm targets (absolute)
        self.last_action = torch.zeros(num_envs, len(ACTION_JOINTS), device=device)
        self.leg_targets = self.default[:, self.leg_ids].clone()
        # history layout of the observation: "frame" (10 whole frames, oldest first, as in Unitree's simulator) or
        # "term" (each term's 10 values, oldest first, term after term, as IsaacLab's observation manager stacks them)
        self.history_layout = history_layout
        self.term_sizes = [3, 3, 4, 26, 26, 29]
        self.ang_vel_scale = ang_vel_scale
        self.joint_vel_scale = joint_vel_scale

    def reset(self, env_ids: torch.Tensor, arm_targets: torch.Tensor):
        """Start a new history for these environments (robot in its default pose).

        Args:
            env_ids: Environment indices.
            arm_targets: Current targets of the 14 arm joints (order :data:`ARM_JOINTS`) of these environments [rad].
        """
        self.fresh[env_ids] = True
        full = self.default[env_ids].clone()
        full[:, self.leg_ids] = 0.0  # leg outputs are relative: 0 = default pose
        full[:, self.arm_ids] = arm_targets
        self.last_action[env_ids] = full[:, self.action_ids]
        self.leg_targets[env_ids] = self.default[env_ids][:, self.leg_ids]

    @torch.inference_mode()
    def step(self, command, ang_vel_b, gravity_b, joint_pos, joint_vel, arm_targets):
        """One 50 Hz step of the walking policy.

        Args:
            command: Walking command (forward speed [m/s], sideways speed [m/s], turn rate [rad/s]), shape [num_envs, 3].
            ang_vel_b: Base angular velocity in the base frame [rad/s], shape [num_envs, 3].
            gravity_b: Gravity direction in the base frame, shape [num_envs, 3].
            joint_pos: All joint positions [rad or m], shape [num_envs, num_joints].
            joint_vel: All joint velocities [rad/s or m/s], shape [num_envs, num_joints].
            arm_targets: Targets of the 14 arm joints (order :data:`ARM_JOINTS`) [rad], shape [num_envs, 14].

        Returns:
            Leg joint targets (order :data:`LEG_JOINTS`) [rad], shape [num_envs, 12].
        """
        height = torch.full((self.num_envs, 1), self.body_height, device=self.device)
        frame = torch.cat(
            (
                ang_vel_b * self.ang_vel_scale,
                gravity_b,
                command,
                height,
                joint_pos[:, self.obs_ids] - self.default[:, self.obs_ids],
                joint_vel[:, self.obs_ids] * self.joint_vel_scale,
                self.last_action,
            ),
            dim=-1,
        )
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = frame
        if bool(self.fresh.any()):
            self.history[self.fresh] = frame[self.fresh].unsqueeze(1).expand(-1, HISTORY, -1)
            self.fresh[:] = False
        if self.history_layout == "term":
            obs = torch.cat([part.reshape(self.num_envs, -1) for part in self.history.split(self.term_sizes, dim=-1)], -1)
        else:
            obs = self.history.reshape(self.num_envs, -1)
        obs = obs.clamp(-100.0, 100.0)
        actions = self.policy(obs).clamp(-100.0, 100.0)
        # the next observation's "last action": leg outputs, waist at its default pose, arm targets
        full = self.default.clone()
        full[:, self.leg_ids] = actions
        full[:, self.arm_ids] = arm_targets
        self.last_action = full[:, self.action_ids]
        self.leg_targets = actions * ACTION_SCALE + self.default[:, self.leg_ids]
        return self.leg_targets
