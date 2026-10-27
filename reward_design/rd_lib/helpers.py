"""Tested geometry helpers for LLM-written derived quantities (passed to ``compute(env, q, h)`` as ``h``).

All functions work on batched torch tensors (first dimension = environments). Quaternions follow IsaacLab:
(x, y, z, w). Positions returned by the ``body_*``/``root_*`` helpers are environment-local (world position
minus ``env.scene.env_origins``), like the task quantities.
"""

import math

import torch

from isaaclab.utils import math as math_utils

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _body_index(asset, name):
    ids, names = asset.find_bodies(name)
    if len(ids) != 1:
        raise KeyError(f"body '{name}' matches {names or 'nothing'}; available bodies: {asset.body_names}")
    return ids[0]


def body_pos(env, name, asset="_robot"):
    """Position of a robot body [m], environment-local, shape (num_envs, 3)."""
    a = getattr(env, asset)
    return a.data.body_pos_w.torch[:, _body_index(a, name)] - env.scene.env_origins


def body_quat(env, name, asset="_robot"):
    """World orientation of a robot body, quaternion (x, y, z, w), shape (num_envs, 4)."""
    a = getattr(env, asset)
    return a.data.body_quat_w.torch[:, _body_index(a, name)]


def root_pos(env, asset):
    """Root position of an asset (its attribute name on the env) [m], environment-local, shape (num_envs, 3)."""
    return getattr(env, asset).data.root_pos_w.torch - env.scene.env_origins


def root_quat(env, asset):
    """Root orientation of an asset (its attribute name on the env), quaternion (x, y, z, w), shape (num_envs, 4)."""
    return getattr(env, asset).data.root_quat_w.torch


def rotate(quat, vec):
    """Rotate vectors by quaternions (x, y, z, w): (N, 4), (N, 3) -> (N, 3)."""
    return math_utils.quat_apply(quat, vec)


def rotate_inverse(quat, vec):
    """Rotate vectors by the inverse of quaternions (x, y, z, w) (world to body frame)."""
    return math_utils.quat_apply_inverse(quat, vec)


def axis(quat, name):
    """World direction of a local axis ("x", "y" or "z") of frames with orientation ``quat``, shape (N, 3)."""
    v = torch.tensor(AXES[name], device=quat.device, dtype=quat.dtype).expand(quat.shape[0], 3)
    return math_utils.quat_apply(quat, v)


def yaw(quat):
    """Yaw angle [rad] (rotation about the world z axis) of quaternions (x, y, z, w), in (-pi, pi]."""
    _, _, yaw_angle = math_utils.euler_xyz_from_quat(quat)
    return wrap_to_pi(yaw_angle)


def tilt(quat):
    """Angle [rad] between the local z axis of the frames and the world z axis (0 = upright)."""
    z = axis(quat, "z")
    return torch.acos(z[:, 2].clamp(-1.0, 1.0))


def heading(vec):
    """Yaw angle [rad] of the horizontal part of vectors (N, 3), in (-pi, pi]."""
    return torch.atan2(vec[:, 1], vec[:, 0])


def wrap_to_pi(angle):
    """Wrap angles [rad] to (-pi, pi]."""
    return math_utils.wrap_to_pi(angle)


def angle_diff_mod(a, b, period):
    """Smallest absolute difference [rad] between angles a and b modulo ``period``, in [0, period / 2].

    Use the symmetry of the object: period = pi / 2 for a cube's faces about its vertical axis, pi for a
    two-sided direction (e.g. a parallel gripper's closing line), 2 pi for a direction with a sign.
    """
    d = torch.remainder(a - b, period)
    return torch.minimum(d, period - d)


def angle_between(u, v):
    """Angle [rad] between vectors u and v (N, 3), in [0, pi]."""
    cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-9)
    return torch.acos(cos.clamp(-1.0, 1.0))


def to_frame(pos, frame_pos, frame_quat):
    """Express points (N, 3) in a frame given by its position and orientation (x, y, z, w)."""
    return math_utils.quat_apply_inverse(frame_quat, pos - frame_pos)


def distance(a, b):
    """Euclidean distance between points (N, 3) and (N, 3)."""
    return torch.norm(a - b, dim=-1)


PI = math.pi

from .derived import HELPERS_DOC as DOC  # noqa: E402,F401
