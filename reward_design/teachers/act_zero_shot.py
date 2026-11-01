"""Zero-shot test of a LeRobot ACT Dex1 picking policy (exported by export_act.py) in the two-tables task.

The robot is placed standing at the cube table as the robot of Unitree's red-block task stood (cube about 0.33 m in front
of the pelvis, 0.05 m to its right, cube top a few cm above the pelvis), with Unitree's simulator camera mounts (head on
d435_link, wrists on the hand base links), the arms in the dataset's start pose. ACT drives the right arm and gripper
(absolute joint targets, 30 Hz chunks); the left arm holds its pose. Reports how often the right gripper reaches, grasps
and lifts the cube, and saves camera frames of environment 0 next to the dataset's.
"""

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=14.0)
parser.add_argument("--policy", default=os.path.join(os.path.dirname(__file__), "act_redblock.pt"))
parser.add_argument("--replan", type=int, default=10, help="ACT steps (30 Hz) executed per chunk before re-querying")
parser.add_argument("--table_above_pelvis", type=float, default=0.05)
parser.add_argument("--forward", type=float, default=0.0, help="cube ahead of the right grasp point [m]")
parser.add_argument("--right", type=float, default=0.09, help="cube to the right of the right grasp point [m]")
parser.add_argument("--red_grey", type=int, default=1, help="1: red 6 cm cube on a grey table as in the dataset")
parser.add_argument("--out", default="act_zero_shot")
parser.add_argument("--server", type=int, default=0, help="port of vla_server.py (a VLA) instead of the --policy TorchScript")
parser.add_argument("--task", default="Pick up the red cube.")
parser.add_argument("--pelvis_height", type=float, default=0.786, help="standing pelvis height (measured)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis

ROOT = os.path.expanduser("~/automatic_reward_design")
sys.path.insert(0, os.path.join(ROOT, "tasks"))
sys.path.insert(0, os.path.join(ROOT, "reward_design"))
import g1_dex1_two_tables  # noqa: F401
from g1_dex1_two_tables import env_cfg as EC
from g1_dex1_two_tables.env import _rotate_xy
from g1_dex1_two_tables.env_cfg import TABLE_BLOCK_H
from rd_lib.quantities import get_adapter

os.makedirs(args.out, exist_ok=True)


def ros_to_usd(q_wxyz):
    """Camera orientation in Unitree's config (ROS convention: +z forward, y down; w, x, y, z) as the USD camera prim's
    orientation (OpenGL convention: -z forward, y up), in (x, y, z, w) as the spawners here take it."""
    w, x, y, z = q_wxyz
    # q * (0, 1, 0, 0): a half turn about the camera's own x axis
    q = (-x, w, z, -y)  # (w, x, y, z) of the product
    return (q[1], q[2], q[3], q[0])


# Unitree's unitree_sim_isaaclab camera presets (tasks/common_config/camera_configs.py), 640 x 480
ACT_CAMS = [
    ("act_head", "d435_link", 7.6, 20.0, (0.0, 0.0, 0.0), (0.5, -0.5, 0.5, -0.5)),
    ("act_left_wrist", "left_hand_base_link", 12.0, 20.0, (0.02541028, 0.045, 0.135), (-0.34202, 0.93969, 0.0, 0.0)),
    ("act_right_wrist", "right_hand_base_link", 12.0, 20.0, (-0.02541028, 0.045, 0.135), (-0.34202, 0.93969, 0.0, 0.0)),
]


def spawn_with_act_cams(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from isaaclab.sim.utils import find_matching_prim_paths

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    for robot_path in find_matching_prim_paths(prim_path):
        for name, parent, focal, aperture, pos, rot in ACT_CAMS:
            spawn = sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=aperture,
                                               clipping_range=(0.01, 100.0))
            spawn.func(f"{robot_path}/{parent}/{name}", spawn, translation=pos, orientation=ros_to_usd(rot))
    return prim


@configclass
class RobotWithActCamsCfg(sim_utils.UsdFileCfg):
    func = spawn_with_act_cams


def cam_cfg(name, parent):
    return TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/" + parent + "/" + name, update_period=0.0, width=640,
                          height=480, data_types=["rgb"], spawn=None)


@configclass
class ActScene(EC.G1Dex1TwoTablesSceneCfg):
    act_head: TiledCameraCfg = None
    act_left_wrist: TiledCameraCfg = None
    act_right_wrist: TiledCameraCfg = None

    def __post_init__(self):
        super().__post_init__()
        spawn = self.robot.spawn
        self.robot = self.robot.replace(spawn=RobotWithActCamsCfg(
            **{k: getattr(spawn, k) for k in spawn.__dataclass_fields__ if k != "func"}))
        for name, parent, *_ in ACT_CAMS:
            setattr(self, name, cam_cfg(name, parent))


torch.set_grad_enabled(False)
inf = torch.inference_mode()
inf.__enter__()
cfg = EC.G1Dex1TwoTablesEnvCfg()
cfg.scene = ActScene(num_envs=args.num_envs, env_spacing=16.0, replicate_physics=True)
cfg.scene.tray = EC._tray_cfg(cfg.tray_variants, cfg.tray_inner_range, cfg.tray_wall_height_range,
                              cfg.tray_variant_seed)
if args.red_grey:
    cube = sim_utils.CuboidCfg(
        size=(0.06, 0.06, 0.06), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.0))
    cfg.scene.item = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Item", spawn=cube,
                                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)))
    cfg.item_sizes = (0.06,)
    grey = cfg.scene.cube_table.spawn.assets_cfg[2]
    cfg.scene.cube_table = cfg.scene.cube_table.replace(spawn=grey)
cfg.episode_length_s = args.seconds + 5.0
# the cube table at the dataset's height above the pelvis, set before the reset places the cube on it
cfg.cube_table_height = (args.pelvis_height + args.table_above_pelvis,) * 2
env = gym.make("G1-Dex1-Two-Tables-v0", cfg=cfg)
e = env.unwrapped
q = get_adapter("g1_dex1_two_tables")
n, dev = e.num_envs, e.device
# no resets during the test (the placement below moves robot, cube and table outside the task's own reset); falls and
# dropped cubes are counted instead
_dones = e._get_dones
terminations = torch.zeros(n, dtype=torch.bool, device=dev)


def _no_dones():
    terminated, time_out = _dones()
    terminations.logical_or_(terminated)
    return torch.zeros_like(terminated), torch.zeros_like(time_out)


e._get_dones = _no_dones
if args.server:
    import pickle
    import socket
    import struct

    conn = socket.create_connection(("127.0.0.1", args.server))

    def policy(state, head, left_wrist, right_wrist):
        data = pickle.dumps({"state": state.cpu().numpy(), "head": head.cpu().numpy(), "left_wrist": left_wrist.cpu().numpy(),
                             "right_wrist": right_wrist.cpu().numpy(), "task": args.task}, protocol=5)
        conn.sendall(struct.pack("!Q", len(data)) + data)
        size = struct.unpack("!Q", conn.recv(8, socket.MSG_WAITALL))[0]
        return torch.as_tensor(pickle.loads(conn.recv(size, socket.MSG_WAITALL)), device=dev)
else:
    policy = torch.jit.load(args.policy, map_location=dev).eval()
names = list(e._robot.joint_names)
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
left_ids = [names.index(f"left_{j}_joint") for j in ARM]
right_ids = [names.index(f"right_{j}_joint") for j in ARM]
arm_ids = list(e._arm_ids)
right_in_targets = [arm_ids.index(i) for i in right_ids]
left_in_targets = [arm_ids.index(i) for i in left_ids]
START = torch.tensor([-0.13, 0.04, -0.063, 0.415, 0.008, -0.34, 0.31,
                      -0.244, -0.125, -0.044, 0.4, 0.147, -0.235, 0.446], device=dev)  # dataset start pose (mean)

env.reset(seed=0)
zero = torch.zeros(n, e.cfg.action_space, device=dev)
zero[:, 10] = -1.0
for _ in range(10):
    env.step(zero)
pelvis_z = e._robot.data.root_pos_w.torch[:, 2].mean().item()
print(f"standing pelvis height {pelvis_z:.3f}")

# arms: the dataset's start pose, while the robot still stands away from the tables
origins = e.scene.env_origins
ids = torch.arange(n, device=dev)
qpos = e._robot.data.joint_pos.torch.clone()
qpos[:, left_ids + right_ids] = START
e._robot.write_joint_state_to_sim_index(position=qpos, velocity=torch.zeros_like(qpos), env_ids=ids)
e._arm_targets[:, left_in_targets + right_in_targets] = START
e._walker.reset(ids, e._walker_arm_targets())
for _ in range(25):
    env.step(zero)
# the cube table just below the lowest fingertip (in the dataset the open grippers rest on the table at the start)
tips = e._robot.data.body_pos_w.torch[:, e._tip_ids.flatten(), 2].min(dim=-1).values
pelvis = e._robot.data.root_pos_w.torch[:, 2]
print(f"lowest fingertip {(tips - pelvis).mean().item():+.3f} m from the pelvis")
table_h = tips - 0.01
e._cube_table_top[:, 2] = table_h - origins[:, 2]
pose = e._cube_table_top + origins  # as the task's reset writes it: block center below the top
pose[:, 2] -= TABLE_BLOCK_H / 2
z_up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
e.scene["cube_table"].write_root_pose_to_sim_index(
    root_pose=torch.cat((pose, quat_from_angle_axis(e._cube_table_yaw, z_up)), -1), env_ids=ids)
e._item_rest_z[:] = e._cube_table_top[:, 2] + e._item_size / 2
item = e._item.data.root_pos_w.torch.clone()
item[:, 2] = table_h + e._item_size / 2 + 0.001
e._item.write_root_pose_to_sim_index(root_pose=torch.cat((item, e._item.data.root_quat_w.torch), -1), env_ids=ids)
e._item.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
for _ in range(5):
    env.step(zero)
e._pickup_xy[:] = e._item_pos[:, :2]


def where(tag):
    rel = _rotate_xy(e._item_pos[:, :2] - e._cube_table_top[:, :2], -e._cube_table_yaw)
    print(tag, "cube above the table top [m]:", [round(v, 3) for v in (e._item_pos[:, 2] - e._cube_table_top[:, 2]).tolist()])
    print(tag, "cube in the table frame:", [[round(a, 2), round(b, 2)] for a, b in rel.tolist()])
    print(tag, "terminated so far:", terminations.nonzero().flatten().tolist())


where("after the table move:")

# place the robot at the cube's edge, as Unitree's robot stood
top, yaw = e._cube_table_top.clone(), e._cube_table_yaw
cube_local = _rotate_xy(e._item_pos[:, :2] - top[:, :2], -yaw)  # in the table frame
half = torch.tensor(EC.TABLE_TOP, device=dev) / 2
to_side = half[0] - cube_local[:, 0].abs()
to_end = half[1] - cube_local[:, 1]
normal = torch.where((to_side < to_end).unsqueeze(-1),
                     torch.stack((torch.sign(cube_local[:, 0]), torch.zeros(n, device=dev)), -1),
                     torch.tensor([0.0, 1.0], device=dev).expand(n, 2))
normal = _rotate_xy(normal, yaw)  # outward edge normal, env frame
forward = -normal
left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
# the right grasp point in the pelvis frame (start pose): the cube goes beside it, as in the dataset's first frames
root_now = e._robot.data.root_pos_w.torch[:, :2] - origins[:, :2]
yaw_now = 2 * torch.atan2(e._robot.data.root_quat_w.torch[:, 2], e._robot.data.root_quat_w.torch[:, 3])
grip = _rotate_xy(e._palm_pos[:, :2] - root_now, -yaw_now)  # x forward, y left
print("right grasp point in the pelvis frame (mean):", [round(v, 3) for v in grip.mean(0).tolist()])
cube_in_pelvis = grip + torch.tensor([args.forward, -args.right], device=dev)
pelvis_xy = e._item_pos[:, :2] - forward * cube_in_pelvis[:, :1] - left * cube_in_pelvis[:, 1:]
heading = torch.atan2(forward[:, 1], forward[:, 0])
z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
root = e._robot.data.root_pos_w.torch.clone()
root[:, :2] = pelvis_xy + origins[:, :2]
e._robot.write_root_pose_to_sim_index(root_pose=torch.cat((root, quat_from_angle_axis(heading, z_axis)), -1), env_ids=ids)
e._robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
for _ in range(25):  # settle (still holding the start pose)
    env.step(zero)
where("after placing the robot:")
print("terminations while settling (fell or cube dropped):", terminations.nonzero().flatten().tolist(),
      "fell:", e._fell.nonzero().flatten().tolist())
terminations[:] = False


def gripper_value(jaw):
    """Unitree's Dex1 value (0 closed .. 5.6 open) of a jaw joint position (their map: value 0 -> 0.03, 5.6 -> -0.02)."""
    return ((0.03 - jaw) / 0.05 * 5.6).clamp(0.0, 5.6)


def frames():
    return [e.scene[name].data.output["rgb"][..., :3].to(torch.uint8) for name, *_ in ACT_CAMS]


steps = int(args.seconds / e.step_dt)
chunk, t_query = None, 0.0
reached = torch.zeros(n, dtype=torch.bool, device=dev)
grasp = torch.zeros(n, dtype=torch.bool, device=dev)
jaws = torch.zeros(n, dtype=torch.bool, device=dev)
lifted = torch.zeros(n, dtype=torch.bool, device=dev)
closest = torch.full((n,), 9.0, device=dev)
log, saved = [], []
base_to_cube = _rotate_xy(e._item_pos[:, :2] - e._robot.data.root_pos_w.torch[:, :2] + origins[:, :2],
                          -heading)  # cube in the pelvis frame (x forward, y left)
print("cube in the pelvis frame per env (x fwd, y left, z above pelvis):")
for i in range(n):
    print(i, [round(v, 3) for v in base_to_cube[i].tolist()],
          round((e._item_pos[i, 2] - e._robot.data.root_pos_w.torch[i, 2]).item(), 3),
          "grasp-point dist", round(q["palm_to_cube_dist"].fn(e)[i].item(), 3), "length", int(e.episode_length_buf[i]))
for k in range(steps):
    t = k * e.step_dt
    idx = int(round((t - t_query) * 30))
    if chunk is None or idx >= args.replan:
        jp = e._robot.data.joint_pos.torch
        jaw_r = e._right_jaw_pos
        jaw_l = jp[:, e._hand_ids[2:]].mean(-1)
        state = torch.cat((jp[:, left_ids], jp[:, right_ids], gripper_value(jaw_l).unsqueeze(-1),
                           gripper_value(jaw_r).unsqueeze(-1)), -1)
        chunk = policy(state, *frames())
        t_query, idx = t, 0
    a = chunk[:, min(idx, chunk.shape[1] - 1)]
    e._arm_targets[:, right_in_targets] = torch.max(torch.min(a[:, 7:14], e._arm_upper[right_in_targets]),
                                                    e._arm_lower[right_in_targets])
    jaw_target = 0.03 - 0.05 * a[:, 15] / 5.6
    cmd = ((jaw_target + 0.02) / 0.044).clamp(0.0, 1.0)
    act = zero.clone()
    act[:, 10] = 2 * cmd - 1
    env.step(act)
    d = q["palm_to_cube_dist"].fn(e)
    closest = torch.minimum(closest, d)
    reached |= d < 0.05
    jaws |= q["jaws_on_cube"].fn(e)
    grasp |= q["secure_grasp"].fn(e)
    lifted |= q["cube_controlled"].fn(e) & (q["cube_bottom_height"].fn(e) > 0.03)
    if k % 50 == 0:
        log.append(dict(t=round(t, 1), dist0=round(d[0].item(), 3), cmd0=round(cmd[0].item(), 2),
                        reached=reached.float().mean().item(), grasp=grasp.float().mean().item()))
        print(log[-1], flush=True)
    if k % 100 == 0:
        import PIL.Image
        imgs = [f[0].cpu().numpy() for f in frames()]
        PIL.Image.fromarray(np.concatenate(imgs, 1)).resize((960, 240)).save(f"{args.out}/env0_t{t:04.1f}.jpg")
        imgs = [f[3].cpu().numpy() for f in frames()]
        PIL.Image.fromarray(np.concatenate(imgs, 1)).resize((960, 240)).save(f"{args.out}/env3_t{t:04.1f}.jpg")
print("closest approach per env [m]:", [round(v, 3) for v in closest.tolist()])
result = {
    "envs": n, "seconds": args.seconds,
    "grasp point within 5 cm of the cube": reached.float().mean().item(),
    "jaws on the cube": jaws.float().mean().item(),
    "secure grasp": grasp.float().mean().item(),
    "lifted 3 cm and held": lifted.float().mean().item(),
    "closest approach [m] (median)": closest.median().item(),
    "fell": e._fell.float().mean().item(),
    "terminated during the test (fell or cube dropped)": terminations.float().mean().item(),
}
print(json.dumps(result, indent=1))
json.dump({"result": result, "log": log, "args": vars(args)}, open(f"{args.out}/result.json", "w"), indent=1)
env.close()
app.close()
