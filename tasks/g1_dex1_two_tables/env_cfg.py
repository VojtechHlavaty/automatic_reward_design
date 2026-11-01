"""Configurations of the G1 + Dex1-1 two-table task: the robot walks to a table, picks the cube and places it into the
tray on a second table.

The robot is set up as in the table pick-and-place task (``g1_dex1_table``): Unitree's G1 29-DoF + Dex1-1 model, its
arm, wrist and gripper drives and start pose, here with a floating base. Unitree's walking policy (``walker.py``)
drives the legs; the task policy commands the walk (forward and sideways speed, turn rate) and moves both arms and
the right gripper: the right hand picks, carries and places the cube, the left arm aims its wrist camera at the right
hand and the cube (a second view for camera policies), and its gripper stays open. The two tables stand end to end,
joined by their short sides. The robot starts 1-5 m from the cube table, in front of it, behind it or beyond its far
end (never on the tray table's side), facing it. Table heights, the gap between the tables, the cube (size, color,
mass) and the tray (shape, color) vary. The camera variant adds the calibrated head and wrist cameras of the real G1-D.
"""

from __future__ import annotations

import math
import os
import random

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from g1_dex1_table.env_cfg import GRIPPER_CLOSED, GRIPPER_OPEN
from g1_dex1_table.tray_spawner import TrayCfg, tray_variants

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
WALKER_POLICY = os.path.join(TASK_DIR, "assets", "unitree_walker_policy.onnx")
# Unitree's G1 29-DoF + Dex1-1 model of the table task with a floating base: a small layer over the table task's USD
# that disables the joint fixing the pelvis to the world and makes the pelvis the articulation root
ROBOT_USD = os.path.join(TASK_DIR, "assets", "g1_29dof_dex1_floating.usda")

TABLE_TOP = (0.6, 1.0)  # table top depth (x) and width (y) in the table frame [m]
TABLE_BLOCK_H = 1.2  # tables are blocks this tall, sunk into the floor so that the top is at the table height [m]
TRAY_T = 0.01  # tray wall and floor thickness [m]

ITEM_SIZES = tuple(round(0.025 + 0.005 * i, 3) for i in range(10))  # cube edge lengths 0.025-0.07 m [m]
ITEM_COLORS = (
    (0.10, 0.45, 0.12),  # green
    (0.10, 0.25, 0.70),  # blue
    (0.85, 0.75, 0.10),  # yellow
    (0.55, 0.15, 0.60),  # purple
    (0.90, 0.40, 0.05),  # orange
    (0.05, 0.60, 0.60),  # teal
)
ITEM_MASS_RANGE = (0.05, 0.20)  # cube mass [kg], one value per cube variant
TRAY_COLORS = ((0.80, 0.08, 0.08), (0.15, 0.15, 0.15), (0.85, 0.85, 0.85), (0.20, 0.35, 0.20))
TABLE_COLORS = ((0.22, 0.12, 0.05), (0.45, 0.30, 0.15), (0.60, 0.60, 0.62), (0.10, 0.10, 0.10))
VARIANT_SEED = 0

# The walking policy's standing pose (Unitree's whole-body simulation, G129_CFG_WITH_DEX1_WHOLEBODY): the walker
# observes the joints relative to it, whatever the robot's own start pose is
WALKER_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_pitch_joint": 0.35,
}

# The robot as in the table pick-and-place task (g1_dex1_table: G1_DEX1_CFG and its scene): Unitree's G1 29-DoF +
# Dex1-1 model, its arm, wrist and gripper drives, and the right arm raised in front with the gripper pointing forward
# (here the left arm too, mirrored: it carries the camera that watches the right hand). The legs, feet and waist get the drives of Unitree's whole-body simulation, which
# the walking policy was trained with, instead of the fixed-base task's locked legs.
G1_DEX1_WALK_CFG = G1_29DOF_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=G1_29DOF_CFG.spawn.replace(
        usd_path=ROBOT_USD,
        articulation_props=G1_29DOF_CFG.spawn.articulation_props.replace(fix_root_link=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            # left arm: the mirror image of the right arm's start pose
            "left_shoulder_pitch_joint": -0.512,
            "left_shoulder_roll_joint": 0.300,
            "left_shoulder_yaw_joint": 0.004,
            "left_elbow_joint": 0.140,
            "left_wrist_roll_joint": -0.398,
            "left_wrist_pitch_joint": 0.640,
            "left_wrist_yaw_joint": -0.627,
            # right arm: raised in front of the body, gripper pointing forward (as in the pick-and-place task)
            "right_shoulder_pitch_joint": -0.512,
            "right_shoulder_roll_joint": -0.300,
            "right_shoulder_yaw_joint": -0.004,
            "right_elbow_joint": 0.140,
            "right_wrist_roll_joint": 0.398,
            "right_wrist_pitch_joint": 0.640,
            "right_wrist_yaw_joint": 0.627,
            ".*_hand_Joint[12]_1": GRIPPER_OPEN,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Unitree's whole-body drives (unitree_sim_isaaclab, robots/unitree.py: G129_CFG_WITH_DEX1_WHOLEBODY)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", ".*waist.*"],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                ".*waist.*": 200.0,
            },
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        # the pick-and-place task's arm and wrist drives and gripper drives
        "arms": G1_29DOF_CFG.actuators["arms"],
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_Joint[12]_1"],
            stiffness=800.0,
            damping=3.0,
        ),
    },
)


def _kinematic_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)


def _table_cfg(name: str) -> RigidObjectCfg:
    """A table: a kinematic block (one of the table colors per environment); its pose is written at every reset."""
    blocks = [
        sim_utils.CuboidCfg(
            size=(TABLE_TOP[0], TABLE_TOP[1], TABLE_BLOCK_H),
            rigid_props=_kinematic_props(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        )
        for color in TABLE_COLORS
    ]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=blocks, random_choice=False),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -TABLE_BLOCK_H)),
    )


def item_variants(seed: int = VARIANT_SEED) -> list[tuple[float, tuple[float, float, float], float]]:
    """The cube variants (edge length [m], color, mass [kg]): every size in every color, masses drawn reproducibly."""
    rng = random.Random(seed)
    return [(s, c, round(rng.uniform(*ITEM_MASS_RANGE), 3)) for s in ITEM_SIZES for c in ITEM_COLORS]


def _item_cfg() -> RigidObjectCfg:
    cubes = [
        sim_utils.CuboidCfg(
            size=(s, s, s),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=m),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c, roughness=0.6),
        )
        for s, c, m in item_variants()
    ]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Item",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=cubes, random_choice=False),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _tray_cfg(count: int, inner_range, height_range, seed: int) -> RigidObjectCfg:
    trays = [
        TrayCfg(
            inner_size=(ix, iy),
            wall_height=h,
            wall_thickness=TRAY_T,
            rigid_props=_kinematic_props(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TRAY_COLORS[i % len(TRAY_COLORS)], roughness=0.6),
        )
        for i, (ix, iy, h) in enumerate(tray_variants(count, inner_range, height_range, seed))
    ]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tray",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=trays, random_choice=False),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@configclass
class G1Dex1TwoTablesSceneCfg(InteractiveSceneCfg):
    # ground plane and light are spawned in the environment's _setup_scene
    robot: ArticulationCfg = G1_DEX1_WALK_CFG
    cube_table = _table_cfg("cube_table")
    tray_table = _table_cfg("tray_table")
    item = _item_cfg()
    tray: RigidObjectCfg = None  # set by the env config (it knows the tray variants)


@configclass
class G1Dex1TwoTablesEnvCfg(DirectRLEnvCfg):
    # env: 50 Hz control, like the walking policy (physics 200 Hz)
    episode_length_s = 40.0
    decimation = 4
    action_space = 18  # walk command (3), right arm (7), right gripper (1), left arm (7)
    observation_space = 95
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    light_intensity = 1000.0
    scene: G1Dex1TwoTablesSceneCfg = G1Dex1TwoTablesSceneCfg(num_envs=1024, env_spacing=16.0, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(eye=(-4.0, -4.0, 3.5), lookat=(1.5, 1.5, 0.6), origin_type="env", env_index=0)
    review_camera_follow = True  # videos: the camera follows the robot (env.review_camera_view), not the fixed viewer

    # joints / bodies (the task policy drives both arms and the right gripper: the right hand picks and places, the left
    # arm aims its wrist camera at the right hand; the left gripper stays open; the walker holds the waist)
    arm_joint_names = [
        "right_shoulder_.*", "right_elbow_joint", "right_wrist_.*", "left_shoulder_.*", "left_elbow_joint", "left_wrist_.*"
    ]
    hand_joint_names = ["right_hand_Joint1_1", "right_hand_Joint2_1", "left_hand_Joint1_1", "left_hand_Joint2_1"]
    palm_body_names = ["right_hand_Link1_3", "right_hand_Link2_3"]  # (the base task's; the grasp points come from below)
    # fingertips of each gripper (right, left): the grasp point of a gripper is the midpoint between its fingertips
    hand_tip_body_names = (("right_hand_Link1_3", "right_hand_Link2_3"), ("left_hand_Link1_3", "left_hand_Link2_3"))
    # half field of view of the left wrist camera, horizontal and vertical [rad] (Adam Basl's calibration: 94.0 x 76.9 deg)
    left_camera_half_fov = (0.820, 0.671)
    hand_open_pos = {"Joint": GRIPPER_OPEN}
    hand_closed_pos = {"Joint": GRIPPER_CLOSED}
    arm_action_scale = 1.5  # max arm joint target speed [rad/s]
    # walking command limits (actions in [-1, 1] are scaled to these), the ranges of Unitree's keyboard tool. From
    # standing, the walking policy ignores small commands (about < 0.5 m/s forward, < 0.3-0.5 m/s sideways,
    # < 1 rad/s turning): it walks with larger ones and stops below them.
    max_forward_speed = 1.0  # [m/s] (backwards up to half of it)
    max_side_speed = 0.5  # [m/s]
    max_turn_rate = 1.5  # [rad/s]
    walker_policy = WALKER_POLICY
    walker_body_height = 0.8  # body height command of the walking policy [m]
    walker_default_joint_pos = WALKER_DEFAULT_JOINT_POS  # the walker's standing pose (its observations are relative to it)

    # layout (new at every reset). The tables stand end to end, joined by their short sides (each table's long side
    # along the line). The robot starts on the C-shaped band around the cube table that opens towards the tray table
    # (in front of the cube table, behind it or beyond its far end), facing the cube table.
    cube_table_height = (0.72, 0.90)  # table top heights [m] (the cube table: below about 0.72 m the right gripper
    # cannot reach down to the cube from the standing walker pose)
    tray_table_height = (0.60, 0.90)
    table_gap_min = 0.2  # free gap between the two tables' short sides [m], drawn from [min, max]
    table_gap_max = 1.5
    robot_distance_min = 1.0  # robot start to the nearest point of the cube table [m], drawn from [min, max]
    robot_distance_max = 5.0
    heading_noise = math.radians(15.0)  # start heading +- this from facing the cube table center
    table_margin = 0.10  # cube and tray stay this far inside the table edges [m]
    # the cube lies near one of the cube table's free edges (the long sides and the far end; not the end towards the
    # tray table), its center this far in from that edge [m], drawn from [min, max]: within the right gripper's reach
    # from a robot standing at the edge (reach check: the grasp point reaches
    # 0.29-0.38 m ahead of the pelvis, the table edge stays about 0.17 m ahead of it)
    cube_edge_depth = (0.05, 0.10)
    # the tray lies near one of the tray table's free edges (the long sides and the far end), its center its outer radius
    # plus this gap in from that edge [m], drawn from [min, max]
    tray_edge_gap = (0.0, 0.03)
    tray_variants = 24  # tray shapes (one per environment): rectangular floor, inner size and wall height below
    tray_inner_range = (0.12, 0.24)
    tray_wall_height_range = (0.04, 0.12)
    tray_variant_seed = 0
    item_sizes = tuple(s for s, _, _ in item_variants())  # cube edge length of each cube variant [m]
    arm_joint_pos_noise = 0.1  # +- around the default arm joint positions [rad] (as in the pick-and-place task)
    cameras: tuple = ()  # calibrated cameras in the scene (the camera variant sets them; see camera_images())

    # task (as in the table pick-and-place task)
    lift_height = 0.12
    carry_height = 0.18
    fall_height = 0.1
    grasp_distance = 0.10
    grasp_min_lift = 0.01
    grasp_closed_cmd = 0.5
    pickup_radius = 0.10
    grasp_hold_time = 0.10
    success_hold_time = 0.25
    terminate_on_success = True
    settle_speed = 0.10
    settle_angular_speed = 1.0
    wall_clearance = 0.015
    placement_anywhere_in_tray = True
    substep_release_check = False
    success_requires_return = False
    observe_layout = True
    table_height_range = 0.0  # (the base task's table; this task sets both table heights itself)
    item_pos_noise = 0.0
    tray_pos_noise = 0.0
    tray_yaw_range = 0.0
    # picking only (the Pick variant): the robot starts standing at the cube table with the cube in reach (in the pelvis
    # frame: cube_ahead m ahead, cube_right m to the right, heading +- stance_yaw_noise from facing the cube's table
    # edge); success is a secure grasp holding the cube pick_lift_height above its rest height for pick_hold_time
    start_at_cube_table = False
    cube_ahead = (0.28, 0.42)
    cube_right = (-0.02, 0.22)
    stance_yaw_noise = math.radians(20.0)
    success_on_lift = False
    pick_lift_height = 0.10
    pick_hold_time = 0.5
    # the robot falls: base height below this [m] or the body tilted more than fall_tilt [rad]
    fall_base_height = 0.45
    fall_tilt = math.radians(50.0)

    # baseline reward (the pipeline replaces it)
    rew_reach = 1.0
    rew_grasp = 1.0
    rew_lift = 3.0
    rew_transport = 5.0
    rew_transport_fine = 5.0
    rew_in_box = 10.0
    rew_released = 5.0
    rew_placed = 50.0
    rew_success = 100.0
    rew_wall_contact = -1.0
    rew_push = -0.5
    rew_action_rate = -0.01
    rew_joint_vel = -0.001
    rew_fall = -50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.tray = _tray_cfg(
            self.tray_variants, self.tray_inner_range, self.tray_wall_height_range, self.tray_variant_seed
        )


# ---------------------------------------------------------------------------------------------------------------
# cameras (for student policies): the head and wrist cameras of the real G1-D, calibrated by Adam Basl (2026-10-27,
# cameras/g1d_cameras_20261027.json; method and validation in his calibration notes)


CAMERA_CALIBRATION = os.path.join(TASK_DIR, "cameras", "g1d_cameras_20261027.json")
# G1-D URDF (unitree_ros, g1_d_with_dex1_1.urdf): {side}_dex1_base_link is {side}_wrist_yaw_link moved 0.0415 m along x,
# not turned. This model's {side}_hand_base_link is turned 90 deg about z, so the wrist cameras are mounted on the
# wrist yaw link instead; torso and arm kinematics of both models agree (checked: 0.0 mm, 0.01 deg).
DEX1_BASE_IN_WRIST_YAW = (0.0415, 0.0, 0.0)


def _calibrated_cameras() -> list[tuple[str, str, sim_utils.PinholeCameraCfg, tuple, tuple, dict]]:
    """(name, parent link, render spawn config, position, USD orientation (x, y, z, w), calibration) of each camera:
    Adam's centred square-pixel pinhole that covers every ray of the real lens (``isaac_render``) at the calibrated
    pose; :class:`~cameras.camera_lens.LensWarp` turns the renders into the real 640 x 480 images."""
    import json

    with open(CAMERA_CALIBRATION) as f:
        cameras = json.load(f)["cameras"]
    out = []
    for name, parent, shift in (
        ("head", "torso_link", (0.0, 0.0, 0.0)),
        ("left_wrist", "left_wrist_yaw_link", DEX1_BASE_IN_WRIST_YAW),
        ("right_wrist", "right_wrist_yaw_link", DEX1_BASE_IN_WRIST_YAW),
    ):
        c = cameras[name]
        render = c["isaac_render"]
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            render["intrinsic_matrix"], render["width"], render["height"], clipping_range=(0.01, 20.0), focal_length=1.0
        )
        w, x, y, z = c["pose"]["quat_wxyz_opengl"]  # the USD camera prim's orientation
        pos = tuple(p + s for p, s in zip(c["pose"]["position_m"], shift))
        out.append((name, parent, spawn, pos, (x, y, z, w), c))
    return out


def spawn_robot_with_cameras(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the robot USD, then the calibrated camera prims under its links. In a scene with per-environment variants
    a camera sensor with its own spawner under the robot clashes with the robot's clone template, so the cameras are
    part of the robot: they are cloned with it, and the camera sensors (spawn=None) use these prims."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from isaaclab.sim.utils import find_matching_prim_paths

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    for robot_path in find_matching_prim_paths(prim_path):
        for name, parent, spawn, pos, rot, _ in _calibrated_cameras():
            spawn.func(f"{robot_path}/{parent}/{name}_camera", spawn, translation=pos, orientation=rot)
    return prim


@configclass
class RobotWithCamerasCfg(sim_utils.UsdFileCfg):
    """The robot USD with the calibrated cameras under its links."""

    func = spawn_robot_with_cameras


def calibrated_camera_cfgs() -> dict[str, TiledCameraCfg]:
    """Camera sensors on the camera prims that :func:`spawn_robot_with_cameras` creates."""
    cfgs = {}
    for name, parent, _, _, _, c in _calibrated_cameras():
        render = c["isaac_render"]
        cfgs[name] = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + parent + "/" + name + "_camera",
            update_period=0.0,
            width=render["width"],
            height=render["height"],
            data_types=["rgb"],
            spawn=None,
        )
    return cfgs


@configclass
class G1Dex1TwoTablesCameraSceneCfg(G1Dex1TwoTablesSceneCfg):
    """The scene with the robot's head and wrist cameras."""

    head_camera: TiledCameraCfg = None
    left_wrist_camera: TiledCameraCfg = None
    right_wrist_camera: TiledCameraCfg = None

    def __post_init__(self):
        super().__post_init__()
        spawn = self.robot.spawn
        self.robot = self.robot.replace(spawn=RobotWithCamerasCfg(**{k: getattr(spawn, k) for k in spawn.__dataclass_fields__ if k != "func"}))
        cfgs = calibrated_camera_cfgs()
        self.head_camera = cfgs["head"]
        self.left_wrist_camera = cfgs["left_wrist"]
        self.right_wrist_camera = cfgs["right_wrist"]


@configclass
class G1Dex1TwoTablesCameraEnvCfg(G1Dex1TwoTablesEnvCfg):
    """The task with the calibrated head and wrist cameras (needs --enable_cameras). The policy observations are
    unchanged; the environment's ``camera_images()`` gives the images as the real cameras see them (640 x 480 RGB)."""

    scene: G1Dex1TwoTablesCameraSceneCfg = G1Dex1TwoTablesCameraSceneCfg(
        num_envs=64, env_spacing=16.0, replicate_physics=True
    )
    cameras = ("head", "left_wrist", "right_wrist")


@configclass
class G1Dex1TwoTablesPickEnvCfg(G1Dex1TwoTablesEnvCfg):
    """Picking only: the robot starts standing at the cube table (cube in reach, arms in the task's start pose) and has
    to grasp the cube with the right gripper and lift it; the teacher for the picking part of the full task."""

    episode_length_s = 8.0
    start_at_cube_table = True
    success_on_lift = True
    pick_lift_height = 0.15  # above the full task's valid-lift height (0.12 m)
