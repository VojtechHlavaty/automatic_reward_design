"""Configurations of the G1 + Dex1-1 table pick-and-place tasks: robot, scene and task parameters.

The robot is Unitree's G1 29-DoF with Dex1-1 parallel grippers (Unitree's USD model, ``assets/g1_29dof_dex1_unitree``)
on a fixed base; the policy drives the waist yaw, the right arm and the right gripper. The scene is a table with a cube
and an open tray; variants vary the cube size, the tray pose and shape and the table height.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Gripper joint positions of the Dex1-1 jaws [m] (both jaws get the same command, as on the real gripper)
GRIPPER_OPEN = -0.02
GRIPPER_CLOSED = 0.024

# Unitree G1 29-DoF with Dex1-1 grippers; the Dex3 hand actuators are replaced by the gripper actuators of Unitree's
# simulation (unitree_sim_isaaclab: G129_CFG_WITH_DEX1_BASE_FIX)
G1_DEX1_CFG = G1_29DOF_CFG.replace(
    spawn=G1_29DOF_CFG.spawn.replace(
        usd_path=os.path.join(ASSET_DIR, "g1_29dof_dex1_unitree", "g1_29dof_with_dex1_base_fix1.usd")
    ),
    init_state=G1_29DOF_CFG.init_state.replace(
        joint_pos={**G1_29DOF_CFG.init_state.joint_pos, ".*_hand_Joint[12]_1": GRIPPER_OPEN},
    ),
    actuators={
        **{name: cfg for name, cfg in G1_29DOF_CFG.actuators.items() if name != "hands"},
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_Joint[12]_1"],
            stiffness=800.0,
            damping=3.0,
        ),
    },
)


# scene layout in each environment's local frame (robot at the origin, facing +x)
TABLE_H = 0.70
TABLE_DX = -0.10  # shift of the table and everything on it towards the robot [m] (was 0: table center at x = 0.55)
TABLE_POS = (0.55 + TABLE_DX, 0.0, TABLE_H / 2)
ITEM_SIZE = 0.05
ITEM_POS = (0.40 + TABLE_DX, -0.15, TABLE_H + ITEM_SIZE / 2)
BOX_POS = (0.45 + TABLE_DX, 0.20, TABLE_H)  # bottom center of the tray
BOX_IN, BOX_WALL_H, BOX_T = 0.20, 0.08, 0.01  # inner size, wall height, wall thickness
ITEM_COLOR = (0.10, 0.45, 0.12)  # light-green cube in the colored scenes


def _static_cuboid(size, pos) -> RigidObjectCfg:
    # kinematic rigid body: does not move, and is tracked by the physics views instead of a
    # separate frame view (avoids a scene-creation hang seen on some systems)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/placeholder",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # no shader (the colored scenes add simple materials)
            visual_material=None,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _cube_spawn(size: float, color=None) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=(size, size, size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
        # Keep the same shader-free default as the table and tray (colors only for the colored scenes).
        visual_material=None if color is None else sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
    )


def _item_cfg(sizes, color=None) -> RigidObjectCfg:
    """The cube; with several sizes, each environment gets one of them at scene creation (the environment reads
    which one from the scene's clone plan)."""
    sizes = tuple(sizes)
    spawn = (
        _cube_spawn(sizes[0], color)
        if len(sizes) == 1
        else sim_utils.MultiAssetSpawnerCfg(assets_cfg=[_cube_spawn(s, color) for s in sizes], random_choice=False)
    )
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Item",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(ITEM_POS[0], ITEM_POS[1], TABLE_H + max(sizes) / 2)),
    )


@configclass
class G1Dex1TablePickPlaceSceneCfg(InteractiveSceneCfg):
    # ground plane and light are spawned in the environment's _setup_scene (as in the cartpole task)

    robot: ArticulationCfg = G1_DEX1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # quaternion is (x, y, z, w): identity -> robot faces +x, towards the table
        init_state=G1_DEX1_CFG.init_state.replace(pos=(0.0, 0.0, 0.78), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    table = _static_cuboid((0.6, 1.0, TABLE_H), TABLE_POS)

    # open-top box = floor + 4 walls
    box_floor = _static_cuboid((BOX_IN, BOX_IN, BOX_T), (BOX_POS[0], BOX_POS[1], TABLE_H + BOX_T / 2))
    box_wall_px = _static_cuboid(
        (BOX_T, BOX_IN, BOX_WALL_H),
        (BOX_POS[0] + BOX_IN / 2, BOX_POS[1], TABLE_H + BOX_WALL_H / 2),
    )
    box_wall_nx = _static_cuboid(
        (BOX_T, BOX_IN, BOX_WALL_H),
        (BOX_POS[0] - BOX_IN / 2, BOX_POS[1], TABLE_H + BOX_WALL_H / 2),
    )
    box_wall_py = _static_cuboid(
        (BOX_IN, BOX_T, BOX_WALL_H),
        (BOX_POS[0], BOX_POS[1] + BOX_IN / 2, TABLE_H + BOX_WALL_H / 2),
    )
    box_wall_ny = _static_cuboid(
        (BOX_IN, BOX_T, BOX_WALL_H),
        (BOX_POS[0], BOX_POS[1] - BOX_IN / 2, TABLE_H + BOX_WALL_H / 2),
    )

    item = _item_cfg((ITEM_SIZE,))  # the env config's item_sizes replaces it when the cube size varies

    def __post_init__(self):
        super().__post_init__()
        # give each static part its own prim path
        for name in [
            "table",
            "box_floor",
            "box_wall_px",
            "box_wall_nx",
            "box_wall_py",
            "box_wall_ny",
        ]:
            if getattr(self, name) is not None:  # a varied-tray layout replaces the five parts by one tray asset
                getattr(self, name).prim_path = "{ENV_REGEX_NS}/" + name
        # fix the robot base so it doesn't need to balance
        self.robot.spawn.articulation_props.fix_root_link = True
        # The unused arm hangs relaxed at the side of the torso (the G1's natural standing pose), slightly bent.
        self.robot.init_state.joint_pos = dict(self.robot.init_state.joint_pos)
        self.robot.init_state.joint_pos.update(
            {
                "left_shoulder_pitch_joint": 0.15,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 1.25,
                # right arm: raised in front of the body, gripper pointing forward
                "right_shoulder_pitch_joint": -0.512,
                "right_shoulder_roll_joint": -0.300,
                "right_shoulder_yaw_joint": -0.004,
                "right_elbow_joint": 0.140,
                "right_wrist_roll_joint": 0.398,
                "right_wrist_pitch_joint": 0.640,
                "right_wrist_yaw_joint": 0.627,
            }
        )
        # lock the legs and the waist roll/pitch: very stiff PD holding the default pose (legs are not used)
        self.robot.actuators = dict(self.robot.actuators)
        for group in ("legs", "feet", "waist"):
            self.robot.actuators.pop(group, None)
        self.robot.actuators["locked_body"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "waist_roll_joint", "waist_pitch_joint"],
            stiffness=1.0e5,
            damping=1.0e3,
            effort_limit_sim=1.0e5,
        )
        # the torso can turn: waist yaw is controlled by the policy (G1 waist yaw motor: 88 N m, 32 rad/s)
        self.robot.actuators["waist_yaw"] = ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            stiffness=300.0,
            damping=10.0,
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
        )


@configclass
class G1Dex1TablePickPlaceColoredSceneCfg(G1Dex1TablePickPlaceSceneCfg):
    """The scene with colored objects (for videos): brown table, red tray, light-green cube."""

    def __post_init__(self):
        super().__post_init__()
        # brown table, red tray, light-green cube (simple preview-surface materials)
        colors = {
            "table": (0.22, 0.12, 0.05),
            "box_floor": (0.80, 0.08, 0.08),
            "box_wall_px": (0.80, 0.08, 0.08),
            "box_wall_nx": (0.80, 0.08, 0.08),
            "box_wall_py": (0.80, 0.08, 0.08),
            "box_wall_ny": (0.80, 0.08, 0.08),
        }
        for name, color in colors.items():
            if getattr(self, name) is not None:
                getattr(self, name).spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)
        self.item = _item_cfg((ITEM_SIZE,), color=ITEM_COLOR)


@configclass
class G1Dex1TablePickPlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.0
    decimation = 2
    action_space = 9  # waist yaw + 7 right-arm joint target deltas, 1 gripper open/close
    observation_space = 43  # 8 joint positions, 8 joint velocities, 2 gripper joints, 16 task values, 9 actions
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    light_intensity = 3000.0  # dome light intensity

    # scene
    scene: G1Dex1TablePickPlaceSceneCfg = G1Dex1TablePickPlaceSceneCfg(
        num_envs=1024, env_spacing=3.0, replicate_physics=True
    )

    # joints / bodies
    # joints driven by the policy's joint-target deltas: the torso (waist yaw) and the right arm
    arm_joint_names = ["waist_yaw_joint", "right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"]
    hand_joint_names = ["right_hand_Joint1_1", "right_hand_Joint2_1"]
    # grasp reference point: the midpoint between the two fingertips (the task code calls it the palm)
    palm_body_names = ["right_hand_Link1_3", "right_hand_Link2_3"]
    # gripper joint targets [m], matched by substring of the joint name
    hand_open_pos = {"Joint": GRIPPER_OPEN}
    hand_closed_pos = {"Joint": GRIPPER_CLOSED}

    # actions
    arm_action_scale = 1.5  # max arm joint target speed [rad/s]

    # reset randomization
    item_pos_noise = 0.05  # +- range around ITEM_POS in x and y [m]
    tray_pos_noise = 0.0  # +- range around BOX_POS in x and y [m], new at every reset
    tray_yaw_range = 0.0  # +- range of the tray yaw about its center [rad], new at every reset
    item_sizes = (ITEM_SIZE,)  # cube edge lengths [m]; several = spread over the environments (fixed per environment)
    table_height_range = 0.0  # +- range of the table height around TABLE_H [m], new at every reset (0 = fixed)
    table_height_center = TABLE_H  # centre of that range [m] (used only when table_height_range > 0)
    # check the release condition at every physics substep too (needed for long control steps)
    substep_release_check = False
    # varied trays (0 = the fixed square tray): this many tray shapes, each a rectangular floor of inner size
    # tray_inner_range along x and y and walls of tray_wall_height_range, spread over the environments (fixed per
    # environment, like the cube sizes); set by _random_tray_and_table
    tray_variants = 0
    tray_inner_range = (0.10, 0.22)
    tray_wall_height_range = (0.05, 0.15)
    tray_variant_seed = 0
    # observe the task layout (tray center in the robot frame, tray yaw, cube size): needed when they vary
    observe_layout = False
    arm_joint_pos_noise = 0.1  # +- range around the default arm joint positions [rad]

    # task
    lift_height = 0.12  # item center rise above rest required before transport [m]
    carry_height = 0.18  # item center height above the table at the transport target [m]
    fall_height = 0.1  # item this far below the table top terminates the episode [m]
    grasp_distance = 0.10  # palm-to-item distance below which the hand counts as at the item [m]
    grasp_min_lift = 0.01  # item rise above rest that counts as off the table [m]
    grasp_closed_cmd = 0.5  # hand command above which the hand counts as closed (0 open .. 1 closed)
    pickup_radius = 0.10  # maximum horizontal distance from nominal pickup for lift credit [m]
    grasp_hold_time = 0.10  # continuous unsupported grasp before lift credit [s]
    success_hold_time = 0.25  # released item must remain settled in the tray [s]
    terminate_on_success = True  # end the episode at success (False: keep running, e.g. to return to the start pose)
    settle_speed = 0.10  # maximum item linear speed for success [m/s]
    settle_angular_speed = 1.0  # maximum item angular speed for success [rad/s]
    wall_clearance = 0.015  # extra clearance above the rim and inside the opening [m]
    # placement counts anywhere in the tray (also against a wall) once the release started over the opening; False =
    # a stricter rule (release cancelled when the cube comes near a wall, and "in the tray" needs the cube's
    # circumscribed sphere clear of the walls)
    placement_anywhere_in_tray = True
    # success with return: after the placement (the success above, latched) the arm must come back to its start pose
    success_requires_return = False
    return_joint_tolerance = 0.3  # norm of the 8 arm joint errors from the default pose [rad]
    return_max_joint_speed = 0.5  # norm of the 8 arm joint speeds [rad/s]
    return_hold_time = 0.25  # the returned pose must be held this long [s]

    # reward scales
    rew_reach = 1.0
    rew_grasp = 0.5
    rew_lift = 3.0  # best-height progress and one-time valid-lift bonus
    rew_transport = 4.0  # best-distance progress while grasped above the rim
    rew_transport_fine = 4.0
    rew_in_box = 10.0  # one-time grasped arrival above the opening
    rew_released = 5.0  # one-time release above the opening
    rew_action_rate = -1.0e-3
    rew_joint_vel = -1.0e-4
    rew_wall_contact = -5.0
    rew_push = -2.0
    rew_success = 100.0  # one-time terminal bonus; not multiplied by the time step
    rew_placed = 50.0  # one-time placement bonus when success_requires_return (the placement is not yet success)


@configclass
class G1Dex1TablePickPlaceReturnEnvCfg(G1Dex1TablePickPlaceEnvCfg):
    """Pick, place and return: success needs the arm back at its start pose after the cube is placed in the tray."""

    success_requires_return = True


# ---------------------------------------------------------------------------------------------------------------
# varied layout: cube size per environment, tray position and yaw new at every reset


ITEM_SIZES_RANDOM = (0.04, 0.045, 0.05, 0.055, 0.06)  # cube edge lengths spread over the environments [m]


def _random_layout(cfg):
    cfg.item_sizes = ITEM_SIZES_RANDOM
    cfg.scene.item = _item_cfg(
        ITEM_SIZES_RANDOM, color=ITEM_COLOR if isinstance(cfg.scene, G1Dex1TablePickPlaceColoredSceneCfg) else None
    )
    cfg.tray_pos_noise = 0.05
    cfg.tray_yaw_range = 0.7854  # +-45 deg (the square tray repeats every 90 deg)
    cfg.observe_layout = True
    cfg.observation_space = 49  # + tray center (3), tray yaw sin/cos (2), cube size (1)


@configclass
class G1Dex1TablePickPlaceReturnRandomEnvCfg(G1Dex1TablePickPlaceReturnEnvCfg):
    """Pick, place and return with a cube of varying size (4-6 cm) and a tray at a varying position and yaw."""

    def __post_init__(self):
        super().__post_init__()
        _random_layout(self)


@configclass
class G1Dex1TablePickPlaceReturnRandomColoredEnvCfg(G1Dex1TablePickPlaceReturnRandomEnvCfg):
    """The varied-layout task in the colored scene (videos: the objects are easy to tell apart)."""

    scene: G1Dex1TablePickPlaceColoredSceneCfg = G1Dex1TablePickPlaceColoredSceneCfg(
        num_envs=16, env_spacing=3.0, replicate_physics=True
    )
    light_intensity = 1000.0  # dimmer light so the object colors are not washed out


@configclass
class G1Dex1TablePickPlaceRandomEnvCfg(G1Dex1TablePickPlaceReturnRandomEnvCfg):
    """The varied-layout task without the return: success is the placement (0.25 s hold) and ends the episode."""

    success_requires_return = False


@configclass
class G1Dex1TablePickPlaceRandomColoredEnvCfg(G1Dex1TablePickPlaceReturnRandomColoredEnvCfg):
    """The varied-layout placing task (no return) in the colored scene, for videos."""

    success_requires_return = False


# ---------------------------------------------------------------------------------------------------------------
# varied table height and tray shape (on top of the varied layout)


TRAY_VARIANTS = 32  # tray shapes spread over the environments


def _random_tray_and_table(cfg):
    """Table height +-5 cm (new at every reset) and one of TRAY_VARIANTS trays per environment: rectangular floor
    10-22 cm inside along x and y, walls 5-15 cm high."""
    from isaaclab.assets import RigidObjectCfg as _RigidObjectCfg

    from .tray_spawner import TrayCfg, tray_variants

    cfg.table_height_range = 0.05
    cfg.tray_variants = TRAY_VARIANTS
    colored = isinstance(cfg.scene, G1Dex1TablePickPlaceColoredSceneCfg)
    shapes = tray_variants(cfg.tray_variants, cfg.tray_inner_range, cfg.tray_wall_height_range, cfg.tray_variant_seed)
    trays = [
        TrayCfg(
            inner_size=(ix, iy),
            wall_height=h,
            wall_thickness=BOX_T,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.08, 0.08), roughness=0.6)
            if colored
            else None,
        )
        for ix, iy, h in shapes
    ]
    cfg.scene.tray = _RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tray",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=trays, random_choice=False),
        init_state=_RigidObjectCfg.InitialStateCfg(pos=BOX_POS),
    )
    for name in ("box_floor", "box_wall_px", "box_wall_nx", "box_wall_py", "box_wall_ny"):
        setattr(cfg.scene, name, None)


@configclass
class G1Dex1TablePickPlaceRandomTrayEnvCfg(G1Dex1TablePickPlaceRandomEnvCfg):
    """The varied-layout placing task (no return) with a varied table height (+-5 cm) and varied trays (rectangular,
    10-22 cm inside, walls 5-15 cm)."""

    def __post_init__(self):
        super().__post_init__()
        _random_tray_and_table(self)


@configclass
class G1Dex1TablePickPlaceRandomTrayColoredEnvCfg(G1Dex1TablePickPlaceRandomColoredEnvCfg):
    """The varied-tray placing task in the colored scene, for videos."""

    def __post_init__(self):
        super().__post_init__()
        _random_tray_and_table(self)


