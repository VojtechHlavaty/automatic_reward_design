"""An open-top tray (floor and four walls) spawned as one kinematic rigid body.

The five parts are collision shapes under one rigid body, so a tray is a single asset: with a
:class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg` of several :class:`TrayCfg` (different inner sizes and
wall heights), each environment gets one whole tray, chosen by the scene's clone plan like the cube sizes.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd


def spawn_tray(
    prim_path: str,
    cfg: TrayCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the tray at ``prim_path`` (a regex path spawns it at every match).

    USD and the simulation utilities are imported here, not at module import: the task config imports this module
    before the simulator starts, and USD must not be loaded earlier.
    """
    from isaaclab.sim.utils import clone

    return clone(_spawn_tray)(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)


def _spawn_tray(
    prim_path: str,
    cfg: TrayCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the tray: an Xform with the rigid body at the bottom center of the tray, the parts as children.

    Args:
        prim_path: The prim path of the tray.
        cfg: The tray configuration.
        translation: Translation of the tray w.r.t. its parent prim.
        orientation: Orientation (x, y, z, w) of the tray w.r.t. its parent prim.
        **kwargs: Ignored (cloning options of the multi-asset spawner).

    Returns:
        The tray prim.
    """
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import create_prim, get_current_stage

    stage = get_current_stage()
    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    ix, iy = cfg.inner_size
    t, h = cfg.wall_thickness, cfg.wall_height
    parts = {
        "floor": ((ix + 2 * t, iy + 2 * t, t), (0.0, 0.0, t / 2)),
        "wall_px": ((t, iy + 2 * t, h), (ix / 2 + t / 2, 0.0, h / 2)),
        "wall_nx": ((t, iy + 2 * t, h), (-ix / 2 - t / 2, 0.0, h / 2)),
        "wall_py": ((ix, t, h), (0.0, iy / 2 + t / 2, h / 2)),
        "wall_ny": ((ix, t, h), (0.0, -iy / 2 - t / 2, h / 2)),
    }
    for name, (size, offset) in parts.items():
        part = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=cfg.visual_material,
            physics_material=cfg.physics_material,
        )
        part.func(f"{prim_path}/{name}", part, translation=offset)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    return stage.GetPrimAtPath(prim_path)


@configclass
class TrayCfg(sim_utils.RigidObjectSpawnerCfg):
    """An open-top tray with a rectangular floor; the root is at the bottom center of the floor."""

    func = spawn_tray

    inner_size: tuple[float, float] = (0.19, 0.19)
    """Inside dimensions between the walls along x and y [m]."""

    wall_height: float = 0.08
    """Height of the walls above the bottom of the floor [m]."""

    wall_thickness: float = 0.01
    """Thickness of the walls and the floor [m]."""

    visual_material: sim_utils.VisualMaterialCfg | None = None
    """Visual material of every part (None: no shader)."""

    physics_material: sim_utils.RigidBodyMaterialCfg | None = None
    """Physics material of every part."""


def tray_variants(count: int, inner_range: tuple[float, float], height_range: tuple[float, float],
                  seed: int = 0) -> list[tuple[float, float, float]]:
    """``count`` tray shapes (inner x [m], inner y [m], wall height [m]), drawn uniformly and reproducibly."""
    rng = random.Random(seed)
    return [
        (round(rng.uniform(*inner_range), 3), round(rng.uniform(*inner_range), 3), round(rng.uniform(*height_range), 3))
        for _ in range(count)
    ]
