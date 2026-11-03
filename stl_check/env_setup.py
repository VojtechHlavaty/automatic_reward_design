"""The Franka lift-cube environment as the suite records and replays it (import after the simulator has started)."""

TASK = "Isaac-Lift-Cube-Franka-v0"


def add_cube_contact_sensor(env_cfg):
    """A contact sensor on the cube (the net force on it): the task has none. It only reports forces."""
    from isaaclab.sensors import ContactSensorCfg

    env_cfg.scene.object.spawn.activate_contact_sensors = True
    env_cfg.scene.cube_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object", update_period=0.0)


def make_env(task, num_envs, seed, device):
    """The task with a fixed seed, no observation noise and the cube contact sensor (the same in recording and
    scoring, so that recorded actions replay to the recorded states)."""
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg.seed = seed
    env_cfg.observations.policy.enable_corruption = False
    add_cube_contact_sensor(env_cfg)
    return gym.make(task, cfg=env_cfg)
