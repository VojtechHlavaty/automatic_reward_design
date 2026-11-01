"""Unitree G1 (29 DoF, walking) with Dex1-1 parallel grippers: carry a cube from one table into a tray on another.

Importing this package registers the tasks with gymnasium."""

import gymnasium as gym

from . import agents

# walk to the cube table, pick the cube and place it into the tray on the tray table
gym.register(
    id="G1-Dex1-Two-Tables-v0",
    entry_point=f"{__name__}.env:G1Dex1TwoTablesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TwoTablesEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TwoTablesPPORunnerCfg",
    },
)

# the same task with the calibrated head and wrist cameras of the real G1-D (student policies; needs --enable_cameras)
gym.register(
    id="G1-Dex1-Two-Tables-Camera-v0",
    entry_point=f"{__name__}.env:G1Dex1TwoTablesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TwoTablesCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TwoTablesPPORunnerCfg",
    },
)

# picking only: start standing at the cube table, grasp the cube with the right gripper and lift it (a teacher for the
# picking part of the full task)
gym.register(
    id="G1-Dex1-Two-Tables-Pick-v0",
    entry_point=f"{__name__}.env:G1Dex1TwoTablesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TwoTablesPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TwoTablesPPORunnerCfg",
    },
)
