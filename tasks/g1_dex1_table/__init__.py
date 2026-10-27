"""Unitree G1 (29 DoF, fixed base) with Dex1-1 parallel grippers: table pick-and-place tasks.

Importing this package registers the tasks with gymnasium."""

import gymnasium as gym

from . import agents

# pick the cube and place it into the tray (fixed layout)
gym.register(
    id="G1-Dex1-Pick-Place-v0",
    entry_point=f"{__name__}.env:G1Dex1TablePickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TablePickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TablePickPlacePPORunnerCfg",
    },
)

# pick, place, then return the arm to its start pose
gym.register(
    id="G1-Dex1-Pick-Place-Return-v0",
    entry_point=f"{__name__}.env:G1Dex1TablePickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TablePickPlaceReturnEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TablePickPlacePPORunnerCfg",
    },
)

# pick, place and return with varied cube sizes and tray poses
gym.register(
    id="G1-Dex1-Pick-Place-Return-Random-v0",
    entry_point=f"{__name__}.env:G1Dex1TablePickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TablePickPlaceReturnRandomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TablePickPlacePPORunnerCfg",
    },
)

# the same in a colored scene (for review videos)
gym.register(
    id="G1-Dex1-Pick-Place-Return-Random-Colored-v0",
    entry_point=f"{__name__}.env:G1Dex1TablePickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TablePickPlaceReturnRandomColoredEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TablePickPlacePPORunnerCfg",
    },
)

# pick and place with varied cube sizes and tray poses (success = placement)
gym.register(
    id="G1-Dex1-Pick-Place-Random-v0",
    entry_point=f"{__name__}.env:G1Dex1TablePickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TablePickPlaceRandomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TablePickPlacePPORunnerCfg",
    },
)

# the same in a colored scene (for review videos)
gym.register(
    id="G1-Dex1-Pick-Place-Random-Colored-v0",
    entry_point=f"{__name__}.env:G1Dex1TablePickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TablePickPlaceRandomColoredEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TablePickPlacePPORunnerCfg",
    },
)

# pick and place with varied table heights and tray shapes
gym.register(
    id="G1-Dex1-Pick-Place-Random-Tray-v0",
    entry_point=f"{__name__}.env:G1Dex1TablePickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TablePickPlaceRandomTrayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TablePickPlacePPORunnerCfg",
    },
)

# the same in a colored scene (for review videos)
gym.register(
    id="G1-Dex1-Pick-Place-Random-Tray-Colored-v0",
    entry_point=f"{__name__}.env:G1Dex1TablePickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1Dex1TablePickPlaceRandomTrayColoredEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dex1TablePickPlacePPORunnerCfg",
    },
)
