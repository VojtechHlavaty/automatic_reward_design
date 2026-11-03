# STL check: Franka lift-cube

## Spec validation

| Episode type | Episodes | rho min | rho max | on the wrong side |
|---|---|---|---|---|
| hover | 4 | -10.858 | -4.916 | 0 |
| lift_drop | 4 | -8.570 | -3.606 | 0 |
| push | 4 | -7.118 | -5.430 | 0 |
| random | 4 | -10.565 | -6.032 | 0 |
| success | 4 | +0.307 | +0.310 | 0 |

## Rewards

| Reward | Kendall tau vs rho | Hacks ranked above a success | LLM #2 verdict | Policy outcome |
|---|---|---|---|---|
| reward_1 | -0.41 (p = 0.011) | 12 (hover 4, push 4, lift_drop 4) | major: The arm reaches the cube with the gripper open and pushes or drags the cube along the table until the end effector is within 0.03 m of the cube center | stage 0 of the trial run: at the cube 44% (0% success) |
| reward_2 | +0.21 (p = 0.21) | 0 (none) | major: Close the fingers on the cube while it stays on the table (or drag it along the table) with the closing command held, the end effector within 0.03 m of the cube | stage 1: grasped 0% |
| reward_3 | +0.18 (p = 0.29) | 0 (none) | major: Get the cube above 0.04 m without a secure grasp, for example by tipping or balancing it on an edge, jamming it against a finger or the gripper body, or pushing | stage 2: held_lifted 0% |
| reward_4 | -0.26 (p = 0.11) | 0 (none) | major: Grip the cube on the table (or drag it along the table) with the fingers stopped at about 0.048 m and the hand at grasp height, so secure_grip holds without any | stage 3: 0% success |

Rewards:

- reward_1: `/home/hlavavo7/automatic_reward_design/reward_design/outputs/runs/franka_try1/iter0/cand0/stage0.yaml`
- reward_2: `/home/hlavavo7/automatic_reward_design/reward_design/outputs/runs/franka_try1/iter0/cand0/stage1.yaml`
- reward_3: `/home/hlavavo7/automatic_reward_design/reward_design/outputs/runs/franka_try1/iter0/cand0/stage2.yaml`
- reward_4: `/home/hlavavo7/automatic_reward_design/reward_design/outputs/runs/franka_try1/iter0/cand0/stage3.yaml`
