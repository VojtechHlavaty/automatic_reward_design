# Trajectory suite (first recording, 3 Nov 2026)

Recorded with `python stl_check/run_suite.py --gpu 1 --num_envs 4 --out_dir stl_check/suite`: 4 episodes per kind,
5 s each, seeds 1-5 in this order.

| File | Kind | Task | Episodes | ρ (specs/lift_cube.py) |
|---|---|---|---|---|
| success.pt | scripted IK: above the cube, down, close, lift to the goal, hold | Isaac-Lift-Cube-Franka-IK-Abs-v0 | 4 | +0.307 .. +0.310 |
| hover.pt | hack (a): open gripper parked 7 cm beside the cube | Isaac-Lift-Cube-Franka-IK-Abs-v0 | 4 | -10.86 .. -4.92 |
| push.pt | hack (b): closed fingers sweep the cube along the table | Isaac-Lift-Cube-Franka-IK-Abs-v0 | 4 | -7.12 .. -5.43 |
| lift_drop.pt | hack (c): lifted, then dropped 0.3-0.7 s into the lift | Isaac-Lift-Cube-Franka-IK-Abs-v0 | 4 | -8.57 .. -3.61 |
| random.pt | uniform random joint-position actions | Isaac-Lift-Cube-Franka-v0 | 4 | -10.57 .. -6.03 |

`results.md` is the spec validation of this suite. Every episode is on the right side. The rtamt cross-check gives
the same robustness on 15 of these episodes.
