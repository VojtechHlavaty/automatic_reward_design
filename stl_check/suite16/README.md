# Trajectory suite (16 episodes per kind, 8–9 Nov 2026)

The suite of the results table. Each kind is 16 episodes of 5 s (250 steps at 0.02 s). ρ is the robustness under
the final spec (`specs/lift_cube.py`).

| File | Kind | Task | Seed | ρ min .. max |
|---|---|---|---|---|
| success.pt | scripted IK: above the cube, down, close, lift the cube to the goal, hold | Isaac-Lift-Cube-Franka-IK-Abs-v0 | 7 | +0.308 .. +0.310 |
| policy.pt | trained policy, deterministic: `franka_progress` iteration 1, candidate 1, stage 0, `model_2344.pt` (98.8% success in the pipeline's 768-episode evaluation) | Isaac-Lift-Cube-Franka-v0 | 6 | +0.093 .. +0.310 |
| random.pt | uniform random joint-position actions | Isaac-Lift-Cube-Franka-v0 | 7 | -11.54 .. -4.29 |
| hover.pt | hack (a): the open gripper parks 7 cm beside the cube | Isaac-Lift-Cube-Franka-IK-Abs-v0 | 7 | -11.54 .. -4.03 |
| push.pt | hack (b): the closed fingers sweep the cube along the table, no grasp | Isaac-Lift-Cube-Franka-IK-Abs-v0 | 7 | -11.54 .. -3.67 |
| lift_drop.pt | hack (c): as success, but the gripper opens 0.3–0.7 s after the lift starts | Isaac-Lift-Cube-Franka-IK-Abs-v0 | 7 | -8.84 .. -3.58 |

Every episode is on the right side of the spec: 32 successes with ρ > 0 and 64 random or hack episodes with
ρ < 0.

The policy checkpoint is
`reward_design/outputs/runs/franka_progress/logs/rsl_rl/franka_lift/2026-11-05_16-09-00_franka_lift_iter1_cand1_stage0/model_2344.pt`
in `~/automatic_reward_design`.

## Other files

- `<kind>_returns.json` and `<kind>_returns_progress.json`: the return of every scored reward on every episode
  (`score_rewards.py`), both as a plain sum and discounted. The first file holds the 16 rewards of `franka_full`,
  scored with this repository's `reward_design`. The second holds the 14 rewards of `franka_progress`, scored with
  `~/automatic_reward_design/reward_design`, which has the level pay those rewards were trained with.
- Replay check: every file replays its recorded episodes exactly, with a cube-position deviation of 0.0 m.
- `results.md` / `results.json`: the spec validation and the table for all 30 rewards, ranked by the discounted
  return.
- `results_undiscounted.md`: the same table ranked by the plain sum of the rewards.
- `check_rtamt.log`: our robustness against rtamt 0.4.10 on 3 episodes of each kind (18 in total), with 0 mismatches.
- `record_*.log`, `score_*.log`: the logs of each step.

## Reproduce

```bash
R=~/automatic_reward_design/reward_design/outputs/runs
CK=$R/franka_progress/logs/rsl_rl/franka_lift/2026-11-05_16-09-00_franka_lift_iter1_cand1_stage0/model_2344.pt
D=stl_check/suite16
for k in success random hover push lift_drop; do
  python stl_check/record_suite.py --kind $k --num_envs 16 --seed 7 --out $D/$k.pt --headless
done
python stl_check/record_suite.py --kind policy --checkpoint $CK --num_envs 16 --seed 6 --out $D/policy.pt --headless
# score the franka_full rewards (this repository's reward_design), then the franka_progress rewards with the
# level-pay code, then analyze all 30
python stl_check/run_suite.py --gpu 0 --out_dir $D --checkpoint $CK --skip_record \
    --rewards $R/franka_full/iter*/cand*/stage?.yaml
python stl_check/run_suite.py --gpu 0 --out_dir $D --checkpoint $CK --skip_record \
    --reward_design ~/automatic_reward_design/reward_design --group progress \
    --rewards $R/franka_progress/iter*/cand*/stage?.yaml \
    --analyze_rewards $R/franka_full/iter*/cand*/stage?.yaml $R/franka_progress/iter*/cand*/stage?.yaml
python -m stl_check.analyze --undiscounted --suite_dir $D --out $D/results_undiscounted.md \
    --rewards $R/franka_full/iter*/cand*/stage?.yaml $R/franka_progress/iter*/cand*/stage?.yaml
```
