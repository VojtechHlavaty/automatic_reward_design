# STL check for LLM-generated rewards: Franka lift-cube

A deterministic Signal Temporal Logic (STL) check for one manipulation task, IsaacLab's
`Isaac-Lift-Cube-Franka-v0`. It scores recorded episodes with a hand-written spec (robustness ρ: > 0 the
task was done, < 0 it was not, the size says by how much) and tests whether LLM-generated rewards rank hacked
episodes above real successes.

## Status

| Part | State |
|---|---|
| Robustness code (`robustness.py`): predicates, ¬, ∧, ∨, F[a,b], G[a,b], batched in PyTorch | done; unit tests pass (`tests/`) |
| Predicate library (`predicates.py`): dist, height, grasp, gripper_open, contact_force, speed, tilt | done; tested on synthetic trajectories only |
| Spec (`specs/lift_cube.py`) with thresholds and reasons | done; one change after the first real suite: the contact-force limit 50 → 250 N (a normal grasp peaks at 20–134 N) |
| Recorder (`record_suite.py`): scripted success, hover, push, lift-then-drop; random; trained policy | done: first suite in `suite/` (4 episodes per kind), every episode on the right side of the spec |
| Reward scoring (`score_rewards.py`): replays the suite under each LLM reward | written, **not run yet** |
| Analysis (`analyze.py`): validation, Kendall τ, hacks above a success, LLM #2 verdict, results table | done; tested on a synthetic suite |
| rtamt cross-check (`check_rtamt.py`) | done: 15 synthetic and 15 recorded episodes, all equal to rtamt 0.4.10 |
| Pipeline setup for the task (`tasks/franka_lift/`, `reward_design/cfg/env/franka_lift.yaml`) | done; the LLM rewards for the results table come from pipeline runs on it |

## Layout

```
stl_check/
  predicates.py        read_state(env) (live) and signals(traj) (offline): the predicate signals, [episodes, steps]
  robustness.py        STL nodes (Pred, Not, And, Or, F, G), robustness, explain (which part failed), rtamt syntax
  specs/lift_cube.py   the hand-written spec, one documented threshold per predicate
  env_setup.py         the task as recorded and replayed (fixed seed, no observation noise, cube contact sensor)
  record_suite.py      records one kind of episode -> <kind>.pt
  score_rewards.py     replays a suite file under LLM rewards -> <kind>_returns.json
  analyze.py           spec validation and the results table -> results.md, results.json
  check_rtamt.py       our robustness against rtamt on a few episodes
  run_suite.py         all of it: record, score, analyze
  tests/               CPU tests of the robustness code and the spec
```

## The spec

Episode: 5 s at a 0.02 s control step (250 steps). Every window ends by 4.9 s.

```
phi_lift =
     F[0,2.5]( grasp > 0  &  F[0,1.5]( G[0,0.5]( grasp > 0  &  cube_height > 0.10 ) ) )     (1)
  &  F[0,4.4]( G[0,0.5]( grasp > 0  &  cube_goal_dist < 0.05 ) )                              (2)
  &  G[0,4.9]( cube_height > -0.01  &  cube_speed < 1.0  &  contact_force < 250 )             (3)
```

1. Grasp the cube within 2.5 s. Within 1.5 s after that, it is held at least 0.10 m high for 0.5 s straight.
2. Before 4.4 s, the cube is held within 0.05 m of the goal for 0.5 s straight.
3. The cube never falls off the table, never moves faster than 1 m/s, and is never pushed harder than 250 N.

Why these thresholds, and why they differ from the task's own success test, is written next to each predicate in
`specs/lift_cube.py`. In short:

- The task's "lifted" (cube center above 0.04 m) is only about 1.6 cm off the table.
- The task's success needs only one step at the goal, so a flung cube passes it.

**Time windows.** F[a,b] means "at some time between a and b seconds later"; G[a,b] means "at every time in
it". Nested windows count from the time the enclosing operator picked.

**Scales.** Every margin is divided by a scale before min/max. Distances are divided by their threshold, and
`grasp` is already normalized. Without scales, a predicate in newtons or in action units swamps every distance in
metres.

**Grasp.** The task has no finger contact sensor, so `grasp` combines three margins:
- fingertip center within 2 cm of the cube center;
- the gripper commanded closed;
- the fingers not fully shut, so something is between them.

**Contact force** comes from a contact sensor on the cube, which `env_setup.py` adds to the task. It only reports
forces.

## Running it

From the repository root, with IsaacLab's Python:

```bash
# CPU tests (no simulator)
python -m pytest stl_check/tests -q

# the whole check on GPU 0: record the suite, score the rewards, write stl_check/suite/results.md
python stl_check/run_suite.py --gpu 0 --rewards <stage YAMLs> --outcomes "<policy outcome>" ...

# single steps
python stl_check/record_suite.py --kind hover --num_envs 16 --seed 3 --out stl_check/suite/hover.pt --headless
python stl_check/score_rewards.py --suite stl_check/suite/hover.pt --rewards <stage.yaml> \
    --out stl_check/suite/hover_returns.json --headless
python -m stl_check.analyze --suite_dir stl_check/suite --rewards <stage.yaml> --out stl_check/suite/results.md
PYTHONPATH=~/envs/rtamt_pkgs python -m stl_check.check_rtamt --suite_dir stl_check/suite --episodes 3
```

The LLM rewards are the stage YAMLs of pipeline runs on the task:

```bash
cd reward_design && python pipeline.py env=franka_lift llm=openrouter_haiku5.5 gpus=[0,1] visual=True iteration=2
```

Each stage YAML stores LLM #2's verdict under `hacking_review`; `analyze.py` puts it in the table.

**rtamt** needs `antlr4-python3-runtime==4.7`, while Hydra and OmegaConf (the pipeline) need 4.9. Do not install
it into the IsaacLab environment. Install it into a separate folder that only the cross-check puts on its path:

```bash
python -m pip install --target ~/envs/rtamt_pkgs --no-deps rtamt==0.4.10 "antlr4-python3-runtime==4.7"
```

## Scripted episodes run in the IK-Abs variant

The scripted kinds (success, hover, push, lift_drop) run in `Isaac-Lift-Cube-Franka-IK-Abs-v0`. It is the same
scene, cube, goal, rewards and success. Only the arm differs: stiff joint gains, and the action is the target
end-effector pose. In the joint-position task the pipeline trains on, the arm's soft gains let it lag and sag under
gravity, and scripted position control missed the cube by 4–6 cm.

Random and policy episodes use the joint-position task. Each suite file records its task, and the replay uses it.

Rewards with action terms (action rate) see different actions in the two variants. Compare their returns across
kinds with that in mind.

## Trajectory suite

`stl_check/suite/<kind>.pt`, one file per kind, `--num_envs` episodes each (16 by default):

| Kind | What it is | Expected ρ |
|---|---|---|
| success | scripted (IK-Abs task): above the cube, down, close, lift to the goal, hold | > 0 |
| policy | a trained checkpoint (`--checkpoint`), deterministic | > 0 for its successful episodes |
| random | uniform random actions | < 0 |
| hover | hack (a): the open gripper parks 7 cm beside the cube | < 0 |
| push | hack (b): closed fingers sweep the cube along the table, no grasp | < 0 |
| lift_drop | hack (c): as success, but the gripper opens 0.3–0.7 s after the lift starts | < 0 |

Each file holds:
- `traj`: the raw state per step, {name: [episodes, steps, ...]};
- `actions`: [episodes, steps, 8];
- `valid`: [episodes, steps], False after an episode ended early;
- `seed`, `num_envs`, `dt`, `kind`.

The recorded actions replay to the same states with the same seed and number of environments. `score_rewards.py`
reports the largest cube-position deviation of the replay, to check this.

The scripted success can fail in some layouts (IK limits, a slipping grasp). Its episodes are kept and judged by
ρ like any other. Check `results.md` before reading the success set as all successes.

## Results table (filled by analyze.py)

| Reward | Kendall τ vs ρ | Hacks ranked above a success | LLM #2 verdict | Policy outcome |
|---|---|---|---|---|
| reward_1 | | | | |
| reward_2 | | | | |
| reward_3 | | | | |

## Questions (to answer after the run)

1. Did the spec separate successes from random and hack episodes on the first try? What did you change?
2. Which predicates or thresholds were hard to choose?
3. Did any reward rank a hack above a real success? Did LLM #2 approve that reward?
4. Would this check have saved us a training run? Which one?

## Not in scope this week

- Connecting the check to the training loop, or replacing LLM #2 in the code.
- Using robustness as the training reward.
- Searching automatically for hack trajectories.
- Other tasks.
