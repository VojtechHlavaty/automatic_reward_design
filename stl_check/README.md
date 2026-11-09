# STL check for LLM-generated rewards: Franka lift-cube

A deterministic Signal Temporal Logic (STL) check for one manipulation task, IsaacLab's
`Isaac-Lift-Cube-Franka-v0`. It scores recorded episodes with a hand-written spec (robustness ρ: > 0 the
task was done, < 0 it was not, the size says by how much) and tests whether LLM-generated rewards rank hacked
episodes above real successes.

## Status

| Part | State |
|---|---|
| Robustness code (`robustness.py`): predicates, ¬, ∧, ∨, F[a,b], G[a,b], batched in PyTorch | done; unit tests pass (`tests/`) |
| Predicate library (`predicates.py`): dist, height, grasp, gripper_open, contact_force, speed, tilt | done; tested on synthetic trajectories and on the 96 recorded episodes |
| Spec (`specs/lift_cube.py`) with thresholds and reasons | done; two changes after real episodes: the contact-force limit 50 → 250 N, and the cube speed limit (1 m/s) dropped (see the answers) |
| Recorder (`record_suite.py`): scripted success, hover, push, lift-then-drop; random; trained policy | done: the suite of the results is `suite16/` (16 episodes per kind, 96 in all); `suite/` is the first 4-episode trial |
| Reward scoring (`score_rewards.py`): replays the suite under each LLM reward | done: the replay reproduces every recorded episode exactly (deviation 0.0 m); plain and discounted returns; a run's level pay and gamma as in training |
| Analysis (`analyze.py`): validation, Kendall τ, hacks above a success, LLM #2 verdict, results table | done: `suite16/results.md`, 30 rewards |
| rtamt cross-check (`check_rtamt.py`) | done: 15 synthetic episodes and 33 recorded ones (15 of `suite/`, 18 of `suite16/`), all equal to rtamt 0.4.10 |
| Pipeline setup for the task (`tasks/franka_lift/`, `reward_design/cfg/env/franka_lift.yaml`) | done; the LLM rewards come from two pipeline runs on it |

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
  &  G[0,4.9]( cube_height > -0.01  &  contact_force < 250 )                                  (3)
```

1. Grasp the cube within 2.5 s. Within 1.5 s after that, it is held at least 0.10 m high for 0.5 s straight.
2. Before 4.4 s, the cube is held within 0.05 m of the goal for 0.5 s straight.
3. The cube never falls off the table and is never pushed harder than 250 N.

Why these thresholds, and why they differ from the task's own success test, is written next to each predicate in
`specs/lift_cube.py`. In short:

- The task's "lifted" (cube center above 0.04 m) is only about 1.6 cm off the table.
- The task's success needs only one step at the goal, so a flung cube passes it. Clause 2 needs the cube held at
  the goal for 0.5 s.

**Stage specs.** The pipeline trains a curriculum (reach, grasp, lift, transport), and a stage's reward only pays
for that stage's goal. Each stage reward is therefore also judged against the spec of its own stage
(`STAGE_SPECS`): the stage's goal held for 0.5 s within 2.5 s, plus clause 3. The transport stage uses the whole
spec.

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

`stl_check/suite16/<kind>.pt` (the suite of the results; `suite/` is the first 4-episode trial), one file per kind, `--num_envs` episodes each (16 by default):

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

## Results

Suite: `suite16/` (96 episodes; see its README). Rewards: all 30 stage rewards of two pipeline runs on
`franka_lift`:

- `franka_full` (4 Nov, rewards 1–16) ended at 0% success.
- `franka_progress` (5 Nov, rewards 17–30) adds the framework's level pay and reached 98.8%.

`suite16/results.md` has the full table, with LLM #2's finding for each reward.

### Spec validation

All 96 episodes are on the right side of the final spec:

| Kind | ρ min | ρ max |
|---|---|---|
| success (scripted) | +0.308 | +0.310 |
| policy (trained, 98.8%) | +0.093 | +0.310 |
| random | -11.54 | -4.29 |
| hover | -11.54 | -4.03 |
| push | -11.54 | -3.67 |
| lift_drop | -8.84 | -3.58 |

### Results table

These are the full-task rewards that led to bad policies, and the two rewards behind the trained policies.

- Returns are discounted with the stage's PPO γ (0.98). That is the quantity PPO maximises.
- τ is Kendall's τ between a reward's return and the robustness ρ under the spec of the reward's own stage.
- "Hacks ranked above a success" counts the hack episodes (hover, push, lift_drop; 48 in all) whose return beats the
  lowest-return success episode (scripted or policy, ρ > 0).

| Reward | Kendall τ vs ρ | Hacks ranked above a success | LLM #2 verdict | Policy outcome |
|---|---|---|---|---|
| reward_8: `franka_full` iteration 0, candidate 1, transport_to_goal | -0.04 (p = 0.54) | 48 of 48 (hover 16, push 16, lift_drop 16) | minor: "This matches the objective" | at_goal 0%; run 0% success |
| reward_4: `franka_full` iteration 0, candidate 0, transport_to_goal | +0.27 (p = 8e-05) | 44 (hover 12, push 16, lift_drop 16) | major (an early-termination exploit) | at_goal 0%; run 0% success |
| reward_16: `franka_full` iteration 1, candidate 1, transport_to_goal | +0.50 (p = 9e-13) | 23 (hover 7, lift_drop 16) | major ("the release penalty is still escapable") | at_goal 0%; run 0% success |
| reward_20: `franka_progress` iteration 0, candidate 0, transport_to_goal | +0.52 (p = 6e-14) | 0 | minor | 97% success at the end of training (model_1945.pt, 97.4% when evaluated) |
| reward_28: `franka_progress` iteration 1, candidate 1, grasp_lift_stabilize (lift spec) | +0.72 (p = 3e-23) | 0 | major | 98.8% success when evaluated (model_2344.pt, the `policy` episodes) |

**Over all 30 rewards**, a reward is flagged when some episode that fails its stage spec earns more than an episode
that passes it.
- Flagged: 15 of 30. That is 14 of the 16 `franka_full` rewards (every iteration-0 reward), and 1 of the 14
  `franka_progress` rewards (reward_27: 3 lift_drop episodes).
- LLM #2's verdict barely tracks the flag:
  - flagged rewards: 10 major, 3 minor, 2 not re-reviewed;
  - unflagged rewards: 8 major, 7 minor.

**The plain sum is misleading for potential terms.** The pipeline's potential terms pay γΦ(s') − Φ(s) per step.
Summed without discounting, they charge (1 − γ)·Φ for every step spent at a high potential, so a policy that reaches
the goal early and stays there is penalised. Ranked by the plain sum (`suite16/results_undiscounted.md`), reward_28
gets τ = +0.15 and 16 hacks above a success, and reward_30 gets 48. Discounted, both rank every hack below every
success.

## Answers

**1. Did the spec separate successes from random and hack episodes on the first try? What did you change?**

No; there were two changes, both on the success side. Random and hack episodes were below 0 from the start, and are
now at ρ ≤ −3.58.

- **Contact force.** The first guess, 50 N, rejected real grasps: the binary gripper snaps shut at 20–134 N. Raised
  to 250 N, so the limit only catches crushing or slamming.
- **Cube speed.** The limit `cube_speed < 1 m/s` rejected all 16 episodes of the trained 98.8% policy, which lifts
  the held cube at up to 2.4 m/s, and one scripted success. The limit separated no hack (every hack already fails
  the grasp-and-lift clause by at least 0.7). A thrown cube already fails clause 2, which needs it held at the goal
  for 0.5 s. The limit was dropped.

The scripted success controller also needed fixes before its episodes were real successes:
- it now brings the cube, not the fingertips, to the goal;
- it has a looser arrival tolerance for the IK arm.

After the changes, all 96 episodes are on the right side. The smallest success margin is +0.09 (a policy episode).

**2. Which predicates or thresholds were hard to choose?**

- **Grasp.** The task has no finger contact sensor, so `grasp` combines three margins: the fingertip centre within
  2 cm of the cube, the gripper commanded closed, and the fingers not fully shut.
- **Contact force.** Pushes (59–83 N) and grasps (20–134 N) overlap, so no force limit separates them. The grasp and
  height clauses do that job.
- **Cube speed.** The threshold was picked from scripted episodes, and a trained policy moved faster.
- **Lift height.** We use 0.10 m instead of the task's 0.04 m. The task's lift test is only about 2 cm off the table,
  which a tipped cube nearly reaches.
- **Time bounds.** The bounds (grasp within 2.5 s, hold 0.5 s) came from the scripted timing. The trained policy
  grasps after 0.44–0.54 s, so they are loose.
- **Scales.** Margins in metres, newtons and normalised units have to be scaled before min and max, or the newtons
  dominate.

**3. Did any reward rank a hack above a real success? Did LLM #2 approve that reward?**

Yes.
- **reward_8:** a full-task reward. It ranks all 48 hack episodes above the worst success (τ = −0.04), and LLM #2
  rated it "minor", saying the best behaviour "matches the objective". It trained a policy that never reached the
  goal.
- **reward_5:** a reach-stage reward that LLM #2 also rated "minor". It ranks every hover, push and random episode
  above real reaches.
- **The other direction:** LLM #2 rated reward_28 "major", and it trained the best policy (98.8%). It ranks no hack
  above a success.

In this sample, LLM #2's severity did not predict either the STL flag or the policy outcome.

**4. Would this check have saved us a training run? Which one?**

Yes: iteration 0 of `franka_full` (4 Nov).
- All 8 of its stage rewards are flagged, both full-task rewards among them.
- Both curricula trained for about 34 minutes on two GPUs and reached 0% success.
- Iteration 1 of the same run was also mostly flagged (6 of 8 rewards), and the run ended at 0%.

The check would have let `franka_progress` through: 13 of 14 rewards pass, and the run reached 98.8%.

It is not sufficient on its own. Some rewards that pass still trained failing stages (reward_12, reward_24,
reward_30), because a reward that ranks the suite correctly can still be hard to learn. The check catches rewards
that pay for the wrong behaviour, not rewards that are too hard to learn.

**Caveats.**
- The hack episodes are hand-scripted, one controller per hack. A reward can still pay for a hack the suite lacks.
- The scripted kinds run in the IK-Abs variant of the task, so action-rate terms see different actions there than
  in the joint-position task.
- The `franka_progress` returns include the framework's level pay, which is part of the reward those policies
  trained on.

## Not in scope this week

- Connecting the check to the training loop, or replacing LLM #2 in the code.
- Using robustness as the training reward.
- Searching automatically for hack trajectories.
- Other tasks.
