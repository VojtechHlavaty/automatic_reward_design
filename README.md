# Automatic reward design with LLMs

A pipeline in which large language models (LLMs) design reward functions, as multi-stage curricula, for
reinforcement learning in [IsaacLab](https://github.com/isaac-sim/IsaacLab). The LLM writes each stage as a YAML
file of reward terms and training settings. The pipeline checks the stage for errors and reward hacking, trains it
with PPO (RSL-RL), evaluates the result and feeds what it learned back into the next design iteration.

The repository includes a humanoid manipulation task to run it on: the Unitree G1 (29 DoF, fixed base) with Dex1-1
parallel grippers picks a cube from a table and places it into a tray, optionally returning the arm to its start pose
afterwards.

## How a run works

Each iteration:

1. **Retrieval.** A query agent searches a local vector database of earlier curricula, their outcomes and feedback.
   A selector agent picks the most useful one as an example, or none.
2. **Planning.** A planner agent proposes candidate curricula: 1–5 stages, each with an objective, a transition
   condition (a task progress flag and a target rate) and a training budget.
3. **Stage generation.** A stage agent writes each stage as YAML. The available pieces are reward terms built from
   registered task quantities, LLM-written *derived quantities* (small Python functions of the environment state),
   PPO settings and allowed randomization parameters.
4. **Checks before training.**
   - Schema validation: structure, references, types and budgets.
   - A live check of the derived quantities on real simulator states.
   - Automatic reward-ordering checks: the reward must not pay for standing still, or drop when the task advances.
   - An independent LLM review for reward hacking.

   Errors go back to the stage agent, which can fix them with a small patch.
5. **Training.** The candidates train in parallel on the given GPUs, stage after stage. A stage can be extended
   while its key metric still improves, and can start from the best earlier stage (rollback).
6. **Evaluation.**
   - Task success and the original task reward from the training logs.
   - A deterministic checkpoint evaluation with a breakdown of the success conditions.
   - An LLM audit of stages whose statistics suggest reward hacking.
   - Optionally, a vision-language model reviews rollout videos and recordings of failed episodes.
7. **Feedback.** A feedback agent analyses every candidate. The next iteration uses this feedback and the database.
   With `warm_start_across_iterations=True`, it also starts from the best policy so far; by default each candidate
   starts from scratch, or from `init_checkpoint` when one is supplied.

Every artefact is written to the run folder, so an interrupted run continues where it stopped (`resume=<run folder>`).

Other training aids:
- **Exploration control:** per-action noise floors, ceilings and output squashing.
- **Hard-case starts:** episodes restart from earlier failed start states.
- **Progress flags as observations:** the policy can observe which task steps it has already done.

## Repository layout

```
reward_design/              the pipeline (run it from this folder)
  pipeline.py               main program (Hydra)
  core.py                   LLM client, training launcher, log parsing, run resume
  review.py                 vision-language review of rollout videos and failed episodes
  rd_lib/                   stage schema, LLM agents, reward registry, task-adapter loader, derived quantities and
                            their checker, exploration control, hard-case starts, retrieval database
  isaaclab_backend/         processes that run inside IsaacLab: training with the generated reward, the live
                            checker, checkpoint evaluation, video and failure recording
  prompts/                  prompts of the LLM agents and of the vision-language review
  cfg/pipeline.yaml         pipeline settings
  cfg/env/                  task configurations (task description, success metric, progress flags, ...)
  cfg/llm/                  LLM configurations (local vLLM servers or OpenAI-compatible APIs)
  env_context/              environment source shown to the LLM (generated, see below)
tasks/
  g1_dex1_table/            IsaacLab task package: G1 + Dex1-1 robot configuration, table, cube and tray scenes
                            (tray variants: tray_spawner.py), task logic, PPO settings and the task
                            quantities (quantities.py)
scripts/run_pipeline.sh     starts a run
```

## Tasks

| Gym ID | Task |
|---|---|
| `G1-Dex1-Pick-Place-v0` | pick and place, fixed layout |
| `G1-Dex1-Pick-Place-Return-v0` | pick, place, then return the arm to its start pose |
| `G1-Dex1-Pick-Place-Return-Random-v0` | pick, place and return; cube size and tray pose vary |
| `G1-Dex1-Pick-Place-Random-v0` | pick and place; cube size and tray pose vary |
| `G1-Dex1-Pick-Place-Random-Tray-v0` | pick and place; table height and tray shape vary too |
| `...-Colored-v0` | the same tasks in a coloured scene, used for review videos |

The policy observes state only: joint positions and velocities, the gripper, and the cube and tray poses. The
corresponding pipeline configurations are in `reward_design/cfg/env/`. Coloured variants are registered for
`Return-Random`, `Random` and `Random-Tray`. The `Random-Tray` task has no dedicated pipeline configuration;
to use it, copy the random-task configuration and update its task description, geometry and video task.

## Requirements

- Linux with an NVIDIA RTX GPU (the video review renders with RTX).
- [Isaac Sim](https://developer.nvidia.com/isaac-sim) 6.0 and IsaacLab. Tested with IsaacLab commit
  `0603cb1710dcda13087665f63abf9ac483b63c05`, Python 3.12, PyTorch 2.10 (CUDA 12.8), `rsl-rl-lib` 5.0.1.
- The packages in `requirements.txt`, installed in the IsaacLab Python environment.
- An LLM, either of:
  - a local model served by [vLLM](https://github.com/vllm-project/vllm) through its OpenAI-compatible API
    (`cfg/llm/qwen3_8b.yaml`, `cfg/llm/qwen3.8_27b_fp8.yaml`);
  - a commercial OpenAI-compatible API (`cfg/llm/openai.yaml`, `cfg/llm/openrouter_haiku5.5.yaml`).

  The video review needs a model that accepts images.

## Installation

```bash
# Run these commands from the automatic_reward_design directory.
# Install IsaacLab first and activate its Python environment.
export PYTHON=/path/to/isaaclab/env/bin/python
"$PYTHON" -m pip install -r requirements.txt

# Where the IsaacLab repository is
export ISAACLAB_PATH=/path/to/IsaacLab

# Check and apply the supplied fix for nested asset cloning at the tested commit.
git -C "$ISAACLAB_PATH" apply --check "$PWD/patches/isaaclab_cloner_utils.patch"
git -C "$ISAACLAB_PATH" apply "$PWD/patches/isaaclab_cloner_utils.patch"
```

The patch fixes clone-source selection when an asset, such as a camera, is nested inside another planned asset.
If it is already applied, the forward check fails; `git apply --reverse --check` with the same patch path checks
that case. For a different IsaacLab revision, inspect whether the fix is still needed before applying it.
`PYTHON` is optional when `python` already points to the IsaacLab environment. No editable installation of this
repository is required: the launcher and backend import the bundled task packages directly.

The robot model (`tasks/g1_dex1_table/assets/g1_29dof_dex1_unitree`) is Unitree's G1 29-DoF with Dex1-1 grippers,
from Unitree's simulation assets. Keep its `configuration/` directory alongside
`g1_29dof_with_dex1_base_fix1.usd`: the USD uses those files as dependencies.

The retrieval encoder (`BAAI/bge-small-en-v1.5` by default) is downloaded from Hugging Face on first use and runs
on the CPU. Set `HF_HOME` to choose its cache directory; offline runs need the model cached in advance.

API keys are never stored in the repository. They are read from the environment variable named in the LLM
configuration (for example `OPENAI_API_KEY` or `OPENROUTER_API_KEY`), or from a file such as
`~/.config/reward_design/openrouter_key` (one line, `chmod 600`). Local vLLM servers need no key.

## Running

```bash
# local model: start the vLLM server first (OpenAI-compatible endpoint on localhost:8000), then
scripts/run_pipeline.sh g1_dex1_pick_place_return_random qwen3_8b "gpus=[0,1]" iteration=3 num_candidates=2

# commercial API, with the video review
scripts/run_pipeline.sh g1_dex1_pick_place_random openrouter_haiku5.5 "gpus=[0]" visual=True

# continue an interrupted run
scripts/run_pipeline.sh g1_dex1_pick_place_random openrouter_haiku5.5 resume=outputs/runs/<date_time>
```

Reserve separate GPUs for a local LLM and training. For example, if the server uses GPU 0, change the first
command to `"gpus=[1]"` on a two-GPU machine. Both local presets otherwise select GPUs 1–7; the commercial
preset selects GPUs 0–2. Override `gpus` for your machine and set `max_runs_per_gpu=1` initially to limit memory
use. Local model servers are managed separately; this repository includes no server launch scripts. The
configured `model` must match the name exposed by the server, and `llm_base_url` must point to its `/v1` endpoint.

For a short end-to-end trial, with the local server on GPU 0 and training on GPU 1:

```bash
scripts/run_pipeline.sh g1_dex1_pick_place_random qwen3_8b \
  "gpus=[1]" max_runs_per_gpu=1 iteration=1 num_candidates=1 \
  curriculum_iterations=100 min_stage_iterations=100 num_envs=256 \
  eval_num_envs=32 eval_episodes_per_env=1 stage_extensions=0 num_eval=0 visual=False
```

This still calls the LLM, checks generated stages in the simulator and trains a policy. It checks the setup;
the reduced budget is not intended to solve the task. The live checker also uses GPU memory.

All settings are Hydra overrides of `cfg/pipeline.yaml`. Common ones:

| Setting | Meaning |
|---|---|
| `iteration` | design iterations |
| `num_candidates`, `num_candidates_schedule` | curricula per iteration |
| `curriculum_iterations` | PPO iterations per curriculum |
| `num_envs` | parallel environments per training run |
| `gpus`, `max_runs_per_gpu` | GPU indices, and training runs per GPU |
| `init_checkpoint` | start from a pretrained policy instead of from scratch |
| `warm_start_across_iterations` | start new candidates from the best policy so far (default `False`) |
| `num_eval` | retrain the winning curriculum with new seeds after the search (default `1`; `0` skips) |
| `hacking_review`, `hacking_audit` | LLM reward-hacking checks |
| `visual`, `visual_failures` | vision-language review of videos and failed episodes |

Results:
- **Run folder:** `reward_design/outputs/runs/<date_time>/`. Per iteration it holds the plans, the stage YAMLs, the
  compiled rewards, the training logs, the evaluations and the feedback. `pipeline.log` summarises the run, including
  the LLM token use per role.
- **Checkpoints and TensorBoard logs:** in each training run's log folder. The generated reward's terms are logged
  under `Generated/`.

For example, a run contains:

```text
run_config.yaml                  saved configuration
pipeline_state.json              completed iterations and best curriculum/checkpoint
llm_usage.jsonl                  token use and timings per LLM request
iter0/
  retrieval.json                retrieved examples and selection
  plans.json                    proposed curricula
  cand0/
    stage0.yaml                 accepted stage specification
    stage0_attempts.json        generation attempts and validation feedback
    stage0_reward.py            generated Python wrapper loading the stage YAML
    stage0.txt                  training process output
    checkpoint_eval.json        deterministic checkpoint evaluation, when enabled
    feedback.json               feedback for the next iteration
final_eval/summary.json          results of retraining the best curriculum, when enabled
logs/rsl_rl/<experiment>/<run>/  checkpoints and TensorBoard events
```

Iteration, candidate and stage indices start at zero. The `best` entry in `pipeline_state.json` identifies the
selected stage YAMLs and checkpoint; keep the YAMLs with their generated Python wrappers. The shared retrieval
database lives in `reward_design/vdb/` by default and is created automatically. Set `vdb_path=/absolute/path`
to use a separate database for an experiment.

```bash
"$PYTHON" -m tensorboard.main --logdir reward_design/outputs/runs/<date_time>/logs
```

Resume paths in the launch script are relative to `reward_design/`, even when the script is called from the
repository root. Absolute paths also work. Resume restores the saved environment and LLM configuration and
reuses completed work; ordinary overrides such as `gpus` or `iteration` can still be changed. A stage interrupted
during training may need to run again.

To give feedback yourself, write a few sentences to `iter<i>/human_feedback.txt` in the run folder. Set
`human_feedback_wait=60` to wait up to a minute between iterations; otherwise a file already present is picked
up when the next iteration begins.

## Stage format

The planner allocates the curriculum budget; each stage defines its transition target, reward and training
settings. A small example of the format is:

```yaml
stage:
  name: reach_and_lift
  objective: Reach the cube and establish a valid lift.
  transition:
    flag: lifted_once
    threshold: 0.8
reward:
  scale_by_dt: true
  terms:
    - name: approach
      kind: shaping
      quantity: palm_to_cube_dist
      fn: exp_kernel
      params: {scale: 0.1}
      weight: 1.0
      forbid: [lifted_once]
    - name: lift_bonus
      kind: event
      quantity: lifted_once
      weight: 10.0
training:
  max_iterations: 100
  entropy_coef: 0.0
  learning_rate: 0.0003
```

This illustrates the schema, rather than a complete curriculum for placement. When the pipeline saves a stage it
adds `adapter: <task adapter>` (from the task configuration); a stage YAML written by hand needs that line too
before it can be compiled into a reward. `quantity` references the task's
adapter or a stage-defined derived quantity. `shaping` terms use bounded transforms, `penalty` terms have negative
weights, `event` terms pay when a flag becomes true, and `potential` terms reward changes in a potential.
`require` and `forbid` gate terms with task flags. Continuous shaping and penalties are scaled by the control
time step when `scale_by_dt` is true; events and potential terms are not.

The full schema is in `reward_design/rd_lib/schema.py`, the transforms are in `rd_lib/reward_registry.py`, and
the derived-quantity rules are in `rd_lib/derived.py`. Stages may also set allowed randomization parameters and
exploration settings. Checks can catch errors and suspicious incentives, but task success still needs to be
measured independently of the generated reward.

## Evaluation

`checkpoint_evaluation=True` evaluates each candidate's selected checkpoint with a deterministic policy. Its
`checkpoint_eval.json` reports episode success, success-condition rates and diagnostic statistics. This is
separate from `num_eval`, which trains the winning curriculum again with seeds `100`, `101`, and so on. Set
`num_eval=3` for three retraining seeds. By default those runs start from scratch, or from `init_checkpoint`;
`final_eval_from_best=True` instead continues from the best search checkpoint.

For a separate checkpoint evaluation, run from `reward_design/`:

```bash
"$PYTHON" -m isaaclab_backend.evaluate_checkpoint \
  --task G1-Dex1-Pick-Place-Random-v0 \
  --adapter g1_dex1_pick_place_return_random \
  --checkpoint /absolute/path/to/model_100.pt \
  --num_envs 256 --episodes_per_env 3 --headless
```

For pipeline checkpoints, first restore the policy settings from that run's `run_config.yaml`: set
`RD_EXPLORATION` to the JSON form of `env.exploration`, and, when `env.obs_progress_flags` is true, set
`RD_OBS_FLAGS` to a JSON object with `adapter` and `flags` (the ordered `env.progress_flags` list with
`env.success_flag` removed). The pipeline sets these automatically during its own evaluations. Omitting them
can change action squashing or cause an observation-size mismatch. Evaluation prints its result as JSON on a
line prefixed with `@@RD_EVAL@@`. Add `--stochastic` to evaluate sampled actions instead.

For video review, use `visual=True` with an image-capable endpoint. `visual_failures` controls the number of
failed episodes sent for review; `vlm_model` and `vlm_base_url` can select a separate vision model. Videos and
visual reports are stored with the run.

## Adding a task

1. **Task package:** add an IsaacLab task package to `tasks/` that registers its gym IDs when imported, like
   `tasks/g1_dex1_table/__init__.py`. Every package in `tasks/` is imported automatically.
2. **Task quantities:** add `tasks/<package>/quantities.py` with an adapter listing the quantities reward terms
   may use, with descriptions and progress flags, and register its factory in `ADAPTERS` (see
   `tasks/g1_dex1_table/quantities.py`; it must not import the task's environment module). Adapter names must be
   unique over all task packages. Each `Quantity` returns
   a tensor with one value per environment: a float for a scalar or a bool for a flag. Mark flags that stay true
   through the episode with `latched=True`. Optionally register success conditions in `FUNNELS` and a continuous
   success-hold diagnostic in `HOLDS` for checkpoint evaluation.
3. **Task configuration:** add `reward_design/cfg/env/<name>.yaml` with the task description, success metric,
   adapter, progress flags, exploration settings and allowed randomization. The existing files show every field.
4. **Environment context:** generate the source shown to the LLM:

   ```bash
   cd reward_design
   "$PYTHON" -m isaaclab_backend.context <gym-id> env_context/<env_name>_obs.py
   ```

   Set `env_name` in the task configuration to the same name. The generator removes the original reward from
   direct environments and reward configurations from manager-based environments. Regenerate the context when
   the task source changes, and check that its geometry and state descriptions agree with the task YAML.
5. **Success and observations:** implement the task metric independently of the generated reward. The bundled
   evaluator uses `env._succeeded` at episode reset to count successes; a new task must expose that flag or adapt
   the evaluator. Progress observations, hard-case starts and the live checker also need compatible task state;
   inspect their backend hooks when integrating a different environment type.
6. **Trial run:** start with one candidate and a short budget. Check that quantity values, success metrics,
   generated reward terms and checkpoint evaluation agree before increasing parallelism or training length.

## Troubleshooting

- **Task not found:** use the repository's backend entry points, which import `tasks/` automatically. Running
  IsaacLab's scripts directly does not register these task packages.
- **CUDA out of memory:** lower `num_envs`, `eval_num_envs` or `max_runs_per_gpu`, and leave room for the local
  LLM and live checker. Confirm that `gpus` contains only devices available on your machine.
- **LLM connection or authentication error:** check the configured endpoint, served model name and key variable.
  Provider presets are configuration examples; update them to models available through your account or server.
- **All stages fail validation:** read `stage*_attempts.json` and `pipeline.log`. Check the task description,
  adapter quantities and environment context for inconsistent names or geometry.
- **Missing robot geometry:** check the main USD and its referenced `configuration/` files are present at the
  path in `tasks/g1_dex1_table/env_cfg.py`.
