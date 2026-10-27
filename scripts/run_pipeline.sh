#!/usr/bin/env bash
# Start a reward-design run.
#
#   scripts/run_pipeline.sh <env config> <llm config> [Hydra overrides ...]
#
#   <env config>  a file of reward_design/cfg/env without .yaml, e.g. g1_dex1_pick_place_return_random
#   <llm config>  a file of reward_design/cfg/llm without .yaml, e.g. qwen3_8b (local vLLM server) or openai
#
# Examples:
#   scripts/run_pipeline.sh g1_dex1_pick_place_return_random qwen3_8b "gpus=[0,1]" iteration=3
#   scripts/run_pipeline.sh g1_dex1_pick_place_random openrouter_haiku5.5 "gpus=[0]" visual=True
#
# Environment: ISAACLAB_PATH (IsaacLab repository, required); PYTHON (the IsaacLab Python, default: python).
# The run folder is reward_design/outputs/runs/<date_time>; resume a run with resume=<run folder>.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${ISAACLAB_PATH:?set ISAACLAB_PATH to the IsaacLab repository}"
PYTHON=${PYTHON:-python}
if [ $# -lt 2 ]; then sed -n '2,15p' "$0"; exit 1; fi
ENV=$1; LLM=$2; shift 2
cd "$REPO/reward_design"
exec "$PYTHON" pipeline.py "env=$ENV" "llm=$LLM" "$@"
