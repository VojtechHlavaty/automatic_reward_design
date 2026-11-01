#!/usr/bin/env bash
# Wait for the picking teacher's training to end, evaluate its last checkpoint without noise and start the two-table
# pipeline with that policy as the initial policy of every curriculum.
# Usage: teachers/picker_then_pipeline.sh <picker train pid> <picker run folder>   (log: outputs/two_tables_run4.log)
RD=/home/hlavavo7/automatic_reward_design/reward_design
PY=/home/hlavavo7/envs/isaacLab/bin/python
cd "$RD" || exit 1
while kill -0 "$1" 2>/dev/null; do sleep 60; done
CKPT=$(realpath "$(ls "$2"/model_*.pt | sort -V | tail -1)")  # absolute: the pipeline runs in its own run folder
echo "picker checkpoint: $CKPT"
export OMNI_KIT_ACCEPT_EULA=YES
export RD_EXPLORATION='{"min_std": 0.3, "min_std_final": 0.1, "max_std": 1.0, "min_std_per_action": {"10": 0.8}, "squash": {"10": 1.5}}'
CUDA_VISIBLE_DEVICES=1 timeout 1500 $PY -m isaaclab_backend.evaluate_checkpoint --task G1-Dex1-Two-Tables-Pick-v0 \
    --checkpoint "$CKPT" --adapter g1_dex1_two_tables --num_envs 256 --episodes_per_env 3 --headless \
    < /dev/null > outputs/pick_teacher/eval_final.log 2>&1
RATE=$(grep "@@RD_EVAL@@" outputs/pick_teacher/eval_final.log | $PY -c "
import json, sys; print(round(100 * json.loads(sys.stdin.read().split('@@RD_EVAL@@', 1)[1])['success_rate']))")
echo "picker success ${RATE}%"
unset RD_EXPLORATION
# start only on empty GPUs 0 and 1 (no compute process on either)
while [ -n "$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
DESC="The initial policy is a picking teacher trained with a hand-written reward in a pick-only version of this task:"\
" every episode started with the robot already standing at the cube table (pelvis 0.28-0.42 m behind and up to 0.22 m"\
" beside the cube, facing the table within +-20 deg), and success was this task's valid lift followed by a secure grasp held"\
" at least 0.15 m above the cube's rest height and clear of the tray rim for 0.5 s."\
" Without exploration noise it does this in about ${RATE} percent of those episodes. It has never"\
" walked from far, carried the cube to the tray table or released it; its left arm was only kept near its start pose,"\
" not trained to aim the wrist camera. Plan the curriculum to keep the grasp and lift and add the walk to the cube"\
" table, the camera aiming, the carry and the release."
exec $PY pipeline.py env=g1_dex1_two_tables llm=openrouter_haiku5.5 gpus=[0,1] visual=True \
    init_checkpoint="$CKPT" "env.init_policy_description=\"$DESC\""
