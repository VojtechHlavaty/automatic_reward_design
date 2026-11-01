#!/usr/bin/env bash
# Train the picking teacher (G1-Dex1-Two-Tables-Pick-v0, hand-written reward teachers/pick_reward.py) with the
# pipeline's training wrapper: exploration floor and squashed gripper output, hard-case and progress starts.
# No entropy bonus (it pins the noise at max_std); the noise floor keeps exploring instead.
# Usage: GPU=1 ITER=2000 teachers/train_pick.sh   (logs under outputs/pick_teacher/)
RD=/home/hlavavo7/automatic_reward_design/reward_design
cd "$RD/outputs/pick_teacher" || exit 1
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=${GPU:-1} RD_TENSOR_LOGS=1 RD_BATCHED_STATS=1 RD_SUCCESS_FLAG=succeeded
export RD_EXPLORATION='{"min_std": 0.3, "min_std_final": 0.1, "max_std": 1.0, "min_std_per_action": {"10": 0.8}, "squash": {"10": 1.5}}'
export RD_HARD_STARTS='{"prob": 0.25, "slots": 4, "max_tries": 3, "progress_prob": 0.3, "progress_commit_steps": 10, "adapter": "g1_dex1_two_tables", "success_flag": "succeeded", "progress_flags": ["jaws_on_cube", "secure_grasp", "cube_controlled"]}'
exec /home/hlavavo7/envs/isaacLab/bin/python -u "$RD/isaaclab_backend/train.py" --task G1-Dex1-Two-Tables-Pick-v0 \
    --reward_file "$RD/teachers/pick_reward.py" --isaaclab_root /home/hlavavo7/IsaacLab --run_name ${RUN_NAME:-pick_teacher} \
    --max_iterations ${ITER:-2000} --num_envs 4096 --headless \
    agent.algorithm.entropy_coef=0.0 agent.actor.distribution_cfg.init_std=0.6 "$@"
