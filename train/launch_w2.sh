#!/bin/bash
# SECOND WRITER (protocol declared in the project log 2026-09-26 19:25): Qwen2.5-VL-7B-Instruct writer, the same
# frozen Qwen3-VL-8B judge, the dose recipe unchanged, seed 101 everywhere, 40 steps. A smoke (2 steps, same 3-GPU
# layout) gates the run. LoRA restricted to the language model (exclude '.*visual.*').
set -uo pipefail
B=${CLAIMBLIND_ROOT:-.}
cd $B
RP=$B/code/rl_verl_reward_logged.py
for E in smoke_w2_qwen25_s101 w2_qwen25_s101; do
  for D in runs/verl/$E runs/rollouts/$E runs/reward_logs/$E; do [ -e $D ] && { echo "EXISTS $D"; exit 2; }; done
done
exp() {  # EXP -> export string
  local S=101
  echo "ALL,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,JUDGE_MODEL=Qwen/Qwen3-VL-8B-Instruct,EXP=$1,ACTOR_LR=1e-4,SEED=$S,DATA=data/rep_s$S,TRAIN_BS=8,MINI_BS=4,ROLL_N=16,ROLL_TEMP=1.0,REWARD_PATH=$RP,REWARD_LOG_DIR=$B/runs/reward_logs/$1,KEEP_LAST=0,VERL_EXTRA=algorithm.filter_groups.enable=True algorithm.filter_groups.metric=reward data.shuffle=False trainer.rollout_data_dir=$B/runs/rollouts/$1 actor_rollout_ref.actor.fsdp_config.seed=$S actor_rollout_ref.actor.data_loader_seed=$S actor_rollout_ref.ref.fsdp_config.seed=$S actor_rollout_ref.rollout.seed=$S actor_rollout_ref.model.exclude_modules=.*visual.*"
}
S=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 2:00:00 -J rl_w2_smoke \
  --export="$(exp smoke_w2_qwen25_s101),TOTAL_STEPS=2,SAVE_FREQ=2,TEST_FREQ=2" code/rl_verl_grpo.sbatch) || exit 1
echo "SMOKE=$S"
J=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 24:00:00 -J rl_w2_s101 \
  --dependency=afterok:$S --kill-on-invalid-dep=yes \
  --export="$(exp w2_qwen25_s101),TOTAL_STEPS=40,SAVE_FREQ=10,TEST_FREQ=20" code/rl_verl_grpo.sbatch) || exit 1
echo "RUN=$J"
