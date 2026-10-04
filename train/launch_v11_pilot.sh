#!/bin/bash
# v1.1 PILOT (2026-09-25): v1 + DAPO dynamic sampling (verl V1 ReplayBuffer, algorithm.filter_groups
# on the canonical reward): groups whose rollouts all score the same carry zero GRPO gradient and are evicted
# and refilled, so every optimizer batch is made of informative groups. Reward, estimand, gates unchanged.
# Paired control = seed 2 of v1: same batch 16 / mini 8, n=8, T=0.8, data seed 2, LR 1e-5, KL 0.01.
# Pilot only: not one of the five replicate seeds.
set -uo pipefail
B=${CLAIMBLIND_ROOT:-.}
cd $B
DAPO="algorithm.filter_groups.enable=True algorithm.filter_groups.metric=reward"
S=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 1:30:00 -J rl_grpo_v11_smoke \
  --export="ALL,SMOKE=1,EXP=smoke_v11_dapo,SEED=2,VERL_EXTRA=$DAPO" code/rl_verl_grpo.sbatch) || exit 1
echo "SMOKE=$S"
F=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 24:00:00 -J rl_grpo_v11s2 \
  --dependency=afterok:$S --kill-on-invalid-dep=yes \
  --export="ALL,EXP=v11_dapo_seed2,SEED=2,TRAIN_BS=16,MINI_BS=8,ROLL_N=8,TOTAL_STEPS=50,SAVE_FREQ=5,TEST_FREQ=10,VERL_EXTRA=$DAPO" \
  code/rl_verl_grpo.sbatch) || exit 1
echo "FULL=$F"
squeue -u "$USER" -o "%.8i %.18j %.10P %.8T %.8M %R"
