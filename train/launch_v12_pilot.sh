#!/bin/bash
# v1.2 PILOT (2026-09-25): content-only reward + exploration. Pre-registered ADDITION; estimand,
# evaluation and gates unchanged; not one of the five replicate seeds.
#   reward   code/rl_verl_reward_lenient.py: the record is repaired to canonical form without inventing
#            content, so schema validity carries no reward (v1 GRPO spent its gradient on format:
#            results/gvar step 0 vs 35); records valid under v1 get byte-identical canonical bytes
#   sampling DAPO dynamic sampling (filter_groups on reward), temperature 1.0, 16 rollouts per image
#            (base writer at T=1.0: content-informative groups 50 percent at k=64, pass@64 0.72)
#   batch    8 images x 16 rollouts = 128 sequences and 2 optimizer updates per step, as v1 (16 x 8)
# LR 1e-5, KL 0.01, LoRA r16, data seed 2: same as v1 seed 2 and the v1.1 pilot.
set -uo pipefail
B=${CLAIMBLIND_ROOT:-.}
cd $B
DAPO="algorithm.filter_groups.enable=True algorithm.filter_groups.metric=reward"
RP=$B/code/rl_verl_reward_lenient.py
S=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 1:30:00 -J rl_grpo_v12_smoke \
  --export="ALL,SMOKE=1,EXP=smoke_v12_lenient,SEED=2,ROLL_TEMP=1.0,REWARD_PATH=$RP,VERL_EXTRA=$DAPO" code/rl_verl_grpo.sbatch) || exit 1
echo "SMOKE=$S"
F=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 24:00:00 -J rl_grpo_v12s2 \
  --dependency=afterok:$S --kill-on-invalid-dep=yes \
  --export="ALL,EXP=v12_lenient_seed2,SEED=2,TRAIN_BS=8,MINI_BS=4,ROLL_N=16,ROLL_TEMP=1.0,TOTAL_STEPS=60,SAVE_FREQ=5,TEST_FREQ=10,REWARD_PATH=$RP,VERL_EXTRA=$DAPO" \
  code/rl_verl_grpo.sbatch) || exit 1
echo "FULL=$F"
squeue -u "$USER" -o "%.8i %.18j %.10P %.8T %.8M %R"
