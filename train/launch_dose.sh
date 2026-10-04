#!/bin/bash
# BOUNDED DOSE TEST (2026-09-25; protocol = red team round 2 follow-up 1, frozen in the project log at 17:45;
# implementation-review fixes 4 and 6 applied). Two arms from the SAME base model and fresh adapter / optimizer,
# differing ONLY in LR (3e-5 vs 1e-4): v1.2 configuration (lenient reward value, DAPO filter, T=1.0, 16 rollouts,
# 8 images per step, mini-batch 4, KL 0.01), 20 steps, saves at 10 and 20, all seeds 42, data.shuffle=False over
# data/dose_v1/train.parquet (frozen probe prefix, interleaved by truth, then a fixed shuffle). Every rollout is
# dumped (trainer.rollout_data_dir) and every judged payload / per-claim verdict logged (REWARD_LOG_DIR); the reward
# VALUE is v1.2's. KEEP_LAST=1 keeps the newest save resumable for the extension.
#   smoke          = PRODUCTION settings (LR 1e-4, 8 / 4 / 16, T 1, filter, logging), 2 steps, 3 GPUs (2 verl + judge)
#   resume smoke   = same experiment, TOTAL_STEPS=3: must restore step 2 and run step 3 (restoration test for the
#                    extension); runs after the smoke, in parallel with the arms
#   arms           = after the smoke; refuse to start over existing output directories
set -uo pipefail
B=${CLAIMBLIND_ROOT:-.}
cd $B
[ -s data/dose_v1/probe_manifest.json ] && [ -s data/dose_v1/train.parquet ] || { echo "PROBE_NOT_FROZEN"; exit 2; }
[ -e data/dose_v1/val.parquet ] || cp -p data/verl_v1/val.parquet data/dose_v1/val.parquet
for E in smoke_dose_v1 dose_v1_lr3e-5 dose_v1_lr1e-4; do
  for D in runs/verl/$E runs/rollouts/$E runs/reward_logs/$E; do [ -e $D ] && { echo "EXISTS $D (fresh start refused)"; exit 2; }; done
done
RP=$B/code/rl_verl_reward_logged.py
common() {  # EXP LR -> shared export string (production batch settings)
  echo "ALL,EXP=$1,ACTOR_LR=$2,SEED=2,DATA=data/dose_v1,TRAIN_BS=8,MINI_BS=4,ROLL_N=16,ROLL_TEMP=1.0,REWARD_PATH=$RP,REWARD_LOG_DIR=$B/runs/reward_logs/$1,KEEP_LAST=1,VERL_EXTRA=algorithm.filter_groups.enable=True algorithm.filter_groups.metric=reward data.shuffle=False trainer.rollout_data_dir=$B/runs/rollouts/$1"
}
S=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 2:00:00 -J rl_dose_smoke \
  --export="$(common smoke_dose_v1 1e-4),TOTAL_STEPS=2,SAVE_FREQ=1,TEST_FREQ=2" code/rl_verl_grpo.sbatch) || exit 1
echo "SMOKE=$S"
R=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 2:00:00 -J rl_dose_resume \
  --dependency=afterok:$S --kill-on-invalid-dep=yes \
  --export="$(common smoke_dose_v1 1e-4),TOTAL_STEPS=3,SAVE_FREQ=1,TEST_FREQ=3" code/rl_verl_grpo.sbatch) || exit 1
echo "RESUME_SMOKE=$R"
A=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 24:00:00 -J rl_dose_3e5 \
  --dependency=afterok:$S --kill-on-invalid-dep=yes \
  --export="$(common dose_v1_lr3e-5 3e-5),TOTAL_STEPS=20,SAVE_FREQ=10,TEST_FREQ=10" code/rl_verl_grpo.sbatch) || exit 1
echo "ARM_3e-5=$A"
C=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 24:00:00 -J rl_dose_1e4 \
  --dependency=afterok:$S --kill-on-invalid-dep=yes \
  --export="$(common dose_v1_lr1e-4 1e-4),TOTAL_STEPS=20,SAVE_FREQ=10,TEST_FREQ=10" code/rl_verl_grpo.sbatch) || exit 1
echo "ARM_1e-4=$C"
squeue -u "$USER" -o "%.8i %.16j %.10P %.8T %.8M %R"
