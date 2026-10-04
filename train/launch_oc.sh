#!/bin/bash
# OMISSION-COST ARM (protocol registered before any run): the replication
# recipe exactly (launch_rep.sh) with REWARD_PATH = code/rl_verl_reward_omitcost.py (score = max(0, content -
# 0.25 * missing / 12)); seeds 101 / 202 / 303 in every slot and in the data order (data/rep_s<S>); 40 steps. A smoke
# (seed 101, 2 steps, the runs' 3-GPU layout) gates all three. Refuses existing output directories.
set -uo pipefail
B=${CLAIMBLIND_ROOT:-.}
cd $B
RP=$B/code/rl_verl_reward_omitcost.py
for E in smoke_oc_s101 oc_s101 oc_s202 oc_s303; do
  for D in runs/verl/$E runs/rollouts/$E runs/reward_logs/$E; do [ -e $D ] && { echo "EXISTS $D"; exit 2; }; done
done
exp() {  # EXP SEED -> export string
  local S=$2
  echo "ALL,EXP=$1,ACTOR_LR=1e-4,SEED=$S,DATA=data/rep_s$S,TRAIN_BS=8,MINI_BS=4,ROLL_N=16,ROLL_TEMP=1.0,REWARD_PATH=$RP,REWARD_LOG_DIR=$B/runs/reward_logs/$1,KEEP_LAST=0,VERL_EXTRA=algorithm.filter_groups.enable=True algorithm.filter_groups.metric=reward data.shuffle=False trainer.rollout_data_dir=$B/runs/rollouts/$1 actor_rollout_ref.actor.fsdp_config.seed=$S actor_rollout_ref.actor.data_loader_seed=$S actor_rollout_ref.ref.fsdp_config.seed=$S actor_rollout_ref.rollout.seed=$S"
}
S=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:3 -c 16 --mem=240G -t 2:00:00 -J rl_oc_smoke \
  --export="$(exp smoke_oc_s101 101),TOTAL_STEPS=2,SAVE_FREQ=2,TEST_FREQ=2" code/rl_verl_grpo.sbatch) || exit 1
echo "SMOKE=$S"
for SEED in 101 202 303; do
  P="-p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS}"
  J=$(sbatch --parsable $P --gres=gpu:a6000:3 -c 16 --mem=240G -t 24:00:00 -J rl_oc_s$SEED \
    --dependency=afterok:$S --kill-on-invalid-dep=yes \
    --export="$(exp oc_s$SEED $SEED),TOTAL_STEPS=40,SAVE_FREQ=10,TEST_FREQ=20" code/rl_verl_grpo.sbatch) || exit 1
  echo "RUN_s$SEED=$J"
done
