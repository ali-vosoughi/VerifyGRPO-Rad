#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 1:00:00
#SBATCH -J rl_merge
#SBATCH -o logs/%x_%j.out
# Convert ONE saved verl checkpoint to a PEFT adapter while the run is still training (2026-09-25),
# so the curve is read as saves land instead of after the last step. CPU only.
# Same merger call as rl_verl_grpo.sbatch's end-of-run loop. Pruning rule: the merged full
# model is deleted at once; FSDP shards are pruned only for checkpoints OLDER than the newest completed
# save that already hold an adapter, so the newest save always stays resumable. The end-of-run loop skips
# checkpoints that already have an adapter, so the last one merged here keeps its shards until pruned by
# hand after the run. Usage: sbatch --export=ALL,EXP=<exp>,STEP=<n> code/rl_merge_ckpt.sbatch
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) MERGE EXP=${EXP:-} STEP=${STEP:-}"
ROOT=${CLAIMBLIND_ROOT:-.}
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
EXP=${EXP:?set EXP}; STEP=${STEP:?set STEP}
CKPT=$ROOT/runs/verl/$EXP; D=$CKPT/global_step_$STEP
LATEST=$(cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
[ "$LATEST" -ge "$STEP" ] || { echo "SAVE_NOT_COMPLETE latest=$LATEST step=$STEP"; exit 2; }
export HF_HOME=$RAD/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${VERL_VENV:?set VERL_VENV to the verl environment}/bin/activate"
have_adapter() { [ -f "$1/hf/lora_adapter/adapter_config.json" ] || [ -f "$1/hf/adapter_config.json" ]; }
if ! have_adapter "$D"; then
  [ -f "$D/actor/model_world_size_2_rank_0.pt" ] || ls $D/actor/model_world_size_*.pt >/dev/null 2>&1 || { echo "NO_SHARDS $D"; exit 2; }
  CUDA_VISIBLE_DEVICES="" python3 -m verl.model_merger merge --backend fsdp --local_dir $D/actor --target_dir $D/hf > $D/merge.log 2>&1
fi
have_adapter "$D" || { echo "MERGE_FAILED $D"; tail -20 $D/merge.log; exit 4; }
rm -f $D/hf/model.safetensors $D/hf/model-*.safetensors
LATEST=$(cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
for O in $(ls -d $CKPT/global_step_* | sort -t_ -k3 -n); do
  N=${O##*_}
  if [ "$N" -lt "$LATEST" ] && have_adapter "$O" && ls $O/actor/model_world_size_*.pt >/dev/null 2>&1; then
    rm -f $O/actor/model_world_size_*.pt $O/actor/optim_world_size_*.pt $O/data.pt && echo "PRUNED_SHARDS $O"
  fi
done
ls -la $D/hf/lora_adapter 2>/dev/null || ls -la $D/hf
echo "MERGE_EXIT rc=0 EXP=$EXP STEP=$STEP LATEST=$LATEST $(date)"
