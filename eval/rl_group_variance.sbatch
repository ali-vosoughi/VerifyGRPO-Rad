#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -J rl_gvar
#SBATCH -o logs/%x_%j.out
# GRPO group-variance decomposition at training settings (2026-09-25). One GPU: vLLM (radopen, vLLM
# 0.29) serves the base model and, when ADAPTER is set, the LoRA as "record"; the same server is the frozen
# adjudicator for the training reward. The client (rl_group_variance.py) runs under envs/verl.
# Usage: sbatch --export=ALL,ADAPTER=none|runs/verl/<exp>/global_step_N/hf/lora_adapter,OUT=results/gvar/<tag>[,N_IMAGES=64,K=8,TEMP=0.8] code/rl_group_variance.sbatch
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) GVAR=v1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export RADOPEN_ROOT=$RAD
ROOT=${CLAIMBLIND_ROOT:-.}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
ADAPTER=${ADAPTER:?set ADAPTER or none}
OUT=$ROOT/${OUT:?set OUT relative to ROOT}
N_IMAGES=${N_IMAGES:-64}; K=${K:-8}; TEMP=${TEMP:-0.8}
export HF_HOME=$RAD/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f ~/.hf_env ] && source ~/.hf_env
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
if [ "$ADAPTER" = "none" ]; then
  WRITER=$MODEL
  start_vllm "$MODEL" 16384 --gpu-memory-utilization 0.92
else
  [ -f "$ROOT/$ADAPTER/adapter_config.json" ] || { echo "NO_ADAPTER $ROOT/$ADAPTER"; exit 2; }
  WRITER=record
  start_vllm "$MODEL" 16384 --gpu-memory-utilization 0.92 --enable-lora --max-lora-rank 64 --lora-modules record=$ROOT/$ADAPTER
  curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id":"record"' || { echo "LORA_NOT_SERVED"; exit 1; }
fi
export ADJ_BASE_URL=$BASE_URL ADJ_MODEL=$MODEL
echo "WRITER=$WRITER ADAPTER=$ADAPTER N_IMAGES=$N_IMAGES K=$K TEMP=$TEMP OUT=$OUT"
"${VERL_VENV:?set VERL_VENV to the verl environment}/bin/python3" $ROOT/code/rl_group_variance.py --writer "$WRITER" --base-url "$BASE_URL" \
  --n-images $N_IMAGES --k $K --temp $TEMP --out $OUT --workers 8 ${LENIENT:+--lenient}
rc=$?
cleanup
[ "$rc" -eq 0 ] || { echo "GVAR_FAILED rc=$rc"; exit 4; }
echo "GVAR_EXIT rc=0 $(date)"
