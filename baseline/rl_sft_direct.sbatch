#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -J rl_sft
#SBATCH -o logs/%x_%j.out
# Direct-supervision baseline (2026-09-26; protocol declared before any run).
# Env: MODE=full|target, ARM=<name>, PATHS=<file relative to ROOT or empty>, SMOKE=1 -> 16 examples.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) SFT=v1 MODE=${MODE:?} ARM=${ARM:?} PATHS=${PATHS:-} SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
ROOT=${CLAIMBLIND_ROOT:-.}
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export HF_HOME=$RAD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${VERL_VENV:?set VERL_VENV to the verl environment}/bin/activate"
DATA=$ROOT/data/rep_s101/train.parquet
REF=$ROOT/runs/verl/dose_v1_lr1e-4/global_step_40/hf/lora_adapter
OUT=$ROOT/runs/sft/$ARM
[ -e $OUT/lora_adapter ] && { echo "EXISTS $OUT (fresh start refused)"; exit 2; }
P=""; [ -n "${PATHS:-}" ] && P="--paths $ROOT/$PATHS"
L=""; [ "${SMOKE:-0}" = "1" ] && L="--limit 16"
sha256sum $DATA ${PATHS:+$ROOT/$PATHS}
python3 $ROOT/code/rl_sft_direct.py --data $DATA $P --mode $MODE --out $OUT --ref-adapter $REF $L ${EPOCHS:+--epochs $EPOCHS}
RC=$?
echo "SFT_EXIT rc=$RC ARM=$ARM $(date)"
exit $RC
