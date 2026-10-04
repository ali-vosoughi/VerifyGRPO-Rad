#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -J rl_readj
#SBATCH -o logs/%x_%j.out
# text-only re-adjudication of the cached round-0 samples on canonical record bytes (and a
# flags-only payload), frozen base adjudicator through vLLM. Fails loud (script exit 4 on zero judged,
# nonzero rc otherwise).
# Usage: sbatch --export=ALL,LIMIT=200 -t 0:40:00 -J rl_readj_smoke code/rl_readjudicate.sbatch   (smoke)
#        sbatch --export=ALL,LIMIT=0 --dependency=afterok:<smoke> --kill-on-invalid-dep=yes code/rl_readjudicate.sbatch
set -uo pipefail
echo "HOST=$(hostname) ARCH=$(uname -m) JOB=$SLURM_JOB_ID DATE=$(date) SITE=B"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export RADOPEN_ROOT=$RAD
ROOT=${CLAIMBLIND_ROOT:-.}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
MAXLEN=${MAXLEN:-8192}
LIMIT=${LIMIT:-0}
SAMPLES=${SAMPLES:-results/round0_pool/samples_all.jsonl}
OUT=${OUT:-results/round0_readj}
[ "$LIMIT" != "0" ] && OUT=${OUT}_smoke
SAMPLES=$ROOT/$SAMPLES; OUT=$ROOT/$OUT
[ -s "$SAMPLES" ] || { echo "NO_SAMPLES $SAMPLES"; exit 2; }
echo "MODEL=$MODEL SAMPLES=$SAMPLES OUT=$OUT LIMIT=$LIMIT"
export HF_HOME=$RAD/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f ~/.hf_env ] && source ~/.hf_env
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
mkdir -p $ROOT/logs "$OUT"
source $RAD/code/scripts/vllm_common.sh
start_vllm "$MODEL" "$MAXLEN" --gpu-memory-utilization 0.92 ${VLLM_EXTRA:-}
python3 $ROOT/code/rl_readjudicate.py --samples "$SAMPLES" --model "$MODEL" --base-url "$BASE_URL" --out "$OUT" --workers 8 --limit "$LIMIT"
rc=$?
cleanup
[ "$rc" -eq 0 ] || { echo "READJ_FAILED rc=$rc"; exit 4; }
sha256sum "$OUT/samples_canon.jsonl"
echo "READJ_EXIT rc=0 $(date)"
