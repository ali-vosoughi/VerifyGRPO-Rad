#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J rl_probe
#SBATCH -o logs/%x_%j.out
# Dose test: probe sampling (selection pool or probe measurement), one shard per job.
# Usage: sbatch -p <part> -q <qos> --export=ALL,MODE=select|measure,PARQUET=<rel>,OUT=<rel>,SHARD=i,NSHARDS=n[,ADAPTER=<rel>|none,N_CAND=640,K=16,TEMP=1.0] code/rl_probe_sample.sbatch
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) PROBE MODE=${MODE:-} SHARD=${SHARD:-0}/${NSHARDS:-1} ADAPTER=${ADAPTER:-none}"
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export RADOPEN_ROOT=$RAD
ROOT=${CLAIMBLIND_ROOT:-.}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
ADAPTER=${ADAPTER:-none}
export HF_HOME=$RAD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f ~/.hf_env ] && source ~/.hf_env
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
if [ "$ADAPTER" = "none" ]; then
  WRITER=$MODEL; start_vllm "$MODEL" 16384 --gpu-memory-utilization 0.92
else
  [ -f "$ROOT/$ADAPTER/adapter_config.json" ] || { echo "NO_ADAPTER $ROOT/$ADAPTER"; exit 2; }
  WRITER=record; start_vllm "$MODEL" 16384 --gpu-memory-utilization 0.92 --enable-lora --max-lora-rank 64 --lora-modules record=$ROOT/$ADAPTER
  curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id":"record"' || { echo "LORA_NOT_SERVED"; exit 1; }
fi
export ADJ_BASE_URL=$BASE_URL ADJ_MODEL=$MODEL
"${VERL_VENV:?set VERL_VENV to the verl environment}/bin/python3" $ROOT/code/rl_probe_sample.py --mode ${MODE:?} --parquet $ROOT/${PARQUET:?} \
  --out $ROOT/${OUT:?} --shard ${SHARD:-0} --nshards ${NSHARDS:-1} --n-candidates ${N_CAND:-640} --k ${K:-16} --temp ${TEMP:-1.0} \
  --writer "$WRITER" --base-url "$BASE_URL" --workers 8
rc=$?
cleanup
[ "$rc" -eq 0 ] || { echo "PROBE_FAILED rc=$rc"; exit 4; }
echo "PROBE_EXIT rc=0 $(date)"
