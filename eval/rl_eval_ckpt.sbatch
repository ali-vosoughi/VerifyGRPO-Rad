#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -J rl_eval
#SBATCH -o logs/%x_%j.out
# S3: dev-slice read of one checkpoint with the harness instrument (2026-09-23). vLLM serves the
# frozen base model plus the LoRA adapter (name "record"); the writer role is routed to the adapter,
# the adjudicator stays on the base. Arms: sweep_gated_default, its claim-swap arm, and (once) the
# no-image arm, all on cells 41-60 (--cell-skip 40 --per-cell 20). Scored with score_run.py.
# Usage: sbatch --export=ALL,ADAPTER=runs/round0/step_25,OUT=results/eval/round0_step_25,NO_IMAGE=results/eval/round0_step_0/no_image/rows.jsonl code/rl_eval_ckpt.sbatch
#        step 0 (prompted base, also produces the no-image arm): sbatch --export=ALL,ADAPTER=none,OUT=results/eval/round0_step_0,RUN_NO_IMAGE=1 code/rl_eval_ckpt.sbatch
set -uo pipefail
echo "HOST=$(hostname) ARCH=$(uname -m) JOB=$SLURM_JOB_ID DATE=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export RADOPEN_ROOT=$RAD
ROOT=${CLAIMBLIND_ROOT:-.}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
MAXLEN=${MAXLEN:-16384}
ADAPTER=${ADAPTER:?set ADAPTER relative to ROOT, or none}
OUT=$ROOT/${OUT:?set OUT relative to ROOT}
PER_CELL=${PER_CELL:-20}
CELL_SKIP=${CELL_SKIP:-40}
PROTO=${PROTO:-sweep_gated_default}
RUN_NO_IMAGE=${RUN_NO_IMAGE:-0}
NO_IMAGE=${NO_IMAGE:-}
PAIRS=${PAIRS:-$(ls -t $RAD/results/e6_pairs/pairs_*.jsonl | head -1)}
[ "${PAIRS#/}" = "$PAIRS" ] && PAIRS=$ROOT/$PAIRS   # relative PAIRS resolve under ROOT
[ -n "$PAIRS" ] || { echo "NO_PAIRS_FILE"; exit 2; }
echo "MODEL=$MODEL ADAPTER=$ADAPTER OUT=$OUT PROTO=$PROTO CELL_SKIP=$CELL_SKIP PER_CELL=$PER_CELL PAIRS=$PAIRS"
export HF_HOME=$RAD/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f ~/.hf_env ] && source ~/.hf_env
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
mkdir -p "$OUT" $ROOT/logs
source $RAD/code/scripts/vllm_common.sh
if [ "$ADAPTER" = "none" ]; then
  export WRITER_MODEL=BASE
  start_vllm "$MODEL" "$MAXLEN" --gpu-memory-utilization 0.92 ${VLLM_EXTRA:-}
else
  [ -f "$ROOT/$ADAPTER/adapter_config.json" ] || { echo "NO_ADAPTER $ROOT/$ADAPTER"; exit 2; }
  export WRITER_MODEL=record
  start_vllm "$MODEL" "$MAXLEN" --gpu-memory-utilization 0.92 --enable-lora --max-lora-rank 64 --lora-modules record=$ROOT/$ADAPTER
  curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id":"record"' || { echo "LORA_NOT_SERVED"; exit 1; }
fi
cd $RAD/code
arm() {  # NAME [verify_run args...]
  local NAME=$1; shift
  echo "=== ARM $NAME $* $(date)"
  local LOG; LOG=$(mktemp)
  python3 $ROOT/code/rl_eval_ckpt.py --pairs "$PAIRS" --model "$MODEL" --per-cell "$PER_CELL" --cell-skip "$CELL_SKIP" \
    --base-url "$BASE_URL" --out "$OUT/$NAME" "$@" | tee "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "ARM_FAILED $NAME rc=$rc"; rm -f "$LOG"; exit 4; }
  local line rows fails
  line=$(grep "^DONE rows=" "$LOG" | tail -1); rm -f "$LOG"
  rows=$(echo "$line" | sed -E 's/.*rows=([0-9]+).*/\1/'); fails=$(echo "$line" | sed -E 's/.*failures=([0-9]+).*/\1/')
  [ -n "$rows" ] && [ "$rows" != "$line" ] && [ "$rows" -gt 0 ] || { echo "ARM_EMPTY $NAME"; exit 4; }
  python3 -c "import sys; sys.exit(0 if $fails <= 0.10 * $rows else 1)" || { echo "ARM_TOO_MANY_FAILURES $NAME $fails/$rows"; exit 4; }
}
arm "$PROTO" --protocol "$PROTO"
arm "${PROTO}_claimswap" --protocol "$PROTO" --claim-swap
if [ "$RUN_NO_IMAGE" = "1" ]; then arm no_image --protocol no_image; NO_IMAGE=$OUT/no_image/rows.jsonl; fi
cleanup
[ -n "$NO_IMAGE" ] && [ -s "$ROOT/$NO_IMAGE" -o -s "$NO_IMAGE" ] || { echo "NO_IMAGE_ROWS_MISSING"; exit 2; }
[ -s "$NO_IMAGE" ] || NO_IMAGE=$ROOT/$NO_IMAGE
python3 scripts/score_run.py --runs "$OUT/$PROTO/rows.jsonl" --no-image "$NO_IMAGE" \
  --claim-swap "$OUT/${PROTO}_claimswap/rows.jsonl" --out "$OUT/summary" || { echo "SCORE_FAILED"; exit 5; }
sha256sum "$OUT/summary/protocol_summary.csv"; cat "$OUT/summary/protocol_summary.csv"
echo "EVAL_EXIT rc=0 ADAPTER=$ADAPTER $(date)"
