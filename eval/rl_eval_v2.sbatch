#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J rl_eval2
#SBATCH -o logs/%x_%j.out
# Evaluation v2 (2026-09-24, after the external red team). One checkpoint read = (1) the record
# arm through the harness (writer routed to the adapter, adjudicator on the base), (2) the
# claim-swap arm on the CACHED records (rl_swap_adjudicate.py; text-only; exact record reuse by
# hash; swap truth from the study labels), (3) directional scoring on a frozen image-necessary
# manifest (rl_score_directional.py: G, B, Y, FS, C_pres, C_flip_correct, d, pair bootstrap). The
# no-image arm is run once per pairs file (RUN_NO_IMAGE=1) and reused. The harness's score_run.py still
# runs for the legacy columns. Fails loud at every step.
# Usage (validation set, step 0):
#   sbatch --export=ALL,ADAPTER=none,PAIRS=results/val_clean/pairs_val_clean_2026-09-24.jsonl,CELL_SKIP=0,PER_CELL=200,OUT=results/eval2/val_step_0,RUN_NO_IMAGE=1,LABELS=results/labels_eval_dev_val.jsonl,MANIFEST=results/val_clean/manifest_image_necessary.txt code/rl_eval_v2.sbatch
# Usage (checkpoint): same with ADAPTER=runs/<run>/step_N, OUT=results/eval2/<run>_step_N, NO_IMAGE=results/eval2/val_step_0/no_image/rows.jsonl
set -uo pipefail
echo "HOST=$(hostname) ARCH=$(uname -m) JOB=$SLURM_JOB_ID DATE=$(date) EVAL=v2"
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
RECORD_ROWS=${RECORD_ROWS:-}   # existing record-arm rows.jsonl (relative to ROOT) to reuse instead of regenerating
LABELS=$ROOT/${LABELS:?set LABELS relative to ROOT}
MANIFEST=$ROOT/${MANIFEST:?set MANIFEST relative to ROOT (written on first use)}
PAIRS=${PAIRS:-$(ls -t $RAD/results/e6_pairs/pairs_*.jsonl | head -1)}
[ "${PAIRS#/}" = "$PAIRS" ] && PAIRS=$ROOT/$PAIRS
[ -s "$PAIRS" ] || { echo "NO_PAIRS_FILE $PAIRS"; exit 2; }
[ -s "$LABELS" ] || { echo "NO_LABELS $LABELS"; exit 2; }
echo "MODEL=$MODEL ADAPTER=$ADAPTER OUT=$OUT PROTO=$PROTO CELL_SKIP=$CELL_SKIP PER_CELL=$PER_CELL PAIRS=$PAIRS LABELS=$LABELS MANIFEST=$MANIFEST"
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
  start_vllm "$MODEL" "$MAXLEN" --gpu-memory-utilization 0.92 ${VLLM_EXTRA:-} --enable-lora --max-lora-rank 64 --lora-modules record=$ROOT/$ADAPTER
  curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id":"record"' || { echo "LORA_NOT_SERVED"; exit 1; }
fi
cd $RAD/code
check() {  # NAME LOGFILE
  local line rows fails
  line=$(grep "^DONE rows=" "$2" | tail -1)
  rows=$(echo "$line" | sed -E 's/.*rows=([0-9]+).*/\1/'); fails=$(echo "$line" | sed -E 's/.*failures=([0-9]+).*/\1/')
  [ -n "$rows" ] && [ "$rows" != "$line" ] && [ "$rows" -gt 0 ] || { echo "ARM_EMPTY $1"; exit 4; }
  python3 -c "import sys; sys.exit(0 if $fails <= 0.10 * $rows else 1)" || { echo "ARM_TOO_MANY_FAILURES $1 $fails/$rows"; exit 4; }
}
if [ -n "$RECORD_ROWS" ]; then
  [ -s "$ROOT/$RECORD_ROWS" ] || { echo "NO_RECORD_ROWS $ROOT/$RECORD_ROWS"; exit 2; }
  mkdir -p "$OUT/$PROTO"; cp "$ROOT/$RECORD_ROWS" "$OUT/$PROTO/rows.jsonl"; echo "REUSED_RECORD_ROWS $RECORD_ROWS $(wc -l < "$OUT/$PROTO/rows.jsonl") rows"
else
  # 2026-09-25: the harness loops rows sequentially (~20 s per trained record on an A6000 = 2.6 h per
  # read), so the record arm runs SHARDS concurrent harness processes against the one vLLM server, which
  # batches them. Pairs are selected by the per-cell rule FIRST, then dealt round-robin into shards run
  # with --per-cell 0 (all pairs of the shard); rows are keyed by (pair_id, condition), so the merged
  # file is the same rows in another order. The swap pass below still gets the full PAIRS file, so its
  # donors are unchanged. SHARDS=1 = the old sequential path.
  NS=${SHARDS:-8}
  echo "=== ARM record $(date) SHARDS=$NS"
  SH=$OUT/_record_shards; rm -rf "$SH"; mkdir -p "$SH"
  python3 $ROOT/code/rl_shard_pairs.py --pairs "$PAIRS" --per-cell "$PER_CELL" --cell-skip "$CELL_SKIP" --shards "$NS" --out-dir "$SH" \
    || { echo "SHARDING_FAILED"; exit 4; }
  pids=()
  for P in $SH/pairs_*.jsonl; do
    K=$(basename $P .jsonl)
    python3 $ROOT/code/rl_eval_ckpt.py --pairs "$P" --model "$MODEL" --per-cell 0 --cell-skip 0 \
      --base-url "$BASE_URL" --out "$SH/rec_$K" --protocol "$PROTO" > "$SH/rec_$K.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ "$rc" -eq 0 ] || { echo "ARM_FAILED record (a shard exited nonzero)"; tail -3 $SH/rec_*.log; exit 4; }
  for L in $SH/rec_*.log; do check "record $(basename $L)" "$L"; done
  mkdir -p "$OUT/$PROTO"; cat $SH/rec_*/rows.jsonl > "$OUT/$PROTO/rows.jsonl"
  NR=$(wc -l < "$OUT/$PROTO/rows.jsonl"); NP=$(cat $SH/n_pairs.txt)
  [ "$NR" -eq $((2 * NP)) ] || { echo "RECORD_ROWS_MISMATCH rows=$NR expected=$((2 * NP))"; exit 4; }
  echo "DONE rows=$NR failures=$(grep -c '"ok": false' "$OUT/$PROTO/rows.jsonl") (record arm, $NS shards merged) $(date)"
fi
echo "=== ARM swap_cached $(date)"
LOG=$(mktemp)
python3 $ROOT/code/rl_swap_adjudicate.py --rows "$OUT/$PROTO/rows.jsonl" --pairs "$PAIRS" --per-cell "$PER_CELL" --cell-skip "$CELL_SKIP" \
  --labels "$LABELS" --model "$MODEL" --base-url "$BASE_URL" --out "$OUT/${PROTO}_swapcached" --workers 8 | tee "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM_FAILED swap_cached"; exit 4; }
check swap_cached "$LOG"; rm -f "$LOG"
if [ "$RUN_NO_IMAGE" = "1" ]; then
  echo "=== ARM no_image $(date)"
  LOG=$(mktemp)
  python3 $ROOT/code/rl_eval_ckpt.py --pairs "$PAIRS" --model "$MODEL" --per-cell "$PER_CELL" --cell-skip "$CELL_SKIP" \
    --base-url "$BASE_URL" --out "$OUT/no_image" --protocol no_image | tee "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM_FAILED no_image"; exit 4; }
  check no_image "$LOG"; rm -f "$LOG"
  NO_IMAGE=$OUT/no_image/rows.jsonl
fi
cleanup
[ -s "$NO_IMAGE" ] || NO_IMAGE=$ROOT/$NO_IMAGE
[ -s "$NO_IMAGE" ] || { echo "NO_IMAGE_ROWS_MISSING"; exit 2; }
python3 $ROOT/code/rl_score_directional.py --rows "$OUT/$PROTO/rows.jsonl" --swap-rows "$OUT/${PROTO}_swapcached/rows.jsonl" \
  --no-image "$NO_IMAGE" --manifest "$MANIFEST" --out "$OUT/directional" || { echo "DIRECTIONAL_FAILED"; exit 5; }
python3 scripts/score_run.py --runs "$OUT/$PROTO/rows.jsonl" --no-image "$NO_IMAGE" --out "$OUT/summary" || { echo "SCORE_FAILED"; exit 5; }
sha256sum "$OUT/directional/directional.csv" "$OUT/summary/protocol_summary.csv" "$MANIFEST"
cat "$OUT/directional/directional.csv"
echo "EVAL2_EXIT rc=0 ADAPTER=$ADAPTER $(date)"
