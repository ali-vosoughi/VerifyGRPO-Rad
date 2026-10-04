# Shared vLLM launch, readiness, teardown and protocol runner for the radagent-open sbatch scripts.
# Added 2026-09-20 after two jobs on one node both bound port 8000, the second
# server died ("address already in use"), the health check passed against the first job's server
# (another model), every row failed, and the job still exited 0. Now:
#   - one port per job, derived from the job id;
#   - readiness requires /v1/models to list the requested model id, not just /health;
#   - every protocol's DONE line is checked and the job aborts (non-zero) on a failed protocol,
#     an empty protocol, or a failure fraction above MAX_FAIL_FRAC;
#   - the server is killed on any exit path.
# Expects ROOT, MODEL, PAIRS, PER_CELL to be set by the caller; sets PORT, BASE_URL, VPID.
PORT=$((20000 + SLURM_JOB_ID % 20000))
BASE_URL="http://127.0.0.1:${PORT}/v1"
VPID=""
MAX_FAIL_FRAC=${MAX_FAIL_FRAC:-0.10}
cleanup() { [ -n "$VPID" ] && kill "$VPID" 2>/dev/null; return 0; }
trap cleanup EXIT

start_vllm() {  # MODEL MAXLEN [extra vllm args...]
  local M=$1 L=$2; shift 2
  python3 -m vllm.entrypoints.openai.api_server --model "$M" --port "$PORT" --max-model-len "$L" \
    --dtype bfloat16 --seed 20260919 "$@" > "$ROOT/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
  VPID=$!
  echo "VLLM port=$PORT pid=$VPID model=$M maxlen=$L"
  local i
  for i in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q "\"id\":\"$M\""; then
      echo "VLLM_READY after $((i * 10))s"; return 0
    fi
    kill -0 "$VPID" 2>/dev/null || { echo "VLLM_DIED"; tail -60 "$ROOT/logs/vllm_${SLURM_JOB_ID}.log"; exit 1; }
    sleep 10
  done
  echo "VLLM_TIMEOUT"; tail -40 "$ROOT/logs/vllm_${SLURM_JOB_ID}.log"; exit 1
}

# run PROTOCOL [verify_run args...]  (uses PAIRS, MODEL, PER_CELL, BASE_URL)
run() {
  local P=$1; shift
  echo "=== PROTOCOL $P $* $(date)"
  local LOG; LOG=$(mktemp)
  python3 scripts/verify_run.py --pairs "$PAIRS" --model "$MODEL" --protocol "$P" --per-cell "$PER_CELL" \
    --base-url "$BASE_URL" "$@" | tee "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "PROTOCOL_FAILED $P rc=$rc"; rm -f "$LOG"; exit 4; }
  local line rows fails
  line=$(grep "^DONE rows=" "$LOG" | tail -1); rm -f "$LOG"
  rows=$(echo "$line" | sed -E 's/.*rows=([0-9]+).*/\1/')
  fails=$(echo "$line" | sed -E 's/.*failures=([0-9]+).*/\1/')
  if [ -z "$rows" ] || [ "$rows" = "$line" ] || [ "$rows" -eq 0 ]; then echo "PROTOCOL_EMPTY $P"; exit 4; fi
  if ! python3 -c "import sys; sys.exit(0 if $fails <= $MAX_FAIL_FRAC * $rows else 1)"; then
    echo "PROTOCOL_TOO_MANY_FAILURES $P failures=$fails rows=$rows max_frac=$MAX_FAIL_FRAC"; exit 4
  fi
}
