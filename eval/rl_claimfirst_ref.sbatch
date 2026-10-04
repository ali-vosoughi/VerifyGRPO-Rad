#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J rl_cfref
#SBATCH -o logs/%x_%j.out
# Claim-first REFERENCE on the clean validation set (2026-09-25). The GO gate's false-strike guard
# and Y comparison are against claim-first, which had never been read on this population. Per protocol
# (claim_first, claim_first_cot): the harness unchanged on the 238 manifest pairs with the base model
# (WRITER_MODEL=BASE; no record is written), then its --claim-swap arm (same seed and pairs file, so the
# swapped sentences equal the cached-record swap pass), swap truth added from the labels, and the
# directional scorer on the frozen manifest. Claim-first C_pres is contamination proper (the verdict sees
# the claim and the image together). SMOKE=1: first 10 pairs, claim_first only. Fails loud.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) CFREF=v1 SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export RADOPEN_ROOT=$RAD
ROOT=${CLAIMBLIND_ROOT:-.}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
MAXLEN=${MAXLEN:-16384}
PAIRS=$ROOT/results/val_clean/pairs_val_clean_in.jsonl
MANIFEST=$ROOT/results/val_clean/manifest_image_necessary.txt
NO_IMAGE=$ROOT/results/eval2/val_step_0/no_image/rows.jsonl
LABELS=$ROOT/results/labels_eval_dev_val.jsonl
for f in $PAIRS $MANIFEST $NO_IMAGE $LABELS; do [ -s "$f" ] || { echo "MISSING_INPUT $f"; exit 2; }; done
if [ "${SMOKE:-0}" = "1" ]; then
  OUTBASE=$ROOT/results/eval2/claimfirst_ref_smoke; PROTOS=${PROTOS:-claim_first}
  mkdir -p $OUTBASE
  # one pair per (finding, direction) cell: the swap arm needs donors from other findings (an earlier smoke drew 10
  # pairs of one finding and built zero swap rows)
  python3 -c "import json,sys; seen=set(); [ (seen.add((d['finding'],d['direction'])), print(l.strip())) for l in open(sys.argv[1]) if l.strip() for d in [json.loads(l)] if (d['finding'],d['direction']) not in seen ]" $PAIRS > $OUTBASE/pairs_smoke.jsonl
  PAIRS=$OUTBASE/pairs_smoke.jsonl; [ "$(wc -l < $PAIRS)" -ge 2 ] || { echo "SMOKE_PAIRS_TOO_FEW"; exit 2; }
  python3 -c "import json,sys; [print(json.loads(l)['pair_id']) for l in open(sys.argv[1]) if l.strip()]" $PAIRS > $OUTBASE/manifest_smoke.txt
  MANIFEST=$OUTBASE/manifest_smoke.txt
else
  OUTBASE=$ROOT/results/eval2/claimfirst_ref_v1; PROTOS=${PROTOS:-claim_first+claim_first_cot}
fi
mkdir -p $OUTBASE
echo "OUTBASE=$OUTBASE PROTOS=$PROTOS PAIRS=$PAIRS MANIFEST=$MANIFEST"
sha256sum $PAIRS $MANIFEST $NO_IMAGE
export HF_HOME=$RAD/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f ~/.hf_env ] && source ~/.hf_env
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
export WRITER_MODEL=BASE
start_vllm "$MODEL" "$MAXLEN" --gpu-memory-utilization 0.92
cd $RAD/code
check() {  # NAME LOGFILE
  local line rows fails
  line=$(grep "^DONE rows=" "$2" | tail -1)
  rows=$(echo "$line" | sed -E 's/.*rows=([0-9]+).*/\1/'); fails=$(echo "$line" | sed -E 's/.*failures=([0-9]+).*/\1/')
  [ -n "$rows" ] && [ "$rows" != "$line" ] && [ "$rows" -gt 0 ] || { echo "ARM_EMPTY $1"; exit 4; }
  python3 -c "import sys; sys.exit(0 if $fails <= 0.10 * $rows else 1)" || { echo "ARM_TOO_MANY_FAILURES $1 $fails/$rows"; exit 4; }
}
IFS='+' read -ra PL <<< "$PROTOS"
for P in "${PL[@]}"; do
  O=$OUTBASE/$P
  LOG=$(mktemp)
  echo "=== $P original $(date)"
  python3 $ROOT/code/rl_eval_ckpt.py --pairs $PAIRS --model "$MODEL" --per-cell 200 --cell-skip 0 \
    --base-url "$BASE_URL" --out $O/orig --protocol $P | tee "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM_FAILED $P orig"; exit 4; }
  check "$P orig" "$LOG"
  echo "=== $P claim swap $(date)"
  python3 $ROOT/code/rl_eval_ckpt.py --pairs $PAIRS --model "$MODEL" --per-cell 200 --cell-skip 0 \
    --base-url "$BASE_URL" --out $O/swap_raw --protocol $P --claim-swap | tee "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM_FAILED $P swap"; exit 4; }
  check "$P swap" "$LOG"
  python3 $ROOT/code/rl_add_swap_truth.py --rows $O/swap_raw/rows.jsonl --pairs $PAIRS --labels $LABELS --out $O/swap | tee "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "SWAP_TRUTH_FAILED $P"; exit 4; }
  check "$P swap_truth" "$LOG"; rm -f "$LOG"
  python3 $ROOT/code/rl_score_directional.py --rows $O/orig/rows.jsonl --swap-rows $O/swap/rows.jsonl \
    --no-image $NO_IMAGE --manifest $MANIFEST --out $O/directional || { echo "DIRECTIONAL_FAILED $P"; exit 5; }
done
cleanup
sha256sum $OUTBASE/*/directional/directional.csv
echo "CFREF_EXIT rc=0 $(date)"
