#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J rl_xadj
#SBATCH -o logs/%x_%j.out
# ADJUDICATOR-TRANSFER control (2026-09-25; design v1 item 6, red-team plan item 9). The writer is
# trained against the frozen Qwen3-VL-8B adjudicator, so a trained record could exploit that judge. This
# job re-judges a read's CACHED records (no record is regenerated) with an independent-family text
# adjudicator (default MedGemma-4B-it, Gemma family): original claims (rl_control_records.py --policy
# readj), the cached-swap pass, and the directional scorer on the frozen 238-pair manifest. A gain that
# appears under the training judge and not here is a judge exploit, not image use.
# Usage: sbatch --export=ALL,ROWS=results/eval2/<read>/sweep_gated_default/rows.jsonl,OUT=results/eval2/<read>_xadj code/rl_xadj.sbatch
# SMOKE=1: first 10 manifest pairs. Fails loud.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) XADJ=v1 SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export RADOPEN_ROOT=$RAD
ROOT=${CLAIMBLIND_ROOT:-.}
ADJ_MODEL=${ADJ_MODEL:-google/medgemma-4b-it}
MAXLEN=${MAXLEN:-8192}
ROWS=$ROOT/${ROWS:?set ROWS relative to ROOT}
OUT=$ROOT/${OUT:?set OUT relative to ROOT}
PAIRS=$ROOT/${PAIRS:-results/val_clean/pairs_val_clean_in.jsonl}   # 2026-09-26: overridable for the locked set
MANIFEST=$ROOT/${MANIFEST:-results/val_clean/manifest_image_necessary.txt}
NO_IMAGE=$ROOT/${NO_IMAGE:-results/eval2/val_step_0/no_image/rows.jsonl}
LABELS=$ROOT/results/labels_eval_dev_val.jsonl
for f in $ROWS $PAIRS $MANIFEST $NO_IMAGE $LABELS; do [ -s "$f" ] || { echo "MISSING_INPUT $f"; exit 2; }; done
mkdir -p $OUT
if [ "${SMOKE:-0}" = "1" ]; then head -10 $MANIFEST > $OUT/manifest_smoke.txt; MANIFEST=$OUT/manifest_smoke.txt; fi
echo "ADJ_MODEL=$ADJ_MODEL ROWS=$ROWS OUT=$OUT MANIFEST=$MANIFEST"
sha256sum $ROWS $PAIRS $MANIFEST
export HF_HOME=$RAD/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f ~/.hf_env ] && source ~/.hf_env
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
start_vllm "$ADJ_MODEL" "$MAXLEN" --gpu-memory-utilization 0.90
cd $RAD/code
check() {  # NAME LOGFILE
  local line rows fails
  line=$(grep "^DONE rows=" "$2" | tail -1)
  rows=$(echo "$line" | sed -E 's/.*rows=([0-9]+).*/\1/'); fails=$(echo "$line" | sed -E 's/.*failures=([0-9]+).*/\1/')
  [ -n "$rows" ] && [ "$rows" != "$line" ] && [ "$rows" -gt 0 ] || { echo "ARM_EMPTY $1"; exit 4; }
  python3 -c "import sys; sys.exit(0 if $fails <= 0.10 * $rows else 1)" || { echo "ARM_TOO_MANY_FAILURES $1 $fails/$rows"; exit 4; }
}
LOG=$(mktemp)
echo "=== readj original claims $(date)"
python3 $ROOT/code/rl_control_records.py --rows $ROWS --manifest $MANIFEST --policy readj --model "$ADJ_MODEL" \
  --base-url "$BASE_URL" --out $OUT/record --workers 8 | tee "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM_FAILED readj"; exit 4; }
check readj "$LOG"
echo "=== swap_cached $(date)"
python3 $ROOT/code/rl_swap_adjudicate.py --rows $OUT/record/rows.jsonl --pairs $PAIRS --per-cell ${PER_CELL:-200} --cell-skip ${CELL_SKIP:-0} \
  --labels $LABELS --model "$ADJ_MODEL" --base-url "$BASE_URL" --out $OUT/swapcached --workers 8 | tee "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM_FAILED swap_cached"; exit 4; }
check swap_cached "$LOG"; rm -f "$LOG"
cleanup
python3 $ROOT/code/rl_score_directional.py --rows $OUT/record/rows.jsonl --swap-rows $OUT/swapcached/rows.jsonl \
  --no-image $NO_IMAGE --manifest $MANIFEST --out $OUT/directional || { echo "DIRECTIONAL_FAILED"; exit 5; }
sha256sum $OUT/directional/directional.csv
cat $OUT/directional/directional.csv
echo "XADJ_EXIT rc=0 ADJ_MODEL=$ADJ_MODEL $(date)"
