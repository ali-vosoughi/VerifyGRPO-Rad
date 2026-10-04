#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J rl_controls
#SBATCH -o logs/%x_%j.out
# Falsification controls on the evaluation-v2 metric (2026-09-25; design v1 item 6, red-team plan item 7).
# (0) leakage assertion: verl parquet patients vs the clean validation set and the harness's eval + dev (exit 3);
# (1) per policy: control records adjudicated against the original claim (rl_control_records.py), the
#     unchanged cached-swap pass, the unchanged directional scorer on the frozen 238-pair manifest;
# (2) one summary table. Policy step0_in = the step-0 point with the swap donors of the 238-pair file,
#     which is the file every checkpoint read uses; the earlier step-0
#     numbers used the 507-pair donor pool and are not on the same swap claims.
# SMOKE=1: 10 manifest pairs, three policies, results/eval2/controls_smoke. Fails loud at every step.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) CONTROLS=v1 SMOKE=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export RADOPEN_ROOT=$RAD
ROOT=${CLAIMBLIND_ROOT:-.}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
MAXLEN=${MAXLEN:-16384}
STEP0=$ROOT/${ROWS_SRC:-results/eval2/val_step_0/sweep_gated_default/rows.jsonl}   # ROWS_SRC: another read (interface audit)
PAIRS=$ROOT/${PAIRS_REL:-results/val_clean/pairs_val_clean_in.jsonl}   # 2026-09-27 override for the locked reread
MANIFEST=$ROOT/${MANIFEST_REL:-results/val_clean/manifest_image_necessary.txt}
NO_IMAGE=$ROOT/${NO_IMAGE_REL:-results/eval2/val_step_0/no_image/rows.jsonl}
LABELS=$ROOT/results/labels_eval_dev_val.jsonl
TRAIN_LABELS=$ROOT/results/train_slice/images.jsonl
VAL_IMAGES=$ROOT/results/val_clean/images.txt
for f in $STEP0 $PAIRS $MANIFEST $NO_IMAGE $LABELS $TRAIN_LABELS $VAL_IMAGES $ROOT/data/verl_v1/train.parquet; do
  [ -s "$f" ] || { echo "MISSING_INPUT $f"; exit 2; }
done
if [ "${SMOKE:-0}" = "1" ]; then
  OUTBASE=$ROOT/results/eval2/controls_smoke
  POLICIES=${POLICIES:-step0_in+all_absent+permute_global}
  mkdir -p $OUTBASE; head -10 $MANIFEST > $OUTBASE/manifest_smoke.txt; MANIFEST=$OUTBASE/manifest_smoke.txt
else
  OUTBASE=$ROOT/${OUTDIR:-results/eval2/controls_v1}
  POLICIES=${POLICIES:-step0_in+all_absent+all_present+top5+permute_cell+permute_global}
fi
mkdir -p $OUTBASE $ROOT/logs
echo "OUTBASE=$OUTBASE POLICIES=$POLICIES MANIFEST=$MANIFEST PAIRS=$PAIRS"
sha256sum $STEP0 $PAIRS $MANIFEST $NO_IMAGE

echo "=== leakage assertion $(date)"
"${VERL_VENV:?set VERL_VENV}/bin/python3" $ROOT/code/rl_check_disjoint.py --data $ROOT/data/verl_v1 \
  --val-images $VAL_IMAGES --rad $RAD | tee $OUTBASE/leakage_check.txt
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "LEAKAGE_CHECK_FAILED"; exit 3; }

export HF_HOME=$RAD/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f ~/.hf_env ] && source ~/.hf_env
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
export WRITER_MODEL=BASE
start_vllm "$MODEL" "$MAXLEN" --gpu-memory-utilization 0.92 ${VLLM_EXTRA:-}
cd $RAD/code
check() {  # NAME LOGFILE
  local line rows fails
  line=$(grep "^DONE rows=" "$2" | tail -1)
  rows=$(echo "$line" | sed -E 's/.*rows=([0-9]+).*/\1/'); fails=$(echo "$line" | sed -E 's/.*failures=([0-9]+).*/\1/')
  [ -n "$rows" ] && [ "$rows" != "$line" ] && [ "$rows" -gt 0 ] || { echo "ARM_EMPTY $1"; exit 4; }
  python3 -c "import sys; sys.exit(0 if $fails <= 0.10 * $rows else 1)" || { echo "ARM_TOO_MANY_FAILURES $1 $fails/$rows"; exit 4; }
}
IFS='+' read -ra PL <<< "$POLICIES"
for P in "${PL[@]}"; do
  O=$OUTBASE/$P
  echo "=== policy $P record $(date)"
  LOG=$(mktemp)
  python3 $ROOT/code/rl_control_records.py --rows $STEP0 --manifest $MANIFEST --policy $P --train-labels $TRAIN_LABELS \
    --exclude-paths $VAL_IMAGES --oracle-labels $LABELS ${COMPACT:+--compact} ${BASE_ROWS:+--base-rows $ROOT/$BASE_ROWS} ${TRAINED_ROWS:+--trained-rows $ROOT/$TRAINED_ROWS} --model "$MODEL" --base-url "$BASE_URL" --out $O/record --workers 8 | tee "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM_FAILED $P record"; exit 4; }
  check "$P record" "$LOG"
  echo "=== policy $P swap_cached $(date)"
  python3 $ROOT/code/rl_swap_adjudicate.py --rows $O/record/rows.jsonl --pairs $PAIRS --per-cell ${SWAP_PER_CELL:-200} --cell-skip 0 \
    --labels $LABELS --model "$MODEL" --base-url "$BASE_URL" --out $O/swapcached --workers 8 | tee "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM_FAILED $P swap_cached"; exit 4; }
  check "$P swap_cached" "$LOG"; rm -f "$LOG"
  python3 $ROOT/code/rl_score_directional.py --rows $O/record/rows.jsonl --swap-rows $O/swapcached/rows.jsonl \
    --no-image $NO_IMAGE --manifest $MANIFEST --out $O/directional || { echo "DIRECTIONAL_FAILED $P"; exit 5; }
done
cleanup
python3 $ROOT/code/rl_controls_summary.py $OUTBASE || { echo "SUMMARY_FAILED"; exit 5; }
sha256sum $OUTBASE/*/directional/directional.csv $OUTBASE/controls_summary.csv
echo "CONTROLS_EXIT rc=0 $(date)"
