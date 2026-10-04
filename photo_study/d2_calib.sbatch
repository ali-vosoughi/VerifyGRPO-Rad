#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -J d2_calib
#SBATCH -o logs/%x_%j.out
# CONVENTION CALIBRATION (engineering, before the freeze): calib_v2 = 41 out-of-pool pairs in 17 categories (no study
# image). Both writers write fresh sparse records at their pinned commits; then each primary judge scores K3, K4, K5 on
# the synthetic comprehension set and the target controls with the strict parser. Adoption rule logged in the project log
# before submission. Fails loud.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) D2_CALIB"
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
ROOT=${CLAIMBLIND_ROOT:-.}
D=$ROOT/domain2; CAL=$D/calib_v2; OUT=$CAL/run_$SLURM_JOB_ID; mkdir -p $OUT
export HF_HOME=$RAD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
cd $D/code_r2
sha256sum d2_forms.py d2_judge.py d2_write_records.py d2_conv_probe.py $CAL/pool.jsonl
rev() { awk -v m="$1" '$1==m {print $2}' $D/study/model_revisions.txt; }
stop() { cleanup; wait "$VPID" 2>/dev/null; VPID=""; sleep 10; }
for W in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-VL-7B-Instruct; do
  R=$(rev "$W"); [ -n "$R" ] || { echo "NO_PINNED_REVISION $W"; exit 9; }
  start_vllm "$W" 16384 --gpu-memory-utilization 0.92 --revision "$R" --tokenizer-revision "$R"
  python3 d2_write_records.py --images $CAL/images.txt --model "$W" --base-url "$BASE_URL" --style sparse \
    --out $OUT/writer/$(echo $W | tr '/' '_')_sparse || { echo "WRITER_FAILED $W"; exit 3; }
  stop
done
for J in Qwen/Qwen3-VL-8B-Instruct google/gemma-3-12b-it; do
  R=$(rev "$J"); [ -n "$R" ] || { echo "NO_PINNED_REVISION $J"; exit 9; }
  start_vllm "$J" 8192 --gpu-memory-utilization 0.92 --revision "$R" --tokenizer-revision "$R"
  python3 d2_conv_probe.py --model "$J" --base-url "$BASE_URL" --smoke-dir $CAL --writer-run $OUT \
    --out $OUT/probe_$(echo $J | tr '/' '_').json --only K3_list_rule_plus_precedence,K4_objects_field,K5_explicit_default \
    || { echo "PROBE_FAILED $J"; exit 5; }
  stop
done
echo "D2_CALIB_EXIT rc=0 $(date)"
