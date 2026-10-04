#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J d2_conv_probe
#SBATCH -o logs/%x_%j.out
# ENGINEERING probe of the convention wording on out-of-pool smoke pairs + the synthetic comprehension set (no study
# data). Both primary judges at their pinned commits; writer records reused from smoke run 23291. Adoption rule fixed in
# d2_conv_probe.py before the run. Fails loud.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) D2_CONV_PROBE"
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
ROOT=${CLAIMBLIND_ROOT:-.}
D=$ROOT/domain2; OUT=$D/smoke/probe_$SLURM_JOB_ID; mkdir -p $OUT
export HF_HOME=$RAD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
cd ${CODE:-$D/code}
rev() { awk -v m="$1" '$1==m {print $2}' $D/study/model_revisions.txt; }
stop() { cleanup; wait "$VPID" 2>/dev/null; VPID=""; sleep 10; }
for J in Qwen/Qwen3-VL-8B-Instruct google/gemma-3-12b-it; do
  R=$(rev "$J"); [ -n "$R" ] || { echo "NO_PINNED_REVISION $J"; exit 9; }
  start_vllm "$J" 8192 --gpu-memory-utilization 0.92 --revision "$R" --tokenizer-revision "$R"
  python3 d2_conv_probe.py --model "$J" --base-url "$BASE_URL" --smoke-dir $D/smoke --writer-run $D/smoke/run_23291 \
    --out $OUT/$(echo $J | tr '/' '_').json ${ONLY:+--only $ONLY} || { echo "PROBE_FAILED $J"; exit 5; }
  stop
done
echo "D2_PROBE_EXIT rc=0 $(date)"
