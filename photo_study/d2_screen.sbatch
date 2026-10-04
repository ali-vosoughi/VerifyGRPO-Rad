#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J d2_screen
#SBATCH -o logs/%x_%j.out
# Second domain: the registered automated presence screen (protocol section 2, revised 2026-09-30). Runs once, before
# any writer record, on every pool image; writes domain2/study/{labels.json, screen.jsonl, screen_pass.txt}.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) D2_SCREEN"
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
ROOT=${CLAIMBLIND_ROOT:-.}
D=$ROOT/domain2; S=$D/study; mkdir -p $S
[ -s $S/screen.jsonl ] && { echo "SCREEN_EXISTS: the screen runs once"; exit 9; }
export HF_HOME=$RAD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
cd $D/code
sha256sum d2_screen.py d2_forms.py $D/pool_v8_cap60_n20/pool.jsonl | tee $S/screen_hashes.txt
# Screener changed before any study read (smoke 23339, 2026-09-30): InternVL2_5-8B's remote-code InternLM2 tokenizer fails
# to load in this vLLM / Transformers stack ("piece must not include null character"); Idefics3-8B-Llama3 is native in vLLM.
SCREEN_MODEL=${SCREEN_MODEL:-HuggingFaceM4/Idefics3-8B-Llama3}
start_vllm "$SCREEN_MODEL" 8192 --gpu-memory-utilization 0.92 ${SCREEN_EXTRA:-}
python3 d2_screen.py --pool ${POOL_OVERRIDE:-$D/pool_v8_cap60_n20/pool.jsonl} --model "$SCREEN_MODEL" --base-url "$BASE_URL" --out ${OUT_OVERRIDE:-$S} || { echo "SCREEN_FAILED"; exit 5; }
echo "D2_SCREEN_EXIT rc=0 $(date)"
