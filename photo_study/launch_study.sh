#!/bin/bash
# Launch the frozen photograph study (protocol section 8, steps 4-5), explicitly, once. Usage: launch_study.sh DIGEST
# Writers (both, 1 GPU each) first; the primary judges start after both writers succeed; the secondary panel starts after
# both primary judges succeed (protocol order: primary pair first). Qwen3-VL-30B-A3B gets 2 A6000 and TP=2. Variables
# reach the job through `env ... sbatch --export=ALL` (never through --export lists, which split at commas).
# DIGEST is the sha256 of domain2/study/FROZEN.sha256 printed by d2_freeze.sh. The study itself submitted these same
# jobs from a remote shell; this is the cluster-side form of that launcher.
set -euo pipefail
DIG=${1:?manifest digest}
cd "${CLAIMBLIND_ROOT:?set CLAIMBLIND_ROOT (absolute path)}/domain2"
mkdir -p logs
S="sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --export=ALL"
E="env MANIFEST_SHA256=$DIG"
W1=$($E STAGE=writers WRITER=Qwen/Qwen3-VL-8B-Instruct $S -J d2_w_q3 code/d2_study.sbatch)
W2=$($E STAGE=writers WRITER=Qwen/Qwen2.5-VL-7B-Instruct $S -J d2_w_q25 code/d2_study.sbatch)
D1="--dependency=afterok:$W1:$W2 --kill-on-invalid-dep=yes"
J1=$($E STAGE=judges JUDGE=Qwen/Qwen3-VL-8B-Instruct $S -J d2_j_q8 $D1 code/d2_study.sbatch)
J2=$($E STAGE=judges JUDGE=google/gemma-3-12b-it $S -J d2_j_gem $D1 code/d2_study.sbatch)
D2="--dependency=afterok:$J1:$J2 --kill-on-invalid-dep=yes"
J3=$($E STAGE=judges JUDGE=Qwen/Qwen3-VL-30B-A3B-Instruct TP=2 $S -J d2_j_q30 --gres=gpu:a6000:2 --mem=96G $D2 code/d2_study.sbatch)
J4=$($E STAGE=judges JUDGE=microsoft/Phi-4-mini-instruct $S -J d2_j_phi $D2 code/d2_study.sbatch)
J5=$($E STAGE=judges JUDGE=allenai/OLMo-2-1124-7B-Instruct $S -J d2_j_olmo $D2 code/d2_study.sbatch)
echo "WRITERS $W1 $W2 PRIMARY $J1 $J2 SECONDARY $J3 $J4 $J5"
