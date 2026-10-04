#!/bin/bash
# Second-writer reads (protocol declared 2026-09-26 19:25). Two phases per record set:
# (A) the Qwen2.5-VL writer (base or adapter) writes the records with the unchanged evaluation-v2 harness; the verdicts
#     it also produces come from the served Qwen2.5-VL base and are DISCARDED;
# (B) the cached records are re-judged as written by the Qwen3-VL-8B training judge and by MedGemma
#     (rl_xadj.sbatch, readj + cached swap + directional), and reread in the training form by both
#     (rl_controls.sbatch, canon_readj).
# Usage: launch_w2_reads.sh <tag> <adapter|none> [dependency_job]
set -euo pipefail
ROOT=${CLAIMBLIND_ROOT:-.}
cd $ROOT
TAG=$1; AD=$2; DEP=${3:-}
D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP --kill-on-invalid-dep=yes"
OUT=results/eval2/w2_$TAG
A=$(sbatch --parsable $D -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a100:1 -t 8:00:00 -J rl_w2read_$TAG \
    --export=ALL,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,ADAPTER=$AD,OUT=$OUT,PAIRS=results/val_clean/pairs_val_clean_in.jsonl,CELL_SKIP=0,PER_CELL=200,NO_IMAGE=results/eval2/val_step_0/no_image/rows.jsonl,LABELS=results/labels_eval_dev_val.jsonl,MANIFEST=results/val_clean/manifest_image_necessary.txt \
    code/rl_eval_v2.sbatch)
ROWS=$OUT/sweep_gated_default/rows.jsonl
Q3=$(sbatch --parsable --dependency=afterok:$A --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a100:1 -J rl_w2q3_$TAG \
    --export=ALL,ADJ_MODEL=Qwen/Qwen3-VL-8B-Instruct,MAXLEN=16384,ROWS=$ROWS,OUT=results/eval2/xadj_qwen3/w2_$TAG code/rl_xadj.sbatch)
MG=$(sbatch --parsable --dependency=afterok:$A --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:rtx_4090:1 -J rl_w2mg_$TAG \
    --export=ALL,ROWS=$ROWS,OUT=results/eval2/xadj_medgemma/w2_$TAG code/rl_xadj.sbatch)
CQ=$(sbatch --parsable --dependency=afterok:$A --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a100:1 -J rl_w2cq_$TAG \
    --export=ALL,ROWS_SRC=$ROWS,OUTDIR=results/eval2/canon/qwen_w2_$TAG,POLICIES=canon_readj code/rl_controls.sbatch)
CM=$(sbatch --parsable --dependency=afterok:$A --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:rtx_4090:1 -J rl_w2cm_$TAG \
    --export=ALL,MODEL=google/medgemma-4b-it,MAXLEN=8192,ROWS_SRC=$ROWS,OUTDIR=results/eval2/canon/medgemma_w2_$TAG,POLICIES=canon_readj code/rl_controls.sbatch)
echo "w2 $TAG: records $A -> qwen3 judge $Q3, medgemma $MG, canonical qwen3 $CQ, canonical medgemma $CM"
