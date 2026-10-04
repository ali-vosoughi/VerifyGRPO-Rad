#!/bin/bash
# OMISSION-COST reads (protocol registered 2026-09-26 22:20; launcher written 2026-09-27 03:20 EDT). Per seed,
# after its training job succeeds, exactly the replication's validation reads (launch_rep_reads.sh: evaluation v2 of
# the step-40 adapter on the 238-pair manifest, then the MedGemma re-judge of the cached records), plus the controls
# needed for the registered mechanism endpoint (training judge and MedGemma re-read the cached records under
# canon_readj, fill_only, fill_target, fill_other; the base arms exist). Hardware as the replication (writer read on one
# A6000, MedGemma on one RTX 4090, training-judge controls on one A100). The locked reads are NOT chained here: the
# protocol reads the oc adapters on the locked
# set once, after the validation endpoints.
# Usage: launch_oc_reads.sh seed=trainjob [seed=trainjob ...]
set -euo pipefail
ROOT=${CLAIMBLIND_ROOT:-.}
cd $ROOT
P=canon_readj+fill_only+fill_target+fill_other
for spec in "$@"; do
  S=${spec%%=*}; J=${spec#*=}; E=oc_s$S; STEP=40
  if [ -d results/eval2/${E}_step_$STEP/directional ]; then echo "ALREADY_READ $E step $STEP"; continue; fi
  R=$(sbatch --parsable --dependency=afterok:$J --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:1 -t 3:00:00 \
      -J rl_read_${E}_$STEP \
      --export=ALL,ADAPTER=runs/verl/$E/global_step_$STEP/hf/lora_adapter,OUT=results/eval2/${E}_step_$STEP,PAIRS=results/val_clean/pairs_val_clean_in.jsonl,CELL_SKIP=0,PER_CELL=200,NO_IMAGE=results/eval2/val_step_0/no_image/rows.jsonl,LABELS=results/labels_eval_dev_val.jsonl,MANIFEST=results/val_clean/manifest_image_necessary.txt \
      code/rl_eval_v2.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$R --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:rtx_4090:1 \
      -J rl_xadj_${E}_$STEP --export=ALL,ROWS=results/eval2/${E}_step_$STEP/sweep_gated_default/rows.jsonl,OUT=results/eval2/xadj_medgemma/${E}_step_$STEP \
      code/rl_xadj.sbatch)
  ROWS=results/eval2/${E}_step_$STEP/sweep_gated_default/rows.jsonl
  A=$(sbatch --parsable --dependency=afterok:$R --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a100:1 -J rl_occ_q_$S \
      --export=ALL,ROWS_SRC=$ROWS,OUTDIR=results/eval2/oc_controls/q$S,POLICIES=$P code/rl_controls.sbatch)
  B=$(sbatch --parsable --dependency=afterok:$R --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:rtx_4090:1 -J rl_occ_m_$S \
      --export=ALL,MODEL=google/medgemma-4b-it,MAXLEN=8192,ROWS_SRC=$ROWS,OUTDIR=results/eval2/oc_controls/m$S,POLICIES=$P code/rl_controls.sbatch)
  echo "$E: train $J -> read $R -> medgemma $X, controls training judge $A, independent judge $B"
done
