#!/bin/bash
# OMISSION-COST LOCKED READ (protocol registered before the runs: "the oc adapters read on the locked set once,
# after their validation endpoints"). Exactly the replication's locked
# read: evaluation v2 of each step-40 adapter on the locked pairs (A100, per cell 40),
# then the MedGemma re-judge of the cached records (RTX 4090). Refuses existing output directories (read once).
set -euo pipefail
ROOT=${CLAIMBLIND_ROOT:-.}
cd $ROOT
LP=results/locked/pairs_e6_2026-09-19_0405.jsonl; LM=results/locked/manifest_image_necessary.txt; LN=results/eval2/locked_base/no_image/rows.jsonl
for SEED in 101 202 303; do
  A=runs/verl/oc_s$SEED/global_step_40/hf/lora_adapter; OUT=results/eval2/locked_oc_s${SEED}_step_40
  [ -e "$OUT" ] && { echo "EXISTS $OUT (the locked set is read once)"; exit 2; }
  [ -s "$A/adapter_model.safetensors" ] || { echo "MISSING $A"; exit 2; }
  R=$(sbatch --parsable -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a100:1 -t 8:00:00 -J rl_lockoc_s$SEED \
      --export=ALL,ADAPTER=$A,OUT=$OUT,PAIRS=$LP,CELL_SKIP=0,PER_CELL=40,NO_IMAGE=$LN,LABELS=results/labels_eval_dev_val.jsonl,MANIFEST=$LM \
      code/rl_eval_v2.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$R --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:rtx_4090:1 -J rl_lockocmg_s$SEED \
      --export=ALL,ROWS=$OUT/sweep_gated_default/rows.jsonl,OUT=results/eval2/xadj_medgemma/locked_oc_s${SEED}_step_40,PAIRS=$LP,MANIFEST=$LM,NO_IMAGE=$LN,PER_CELL=40,CELL_SKIP=0 \
      code/rl_xadj.sbatch)
  echo "locked oc_s$SEED: read $R -> medgemma $X"
done
