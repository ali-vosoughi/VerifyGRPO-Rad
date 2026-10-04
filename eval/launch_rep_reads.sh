#!/bin/bash
# REPLICATION reads (2026-09-26; protocol frozen before the runs). Per seed, after its training job succeeds (the GRPO
# sbatch converts every save to a PEFT adapter at the end of the run): evaluation v2 of the step-40 adapter on the
# 238-pair manifest exactly as every checkpoint read, then the MedGemma re-judge of
# the cached records. The flag reader and the endpoint analysis run on CPU afterwards (rl_rep_endpoints.py).
# Usage: launch_rep_reads.sh <qos> seed=trainjob [seed=trainjob ...]   (the qos name doubles as the
# partition name; the read uses one A6000 freed by the finished run)
set -euo pipefail
ROOT=${CLAIMBLIND_ROOT:-.}
cd $ROOT
Q=$1; shift
for spec in "$@"; do
  S=${spec%%=*}; J=${spec#*=}; E=rep_s$S; STEP=${STEP:-40}
  if [ -d results/eval2/${E}_step_$STEP/directional ]; then echo "ALREADY_READ $E step $STEP"; continue; fi
  R=$(sbatch --parsable --dependency=afterok:$J --kill-on-invalid-dep=yes -p $Q -q $Q --gres=gpu:a6000:1 -t 8:00:00 \
      -J rl_read_${E}_$STEP \
      --export=ALL,ADAPTER=runs/verl/$E/global_step_$STEP/hf/lora_adapter,OUT=results/eval2/${E}_step_$STEP,PAIRS=results/val_clean/pairs_val_clean_in.jsonl,CELL_SKIP=0,PER_CELL=200,NO_IMAGE=results/eval2/val_step_0/no_image/rows.jsonl,LABELS=results/labels_eval_dev_val.jsonl,MANIFEST=results/val_clean/manifest_image_necessary.txt \
      code/rl_eval_v2.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$R --kill-on-invalid-dep=yes -p $Q -q $Q --gres=gpu:rtx_4090:1 \
      -J rl_xadj_${E}_$STEP --export=ALL,ROWS=results/eval2/${E}_step_$STEP/sweep_gated_default/rows.jsonl,OUT=results/eval2/xadj_medgemma/${E}_step_$STEP \
      code/rl_xadj.sbatch)
  echo "$E step $STEP: train $J -> read $R -> medgemma $X"
done
