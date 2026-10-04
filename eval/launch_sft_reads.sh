#!/bin/bash
# chain the reads of the direct-supervision adapters (2026-09-26). Per arm, after its SFT job succeeds:
# evaluation v2 on the 238-pair manifest exactly as an RL checkpoint read (merge_and_read.sh arguments), then
# the MedGemma re-judge of the cached records. The flag reader runs on CPU afterwards. A100 / RTX 4090
# GPUs, so the replication keeps the A6000s. Usage: launch_sft_reads.sh arm=jobid [arm=jobid ...]
set -euo pipefail
ROOT=${CLAIMBLIND_ROOT:-.}
cd $ROOT
for spec in "$@"; do
  ARM=${spec%%=*}; J=${spec#*=}
  if [ -d results/eval2/$ARM/directional ]; then echo "ALREADY_READ $ARM"; continue; fi
  R=$(sbatch --parsable --dependency=afterok:$J --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a100:1 -t 8:00:00 \
      -J rl_read_$ARM \
      --export=ALL,ADAPTER=runs/sft/$ARM/lora_adapter,OUT=results/eval2/$ARM,PAIRS=results/val_clean/pairs_val_clean_in.jsonl,CELL_SKIP=0,PER_CELL=200,NO_IMAGE=results/eval2/val_step_0/no_image/rows.jsonl,LABELS=results/labels_eval_dev_val.jsonl,MANIFEST=results/val_clean/manifest_image_necessary.txt \
      code/rl_eval_v2.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$R --kill-on-invalid-dep=yes -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:rtx_4090:1 \
      -J rl_xadj_$ARM --export=ALL,ROWS=results/eval2/$ARM/sweep_gated_default/rows.jsonl,OUT=results/eval2/xadj_medgemma/$ARM \
      code/rl_xadj.sbatch)
  echo "$ARM: sft $J -> read $R -> medgemma $X"
done
