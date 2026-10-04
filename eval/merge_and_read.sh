#!/bin/bash
# For a verl run whose checkpoints were not merged in the training job: convert saved checkpoints and read them (2026-09-25).
# usage: merge_and_read.sh <exp_name> <step> [<step> ...]
# Per step: CPU merge job (rl_merge_ckpt.sbatch) -> afterok evaluation-v2 read with the same arguments
# as launch_reads.sh (238 image-necessary validation pairs, cached-swap pass, directional scoring, the
# step-0 no-image rows and the frozen manifest). For adapters that already exist use launch_reads.sh.
set -uo pipefail
B=${CLAIMBLIND_ROOT:-.}
EXP=$1; shift
cd $B
for S in "$@"; do
  if [ -d results/eval2/${EXP}_step_$S/directional ]; then echo "ALREADY_READ ${EXP} step_$S"; continue; fi
  M=$(sbatch --parsable --export=ALL,EXP=$EXP,STEP=$S -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} -J rl_merge_$S code/rl_merge_ckpt.sbatch) || { echo "MERGE_SUBMIT_FAILED $S"; continue; }
  A=runs/verl/$EXP/global_step_$S/hf/lora_adapter
  J=$(sbatch --parsable --dependency=afterok:$M --kill-on-invalid-dep=yes \
    --export=ALL,ADAPTER=$A,OUT=results/eval2/${EXP}_step_$S,PAIRS=results/val_clean/pairs_val_clean_in.jsonl,CELL_SKIP=0,PER_CELL=200,NO_IMAGE=results/eval2/val_step_0/no_image/rows.jsonl,LABELS=results/labels_eval_dev_val.jsonl,MANIFEST=results/val_clean/manifest_image_necessary.txt \
    -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} -t 6:00:00 -J rl_read_${S} code/rl_eval_v2.sbatch) && echo "STEP ${S}: merge ${M} -> read ${J}"
done
squeue -u "$USER" -h -o "%.9i %.18j %.3t %.8M %R" | grep -E "rl_merge|rl_read" | head -20
