#!/bin/bash
# Submit evaluation-v2 reads for the adapters of a verl run.
# usage: launch_reads.sh <exp_name> <step> [<step> ...]   (adapters at runs/verl/<exp>/global_step_<step>/hf/lora_adapter)
# Each read: record arm on the 238 image-necessary validation pairs, cached-swap pass, directional scoring;
# reuses the step-0 no-image rows and the frozen manifest. The per-user cap of 4 GPUs queues the rest.
set -uo pipefail
B=${CLAIMBLIND_ROOT:-.}
EXP=$1; shift
# (the runs that trained on a second cluster first copied their adapters here; copy yours to the same path)
for S in "$@"; do
  A=runs/verl/$EXP/global_step_$S/hf/lora_adapter
  [ -f "$B/$A/adapter_config.json" ] || { echo "NO_ADAPTER $A"; continue; }
  J=$(cd $B && sbatch --parsable --export=ALL,ADAPTER=$A,OUT=results/eval2/${EXP}_step_$S,PAIRS=results/val_clean/pairs_val_clean_in.jsonl,CELL_SKIP=0,PER_CELL=200,NO_IMAGE=results/eval2/val_step_0/no_image/rows.jsonl,LABELS=results/labels_eval_dev_val.jsonl,MANIFEST=results/val_clean/manifest_image_necessary.txt -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} -t 6:00:00 -J rl_read_${S} code/rl_eval_v2.sbatch) && echo "READ ${EXP} step_${S} job ${J}"
done
squeue -u "$USER" -h -o "%.9i %.18j %.3t %.8M %R" | grep rl_read | head -20
