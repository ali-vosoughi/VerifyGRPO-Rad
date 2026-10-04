#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x_%j.out
# Run rl_controls.sbatch once per record set inside one allocation, so that a
# multi-read reread fits per-user submit caps. SPECS = comma-separated ROWS_SRC:OUTDIR pairs (paths relative to
# ROOT); every other variable (MODEL, MAXLEN, POLICIES, PAIRS_REL, MANIFEST_REL, NO_IMAGE_REL, SWAP_PER_CELL) passes
# through. Fails loud on the first failing record set.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) MULTI SPECS=$SPECS"
IFS=',' read -ra SP <<< "$SPECS"
for s in "${SP[@]}"; do
  export ROWS_SRC=${s%%:*} OUTDIR=${s#*:}
  echo "=== MULTI record set ROWS_SRC=$ROWS_SRC OUTDIR=$OUTDIR $(date)"
  bash ${CLAIMBLIND_ROOT:-.}/code/rl_controls.sbatch || { echo "MULTI_FAILED $s"; exit 6; }
done
echo "MULTI_EXIT rc=0 $(date)"
