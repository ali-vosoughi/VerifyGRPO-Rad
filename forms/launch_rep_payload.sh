#!/bin/bash
# Replication payload factorial (declared 2026-09-26 15:00 and 17:45, before any replicated record was read):
# per seed, after its evaluation-v2 read succeeds, each judge re-reads the cached step-40 records under canon_readj,
# notes_removed, flags_only and side_neutral (one controls job per judge). Base arms exist from the discovery phase.
# Usage: launch_rep_payload.sh <qos> seed=readjob [seed=readjob ...]
set -euo pipefail
ROOT=${CLAIMBLIND_ROOT:-.}
cd $ROOT
Q=$1; shift
S=code/rl_controls.sbatch
P=canon_readj+notes_removed+flags_only+side_neutral
for spec in "$@"; do
  SEED=${spec%%=*}; R=${spec#*=}
  ROWS=results/eval2/rep_s${SEED}_step_40/sweep_gated_default/rows.jsonl
  A=$(sbatch --parsable --dependency=afterok:$R --kill-on-invalid-dep=yes -p $Q -q $Q --gres=gpu:a100:1 -J rl_payrep_q_$SEED \
      --export=ALL,ROWS_SRC=$ROWS,OUTDIR=results/eval2/payload_rep/qwen_s$SEED,POLICIES=$P $S)
  B=$(sbatch --parsable --dependency=afterok:$R --kill-on-invalid-dep=yes -p $Q -q $Q --gres=gpu:rtx_4090:1 -J rl_payrep_m_$SEED \
      --export=ALL,MODEL=google/medgemma-4b-it,MAXLEN=8192,ROWS_SRC=$ROWS,OUTDIR=results/eval2/payload_rep/medgemma_s$SEED,POLICIES=$P $S)
  echo "seed $SEED: read $R -> training judge $A, independent judge $B"
done
