#!/bin/bash
# Payload factorial on the discovery pair set (2026-09-26; secondary analysis declared in the replication
# protocol frozen 14:40). Records are the CACHED base and dose step-40 records; each arm changes one part of the
# record and re-judges it. Qwen notes_removed exists (an earlier controls read); this adds Qwen flags_only and
# MedGemma notes_removed + flags_only. Runs on A100 / RTX 4090 GPUs so the replication keeps the A6000s.
set -euo pipefail
ROOT=${CLAIMBLIND_ROOT:-.}
cd $ROOT
S=code/rl_controls.sbatch
B=results/eval2/val_step_0/sweep_gated_default/rows.jsonl
D=results/eval2/dose_v1_lr1e-4_step_40/sweep_gated_default/rows.jsonl
for f in $B $D; do [ -s $f ] || { echo MISSING $f; exit 2; }; done
Q="-p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a100:1"
M="-p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:rtx_4090:1"
sbatch $Q -J rl_pay_qwen_base   --export=ALL,ROWS_SRC=$B,OUTDIR=results/eval2/payload/qwen_base,POLICIES=flags_only $S
sbatch $Q -J rl_pay_qwen_dose40 --export=ALL,ROWS_SRC=$D,OUTDIR=results/eval2/payload/qwen_dose40,POLICIES=flags_only $S
sbatch $M -J rl_pay_mg_base     --export=ALL,MODEL=google/medgemma-4b-it,MAXLEN=8192,ROWS_SRC=$B,OUTDIR=results/eval2/payload/medgemma_base,POLICIES=notes_removed+flags_only $S
sbatch $M -J rl_pay_mg_dose40   --export=ALL,MODEL=google/medgemma-4b-it,MAXLEN=8192,ROWS_SRC=$D,OUTDIR=results/eval2/payload/medgemma_dose40,POLICIES=notes_removed+flags_only $S
