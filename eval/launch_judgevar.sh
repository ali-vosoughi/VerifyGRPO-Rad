#!/bin/bash
# Judge-size and judge-prompt reads of the SAME cached records (declared 2026-09-30 23:34 EDT, post hoc, before any read).
# Usage: launch_judgevar.sh smoke | size | g27 | prompt
# Each full job = rl_controls_multi.sbatch over 4 record sets (base + s101/s202/s303) x 4 forms, one population.
# Variables are exported in the environment and passed with --export=ALL, because sbatch splits --export values at commas.
set -uo pipefail
ROOT=${CLAIMBLIND_ROOT:?set CLAIMBLIND_ROOT (absolute path)}
SB=$ROOT/code/rl_controls_multi.sbatch
FORMS=readj+canon_readj+order_only+fill_only
VAL_SETS="results/eval2/val_step_0/sweep_gated_default/rows.jsonl results/eval2/rep_s101_step_40/sweep_gated_default/rows.jsonl results/eval2/rep_s202_step_40/sweep_gated_default/rows.jsonl results/eval2/rep_s303_step_40/sweep_gated_default/rows.jsonl"
LOCKED_SETS="results/eval2/locked_base/sweep_gated_default/rows.jsonl results/eval2/locked_rep_s101_step_40/sweep_gated_default/rows.jsonl results/eval2/locked_rep_s202_step_40/sweep_gated_default/rows.jsonl results/eval2/locked_rep_s303_step_40/sweep_gated_default/rows.jsonl"
specs() {  # OUTBASE SETS...
  local ob=$1; shift; local names=(base s101 s202 s303) out=() i=0
  for s in "$@"; do out+=("$s:$ob/${names[$i]}"); i=$((i+1)); done
  local IFS=,; echo "${out[*]}"
}
clean_env() { unset PAIRS_REL MANIFEST_REL NO_IMAGE_REL JUDGE_PROMPT_VARIANT VLLM_EXTRA SMOKE OUTDIR ROWS_SRC SPECS MODEL MAXLEN POLICIES; }
submit() {  # NAME GRES MODEL MAXLEN OUTBASE POP VARIANT VLLM_EXTRA
  local name=$1 gres=$2 model=$3 maxlen=$4 ob=$5 pop=$6 variant=$7 vextra=$8 sets
  # idempotent: skip when a job of this name is queued or running, or when its last output already exists
  if squeue -u "$USER" -h -o "%j" | grep -qx "$name"; then echo "SKIP $name (already queued or running)"; return 0; fi
  if [ -s "$ROOT/$ob/s303/fill_only/swapcached/rows.jsonl" ]; then echo "SKIP $name (output exists)"; return 0; fi
  clean_env
  if [ "$pop" = val ]; then sets=$VAL_SETS; else
    sets=$LOCKED_SETS
    export PAIRS_REL=results/locked/pairs_e6_2026-09-19_0405.jsonl MANIFEST_REL=results/locked/manifest_image_necessary.txt NO_IMAGE_REL=results/eval2/locked_base/no_image/rows.jsonl
  fi
  export SPECS="$(specs "$ob" $sets)" MODEL="$model" MAXLEN="$maxlen" POLICIES="$FORMS"
  [ -n "$variant" ] && export JUDGE_PROMPT_VARIANT="$variant"
  [ -n "$vextra" ] && export VLLM_EXTRA="$vextra"
  sbatch -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=$gres -c 8 --mem=64G -t 6:00:00 -J "$name" --export=ALL "$SB" \
    || echo "SUBMIT_FAILED $name"
  clean_env
}
case "${1:-}" in
  smoke)
    # 10 manifest pairs (results/val_clean/manifest_smoke10.txt), s101 records, 2 forms; separate OUTDIR per smoke so they never clash
    SM=MANIFEST_REL=results/val_clean/manifest_smoke10.txt,ROWS_SRC=results/eval2/rep_s101_step_40/sweep_gated_default/rows.jsonl,POLICIES=readj+canon_readj
    for v in cite reason; do
      sbatch -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:1 -c 8 --mem=64G -t 1:00:00 -J rl_jv_smoke_$v \
        --export=ALL,$SM,OUTDIR=results/eval2/judgevar_smoke/q8_$v,MODEL=Qwen/Qwen3-VL-8B-Instruct,MAXLEN=16384,JUDGE_PROMPT_VARIANT=$v \
        $ROOT/code/rl_controls.sbatch
    done
    sbatch -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} --gres=gpu:a6000:1 -c 8 --mem=64G -t 1:00:00 -J rl_jv_smoke_g12 \
      --export=ALL,$SM,OUTDIR=results/eval2/judgevar_smoke/g12,MODEL=google/gemma-3-12b-it,MAXLEN=8192 \
      $ROOT/code/rl_controls.sbatch ;;
  size)
    for pop in val locked; do
      submit rl_jsz_g12_$pop gpu:a6000:1 google/gemma-3-12b-it 8192 results/eval2/judgesize/g12_$pop $pop "" ""
    done ;;
  g27)
    for pop in val locked; do
      submit rl_jsz_g27_$pop gpu:a6000:2 google/gemma-3-27b-it 8192 results/eval2/judgesize/g27_$pop $pop "" "--tensor-parallel-size 2"
    done ;;
  prompt)
    for v in cite reason; do
      for pop in val locked; do
        submit rl_jv_q8_${v}_$pop gpu:a6000:1 Qwen/Qwen3-VL-8B-Instruct 16384 results/eval2/judgeprompt/q8_${v}_$pop $pop "$v" ""
        submit rl_jv_mg_${v}_$pop gpu:a6000:1 google/medgemma-4b-it 8192 results/eval2/judgeprompt/mg_${v}_$pop $pop "$v" ""
      done
    done ;;
  rest)
    # the 5 prompt jobs that hit the submit caps at the first launch (2026-09-30 23:4x); resubmit explicitly, never from a watcher
    submit rl_jv_mg_cite_locked gpu:a6000:1 google/medgemma-4b-it 8192 results/eval2/judgeprompt/mg_cite_locked locked cite ""
    submit rl_jv_q8_reason_val gpu:a6000:1 Qwen/Qwen3-VL-8B-Instruct 16384 results/eval2/judgeprompt/q8_reason_val val reason ""
    submit rl_jv_mg_reason_val gpu:a6000:1 google/medgemma-4b-it 8192 results/eval2/judgeprompt/mg_reason_val val reason ""
    submit rl_jv_q8_reason_locked gpu:a6000:1 Qwen/Qwen3-VL-8B-Instruct 16384 results/eval2/judgeprompt/q8_reason_locked locked reason ""
    submit rl_jv_mg_reason_locked gpu:a6000:1 google/medgemma-4b-it 8192 results/eval2/judgeprompt/mg_reason_locked locked reason "" ;;
  one)
    # one named job: launch_judgevar.sh one NAME GRES MODEL MAXLEN OUTBASE POP VARIANT VLLM_EXTRA
    shift; submit "$@" ;;
  *) echo "usage: $0 smoke|size|g27|prompt|rest|one"; exit 2 ;;
esac
