#!/bin/bash
# The frozen analysis, ONCE (protocol section 8.6), after every study judge job finished. Verifies the manifest digest and
# every registered file, checks that every expected judge output verifies (d2_judge.py --verify-only), runs d2_analyze.py
# on a compute node into study/analysis_frozen (the runner refuses an existing directory), then d2_report.py (reads only).
# Usage: run_frozen_analysis.sh DIGEST   (the sha256 of domain2/study/FROZEN.sha256 printed by d2_freeze.sh)
# The study ran these same commands from a remote shell; this is the cluster-side form of that runner.
set -euo pipefail
DIG=${1:?manifest digest}
D="${CLAIMBLIND_ROOT:?set CLAIMBLIND_ROOT (absolute path)}/domain2"; S=$D/study
cd $D/code
[ "$(sha256sum $S/FROZEN.sha256 | awk '{print $1}')" = "$DIG" ] || { echo MANIFEST_DIGEST_MISMATCH; exit 9; }
sha256sum --quiet -c $S/FROZEN.sha256 && echo MANIFEST_VERIFIES
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
POOL=$D/pool_v8_cap60_n20/pool.jsonl
A=A_unfilled,A_absent,A_notreported,B_complete,B_sparse,C_corr_complete,C_corr_sparse,C_rev_complete,C_rev_sparse,T_complete,T_sparse
for J in Qwen/Qwen3-VL-8B-Instruct google/gemma-3-12b-it Qwen/Qwen3-VL-30B-A3B-Instruct microsoft/Phi-4-mini-instruct allenai/OLMo-2-1124-7B-Instruct; do
  jt=$(echo $J | tr '/' '_')
  python3 d2_judge.py --base-url http://127.0.0.1:0 --verify-only --model $J --arms COMP --out $S/judge/$jt/COMP
  for W in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-VL-7B-Instruct; do
    wt=$(echo $W | tr '/' '_')
    python3 d2_judge.py --base-url http://127.0.0.1:0 --verify-only --model $J --arms $A --pool $POOL --labels $S/labels.json \
      --records $S/writer/${wt}_sparse/rows.jsonl --style sparse --writer $W --out $S/judge/$jt/${wt}_sparse
    python3 d2_judge.py --base-url http://127.0.0.1:0 --verify-only --model $J --arms A_unfilled,A_absent,A_notreported --pool $POOL \
      --labels $S/labels.json --records $S/writer/${wt}_checklist/rows.jsonl --style checklist --writer $W --out $S/judge/$jt/${wt}_checklist
  done
done
JR=$(ls $S/judge/*/*/rows.jsonl | paste -sd,); WR=$(ls $S/writer/*/rows.jsonl | paste -sd,)
echo judge_row_files $(echo $JR | tr ',' '\n' | wc -l) writer_row_files $(echo $WR | tr ',' '\n' | wc -l)
srun -p ${PARTITION:?set PARTITION} -q ${QOS:?set QOS} -c 4 --mem=32G -t 1:00:00 python3 d2_analyze.py --judge-rows $JR --writer-rows $WR \
  --pool $POOL --labels $S/labels.json --screen-pass $S/screen_pass.txt --screen-rows $S/screen.jsonl --out $S/analysis_frozen
sha256sum $S/analysis_frozen/analysis.json | tee $S/analysis_frozen/analysis.sha256
python3 d2_report.py $S/analysis_frozen/analysis.json $S/analysis_frozen/report
echo FROZEN_ANALYSIS_DONE
