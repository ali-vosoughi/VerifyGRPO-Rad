#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -J d2_smoke
#SBATCH -o logs/%x_%j.out
# Second domain ENGINEERING smoke (2026-09-29; protocol DRAFT section 8.3): out-of-pool pairs only (domain2/smoke, no
# study image) and the synthetic comprehension set. Both writers x both styles on the smoke images, then each 1-GPU
# judge: COMP (96 synthetic cases) + every study arm on the smoke pairs, then the frozen analysis on the smoke rows as a
# pipeline check (never a result). Fails loud. JUDGES may be overridden (comma list).
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) D2_SMOKE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
ROOT=${CLAIMBLIND_ROOT:-.}
D=$ROOT/domain2
SM=${SMOKE_DIR:-$D/smoke}; OUT=$SM/run_$SLURM_JOB_ID; mkdir -p $OUT $ROOT/logs
export HF_HOME=$RAD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
cd $D/code
python3 d2_forms.py --selftest || { echo "SELFTEST_FAILED"; exit 2; }
python3 d2_analyze.py --selftest || { echo "ANALYSIS_SELFTEST_FAILED"; exit 2; }
sha256sum d2_forms.py d2_write_records.py d2_judge.py d2_analyze.py $SM/pool.jsonl
REVS=$D/study/model_revisions.txt   # the same pinned commits as the study (loaded offline with --revision)
rev() { awk -v m="$1" '$1==m {print $2}' $REVS; }
NG=$(nvidia-smi -L | wc -l); [ "$NG" -ge "${TP:-1}" ] || { echo "GPUS_BELOW_TP allocated=$NG TP=${TP:-1}"; exit 9; }
stop() { cleanup; wait "$VPID" 2>/dev/null; VPID=""; sleep 10; }
chk() {  # LOGFILE NAME
  local line rows fails; line=$(grep "^DONE rows=" "$1" | tail -1)
  rows=$(echo "$line" | sed -E 's/.*rows=([0-9]+).*/\1/'); fails=$(echo "$line" | sed -E 's/.*failures=([0-9]+).*/\1/')
  [ -n "$rows" ] && [ "$rows" != "$line" ] && [ "$rows" -gt 0 ] || { echo "EMPTY $2"; exit 4; }
  python3 -c "import sys; sys.exit(0 if $fails <= 0.10 * $rows else 1)" || { echo "TOO_MANY_FAILURES $2 $fails/$rows"; exit 4; }
}
for W in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-VL-7B-Instruct; do
  if [ -n "${WRITER_FROM:-}" ]; then   # reuse the writer records of an earlier smoke run (judge-side changes only)
    mkdir -p $OUT/writer; cp -r $WRITER_FROM/writer/$(echo $W | tr '/' '_')_* $OUT/writer/ || exit 3; continue
  fi
  R=$(rev "$W"); [ -n "$R" ] || { echo "NO_PINNED_REVISION $W"; exit 9; }
  start_vllm "$W" 16384 --gpu-memory-utilization 0.92 --revision "$R" --tokenizer-revision "$R"
  for ST in sparse checklist; do
    L=$(mktemp); tag=$(echo $W | tr '/' '_')_$ST
    python3 d2_write_records.py --images $SM/images.txt --model "$W" --base-url "$BASE_URL" --style $ST \
      --out $OUT/writer/$tag | tee $L; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "WRITER_FAILED $tag"; exit 3; }; chk $L "$tag"
  done
  stop
done
[ -n "${WRITERS_ONLY:-}" ] && { echo "WRITERS_ONLY done"; echo "D2_SMOKE_EXIT rc=0 $(date)"; exit 0; }
IFS=',' read -ra JL <<< "${JUDGES:-Qwen/Qwen3-VL-8B-Instruct,microsoft/Phi-4-mini-instruct,allenai/OLMo-2-1124-7B-Instruct}"
ARMS=A_unfilled,A_absent,A_notreported,B_complete,B_sparse,C_corr_complete,C_corr_sparse,C_rev_complete,C_rev_sparse,T_complete,T_sparse
for J in "${JL[@]}"; do
  JL_MAX=8192; case "$J" in *OLMo*) JL_MAX=4096;; esac   # OLMo-2 max_position_embeddings 4096 (smoke 23273)
  R=$(rev "$J"); [ -n "$R" ] || { echo "NO_PINNED_REVISION $J"; exit 9; }
  start_vllm "$J" $JL_MAX --gpu-memory-utilization 0.92 --tensor-parallel-size ${TP:-1} --revision "$R" --tokenizer-revision "$R"
  jt=$(echo $J | tr '/' '_')
  L=$(mktemp); python3 d2_judge.py --model "$J" --base-url "$BASE_URL" --arms COMP --out $OUT/judge/$jt/COMP | tee $L
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "JUDGE_FAILED $jt COMP"; exit 5; }; chk $L "$jt COMP"
  python3 d2_judge.py --base-url http://127.0.0.1:0 --verify-only --model "$J" --arms COMP --out $OUT/judge/$jt/COMP || exit 5
  python3 d2_judge.py --model "$J" --base-url "$BASE_URL" --arms COMP --out $OUT/judge/$jt/COMP >/dev/null 2>&1 \
    && { echo "IMMUTABILITY_BROKEN"; exit 7; } || echo "IMMUTABILITY_OK second call refused"
  for W in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-VL-7B-Instruct; do
    wt=$(echo $W | tr '/' '_')
    L=$(mktemp); python3 d2_judge.py --model "$J" --base-url "$BASE_URL" --arms $ARMS --pool $SM/pool.jsonl \
      --labels $SM/labels.json --records $OUT/writer/${wt}_sparse/rows.jsonl --style sparse --writer "$W" \
      --out $OUT/judge/$jt/$wt | tee $L
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "JUDGE_FAILED $jt $wt"; exit 5; }; chk $L "$jt $wt"
    python3 d2_judge.py --base-url http://127.0.0.1:0 --verify-only --model "$J" --arms $ARMS --pool $SM/pool.jsonl \
      --labels $SM/labels.json --records $OUT/writer/${wt}_sparse/rows.jsonl --style sparse --writer "$W" \
      --out $OUT/judge/$jt/$wt || exit 5
  done
  stop
done
ROWS=$(ls $OUT/judge/*/*/rows.jsonl | paste -sd,); WROWS=$(ls $OUT/writer/*/rows.jsonl | paste -sd,)
python3 -c "import json,sys; [print(json.loads(l)['pair_id']) for l in open(sys.argv[1]) if l.strip()]" $SM/pool.jsonl > $OUT/all_pairs.txt
python3 d2_analyze.py --judge-rows "$ROWS" --writer-rows "$WROWS" --pool $SM/pool.jsonl --labels $SM/labels.json   --screen-pass $OUT/all_pairs.txt --expect-pairs 0 --out $OUT/analysis_pipeline_check || { echo "ANALYSIS_FAILED"; exit 6; }
python3 - $OUT/analysis_pipeline_check/analysis.json <<'EOF2'
import json, sys
g = json.load(open(sys.argv[1]))["primary_all_pool_pairs"]["gates"]
for k, v in g.items():
    if k != "all_pass":
        print("GATE %-90s rows=%s agreement=%.3f pass=%s" % (k, v["rows"], v["agreement"], v["pass"]))
print("GATES_ALL_PASS", g["all_pass"])
sys.exit(0 if g["all_pass"] else 1)
EOF2
[ $? -eq 0 ] || { echo "CRITERIA_FAILED"; exit 6; }
echo "D2_SMOKE_EXIT rc=0 $(date)"
