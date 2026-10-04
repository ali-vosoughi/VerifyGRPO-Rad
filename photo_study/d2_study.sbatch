#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J d2_study
#SBATCH -o logs/%x_%j.out
# Second domain STUDY runs (protocol sections 8.4-8.5). Submit with MANIFEST_SHA256 = the digest of study/FROZEN.sha256
# logged in the project log at the freeze; verify() (at start and before every study call) checks that digest, that the
# manifest lists EVERY registered file, and that every listed file matches byte for byte. STAGE=writers runs both styles
# for WRITER; STAGE=judges runs JUDGE on every writer and style. Qwen3-VL-30B-A3B must be submitted with
# --gres=gpu:a6000:2 and TP=2 (enforced). Models load at the commit pinned in study/model_revisions.txt. Output
# directories are reserved atomically by the runners and never overwritten. A judge output is reused, and every new one
# accepted, only if d2_judge.py --verify-only regenerates exactly its slots for this judge, writer, style, arm set, and
# input hashes; writer records only if their DONE marker names this writer, style and image list, and they hold the
# 1,032 pool images exactly once, all by this writer in this style. Fails loud.
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) D2_STUDY STAGE=${STAGE:?} WRITER=${WRITER:-} JUDGE=${JUDGE:-} TP=${TP:-1}"
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
ROOT=${CLAIMBLIND_ROOT:-.}
D=$ROOT/domain2; S=$D/study; C=$D/code
POOL=$D/pool_v8_cap60_n20/pool.jsonl; IMAGES=$D/pool_v8_cap60_n20/images.txt
REQUIRED=($S/protocol_registered.md $POOL $IMAGES $S/labels.json $S/screen.jsonl $S/screen_pass.txt
  $S/screen_summary.json $S/model_revisions.txt $C/d2_forms.py $C/d2_write_records.py $C/d2_judge.py $C/d2_analyze.py
  $C/d2_screen.py $C/d2_screen.sbatch $C/d2_study.sbatch $C/d2_build_pool.py $C/d2_freeze.sh $C/d2_conv_probe.py
  $C/d2_conv_probe.sbatch $C/d2_smoke.sbatch $C/d2_make_smoke.py $C/d2_make_calib.py $C/d2_calib.sbatch
  $RAD/code/scripts/vllm_common.sh)
verify() {  # WHERE
  [ -s $S/FROZEN.sha256 ] || { echo "NOT_FROZEN before $1"; exit 9; }
  [ "$(sha256sum $S/FROZEN.sha256 | awk '{print $1}')" = "${MANIFEST_SHA256:?submit with the logged manifest digest}" ] \
    || { echo "MANIFEST_DIGEST_MISMATCH before $1"; exit 9; }
  local f; for f in "${REQUIRED[@]}"; do
    awk '{print $2}' $S/FROZEN.sha256 | grep -qxF "$f" || { echo "NOT_IN_MANIFEST $f before $1"; exit 9; }
  done
  sha256sum --quiet -c $S/FROZEN.sha256 || { echo "MANIFEST_MISMATCH before $1"; exit 9; }
}
verify "start"
rev() { awk -v m="$1" '$1==m {print $2}' $S/model_revisions.txt; }
NG=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
[ "$NG" -ge "${TP:-1}" ] || { echo "GPUS_BELOW_TP allocated=$NG TP=${TP:-1}"; exit 9; }
case "${JUDGE:-}" in *30B*) [ "${TP:-1}" -eq 2 ] && [ "$NG" -ge 2 ] || { echo "30B_NEEDS_TP2_ON_2_GPUS NG=$NG TP=${TP:-1}"; exit 9; };; esac
export HF_HOME=$RAD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
source $RAD/code/scripts/vllm_common.sh
cd $C
cp $S/FROZEN.sha256 $S/run_manifest_$SLURM_JOB_ID.sha256
chk() {
  local line rows fails; line=$(grep "^DONE rows=" "$1" | tail -1)
  rows=$(echo "$line" | sed -E 's/.*rows=([0-9]+).*/\1/'); fails=$(echo "$line" | sed -E 's/.*failures=([0-9]+).*/\1/')
  [ -n "$rows" ] && [ "$rows" != "$line" ] && [ "$rows" -gt 0 ] || { echo "EMPTY $2"; exit 4; }
  python3 -c "import sys; sys.exit(0 if $fails <= 0.10 * $rows else 1)" || { echo "TOO_MANY_FAILURES $2 $fails/$rows"; exit 4; }
}
writer_ok() {  # DIR MODEL STYLE -> 0 iff DONE marker matches rows and identity, and exactly the 1,032 pool images once
  python3 - "$1" "$IMAGES" "$2" "$3" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d, images, model, style = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
try:
    m = json.loads((d / "DONE.json").read_text()); b = (d / "rows.jsonl").read_bytes()
except FileNotFoundError:
    print("WRITER_NOT_DONE", d); sys.exit(1)
rows = [json.loads(l) for l in b.splitlines() if l.strip()]
ks = [r["k"] for r in rows]
want = set()
for l in images.read_text().splitlines():
    l = l.strip()
    if l:
        want.add("%s_%d" % (l.split("/")[0].replace("2014", ""), int(l.rsplit("_", 1)[1].split(".")[0])))
ident = {"model": model, "style": style, "images_sha256": hashlib.sha256(images.read_bytes()).hexdigest()}
checks = {"marker_rows": m["rows"] == m["tasks"] == len(rows), "marker_sha": hashlib.sha256(b).hexdigest() == m["rows_sha256"],
          "marker_identity": m.get("identity") == ident, "unique_1032": len(ks) == len(set(ks)) == 1032,
          "match_pool": set(ks) == want, "owner": all(r.get("model") == model and r.get("style") == style for r in rows)}
print("WRITER_RECORDS", d, checks); sys.exit(0 if all(checks.values()) else 1)
EOF
}
judge_ok() {  # the same arguments as the judge call -> d2_judge.py --verify-only
  python3 d2_judge.py --base-url http://127.0.0.1:0 --verify-only "$@"
}
if [ "$STAGE" = "writers" ]; then
  R=$(rev "$WRITER"); [ -n "$R" ] || { echo "NO_PINNED_REVISION $WRITER"; exit 9; }
  start_vllm "$WRITER" 16384 --gpu-memory-utilization 0.92 --revision "$R" --tokenizer-revision "$R"
  for ST in sparse checklist; do
    O=$S/writer/$(echo $WRITER | tr '/' '_')_$ST
    if [ -e $O ]; then
      writer_ok $O "$WRITER" $ST || { echo "INCOMPLETE_EXISTING $O (never overwritten; resolve by hand)"; exit 3; }
      echo "EXISTS_COMPLETE $O (records are written once)"; continue
    fi
    verify "writer $ST"
    L=$(mktemp); python3 d2_write_records.py --images $IMAGES --model "$WRITER" --base-url "$BASE_URL" --style $ST --out $O | tee $L
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "WRITER_FAILED $ST"; exit 3; }; chk $L "writer $ST"
    writer_ok $O "$WRITER" $ST || { echo "WRITER_INCOMPLETE $ST"; exit 3; }
    sha256sum $O/rows.jsonl | tee -a $S/outputs.sha256
  done
elif [ "$STAGE" = "judges" ]; then
  R=$(rev "$JUDGE"); [ -n "$R" ] || { echo "NO_PINNED_REVISION $JUDGE"; exit 9; }
  MAXL=8192; case "$JUDGE" in *OLMo*) MAXL=4096;; esac
  start_vllm "$JUDGE" $MAXL --gpu-memory-utilization 0.92 --tensor-parallel-size ${TP:-1} --revision "$R" --tokenizer-revision "$R"
  jt=$(echo $JUDGE | tr '/' '_')
  O=$S/judge/$jt/COMP
  if [ -e $O ]; then
    judge_ok --model "$JUDGE" --arms COMP --out $O || { echo "INCOMPLETE_EXISTING $O (never overwritten; resolve by hand)"; exit 5; }
    echo "EXISTS_COMPLETE $O"
  else
    verify "COMP"
    L=$(mktemp); python3 d2_judge.py --model "$JUDGE" --base-url "$BASE_URL" --arms COMP --out $O | tee $L
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "JUDGE_FAILED COMP"; exit 5; }; chk $L "COMP"
    judge_ok --model "$JUDGE" --arms COMP --out $O || exit 5
    sha256sum $O/rows.jsonl | tee -a $S/outputs.sha256
  fi
  for W in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-VL-7B-Instruct; do
    for ST in ${STYLES:-sparse checklist}; do
      ARMS_ST=A_unfilled,A_absent,A_notreported,B_complete,B_sparse,C_corr_complete,C_corr_sparse,C_rev_complete,C_rev_sparse,T_complete,T_sparse
      [ "$ST" = "checklist" ] && ARMS_ST=A_unfilled,A_absent,A_notreported
      WD=$S/writer/$(echo $W | tr '/' '_')_$ST
      writer_ok $WD "$W" $ST || { echo "NO_COMPLETE_RECORDS $WD"; exit 3; }
      O=$S/judge/$jt/$(echo $W | tr '/' '_')_$ST
      JA=(--model "$JUDGE" --arms $ARMS_ST --pool $POOL --labels $S/labels.json --records $WD/rows.jsonl --style $ST --writer "$W" --out $O)
      if [ -e $O ]; then
        judge_ok "${JA[@]}" || { echo "INCOMPLETE_EXISTING $O (never overwritten; resolve by hand)"; exit 5; }
        echo "EXISTS_COMPLETE $O"; continue
      fi
      verify "judge $W $ST"
      L=$(mktemp); python3 d2_judge.py --base-url "$BASE_URL" "${JA[@]}" | tee $L
      [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "JUDGE_FAILED $W $ST"; exit 5; }; chk $L "$W $ST"
      judge_ok "${JA[@]}" || exit 5
      sha256sum $O/rows.jsonl | tee -a $S/outputs.sha256
    done
  done
fi
echo "D2_STUDY_EXIT rc=0 $(date)"
