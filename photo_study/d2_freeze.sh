#!/bin/bash
# Freeze the second-domain study (protocol section 8.2). Run ONCE on the login node after the registered protocol copy
# has been placed at study/protocol_registered.md. Refuses if a manifest exists already, if any writer or judge output
# exists, or if a listed file is missing. Writes study/FROZEN.sha256 (byte-level manifest, absolute paths) and
# study/FROZEN.txt (the manifest's own hash and the time). No model is loaded.
set -euo pipefail
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
D=${CLAIMBLIND_ROOT:-.}/domain2; S=$D/study; C=$D/code
[ -e $S/FROZEN.sha256 ] && { echo "ALREADY_FROZEN"; exit 9; }
[ -e $S/writer ] || [ -e $S/judge ] && { echo "STUDY_OUTPUT_EXISTS: freeze must precede every study read"; exit 9; }
FILES=(
  $S/protocol_registered.md
  $D/pool_v8_cap60_n20/pool.jsonl $D/pool_v8_cap60_n20/images.txt
  $S/labels.json $S/screen.jsonl $S/screen_pass.txt $S/screen_summary.json $S/model_revisions.txt
  $C/d2_forms.py $C/d2_write_records.py $C/d2_judge.py $C/d2_analyze.py $C/d2_screen.py $C/d2_screen.sbatch
  $C/d2_study.sbatch $C/d2_build_pool.py $C/d2_freeze.sh $C/d2_conv_probe.py $C/d2_conv_probe.sbatch $C/d2_smoke.sbatch
  $C/d2_make_smoke.py $C/d2_make_calib.py $C/d2_calib.sbatch
  $RAD/code/scripts/vllm_common.sh
)
for f in "${FILES[@]}"; do [ -s "$f" ] || { echo "MISSING $f"; exit 9; }; done
cd $C
python3 d2_forms.py --selftest
python3 d2_analyze.py --selftest
[ "$(wc -l < $D/pool_v8_cap60_n20/pool.jsonl)" -eq 516 ] && [ "$(wc -l < $D/pool_v8_cap60_n20/images.txt)" -eq 1032 ] \
  || { echo "POOL_SIZE"; exit 9; }
sha256sum "${FILES[@]}" > $S/FROZEN.sha256
chmod a-w $S/FROZEN.sha256 "${FILES[@]:0:8}"
H=$(sha256sum $S/FROZEN.sha256 | awk '{print $1}')
printf 'FROZEN %s\nmanifest sha256 %s\nfiles %d\n' "$(date)" "$H" "${#FILES[@]}" > $S/FROZEN.txt
cat $S/FROZEN.sha256; cat $S/FROZEN.txt
sha256sum --quiet -c $S/FROZEN.sha256 && echo "MANIFEST_VERIFIES"
