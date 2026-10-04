#!/bin/bash
# Assemble the flat code directories the scripts expect. Every sbatch file calls its neighbours as
# $CLAIMBLIND_ROOT/code/<file>, and every Python script puts $CLAIMBLIND_ROOT/code (or its own directory) on sys.path
# before importing a neighbour, so all scripts and configs must sit together. The repository keeps them in topic
# folders for reading; this copies them into $CLAIMBLIND_ROOT/code, and the photograph study's scripts into
# $CLAIMBLIND_ROOT/domain2/code (the layout its sbatch files and frozen manifest use).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${CLAIMBLIND_ROOT:?set CLAIMBLIND_ROOT to the working directory (it will hold code/, data/, results/, runs/, logs/)}"
mkdir -p "$ROOT/code" "$ROOT/logs" "$ROOT/domain2/code" "$ROOT/domain2/logs"
n=0
for d in train eval forms stats data baseline configs figures; do
  for f in "$HERE/$d"/*.py "$HERE/$d"/*.sh "$HERE/$d"/*.sbatch "$HERE/$d"/*.json; do
    if [ -f "$f" ]; then cp -p "$f" "$ROOT/code/"; n=$((n + 1)); fi
  done
done
m=0
for f in "$HERE/photo_study"/*.py "$HERE/photo_study"/*.sh "$HERE/photo_study"/*.sbatch; do
  if [ -f "$f" ]; then cp -p "$f" "$ROOT/domain2/code/"; m=$((m + 1)); fi
done
# the study directory holds the pinned model revisions and the registered protocol (both listed in the freeze manifest)
mkdir -p "$ROOT/domain2/study"
[ -e "$ROOT/domain2/study/model_revisions.txt" ] || cp -p "$HERE/photo_study/model_revisions.txt" "$ROOT/domain2/study/"
[ -e "$ROOT/domain2/study/protocol_registered.md" ] || cp -p "$HERE/photo_study/protocol_registered.md" "$ROOT/domain2/study/"
echo "copied $n files into $ROOT/code and $m files into $ROOT/domain2/code"
echo "next: export RADOPEN_ROOT=$HERE/harness (it must also hold datasets/, results/e6_pairs/, runs/image_cache/, hf_cache/)"
