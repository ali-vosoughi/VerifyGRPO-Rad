#!/bin/bash
# Judge-size / judge-prompt analysis (declared 2026-09-30 23:34 EDT): builds the arms files, runs the FROZEN
# rl_orderfill_contrast_locked.py once per arms file whose reads are all complete, then the cite-evidence readout and a
# summary table. Skips an arms file (and says so) when any of its read directories is missing or lacks rows.jsonl.
# Usage: run_judgevar_analysis.sh [size|prompt|all]
set -uo pipefail
ROOT=${CLAIMBLIND_ROOT:?set CLAIMBLIND_ROOT (absolute path)}
cd $ROOT/code
source "${HARNESS_VENV:?set HARNESS_VENV to the harness (vLLM) environment}/bin/activate"
export RADOPEN_ROOT=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
WHAT=${1:-all}
python3 make_judgevar_arms.py "$WHAT" || exit 2
case "$WHAT" in size) PAT="arms_jsz_*.json";; prompt) PAT="arms_jv_*.json arms_jvpair_*.json";; *) PAT="arms_jsz_*.json arms_jv_*.json arms_jvpair_*.json";; esac
complete() {  # ARMS -> 0 if every read dir in the file has record/rows.jsonl and swapcached/rows.jsonl
  python3 - "$1" <<'PY'
import json, os, sys, pathlib
root = pathlib.Path(os.environ.get("CLAIMBLIND_ROOT", "."))
spec = json.load(open(sys.argv[1])); missing = []
for slot in ("training_judge", "independent_judge"):
    for arm, v in spec[slot].items():
        if isinstance(v, dict):
            b = v["b"]
            sides = [v["a"]] + (list(b) if b and isinstance(b[0], (list, tuple)) else [b])
        else:
            sides = [v[:3], v[3:]]
        for s in sides:
            d = root / s[0]
            for sub in (s[1], s[2]):
                if sub is None:
                    continue
                if not (d / sub / "rows.jsonl").exists():
                    missing.append(str(d / sub))
print("\n".join(missing)); sys.exit(1 if missing else 0)
PY
}
for A in $PAT; do
  [ -f "$A" ] || continue
  name=${A#arms_}; name=${name%.json}
  OUT=results/analysis/judgevar/$name
  if m=$(complete "$A"); then
    echo "=== $name $(date)"
    python3 rl_orderfill_contrast_locked.py "$A" "$OUT" && sha256sum $ROOT/$OUT/orderfill.json
  else
    echo "SKIP $name (incomplete reads): $(echo "$m" | head -2 | tr '\n' ' ') ..."
  fi
done
if [ "$WHAT" != size ]; then
  python3 rl_cite_evidence.py results/analysis/judgevar/cite_evidence || echo "CITE_READOUT_FAILED"
fi
python3 summarize_judgevar.py results/analysis/judgevar || echo "SUMMARY_FAILED"
echo "JUDGEVAR_ANALYSIS_EXIT $(date)"
