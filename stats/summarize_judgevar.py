#!/usr/bin/env python3
"""Summary table for the judge-size / judge-prompt analyses: one row per arms output under results/analysis/judgevar/.
Columns (percent, 95% interval): form effect on the gain for the slot-A judge/instruction (training_judge_Y_interface) and
slot-B (independent_judge_Y_interface), their difference (diff_Y_interface), the fill effect on false strikes for both
slots (FS_fill) and its difference. Slot meanings per arms file: jsz_* = training judge vs larger Gemma-family judge;
jv_<J>_<v>_* = judge J default vs variant v; jvpair_<v>_* = training judge vs MedGemma, both under variant v.
Usage: summarize_judgevar.py results/analysis/judgevar
"""
import os
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
base = ROOT / sys.argv[1]
KEYS = ["training_judge_Y_interface", "independent_judge_Y_interface", "diff_Y_interface",
        "training_judge_FS_fill", "independent_judge_FS_fill", "diff_FS_fill",
        "training_judge_Y_d_raw", "training_judge_Y_d_canonical", "independent_judge_Y_d_raw", "independent_judge_Y_d_canonical"]


def fmt(v):
    return "%+.1f [%+.1f, %+.1f]" % (100 * v["point"], 100 * v["lo95"], 100 * v["hi95"])


rows = []
for d in sorted(base.iterdir()):
    f = d / "orderfill.json"
    if not f.exists():
        continue
    j = json.load(open(f, encoding="utf-8"))
    v = j["values"]
    rows.append((d.name, j["pairs"], [fmt(v[k]) if k in v else "n/a" for k in KEYS]))
hdr = "| arms | pairs | A form effect dY | B form effect dY | B minus A | A FS fill | B FS fill | B minus A FS fill | A dY raw | A dY canon | B dY raw | B dY canon |"
lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
for name, n, vals in rows:
    lines.append("| %s | %d | %s |" % (name, n, " | ".join(vals)))
out = base / "summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("wrote", out)
