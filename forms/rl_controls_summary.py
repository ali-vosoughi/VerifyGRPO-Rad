#!/usr/bin/env python3
"""One table of the control reads (2026-09-25). Reads OUTBASE/<policy>/directional/directional.json."""
import csv
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
keys = ("G", "B", "Y", "FS", "REC", "I", "C_pres", "C_flip_correct", "d", "Y_minus_Cpres", "record_hash_identity")
rows = []
for d in sorted(base.glob("*/directional/directional.json")):
    j = json.loads(d.read_text())
    pt, ci = j["point"], j["ci95"]
    row = {"policy": d.parent.parent.name, "scored": j["scored"], "pres_n": pt["pres_n"], "flip_n": pt["flip_n"]}
    for k in keys:
        v = pt.get(k)
        row[k] = None if v is None else round(v, 4)
        if k in ci:
            row[k + "_ci"] = "[%.3f, %.3f]" % tuple(ci[k])
    rows.append(row)
cols = ["policy", "scored", "pres_n", "flip_n"] + [c for k in keys for c in (k, k + "_ci")]
with (base / "controls_summary.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows:
        w.writerow(r)
for r in rows:
    print("%-15s n=%s G=%s B=%s Y=%s FS=%s I=%s C_pres=%s d=%s" % (r["policy"], r["scored"], r["G"], r["B"], r["Y"], r["FS"], r["I"], r["C_pres"], r["d"]))
print("SUMMARY", base / "controls_summary.csv")
