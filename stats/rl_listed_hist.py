#!/usr/bin/env python3
"""Findings listed per record, per writer, on the validation records (2026-09-28, supplementary figure; descriptive).
Rule: distinct valid finding names among the twelve in the writer's raw record (a boolean or
"true"/"false" present value). Reads the stored records of the base writer, the 3 registered runs, and the 3
omission-cost runs (all 238 validation pairs x 2 images, original claims); writes one small JSON."""
import os
import collections
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
FINDINGS = ("Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture")
SRC = {
    "base": "results/eval2/val_step_0/sweep_gated_default/rows.jsonl",
    "rep_s101": "results/eval2/rep_s101_step_40/sweep_gated_default/rows.jsonl",
    "rep_s202": "results/eval2/rep_s202_step_40/sweep_gated_default/rows.jsonl",
    "rep_s303": "results/eval2/rep_s303_step_40/sweep_gated_default/rows.jsonl",
    "oc_s101": "results/eval2/oc_s101_step_40/sweep_gated_default/rows.jsonl",
    "oc_s202": "results/eval2/oc_s202_step_40/sweep_gated_default/rows.jsonl",
    "oc_s303": "results/eval2/oc_s303_step_40/sweep_gated_default/rows.jsonl",
}


def listed(rec):
    ents = rec.get("record") if isinstance(rec, dict) else None
    if not isinstance(ents, list):
        return 0
    names = set()
    for e in ents:
        if isinstance(e, dict) and e.get("finding") in FINDINGS:
            p = e.get("present")
            if isinstance(p, bool) or (isinstance(p, str) and p.lower() in ("true", "false")):
                names.add(e["finding"])
    return len(names)


MANIFEST = {l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()}
out = {}
for k, rel in SRC.items():
    seen, counts = set(), []
    for l in open(ROOT / rel):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("claim_is_swapped") or r["pair_id"] not in MANIFEST:
            continue
        key = (r["pair_id"], r["condition"])
        if key in seen:
            continue
        seen.add(key)
        counts.append(listed(r.get("record")))
    h = collections.Counter(counts)
    out[k] = {"n_records": len(counts), "mean": round(sum(counts) / len(counts), 3),
              "share_omitting_any": round(sum(c < 12 for c in counts) / len(counts), 4),
              "hist": {str(i): h.get(i, 0) for i in range(13)}}
    print(k, out[k]["n_records"], out[k]["mean"], out[k]["share_omitting_any"])
dst = ROOT / "results/analysis/listed_hist"
dst.mkdir(parents=True, exist_ok=True)
(dst / "listed_hist.json").write_text(json.dumps(out, indent=1))
print("wrote", dst / "listed_hist.json")
