#!/usr/bin/env python3
"""Arms files for the three-judge generality test (declared 2026-09-28 18:34 EDT, post hoc, before any read).

For each further judge J in {q30, phi, olmo} and population in {val, locked}, the frozen analysis
code/rl_orderfill_contrast_locked.py (sha256 d904e2e5...) is run unchanged with its "training_judge" slot holding the
training judge's existing reads (from arms_rep_orderfill_final.json / arms_locked_orderfill.json) and its
"independent_judge" slot holding judge J's reads of the SAME cached records, so every J quantity is computed exactly as
the independent judge's was, and diff_* is J minus the training judge. Writes code/arms_j3_{J}_{pop}.json.
"""
import os
import json
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
FORM_DIR = {"raw": "readj", "canonical": "canon_readj", "order_only": "order_only", "fill_only": "fill_only"}
POP = {
    "val": {"manifest": "results/val_clean/manifest_image_necessary.txt",
            "pairs": "results/val_clean/pairs_val_clean_in.jsonl", "tj": "code/arms_rep_orderfill_final.json"},
    "locked": {"manifest": "results/locked/manifest_image_necessary.txt",
               "pairs": "results/locked/pairs_e6_2026-09-19_0405.jsonl", "tj": "code/arms_locked_orderfill.json"},
}
for J in ("q30", "phi", "olmo"):
    for pop, P in POP.items():
        tj = json.load(open(ROOT / P["tj"]))["training_judge"]
        base = f"results/eval2/judges3/{J}_{pop}"
        ij = {arm: {"a": [f"{base}/base/{d}", "record", "swapcached"],
                    "b": [[f"{base}/{s}/{d}", "record", "swapcached"] for s in ("s101", "s202", "s303")]}
              for arm, d in FORM_DIR.items()}
        spec = {"manifest": P["manifest"], "pairs": P["pairs"], "training_judge": tj, "independent_judge": ij,
                "note": f"independent_judge slot = judge {J}; declared 2026-09-28 18:34 EDT"}
        p = ROOT / "code" / f"arms_j3_{J}_{pop}.json"
        p.write_text(json.dumps(spec, indent=1))
        print("wrote", p.name)
