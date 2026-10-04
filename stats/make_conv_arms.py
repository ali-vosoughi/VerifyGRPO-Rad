#!/usr/bin/env python3
"""Arms for the explicit-convention control (declared in the project log 2026-09-28 20:16 EDT, post hoc, before any read).

For each judge J and population, code/arms_conv_{J}_{pop}.json puts J's existing DEFAULT-instruction reads in the
"training_judge" slot and J's CONVENTION reads (results/eval2/conv/{J}_{pop}/...) in the "independent_judge" slot, so the
frozen rl_orderfill_contrast_locked.py reports J's form and fill effects under both instructions and diff_* = convention
minus default. code/arms_convpair_{pop}.json puts the training judge's and MedGemma's CONVENTION reads in the two slots,
giving the between-judge form contrast under the convention, to compare with 0.062 / 0.077 under the default.
"""
import os
import json
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
FORM_DIR = {"raw": "readj", "canonical": "canon_readj", "order_only": "order_only", "fill_only": "fill_only"}
POP = {
    "val": {"manifest": "results/val_clean/manifest_image_necessary.txt",
            "pairs": "results/val_clean/pairs_val_clean_in.jsonl", "orig": "code/arms_rep_orderfill_final.json"},
    "locked": {"manifest": "results/locked/manifest_image_necessary.txt",
               "pairs": "results/locked/pairs_e6_2026-09-19_0405.jsonl", "orig": "code/arms_locked_orderfill.json"},
}


def conv_arm(J, pop):
    base = f"results/eval2/conv/{J}_{pop}"
    return {arm: {"a": [f"{base}/base/{d}", "record", "swapcached"],
                  "b": [[f"{base}/{s}/{d}", "record", "swapcached"] for s in ("s101", "s202", "s303")]}
            for arm, d in FORM_DIR.items()}


for pop, P in POP.items():
    orig = json.load(open(ROOT / P["orig"]))
    default = {"q8": orig["training_judge"], "mg": orig["independent_judge"]}
    for J in ("q30", "phi", "olmo"):
        default[J] = json.load(open(ROOT / "code" / f"arms_j3_{J}_{pop}.json"))["independent_judge"]
    for J, dflt in default.items():
        spec = {"manifest": P["manifest"], "pairs": P["pairs"], "training_judge": dflt, "independent_judge": conv_arm(J, pop),
                "note": f"judge {J}: default instruction (training_judge slot) vs explicit convention (independent_judge "
                        f"slot); declared 2026-09-28 20:16 EDT"}
        (ROOT / "code" / f"arms_conv_{J}_{pop}.json").write_text(json.dumps(spec, indent=1))
        print("wrote", f"arms_conv_{J}_{pop}.json")
    spec = {"manifest": P["manifest"], "pairs": P["pairs"], "training_judge": conv_arm("q8", pop),
            "independent_judge": conv_arm("mg", pop),
            "note": "training judge vs MedGemma, both under the explicit convention; declared 2026-09-28 20:16 EDT"}
    (ROOT / "code" / f"arms_convpair_{pop}.json").write_text(json.dumps(spec, indent=1))
    print("wrote", f"arms_convpair_{pop}.json")
