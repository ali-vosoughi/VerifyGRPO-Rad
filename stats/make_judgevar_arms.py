#!/usr/bin/env python3
"""Arms files for the judge-size and judge-prompt reads (declared in the project log 2026-09-30 23:34 EDT, post hoc,
before any read). Mirrors make_j3_arms.py / make_conv_arms.py so the frozen rl_orderfill_contrast_locked.py runs unchanged.

Size:   arms_jsz_{g12,g27}_{pop}.json   training_judge slot = training judge's existing default reads,
                                        independent_judge slot = the larger Gemma-family judge's default reads
                                        -> form effect, fill effect on FS, and diff_* = new judge minus training judge.
Prompt: arms_jv_{q8,mg}_{cite,reason}_{pop}.json  training_judge slot = judge J's DEFAULT reads,
                                        independent_judge slot = J's VARIANT reads -> diff_* = variant minus default.
        arms_jvpair_{cite,reason}_{pop}.json  training judge vs MedGemma, both under the variant -> the between-judge
                                        contrast under the variant, to compare with 0.062 / 0.077 under the default.
Usage: make_judgevar_arms.py [size|prompt|all]
"""
import os
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
FORM_DIR = {"raw": "readj", "canonical": "canon_readj", "order_only": "order_only", "fill_only": "fill_only"}
POP = {
    "val": {"manifest": "results/val_clean/manifest_image_necessary.txt",
            "pairs": "results/val_clean/pairs_val_clean_in.jsonl", "orig": "code/arms_rep_orderfill_final.json"},
    "locked": {"manifest": "results/locked/manifest_image_necessary.txt",
               "pairs": "results/locked/pairs_e6_2026-09-19_0405.jsonl", "orig": "code/arms_locked_orderfill.json"},
}
NOTE = "declared 2026-09-30 23:34 EDT (post hoc, before any read)"


def arm(base):
    return {a: {"a": [f"{base}/base/{d}", "record", "swapcached"],
                "b": [[f"{base}/{s}/{d}", "record", "swapcached"] for s in ("s101", "s202", "s303")]}
            for a, d in FORM_DIR.items()}


def write(name, spec):
    (ROOT / "code" / name).write_text(json.dumps(spec, indent=1))
    print("wrote", name)


what = sys.argv[1] if len(sys.argv) > 1 else "all"
for pop, P in POP.items():
    orig = json.load(open(ROOT / P["orig"]))
    default = {"q8": orig["training_judge"], "mg": orig["independent_judge"]}
    if what in ("size", "all"):
        for J in ("g12", "g27", "g4"):
            write(f"arms_jsz_{J}_{pop}.json", {"manifest": P["manifest"], "pairs": P["pairs"], "training_judge": default["q8"],
                                              "independent_judge": arm(f"results/eval2/judgesize/{J}_{pop}"),
                                              "note": f"independent_judge slot = judge {J} (default message); {NOTE}"})
    if what in ("prompt", "all"):
        for v in ("cite", "reason"):
            for J in ("q8", "mg"):
                write(f"arms_jv_{J}_{v}_{pop}.json", {"manifest": P["manifest"], "pairs": P["pairs"], "training_judge": default[J],
                                                     "independent_judge": arm(f"results/eval2/judgeprompt/{J}_{v}_{pop}"),
                                                     "note": f"judge {J}: default reply line (training_judge slot) vs variant {v} (independent_judge slot); {NOTE}"})
            write(f"arms_jvpair_{v}_{pop}.json", {"manifest": P["manifest"], "pairs": P["pairs"],
                                                 "training_judge": arm(f"results/eval2/judgeprompt/q8_{v}_{pop}"),
                                                 "independent_judge": arm(f"results/eval2/judgeprompt/mg_{v}_{pop}"),
                                                 "note": f"training judge vs MedGemma, both under variant {v}; {NOTE}"})
