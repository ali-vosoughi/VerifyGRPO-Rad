#!/usr/bin/env python3
"""Figure 3 data (2026-09-26): the numbers behind the evidence figure, for any list of reads, as JSON.
Same rules as rl_det_checks.py (target flag from the lenient canonical record; within-cell shuffle mixing both
image conditions; 200 permutations, seed 20260926) so every value equals det_checks.txt for base and step 40.
Per read: observed flag-reader Y, the 200 shuffled Y values, and per finding (directions pooled) the target
flag's TPR and FPR with counts, plus the fixed-weight macro over findings.
Usage: rl_fig3_data.py --out results/analysis/fig3/fig3_data.json name=rows.jsonl [name=rows.jsonl ...]
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_det_checks import load, flag, det_y, pop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("reads", nargs="+")
    args = ap.parse_args()
    res = {"manifest_pairs": len(pop), "reads": {}}
    for spec in args.reads:
        name, path = spec.split("=", 1)
        o = load(path)
        cnt = defaultdict(lambda: [0, 0, 0, 0])
        for pid in pop:
            t, p = o[pid]["true_image"], o[pid]["swapped_image"]
            asserts = t["direction"] == "false_finding"
            pos_img, neg_img = (t, p) if asserts else (p, t)
            fpos, fneg = flag(pos_img.get("record"), t["finding"]), flag(neg_img.get("record"), t["finding"])
            c = cnt[t["finding"]]; c[0] += int(fpos is True); c[1] += 1; c[2] += int(fneg is True); c[3] += 1
        per = {f: {"tp": c[0], "pos": c[1], "fp": c[2], "neg": c[3], "tpr": c[0] / c[1], "fpr": c[2] / c[3]} for f, c in sorted(cnt.items())}
        macro_t = sum(v["tpr"] for v in per.values()) / len(per); macro_f = sum(v["fpr"] for v in per.values()) / len(per)
        cells = defaultdict(list)
        for pid in pop:
            t = o[pid]["true_image"]
            for cond in ("true_image", "swapped_image"):
                cells[(t["finding"], t["direction"])].append((pid, cond))
        rng = random.Random(20260926); ys = []
        for _ in range(200):
            recmap = {}
            for key, slots in cells.items():
                recs = [o[pid][cond].get("record") for pid, cond in slots]
                rng.shuffle(recs)
                for (pid, cond), r in zip(slots, recs):
                    recmap[(pid, cond)] = r
            ys.append(det_y(o, recmap))
        res["reads"][name] = {"rows": path, "observed_Y": det_y(o), "null_Y": ys, "per_finding": per,
                              "macro_tpr": macro_t, "macro_fpr": macro_f}
        s = sorted(ys)
        print("%-10s observed Y %.3f  null mean %.3f [%.3f, %.3f]  macro TPR %.3f FPR %.3f" % (
            name, det_y(o), sum(ys) / len(ys), s[5], s[194], macro_t, macro_f), flush=True)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("WROTE", out)


if __name__ == "__main__":
    main()
