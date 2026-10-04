#!/usr/bin/env python3
"""Cheap checks 1 and 2 on the deterministic flag reader (2026-09-26). CPU only.
(1) Calibration: per finding and direction, the target flag's TPR (flag present on the image whose label has
    the finding) and FPR (flag present on the image without it), base vs step 40, and a fixed-weight macro
    average over findings (each finding weight 1 / #findings, direction pooled).
(2) Patient specificity: within each (finding, direction) cell, the records of BOTH image conditions are pooled
    and randomly re-assigned to the cell's images (200 permutations); deterministic Y is recomputed. Records that
    carry image-specific information lose their Y under this shuffle; image-independent ones keep it.
"""
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_verl_reward_lenient import canonical_lenient  # noqa: E402

pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]


def load(p):
    o = defaultdict(dict)
    for l in open(ROOT / p):
        r = json.loads(l)
        if not r.get("claim_is_swapped"):
            o[r["pair_id"]][r["condition"]] = r
    return o


def flag(rec, finding):
    if not isinstance(rec, dict):
        return None
    ok, c, _ = canonical_lenient(rec)
    if not ok:
        return None
    return [e for e in json.loads(c)["record"] if e["finding"] == finding][0]["present"]


def det_y(o, recmap=None):
    g = b = 0
    for pid in pop:
        t, p = o[pid]["true_image"], o[pid]["swapped_image"]
        asserts = t["direction"] == "false_finding"
        rt = recmap[(pid, "true_image")] if recmap else t.get("record")
        rp = recmap[(pid, "swapped_image")] if recmap else p.get("record")
        ft, fp = flag(rt, t["finding"]), flag(rp, p["finding"])
        st = True if ft is None else (ft != asserts); sp = True if fp is None else (fp != asserts)
        g += int(not st and sp); b += int(st and not sp)
    return (g - b) / len(pop)


def main():
    A = load("results/eval2/val_step_0/sweep_gated_default/rows.jsonl")
    B = load("results/eval2/dose_v1_lr1e-4_step_40/sweep_gated_default/rows.jsonl")
    print("(1) TPR / FPR of the target flag (positive = image whose label has the finding)")
    for tag, o in (("base", A), ("step40", B)):
        cnt = defaultdict(lambda: [0, 0, 0, 0])   # tp, pos, fp, neg
        for pid in pop:
            t, p = o[pid]["true_image"], o[pid]["swapped_image"]
            asserts = t["direction"] == "false_finding"
            # false_finding: report (true) image has the finding; missed_finding: the partner has it
            pos_img, neg_img = (t, p) if asserts else (p, t)
            for key in ((t["finding"], t["direction"]), (t["finding"], "all")):
                fpos, fneg = flag(pos_img.get("record"), t["finding"]), flag(neg_img.get("record"), t["finding"])
                c = cnt[key]; c[0] += int(fpos is True); c[1] += 1; c[2] += int(fneg is True); c[3] += 1
        rows = sorted(k for k in cnt if k[1] == "all")
        macro_t = sum(cnt[k][0] / cnt[k][1] for k in rows) / len(rows)
        macro_f = sum(cnt[k][2] / cnt[k][3] for k in rows) / len(rows)
        print("  %-6s macro TPR %.3f  macro FPR %.3f  macro (TPR - FPR) %.3f  [%d findings, fixed weights]" % (tag, macro_t, macro_f, macro_t - macro_f, len(rows)))
        for k in sorted(cnt, key=lambda k: (k[0], k[1])):
            c = cnt[k]
            if k[1] != "all":
                print("     %-20s %-15s n=%3d  TPR %.2f  FPR %.2f" % (k[0], k[1], c[1], c[0] / c[1], c[2] / c[3]))
    print("(2) patient specificity: within-cell shuffle mixing both image conditions (200 permutations)")
    for tag, o in (("base", A), ("step40", B)):
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
        ys.sort()
        print("  %-6s observed det Y %.3f | shuffled mean %.3f, 95%% range [%.3f, %.3f]" % (tag, det_y(o), sum(ys) / len(ys), ys[5], ys[194]))


if __name__ == "__main__":
    main()
