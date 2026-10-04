#!/usr/bin/env python3
"""deterministic flag reader split by sentence direction and by finding, base vs a checkpoint read
(2026-09-26; the confound check of the dose-test secondary result). Usage: rl_det_by_group.py <base rows> <other rows>"""
import os
import collections
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_verl_reward_lenient import canonical_lenient  # noqa: E402

pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]


def load(p):
    o = collections.defaultdict(dict)
    for l in open(ROOT / p):
        r = json.loads(l)
        if not r.get("claim_is_swapped"):
            o[r["pair_id"]][r["condition"]] = r
    return o


def flag(r):
    rec = r.get("record")
    if not isinstance(rec, dict):
        return None
    ok, c, _ = canonical_lenient(rec)
    if not ok:
        return None
    return [e for e in json.loads(c)["record"] if e["finding"] == r["finding"]][0]["present"]


def cell(o, pid):
    t, p = o[pid]["true_image"], o[pid]["swapped_image"]
    asserts = t["direction"] == "false_finding"
    ft, fp = flag(t), flag(p)
    st = True if ft is None else (ft != asserts)
    sp = True if fp is None else (fp != asserts)
    return int(not st and sp), int(st and not sp), ft, fp


A, B = load(sys.argv[1]), load(sys.argv[2])
for key in ("direction", "finding"):
    agg = collections.defaultdict(lambda: [0] * 7)
    for pid in pop:
        k = A[pid]["true_image"][key]
        ga, ba, fta, fpa = cell(A, pid)
        gb, bb, ftb, fpb = cell(B, pid)
        a = agg[k]
        a[0] += 1; a[1] += ga; a[2] += ba; a[3] += gb; a[4] += bb
        a[5] += int(bool(fta)) + int(bool(fpa)); a[6] += int(bool(ftb)) + int(bool(fpb))
    print("== by", key)
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        n = a[0]
        print("  %-18s n=%3d  Y %+.3f -> %+.3f  G %.3f -> %.3f  B %.3f -> %.3f  target-present rate %.2f -> %.2f" % (
            k, n, (a[1] - a[2]) / n, (a[3] - a[4]) / n, a[1] / n, a[3] / n, a[2] / n, a[4] / n, a[5] / (2 * n), a[6] / (2 * n)))
