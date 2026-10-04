#!/usr/bin/env python3
"""reachability of a correct record under the base writer (2026-09-25).

RLVR can only reinforce records the policy already samples. From K rollouts per image (rl_group_variance
output), a rollout is CORRECT if its training reward is 1.0 (every attached claim judged right on the
canonical record). pass@k = 1 - C(n - c, k) / C(n, k) per image (unbiased, Chen et al. 2021), averaged over
images. Also reported: images with no correct rollout in n (unreachable within n), and the same numbers for
the images that were STUCK WRONG at step 0 (all 8 rollouts at T=0.8 judged wrong in results/gvar/v1s2_step_0).
"""
import os
import argparse
import json
from math import comb
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))


def passk(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="results/gvar/<tag> directories with groups.jsonl")
    args = ap.parse_args()
    ref = {}
    rf = ROOT / "results/gvar/v1s2_step_0/groups.jsonl"
    if rf.exists():
        for l in open(rf):
            g = json.loads(l)
            ref[g["path"]] = (g["class"] == "none" and g["rewards"][0] == 0.0)
    for tag in args.tags:
        gs = [json.loads(l) for l in open(ROOT / "results/gvar" / tag / "groups.jsonl") if l.strip()]
        gs = [g for g in gs if g.get("ok")]
        n = min(len(g["rewards"]) for g in gs)
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= n]
        for label, sel in (("all", gs), ("stuck_wrong_at_step0", [g for g in gs if ref.get(g["path"])])):
            if not sel:
                continue
            cs = [sum(1 for r in g["rewards"][:n] if r >= 1.0 - 1e-9) for g in sel]
            row = {k: round(sum(passk(n, c, k) for c in cs) / len(cs), 4) for k in ks}
            unreach = sum(1 for c in cs if c == 0)
            mean_p = sum(c / n for c in cs) / len(cs)
            print("%s %-22s images=%d n=%d mean_p_correct=%.4f unreachable_within_n=%d (%.1f%%) pass@k=%s" % (
                tag, label, len(sel), n, mean_p, unreach, 100.0 * unreach / len(sel), json.dumps(row)))


if __name__ == "__main__":
    main()
