#!/usr/bin/env python3
"""Paired change between two checkpoint reads on the same frozen manifest (2026-09-25).

The pre-registered NO-GO rule (design v1 item 5): both paired intervals, for Y and for C_pres, inside the
equivalence region +/- 0.05 after round 1. Per pair on the manifest, the same quantities as
rl_score_directional.py (struck = verdict starts with "unsupported"; G, B from the original arm on the two
images; the truth-preserving swap flip from the cached-swap rows). Pairs are resampled jointly for both
reads (paired bootstrap, 2,000 draws), so the interval is for the CHANGE. Prints delta Y, delta C_pres,
delta FS, delta G, delta B with 95 percent intervals and the equivalence verdict.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))


def struck(r):
    if not r.get("ok"):
        return None
    v = (r.get("verdict") or "").lower()
    return True if v.startswith("unsupported") else False if v.startswith("supported") else None


def per_pair(read_dir, proto="sweep_gated_default", swap_dir=None):
    d = ROOT / read_dir
    orig, swp = defaultdict(dict), defaultdict(dict)
    for l in open(d / proto / "rows.jsonl"):
        r = json.loads(l)
        if not r.get("claim_is_swapped"):
            orig[r["pair_id"]][r["condition"]] = r
    for l in open(d / (swap_dir or proto + "_swapcached") / "rows.jsonl"):
        r = json.loads(l)
        swp[r["pair_id"]][r["condition"]] = r
    out = {}
    for pid, o in orig.items():
        t, p = o.get("true_image"), o.get("swapped_image")
        st, sp = (struck(t) if t else None), (struck(p) if p else None)
        if st is None or sp is None:
            continue
        rec = {"G": int(not st and sp), "B": int(st and not sp), "FS": int(st), "pres_n": 0, "pres_flip": 0}
        for cond, base_struck, base_truth in (("true_image", st, "supported"), ("swapped_image", sp, "unsupported")):
            sr = swp[pid].get(cond)
            if not sr or not sr.get("ok"):
                continue
            ss = struck(sr)
            if ss is None or sr.get("swap_truth", "unknown") != base_truth:
                continue
            rec["pres_n"] += 1; rec["pres_flip"] += int(ss != base_struck)
        out[pid] = rec
    return out


def stats(per, ids):
    n = len(ids)
    g = sum(per[i]["G"] for i in ids); b = sum(per[i]["B"] for i in ids); fs = sum(per[i]["FS"] for i in ids)
    pn = sum(per[i]["pres_n"] for i in ids); pf = sum(per[i]["pres_flip"] for i in ids)
    return {"G": g / n, "B": b / n, "Y": (g - b) / n, "FS": fs / n, "C_pres": pf / pn if pn else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="read dir relative to ROOT, e.g. results/eval2/controls_v1/step0_in")
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-proto", default="sweep_gated_default")
    ap.add_argument("--a-swap", default=None, help="swap rows dir under --a (controls use swapcached)")
    ap.add_argument("--b-proto", default="sweep_gated_default")
    ap.add_argument("--b-swap", default=None)
    ap.add_argument("--manifest", default="results/val_clean/manifest_image_necessary.txt")
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--cluster", choices=("pair", "patient"), default="patient",
                    help="resampling unit: pair, or patient-connected components (pairs sharing a patient move together)")
    ap.add_argument("--pairs", default="results/val_clean/pairs_val_clean_in.jsonl")
    args = ap.parse_args()
    pop = [l.strip() for l in open(ROOT / args.manifest) if l.strip()]
    A = per_pair(args.a, args.a_proto, args.a_swap); B = per_pair(args.b, args.b_proto, args.b_swap)
    ids = [p for p in pop if p in A and p in B]
    sa, sb = stats(A, ids), stats(B, ids)
    delta = {k: sb[k] - sa[k] for k in sa}
    units = [[i] for i in ids]
    if args.cluster == "patient":
        pr = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / args.pairs) if l.strip()}
        parent = {}
        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in ids:
            a, b = (pr[i][k].split("/")[1] for k in ("report_path", "image_path"))
            parent[find(a)] = find(b)
        comp = defaultdict(list)
        for i in ids:
            comp[find(pr[i]["report_path"].split("/")[1])].append(i)
        units = list(comp.values())
    rng = random.Random(20260925); boots = defaultdict(list)
    for _ in range(args.boots):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        xa, xb = stats(A, sel), stats(B, sel)
        for k in xa:
            boots[k].append(xb[k] - xa[k])
    ci = {}
    for k, v in boots.items():
        v = sorted(v); ci[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1])
    print("paired pairs=%d units=%d (%s)  a=%s  b=%s" % (len(ids), len(units), args.cluster, args.a, args.b))
    for k in ("Y", "C_pres", "FS", "G", "B"):
        print("  delta %-6s %+.4f  [%+.4f, %+.4f]   a %.4f -> b %.4f" % (k, delta[k], ci[k][0], ci[k][1], sa[k], sb[k]))
    inside = all(-args.margin <= ci[k][0] and ci[k][1] <= args.margin for k in ("Y", "C_pres"))
    print("EQUIVALENCE (Y and C_pres paired 95%% intervals inside +/- %.2f): %s" % (args.margin, inside))


if __name__ == "__main__":
    main()
