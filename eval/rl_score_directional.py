#!/usr/bin/env python3
"""Directional scorer (2026-09-24, red-team finding 1). Definitions live here.

Inputs: the original arm rows (record protocol; one row per pair and condition), the cached-record
swap rows (rl_swap_adjudicate.py), the no-image rows (define the image-necessary subset), and an
optional frozen pair-ID manifest (one pair_id per line) so every checkpoint is scored on one
population. Struck = verdict starts with "unsupported".

Per pair on the frozen image-necessary population:
  G   desirable image flip: true image accepted AND partner image struck
  B   reverse flip:          true image struck AND partner image accepted
  Y   = G - B (rate)         Youden on the population (recall_in - false_strike_in)
  FS  false_strike_in        true image struck
  REC recall_in              partner image struck
  I   = G + B                image following (unsigned), for reference only
  C_pres   truth-preserving claim swap: on the TRUE image, the swapped claim is supported per the
           study's label, and the verdict on the cached record differs from the original verdict
           (a flip under wording alone; contamination). Also on the partner image where the swapped
           claim is unsupported and the original verdict was unsupported.
  C_flip_correct  truth-flipping swap: the swapped claim's truth differs from the original claim's
           truth on that image; the response is correct if the verdict matches the swapped truth.
  d = I - C_pres  (the reviewer's mechanistic scalar), and Y - C_pres (the stronger one)
Uncertainty: pair bootstrap (2,000 draws, seed fixed) percentile intervals for Y, FS, I, C_pres,
C_flip_correct, d, and Y - C_pres. Writes OUT/directional.json and OUT/directional.csv.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def struck(r):
    if not r.get("ok"):
        return None
    v = (r.get("verdict") or "").lower()
    return True if v.startswith("unsupported") else False if v.startswith("supported") else None


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--swap-rows", required=True)
    ap.add_argument("--no-image", required=True)
    ap.add_argument("--manifest", default="", help="frozen pair_id list; written if absent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    necessary = {r["pair_id"] for r in load(args.no_image) if struck(r) is False}
    if args.manifest and Path(args.manifest).exists():
        pop = [l.strip() for l in open(args.manifest, encoding="utf-8") if l.strip()]
    else:
        pop = sorted(necessary)
        if args.manifest:
            Path(args.manifest).write_text("\n".join(pop) + "\n", encoding="utf-8")
    pop_set = set(pop)
    orig = defaultdict(dict)
    for r in load(args.rows):
        if not r.get("claim_is_swapped"):
            orig[r["pair_id"]][r["condition"]] = r
    swp = defaultdict(dict)
    for r in load(args.swap_rows):
        swp[r["pair_id"]][r["condition"]] = r
    per = {}
    for pid in pop:
        o = orig.get(pid, {}); s = swp.get(pid, {})
        t, p = o.get("true_image"), o.get("swapped_image")
        st, sp = (struck(t) if t else None), (struck(p) if p else None)
        if st is None or sp is None:
            continue
        rec = {"G": int(not st and sp), "B": int(st and not sp), "FS": int(st), "REC": int(sp), "I": int(st != sp),
               "pres_n": 0, "pres_flip": 0, "flip_n": 0, "flip_correct": 0, "hash_ok": 0, "hash_n": 0}
        for cond, base_row, base_struck, base_truth in (("true_image", t, st, "supported"), ("swapped_image", p, sp, "unsupported")):
            sr = s.get(cond)
            if not sr or not sr.get("ok"):
                continue
            ss = struck(sr)
            if ss is None:
                continue
            rec["hash_n"] += 1
            h = hashlib.sha256(json.dumps(base_row.get("record"), sort_keys=True).encode()).hexdigest()
            rec["hash_ok"] += int(sr.get("record_sha256") == h)
            truth = sr.get("swap_truth", "unknown")
            if truth == "unknown":
                continue
            if truth == base_truth:
                rec["pres_n"] += 1; rec["pres_flip"] += int(ss != base_struck)
            else:
                rec["flip_n"] += 1; rec["flip_correct"] += int(ss == (truth == "unsupported"))
        per[pid] = rec
    ids = sorted(per)

    def rates(sel):
        n = len(sel)
        agg = defaultdict(int)
        for pid in sel:
            for k, v in per[pid].items():
                agg[k] += v
        r = {"n": n, "G": agg["G"] / n, "B": agg["B"] / n, "Y": (agg["G"] - agg["B"]) / n, "FS": agg["FS"] / n, "REC": agg["REC"] / n,
             "I": agg["I"] / n, "C_pres": (agg["pres_flip"] / agg["pres_n"]) if agg["pres_n"] else None,
             "C_flip_correct": (agg["flip_correct"] / agg["flip_n"]) if agg["flip_n"] else None,
             "pres_n": agg["pres_n"], "flip_n": agg["flip_n"], "record_hash_identity": (agg["hash_ok"] / agg["hash_n"]) if agg["hash_n"] else None}
        r["d"] = (r["I"] - r["C_pres"]) if r["C_pres"] is not None else None
        r["Y_minus_Cpres"] = (r["Y"] - r["C_pres"]) if r["C_pres"] is not None else None
        return r

    point = rates(ids)
    rng = random.Random(args.seed)
    boots = defaultdict(list)
    for _ in range(args.boots):
        sel = [ids[rng.randrange(len(ids))] for _ in ids]
        b = rates(sel)
        for k in ("Y", "FS", "REC", "I", "C_pres", "C_flip_correct", "d", "Y_minus_Cpres", "G", "B"):
            if b[k] is not None:
                boots[k].append(b[k])
    ci = {}
    for k, v in boots.items():
        v = sorted(v); ci[k] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
    res = {"population_size": len(pop), "scored": len(ids), "image_necessary_from_no_image": len(necessary),
           "manifest_sha256": hashlib.sha256("\n".join(pop).encode()).hexdigest(), "point": point, "ci95": ci, "boots": args.boots}
    (out / "directional.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with (out / "directional.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["metric", "point", "ci_lo", "ci_hi"])
        for k in ("G", "B", "Y", "FS", "REC", "I", "C_pres", "C_flip_correct", "d", "Y_minus_Cpres", "record_hash_identity"):
            w.writerow([k, point.get(k), ci.get(k, [None, None])[0], ci.get(k, [None, None])[1]])
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in point.items()}))
    print("CI95", {k: [round(a, 4), round(b, 4)] for k, (a, b) in ci.items()})
    print("DIRECTIONAL_DONE scored=%d of %d" % (len(ids), len(pop)))


if __name__ == "__main__":
    main()
