#!/usr/bin/env python3
"""Exact (fraction) values behind Table 1 (added after a review found base Y 0.029 on 237
pairs impossible: rl_rep_endpoints.py rounds to 4 decimals before printing 3, a double rounding). Same loaders and
common-pair rule as rl_rep_endpoints.py; prints each quantity as an exact fraction and its correct 3-decimal rounding
(ROUND_HALF_UP on the exact rational). Also: validation vs locked patient overlap.
Usage: rl_exact_table1.py CONFIG.json [CONFIG.json ...]"""
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_paired_delta import per_pair  # noqa: E402
from rl_payload_table import det_per_pair  # noqa: E402

READERS = ("flag_reader", "independent_judge", "training_judge")


def r3(q):
    s = 1 if q >= 0 else -1; q = abs(q)
    return s * int(q * 1000 + Fraction(1, 2)) / 1000


def load(reader, spec, pop):
    return det_per_pair(spec, pop) if reader == "flag_reader" else per_pair(*spec)


for cf in sys.argv[1:]:
    cfg = json.load(open(ROOT / cf))
    pop = [l.strip() for l in open(ROOT / cfg["manifest"]) if l.strip()]
    seeds = sorted(cfg["seeds"])
    reads = {r: (load(r, cfg["base"][r], pop), {s: load(r, cfg["seeds"][s][r], pop) for s in seeds}) for r in READERS}
    ids = {r: [p for p in pop if p in reads[r][0] and all(p in reads[r][1][s] for s in seeds)] for r in READERS}
    common = [p for p in pop if all(p in ids[r] for r in READERS)]
    n = len(common)
    print("==", cf, "common pairs", n)
    for r in READERS:
        b, ps = reads[r]
        Y0 = Fraction(sum(b[p]["G"] - b[p]["B"] for p in common), n); F0 = Fraction(sum(b[p]["FS"] for p in common), n)
        out = ["%-17s base Y %s = %.5f -> %.3f | base FS %s = %.5f -> %.3f" % (r, Y0, float(Y0), r3(Y0), F0, float(F0), r3(F0))]
        dys, dfs = [], []
        for s in seeds:
            Ys = Fraction(sum(ps[s][p]["G"] - ps[s][p]["B"] for p in common), n); Fs = Fraction(sum(ps[s][p]["FS"] for p in common), n)
            dys.append(Ys - Y0); dfs.append(Fs - F0)
            out.append("    seed %s Y %.3f FS %.3f dY %+.5f -> %+.3f dFS %+.5f -> %+.3f" % (s, r3(Ys), r3(Fs), float(Ys - Y0), r3(Ys - Y0), float(Fs - F0), r3(Fs - F0)))
        dY = sum(dys) / len(dys); dF = sum(dfs) / len(dfs)
        out.append("    seed-avg dY %s = %+.5f -> %+.3f | dFS %s = %+.5f -> %+.3f" % (dY, float(dY), r3(dY), dF, float(dF), r3(dF)))
        print("\n".join(out))

def patients(path):
    s = set()
    for l in open(ROOT / path):
        if l.strip():
            r = json.loads(l)
            for k in ("report_path", "image_path"):
                s.add(r[k].split("/")[1])
    return s


vp = patients("results/val_clean/pairs_val_clean_in.jsonl")
lk = json.load(open(ROOT / "code/cfg_locked_final.json"))
lp_all = patients(lk["pairs"])
man = set(l.strip() for l in open(ROOT / lk["manifest"]) if l.strip())
lp_man = set()
for l in open(ROOT / lk["pairs"]):
    if l.strip():
        r = json.loads(l)
        if r["pair_id"] in man:
            lp_man |= {r["report_path"].split("/")[1], r["image_path"].split("/")[1]}
print("validation patients (507-pair file)", len(vp), "| locked patients (pair file)", len(lp_all), "| locked image-necessary", len(lp_man))
print("overlap validation x locked (all)", len(vp & lp_all), "| validation x locked image-necessary", len(vp & lp_man))
