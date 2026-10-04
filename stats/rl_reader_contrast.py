#!/usr/bin/env python3
"""BETWEEN-READER contrasts (2026-09-26; the test the title's "hide" claim needs, raised in
manuscript review: the co-primary endpoints can pass while the training judge gains just as much).

For the same base and trained records, dY_R is reader R's paired change (trained minus base; seed-averaged when
several trained reads are given). The contrasts are C_flag = dY_flag - dY_train and C_ind = dY_ind - dY_train, and
the false-strike contrasts dFS_train - dFS_flag and dFS_train - dFS_ind, all on the pairs common to every read,
with one patient-component bootstrap (2,000 draws, seed 20260925) recomputing every quantity on each draw.
Same config format as rl_rep_endpoints.py (which stays frozen and untouched). On the discovery run this is POST HOC;
for the replication it is declared before any replicated record is read.
Usage: rl_reader_contrast.py CONFIG.json OUTDIR
"""
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_paired_delta import per_pair  # noqa: E402
from rl_payload_table import det_per_pair, components  # noqa: E402

READERS = ("flag_reader", "independent_judge", "training_judge")


def load(reader, spec, pop):
    return det_per_pair(spec, pop) if reader == "flag_reader" else per_pair(*spec)


def main():
    cfg = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / cfg["manifest"]) if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / cfg["pairs"]) if l.strip()}
    seeds = sorted(cfg["seeds"])
    per = {}
    for r in READERS:
        base = load(r, cfg["base"][r], pop)
        tr = [load(r, cfg["seeds"][s][r], pop) for s in seeds]
        per[r] = (base, tr)
    common = [p for p in pop if all(p in per[r][0] and all(p in t for t in per[r][1]) for r in READERS)]
    d = {r: {p: {"dY": sum(t[p]["G"] - t[p]["B"] for t in per[r][1]) / len(seeds) - (per[r][0][p]["G"] - per[r][0][p]["B"]),
                 "dFS": sum(t[p]["FS"] for t in per[r][1]) / len(seeds) - per[r][0][p]["FS"]} for p in common} for r in READERS}

    def q(sel):
        m = {r: {k: sum(d[r][p][k] for p in sel) / len(sel) for k in ("dY", "dFS")} for r in READERS}
        return {"dY_flag": m["flag_reader"]["dY"], "dY_ind": m["independent_judge"]["dY"], "dY_train": m["training_judge"]["dY"],
                "C_flag_minus_train": m["flag_reader"]["dY"] - m["training_judge"]["dY"],
                "C_ind_minus_train": m["independent_judge"]["dY"] - m["training_judge"]["dY"],
                "C_flag_minus_ind": m["flag_reader"]["dY"] - m["independent_judge"]["dY"],
                "FS_train_minus_flag": m["training_judge"]["dFS"] - m["flag_reader"]["dFS"],
                "FS_train_minus_ind": m["training_judge"]["dFS"] - m["independent_judge"]["dFS"]}

    units = components(common, pairs)
    point = q(common)
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in q(sel).items():
            bo[k].append(v)
    res = {"config": cfg, "pairs": len(common), "units": len(units), "contrasts": {}}
    lines = ["between-reader contrasts, %d pairs, %d components, seeds %s" % (len(common), len(units), ",".join(seeds))]
    for k, v0 in point.items():
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        res["contrasts"][k] = {"point": round(v0, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
        lines.append("  %-22s %+.4f [%+.4f, %+.4f]" % (k, v0, lo, hi))
    (out / "reader_contrasts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "reader_contrasts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
