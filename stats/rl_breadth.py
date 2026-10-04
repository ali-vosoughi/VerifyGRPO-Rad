#!/usr/bin/env python3
"""Breadth of the replicated gain (POST HOC, added after a review objection;
the discovery version was an inline script, breadth_dose40_POSTHOC). For each reader, the seed-averaged per-pair dY
exactly as rl_rep_endpoints.py computes it (same loaders, same config format, pairs common to all readers), then
three summaries: all pairs; without the pleural effusion pairs; finding-macro (the mean over findings of each
finding's mean dY, equal weight per finding). One patient-component bootstrap, 2,000 draws, seed 20260925, shared by
all quantities; the macro is recomputed on each draw over the findings present in it. Also per finding: pairs and
seed-averaged dY (points only).
Usage: rl_breadth.py CONFIG.json OUTDIR   (CONFIG as rl_rep_endpoints.py)
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
EFF = "Pleural Effusion"


def load(reader, spec, pop):
    return det_per_pair(spec, pop) if reader == "flag_reader" else per_pair(*spec)


def main():
    cfg = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / cfg["manifest"]) if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / cfg["pairs"]) if l.strip()}
    seeds = sorted(cfg["seeds"])
    dy = {}
    for r in READERS:
        base = load(r, cfg["base"][r], pop); ps = {s: load(r, cfg["seeds"][s][r], pop) for s in seeds}
        dy[r] = {p: sum(ps[s][p]["G"] - ps[s][p]["B"] for s in seeds) / len(seeds) - (base[p]["G"] - base[p]["B"])
                 for p in pop if p in base and all(p in ps[s] for s in seeds)}
    common = [p for p in pop if all(p in dy[r] for r in READERS)]
    fnd = {p: pairs[p]["finding"] for p in common}
    units = components(common, pairs)

    def q(sel):
        o = {}
        for r in READERS:
            v = [dy[r][p] for p in sel]; o[r + "_all"] = sum(v) / len(v)
            ne = [dy[r][p] for p in sel if fnd[p] != EFF]; o[r + "_without_effusion"] = sum(ne) / len(ne)
            by = defaultdict(list)
            for p in sel:
                by[fnd[p]].append(dy[r][p])
            o[r + "_finding_macro"] = sum(sum(x) / len(x) for x in by.values()) / len(by)
        return o

    point = q(common)
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in q(sel).items():
            bo[k].append(v)
    per_f = defaultdict(list)
    for p in common:
        per_f[fnd[p]].append(p)
    res = {"config": sys.argv[1], "status": "POST HOC", "pairs": len(common), "units": len(units),
           "effusion_pairs": len(per_f[EFF]), "values": {}, "per_finding": {}}
    lines = ["breadth (POST HOC), %s: %d pairs, %d components, %d pleural effusion pairs" % (cfg["manifest"], len(common), len(units), len(per_f[EFF]))]
    for k, v0 in point.items():
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        res["values"][k] = {"point": round(v0, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
        lines.append("  %-36s %+.4f [%+.4f, %+.4f]" % (k, v0, lo, hi))
    lines.append("per finding (n; seed-averaged dY flag / independent / training):")
    for f in sorted(per_f):
        ids = per_f[f]
        vals = [sum(dy[r][p] for p in ids) / len(ids) for r in READERS]
        res["per_finding"][f] = {"n": len(ids), **{r: round(x, 4) for r, x in zip(READERS, vals)}}
        lines.append("  %-28s %3d  %+.3f / %+.3f / %+.3f" % (f, len(ids), *vals))
    (out / "breadth.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "breadth.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
