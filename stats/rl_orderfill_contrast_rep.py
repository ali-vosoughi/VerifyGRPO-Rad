#!/usr/bin/env python3
"""REPLICATION VERSION (seed-averaged via rl_seedavg.load_arm; otherwise identical). ORDER x MISSINGNESS analysis (POST HOC on the discovery run, DECLARED for the replication
before any replicated decomposition was computed).

Arms per judge, each a paired change dY (trained minus base) on the same cached records:
  raw = writer order, omissions kept     order_only = fixed order, omissions kept
  fill_only = writer order, omissions written as absent     canonical = fixed order, omissions written as absent
Effects (averaged over the other factor): order = mean(order_only - raw, canonical - fill_only);
fill = mean(fill_only - raw, canonical - order_only); interaction = (canonical - fill_only) - (order_only - raw);
between-judge differences (independent minus training) of each; Y and FS; common pairs; one patient-component
bootstrap (2,000 draws, seed 20260925).
Usage: rl_orderfill_contrast.py ARMS.json OUTDIR   (ARMS.json: {judge: {arm: [a_dir, a_proto, a_swap, b_dir, b_proto, b_swap]}})
"""
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_paired_delta import per_pair, stats  # noqa: E402
from rl_payload_table import components  # noqa: E402
from rl_seedavg import load_arm  # noqa: E402  (seed-averaged reads; replication version)

JUDGES = ("training_judge", "independent_judge")
ARMS = ("raw", "order_only", "fill_only", "canonical")


def main():
    spec = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    reads = {(j, a): load_arm(spec[j][a]) for j in JUDGES for a in ARMS}
    ids = [p for p in pop if all(p in A and p in B for A, B in reads.values())]
    units = components(ids, pairs)

    def q(sel):
        o = {}
        for m in ("Y", "FS"):
            d = {(j, a): stats(B, sel)[m] - stats(A, sel)[m] for (j, a), (A, B) in reads.items()}
            for j in JUDGES:
                for a in ARMS:
                    o["%s_%s_d_%s" % (j, m, a)] = d[(j, a)]
                o["%s_%s_order" % (j, m)] = ((d[(j, "order_only")] - d[(j, "raw")]) + (d[(j, "canonical")] - d[(j, "fill_only")])) / 2
                o["%s_%s_fill" % (j, m)] = ((d[(j, "fill_only")] - d[(j, "raw")]) + (d[(j, "canonical")] - d[(j, "order_only")])) / 2
                o["%s_%s_interaction" % (j, m)] = (d[(j, "canonical")] - d[(j, "fill_only")]) - (d[(j, "order_only")] - d[(j, "raw")])
            for e in ("order", "fill", "interaction"):
                o["diff_%s_%s" % (m, e)] = o["independent_judge_%s_%s" % (m, e)] - o["training_judge_%s_%s" % (m, e)]
        return o

    point = q(ids)
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in q(sel).items():
            bo[k].append(v)
    res = {"pairs": len(ids), "units": len(units), "values": {}}
    lines = ["order x missingness, %d pairs, %d components" % (len(ids), len(units))]
    for k, v0 in point.items():
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        res["values"][k] = {"point": round(v0, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
        lines.append("  %-40s %+.4f [%+.4f, %+.4f]" % (k, v0, lo, hi))
    (out / "orderfill.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "orderfill.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
