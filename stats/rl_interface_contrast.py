#!/usr/bin/env python3
"""Interface-only contrast (2026-09-27; for the second writer, whose records were read as written and in the
training form but not ablated). Same definitions as rl_payload_contrast_v2 restricted to its raw and canonical arms:
per judge dY and dFS in each form, interface = d(raw) - d(canonical), between judges (independent minus training);
common pairs, one patient-component bootstrap (2,000 draws, seed 20260925). Arms via rl_seedavg.load_arm.
Usage: rl_interface_contrast.py ARMS.json OUTDIR   (ARMS.json {judge: {"raw": arm, "canonical": arm}})
"""
import os
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_paired_delta import stats  # noqa: E402
from rl_payload_table import components  # noqa: E402
from rl_seedavg import load_arm  # noqa: E402

JUDGES = ("training_judge", "independent_judge")


def main():
    spec = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    reads = {(j, a): load_arm(spec[j][a]) for j in JUDGES for a in ("raw", "canonical")}
    ids = [p for p in pop if all(p in A and p in B for A, B in reads.values())]
    units = components(ids, pairs)

    def q(sel):
        o = {}
        for m in ("Y", "FS"):
            d = {(j, a): stats(B, sel)[m] - stats(A, sel)[m] for (j, a), (A, B) in reads.items()}
            for j in JUDGES:
                o["%s_%s_d_raw" % (j, m)] = d[(j, "raw")]; o["%s_%s_d_canonical" % (j, m)] = d[(j, "canonical")]
                o["%s_%s_interface" % (j, m)] = d[(j, "raw")] - d[(j, "canonical")]
            o["diff_%s_interface" % m] = o["independent_judge_%s_interface" % m] - o["training_judge_%s_interface" % m]
        return o

    point = q(ids)
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in q(sel).items():
            bo[k].append(v)
    res = {"pairs": len(ids), "units": len(units), "values": {}}
    lines = ["interface contrast, %d pairs, %d components" % (len(ids), len(units))]
    for k, v0 in point.items():
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        res["values"][k] = {"point": round(v0, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
        lines.append("  %-36s %+.4f [%+.4f, %+.4f]" % (k, v0, lo, hi))
    (out / "interface.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "interface.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
