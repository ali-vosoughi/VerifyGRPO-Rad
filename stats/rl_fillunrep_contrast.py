#!/usr/bin/env python3
"""EXPLORATORY contrast (added after the final review; not registered): does the training
form act by asserting absence or by making missingness explicit? Three forms of the same cached base and trained
records, all in the writer's own order: raw (as written), fill_unreported (every omitted finding appended as an
explicit "not assessed" entry), fill_only (every omitted finding appended as an explicit absent entry).
Per judge and metric (Y, FS): d_<form> = trained - base, seed-averaged via rl_seedavg.load_arm;
explicit = d_raw - d_unreported (effect of stating that a finding was not assessed);
assert = d_unreported - d_fill_only (effect of asserting absence instead of stating not assessed);
fill = d_raw - d_fill_only (the registered fill simple effect at writer order, for reference);
diff_<q> = independent minus training. Common pairs, one patient-component bootstrap (2,000 draws, seed 20260925),
same bootstrap code path as the other contrast scripts.
Usage: rl_fillunrep_contrast.py ARMS.json OUTDIR   (ARMS.json {judge: {"raw": arm, "fill_unreported": arm, "fill_only": arm}})
"""
import json
import os
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
FORMS = ("raw", "fill_unreported", "fill_only")


def main():
    spec = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    reads = {(j, a): load_arm(spec[j][a]) for j in JUDGES for a in FORMS}
    ids = [p for p in pop if all(p in A and p in B for A, B in reads.values())]
    units = components(ids, pairs)

    def q(sel):
        o = {}
        for m in ("Y", "FS"):
            d = {(j, a): stats(B, sel)[m] - stats(A, sel)[m] for (j, a), (A, B) in reads.items()}
            for j in JUDGES:
                for a in FORMS:
                    o["%s_%s_d_%s" % (j, m, a)] = d[(j, a)]
                o["%s_%s_explicit" % (j, m)] = d[(j, "raw")] - d[(j, "fill_unreported")]
                o["%s_%s_assert" % (j, m)] = d[(j, "fill_unreported")] - d[(j, "fill_only")]
                o["%s_%s_fill" % (j, m)] = d[(j, "raw")] - d[(j, "fill_only")]
            for c in ("explicit", "assert", "fill"):
                o["diff_%s_%s" % (m, c)] = o["independent_judge_%s_%s" % (m, c)] - o["training_judge_%s_%s" % (m, c)]
        return o

    point = q(ids)
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in q(sel).items():
            bo[k].append(v)
    res = {"pairs": len(ids), "units": len(units), "status": "EXPLORATORY (not registered)", "values": {}}
    lines = ["EXPLORATORY fill_unreported vs fill_only vs raw, %d pairs, %d components" % (len(ids), len(units))]
    for k, v0 in point.items():
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        res["values"][k] = {"point": round(v0, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
        lines.append("  %-40s %+.4f [%+.4f, %+.4f]" % (k, v0, lo, hi))
    (out / "fillunrep.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "fillunrep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
