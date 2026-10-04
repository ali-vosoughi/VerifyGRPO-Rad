#!/usr/bin/env python3
"""TARGET-FILL x OTHER-FILL analysis (declared for all three seeds, after
seed 101 motivated it). Writer order kept in every cell.
Cells (paired change dY / dFS, trained minus base, per judge): raw (00), fill_target (10: the omitted target stated
absent), fill_other (01: every other omission stated absent), fill_all (11 = fill_only).
Effects of FILLING (moving from raw toward filled; averaged over the other factor): target = mean(10 - 00, 11 - 01),
other = mean(01 - 00, 11 - 10), interaction = (11 - 01) - (10 - 00).
PRIMARY contrasts as declared: K_FS_j = dFS_j(raw) - dFS_j(fill_all); K_between_Y = [dY_ind(raw) - dY_ind(fill_all)] -
[dY_train(raw) - dY_train(fill_all)]. One patient-component bootstrap (2,000 draws, seed 20260925), common pairs.
Usage: rl_filltarget_contrast.py ARMS.json OUTDIR  (ARMS.json {judge: {cell: [a_dir,a_proto,a_swap,b_dir,b_proto,b_swap] or {"a":..,"b":..}}})
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
CELLS = ("raw", "fill_target", "fill_other", "fill_all")


def main():
    spec = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    reads = {(j, c): load_arm(spec[j][c]) for j in JUDGES for c in CELLS}
    ids = [p for p in pop if all(p in A and p in B for A, B in reads.values())]
    units = components(ids, pairs)

    def q(sel):
        o = {}
        for m in ("Y", "FS"):
            d = {(j, c): stats(B, sel)[m] - stats(A, sel)[m] for (j, c), (A, B) in reads.items()}
            for j in JUDGES:
                for c in CELLS:
                    o["%s_%s_d_%s" % (j, m, c)] = d[(j, c)]
                o["%s_%s_target" % (j, m)] = ((d[(j, "fill_target")] - d[(j, "raw")]) + (d[(j, "fill_all")] - d[(j, "fill_other")])) / 2
                o["%s_%s_other" % (j, m)] = ((d[(j, "fill_other")] - d[(j, "raw")]) + (d[(j, "fill_all")] - d[(j, "fill_target")])) / 2
                o["%s_%s_interaction" % (j, m)] = (d[(j, "fill_all")] - d[(j, "fill_other")]) - (d[(j, "fill_target")] - d[(j, "raw")])
                o["K_%s_%s" % (m, j)] = d[(j, "raw")] - d[(j, "fill_all")]
            o["K_between_%s" % m] = o["K_%s_independent_judge" % m] - o["K_%s_training_judge" % m]
            for e in ("target", "other", "interaction"):
                o["diff_%s_%s" % (m, e)] = o["independent_judge_%s_%s" % (m, e)] - o["training_judge_%s_%s" % (m, e)]
        return o

    point = q(ids)
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in q(sel).items():
            bo[k].append(v)
    res = {"pairs": len(ids), "units": len(units), "values": {}}
    lines = ["target-fill x other-fill, %d pairs, %d components" % (len(ids), len(units))]
    for k, v0 in point.items():
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        res["values"][k] = {"point": round(v0, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
        lines.append("  %-40s %+.4f [%+.4f, %+.4f]" % (k, v0, lo, hi))
    (out / "filltarget.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "filltarget.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
