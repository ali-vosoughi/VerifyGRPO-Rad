#!/usr/bin/env python3
"""Crossed flags x notes analysis (2026-09-26; POST HOC on the discovery run, suggested by the outside
manuscript red team; declared for the replication before any replicated record is read).

Cells: records whose present flags come from the base (B) or trained (T) read and whose notes come from the base or
trained read, sides "not applicable" and boxes null everywhere (rl_control_records.py hybrid_XY, X = flags,
Y = notes). For each judge, Y_XY on the pairs common to all eight reads, and
  flag effect under base notes      F_b = Y_TB - Y_BB     flag effect under trained notes  F_t = Y_TT - Y_BT
  note effect under base flags      N_b = Y_BT - Y_BB     note effect under trained flags  N_t = Y_TT - Y_TB
  interaction                       I = F_t - F_b = N_t - N_b
and the between-judge differences of F_t, N_t and I (independent minus training). Same for FS. One
patient-component bootstrap (2,000 draws, seed 20260925) recomputes everything on each draw.
Usage: rl_hybrid_contrast.py BASEDIR OUTDIR   (BASEDIR holds qwen/hybrid_XY and medgemma/hybrid_XY)
"""
import os
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_paired_delta import per_pair  # noqa: E402
from rl_payload_table import components  # noqa: E402

CELLS = ("BB", "BT", "TB", "TT")
JUDGES = (("training_judge", "qwen"), ("independent_judge", "medgemma"))


def main():
    base = sys.argv[1]; out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    per = {(j, c): per_pair("%s/%s/hybrid_%s" % (base, sub, c), "record", "swapcached") for j, sub in JUDGES for c in CELLS}
    common = [p for p in pop if all(p in v for v in per.values())]
    units = components(common, pairs)

    def q(sel):
        n = len(sel); o = {}
        for j, _ in JUDGES:
            for m in ("Y", "FS"):
                v = {c: sum((per[(j, c)][p]["G"] - per[(j, c)][p]["B"]) if m == "Y" else per[(j, c)][p]["FS"] for p in sel) / n for c in CELLS}
                for c in CELLS:
                    o["%s_%s_%s" % (j, m, c)] = v[c]
                o["%s_%s_Fb" % (j, m)] = v["TB"] - v["BB"]; o["%s_%s_Ft" % (j, m)] = v["TT"] - v["BT"]
                o["%s_%s_Nb" % (j, m)] = v["BT"] - v["BB"]; o["%s_%s_Nt" % (j, m)] = v["TT"] - v["TB"]
                o["%s_%s_I" % (j, m)] = (v["TT"] - v["BT"]) - (v["TB"] - v["BB"])
        for m in ("Y", "FS"):
            for e in ("Fb", "Ft", "Nb", "Nt", "I"):
                o["diff_ind_minus_train_%s_%s" % (m, e)] = o["independent_judge_%s_%s" % (m, e)] - o["training_judge_%s_%s" % (m, e)]
        return o

    point = q(common)
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in q(sel).items():
            bo[k].append(v)
    res = {"pairs": len(common), "units": len(units), "post_hoc_on_discovery": True, "values": {}}
    lines = ["crossed flags x notes, %d pairs, %d components" % (len(common), len(units))]
    for k, v0 in point.items():
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        res["values"][k] = {"point": round(v0, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
        lines.append("  %-38s %+.4f [%+.4f, %+.4f]" % (k, v0, lo, hi))
    (out / "hybrid_contrasts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "hybrid_contrasts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
