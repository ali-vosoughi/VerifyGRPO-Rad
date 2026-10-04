#!/usr/bin/env python3
"""REPLICATION endpoints, exactly as frozen before the replication runs (co-primary, conjunctive).

For each reader and each pair of the population: G, B, FS of the base records and of each seed's step-40 records
(judges: rl_paired_delta.per_pair on the cached reads; flag reader: rl_payload_table.det_per_pair). The per-pair
change is the SEED-AVERAGED value minus the base value; the population change is its mean over pairs; intervals
come from one patient-component bootstrap (2,000 draws, seed 20260925) shared by all readers.
Co-primary endpoints: (1) flag reader dY >= +0.05 with the lower 95 percent bound > 0; (2) independent judge dY
>= +0.05 with the lower bound > 0; (3) for EACH of those two readers the one-sided upper 95 percent bound of dFS
<= +0.05. Success = all three. Every seed is also reported alone; the training judge is reported beside them and
never enters the verdict. Pairs: those present in the base read and in every seed's read for that reader.
Usage: rl_rep_endpoints.py CONFIG.json OUTDIR
CONFIG = {"manifest": ..., "pairs": ..., "base": {"training_judge": [dir, proto, swap], "independent_judge": [...],
          "flag_reader": rows}, "seeds": {"101": {...same keys...}, ...}}
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


def mean(sel, per, k):
    return sum(per[i][k] for i in sel) / len(sel)


def main():
    cfg = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / cfg["manifest"]) if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / cfg["pairs"]) if l.strip()}
    seeds = sorted(cfg["seeds"])
    data = {}
    for r in READERS:
        base = load(r, cfg["base"][r], pop)
        per_seed = {s: load(r, cfg["seeds"][s][r], pop) for s in seeds}
        ids = [p for p in pop if p in base and all(p in per_seed[s] for s in seeds)]
        d = {}
        for p in ids:
            y0 = base[p]["G"] - base[p]["B"]; f0 = base[p]["FS"]
            ys = [per_seed[s][p]["G"] - per_seed[s][p]["B"] for s in seeds]; fs = [per_seed[s][p]["FS"] for s in seeds]
            d[p] = {"dY": sum(ys) / len(ys) - y0, "dFS": sum(fs) / len(fs) - f0,
                    **{"dY_" + s: ys[k] - y0 for k, s in enumerate(seeds)}, **{"dFS_" + s: fs[k] - f0 for k, s in enumerate(seeds)},
                    "Y0": y0, "FS0": f0, **{"Y_" + s: ys[k] for k, s in enumerate(seeds)}, **{"FS_" + s: fs[k] for k, s in enumerate(seeds)}}
        data[r] = (ids, d)
    common = [p for p in pop if all(p in data[r][1] for r in READERS)]
    units = components(common, pairs)
    keys = ["dY", "dFS"] + ["dY_" + s for s in seeds] + ["dFS_" + s for s in seeds]
    # every reader is scored on the SAME pairs (common to all reads) so points and intervals share one population
    data = {r: (common, data[r][1]) for r in READERS}
    point = {r: {k: mean(common, data[r][1], k) for k in keys + ["Y0", "FS0"] + ["Y_" + s for s in seeds] + ["FS_" + s for s in seeds]} for r in READERS}
    rng = random.Random(20260925); bo = {r: defaultdict(list) for r in READERS}
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for r in READERS:
            for k in keys:
                bo[r][k].append(mean(sel, data[r][1], k))
    res = {"config": cfg, "pairs_common": len(common), "units": len(units), "readers": {}}
    lines = ["population %s: %d pairs common to all readers, %d patient components, seeds %s" % (cfg["manifest"], len(common), len(units), ",".join(seeds))]
    for r in READERS:
        ids, _ = data[r]; rr = {"pairs": len(ids)}
        for k in keys:
            v = sorted(bo[r][k])
            rr[k] = {"point": round(point[r][k], 4), "lo95": round(v[int(0.025 * len(v))], 4), "hi95": round(v[int(0.975 * len(v)) - 1], 4),
                     "upper95_one_sided": round(v[int(0.95 * len(v)) - 1], 4)}
        rr["base_Y"] = round(point[r]["Y0"], 4); rr["base_FS"] = round(point[r]["FS0"], 4)
        for s in seeds:
            rr["Y_" + s] = round(point[r]["Y_" + s], 4); rr["FS_" + s] = round(point[r]["FS_" + s], 4)
        res["readers"][r] = rr
        lines.append("%-17s n=%3d  base Y %.3f FS %.3f | seed-avg dY %+.4f [%+.4f, %+.4f]  dFS %+.4f (one-sided upper95 %+.4f)" % (
            r, len(ids), rr["base_Y"], rr["base_FS"], rr["dY"]["point"], rr["dY"]["lo95"], rr["dY"]["hi95"], rr["dFS"]["point"], rr["dFS"]["upper95_one_sided"]))
        for s in seeds:
            lines.append("    seed %s: Y %.3f FS %.3f  dY %+.4f [%+.4f, %+.4f]  dFS %+.4f (upper95 %+.4f)" % (
                s, rr["Y_" + s], rr["FS_" + s], rr["dY_" + s]["point"], rr["dY_" + s]["lo95"], rr["dY_" + s]["hi95"],
                rr["dFS_" + s]["point"], rr["dFS_" + s]["upper95_one_sided"]))
    e = res["readers"]
    ep = {"flag_reader_dY": e["flag_reader"]["dY"]["point"] >= 0.05 and e["flag_reader"]["dY"]["lo95"] > 0,
          "independent_judge_dY": e["independent_judge"]["dY"]["point"] >= 0.05 and e["independent_judge"]["dY"]["lo95"] > 0,
          "dFS_noninferior_both": e["flag_reader"]["dFS"]["upper95_one_sided"] <= 0.05 and e["independent_judge"]["dFS"]["upper95_one_sided"] <= 0.05}
    ep["SUCCESS"] = all(ep.values())
    ep["negative_seeds"] = {r: [s for s in seeds if e[r]["dY_" + s]["point"] < 0] for r in READERS}
    res["endpoints"] = ep
    lines.append("ENDPOINTS " + json.dumps(ep))
    (out / "endpoints.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "endpoints.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
