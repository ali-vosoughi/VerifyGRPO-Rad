#!/usr/bin/env python3
"""OMISSION-COST follow-up, endpoint (2) and the record part of endpoint (3), as registered before the runs
(written before any oc record existed).

(2) Paired oc-minus-control contrasts per reader: for every pair, each arm's G, B and FS are averaged over its three
seeds (oc_s101 / oc_s202 / oc_s303 against the registered replication rep_s101 / rep_s202 / rep_s303); the contrast is
the seed-averaged oc value minus the seed-averaged control value, per pair, averaged over pairs. Because both arms
share the base records, this equals dY(oc) - dY(control) and dFS(oc) - dFS(control). Per-seed contrasts (oc_sS minus
rep_sS) are reported beside it. Readers: flag reader (rl_payload_table.det_per_pair on the records), independent judge
and training judge (rl_paired_delta.per_pair on the cached reads), all as written, exactly as rl_rep_endpoints.py
loads them. Pairs: those present in every read of both arms for every reader; one patient-component bootstrap (2,000
draws, seed 20260925) shared by all quantities; two-sided 95 percent intervals, and for dFS also the one-sided upper
95 percent bound.
(3, records) Findings listed per record (distinct names among the twelve in the writer's raw record),
the share of records omitting at least one finding, and the target-finding omission rate split by
sentence direction (false_finding = the sentence asserts the finding, missed_finding = it denies it) and image
condition, on the original-claim records (claim_is_swapped false) of the population, for base, each control seed,
and each oc seed; plus the paired oc-minus-control contrast in findings listed per record, bootstrapped as above.
The K contrasts of endpoint (3) come from rl_filltarget_contrast.py on the oc control reads.
Usage: rl_oc_contrast.py CONFIG.json OUTDIR
CONFIG = {"manifest": ..., "pairs": ..., "base_rows": rows.jsonl,
          "control": {"101": {"training_judge": [dir, proto, swap], "independent_judge": [...], "flag_reader": rows}, ...},
          "oc": {"101": {...same keys...}, ...}}
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
FINDINGS = ("Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture")


def load(reader, spec, pop):
    return det_per_pair(spec, pop) if reader == "flag_reader" else per_pair(*spec)


def listed(rec):
    """Distinct names among the twelve findings in the writer's raw record."""
    ents = rec.get("record") if isinstance(rec, dict) else None
    if not isinstance(ents, list):
        return set()
    return {e.get("finding") for e in ents if isinstance(e, dict) and e.get("finding") in FINDINGS}


def record_profile(rows_path, pop):
    """Per pair: mean findings listed over its original-claim records; plus pooled descriptive counts."""
    popset = set(pop); per = defaultdict(list); desc = defaultdict(lambda: [0, 0])
    n = omit1 = 0; tot = 0
    for l in open(ROOT / rows_path):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("claim_is_swapped") or r.get("pair_id") not in popset:
            continue
        L = listed(r.get("record"))
        per[r["pair_id"]].append(len(L)); n += 1; tot += len(L); omit1 += len(L) < 12
        key = "%s|%s" % (r.get("direction"), r.get("condition"))
        desc[key][0] += 1; desc[key][1] += r.get("finding") not in L
    prof = {"records": n, "listed_mean": round(tot / n, 3) if n else None, "omit_any_share": round(omit1 / n, 3) if n else None,
            "target_omitted_share": {k: {"records": v[0], "share": round(v[1] / v[0], 3)} for k, v in sorted(desc.items())}}
    return {p: sum(v) / len(v) for p, v in per.items()}, prof


def main():
    cfg = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / cfg["manifest"]) if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / cfg["pairs"]) if l.strip()}
    seeds = sorted(cfg["oc"]); assert seeds == sorted(cfg["control"]), "oc and control seeds must match"
    reads = {(arm, s, r): load(r, cfg[arm][s][r], pop) for arm in ("control", "oc") for s in seeds for r in READERS}
    common = [p for p in pop if all(p in v for v in reads.values())]
    prof, lst = {}, {}
    lst["base"], prof["base"] = record_profile(cfg["base_rows"], pop)
    for arm in ("control", "oc"):
        for s in seeds:
            lst["%s_%s" % (arm, s)], prof["%s_%s" % (arm, s)] = record_profile(cfg[arm][s]["flag_reader"], pop)
    common = [p for p in common if all(p in v for v in lst.values())]
    units = components(common, pairs)
    per = {}
    for p in common:
        d = {}
        for r in READERS:
            for m in ("Y", "FS"):
                def val(arm, s):
                    x = reads[(arm, s, r)][p]
                    return x["G"] - x["B"] if m == "Y" else x["FS"]
                o = [val("oc", s) for s in seeds]; c = [val("control", s) for s in seeds]
                d["%s_d%s" % (r, m)] = sum(o) / len(o) - sum(c) / len(c)
                for k, s in enumerate(seeds):
                    d["%s_d%s_%s" % (r, m, s)] = o[k] - c[k]
        o = [lst["oc_" + s][p] for s in seeds]; c = [lst["control_" + s][p] for s in seeds]
        d["listed_d"] = sum(o) / len(o) - sum(c) / len(c)
        per[p] = d
    keys = sorted(per[common[0]])

    def mean(sel, k):
        return sum(per[i][k] for i in sel) / len(sel)

    point = {k: mean(common, k) for k in keys}
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k in keys:
            bo[k].append(mean(sel, k))
    res = {"config": cfg, "pairs_common": len(common), "units": len(units), "contrasts": {}, "records": prof}
    lines = ["oc minus control (registered replication), %d pairs common to all reads, %d patient components, seeds %s"
             % (len(common), len(units), ",".join(seeds))]
    for k in keys:
        v = sorted(bo[k])
        res["contrasts"][k] = {"point": round(point[k], 4), "lo95": round(v[int(0.025 * len(v))], 4),
                               "hi95": round(v[int(0.975 * len(v)) - 1], 4), "upper95_one_sided": round(v[int(0.95 * len(v)) - 1], 4)}
        c = res["contrasts"][k]
        lines.append("  %-30s %+.4f [%+.4f, %+.4f]%s" % (k, c["point"], c["lo95"], c["hi95"],
                                                         "  one-sided upper95 %+.4f" % c["upper95_one_sided"] if "_dFS" in k else ""))
    lines.append("records (original claims, population):")
    for k, v in prof.items():
        lines.append("  %-12s n=%d listed %.3f omit>=1 %.3f target omitted %s" % (
            k, v["records"], v["listed_mean"], v["omit_any_share"],
            ", ".join("%s %.3f" % (kk, vv["share"]) for kk, vv in v["target_omitted_share"].items())))
    (out / "oc_contrast.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "oc_contrast.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
