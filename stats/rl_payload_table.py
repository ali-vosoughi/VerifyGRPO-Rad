#!/usr/bin/env python3
"""Payload factorial table (2026-09-26; secondary analysis declared in the replication protocol).

Readers x payload arms x sentence strata, on the frozen 238-pair image-necessary manifest. For every cell: the
paired change from the base writer's records to the trained writer's records (the SAME cached records, changed
in one part by the payload arm, then read), with patient-component bootstrap intervals (2,000 draws, seed
20260925, components formed within the stratum). Judges use rl_paired_delta.per_pair (the evaluation-v2
scoring); the flag reader uses rl_det_reader's rule (target flag vs sentence direction, no record = strike).
Sentence strata follow the rule frozen BEFORE this script existed:
DETAIL-RICH if the lower-cased target sentence contains a listed severity / location / temporal word as a whole
word, else PLAIN; hedges are counted, not used.
Usage: rl_payload_table.py --out results/analysis/payload [--b-tag dose40] [--arms-json FILE]
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_paired_delta import per_pair, stats  # noqa: E402
from rl_det_reader import flag  # noqa: E402

SEVERITY = "mild mildly moderate moderately severe severely small smaller large larger trace minimal minimally tiny subtle slight slightly marked markedly extensive massive significant borderline"
LOCATION = "left right bilateral bilaterally basilar bibasilar base bases basal apex apical upper lower middle mid lobe lingula lingular retrocardiac perihilar hilar costophrenic lateral medial anterior posterior"
TEMPORAL = "increased increasing decreased decreasing improved improving worsened worsening unchanged stable new persistent resolved resolving interval"
HEDGES = "possible possibly likely may might could questionable suspicious probable suggest suggests suggesting"
DETAIL = set((SEVERITY + " " + LOCATION + " " + TEMPORAL).split())
HEDGE = set(HEDGES.split())

# (reader, arm) -> (a_dir, a_proto, a_swap, b_dir, b_proto, b_swap); det arms give rows files
DEFAULT_ARMS = {
    ("training_judge", "full"): ("results/eval2/controls_v1/step0_in", "record", "swapcached",
                                 "results/eval2/dose_v1_lr1e-4_step_40", "sweep_gated_default", None),
    ("training_judge", "sides_neutral"): ("results/eval2/side_neutral/base/side_neutral", "record", "swapcached",
                                          "results/eval2/side_neutral/dose40/side_neutral", "record", "swapcached"),
    ("training_judge", "notes_removed"): ("results/eval2/notes_removed/base/notes_removed", "record", "swapcached",
                                          "results/eval2/notes_removed/dose40/notes_removed", "record", "swapcached"),
    ("training_judge", "flags_only"): ("results/eval2/payload/qwen_base/flags_only", "record", "swapcached",
                                       "results/eval2/payload/qwen_dose40/flags_only", "record", "swapcached"),
    ("independent_judge", "full"): ("results/eval2/val_step_0_xadj_medgemma", "record", "swapcached",
                                    "results/eval2/xadj_medgemma/dose40", "record", "swapcached"),
    ("independent_judge", "sides_neutral"): ("results/eval2/payload/medgemma_sn_base/side_neutral", "record", "swapcached",
                                             "results/eval2/payload/medgemma_sn_dose40/side_neutral", "record", "swapcached"),
    ("independent_judge", "notes_removed"): ("results/eval2/payload/medgemma_base/notes_removed", "record", "swapcached",
                                             "results/eval2/payload/medgemma_dose40/notes_removed", "record", "swapcached"),
    ("independent_judge", "flags_only"): ("results/eval2/payload/medgemma_base/flags_only", "record", "swapcached",
                                          "results/eval2/payload/medgemma_dose40/flags_only", "record", "swapcached"),
    ("flag_reader", "full"): ("results/eval2/controls_v1/step0_in/record/rows.jsonl", None, None,
                              "results/eval2/dose_v1_lr1e-4_step_40/sweep_gated_default/rows.jsonl", None, None),
}


def words(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def det_per_pair(rows_path, pop):
    o = defaultdict(dict)
    for l in open(ROOT / rows_path):
        r = json.loads(l)
        if not r.get("claim_is_swapped"):
            o[r["pair_id"]][r["condition"]] = r
    per = {}
    for pid in pop:
        t, p = o[pid].get("true_image"), o[pid].get("swapped_image")
        if not t or not p:
            continue
        asserts = t["direction"] == "false_finding"
        ft, fp = flag(t.get("record"), t["finding"]), flag(p.get("record"), p["finding"])
        st = True if ft is None else (ft != asserts)
        sp = True if fp is None else (fp != asserts)
        per[pid] = {"G": int(not st and sp), "B": int(st and not sp), "FS": int(st), "pres_n": 0, "pres_flip": 0}
    return per


def components(ids, pairs):
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in ids:
        a, b = (pairs[i][k].split("/")[1] for k in ("report_path", "image_path"))
        parent[find(a)] = find(b)
    comp = defaultdict(list)
    for i in ids:
        comp[find(pairs[i]["report_path"].split("/")[1])].append(i)
    return list(comp.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms-json", default="", help="optional JSON list of [reader, arm, a_dir, a_proto, a_swap, b_dir, b_proto, b_swap]")
    ap.add_argument("--boots", type=int, default=2000)
    args = ap.parse_args()
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    arms = DEFAULT_ARMS
    if args.arms_json:
        arms = {(a[0], a[1]): tuple(a[2:]) for a in json.load(open(args.arms_json))}

    strata = {"all": list(pop), "plain": [], "detail": []}
    hedged = defaultdict(int)
    for pid in pop:
        w = words(pairs[pid]["target_sentence"])
        s = "detail" if w & DETAIL else "plain"
        strata[s].append(pid)
        hedged[s] += int(bool(w & HEDGE))
    for s in ("plain", "detail"):
        (out / ("manifest_%s.txt" % s)).write_text("\n".join(strata[s]) + "\n", encoding="utf-8")
    counts = {s: {"pairs": len(v), "hedged": hedged.get(s, 0)} for s, v in strata.items()}
    by_finding = defaultdict(lambda: defaultdict(int))
    for s in ("plain", "detail"):
        for pid in strata[s]:
            by_finding[pairs[pid]["finding"]][s] += 1
    counts["by_finding"] = {f: dict(v) for f, v in sorted(by_finding.items())}
    (out / "strata_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print("STRATA", json.dumps({s: counts[s] for s in ("all", "plain", "detail")}))

    rows_out = []
    for (reader, arm), spec in arms.items():
        a_dir, a_proto, a_swap, b_dir, b_proto, b_swap = spec
        try:
            if reader == "flag_reader":
                A, B = det_per_pair(a_dir, pop), det_per_pair(b_dir, pop)
            else:
                A, B = per_pair(a_dir, a_proto, a_swap), per_pair(b_dir, b_proto, b_swap)
        except FileNotFoundError as exc:
            print("SKIP %s %s missing %s" % (reader, arm, exc.filename)); continue
        for sname, sids in strata.items():
            ids = [p for p in sids if p in A and p in B]
            if not ids:
                continue
            sa, sb = stats(A, ids), stats(B, ids)
            units = components(ids, pairs)
            rng = random.Random(20260925); bo = defaultdict(list)
            for _ in range(args.boots):
                sel = [i for _ in units for i in units[rng.randrange(len(units))]]
                xa, xb = stats(A, sel), stats(B, sel)
                for k in ("Y", "FS", "G", "B"):
                    bo[k].append(xb[k] - xa[k])
            rec = {"reader": reader, "arm": arm, "stratum": sname, "pairs": len(ids), "units": len(units)}
            for k in ("Y", "FS", "G", "B"):
                v = sorted(bo[k])
                rec.update({"a_" + k: round(sa[k], 4), "b_" + k: round(sb[k], 4), "d_" + k: round(sb[k] - sa[k], 4),
                            "lo_" + k: round(v[int(0.025 * len(v))], 4), "hi_" + k: round(v[int(0.975 * len(v)) - 1], 4),
                            "hi95_1s_" + k: round(v[int(0.95 * len(v)) - 1], 4)})
            rows_out.append(rec)
            print("%-17s %-14s %-7s n=%3d  dY %+.3f [%+.3f, %+.3f]  dFS %+.3f [%+.3f, %+.3f]  (Y %.3f->%.3f, FS %.3f->%.3f)" % (
                reader, arm, sname, len(ids), rec["d_Y"], rec["lo_Y"], rec["hi_Y"], rec["d_FS"], rec["lo_FS"], rec["hi_FS"],
                sa["Y"], sb["Y"], sa["FS"], sb["FS"]), flush=True)
    with open(out / "payload_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print("WROTE", out / "payload_table.csv")


if __name__ == "__main__":
    main()
