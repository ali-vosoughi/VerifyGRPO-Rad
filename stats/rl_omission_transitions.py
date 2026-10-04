#!/usr/bin/env python3
"""Default-prompt omission transitions (declared in the project log 2026-10-01 06:22 EDT, post hoc, before computing).

For a judge's reads of the SAME saved records in the original form (arm "raw") and in fill_only (every omission written
absent, writer order), rows are matched by (record set, pair_id, condition, pass). Target = the sentence's finding
(swap pass: donor_finding); polarity = label-defined pair direction (false_finding -> 'present' sentence, true where the
finding is present; missed_finding -> 'absent' sentence, true where it is absent; swap pass: donor_direction);
truth = truth (record pass) / swap_truth (swap pass; unknown excluded). OMITTED = no entry for the target in the record
as written (raw row's record). Cells: omitted / listed x polarity x truth. Statistic per cell: acceptance (verdict
'supported') in the original form, in fill_only, and the difference, rows pooled over the 4 record sets; 95% percentile
intervals from 2,000 draws of patient components (seed 20260925), resampling whole components with all their rows.
Usage: rl_omission_transitions.py OUTDIR [record|all]  (record = record-pass rows only, the pair's own sentence; default record)
"""
import os
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_payload_table import components  # noqa: E402

POP = {"val": {"pairs": "results/val_clean/pairs_val_clean_in.jsonl", "manifest": "results/val_clean/manifest_image_necessary.txt",
               "orig": "code/arms_rep_orderfill_final.json"},
       "locked": {"pairs": "results/locked/pairs_e6_2026-09-19_0405.jsonl", "manifest": "results/locked/manifest_image_necessary.txt",
                  "orig": "code/arms_locked_orderfill.json"}}
# (label, arms file pattern with {pop}, slot)
JUDGES = [
    ("training judge", "{orig}", "training_judge"),
    ("MedGemma-4B", "{orig}", "independent_judge"),
    ("Qwen3-VL-30B", "code/arms_j3_q30_{pop}.json", "independent_judge"),
    ("Phi-4-mini", "code/arms_j3_phi_{pop}.json", "independent_judge"),
    ("OLMo-2-7B", "code/arms_j3_olmo_{pop}.json", "independent_judge"),
    ("Gemma-3-4B", "code/arms_jsz_g4_{pop}.json", "independent_judge"),
    ("Gemma-3-12B", "code/arms_jsz_g12_{pop}.json", "independent_judge"),
    ("Gemma-3-27B", "code/arms_jsz_g27_{pop}.json", "independent_judge"),
    ("training judge, cite", "code/arms_jv_q8_cite_{pop}.json", "independent_judge"),
    ("MedGemma-4B, cite", "code/arms_jv_mg_cite_{pop}.json", "independent_judge"),
    ("training judge, reason", "code/arms_jv_q8_reason_{pop}.json", "independent_judge"),
    ("MedGemma-4B, reason", "code/arms_jv_mg_reason_{pop}.json", "independent_judge"),
]
DRAWS, SEED = 2000, 20260925


def norm(s):
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def entries(record):
    rec = record.get("record", record) if isinstance(record, dict) else record
    return [e for e in rec if isinstance(e, dict)] if isinstance(rec, list) else []


def reads(spec):
    """arms entry -> list of 4 reads [dir, proto, swap] in order base, s101, s202, s303."""
    if isinstance(spec, dict):
        b = spec["b"]
        return [spec["a"]] + (list(b) if b and isinstance(b[0], (list, tuple)) else [b])
    return [spec[:3], spec[3:]]


PASSES = sys.argv[2] if len(sys.argv) > 2 else "record"


def rows_of(read):
    d = ROOT / read[0]
    out = {}
    plan = [("record", read[1])] + ([("swap", read[2] or read[1] + "_swapcached")] if PASSES == "all" else [])
    for pas, sub in plan:
        p = d / sub / "rows.jsonl"
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if pas == "record" and r.get("claim_is_swapped"):
                continue
            out[(pas, r["pair_id"], r["condition"])] = r
    return out


def main():
    out = ROOT / sys.argv[1]; out.mkdir(parents=True, exist_ok=True)
    res = {}; lines = []
    for pop, P in POP.items():
        pop_ids = [l.strip() for l in open(ROOT / P["manifest"]) if l.strip()]
        pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / P["pairs"]) if l.strip()}
        comp_of = {}
        for ci, comp in enumerate(components(pop_ids, pairs)):
            for pid in comp:
                comp_of[pid] = ci
        for label, pat, slot in JUDGES:
            af = ROOT / pat.format(pop=pop, orig=P["orig"])
            if not af.exists():
                continue
            spec = json.load(open(af))[slot]
            raw_reads, fill_reads = reads(spec["raw"]), reads(spec["fill_only"])
            cells = collections.defaultdict(list)  # cell -> list of (component, acc_orig, acc_fill)
            n_match = n_unmatched = 0
            for si, (rr, fr) in enumerate(zip(raw_reads, fill_reads)):
                R, F = rows_of(rr), rows_of(fr)
                for key, r in R.items():
                    pas, pid, cond = key
                    if pid not in comp_of:
                        continue
                    f = F.get(key)
                    if not f or not r.get("ok") or not f.get("ok"):
                        n_unmatched += 1; continue
                    truth = r.get("truth") if pas == "record" else r.get("swap_truth")
                    if truth not in ("supported", "unsupported"):
                        continue
                    tf = r.get("finding") if pas == "record" else r.get("donor_finding")
                    direction = r.get("direction") if pas == "record" else r.get("donor_direction")
                    pol = {"false_finding": "present", "missed_finding": "absent"}.get(direction)
                    if pol is None or not tf:
                        continue
                    listed = any(norm(e.get("finding")) == norm(tf) for e in entries(r.get("record")))
                    cell = ("listed" if listed else "omitted", pol, "true" if truth == "supported" else "false")
                    cells[cell].append((comp_of[pid], int(r.get("verdict") == "supported"), int(f.get("verdict") == "supported")))
                    n_match += 1
            # bootstrap over components
            comps = sorted(set(comp_of.values()))
            rng = random.Random(SEED)
            by_comp = {cell: collections.defaultdict(lambda: [0, 0, 0]) for cell in cells}
            for cell, v in cells.items():
                for c, a, b in v:
                    t = by_comp[cell][c]; t[0] += a; t[1] += b; t[2] += 1
            draws = []
            for _ in range(DRAWS):
                pick = [rng.choice(comps) for _ in comps]
                draws.append(pick)
            jr = {"pop": pop, "judge": label, "matched_rows": n_match, "unmatched_or_unparsed": n_unmatched, "cells": {}}
            for cell, v in sorted(cells.items()):
                n = len(v)
                ao = sum(a for _, a, _ in v) / n; af_ = sum(b for _, _, b in v) / n
                dd = []
                for pick in draws:
                    sa = sb = m = 0
                    bc = by_comp[cell]
                    for c in pick:
                        t = bc.get(c)
                        if t:
                            sa += t[0]; sb += t[1]; m += t[2]
                    if m:
                        dd.append((sb - sa) / m)
                dd.sort()
                lo, hi = dd[int(0.025 * len(dd))], dd[int(0.975 * len(dd)) - 1]
                to_acc = sum(1 for _, a, b in v if a == 0 and b == 1) / n
                to_rej = sum(1 for _, a, b in v if a == 1 and b == 0) / n
                jr["cells"]["|".join(cell)] = {"n": n, "accept_original": round(ao, 4), "accept_fill": round(af_, 4),
                                               "diff": round(af_ - ao, 4), "diff_lo95": round(lo, 4), "diff_hi95": round(hi, 4),
                                               "rejected_to_accepted": round(to_acc, 4), "accepted_to_rejected": round(to_rej, 4)}
            res[f"{label}|{pop}"] = jr
    (out / "omission_transitions.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    hdr = "| judge | pop | cell (target, sentence, truth) | rows | accept original | accept omissions-absent | difference [95%] |"
    lines = [hdr, "|---|---|---|---|---|---|---|"]
    for k, jr in res.items():
        for cell in ("omitted|absent|true", "omitted|absent|false", "omitted|present|true", "omitted|present|false",
                     "listed|absent|true", "listed|absent|false"):
            c = jr["cells"].get(cell)
            if not c:
                continue
            lines.append("| %s | %s | %s | %d | %.1f%% | %.1f%% | %+.1f%% [%+.1f, %+.1f] |" % (
                jr["judge"], jr["pop"], cell, c["n"], 100 * c["accept_original"], 100 * c["accept_fill"],
                100 * c["diff"], 100 * c["diff_lo95"], 100 * c["diff_hi95"]))
    (out / "omission_transitions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
