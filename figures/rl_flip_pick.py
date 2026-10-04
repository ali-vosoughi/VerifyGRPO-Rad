#!/usr/bin/env python3
"""Figure 1 flip example and flip counts (rule declared in the project log 2026-09-28 12:39 EDT before any candidate
was looked at; illustrative, post hoc).

Example: among the 238 validation pairs in manifest order, the first pleural-effusion pair with a DENYING sentence
(missed_finding) whose seed-101 step-40 record of the TRUE image leaves the target finding out, and which the training
judge REJECTS as written and ACCEPTS in the training form; if none, the first such pair in any finding.
Counts: for base and each trained run, true-image original-claim records whose training-judge verdict changes between
the record as written and the training form, split into reject -> accept and accept -> reject, on the pairs present in
both reads.
Usage: rl_flip_pick.py OUTDIR
"""
import os
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_paired_delta import struck  # noqa: E402

FINDINGS = ("Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture")
READS = {
    "base": ("results/eval2/controls_v1/step0_in/record/rows.jsonl", "results/eval2/canon/qwen_base/canon_readj/record/rows.jsonl"),
    "s101": ("results/eval2/rep_s101_step_40/sweep_gated_default/rows.jsonl", "results/eval2/payload_rep/qwen_s101/canon_readj/record/rows.jsonl"),
    "s202": ("results/eval2/rep_s202_step_40/sweep_gated_default/rows.jsonl", "results/eval2/payload_rep/qwen_s202/canon_readj/record/rows.jsonl"),
    "s303": ("results/eval2/rep_s303_step_40/sweep_gated_default/rows.jsonl", "results/eval2/payload_rep/qwen_s303/canon_readj/record/rows.jsonl"),
}


def load(path):
    out = {}
    for l in open(ROOT / path):
        if l.strip():
            r = json.loads(l)
            if not r.get("claim_is_swapped"):
                out[(r["pair_id"], r["condition"])] = r
    return out


def listed(rec):
    ents = rec.get("record") if isinstance(rec, dict) else None
    return {e.get("finding") for e in ents if isinstance(e, dict)} if isinstance(ents, list) else set()


def entries(rec):
    ents = rec.get("record") if isinstance(rec, dict) else None
    return [{"finding": e.get("finding"), "present": e.get("present")} for e in ents if isinstance(e, dict) and e.get("finding") in FINDINGS] if isinstance(ents, list) else []


def main():
    out = ROOT / sys.argv[1]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    counts = {}
    for name, (raw_p, can_p) in READS.items():
        raw, can = load(raw_p), load(can_p)
        n = ra = ar = 0
        for p in pop:
            k = (p, "true_image")
            if k in raw and k in can:
                a, b = struck(raw[k]), struck(can[k])
                if a is None or b is None:
                    continue
                n += 1; ra += (a and not b); ar += (b and not a)
        counts[name] = {"true_image_records": n, "reject_to_accept": ra, "accept_to_reject": ar}
    raw, can = load(READS["s101"][0]), load(READS["s101"][1])
    pick, stage, examined = None, "pleural effusion", 0
    for finding in ("Pleural Effusion", None):
        for p in pop:
            pr = pairs[p]
            if pr["direction"] != "missed_finding" or (finding and pr["finding"] != finding):
                continue
            examined += 1
            k = (p, "true_image")
            if k not in raw or k not in can:
                continue
            if pr["finding"] in listed(raw[k].get("record")):
                continue
            if struck(raw[k]) is True and struck(can[k]) is False:
                pick = p; break
        if pick:
            break
        stage = "any finding"
    res = {"rule": "declared in the project log 2026-09-28 12:39 EDT", "stage": stage, "examined": examined, "counts": counts}
    if pick:
        pr = pairs[pick]; k = (pick, "true_image")
        res.update({"pair_id": pick, "finding": pr["finding"], "sentence": raw[k].get("claim"), "image_path": raw[k].get("image_path"),
                    "raw_entries": entries(raw[k].get("record")), "omitted": [f for f in FINDINGS if f not in listed(raw[k].get("record"))],
                    "verdict_as_written": raw[k].get("verdict"), "reason_as_written": raw[k].get("reason"),
                    "verdict_training_form": can[k].get("verdict"), "reason_training_form": can[k].get("reason")})
    (out / "flip_pick.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("raw_entries",)}, indent=1))


if __name__ == "__main__":
    main()
