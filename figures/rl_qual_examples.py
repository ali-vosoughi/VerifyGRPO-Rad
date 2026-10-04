#!/usr/bin/env python3
"""Qualitative records for the supplement (rule declared in the project log 2026-09-28 19:02 EDT before any candidate
was looked at; illustrative, post hoc). Read-only over cached reads; writes one JSON.

Q1  the Figure 1 pair (artifacts flip_pick.json rule of 12:39): s101 true-image record as written; training judge and
    independent judge verdict + reason as written and in the training form; flag reader verdict.
Q2  first pleural-effusion pair (manifest order) with an ASSERTING sentence (false_finding) whose base true-image record
    does not flag effusion present, whose s101 true-image record flags it present and whose s101 partner-image record
    does not, and which all three readers get right on both images with the s101 records as written.
Q3  first cardiomegaly pair whose s202 records flag cardiomegaly present on BOTH images; both images' verdicts.
Q4  the oc_s101 record of the Q1 true image as written, with the training judge's verdict and reason.
A rule that finds no pair reports "none"; nothing is substituted.
Usage: rl_qual_examples.py OUTDIR
"""
import os
import collections
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
R = {
    ("tj", "raw", "base"): "results/eval2/controls_v1/step0_in/record/rows.jsonl",
    ("tj", "raw", "s101"): "results/eval2/rep_s101_step_40/sweep_gated_default/rows.jsonl",
    ("tj", "raw", "s202"): "results/eval2/rep_s202_step_40/sweep_gated_default/rows.jsonl",
    ("tj", "can", "s101"): "results/eval2/payload_rep/qwen_s101/canon_readj/record/rows.jsonl",
    ("ij", "raw", "base"): "results/eval2/val_step_0_xadj_medgemma/record/rows.jsonl",
    ("ij", "raw", "s101"): "results/eval2/xadj_medgemma/rep_s101_step_40/record/rows.jsonl",
    ("ij", "raw", "s202"): "results/eval2/xadj_medgemma/rep_s202_step_40/record/rows.jsonl",
    ("ij", "can", "s101"): "results/eval2/payload_rep/medgemma_s101/canon_readj/record/rows.jsonl",
    ("tj", "raw", "oc101"): "results/eval2/oc_s101_step_40/sweep_gated_default/rows.jsonl",
}
FINDINGS = ("Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture")


def load(path):
    out = {}
    for l in open(ROOT / path):
        if l.strip():
            r = json.loads(l)
            if not r.get("claim_is_swapped"):
                out[(r["pair_id"], r["condition"])] = r
    return out


def entries(rec):
    ents = rec.get("record") if isinstance(rec, dict) else None
    if not isinstance(ents, list):
        return None
    seen, out = set(), []
    for e in ents:
        if isinstance(e, dict) and e.get("finding") in FINDINGS and e["finding"] not in seen:
            p = e.get("present")
            p = {"true": True, "false": False}.get(p, p) if isinstance(p, str) else p
            if isinstance(p, bool):
                seen.add(e["finding"])
                out.append({"finding": e["finding"], "present": p, "side": e.get("side"), "note": e.get("note")})
    return out


def flag_verdict(row):
    ents = entries(row.get("record"))
    if ents is None:
        return "unsupported"
    tgt = next((e for e in ents if e["finding"] == row["finding"]), None)
    present = bool(tgt and tgt["present"])
    asserting = row["direction"] == "false_finding"
    return "supported" if present == asserting else "unsupported"


def flagged(row, finding):
    ents = entries(row.get("record")) or []
    e = next((e for e in ents if e["finding"] == finding), None)
    return None if e is None else e["present"]


def brief(row):
    return None if row is None else {"verdict": row.get("verdict"), "truth": row.get("truth"),
                                     "reason": row.get("reason")}


def main():
    out = ROOT / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    reads = {k: load(v) for k, v in R.items()}
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    conds = collections.Counter(c for (_, c) in reads[("tj", "raw", "s101")])
    print("conditions", dict(conds))
    other = [c for c in conds if c != "true_image"]
    assert len(other) == 1, other
    oc = other[0]
    res = {"rule": "project log 2026-09-28 19:02 EDT", "partner_condition": oc}

    fp = json.load(open(ROOT / "results/analysis/fig1_flip/flip_pick.json")) if (ROOT / "results/analysis/fig1_flip/flip_pick.json").exists() else None
    q1 = fp["pair_id"] if fp else None
    if q1 is None:
        print("Q1 flip_pick.json not found on the cluster; pass through")
    else:
        k = (q1, "true_image")
        tr = reads[("tj", "raw", "s101")][k]
        res["Q1"] = {"pair_id": q1, "finding": tr["finding"], "direction": tr["direction"], "claim": tr["claim"],
                     "truth": tr["truth"], "record_as_written": tr["record"],
                     "training_judge": {"as_written": brief(tr), "training_form": brief(reads[("tj", "can", "s101")].get(k))},
                     "independent_judge": {"as_written": brief(reads[("ij", "raw", "s101")].get(k)),
                                           "training_form": brief(reads[("ij", "can", "s101")].get(k))},
                     "flag_reader": flag_verdict(tr)}
        o = reads[("tj", "raw", "oc101")].get(k)
        res["Q4"] = None if o is None else {"pair_id": q1, "record_as_written": o["record"],
                                            "n_listed": len(entries(o["record"]) or []),
                                            "target_flag": flagged(o, o["finding"]), "training_judge": brief(o),
                                            "flag_reader": flag_verdict(o)}

    res["Q2"] = "none"
    for p in pop:
        t = reads[("tj", "raw", "s101")].get((p, "true_image"))
        q = reads[("tj", "raw", "s101")].get((p, oc))
        b = reads[("tj", "raw", "base")].get((p, "true_image"))
        if not (t and q and b) or t["finding"] != "Pleural Effusion" or t["direction"] != "false_finding":
            continue
        if flagged(b, t["finding"]) is True or flagged(t, t["finding"]) is not True or flagged(q, t["finding"]) is True:
            continue
        ok = True
        for c, row in (("true_image", t), (oc, q)):
            ij = reads[("ij", "raw", "s101")].get((p, c))
            ok &= row["verdict"] == row["truth"] and flag_verdict(row) == row["truth"] and ij is not None and ij["verdict"] == ij["truth"]
        if not ok:
            continue
        res["Q2"] = {"pair_id": p, "claim": t["claim"], "direction": t["direction"],
                     "base_true_record": b["record"], "s101_true_record": t["record"], "s101_partner_record": q["record"],
                     "base_true_flag": flagged(b, t["finding"]),
                     "verdicts": {c: {"training_judge": brief(reads[("tj", "raw", "s101")][(p, c)]),
                                      "independent_judge": brief(reads[("ij", "raw", "s101")][(p, c)]),
                                      "flag_reader": flag_verdict(reads[("tj", "raw", "s101")][(p, c)])}
                                  for c in ("true_image", oc)}}
        break

    res["Q3"] = "none"
    for p in pop:
        t = reads[("tj", "raw", "s202")].get((p, "true_image"))
        q = reads[("tj", "raw", "s202")].get((p, oc))
        if not (t and q) or t["finding"] != "Cardiomegaly":
            continue
        if flagged(t, "Cardiomegaly") is True and flagged(q, "Cardiomegaly") is True:
            res["Q3"] = {"pair_id": p, "claim": t["claim"], "direction": t["direction"],
                         "s202_true_record": t["record"], "s202_partner_record": q["record"],
                         "verdicts": {c: {"training_judge": brief(reads[("tj", "raw", "s202")][(p, c)]),
                                          "independent_judge": brief(reads[("ij", "raw", "s202")].get((p, c))),
                                          "flag_reader": flag_verdict(reads[("tj", "raw", "s202")][(p, c)])}
                                      for c in ("true_image", oc)}}
            break
    json.dump(res, open(out / "qual_examples.json", "w"), indent=1)
    for q in ("Q1", "Q2", "Q3", "Q4"):
        v = res.get(q)
        print(q, "none" if v in (None, "none") else v.get("pair_id"))
    print("wrote", out / "qual_examples.json")


if __name__ == "__main__":
    main()
