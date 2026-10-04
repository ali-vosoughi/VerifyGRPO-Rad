#!/usr/bin/env python3
"""ENGINEERING probe of the closed-world convention wording (2026-09-30, before the freeze). Out-of-pool smoke pairs and
the synthetic comprehension set only; no study image, record, or verdict is read.

Why: smoke 23368 showed that the convention naming the object universe ("The record covers exactly these objects: ...")
made both primary judges read the enumerated names as the record's own findings (comprehension on the sparse encoding
fell from 100% to 85.4% and 81.2%; sparse target controls to 59-75%). This probe scores candidate wordings on the
registered criteria only (comprehension per encoding, target controls per writer x encoding x control) and applies a
rule fixed BEFORE the probe runs: adopt the first candidate, in the order K1, K2, K3, K0, whose every comprehension and
control cell is at least 90% for both primary judges (round 1, job 23375: K3 adopted). Round 2 (after the third
pre-freeze check; rule fixed before the run): K4 names the record's objects field instead of an undisclosed list; adopt
K4 if every cell is >= 90% for both primary judges, otherwise keep K3 if it still passes under the strict parser,
otherwise no freeze. The encoding-contrast result E is never computed here.
Usage: d2_conv_probe.py --model M --base-url URL --smoke-dir DIR --writer-run DIR --out FILE
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_forms as F  # noqa: E402
import d2_judge as J  # noqa: E402

L = F._LIST
PREC = " The listed states and this default take precedence over the notes and the other field."
CANDIDATES = {   # order = preference order of the adoption rule
    "K1_checklist_not_findings": (
        "The note-taker worked from a fixed checklist of 20 object names (possible objects only, not findings about "
        "this photograph): " + L + ". Convention for this record: the states listed in the record are authoritative "
        "for whether a checklist object is in the photograph, and any checklist object that the record does not list "
        "is {DEFAULT}." + PREC),
    "K2_claim_object_on_checklist": (
        "Convention for this record: the note-taker worked from a fixed checklist of 20 object names, and the object "
        "named in the sentence below is on that checklist. The states listed in the record are authoritative for "
        "whether a checklist object is in the photograph, and any checklist object that the record does not list is "
        "{DEFAULT}." + PREC),
    "K3_list_rule_plus_precedence": (
        "Convention for this record: the listed states are authoritative for whether an object from the list is in "
        "the photograph, and any object from the list that the record does not list is {DEFAULT}." + PREC),
    "K0_prior_smoked_wording": (
        "Convention for this record: the listed states are authoritative for whether an object from the list is in "
        "the photograph, and any object from the list that the record does not list is {DEFAULT}. Notes are "
        "auxiliary."),
    # round 2 (third pre-freeze check, outside): name the field instead of an undisclosed list; rule in the docstring
    "K4_objects_field": (
        "Convention for this record: the states in the objects field are authoritative for whether an object is in "
        "the photograph, and any object that the objects field does not list is {DEFAULT}." + PREC),
    # round 3 (calibration set calib_v2, 41 pairs in 17 categories): the failures of rounds 1-2 were all one refusal ("the
    # record does not list any people, but it also does not explicitly state their absence"); K5 states the default
    # outright and drops the undisclosed "from the list". Closed world only is probed.
    "K5_explicit_default": (
        "Convention for this record: the listed states are authoritative for whether an object is in the photograph. "
        "Any object that the record does not list is {DEFAULT} from the photograph, even though the record does not say "
        "so explicitly." + PREC),
}
ARMS = ["B_complete", "B_sparse", "C_corr_complete", "C_corr_sparse", "C_rev_complete", "C_rev_sparse"]
WRITERS = ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--base-url", required=True)
    ap.add_argument("--smoke-dir", required=True); ap.add_argument("--writer-run", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--only", help="comma list of candidate names (round 2: K4_objects_field,K3_list_rule_plus_precedence)")
    a = ap.parse_args()
    assert a.base_url.startswith("http://127.0.0.1")
    sd, wr = Path(a.smoke_dir), Path(a.writer_run)
    pool = [json.loads(l) for l in open(sd / "pool.jsonl") if l.strip()]
    labels = json.load(open(sd / "labels.json"))
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=600)

    def one(t):
        claim = F.claim(t["category"], t["polarity"])
        msgs = F.judge_messages(t["evidence"], claim, convention=t["convention"], ternary=t["ternary"])
        text = ""
        for _ in range(2):
            try:
                r = client.chat.completions.create(model=a.model, messages=msgs, temperature=0.0, seed=20260930,
                                                   max_tokens=128, response_format={"type": "json_object"})
                text = r.choices[0].message.content or ""
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        v = J.parse_verdict(text, False)       # the registered strict parser (round 2; round 1 used tolerant extraction)
        return {"arm": t["arm"], "writer": t["writer"], "verdict": v if v in ("supported", "unsupported") else None,
                "implied": t["record_implied"], "response": text[:200], "claim": claim}

    report = {}
    for name, tmpl in CANDIDATES.items():
        if a.only and name not in a.only.split(","):
            continue
        F.CONVENTION = {"closed": tmpl.replace("{DEFAULT}", "absent"), "open": tmpl.replace("{DEFAULT}", "not reported")}
        tasks = J.tasks_comp()
        for w in WRITERS:
            recs = [json.loads(l) for l in open(wr / "writer" / (w.replace("/", "_") + "_sparse") / "rows.jsonl")]
            tasks += J.tasks_study(pool, labels, recs, "sparse", w, ARMS)[0]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, tasks))
        cells = defaultdict(lambda: [0, 0])
        for r in rows:
            key = r["arm"] if r["arm"].startswith("COMP") else "%s|%s" % (r["writer"], r["arm"])
            cells[key][0] += 1
            cells[key][1] += int(r["verdict"] is not None and r["verdict"] == r["implied"])
        crit = {k: v[1] / v[0] for k, v in cells.items() if k.startswith("COMP") or "|C_" in k}
        report[name] = {"cells": {k: {"n": v[0], "agreement": v[1] / v[0]} for k, v in sorted(cells.items())},
                        "min_criterion_cell": min(crit.values()),
                        "failures": [r for r in rows if r["verdict"] != r["implied"]][:12]}
        print("%s %s min criterion cell %.3f | %s" % (a.model, name, min(crit.values()), " ".join(
            "%s=%.2f" % (k.split("|")[-1] if "|" not in k else k.split("/")[-1][:9] + "|" + k.split("|")[-1], v)
            for k, v in sorted(crit.items()))), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("PROBE_DONE", a.model)


if __name__ == "__main__":
    main()
