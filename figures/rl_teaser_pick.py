#!/usr/bin/env python3
"""Teaser pair, picked by the rule DECLARED in the project log 2026-09-26 13:45, before any replication
record existed: the first pair in manifest order within pleural effusion whose seed-101 step-40 records are right
on BOTH images under the deterministic flag reader AND under the independent judge; if none exists, the first such
pair in any finding in manifest order. The base records and the three readers' verdicts on that pair are exported
as they are, whatever they are. Writes <out>/pair.json and <out>/{report,partner}.png (real CheXpert Plus images,
longest side 512 px) for Figures 1 and 2.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(RAD / "code" / "scripts"))
from rl_det_reader import flag  # noqa: E402
from rl_verl_reward_lenient import canonical_lenient  # noqa: E402


def load(p):
    o = {}
    for l in open(ROOT / p, encoding="utf-8"):
        r = json.loads(l)
        if not r.get("claim_is_swapped"):
            o.setdefault(r["pair_id"], {})[r["condition"]] = r
    return o


def verdict(r):
    if not r or not r.get("ok"):
        return None
    v = (r.get("verdict") or "").lower()
    return "supported" if v.startswith("supported") else "unsupported" if v.startswith("unsupported") else None


def det_verdict(r):
    f = flag(r.get("record"), r["finding"])
    if f is None:
        return "unsupported"
    asserts = r["direction"] == "false_finding"
    return "supported" if f == asserts else "unsupported"


def entry(r):
    ok, c, _ = canonical_lenient(r.get("record")) if isinstance(r.get("record"), dict) else (False, None, 0)
    if not ok:
        return None, []
    rec = json.loads(c)
    tgt = [e for e in rec["record"] if e["finding"] == r["finding"]][0]
    return tgt, [e["finding"] for e in rec["record"] if e["present"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-rows", default="results/eval2/val_step_0/sweep_gated_default/rows.jsonl")
    ap.add_argument("--base-mg", default="results/eval2/val_step_0_xadj_medgemma/record/rows.jsonl")
    ap.add_argument("--trained-rows", default="results/eval2/rep_s101_step_40/sweep_gated_default/rows.jsonl")
    ap.add_argument("--trained-mg", default="results/eval2/xadj_medgemma/rep_s101_step_40/record/rows.jsonl")
    ap.add_argument("--manifest", default="results/val_clean/manifest_image_necessary.txt")
    ap.add_argument("--out", default="results/analysis/teaser")
    args = ap.parse_args()
    pop = [l.strip() for l in open(ROOT / args.manifest) if l.strip()]
    BQ, BM, TQ, TM = (load(p) for p in (args.base_rows, args.base_mg, args.trained_rows, args.trained_mg))

    def qualifies(pid):
        t, m = TQ.get(pid, {}), TM.get(pid, {})
        if not all(k in t for k in ("true_image", "swapped_image")) or not all(k in m for k in ("true_image", "swapped_image")):
            return False
        det_ok = det_verdict(t["true_image"]) == "supported" and det_verdict(t["swapped_image"]) == "unsupported"
        mg_ok = verdict(m["true_image"]) == "supported" and verdict(m["swapped_image"]) == "unsupported"
        return det_ok and mg_ok

    pick, stage, examined = None, None, 0
    for pid in pop:
        if TQ.get(pid, {}).get("true_image", {}).get("finding") != "Pleural Effusion":
            continue
        examined += 1
        if qualifies(pid):
            pick, stage = pid, "pleural effusion"; break
    if pick is None:
        for pid in pop:
            if qualifies(pid):
                pick, stage = pid, "any finding (no pleural effusion pair qualified)"; break
    if pick is None:
        print("NO_PAIR_QUALIFIES"); sys.exit(3)

    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    res = {"rule": "first pair in manifest order within pleural effusion whose seed-101 step-40 records are right on "
                   "both images under the flag reader and the independent judge; else first such pair in any finding",
           "declared": "project log 2026-09-26 13:45", "stage": stage, "pleural_effusion_pairs_examined": examined,
           "pair_id": pick}
    t0 = TQ[pick]["true_image"]
    res.update({"finding": t0["finding"], "direction": t0["direction"], "sentence": t0["claim"],
                "report_path": t0["image_path"], "partner_path": TQ[pick]["swapped_image"]["image_path"]})
    for tag, Q, M in (("base", BQ, BM), ("trained_s101_step40", TQ, TM)):
        for cond, name in (("true_image", "report"), ("swapped_image", "partner")):
            r, m = Q[pick][cond], M.get(pick, {}).get(cond)
            tgt, present = entry(r)
            res.setdefault(tag, {})[name] = {
                "target_entry": tgt, "present_findings": present,
                "training_judge": {"verdict": verdict(r), "reason": r.get("reason")},
                "independent_judge": {"verdict": verdict(m), "reason": (m or {}).get("reason")},
                "flag_reader": {"verdict": det_verdict(r)},
                "truth": "supported" if cond == "true_image" else "unsupported"}
    import verify_run as vr
    from PIL import Image
    idx = vr.load_image_index()
    for name, key in (("report", "report_path"), ("partner", "partner_path")):
        p = vr.resolve_image(res[key], idx, 1024)
        im = Image.open(p).convert("L"); im.thumbnail((512, 512)); im.save(out / (name + ".png"), optimize=True)
    (out / "pair.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("pair_id", "stage", "pleural_effusion_pairs_examined", "sentence")}, ensure_ascii=False))
    for tag in ("base", "trained_s101_step40"):
        for name in ("report", "partner"):
            d = res[tag][name]
            print(tag, name, "flag", (d["target_entry"] or {}).get("present"), "| TJ", d["training_judge"]["verdict"],
                  "| IJ", d["independent_judge"]["verdict"], "| FR", d["flag_reader"]["verdict"], "| truth", d["truth"])


if __name__ == "__main__":
    main()
