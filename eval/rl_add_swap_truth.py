#!/usr/bin/env python3
"""Add swap truth to the harness's --claim-swap rows (2026-09-25) so rl_score_directional.py can score a
claim-first reference arm on the same footing as the cached-record swap pass.

verify_run.py --claim-swap draws donors with build_rows(pairs, True, Random(20260919)), the same call
rl_swap_adjudicate.py makes, so on the same pairs file the swapped sentences are identical. This pass only
annotates each row with donor_finding, donor_direction and swap_truth (from the image's label vector, the
same rule as rl_swap_adjudicate.py); it calls no model. Claim-first arms have no record, so the scorer's
record_hash_identity is meaningless for them (reported as 0). Exit 4 on zero rows.
"""
import argparse
import json
import os
import sys
from pathlib import Path

RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(RAD / "code" / "scripts"))
import verify_run as vr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="verify_run --claim-swap rows.jsonl")
    ap.add_argument("--pairs", required=True, help="the same pairs file the swap arm ran on")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    sent2 = {p["target_sentence"]: (p["finding"], p["direction"]) for p in pairs}
    labels = {}
    for l in open(args.labels, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); labels[d["path"]] = d["labels"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n = fails = unknown = 0
    with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for l in open(args.rows, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l); n += 1
            df, dd = sent2.get(r.get("claim"), (None, None))
            truth = "unknown"
            lab = labels.get(r.get("image_path"))
            if df is not None and lab is not None and df in vr.FINDINGS:
                v = lab[vr.FINDINGS.index(df)]
                asserts_presence = (dd == "false_finding")
                if v in (0, 1):
                    truth = "supported" if ((v == 1) == asserts_presence) else "unsupported"
            r["donor_finding"] = df; r["donor_direction"] = dd; r["swap_truth"] = truth
            fails += int(not r.get("ok")); unknown += int(truth == "unknown")
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("DONE rows=%d failures=%d unknown_truth=%d out=%s" % (n, fails, unknown, out / "rows.jsonl"))
    sys.exit(0 if n else 4)


if __name__ == "__main__":
    main()
