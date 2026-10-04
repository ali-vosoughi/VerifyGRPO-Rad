#!/usr/bin/env python3
"""does training move FORMAT or CONTENT? (2026-09-25)

The verl reward scores a record that fails the strict canonical schema as 0, so within-group reward
variance can come from schema validity alone. This reads the cached records of every checkpoint read of
one run (greedy decoding, the 238 manifest pairs, 476 records) and reports per step: share of records that
pass rl_build_prefs_v1.canonical, the failure reasons, findings listed and marked present, and record
length. CPU only; reads files, calls no model.
"""
import os
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_build_prefs_v1 import canonical  # noqa: E402


def stats(rows_path, pop):
    n = ok_parse = ok_canon = 0
    reasons = collections.Counter(); listed = present = length = 0
    for line in open(rows_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["pair_id"] not in pop or r.get("claim_is_swapped"):
            continue
        n += 1
        rec = r.get("record")
        if not isinstance(rec, dict):
            reasons["no_parsed_record"] += 1
            continue
        ok_parse += 1
        length += len(json.dumps(rec))
        ents = rec.get("record") if isinstance(rec.get("record"), list) else []
        listed += len(ents)
        present += sum(1 for e in ents if isinstance(e, dict) and e.get("present") is True)
        res = canonical(rec)
        ok = res[0]
        if ok:
            ok_canon += 1
        else:
            why = res[2] if len(res) > 2 else "invalid"
            reasons[str(why)[:40]] += 1
    return {"n": n, "parsed": ok_parse, "canonical_ok": ok_canon, "valid_share": round(ok_canon / n, 4) if n else None,
            "mean_listed": round(listed / max(ok_parse, 1), 2), "mean_present": round(present / max(ok_parse, 1), 2),
            "mean_chars": round(length / max(ok_parse, 1)), "reasons": dict(reasons.most_common(5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    ev = ROOT / "results" / "eval2"
    pop = {l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()}
    runs = [(0, ev / "val_step_0" / "sweep_gated_default" / "rows.jsonl")]
    for d in ev.glob("%s_step_*" % args.exp):
        m = re.search(r"_step_(\d+)$", d.name)
        f = d / "sweep_gated_default" / "rows.jsonl"
        if m and f.exists():
            runs.append((int(m.group(1)), f))
    for step, f in sorted(runs):
        print("step %3d %s" % (step, json.dumps(stats(f, pop))))


if __name__ == "__main__":
    main()
