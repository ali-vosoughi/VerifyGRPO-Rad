#!/usr/bin/env python3
"""Repair audit of the reward-scoring form (outside review 5, vulnerability 1; declared in the project log 2026-09-29).

The reward-scoring form (canonical_lenient) does more than fix the order and write omissions as absent: it drops
entries that are not objects, have an unknown or duplicate finding, or an unparseable present flag; parses string
booleans; resets an unknown side to "not applicable"; drops a malformed box; blanks a non-string note; truncates
notes and "other" at 300 characters. order_only and fill_only apply the same repairs, so any repair effect is folded
into the factorial's order and fill factors. This script replays canonical_lenient, instrumented, on every record the
registered reads used (validation: untrained + 3 runs; held-out: untrained + 3 runs), one record per image, and counts
each repair type. No model is run. Output: JSON + text next to --out.
"""
import os
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
import rl_verl_reward_lenient as L  # noqa: E402

FINDINGS, SIDES = L.FINDINGS, L.SIDES

READS = {
    "validation": {"untrained": "results/eval2/controls_v1/step0_in/record",
                   "run1": "results/eval2/rep_s101_step_40/sweep_gated_default",
                   "run2": "results/eval2/rep_s202_step_40/sweep_gated_default",
                   "run3": "results/eval2/rep_s303_step_40/sweep_gated_default"},
    "heldout": {"untrained": "results/eval2/locked_base/sweep_gated_default",
                "run1": "results/eval2/locked_rep_s101_step_40/sweep_gated_default",
                "run2": "results/eval2/locked_rep_s202_step_40/sweep_gated_default",
                "run3": "results/eval2/locked_rep_s303_step_40/sweep_gated_default"},
}
CONTENT = ("record_not_list", "entry_not_object", "finding_unknown", "finding_duplicate", "present_unparseable",
           "side_reset", "box_dropped", "note_not_string", "note_truncated", "other_not_string", "other_truncated")
BENIGN = ("present_string_parsed",)   # same meaning, different type


def audit(rec):
    """Instrumented replay of canonical_lenient: Counter of repair types, plus whether the replay matches it."""
    c = Counter()
    if not isinstance(rec, dict):
        c["not_an_object"] += 1
        return c, None
    entries = rec.get("record")
    if not isinstance(entries, list):
        entries = []; c["record_not_list"] += 1
    seen = set()
    for e in entries:
        if not isinstance(e, dict):
            c["entry_not_object"] += 1; continue
        f = e.get("finding")
        if f not in FINDINGS:
            c["finding_unknown"] += 1; continue
        if f in seen:
            c["finding_duplicate"] += 1; continue
        p = e.get("present")
        if isinstance(p, bool):
            pass
        elif isinstance(p, str) and p.strip().lower() in ("true", "false"):
            c["present_string_parsed"] += 1
        else:
            c["present_unparseable"] += 1; continue
        if e.get("side") not in SIDES:
            c["side_reset"] += 1
        b = e.get("box")
        if b is not None and not (isinstance(b, list) and len(b) == 4 and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in b)):
            c["box_dropped"] += 1
        note = e.get("note", "")
        if not isinstance(note, str):
            c["note_not_string"] += 1
        elif len(note) > 300:
            c["note_truncated"] += 1
        seen.add(f)
    other = rec.get("other", "")
    if not isinstance(other, str):
        c["other_not_string"] += 1
    elif len(other) > 300:
        c["other_truncated"] += 1
    c["listed_valid"] = len(seen)
    ok, _, n_rep = L.canonical_lenient(rec)
    replay = sum(v for k, v in c.items() if k in CONTENT + BENIGN)
    return c, (ok, n_rep, replay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = {"canonicalizer_sha256": hashlib.sha256((ROOT / "code" / "rl_verl_reward_lenient.py").read_bytes()).hexdigest(),
           "populations": {}}
    lines = []
    for pop, reads in READS.items():
        res["populations"][pop] = {}
        for writer, rel in reads.items():
            per_image = {}
            for l in open(ROOT / rel / "rows.jsonl"):
                if not l.strip():
                    continue
                r = json.loads(l)
                per_image.setdefault(r["image_path"], r.get("record"))
            tot = Counter(); affected = Counter(); mismatch = 0; any_content = 0; examples = defaultdict(list)
            for img, rec in per_image.items():
                c, chk = audit(rec)
                if chk is not None and chk[1] != chk[2]:
                    mismatch += 1
                for k, v in c.items():
                    if k == "listed_valid":
                        continue
                    tot[k] += v; affected[k] += 1
                    if len(examples[k]) < 2:
                        examples[k].append(img)
                if any(c[k] for k in CONTENT):
                    any_content += 1
            n = len(per_image)
            res["populations"][pop][writer] = {"read": rel, "records": n, "records_with_content_repair": any_content,
                                               "records_affected_by_type": dict(affected), "repairs_by_type": dict(tot),
                                               "replay_mismatches": mismatch, "examples": dict(examples)}
            lines.append("%-10s %-9s records %4d  with a content repair %4d (%.1f%%)  replay mismatches %d  %s" % (
                pop, writer, n, any_content, 100.0 * any_content / max(n, 1), mismatch,
                ", ".join("%s %d" % (k, affected[k]) for k in sorted(affected))))
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("canonicalizer sha256", res["canonicalizer_sha256"][:16])


if __name__ == "__main__":
    main()
