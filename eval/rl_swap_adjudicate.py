#!/usr/bin/env python3
"""Claim-swap arm on CACHED records (2026-09-24, red-team findings 1 and 10).

The harness's --claim-swap arm regenerates the record, so at trained checkpoints the comparison mixes
record-regeneration noise with the effect of the wording. This pass reads the original arm's
rows.jsonl (one record per pair and condition), rebuilds the swapped claims exactly as
verify_run.build_rows does (same seed, same pair subset, so the donors are identical), and asks the
frozen base adjudicator to judge the CACHED record's canonical payload against the swapped claim.
Output: OUT/rows.jsonl in the harness's row schema plus record_sha256 (identical to the original row's
record hash by construction), donor_finding, donor_direction, and, when a label vector is supplied,
swap_truth (supported/unsupported/unknown) for the truth-preserving versus truth-flipping split.
Text-only, threaded. Exit 4 on zero rows.
"""
import argparse
import hashlib
import json
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(RAD / "code"))
sys.path.insert(0, str(RAD / "code" / "scripts"))
from radagent_open.llm import LLM, parse_json  # noqa: E402
from radagent_open import sweep as sw  # noqa: E402
import verify_run as vr  # noqa: E402


_REPLY_DEFAULT = 'Reply as {"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}'
_REPLY_VARIANT = {
    "reason": 'Reply as {"reasoning": "<two or three sentences that check the sentence against the record>", "verdict": "supported" or "unsupported", "reason": "<one short sentence>"}',
    "cite": 'Reply as {"evidence": "<the record entry about the finding the sentence is about, quoted, or the words not listed if the record has no entry for it>", "verdict": "supported" or "unsupported", "reason": "<one short sentence>"}',
}


def _variant(msg):
    """Opt-in judge prompt variant (added 2026-09-30); unset env var = unchanged message."""
    v = os.environ.get("JUDGE_PROMPT_VARIANT", "")
    if not v:
        return msg
    assert v in _REPLY_VARIANT, v
    assert msg.count(_REPLY_DEFAULT) == 1, "reply line not found exactly once"
    return msg.replace(_REPLY_DEFAULT, _REPLY_VARIANT[v])


def _extra(rec, d):
    e = {k: d[k] for k in ("reasoning", "evidence") if k in d}
    if e:
        rec["judge_extra"] = e

def _convention(msg):
    """Opt-in explicit omission convention (added 2026-09-28); unset env var = unchanged message."""
    c = os.environ.get("JUDGE_CONVENTION", "")
    return msg + "\n\n" + c if c else msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="original arm rows.jsonl (records cached here)")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--per-cell", type=int, default=0)
    ap.add_argument("--cell-skip", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260919, help="verify_run's default seed, so donors match its --claim-swap arm")
    ap.add_argument("--labels", default="", help="jsonl of {path, labels[12]} for swap_truth")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.per_cell:
        cells = {}; kept = []
        for p in pairs:
            k = (p["finding"], p["direction"]); i = cells.get(k, 0); cells[k] = i + 1
            if args.cell_skip <= i < args.cell_skip + args.per_cell:
                kept.append(p)
        pairs = kept
    sent2 = {p["target_sentence"]: (p["finding"], p["direction"]) for p in pairs}
    swap_rows = vr.build_rows(pairs, True, random.Random(args.seed))
    swap_claim = {(r["pair_id"], r["condition"]): r["claim"] for r in swap_rows}
    labels = {}
    if args.labels:
        for l in open(args.labels, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); labels[d["path"]] = d["labels"]
    orig = [json.loads(l) for l in open(args.rows, encoding="utf-8") if l.strip()]
    pool = queue.Queue()
    for w in range(max(1, args.workers)):
        pool.put(LLM(model=args.model, base_url=args.base_url, run_dir=out / "_calls" / ("w%02d" % w), seed=args.seed))
    lock = threading.Lock(); fh = (out / "rows.jsonl").open("w", encoding="utf-8")
    state = {"done": 0, "ok": 0, "fail": 0, "unknown_truth": 0}; t0 = time.time()

    def task(r):
        key = (r["pair_id"], r["condition"])
        rec = dict(r); rec["claim_is_swapped"] = True; rec["cached_record"] = True
        claim = swap_claim.get(key)
        record = r.get("record")
        if claim is None or not r.get("ok") or not isinstance(record, dict):
            rec.update({"ok": False, "error": "no swapped claim or no cached record", "claim": claim})
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); state["fail"] += 1; state["done"] += 1
            return
        payload = json.dumps(record.get("record", record))
        rec["claim"] = claim
        rec["record_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        df, dd = sent2.get(claim, (None, None))
        rec["donor_finding"] = df; rec["donor_direction"] = dd
        truth = "unknown"
        lab = labels.get(r["image_path"])
        if df is not None and lab is not None and df in vr.FINDINGS:
            v = lab[vr.FINDINGS.index(df)]
            asserts_presence = (dd == "false_finding")   # false_finding sentences assert the finding; missed_finding sentences deny it
            if v in (0, 1):
                truth = "supported" if ((v == 1) == asserts_presence) else "unsupported"
        rec["swap_truth"] = truth
        rec["truth"] = "n/a"
        llm = pool.get()
        try:
            text = llm.chat("adjudicator", sw.SYS_VERIFIER, _variant(_convention(sw.A_DEFAULT_RECORD.format(evidence=payload, claim=claim))), json_mode=True)
            d = parse_json(text)
            rec["verdict"] = str(d.get("verdict", "")).strip().lower(); rec["reason"] = d.get("reason", ""); rec["ok"] = True; _extra(rec, d)
        except Exception as exc:  # noqa: BLE001
            rec.update({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
        finally:
            pool.put(llm)
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["done"] += 1; state["ok" if rec.get("ok") else "fail"] += 1; state["unknown_truth"] += int(truth == "unknown")
            if state["done"] % 100 == 0:
                fh.flush(); print("ROWS %d/%d failures=%d elapsed=%.0fs" % (state["done"], len(orig), state["fail"], time.time() - t0), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(task, orig))
    fh.close()
    print("DONE rows=%d failures=%d unknown_truth=%d elapsed=%.1fs out=%s" % (state["done"], state["fail"], state["unknown_truth"], time.time() - t0, out / "rows.jsonl"))
    sys.exit(0 if state["ok"] > 0 else 4)


if __name__ == "__main__":
    main()
