#!/usr/bin/env python3
"""Re-adjudicate cached round-0 samples on their CANONICAL record bytes (2026-09-24).

The sampler judged the raw parsed payload; the redesigned reward judges the canonical 12-entry
record (unlisted findings written as absent, fixed order, no extra keys), so the bytes that earn the
reward are the bytes that are trained. This pass is text-only: for every eligible sample and every
claim attached to its image, the frozen base adjudicator (the harness's default rule) judges the canonical
payload, and pair_reward_canon is written next to the original pair_reward. Also writes
pair_reward_flags, judged on a flags-only payload (finding + present, no side, box, or note), which
is the no-free-text control the red team asked for. Threaded; per-thread clients. Exit 4 on zero
judged samples.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(RAD / "code"))
from radagent_open.llm import LLM, parse_json  # noqa: E402
from radagent_open import sweep as sw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_build_prefs_v1 import canonical, pat  # noqa: E402


def flags_payload(canon_str):
    rec = json.loads(canon_str)
    return json.dumps([{"finding": e["finding"], "present": e["present"]} for e in rec["record"]], separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    pool = queue.Queue()
    for w in range(max(1, args.workers)):
        pool.put(LLM(model=args.model, base_url=args.base_url, run_dir=out / "_calls" / ("w%02d" % w), seed=args.seed))
    lock = threading.Lock()
    fh = (out / "samples_canon.jsonl").open("w", encoding="utf-8")
    state = {"done": 0, "judged": 0, "skipped": 0, "errors": 0}
    t0 = time.time()

    def judge(llm, payload, claim):
        try:
            d = parse_json(llm.chat("adjudicator", sw.SYS_VERIFIER, sw.A_DEFAULT_RECORD.format(evidence=payload, claim=claim), json_mode=True))
            return str(d.get("verdict", "")).strip().lower()
        except Exception:  # noqa: BLE001
            return "error"

    def task(s):
        if not s.get("ok"):
            with lock:
                state["skipped"] += 1; state["done"] += 1
            return
        ok, canon, why = canonical(s["record"])
        if not ok:
            s["canon_ok"] = False; s["canon_reason"] = why
            with lock:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n"); state["skipped"] += 1; state["done"] += 1
            return
        llm = pool.get()
        try:
            payload = json.dumps(json.loads(canon)["record"], separators=(",", ":"))
            fp = flags_payload(canon)
            vc, vf = [], []
            for c in s["claims"]:
                vc.append(judge(llm, payload, c["claim"])); vf.append(judge(llm, fp, c["claim"]))
        finally:
            pool.put(llm)
        n = len(s["claims"]) or 1
        s["canon_ok"] = True; s["canon"] = canon; s["canon_reason"] = why
        s["verdicts_canon"] = vc; s["verdicts_flags"] = vf
        s["pair_reward_canon"] = sum(int(v == c["truth"]) for v, c in zip(vc, s["claims"])) / n
        s["pair_reward_flags"] = sum(int(v == c["truth"]) for v, c in zip(vf, s["claims"])) / n
        with lock:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            state["judged"] += 1; state["done"] += 1; state["errors"] += vc.count("error") + vf.count("error")
            if state["done"] % 200 == 0:
                fh.flush()
                print("SAMPLES %d/%d judged=%d skipped=%d errors=%d elapsed=%.0fs" % (state["done"], len(rows), state["judged"], state["skipped"], state["errors"], time.time() - t0), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(task, rows))
    fh.close()
    print("DONE samples=%d judged=%d skipped=%d errors=%d elapsed=%.0fs" % (len(rows), state["judged"], state["skipped"], state["errors"], time.time() - t0))
    sys.exit(0 if state["judged"] > 0 else 4)


if __name__ == "__main__":
    main()
