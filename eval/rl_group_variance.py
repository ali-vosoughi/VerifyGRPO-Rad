#!/usr/bin/env python3
"""where does GRPO's within-group reward variance come from? (2026-09-25)

Samples K rollouts per image at the TRAINING settings (verl's held-out val.parquet prompt and image,
temperature T, top_p 1, top_k -1, repetition penalty 1, max 1,024 new tokens) from the writer served by
vLLM (base or LoRA "record"), scores every rollout with the TRAINING reward (rl_verl_reward.compute_score:
strict canonical schema else 0, then the frozen adjudicator on the canonical bytes), and classifies each
group:
  none       every rollout has the same reward: zero GRPO advantage
  validity   rewards differ, but all VALID rollouts share one reward: the spread is schema validity alone
  pair       valid rollouts disagree on the pair term: the content signal the design intends
Writes OUT/groups.jsonl and OUT/summary.json. Runs under envs/verl (pyarrow, openai, the reward module).
Environment: ADJ_BASE_URL, ADJ_MODEL (the base model on the same server), RADOPEN_ROOT.
"""
import os
import argparse
import base64
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from openai import OpenAI

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
import rl_verl_reward as R  # noqa: E402
import rl_verl_reward_lenient as RL  # noqa: E402
from rl_build_prefs_v1 import canonical  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(ROOT / "data/verl_v1/val.parquet"))
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--writer", required=True, help="served model name of the writer (base id or 'record')")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lenient", action="store_true", help="score with the v1.2 reward (rl_verl_reward_lenient); validity still reported under the strict schema")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.parquet).to_pylist()[:args.n_images]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)

    def one(i):
        row = rows[i]
        sys_msg = row["prompt"][0]["content"]
        text = row["prompt"][1]["content"].replace("<image>", "", 1)
        img = base64.b64encode(row["images"][0]["bytes"]).decode()
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                             {"type": "text", "text": text}]}]
        gt = row["reward_model"]["ground_truth"]
        try:
            resp = client.chat.completions.create(model=args.writer, messages=msgs, temperature=args.temp, top_p=1.0,
                                                  max_tokens=args.max_tokens, n=args.k,
                                                  extra_body={"top_k": -1, "repetition_penalty": 1.0})
        except Exception as exc:  # noqa: BLE001
            return {"i": i, "ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        rewards, valid, trunc, reasons = [], [], 0, collections.Counter()
        for ch in resp.choices:
            s = ch.message.content or ""
            trunc += int(ch.finish_reason == "length")
            try:
                ok, _, why = canonical(R._parse_json(s))
            except Exception:  # noqa: BLE001
                ok, why = False, "parse"
            valid.append(bool(ok))
            if not ok:
                reasons[why] += 1
            rewards.append(float((RL if args.lenient else R).compute_score("claimblind_cxr", s, gt)))
        vr = {r for r, v in zip(rewards, valid) if v}
        if len(set(rewards)) <= 1:
            cls = "none"
        elif len(vr) > 1:
            cls = "pair"
        else:
            cls = "validity"
        return {"i": i, "ok": True, "path": row["extra_info"]["path"], "rewards": rewards, "valid": valid,
                "truncated": trunc, "invalid_reasons": dict(reasons), "class": cls}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(one, range(len(rows))))
    with (out / "groups.jsonl").open("w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    good = [r for r in res if r.get("ok")]
    if not good:
        print("NO_GROUPS", res[:2]); sys.exit(4)
    n = len(good); ns = sum(len(r["rewards"]) for r in good)
    cls = collections.Counter(r["class"] for r in good)
    reasons = collections.Counter()
    for r in good:
        reasons.update(r["invalid_reasons"])
    summ = {"writer": args.writer, "reward": "lenient" if args.lenient else "strict", "groups": n, "failed_groups": len(res) - n, "k": args.k, "temp": args.temp,
            "share_none": round(cls["none"] / n, 4), "share_validity_only": round(cls["validity"] / n, 4),
            "share_pair": round(cls["pair"] / n, 4),
            "invalid_rate": round(sum(v is False for r in good for v in r["valid"]) / ns, 4),
            "truncated_rate": round(sum(r["truncated"] for r in good) / ns, 4),
            "mean_reward": round(sum(sum(r["rewards"]) for r in good) / ns, 4),
            "invalid_reasons": dict(reasons.most_common(8)), "elapsed_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("SUMMARY", json.dumps(summ))
    print("DONE rows=%d failures=%d" % (len(res), len(res) - n))


if __name__ == "__main__":
    main()
