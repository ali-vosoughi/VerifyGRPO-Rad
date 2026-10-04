#!/usr/bin/env python3
"""Dose test (2026-09-25; red team round 2 follow-up 1): K rollouts per image at TRAINING settings,
scored with the v1.2 training reward, with the target flag of every rollout recorded.

Two uses, one code path:
  selection    candidates from verl's train.parquet (single-claim images, one per patient, excluded
               patients removed, deterministic round-robin over finding x direction strata), base writer;
               rl_probe_select.py then freezes the 64-image probe from this output
  measurement  the frozen probe parquet at base / step 20 / step 40 with the SAME fixed per-image seeds
Per image the request seed is derived from the image PATH and a namespace ("select" or "measure"), so the
base, step-20 and step-40 measurement draws of one image share one seed and never reuse a selection seed.
Transport errors are retried with the same seed; the run exits non-zero unless every image returns K
completions (red team round 2 implementation review, items 2-3). Per rollout: parse_ok, lenient_ok, target_flag (present after lenient canonical form,
None if unparseable), flag_correct (target flag == the image's label for the target finding), reward
(rl_verl_reward_lenient.compute_score), finish_reason; the full text is kept in texts.jsonl.
Runs under envs/verl; the writer (base or LoRA "record") and the judge are served by one vLLM.
"""
import os
import argparse
import base64
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from openai import OpenAI

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
import rl_verl_reward as V  # noqa: E402
import rl_verl_reward_lenient as RL  # noqa: E402
from rl_build_prefs_v1 import FINDINGS  # noqa: E402


def pat(p):
    return p.split("/")[1]


def candidates(parquet, exclude_patients, n):
    rows = pq.read_table(parquet).to_pylist()
    strata = defaultdict(list)
    seen = set()
    for r in sorted(rows, key=lambda r: hashlib.sha256(r["extra_info"]["path"].encode()).hexdigest()):
        gt = json.loads(r["reward_model"]["ground_truth"])
        cl = gt.get("claims", [])
        p = pat(r["extra_info"]["path"])
        if len(cl) != 1 or p in exclude_patients or p in seen:
            continue
        seen.add(p)
        strata[(cl[0]["finding"], cl[0]["direction"], cl[0]["truth"])].append(r)
    keys = sorted(strata)
    out, i = [], 0
    while len(out) < n and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            out.append(strata[k].pop(0))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--mode", choices=("select", "measure"), required=True)
    ap.add_argument("--exclude-parquet", default=str(ROOT / "data/verl_v1/val.parquet"),
                    help="patients of these images are excluded from selection (verl's internal validation)")
    ap.add_argument("--n-candidates", type=int, default=640)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed-base", type=int, default=20260925)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--writer", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.mode == "select":
        excl = {pat(e["path"]) for e in pq.read_table(args.exclude_parquet, columns=["extra_info"]).column("extra_info").to_pylist()}
        rows = candidates(args.parquet, excl, args.n_candidates)
    else:
        rows = pq.read_table(args.parquet).to_pylist()
    idx = [i for i in range(len(rows)) if i % args.nshards == args.shard]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=900)

    def one(i):
        row = rows[i]
        gt_s = row["reward_model"]["ground_truth"]
        gt = json.loads(gt_s)
        c = gt["claims"][0]
        labels = row["extra_info"]["labels"]
        lab = labels[FINDINGS.index(c["finding"])] if c["finding"] in FINDINGS else None
        text = row["prompt"][1]["content"].replace("<image>", "", 1)
        img = base64.b64encode(row["images"][0]["bytes"]).decode()
        msgs = [{"role": "system", "content": row["prompt"][0]["content"]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                             {"type": "text", "text": text}]}]
        seed = (args.seed_base + int(hashlib.sha256((args.mode + ":" + row["extra_info"]["path"]).encode()).hexdigest()[:8], 16)) % (2 ** 31)
        resp, err = None, None
        for _ in range(max(1, args.retries)):
            try:
                resp = client.chat.completions.create(model=args.writer, messages=msgs, temperature=args.temp, top_p=1.0,
                                                      max_tokens=args.max_tokens, n=args.k, seed=seed,
                                                      extra_body={"top_k": -1, "repetition_penalty": 1.0})
                break
            except Exception as exc:  # noqa: BLE001
                err = "%s: %s" % (type(exc).__name__, exc); time.sleep(5)
        if resp is None or len(resp.choices) != args.k:
            return {"i": i, "ok": False, "path": row["extra_info"]["path"], "error": err or "choices=%d" % (len(resp.choices) if resp else -1)}, []
        rolls, texts = [], []
        for k, ch in enumerate(resp.choices):
            s = ch.message.content or ""
            texts.append({"i": i, "k": k, "text": s})
            try:
                rec = V._parse_json(s); parse_ok = True
            except Exception:  # noqa: BLE001
                rec, parse_ok = None, False
            lok, canon, _ = RL.canonical_lenient(rec) if parse_ok else (False, None, 0)
            flag = None
            if lok:
                for e in json.loads(canon)["record"]:
                    if e["finding"] == c["finding"]:
                        flag = bool(e["present"])
            rolls.append({"k": k, "parse_ok": parse_ok, "lenient_ok": bool(lok), "target_flag": flag,
                          "flag_correct": (None if flag is None or lab not in (0, 1) else int(flag == (lab == 1))),
                          "reward": float(RL.compute_score("claimblind_cxr", s, gt_s)), "finish": ch.finish_reason})
        return {"i": i, "ok": True, "image_id": hashlib.sha256(row["extra_info"]["path"].encode()).hexdigest()[:16], "path": row["extra_info"]["path"], "patient": pat(row["extra_info"]["path"]),
                "finding": c["finding"], "direction": c["direction"], "truth": c["truth"], "role": c.get("role"),
                "label_target": lab, "seed": seed, "rollouts": rolls}, texts

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(one, idx))
    with (out / ("groups_%02d.jsonl" % args.shard)).open("w") as fg, (out / ("texts_%02d.jsonl" % args.shard)).open("w") as ft:
        for g, texts in res:
            fg.write(json.dumps(g) + "\n")
            for t in texts:
                ft.write(json.dumps(t, ensure_ascii=False) + "\n")
    ok = sum(1 for g, _ in res if g.get("ok"))
    print("DONE rows=%d failures=%d elapsed=%.0fs shard=%d/%d candidates_total=%d" % (
        len(res), len(res) - ok, time.time() - t0, args.shard, args.nshards, len(rows)))
    sys.exit(0 if ok == len(res) else 4)


if __name__ == "__main__":
    main()
