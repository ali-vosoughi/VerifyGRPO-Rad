#!/usr/bin/env python3
"""Per-finding TRAINING EXPOSURE (2026-09-26).

From a run's per-claim reward log: every (training image, claim) scored during training, grouped over the rollouts
of that image (the GRPO group). Per finding of the claim (from the training parquet's ground truth): images scored,
claim groups, mean reward, and the share of INFORMATIVE groups (reward neither all 0 nor all 1, the only groups that
produce a policy gradient). The 64 in-training validation images are excluded.
Usage: rl_exposure.py RUN_NAME [RUN_NAME ...]  -> results/analysis/exposure/<run>.json and a printed table
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))


def main():
    t = pq.read_table(ROOT / "data/rep_s101/train.parquet", columns=["extra_info", "reward_model"]).to_pylist()
    finding = {}
    for r in t:
        for c in json.loads(r["reward_model"]["ground_truth"])["claims"]:
            finding[(r["extra_info"]["path"], c["claim"])] = c.get("finding")
    val = {e["path"] for e in pq.read_table(ROOT / "data/dose_v1/val.parquet", columns=["extra_info"]).column("extra_info").to_pylist()}
    out = ROOT / "results/analysis/exposure"; out.mkdir(parents=True, exist_ok=True)
    skipped = {}
    for run in sys.argv[1:]:
        groups = defaultdict(list)
        for f in glob.glob(str(ROOT / "runs/reward_logs" / run / "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("path") in val:
                    continue
                if "claim" not in d:          # the logger writes one line without a claim for unparseable output
                    skipped[run] = skipped.get(run, 0) + 1; continue
                groups[(d["path"], d["claim"])].append(float(d.get("hit", 0)))
        per = defaultdict(lambda: {"images": set(), "groups": 0, "informative": 0, "reward_sum": 0.0, "n": 0})
        for (path, claim), hits in groups.items():
            fnd = finding.get((path, claim), "unknown")
            p = per[fnd]; p["images"].add(path); p["groups"] += 1; p["n"] += len(hits); p["reward_sum"] += sum(hits)
            m = sum(hits) / len(hits)
            p["informative"] += int(0 < m < 1)
        res = {f: {"images": len(v["images"]), "groups": v["groups"], "informative_groups": v["informative"],
                   "informative_share": round(v["informative"] / v["groups"], 3), "mean_reward": round(v["reward_sum"] / v["n"], 3),
                   "samples": v["n"]} for f, v in sorted(per.items())}
        (out / (run + ".json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("== %s (%d claim groups; %d log lines without a claim skipped)" % (run, sum(v["groups"] for v in res.values()), skipped.get(run, 0)))
        for f, v in sorted(res.items(), key=lambda kv: -kv[1]["informative_groups"]):
            print("  %-26s images %4d groups %4d informative %4d (%.2f) mean reward %.2f" % (
                f, v["images"], v["groups"], v["informative_groups"], v["informative_share"], v["mean_reward"]))


if __name__ == "__main__":
    main()
