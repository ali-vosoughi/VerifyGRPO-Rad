#!/usr/bin/env python3
"""Select pairs by verify_run's per-cell rule, then deal them round-robin into N shard files
(2026-09-25), so one read's record arm can run N concurrent harness processes against one vLLM server.
Writes OUT/pairs_XX.jsonl and OUT/n_pairs.txt. The selection is verify_run.main's, line for line.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--per-cell", type=int, default=0)
    ap.add_argument("--cell-skip", type=int, default=0)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    pairs = [l for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.per_cell:
        cells, kept = {}, []
        for l in pairs:
            p = json.loads(l)
            k = (p["finding"], p["direction"]); i = cells.get(k, 0); cells[k] = i + 1
            if args.cell_skip <= i < args.cell_skip + args.per_cell:
                kept.append(l)
        pairs = kept
    if not pairs:
        raise SystemExit("NO_PAIRS_SELECTED")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = max(1, min(args.shards, len(pairs)))
    shards = [[] for _ in range(n)]
    for i, l in enumerate(pairs):
        shards[i % n].append(l)
    for k, s in enumerate(shards):
        (out / ("pairs_%02d.jsonl" % k)).write_text("\n".join(s) + "\n", encoding="utf-8")
    (out / "n_pairs.txt").write_text("%d\n" % len(pairs))
    print("SHARDED pairs=%d shards=%d sizes=%s" % (len(pairs), n, [len(s) for s in shards]))


if __name__ == "__main__":
    main()
