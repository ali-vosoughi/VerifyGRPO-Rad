#!/usr/bin/env python3
"""Leakage assertion for the verl training data (2026-09-25; red-team finding on split leakage).

Proves zero patient overlap between the verl GRPO parquet (train + verl's own held-out val) and
(a) the frozen clean validation set every checkpoint is read on, (b) the harness's evaluation sample and dev
slice (cells 0 to 59 of the pairs file). Also prints the parquet sha256 against data/verl_v1/manifest.json
so the local copy is known to be the file the training run uses. Exit 3 on any overlap or sha mismatch.
Runs under the verl environment (pyarrow). Reads only the extra_info column.
"""
import argparse
import collections
import glob
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def patient(path):
    return path.split("/")[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/verl_v1 directory")
    ap.add_argument("--val-images", required=True, help="results/val_clean/images.txt")
    ap.add_argument("--pairs", default="", help="harness pairs file; default = newest in RADOPEN_ROOT/results/e6_pairs")
    ap.add_argument("--rad", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    man = json.loads((data / "manifest.json").read_text())
    bad = 0
    paths = {}
    for split in ("train", "val"):
        f = data / ("%s.parquet" % split)
        got = sha256(f)
        want = man.get("%s_sha256" % split)
        print("SHA %s %s manifest=%s match=%s" % (split, got, want, got == want))
        bad += int(got != want)
        t = pq.read_table(f, columns=["extra_info"]).column("extra_info").to_pylist()
        paths[split] = [e["path"] for e in t]
    train_pat = {patient(p) for p in paths["train"]} | {patient(p) for p in paths["val"]}
    val_pat = {patient(l.strip()) for l in open(args.val_images) if l.strip()}
    pf = args.pairs or sorted(glob.glob(args.rad + "/results/e6_pairs/pairs_*.jsonl"))[-1]
    cells = collections.Counter(); evdev = set()
    for line in open(pf):
        if not line.strip():
            continue
        p = json.loads(line)
        k = (p["finding"], p["direction"]); i = cells[k]; cells[k] += 1
        if i < 60:
            evdev.add(patient(p["report_path"])); evdev.add(patient(p["image_path"]))
    o1 = train_pat & val_pat; o2 = train_pat & evdev
    print("PATIENTS verl_train+val=%d clean_val=%d eval+dev=%d pairs_file=%s" % (len(train_pat), len(val_pat), len(evdev), pf))
    print("OVERLAP verl_vs_clean_val=%d verl_vs_eval_dev=%d" % (len(o1), len(o2)))
    if o1 or o2:
        print("LEAK examples", sorted(o1)[:5], sorted(o2)[:5])
        bad += 1
    print("DISJOINT_OK" if not bad else "DISJOINT_FAILED")
    sys.exit(0 if not bad else 3)


if __name__ == "__main__":
    main()
