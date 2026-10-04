#!/usr/bin/env python3
"""Replication (protocol frozen before the runs): per-seed training parquet = the SAME frozen 64-image
probe prefix (data/dose_v1/probe.parquet, in its frozen order), then every other training row of
data/verl_v1/train.parquet shuffled with random.Random(seed). Used with data.shuffle=False. Writes
data/rep_s<seed>/{train.parquet, val.parquet, manifest.json}; refuses to overwrite.
"""
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main():
    probe_tab = pq.read_table(ROOT / "data/dose_v1/probe.parquet")
    probe = probe_tab.to_pylist()
    ppaths = {r["extra_info"]["path"] for r in probe}
    src = pq.read_table(ROOT / "data/verl_v1/train.parquet")
    rest_all = [r for r in src.to_pylist() if r["extra_info"]["path"] not in ppaths]
    for seed in [int(s) for s in sys.argv[1:]]:
        out = ROOT / ("data/rep_s%d" % seed)
        if (out / "manifest.json").exists():
            print("EXISTS", out); continue
        out.mkdir(parents=True, exist_ok=True)
        rest = list(rest_all)
        random.Random(seed).shuffle(rest)
        pq.write_table(pa.Table.from_pylist(probe + rest, schema=src.schema), out / "train.parquet")
        shutil.copy2(ROOT / "data/verl_v1/val.parquet", out / "val.parquet")
        man = {"seed": seed, "rows": len(probe) + len(rest), "probe_prefix": len(probe),
               "probe_sha256": sha(ROOT / "data/dose_v1/probe.parquet"), "train_sha256": sha(out / "train.parquet"),
               "first_rest_paths": [r["extra_info"]["path"] for r in rest[:3]]}
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        print("BUILT", out, man["rows"], man["train_sha256"][:12], man["first_rest_paths"][0])


if __name__ == "__main__":
    main()
