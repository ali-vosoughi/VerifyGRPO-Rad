#!/usr/bin/env python3
"""Dose test: FREEZE the 64-image training probe and build the dose-test training parquet (2026-09-25).

Protocol (red team round 2 follow-up 1, Q2) with the implementation-review fixes (items 1, 3, 7, 8):
  completeness  exactly --nshards shard files, --n-candidates unique images, every image ok with K rollouts;
                anything else aborts (no silent partial selection)
  eligible      counted ONLY over rollouts that are lenient-canonical objects with a Boolean target flag, on
                images whose target label is certain (0/1): both a correct and an incorrect adjudication AND
                both target-flag values
  select        32 supported + 32 unsupported, deterministic round-robin over (finding, direction) strata
  prefix order  interleaved so every block of 8 holds 4 supported and 4 unsupported (no truth curriculum)
  disjointness  asserted against clean-validation, dev and evaluation patients (harness pairs, cells < 60);
                verl's internal validation (11/64 patient overlap) is QUARANTINED: logged, never decisive
  provenance    sha256 of the source parquet, the patient manifests, and this script, in the manifest
Writes --out-dir/probe.parquet, train.parquet (probe prefix, then every other row in random.Random(--seed)
order, for data.shuffle=False), probe_manifest.json. Refuses to overwrite a frozen probe.
"""
import sys
import os
import argparse
import collections
import glob
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def pat(p):
    return p.split("/")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-dir", default=str(ROOT / "results/probe/select_v1"))
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--n-candidates", type=int, default=640)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--train", default=str(ROOT / "data/verl_v1/train.parquet"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/dose_v1"))
    ap.add_argument("--per-truth", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out_dir)
    if (out / "probe_manifest.json").exists():
        raise SystemExit("PROBE_ALREADY_FROZEN %s" % out)
    files = sorted(Path(args.groups_dir).glob("groups_*.jsonl"))
    if len(files) != args.nshards:
        raise SystemExit("SHARD_COUNT %d != %d" % (len(files), args.nshards))
    groups = [json.loads(l) for f in files for l in open(f) if l.strip()]
    paths = [g.get("path") for g in groups]
    bad = [g for g in groups if not g.get("ok") or len(g.get("rollouts", [])) != args.k]
    if len(groups) != args.n_candidates or len(set(paths)) != len(paths) or bad:
        raise SystemExit("INCOMPLETE groups=%d unique=%d bad=%d" % (len(groups), len(set(paths)), len(bad)))
    groups.sort(key=lambda g: g["i"])
    # disjointness
    val_pat = {pat(l.strip()) for l in open(ROOT / "results/val_clean/images.txt") if l.strip()}
    pairs_file = sorted(glob.glob(str(RAD / "results/e6_pairs/pairs_*.jsonl")))[-1]
    cells = collections.Counter(); evdev = set()
    for l in open(pairs_file):
        if l.strip():
            p = json.loads(l); k = (p["finding"], p["direction"]); i = cells[k]; cells[k] += 1
            if i < 60:
                evdev.add(pat(p["report_path"])); evdev.add(pat(p["image_path"]))
    verlval = {pat(e["path"]) for e in pq.read_table(ROOT / "data/verl_v1/val.parquet", columns=["extra_info"]).column("extra_info").to_pylist()}
    stats = collections.Counter(); elig = []
    for g in groups:
        stats["candidates"] += 1
        if g["label_target"] not in (0, 1):
            stats["label_uncertain"] += 1; continue
        r = [x for x in g["rollouts"] if x["lenient_ok"] and isinstance(x["target_flag"], bool)]
        rew = {x["reward"] for x in r}; flags = {x["target_flag"] for x in r}
        rb = (1.0 in rew) and (0.0 in rew); fb = flags == {True, False}
        stats["reward_both"] += int(rb); stats["flag_both"] += int(fb)
        if rb and fb:
            stats["eligible"] += 1; elig.append(g)
    chosen = {}
    for truth in ("supported", "unsupported"):
        strata = collections.defaultdict(list)
        for g in elig:
            if g["truth"] == truth:
                strata[(g["finding"], g["direction"])].append(g)
        keys = sorted(strata); i = 0; got = []
        while len(got) < args.per_truth and any(strata[k] for k in keys):
            k = keys[i % len(keys)]
            if strata[k]:
                got.append(strata[k].pop(0))
            i += 1
        if len(got) < args.per_truth:
            raise SystemExit("NOT_ENOUGH_ELIGIBLE truth=%s got=%d stats=%s" % (truth, len(got), dict(stats)))
        chosen[truth] = got
    order = []
    for b in range(args.per_truth // 4):   # blocks of 8: 4 supported + 4 unsupported, alternating
        for j in range(4):
            order.append(chosen["supported"][4 * b + j]); order.append(chosen["unsupported"][4 * b + j])
    ppaths = [g["path"] for g in order]
    ppat = {pat(p) for p in ppaths}
    if len(ppat) != len(ppaths):
        raise SystemExit("PATIENT_REPEATED")
    if ppat & val_pat or ppat & evdev or ppat & verlval:
        raise SystemExit("NOT_DISJOINT val=%d evdev=%d verlval=%d" % (len(ppat & val_pat), len(ppat & evdev), len(ppat & verlval)))
    tab = pq.read_table(args.train)
    rows = tab.to_pylist()
    by_path = {r["extra_info"]["path"]: r for r in rows}
    probe_rows = [by_path[p] for p in ppaths]
    pset = set(ppaths)
    rest = [r for r in rows if r["extra_info"]["path"] not in pset]
    random.Random(args.seed).shuffle(rest)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(probe_rows, schema=tab.schema), out / "probe.parquet")
    pq.write_table(pa.Table.from_pylist(probe_rows + rest, schema=tab.schema), out / "train.parquet")
    man = {"frozen": "2026-09-25", "protocol": "red team round 2 follow-up 1 Q2 + implementation review 1,3,7,8",
           "stats": dict(stats), "n_probe": len(ppaths), "order": "blocks of 8 = 4 supported + 4 unsupported, alternating",
           "disjoint_from": {"clean_val_patients": len(val_pat), "eval_dev_patients": len(evdev), "verl_internal_val_patients": len(verlval)},
           "verl_internal_val": "QUARANTINED: 11 of its 64 images share a training patient; its reward is logged, never decisive",
           "probe": [{"pos": n, "path": g["path"], "image_id": g.get("image_id"), "patient": g["patient"], "finding": g["finding"],
                      "direction": g["direction"], "truth": g["truth"], "label_target": g["label_target"]} for n, g in enumerate(order)],
           "train_rows": len(probe_rows) + len(rest), "shuffle_seed_rest": args.seed,
           "sha256": {"probe.parquet": sha(out / "probe.parquet"), "train.parquet": sha(out / "train.parquet"),
                      "source_train.parquet": sha(args.train), "val_clean_images.txt": sha(ROOT / "results/val_clean/images.txt"),
                      "pairs_file": sha(pairs_file), "rl_probe_select.py": sha(__file__)}}
    (out / "probe_manifest.json").write_text(json.dumps(man, indent=1))
    print("FROZEN probe=%d stats=%s sha_probe=%s" % (len(ppaths), dict(stats), man["sha256"]["probe.parquet"][:12]))


if __name__ == "__main__":
    main()
