"""Patient overlap between the locked set and the validation populations (2026-09-28, review check). Read-only."""
import os
import json
import re
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
PAT = re.compile(r"patient(\d+)")


def pats_of_pair(pid):
    return set(PAT.findall(pid))


val_pool = [json.loads(l)["pair_id"] for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()]
val_238 = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
lock_274 = [l.strip() for l in open(ROOT / "results/locked/manifest_image_necessary.txt") if l.strip()]
# the 548-pair locked read set: first 40 per (finding, direction) cell of the e6 pairs file, in file order
cells, lock_548 = {}, []
for l in open(ROOT / "results/locked/pairs_e6_2026-09-19_0405.jsonl"):
    if not l.strip():
        continue
    d = json.loads(l)
    k = (d["finding"], d["direction"])
    if cells.get(k, 0) < 40:
        cells[k] = cells.get(k, 0) + 1
        lock_548.append(d["pair_id"])
U = lambda ids: set().union(*[pats_of_pair(p) for p in ids]) if ids else set()
pv, p238, p274, p548 = U(val_pool), U(val_238), U(lock_274), U(lock_548)
print("patients: val pool", len(pv), "| val 238", len(p238), "| locked 274", len(p274), "| locked 548", len(p548))
print("overlap locked548 & val pool:", len(p548 & pv), "| locked274 & val238:", len(p274 & p238),
      "| locked548 & val238:", len(p548 & p238), "| locked274 & val pool:", len(p274 & pv))
print("locked read-set size", len(lock_548), "cells", len(cells))
