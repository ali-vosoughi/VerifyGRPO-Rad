#!/usr/bin/env python3
"""DETERMINISTIC FLAG READER (2026-09-25).

Scores a read's cached records with no language-model judge: the record is put in lenient canonical form
(unlisted finding = absent) and the claim is judged by its target flag alone. A false_finding sentence asserts
the finding, so it is supported iff the flag is present; a missed_finding sentence denies it, so it is supported
iff the flag is absent (laterality and wording ignored). G, B, Y, FS on the frozen 238-pair manifest, with a
patient-component bootstrap (2,000 draws). Immune to judge exploitation: it measures the writer's flags only.
Usage: rl_det_reader.py name=rows.jsonl [name=rows.jsonl ...]
"""
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_verl_reward_lenient import canonical_lenient  # noqa: E402


def flag(rec, finding):
    if not isinstance(rec, dict):
        return None
    ok, canon, _ = canonical_lenient(rec)
    if not ok:
        return None
    for e in json.loads(canon)["record"]:
        if e["finding"] == finding:
            return bool(e["present"])
    return False


def main():
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for pid in pop:
        a, b = (pairs[pid][k].split("/")[1] for k in ("report_path", "image_path"))
        parent[find(a)] = find(b)
    comp = defaultdict(list)
    for pid in pop:
        comp[find(pairs[pid]["report_path"].split("/")[1])].append(pid)
    units = list(comp.values())
    for spec in sys.argv[1:]:
        name, path = spec.split("=", 1)
        o = defaultdict(dict)
        for l in open(ROOT / path):
            r = json.loads(l)
            if not r.get("claim_is_swapped"):
                o[r["pair_id"]][r["condition"]] = r
        per = {}
        for pid in pop:
            t, p = o[pid].get("true_image"), o[pid].get("swapped_image")
            if not t or not p:
                continue
            asserts = t["direction"] == "false_finding"
            ft, fp = flag(t.get("record"), t["finding"]), flag(p.get("record"), p["finding"])
            st = True if ft is None else (ft != asserts)      # struck = verdict unsupported; no record = strike
            sp = True if fp is None else (fp != asserts)
            per[pid] = (int(not st and sp), int(st and not sp), int(st))
        def rates(sel):
            n = len(sel); g = sum(per[i][0] for i in sel); b = sum(per[i][1] for i in sel); fs = sum(per[i][2] for i in sel)
            return (g - b) / n, g / n, b / n, fs / n
        ids = [i for i in pop if i in per]
        Y, G, B, FS = rates(ids)
        rng = random.Random(20260925); ys = []
        for _ in range(2000):
            sel = [i for _ in units for i in units[rng.randrange(len(units))] if i in per]
            ys.append(rates(sel)[0])
        ys.sort()
        print("%-16s n=%d  G %.3f  B %.3f  Y %.3f [%.3f, %.3f]  FS %.3f" % (name, len(ids), G, B, Y, ys[50], ys[1949], FS))


if __name__ == "__main__":
    main()
