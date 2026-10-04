#!/usr/bin/env python3
"""Dose test, step-40 audit (2026-09-26; secondary analyses, not gates).
(1) Deterministic flag reader: PAIRED patient-component bootstrap of the change base -> step 40 in G, B, Y, FS.
(2) Why the LLM judge strikes more: on true images, compare base vs step-40 records and verdicts:
    target flag, side given for the target, findings marked present, record length; and cross-tab of
    "flag correct" vs "judge struck" at step 40.
"""
import os
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_verl_reward_lenient import canonical_lenient  # noqa: E402

pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
parent = {}
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for pid in pop:
    a, b = (pairs[pid][k].split("/")[1] for k in ("report_path", "image_path")); parent[find(a)] = find(b)
comp = defaultdict(list)
for pid in pop:
    comp[find(pairs[pid]["report_path"].split("/")[1])].append(pid)
units = list(comp.values())


def canon_entry(rec, finding):
    if not isinstance(rec, dict):
        return None, None, 0
    ok, canon, _ = canonical_lenient(rec)
    if not ok:
        return None, None, 0
    ents = json.loads(canon)["record"]
    npres = sum(1 for e in ents if e["present"])
    for e in ents:
        if e["finding"] == finding:
            return bool(e["present"]), e["side"], npres
    return False, "not applicable", npres


def load(path):
    o = defaultdict(dict)
    for l in open(ROOT / path):
        r = json.loads(l)
        if not r.get("claim_is_swapped"):
            o[r["pair_id"]][r["condition"]] = r
    return o


def det(o):
    per = {}
    for pid in pop:
        t, p = o[pid].get("true_image"), o[pid].get("swapped_image")
        asserts = t["direction"] == "false_finding"
        ft = canon_entry(t.get("record"), t["finding"])[0]; fp = canon_entry(p.get("record"), p["finding"])[0]
        st = True if ft is None else (ft != asserts); sp = True if fp is None else (fp != asserts)
        per[pid] = (int(not st and sp), int(st and not sp), int(st))
    return per


A = load("results/eval2/val_step_0/sweep_gated_default/rows.jsonl")
B = load("results/eval2/dose_v1_lr1e-4_step_40/sweep_gated_default/rows.jsonl")
pa, pb = det(A), det(B)
def rates(per, sel):
    n = len(sel); g = sum(per[i][0] for i in sel); b = sum(per[i][1] for i in sel); fs = sum(per[i][2] for i in sel)
    return {"G": g / n, "B": b / n, "Y": (g - b) / n, "FS": fs / n}
ra, rb = rates(pa, pop), rates(pb, pop)
rng = random.Random(20260926); bs = defaultdict(list)
for _ in range(2000):
    sel = [i for _ in units for i in units[rng.randrange(len(units))]]
    x, y = rates(pa, sel), rates(pb, sel)
    for k in x:
        bs[k].append(y[k] - x[k])
print("(1) DETERMINISTIC READER, paired base -> step 40 (%d pairs, %d patient components)" % (len(pop), len(units)))
for k in ("Y", "G", "B", "FS"):
    v = sorted(bs[k]); print("  d%-2s %+.4f [%+.4f, %+.4f]   %.4f -> %.4f" % (k, rb[k] - ra[k], v[50], v[1949], ra[k], rb[k]))

print("(2) TRUE IMAGES: records and judge verdicts, base vs step 40")
for tag, o in (("base", A), ("step40", B)):
    c = Counter(); npres = []; lens = []
    for pid in pop:
        t = o[pid]["true_image"]; asserts = t["direction"] == "false_finding"
        flag, side, n = canon_entry(t.get("record"), t["finding"])
        npres.append(n); lens.append(len(json.dumps(t.get("record"))))
        flag_ok = (flag is not None) and (flag == asserts)
        struck = (t.get("verdict") or "").startswith("unsupported")
        c[("flag_ok" if flag_ok else "flag_wrong", "struck" if struck else "accepted")] += 1
        if flag_ok and struck and flag:
            c["flag_ok_present_struck_side_" + str(side)] += 1
    print("  %-6s present/record %.2f  chars %.0f  %s" % (tag, sum(npres) / len(npres), sum(lens) / len(lens), dict(sorted(c.items(), key=lambda kv: str(kv[0])))))
