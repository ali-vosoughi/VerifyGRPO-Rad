#!/usr/bin/env python3
"""Leakage audit (red-team finding 6, 2026-09-24). Overlap between the round-0 sampled
training images (selected_round0.txt, 2,000 paths) and the dev slice (cells 41-60), the evaluation
sample (cells 1-40), and the gold set, at three granularities: image path, study, patient. Also the
overlap of normalised sentences. CPU only; prints counts and writes a JSON report."""
import glob
import json
import os
import re
import sys
from collections import Counter

RAD = os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)")
B = os.environ.get("CLAIMBLIND_ROOT", ".")
pairs_file = sorted(glob.glob(RAD + "/results/e6_pairs/pairs_*.jsonl"))[-1]
gold_file = sorted(glob.glob(RAD + "/results/e6_pairs/gold_pairs_*.jsonl"))[-1]
pairs = [json.loads(l) for l in open(pairs_file, encoding="utf-8") if l.strip()]
gold = [json.loads(l) for l in open(gold_file, encoding="utf-8") if l.strip()]
sel = [l.strip() for l in open(B + "/results/train_slice/selected_round0.txt", encoding="utf-8") if l.strip()]


def study(p):
    return p.rsplit("/", 1)[0]


def patient(p):
    return p.split("/")[1]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


cells = Counter()
slices = {"eval": [], "dev": [], "train": []}
for p in pairs:
    k = (p["finding"], p["direction"]); i = cells[k]; cells[k] += 1
    slices["eval" if i < 40 else "dev" if i < 60 else "train"].append(p)
print("pairs_file", os.path.basename(pairs_file), "eval", len(slices["eval"]), "dev", len(slices["dev"]), "train", len(slices["train"]), "gold", len(gold))


def keys(rows):
    imgs = {r["report_path"] for r in rows} | {r["image_path"] for r in rows}
    return {"images": imgs, "studies": {study(x) for x in imgs}, "patients": {patient(x) for x in imgs},
            "sentences": {norm(r["target_sentence"]) for r in rows}}


sel_imgs = set(sel)
sel_keys = {"images": sel_imgs, "studies": {study(x) for x in sel_imgs}, "patients": {patient(x) for x in sel_imgs}}
train_rows_sel = [p for p in slices["train"] if p["report_path"] in sel_imgs or p["image_path"] in sel_imgs]
sel_keys["sentences"] = {norm(p["target_sentence"]) for p in train_rows_sel}
report = {"selected_images": len(sel_imgs), "selected_studies": len(sel_keys["studies"]), "selected_patients": len(sel_keys["patients"]),
          "selected_sentences": len(sel_keys["sentences"]), "overlap": {}}
for name, rows in (("eval", slices["eval"]), ("dev", slices["dev"]), ("gold", gold)):
    k = keys(rows)
    report["overlap"][name] = {g: len(sel_keys[g] & k[g]) for g in ("images", "studies", "patients", "sentences")}
    report["overlap"][name]["size"] = {g: len(k[g]) for g in ("images", "studies", "patients", "sentences")}
# within-training reuse: how many selected images also appear in unselected training pairs (partner reuse)
unsel_train = [p for p in slices["train"] if not (p["report_path"] in sel_imgs or p["image_path"] in sel_imgs)]
uk = keys(unsel_train)
report["unselected_train_overlap"] = {g: len(sel_keys[g] & uk[g]) for g in ("images", "studies", "patients", "sentences")}
report["unselected_train_size"] = {"pairs": len(unsel_train), **{g: len(uk[g]) for g in ("images", "studies", "patients")}}
# candidate clean validation pool: unselected training pairs whose BOTH studies' patients are absent from selected + eval + dev + gold
blocked = set(sel_keys["patients"])
for rows in (slices["eval"], slices["dev"], gold):
    blocked |= keys(rows)["patients"]
clean = [p for p in unsel_train if patient(p["report_path"]) not in blocked and patient(p["image_path"]) not in blocked]
cc = Counter((p["finding"], p["direction"]) for p in clean)
report["clean_validation_pool"] = {"pairs": len(clean), "cells": len(cc), "min_cell": min(cc.values()) if cc else 0, "max_cell": max(cc.values()) if cc else 0}
out = B + "/results/train_slice/leakage_audit.json"
json.dump(report, open(out, "w"), indent=1)
print(json.dumps(report, indent=1))
print("AUDIT_DONE", out)
