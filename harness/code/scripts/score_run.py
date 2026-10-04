#!/usr/bin/env python3
"""Score the harness output. Definitions live here so the paper and the code cannot drift.

A pair contributes two rows under the same claim: the true image, on which the sentence is true,
and the counterfactual image, on which it is false because the structured labels differ on exactly
that finding.

  false-strike rate      share of true-image rows struck as unsupported
  recall                 share of counterfactual rows struck as unsupported
  image-necessary subset pairs whose claim the no-image run judged supported, so the text alone
                         gives no reason to doubt the sentence
  recall_in              recall restricted to that subset, which is the number the paper argues
  balanced accuracy      mean of recall and one minus the false-strike rate
  image following        share of pairs whose verdict differs between the two images
  claim following        share of pairs whose verdict differs when the claim is swapped at a
                         fixed image, measured against the matching claim-swap run
  box displacement       median centre distance between the cited box under the original claim
                         and under the swapped claim, in image widths

Usage:
  score_run.py --runs results/e6_week1/*/rows.jsonl --no-image results/e6_week1/no_image/rows.jsonl
               --out results/e6_week1/summary
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def load(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def struck(row: dict) -> bool | None:
    if not row.get("ok"):
        return None
    v = (row.get("verdict") or "").lower()
    if v.startswith("unsupported"):
        return True
    if v.startswith("supported"):
        return False
    return None


def centre(box) -> tuple[float, float] | None:
    if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


SCORER_VERSION = "v2"  # 2026-09-19: subset-consistent metrics, image_wh box normalisation

def norm_box(box, row):
    """Boxes may be normalised or in pixels of the resized image; return normalised or None."""
    if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        v = [float(x) for x in box]
    except (TypeError, ValueError):
        return None
    if max(v) > 1.5:
        wh = row.get("image_wh") or [1024, 1024]
        w, h = float(wh[0]), float(wh[1])
        v = [v[0] / w, v[1] / h, v[2] / w, v[3] / h]
    return v

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--no-image", default=None,
                    help="rows.jsonl of the no_image protocol, which defines the image-necessary subset")
    ap.add_argument("--claim-swap", nargs="*", default=[],
                    help="rows.jsonl files produced with --claim-swap, for the contamination index")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.runs)
    image_necessary: set[str] = set()
    if args.no_image:
        for r in load([args.no_image]):
            if struck(r) is False:          # the text alone raised no doubt
                image_necessary.add(r["pair_id"])

    swap_box: dict[tuple[str, str, str], tuple[float, float]] = {}
    swap_verdict: dict[tuple[str, str, str], bool] = {}
    for r in load(args.claim_swap):
        key = (r["model"], r["protocol"], r["pair_id"])
        nb = norm_box(r.get("box"), r)
        c = centre(nb) if nb else None
        if c:
            swap_box[key] = c
        s = struck(r)
        if s is not None:
            swap_verdict[key] = s

    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r.get("claim_is_swapped"):
            continue
        groups[(r["model"], r["protocol"])].setdefault(r["pair_id"], {})[r["condition"]] = r

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "protocol_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "protocol", "pairs", "pairs_image_necessary", "recall_all",
                    "recall_image_necessary", "false_strike", "balanced_accuracy",
                    "image_following", "claim_following", "box_displacement", "parse_failures",
                    "false_strike_in", "youden_in", "image_following_in",
                    "youden_in_false_finding", "youden_in_missed_finding", "scorer_version"])
        for (model, protocol), pairs in sorted(groups.items()):
            n = hit = miss = fs_num = fs_den = 0
            hit_in = den_in = 0
            follow_img = follow_den = 0
            follow_claim = follow_claim_den = 0
            displacements: list[float] = []
            fs_in_num = fs_in_den = fol_in = fol_in_den = 0
            dir_acc: dict = {"false_finding": [0, 0, 0], "missed_finding": [0, 0, 0]}
            failures = 0
            for pair_id, conds in pairs.items():
                true_row = conds.get("true_image")
                swap_row = conds.get("swapped_image")
                st = struck(true_row) if true_row else None
                ss = struck(swap_row) if swap_row else None
                if true_row and st is None:
                    failures += 1
                if swap_row and ss is None:
                    failures += 1
                n += 1
                if ss is not None:
                    if ss:
                        hit += 1
                    else:
                        miss += 1
                    if pair_id in image_necessary or not image_necessary:
                        den_in += 1
                        if ss:
                            hit_in += 1
                if st is not None:
                    fs_den += 1
                    if st:
                        fs_num += 1
                if st is not None and ss is not None:
                    follow_den += 1
                    if st != ss:
                        follow_img += 1
                    if pair_id in image_necessary or not image_necessary:
                        fs_in_den += 1
                        fs_in_num += int(st)
                        fol_in_den += 1
                        fol_in += int(st != ss)
                        dr = (true_row or swap_row).get("direction", "")
                        if dr in dir_acc:
                            dir_acc[dr][0] += int(ss); dir_acc[dr][1] += int(st); dir_acc[dr][2] += 1
                key = (model, protocol, pair_id)
                if key in swap_verdict and ss is not None:
                    follow_claim_den += 1
                    if swap_verdict[key] != ss:
                        follow_claim += 1
                if key in swap_box and swap_row is not None:
                    nb0 = norm_box(swap_row.get("box"), swap_row)
                    c0 = centre(nb0) if nb0 else None
                    if c0:
                        c1 = swap_box[key]
                        displacements.append(math.hypot(c1[0] - c0[0], c1[1] - c0[1]))

            recall_all = (hit / (hit + miss)) if (hit + miss) else None
            recall_in = (hit_in / den_in) if den_in else None
            false_strike = (fs_num / fs_den) if fs_den else None
            bal = None
            if recall_in is not None and false_strike is not None:
                bal = (recall_in + (1.0 - false_strike)) / 2.0
            w.writerow([
                model, protocol, n, den_in,
                fmt(recall_all), fmt(recall_in), fmt(false_strike), fmt(bal),
                fmt(follow_img / follow_den if follow_den else None),
                fmt(follow_claim / follow_claim_den if follow_claim_den else None),
                fmt(median(displacements)), failures,
                fmt(fs_in_num / fs_in_den if fs_in_den else None),
                fmt((hit_in / den_in - fs_in_num / fs_in_den) if (den_in and fs_in_den) else None),
                fmt(fol_in / fol_in_den if fol_in_den else None),
                fmt((dir_acc["false_finding"][0] - dir_acc["false_finding"][1]) / dir_acc["false_finding"][2] if dir_acc["false_finding"][2] else None),
                fmt((dir_acc["missed_finding"][0] - dir_acc["missed_finding"][1]) / dir_acc["missed_finding"][2] if dir_acc["missed_finding"][2] else None),
                SCORER_VERSION,
            ])

    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    md = out / "protocol_summary.md"
    with md.open("w", encoding="utf-8") as fh:
        fh.write("# Protocol summary\n\n")
        fh.write("Rows scored: %d. Image-necessary pairs from the no-image run: %d.\n\n"
                 % (len(rows), len(image_necessary)))
        with csv_path.open(encoding="utf-8") as src:
            for i, line in enumerate(src):
                cells = line.rstrip("\n").split(",")
                fh.write("| " + " | ".join(cells) + " |\n")
                if i == 0:
                    fh.write("|" + "---|" * len(cells) + "\n")
        fh.write("\n`protocol_summary.csv` sha256 `%s`\n" % digest)
    print(md.read_text(encoding="utf-8"))
    print("SUMMARY_SHA256 %s" % digest)


def fmt(v) -> str:
    return "" if v is None else ("%.4f" % v)


if __name__ == "__main__":
    main()
