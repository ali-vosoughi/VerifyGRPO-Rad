#!/usr/bin/env python3
"""Second domain CONVENTION CALIBRATION set (2026-09-30, before the freeze): pairs OUTSIDE the study pool and the smoke
set, stratified over the 20 categories (up to 3 per category), so that the choice of the convention wording no longer
rests on 8 pairs of 1 category. No study image is read.

Why the labels may be looser than the study's: the calibration scores only comprehension and the target controls, whose
expected verdict follows from the record's stated target state (set by with_target), not from the image, so a label
error cannot move them. Candidates use the study pool's question template and COCO agreement with at least 7 of 10
opposite VQA answers, in the builder's seeded order, image-disjoint from the study pool, the smoke set, and each other.
Writes domain2/calib/{pool.jsonl, labels.json, images.txt, summary.json}.
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

D = Path(os.environ.get("CLAIMBLIND_ROOT", ".")) / "domain2"
sys.path.insert(0, str(D / "code"))
import d2_build_pool as BP  # noqa: E402

# parameters by environment (d2_build_pool reads sys.argv at import); v1 (3, 7, calib) gave 28 pairs in 14 categories
PER_CAT, MIN_VOTES = int(os.environ.get("PER_CAT", 3)), int(os.environ.get("MIN_VOTES", 7))
OUT_NAME = os.environ.get("OUT_NAME", "calib")
study = [json.loads(l) for l in open(D / "pool_v8_cap60_n20" / "pool.jsonl")]
smoke = [json.loads(l) for l in open(D / "smoke" / "pool.jsonl")]
used = set()
for p in study + smoke:
    used |= {(p["split"], p["yes_image_id"]), (p["split"], p["no_image_id"])}
study_ids = {p["pair_id"] for p in study}
cats = list(json.load(open(D / "pool_v8_cap60_n20" / "pool_summary.json"))["checklist"])

cands = []
for split in ("val2014", "train2014"):
    Q, A, P, names, present = BP.load(split)
    for q1, q2 in P:
        if Q[q1]["question"] != Q[q2]["question"] or A[q1]["answer_type"] != "yes/no":
            continue
        m = BP.PAT.match(Q[q1]["question"].strip().lower())
        if not m:
            continue
        obj = m.group("obj").strip()
        cat = obj if obj in cats else (obj[:-1] if obj.endswith("s") and obj[:-1] in cats else ("person" if obj == "people" else None))
        if cat is None:
            continue
        (y1, n1), (y2, n2) = BP.votes(A[q1]), BP.votes(A[q2])
        if y1 >= MIN_VOTES and n2 >= MIN_VOTES:
            yq, nq = q1, q2
        elif y2 >= MIN_VOTES and n1 >= MIN_VOTES:
            yq, nq = q2, q1
        else:
            continue
        iy, ino = Q[yq]["image_id"], Q[nq]["image_id"]
        if cat not in present[iy] or cat in present[ino]:
            continue
        if (split, iy) in used or (split, ino) in used:
            continue
        cands.append({"split": split, "category": cat, "yes_image_id": iy, "no_image_id": ino})
random.Random(20260930).shuffle(cands)
seen, calib, per = set(), [], Counter()
for c in cands:
    k1, k2 = (c["split"], c["yes_image_id"]), (c["split"], c["no_image_id"])
    if per[c["category"]] >= PER_CAT or k1 in seen or k2 in seen:
        continue
    seen |= {k1, k2}
    c["pair_id"] = "CALIB_%s_%s_%d_%d" % (c["category"].replace(" ", "-"), c["split"], c["yes_image_id"], c["no_image_id"])
    assert c["pair_id"] not in study_ids
    calib.append(c)
    per[c["category"]] += 1
out = D / OUT_NAME
out.mkdir(exist_ok=False)
with open(out / "pool.jsonl", "w") as f:
    for c in calib:
        f.write(json.dumps(c) + "\n")
labels, imgs = {}, []
for c in calib:
    s = c["split"].replace("2014", "")
    labels["%s_%d" % (s, c["yes_image_id"])] = "present"
    labels["%s_%d" % (s, c["no_image_id"])] = "absent"
    imgs += ["%s/COCO_%s_%012d.jpg" % (c["split"], c["split"], c["yes_image_id"]),
             "%s/COCO_%s_%012d.jpg" % (c["split"], c["split"], c["no_image_id"])]
json.dump(labels, open(out / "labels.json", "w"))
(out / "images.txt").write_text("\n".join(imgs) + "\n")
summary = {"pairs": len(calib), "per_category": dict(sorted(per.items())), "missing_categories": sorted(set(cats) - set(per)),
           "min_votes": MIN_VOTES, "per_category_cap": PER_CAT, "candidates": len(cands)}
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print(json.dumps(summary))
