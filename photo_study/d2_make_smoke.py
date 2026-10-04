#!/usr/bin/env python3
"""Second domain smoke inputs (protocol section 8.3): pairs OUTSIDE the frozen study pool, so no study image is read.

Rebuilds the builder's image-disjoint candidate list with the same rules (imports d2_build_pool), then keeps the first
N candidates in the builder's seeded order whose images are not in the study pool (for example pairs beyond a category's
cap). Their labels come from VQA + COCO (pipeline test only, never used for any result).
Writes domain2/smoke/{pool.jsonl, labels.json, images.txt}.
"""
import os
import json
import random
import sys
from pathlib import Path

D = Path(os.environ.get("CLAIMBLIND_ROOT", ".")) / "domain2"
sys.path.insert(0, str(D / "code"))
import d2_build_pool as BP  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
study = [json.loads(l) for l in open(D / "pool_v8_cap60_n20" / "pool.jsonl")]
used = {(p["split"], p["yes_image_id"]) for p in study} | {(p["split"], p["no_image_id"]) for p in study}
study_ids = {p["pair_id"] for p in study}
cats = set(json.load(open(D / "pool_v8_cap60_n20" / "pool_summary.json"))["checklist"])

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
        if y1 >= 8 and n2 >= 8:
            yq, nq = q1, q2
        elif y2 >= 8 and n1 >= 8:
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
seen, smoke = set(), []
for c in cands:
    k1, k2 = (c["split"], c["yes_image_id"]), (c["split"], c["no_image_id"])
    if k1 in seen or k2 in seen:
        continue
    seen |= {k1, k2}
    c["pair_id"] = "SMOKE_%s_%s_%d_%d" % (c["category"].replace(" ", "-"), c["split"], c["yes_image_id"], c["no_image_id"])
    assert c["pair_id"] not in study_ids
    smoke.append(c)
    if len(smoke) == N:
        break
out = D / "smoke"; out.mkdir(exist_ok=True)
with open(out / "pool.jsonl", "w") as f:
    for c in smoke:
        f.write(json.dumps(c) + "\n")
labels = {}
imgs = []
for c in smoke:
    s = c["split"].replace("2014", "")
    labels["%s_%d" % (s, c["yes_image_id"])] = "present"
    labels["%s_%d" % (s, c["no_image_id"])] = "absent"
    imgs += ["%s/COCO_%s_%012d.jpg" % (c["split"], c["split"], c["yes_image_id"]),
             "%s/COCO_%s_%012d.jpg" % (c["split"], c["split"], c["no_image_id"])]
(out / "labels.json").write_text(json.dumps(labels))
(out / "images.txt").write_text("\n".join(imgs) + "\n")
print("smoke pairs", len(smoke), [c["category"] for c in smoke])
