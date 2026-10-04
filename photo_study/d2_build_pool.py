#!/usr/bin/env python3
"""Second domain: the frozen pair pool (2026-09-29). Label-only: questions, human answers, COCO instances, pair lists.
No image pixels, no model, no record, no verdict are touched.

Rules (from the outside reviewers design consult 2026-09-29):
- VQA v2 official complementary pairs (train2014 + val2014 files), yes/no answer type, the SAME question on both images.
- Literal, unqualified existence questions only: "is there a(n) X", "is there any X", "are there any X", "are there X",
  optionally followed by "in the picture / image / photo / photograph / scene"; nothing else (no location, attribute,
  relation, action, count).
- X maps to a COCO category by a frozen table: the category name or its plural, plus "people" -> person and
  "television" -> tv. "man", "woman", "anyone", "animal" etc. are excluded (not the category).
- Human answers: at least MIN_VOTES of 10 say "yes" on one image and at least MIN_VOTES say "no" on the other.
- COCO instance screen: presence = any instance of the category on the image, crowd regions included; the yes image
  must have one, the no image none.
- Image-disjoint: in a fixed seeded order, a pair is kept only if neither image is already used.
- Checklist: the 12 categories with the most eligible pairs; at most CAP pairs per category, taken in the seeded order.
Output: pool.jsonl (one pair per line) + pool_summary.json + images.txt (the images the raters will label).
"""
import os
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

V = Path(os.environ.get("VQA_ROOT") or sys.exit("set VQA_ROOT to the VQA v2 and COCO 2014 directory (README, photo_study)"))
D = Path(os.environ.get("CLAIMBLIND_ROOT", ".")) / "domain2"
MIN_VOTES = int(sys.argv[1]) if len(sys.argv) > 1 else 9
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
SEED = 20260929
N_CATS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
TAIL = r"(?: in (?:the|this) (?:picture|image|photo|photograph|scene))?"
PAT = re.compile(r"^(?:is there an?|is there any|are there any|are there) (?P<obj>[a-z][a-z ]*?)" + TAIL + r"\?$")


def load(split):
    Q = {q["question_id"]: q for q in json.load(open(V / f"v2_OpenEnded_mscoco_{split}_questions.json"))["questions"]}
    A = {a["question_id"]: a for a in json.load(open(V / f"v2_mscoco_{split}_annotations.json"))["annotations"]}
    P = json.load(open(V / f"v2_mscoco_{split}_complementary_pairs.json"))
    inst = json.load(open(D / f"data/annotations/instances_{split}.json"))
    names = {c["id"]: c["name"] for c in inst["categories"]}
    present = defaultdict(set)
    for a in inst["annotations"]:              # crowd regions count as presence
        present[a["image_id"]].add(names[a["category_id"]])
    return Q, A, P, set(names.values()), present


def votes(a):
    c = Counter(x["answer"].strip().lower() for x in a["answers"])
    return c.get("yes", 0), c.get("no", 0)


def main():
    cands, excluded = [], Counter()
    cat_names = None
    for split in ("val2014", "train2014"):
        Q, A, P, cats, present = load(split)
        if cat_names is None:
            cat_names = cats
            plural = {c: (c + "es" if c.endswith(("s", "sh", "ch")) else c + "s") for c in cats}
            table = {c: c for c in cats}
            table.update({v: k for k, v in plural.items()})
            table.update({"people": "person", "television": "tv", "televisions": "tv", "sheep": "sheep"})
        for q1, q2 in P:
            if Q[q1]["question"] != Q[q2]["question"]:
                excluded["different question text"] += 1; continue
            if A[q1]["answer_type"] != "yes/no":
                continue
            m = PAT.match(Q[q1]["question"].strip().lower())
            if not m:
                excluded["not a literal existence question"] += 1; continue
            cat = table.get(m.group("obj").strip())
            if cat is None:
                excluded["object not a COCO category"] += 1; continue
            (y1, n1), (y2, n2) = votes(A[q1]), votes(A[q2])
            if y1 >= MIN_VOTES and n2 >= MIN_VOTES:
                yes_q, no_q = q1, q2
            elif y2 >= MIN_VOTES and n1 >= MIN_VOTES:
                yes_q, no_q = q2, q1
            else:
                excluded["answers not near unanimous and opposite"] += 1; continue
            iy, ino = Q[yes_q]["image_id"], Q[no_q]["image_id"]
            if cat not in present[iy] or cat in present[ino]:
                excluded["COCO instances disagree"] += 1; continue
            cands.append({"split": split, "category": cat, "question": Q[q1]["question"], "yes_image_id": iy,
                          "no_image_id": ino, "yes_votes": votes(A[yes_q])[0], "no_votes": votes(A[no_q])[1],
                          "yes_question_id": yes_q, "no_question_id": no_q})
    rng = random.Random(SEED)
    rng.shuffle(cands)
    used, disjoint = set(), []
    for c in cands:
        k1, k2 = (c["split"], c["yes_image_id"]), (c["split"], c["no_image_id"])
        if k1 in used or k2 in used:
            excluded["image already used"] += 1; continue
        used |= {k1, k2}; disjoint.append(c)
    per_cat = Counter(c["category"] for c in disjoint)
    checklist = [c for c, _ in sorted(per_cat.items(), key=lambda kv: (-kv[1], kv[0]))[:N_CATS]]
    taken, pool = Counter(), []
    for c in disjoint:
        if c["category"] in checklist and taken[c["category"]] < CAP:
            taken[c["category"]] += 1
            c["pair_id"] = "%s_%s_%d_%d" % (c["category"].replace(" ", "-"), c["split"], c["yes_image_id"], c["no_image_id"])
            pool.append(c)
    out = D / ("pool_v%d_cap%d_n%d" % (MIN_VOTES, CAP, N_CATS)); out.mkdir(parents=True, exist_ok=True)
    with open(out / "pool.jsonl", "w") as f:
        for c in pool:
            f.write(json.dumps(c) + "\n")
    imgs = sorted({(c["split"], c["yes_image_id"]) for c in pool} | {(c["split"], c["no_image_id"]) for c in pool})
    (out / "images.txt").write_text("\n".join("%s/COCO_%s_%012d.jpg" % (s, s, i) for s, i in imgs) + "\n")
    summ = {"min_votes": MIN_VOTES, "cap": CAP, "seed": SEED, "candidates": len(cands), "image_disjoint": len(disjoint),
            "checklist": checklist, "per_category_available": {c: per_cat[c] for c in checklist},
            "pairs": len(pool), "per_category_taken": dict(taken), "images": len(imgs),
            "splits": dict(Counter(c["split"] for c in pool)), "excluded": dict(excluded)}
    (out / "pool_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
