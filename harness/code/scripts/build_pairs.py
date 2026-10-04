#!/usr/bin/env python3
"""E6 instrument, step one: single-label matched pairs from CheXpert Plus.

A pair is two frontal studies whose CheXpert findings-section labels agree on every finding but
one. The report of study A is read against the image of study B, so exactly the sentence about
the differing finding is false on B by construction. Nothing is generated; the ground truth is
the label difference.

Outputs (results/e6_pairs/):
  pairs_<stamp>.jsonl   one row per candidate pair
  summary_<stamp>.csv   counts per finding and direction
  notes_<stamp>.md      provenance, filters, sha256 of the outputs

Run inside the harness environment (needs pandas + pyarrow). CPU only.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime

ROOT = os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)")
CXP = os.path.join(ROOT, "datasets", "chexpert_plus")
TABLE = os.path.join(CXP, "tables", "df_chexpert_plus_240401.parquet")
LABELS = os.path.join(CXP, "files", "CheXpert_Labels", "findings_fixed.json")
OUTDIR = os.path.join(ROOT, "results", "e6_pairs")

# The 12 localizable findings. No Finding and Support Devices are excluded because neither names
# a pathology a verifier could be asked to locate.
FINDINGS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture",
]

# Targets for the first build: the findings the corpus carries in quantity, plus the two thin
# ones, which are reported separately rather than pooled.
TARGETS = [
    "Pleural Effusion", "Cardiomegaly", "Atelectasis", "Edema", "Consolidation",
    "Pneumothorax", "Lung Lesion", "Fracture",
]

# Surface forms used to locate the sentence that carries the finding. A pair is dropped when the
# finding cannot be tied to exactly one sentence, which is the cheap stand-in for an entity parse
# and is replaced by RadGraph-XL spans in step two.
TERMS = {
    "Pleural Effusion": r"effusion|pleural fluid",
    "Cardiomegaly": r"cardiomegaly|enlarged cardiac silhouette|cardiac enlargement|enlargement of the cardiac silhouette|heart size is enlarged",
    "Atelectasis": r"atelecta",
    "Edema": r"edema",
    "Consolidation": r"consolidat",
    "Pneumothorax": r"pneumothora",
    "Lung Lesion": r"nodule|nodular|mass\b|masses\b|lesion",
    "Fracture": r"fractur",
    "Lung Opacity": r"opacit|infiltrat",
    "Pneumonia": r"pneumonia|infectio",
    "Enlarged Cardiomediastinum": r"cardiomediastin|mediastinal silhouette",
    "Pleural Other": r"pleural thickening|pleural scarring",
}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SECTION_HEAD = re.compile(
    r"^\s*(FINDINGS|IMPRESSION|TECHNIQUE|COMPARISON|NARRATIVE|HISTORY|CLINICAL HISTORY|SUMMARY"
    r"|PROCEDURE COMMENTS)\s*:\s*", re.IGNORECASE)
MIN_SENT_CHARS, MAX_SENT_CHARS = 15, 250

# A sentence that compares this study with a prior one cannot be judged from a single image, so
# comparison claims are outside what these pairs can test and are dropped rather than scored.
COMPARISON = re.compile(
    r"\b(compar\w*|previous\w*|prior|interval|unchanged|redemonstrat\w*|again|persistent"
    r"|improv\w*|worsen\w*|increas\w*|decreas\w*|resolv\w*|resolution|stable|new|since"
    r"|similar|still|recent\w*|subsequent\w*|clearing|follow[- ]?up|progress\w*|postoperative"
    r"|as before|no change|first film|earlier)\b",
    re.IGNORECASE)


def norm_label(v):
    """1.0 present, 0.0 absent, -1.0 uncertain, missing treated as absent (CheXpert convention)."""
    if v is None:
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if f == 1.0:
        return 1
    if f == -1.0:
        return -1
    return 0


def sentences(text):
    """Unwrap the report before splitting it.

    CheXpert Plus reports are hard-wrapped at a fixed column, so splitting on newlines cuts
    sentences in half and yields claims like "No consolidation, effusion or pneumothorax is".
    A single newline is joined with a space and only sentence punctuation ends a sentence.
    """
    text = re.sub(r"\s*\n\s*", " ", str(text))
    text = SECTION_HEAD.sub("", text)
    out = []
    for s in SENT_SPLIT.split(text):
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) >= 5:
            out.append(s)
    return out


def locate(text, finding):
    """Return (index, sentence) when exactly one sentence carries the finding and nothing else.

    Both conditions matter for the pair to mean what the paper says it means. The finding must
    appear in exactly one sentence, so the claim is identifiable, and that sentence must mention
    no other finding, so that swapping the image makes exactly one thing false. A sentence like
    "No consolidation, effusion or pneumothorax is seen" fails the second condition and is dropped.
    """
    pat = re.compile(TERMS[finding], re.IGNORECASE)
    others = [(f, re.compile(TERMS[f], re.IGNORECASE)) for f in FINDINGS if f != finding]
    hits = [(i, s) for i, s in enumerate(sentences(text)) if pat.search(s)]
    if len(hits) != 1:
        return None
    idx, sent = hits[0]
    if not (MIN_SENT_CHARS <= len(sent) <= MAX_SENT_CHARS):
        return None
    if any(p.search(sent) for _, p in others):
        return None
    if COMPARISON.search(sent):
        return None
    return idx, sent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-finding", type=int, default=400,
                    help="maximum pairs kept per finding per direction")
    ap.add_argument("--gold-per-finding", type=int, default=150,
                    help="extra pairs admitted per finding per direction when the image carries "
                         "an expert segmentation, which is where box tracking runs")
    ap.add_argument("--gold-partners", type=int, default=6,
                    help="how many different reports each mask-bearing image is paired with")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--min-findings-chars", type=int, default=40)
    ap.add_argument("--train-only", action="store_true",
                    help="drop validation studies, whose expert segmentations disagree with the "
                         "report-derived labels; those studies are the separate gold build")
    args = ap.parse_args()
    random.seed(args.seed)

    import pandas as pd

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    os.makedirs(OUTDIR, exist_ok=True)

    cols = ["path_to_image", "frontal_lateral", "ap_pa", "section_findings", "split",
            "deid_patient_id"]
    df = pd.read_parquet(TABLE, columns=cols)
    n_all = len(df)
    df = df[df["frontal_lateral"].astype(str).str.lower() == "frontal"]
    df = df[df["section_findings"].notna()]
    df = df[df["section_findings"].astype(str).str.len() >= args.min_findings_chars]
    n_usable = len(df)

    labels = {}
    n_uncertain = 0
    with open(LABELS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vec = tuple(norm_label(rec.get(f)) for f in FINDINGS)
            if -1 in vec:
                n_uncertain += 1
                continue
            labels[rec["path_to_image"]] = vec

    studies = []
    for row in df.itertuples(index=False):
        vec = labels.get(row.path_to_image)
        if vec is None:
            continue
        studies.append({
            "path": row.path_to_image,
            "patient": str(row.deid_patient_id),
            "split": str(row.split),
            "text": str(row.section_findings),
            "vec": vec,
        })
    if args.train_only:
        studies = [s for s in studies if s["split"].strip().lower() == "train"]
    n_labelled = len(studies)

    pairs = []
    per_finding = defaultdict(lambda: defaultdict(int))
    per_finding_gold = defaultdict(lambda: defaultdict(int))

    def is_gold(study):
        """A study carries an expert segmentation only if it is outside the training split."""
        return study["split"].strip().lower() not in ("train", "")

    def take(report_study, image_study, finding, direction, context):
        """Admit a pair if the ordinary cap allows it, or if its image carries expert masks.

        The gold allowance exists because the box-tracking experiment can only run where a mask
        exists, and those studies are a few hundred against a corpus of a quarter of a million,
        so random sampling starves the experiment that carries the visual claim.
        """
        gold = is_gold(image_study)
        room = per_finding[finding][direction] < args.per_finding
        gold_room = gold and per_finding_gold[finding][direction] < args.gold_per_finding
        if not (room or gold_room):
            return
        if report_study["patient"] == image_study["patient"]:
            return
        hit = locate(report_study["text"], finding)
        if hit is None:
            return
        pairs.append(mkpair(report_study, image_study, finding, direction, hit, context))
        per_finding[finding][direction] += 1
        if gold:
            per_finding_gold[finding][direction] += 1

    for finding in TARGETS:
        idx = FINDINGS.index(finding)
        buckets = defaultdict(lambda: {1: [], 0: []})
        for st in studies:
            context = st["vec"][:idx] + st["vec"][idx + 1:]
            buckets[context][st["vec"][idx]].append(st)
        for context, sides in buckets.items():
            pos, neg = sides[1], sides[0]
            if not pos or not neg:
                continue
            random.shuffle(pos)
            random.shuffle(neg)
            # studies with expert masks are paired first, so the gold subset is not left to chance
            pos.sort(key=lambda s: not is_gold(s))
            neg.sort(key=lambda s: not is_gold(s))
            for a, b in zip(pos, neg):
                # the report asserts the finding, the partner image lacks it
                take(a, b, finding, "false_finding", context)
                # the report denies the finding, the partner image has it
                take(b, a, finding, "missed_finding", context)
            # Second pass for the studies that carry expert masks. One partner each leaves the
            # box-tracking experiment with a few dozen rows, so each such study is paired with
            # several different reports; the image, which is what the masks belong to, stays the
            # unit, and the reports differ.
            for image_study in [s for s in neg if is_gold(s)]:
                for report_study in pos[:args.gold_partners]:
                    take(report_study, image_study, finding, "false_finding", context)
            for image_study in [s for s in pos if is_gold(s)]:
                for report_study in neg[:args.gold_partners]:
                    take(report_study, image_study, finding, "missed_finding", context)

    pairs_path = os.path.join(OUTDIR, "pairs_%s.jsonl" % stamp)
    with open(pairs_path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    summary_path = os.path.join(OUTDIR, "summary_%s.csv" % stamp)
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["finding", "false_finding", "missed_finding", "total", "gold_subset",
                    "median_claim_chars"])
        gold = defaultdict(int)
        for p in pairs:
            if p["image_split"] in ("valid", "test"):
                gold[p["finding"]] += 1
        lens = defaultdict(list)
        for p in pairs:
            lens[p["finding"]].append(len(p["target_sentence"]))
        for finding in TARGETS:
            ff = per_finding[finding]["false_finding"]
            mf = per_finding[finding]["missed_finding"]
            xs = sorted(lens[finding])
            med = xs[len(xs) // 2] if xs else 0
            w.writerow([finding, ff, mf, ff + mf, gold[finding], med])
        w.writerow(["TOTAL",
                    sum(per_finding[f]["false_finding"] for f in TARGETS),
                    sum(per_finding[f]["missed_finding"] for f in TARGETS),
                    len(pairs), sum(gold.values()), ""])

    notes_path = os.path.join(OUTDIR, "notes_%s.md" % stamp)
    with open(notes_path, "w", encoding="utf-8") as fh:
        fh.write("# E6 matched pairs, build %s\n\n" % stamp)
        fh.write("Source table: %s\n\n" % TABLE)
        fh.write("Source labels: %s\n\n" % LABELS)
        fh.write("Rows in table: %d; frontal with a usable findings section: %d; "
                 "with a certain label vector: %d; studies dropped for an uncertain label: %d\n\n"
                 % (n_all, n_usable, n_labelled, n_uncertain))
        fh.write("Pair rule: identical labels on the other 11 findings, opposite on the target, "
                 "different patients, target tied to exactly one sentence.\n\n")
        fh.write("Cap per finding per direction: %d; seed %d\n\n"
                 % (args.per_finding, args.seed))
        fh.write("Pairs written: %d\n\n" % len(pairs))
        for path in (pairs_path, summary_path):
            fh.write("- `%s` sha256 `%s`\n" % (os.path.basename(path), sha256(path)))

    with open(summary_path, encoding="utf-8") as fh:
        sys.stdout.write(fh.read())
    print("PAIRS_WRITTEN %d" % len(pairs))
    print("OUT %s" % pairs_path)


def mkpair(report_study, image_study, finding, direction, hit, context):
    idx, sentence = hit
    return {
        "pair_id": "%s__%s__%s" % (finding.replace(" ", "_"),
                                   report_study["path"].replace("/", "-"),
                                   image_study["path"].replace("/", "-")),
        "finding": finding,
        "direction": direction,
        "report_path": report_study["path"],
        "report_split": report_study["split"],
        "image_path": image_study["path"],
        "image_split": image_study["split"],
        "target_sentence": sentence,
        "target_sentence_index": idx,
        "report_findings": report_study["text"],
        "context_vector": list(context),
    }


if __name__ == "__main__":
    main()
