#!/usr/bin/env python3
"""E6 to E8 harness: run one verifier protocol over the matched pairs under one intervention.

Each matched pair supplies one radiologist-written sentence and two studies. The sentence is true
of the report study and false of the partner study, whose structured labels differ on exactly the
finding the sentence carries. Running the same claim against both images is the measurement: a
verifier that uses the image accepts the first and strikes the second.

Protocols
  no_image                 claim only, defines the text-only ceiling
  claim_first              image and claim together, one call
  claim_first_boxes        as above, and the verdict must cite a region
  observe_first            describe the image first, with the claim visible, then judge
  claim_blind              write a structured evidence record from the image with no claim in
                           context, then judge the claim against the record alone
  claim_blind_zoom         as claim_blind, and the first call may request one native-resolution
                           crop, which is appended to the record before judging

Interventions
  --claim-swap             replace the claim with a matched-length sentence about a different
                           finding on the same image; the evidence of an uncontaminated verifier
                           should not move
  --shuffle-boxes          permute cited regions across claims (control for claim_first_boxes)

Output: one JSON line per evaluated row in <out>/rows.jsonl, plus the raw call log the client
writes to <run_dir>/calls.jsonl. Scoring is a separate step (score_run.py); nothing here computes
a paper number.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(ROOT / "code"))

from radagent_open.llm import LLM, parse_json  # noqa: E402
from radagent_open import sweep as _sweep  # noqa: E402

CXP_ZIPDIR = ROOT / "datasets" / "chexpert_plus" / "files" / "PNG_compressed"
IMAGE_INDEX = ROOT / "results" / "e6_pairs" / "image_index.tsv"
CACHE = ROOT / "runs" / "image_cache"

FINDINGS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture",
]

PROMPT_VERSION = "v3"
HARNESS_VERSION = "v5"  # 2026-09-22: --cell-skip, sweep_<record>_<adjudicator> arms (add-only)  # 2026-09-19 evening: absent-default record, both verdicts named

SYS_VERIFIER = (
    "You are verifying a single sentence taken from a chest radiograph report. "
    "Sentences are sometimes true of the radiograph and sometimes not; both verdicts are common. "
    "Answer only with the JSON object asked for. Do not add commentary."
)
SYS_OBSERVER = (
    "You are reading a chest radiograph and recording what you see. "
    "Answer only with the JSON object asked for. Do not add commentary."
)

U_NO_IMAGE = (
    "Sentence from a chest radiograph report:\n{claim}\n\n"
    "Without seeing the radiograph, judge whether this sentence is likely to be true of the study "
    "it was written about.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}}'
)

U_CLAIM_FIRST = (
    "Sentence from a chest radiograph report:\n{claim}\n\n"
    "The radiograph is attached. Judge whether the sentence is true of this radiograph. "
    "Answer unsupported when the radiograph does not show what the sentence says, and "
    "supported when it does.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}}'
)

U_CLAIM_FIRST_BOXES = (
    "Sentence from a chest radiograph report:\n{claim}\n\n"
    "The radiograph is attached. Judge whether the sentence is true of this radiograph and cite "
    "the region you used, as a box in normalised coordinates with the origin at the top left. "
    "Answer unsupported when the radiograph does not show what the sentence says.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "box": [x1, y1, x2, y2], '
    '"reason": "<one short sentence>"}}'
)

U_OBSERVE_FIRST = (
    "Sentence from a chest radiograph report:\n{claim}\n\n"
    "The radiograph is attached. First write down what you observe in the radiograph, then judge "
    "whether the sentence is true of it, citing the region you used. Answer unsupported when the "
    "radiograph does not show what the sentence says.\n"
    'Reply as {{"observations": "<two or three sentences>", '
    '"verdict": "supported" or "unsupported", "box": [x1, y1, x2, y2], '
    '"reason": "<one short sentence>"}}'
)

U_RECORD = (
    "A chest radiograph is attached. Record what it shows, finding by finding, before any claim "
    "about it is made. Most chest radiographs show none or only one of the findings listed, so "
    "the default for every finding is present false; mark a finding present only when you see "
    "clear evidence of it in this image, and say what that evidence is in the note. For each "
    "finding state whether it is present, which side, and the region you would point to.\n"
    "Findings to cover: {findings}.\n"
    'Reply as {{"record": [{{"finding": "<name>", "present": true or false, '
    '"side": "left" or "right" or "bilateral" or "not applicable", '
    '"box": [x1, y1, x2, y2] or null, "note": "<short>"}}], '
    '"other": "<anything notable not in the list, one sentence>"}}'
)

U_RECORD_ZOOM = U_RECORD.rstrip(")") + (
    '\nIf one region needs a closer look before you can fill the record, name it as '
    '"zoom": [x1, y1, x2, y2]; otherwise set "zoom" to null.'
)

U_PROSE = (
    "A chest radiograph is attached. Describe what it shows in plain prose, as a radiologist would "
    "dictate the findings section, before any claim about it is made. Describe the heart size, the "
    "mediastinum, the lungs, the pleural spaces, and the bones, naming only abnormalities you can "
    "actually see; a normal structure is described as normal. Four to seven sentences, no lists.\n"
    'Reply as {{"description": "<the prose>"}}'
)
U_RECORD_SPARSE = (
    "A chest radiograph is attached. Record what it shows before any claim about it is made. Most "
    "chest radiographs show none or only one abnormality, so the default for every finding is "
    "present false. You may mark AT MOST TWO findings present, and only the ones you are most "
    "confident about; give the region for each. Everything else must be present false.\n"
    "Findings to cover: {findings}.\n"
    'Reply as {{"record": [{{"finding": "<name>", "present": true or false, '
    '"side": "left" or "right" or "bilateral" or "not applicable", '
    '"box": [x1, y1, x2, y2] or null, "note": "<short>"}}], '
    '"other": "<anything notable not in the list, one sentence>"}}'
)
U_ADJUDICATE_PROSE = (
    "Here is a description of a chest radiograph, written before any claim was made:\n"
    "{description}\n\n"
    "Sentence from the report of that radiograph:\n{claim}\n\n"
    "Using only the description, judge whether the sentence is true. If the description does not "
    "mention what the sentence asserts, the sentence is unsupported.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}}'
)
U_CLAIM_FIRST_COT = (
    "Sentence from a chest radiograph report:\n{claim}\n\n"
    "The radiograph is attached. Reason step by step about what the radiograph shows in the "
    "region the sentence concerns, then judge whether the sentence is true of this radiograph. "
    "Answer unsupported when the radiograph does not show what the sentence says.\n"
    'Reply as {{"reasoning": "<three to six sentences>", "verdict": "supported" or "unsupported", '
    '"reason": "<one short sentence>"}}'
)
U_ADJUDICATE = (
    "Here is a record of what a chest radiograph shows, written before any claim was made:\n"
    "{record}\n\n"
    "Sentence from the report of that radiograph:\n{claim}\n\n"
    "Using only the record, judge whether the sentence is true.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}}'
)


def load_image_index() -> dict[str, tuple[str, str]]:
    idx: dict[str, tuple[str, str]] = {}
    if not IMAGE_INDEX.exists():
        raise SystemExit("image index missing: run code/scripts/index_images.py first (%s)"
                         % IMAGE_INDEX)
    with IMAGE_INDEX.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                idx[parts[0]] = (parts[1], parts[2])
    return idx


_ZIPS: dict[str, zipfile.ZipFile] = {}


def open_archive(name: str) -> zipfile.ZipFile:
    """Keep one handle per archive. Each holds a 48,000-entry central directory, and reopening a
    155 GB zip for every image costs more than the inference it feeds."""
    if name not in _ZIPS:
        _ZIPS[name] = zipfile.ZipFile(CXP_ZIPDIR / name)
    return _ZIPS[name]


def resolve_image(study_path: str, index: dict[str, tuple[str, str]], max_side: int) -> Path:
    """Extract the study's PNG from its archive once, downscaled, and cache it."""
    key = study_path.rsplit(".", 1)[0]
    dest = CACHE / (key.replace("/", "__") + ("_s%d.png" % max_side))
    if dest.exists():
        return dest
    if key not in index:
        raise KeyError("no archive member for %s" % study_path)
    archive, member = index[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    data = open_archive(archive).read(member)
    tmp = dest.with_suffix(".raw")
    tmp.write_bytes(data)
    if max_side > 0:
        from PIL import Image
        with Image.open(tmp) as im:
            im = im.convert("L")
            if max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
            im.save(dest, format="PNG")
        tmp.unlink()
    else:
        tmp.rename(dest)
    return dest


def build_rows(pairs: list[dict], claim_swap: bool, rng: random.Random) -> list[dict]:
    """Each pair yields the same claim on the true image and on the counterfactual image."""
    pool: dict[str, list[str]] = {}
    for p in pairs:
        pool.setdefault(p["finding"], []).append(p["target_sentence"])
    rows = []
    for p in pairs:
        claim = p["target_sentence"]
        if claim_swap:
            others = [f for f in pool if f != p["finding"] and pool[f]]
            if not others:
                continue
            donor = rng.choice(others)
            candidates = sorted(pool[donor], key=lambda s: abs(len(s) - len(claim)))
            claim = candidates[0]
        for condition, image_path, truth in (
            ("true_image", p["report_path"], "supported"),
            ("swapped_image", p["image_path"], "unsupported"),
        ):
            rows.append({
                "pair_id": p["pair_id"],
                "finding": p["finding"],
                "direction": p["direction"],
                "condition": condition,
                "claim": claim,
                "claim_is_swapped": claim_swap,
                "image_path": image_path,
                "truth": "n/a" if claim_swap else truth,
            })
    return rows


def run_protocol(llm: LLM, protocol: str, row: dict, image: Path | None) -> dict:
    images = [] if (protocol == "no_image" or image is None) else [image]
    out: dict = {"record": None, "observations": None, "box": None, "zoom": None}
    if protocol == "no_image":
        text = llm.chat("verifier", SYS_VERIFIER, U_NO_IMAGE.format(claim=row["claim"]),
                        json_mode=True)
        data = parse_json(text)
    elif protocol == "claim_first":
        text = llm.chat("verifier", SYS_VERIFIER, U_CLAIM_FIRST.format(claim=row["claim"]),
                        images=images, json_mode=True)
        data = parse_json(text)
    elif protocol == "claim_first_boxes":
        text = llm.chat("verifier", SYS_VERIFIER, U_CLAIM_FIRST_BOXES.format(claim=row["claim"]),
                        images=images, json_mode=True)
        data = parse_json(text)
        out["box"] = data.get("box")
    elif protocol == "observe_first":
        text = llm.chat("verifier", SYS_VERIFIER, U_OBSERVE_FIRST.format(claim=row["claim"]),
                        images=images, json_mode=True)
        data = parse_json(text)
        out["box"] = data.get("box")
        out["observations"] = data.get("observations")
    elif protocol == "claim_first_cot":
        text = llm.chat("verifier", SYS_VERIFIER, U_CLAIM_FIRST_COT.format(claim=row["claim"]),
                        images=images, json_mode=True, max_tokens=1024)
        data = parse_json(text)
        out["observations"] = data.get("reasoning")
    elif protocol == "claim_first_sc":
        votes = []
        for k in range(5):
            sampler = LLM(model=llm.model, base_url=llm.base_url, run_dir=llm.run_dir,
                          seed=llm.seed + k + 1, temperature=0.7)
            t = sampler.chat("verifier", SYS_VERIFIER, U_CLAIM_FIRST.format(claim=row["claim"]),
                             images=images, json_mode=True)
            v = str(parse_json(t).get("verdict", "")).strip().lower()
            if v in ("supported", "unsupported"):
                votes.append(v)
        out["observations"] = votes
        data = {"verdict": ("unsupported" if votes.count("unsupported") > len(votes) / 2 else "supported")
                if votes else "", "reason": "majority of %d samples" % len(votes)}
    elif protocol == "claim_blind_xjudge":
        record = RECORDS_FROM.get((row["pair_id"], row["condition"]))
        if record is None:
            raise KeyError("no saved record for %s %s" % (row["pair_id"], row["condition"]))
        out["record"] = record
        payload = record.get("description") if isinstance(record, dict) and "description" in record \
            else json.dumps(record.get("record", record) if isinstance(record, dict) else record)
        template = U_ADJUDICATE_PROSE if (isinstance(record, dict) and "description" in record) else U_ADJUDICATE
        text = llm.chat("adjudicator", SYS_VERIFIER,
                        template.format(description=payload, record=payload, claim=row["claim"]) if template is U_ADJUDICATE_PROSE
                        else template.format(record=payload, claim=row["claim"]),
                        json_mode=True)
        data = parse_json(text)
    elif protocol == "claim_blind_prose":
        desc_text = llm.chat("observer", SYS_OBSERVER, U_PROSE, images=images, json_mode=True,
                             max_tokens=768)
        desc = parse_json(desc_text)
        description = desc.get("description", desc_text) if isinstance(desc, dict) else desc_text
        out["record"] = {"description": description}
        text = llm.chat("adjudicator", SYS_VERIFIER,
                        U_ADJUDICATE_PROSE.format(description=description, claim=row["claim"]),
                        json_mode=True)
        data = parse_json(text)
    elif protocol in ("claim_blind", "claim_blind_zoom", "claim_blind_sparse"):
        template = {"claim_blind_zoom": U_RECORD_ZOOM, "claim_blind_sparse": U_RECORD_SPARSE}.get(protocol, U_RECORD)
        rec_text = llm.chat("observer", SYS_OBSERVER,
                            template.format(findings=", ".join(FINDINGS)),
                            images=images, json_mode=True, max_tokens=1536)
        record = parse_json(rec_text)
        out["record"] = record
        out["zoom"] = record.get("zoom")
        text = llm.chat("adjudicator", SYS_VERIFIER,
                        U_ADJUDICATE.format(record=json.dumps(record.get("record", record)),
                                            claim=row["claim"]),
                        json_mode=True)
        data = parse_json(text)
        for entry in record.get("record", []) or []:
            if isinstance(entry, dict) and entry.get("finding") == row["finding"]:
                out["box"] = entry.get("box")
                break
    elif protocol.startswith("sweep_"):
        fields, data = _sweep.run_protocol(llm, parse_json, protocol, row, images, FINDINGS)
        out.update(fields)
    else:
        raise SystemExit("unknown protocol %s" % protocol)
    out["verdict"] = str(data.get("verdict", "")).strip().lower()
    out["reason"] = data.get("reason", "")
    return out


RECORDS_FROM: dict = {}

def load_records(path: str) -> dict:
    """(pair_id, condition) -> saved record from another model's claim-blind rows.jsonl."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record") is not None:
            out[(r["pair_id"], r["condition"])] = r["record"]
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=0,
                    help="up to N pairs per (finding, direction); stratified, replaces --limit")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--cell-skip", type=int, default=0,
                    help="skip the first N pairs of every (finding, direction) cell before --per-cell")
    ap.add_argument("--claim-swap", action="store_true")
    ap.add_argument("--records-from", default=None,
                    help="rows.jsonl of another model's claim-blind run, for claim_blind_xjudge")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.per_cell:
        cells: dict = {}
        kept = []
        for p in pairs:
            k = (p["finding"], p["direction"])
            i = cells.get(k, 0)
            cells[k] = i + 1
            if args.cell_skip <= i < args.cell_skip + args.per_cell:
                kept.append(p)
        pairs = kept
    elif args.limit:
        pairs = pairs[:args.limit]
    rows = build_rows(pairs, args.claim_swap, rng)
    if args.records_from:
        RECORDS_FROM.update(load_records(args.records_from))
        print("RECORDS_FROM %d records" % len(RECORDS_FROM), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model, base_url=args.base_url, run_dir=out_dir, seed=args.seed)
    index = {} if args.protocol == "no_image" else load_image_index()

    written = 0
    failures = 0
    t0 = time.time()
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            rec = dict(row)
            rec.update({"model": args.model, "protocol": args.protocol,
                        "prompt_version": PROMPT_VERSION, "harness_version": HARNESS_VERSION})
            try:
                image = None
                if args.protocol != "no_image":
                    image = resolve_image(row["image_path"], index, args.max_side)
                    try:
                        from PIL import Image as _Im
                        with _Im.open(image) as _im:
                            rec["image_wh"] = list(_im.size)
                    except Exception:  # noqa: BLE001
                        rec["image_wh"] = None
                rec.update(run_protocol(llm, args.protocol, row, image))
                rec["ok"] = True
            except Exception as exc:  # noqa: BLE001 - a failed row is data, not a crash
                rec.update({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
                failures += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written >= 20 and failures == written:
                fh.flush()
                print("ABORT_ALL_ROWS_FAILED rows=%d last_error=%s" % (written, rec.get("error")), flush=True)
                sys.exit(3)
            if written % 25 == 0:
                print("ROWS %d/%d failures=%d elapsed=%.1fs"
                      % (written, len(rows), failures, time.time() - t0), flush=True)
    print("DONE rows=%d failures=%d elapsed=%.1fs out=%s"
          % (written, failures, time.time() - t0, out_dir / "rows.jsonl"))


if __name__ == "__main__":
    main()
