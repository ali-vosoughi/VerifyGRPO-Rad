#!/usr/bin/env python3
"""Second domain: writer records (protocol sections 3 and 8.4). One record per image, writer, and style, generated once.

The writer sees only the image and the fixed prompt; no question, object, claim, answer, pair, or partner reaches it.
Usage: d2_write_records.py --images LIST --model M --base-url URL --style sparse|checklist --out DIR [--workers 16]
LIST = a text file of image paths relative to $VQA_ROOT (study images: domain2 pool; smoke: images
outside the pool). Prints "DONE rows=N failures=F" for the sbatch checker.
"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_forms as F  # noqa: E402

V = Path(os.environ.get("VQA_ROOT") or sys.exit("set VQA_ROOT to the VQA v2 and COCO 2014 directory (README, photo_study)"))


def parse_json(text):
    """The radiology harness's tolerant extraction (strips code fences and leading prose), kept identical."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(s[a:b + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--style", choices=("sparse", "checklist"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    assert a.base_url.startswith("http://127.0.0.1"), "local server only"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.mkdir(exist_ok=False)               # atomic reservation BEFORE any call: records are written once
    except FileExistsError:
        raise SystemExit("OUTPUT_EXISTS %s (records are written once)" % out)
    rels = [l.strip() for l in open(a.images) if l.strip()]
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=600)
    user = F.WRITER_USER[a.style]

    def one(rel):
        raw = (V / rel).read_bytes()
        img = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}}
        msgs = [{"role": "system", "content": F.WRITER_SYSTEM},
                {"role": "user", "content": [img, {"type": "text", "text": user}]}]
        text, err, finish = "", None, None
        for attempt in range(2):                  # 1 identical transport retry, never a changed prompt
            try:
                r = client.chat.completions.create(model=a.model, messages=msgs, temperature=0.0, seed=20260930,
                                                   max_tokens=2048, response_format={"type": "json_object"})
                text, err, finish = r.choices[0].message.content or "", None, r.choices[0].finish_reason
                break
            except Exception as e:  # noqa: BLE001
                err = "%s: %s" % (type(e).__name__, str(e)[:200]); time.sleep(2)
        parsed = None
        if err is None:
            try:
                parsed = F.parse_writer(parse_json(text), a.style)
            except Exception:  # noqa: BLE001
                parsed = None
        prefix = None
        if finish == "length":                    # a reply stopped at the token limit is never a record (round 6),
            parsed = None                         # even if part of it parses; rule R2 keeps the prefix for diagnostics
            prefix = F.repair_cut_off(text, a.style)
        split = rel.split("/")[0]
        iid = int(rel.rsplit("_", 1)[1].split(".")[0])
        return {"k": "%s_%d" % (split.replace("2014", ""), iid), "path": rel, "image_sha256": hashlib.sha256(raw).hexdigest(),
                "style": a.style, "model": a.model, "response": text, "error": err, "parsed": parsed,
                "finish_reason": finish, "parsed_prefix": prefix,
                "ok": parsed is not None}

    rows = []
    with open(out / "rows.partial.jsonl", "w") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, rel) for rel in rels]):   # persisted the moment each record finishes
            r = fut.result()
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    os.replace(out / "rows.partial.jsonl", out / "rows.jsonl")
    (out / "DONE.json").write_text(json.dumps({
        "tasks": len(rels), "rows": len(rows), "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest(),
        "identity": {"model": a.model, "style": a.style,
                     "images_sha256": hashlib.sha256(Path(a.images).read_bytes()).hexdigest()}}))   # written last
    fails = sum(not r["ok"] for r in rows)
    listed = [len(r["parsed"]["states"]) for r in rows if r["ok"]]
    print("parsed %d/%d, mean objects listed %.2f, dropped entries %d" % (
        len(listed), len(rows), sum(listed) / max(len(listed), 1), sum(r["parsed"]["dropped"] for r in rows if r["ok"])))
    print("DONE rows=%d failures=%d" % (len(rows), fails))


if __name__ == "__main__":
    main()
