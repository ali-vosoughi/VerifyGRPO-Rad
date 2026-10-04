#!/usr/bin/env python3
"""Second domain: the automated presence screen that replaces the full human check (protocol section 2, revised
2026-09-30 after the outside reviewers acceptance consult). An image-seeing model from a family used nowhere else in the study
(HuggingFaceM4/Idefics3-8B-Llama3, Llama 3 language model; InternVL2_5-8B was the first choice but its tokenizer does not load in this vLLM stack, smoke 23339) answers 1 question per pool image, "Is there a <object> in this
image?", blind to every label, record, and verdict. A pair passes the screen when the answer is yes on the image VQA and
COCO mark present and no on the other; an unparsed answer fails the screen. The primary analysis uses every pool pair;
the screened subset is a registered secondary analysis. Writes labels.json (VQA + COCO state per image) and screen.jsonl.
Usage: d2_screen.py --pool POOL --model M --base-url URL --out DIR [--workers 16]
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_forms as F  # noqa: E402

V = Path(os.environ.get("VQA_ROOT") or sys.exit("set VQA_ROOT to the VQA v2 and COCO 2014 directory (README, photo_study)"))


def question(cat):
    n = F.DISPLAY[cat]
    return "Is there %s %s in this image? Answer with yes or no." % ("an" if cat in F.AN else "a", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    assert a.base_url.startswith("http://127.0.0.1"), "local server only"
    pool = [json.loads(l) for l in open(a.pool) if l.strip()]
    out = Path(a.out)
    if (out / "screen.jsonl").exists():
        raise SystemExit("SCREEN_EXISTS %s (the screen runs once)" % out)
    out.mkdir(parents=True, exist_ok=True)
    try:                                         # atomic reservation BEFORE any call (a concurrent run fails here)
        open(out / "screen.reserved", "x").close()
    except FileExistsError:
        raise SystemExit("SCREEN_EXISTS %s (reserved by another run)" % out)
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=600)
    items, labels = [], {}
    for p in pool:
        s = p["split"].replace("2014", "")
        for role, iid, state in (("yes", p["yes_image_id"], "present"), ("no", p["no_image_id"], "absent")):
            k = "%s_%d" % (s, iid)
            labels[k] = state                                # VQA >= 8/10 and COCO instances agree (pool rule)
            items.append({"pair_id": p["pair_id"], "k": k, "role": role, "category": p["category"],
                          "path": "%s/COCO_%s_%012d.jpg" % (p["split"], p["split"], iid)})

    def one(it):
        raw = (V / it["path"]).read_bytes()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}},
            {"type": "text", "text": question(it["category"])}]}]
        text, err = "", None
        for attempt in range(2):
            try:
                r = client.chat.completions.create(model=a.model, messages=msgs, temperature=0.0, seed=20260930, max_tokens=8)
                text, err = (r.choices[0].message.content or "").strip(), None
                break
            except Exception as e:  # noqa: BLE001
                err = "%s: %s" % (type(e).__name__, str(e)[:200]); time.sleep(2)
        m = re.fullmatch(r"\s*(yes|no)[\s.!]*", text, flags=re.I)   # full match only (pre-freeze red-team); anything
        ans = m.group(1).lower() if m else None                          # else, "not sure" included, fails the screen
        return dict(it, answer=ans, response=text[:40], error=err)

    rows = []
    with open(out / "screen.partial.jsonl", "w") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, it) for it in items]):   # persisted the moment each answer finishes
            r = fut.result()
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    os.replace(out / "screen.partial.jsonl", out / "screen.jsonl")
    (out / "labels.json").write_text(json.dumps(labels))
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], {})[r["role"]] = r["answer"]
    passed = [pid for pid, d in by.items() if d.get("yes") == "yes" and d.get("no") == "no"]
    (out / "screen_pass.txt").write_text("\n".join(sorted(passed)) + "\n")
    unparsed = sum(r["answer"] is None for r in rows)
    agree = sum((r["answer"] == "yes") == (r["role"] == "yes") for r in rows if r["answer"] is not None)
    print("images %d, unparsed %d, agreement with VQA+COCO %.1f%%, pairs passing the screen %d of %d" % (
        len(rows), unparsed, 100.0 * agree / max(len(rows) - unparsed, 1), len(passed), len(by)))
    print("DONE rows=%d failures=%d" % (len(rows), unparsed))


if __name__ == "__main__":
    main()
