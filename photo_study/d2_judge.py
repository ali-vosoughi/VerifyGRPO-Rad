#!/usr/bin/env python3
"""Second domain: judge reads (protocol sections 5, 6, 8.5). Text only, never the image.

Arms: A_unfilled, A_absent, A_notreported (default instruction); B_complete, B_sparse (closed-world contract);
C_corr_complete, C_corr_sparse, C_rev_complete, C_rev_sparse (target controls, closed-world contract);
T_complete, T_sparse (open-world ternary); COMP (the 96 synthetic comprehension cases, no study data).
Before any call the B and C arms assert that their 2 encodings decode to the identical record and log its hash.
Usage: d2_judge.py --model M --base-url URL --arms A_unfilled,B_sparse,... --out DIR [--pool P --labels L --records R
       --style S --writer W] [--workers 32]
Prints "DONE rows=N failures=F".
"""
import argparse
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

ARM = {"A_unfilled": ("unfilled", None, None, False), "A_absent": ("absent", None, None, False),
       "A_notreported": ("not_reported", None, None, False),
       "B_complete": ("complete", "closed", None, False), "B_sparse": ("sparse", "closed", None, False),
       "C_corr_complete": ("complete", "closed", "corr", False), "C_corr_sparse": ("sparse", "closed", "corr", False),
       "C_rev_complete": ("complete", "closed", "rev", False), "C_rev_sparse": ("sparse", "closed", "rev", False),
       "T_complete": ("ternary_complete", "open", None, True), "T_sparse": ("ternary_sparse", "open", None, True)}


def tasks_study(pool, labels, records, style, writer, arms):
    """One task per (eligible pair, image, polarity, arm). labels[k] = 'present' | 'absent' (adjudicated)."""
    recs = {r["k"]: r for r in records}
    assert len(recs) == len(records), "duplicate writer records"
    bad = [r["k"] for r in records if r.get("model") != writer or r.get("style") != style]
    if bad:                                     # a misfiled record must never be read as another writer's or style's
        raise ValueError("records not written by %s in style %s: %s" % (writer, style, bad[:5]))
    out, eq_hashes, missing = [], {}, []
    for p in pool:
        ky = "%s_%d" % (p["split"].replace("2014", ""), p["yes_image_id"])
        kn = "%s_%d" % (p["split"].replace("2014", ""), p["no_image_id"])
        if labels.get(ky) != "present" or labels.get(kn) != "absent":   # every pool pair must carry its pool labels
            raise ValueError("label mismatch for %s" % p["pair_id"])       # (pre-freeze red-team: never a silent skip)
        for role, k in (("yes", ky), ("no", kn)):
            r = recs.get(k)
            if r is None or not r["ok"]:
                missing.append({"pair_id": p["pair_id"], "k": k, "reason": "no record" if r is None else "unparsed"})
                continue                               # counted as missing in the analysis (pool-based accounting)
            rec, cat = r["parsed"], p["category"]
            truth_state = labels[k]
            for arm in arms:
                form, conv, ctrl, tern = ARM[arm]
                base = rec
                if ctrl == "corr":
                    base = F.with_target(rec, cat, truth_state)
                elif ctrl == "rev":
                    base = F.with_target(rec, cat, "absent" if truth_state == "present" else "present")
                ev = F.render(base, form)
                if conv:                                # the 2 encodings must carry the identical record
                    pair_form = {"complete": "sparse", "sparse": "complete", "ternary_complete": "ternary_sparse",
                                 "ternary_sparse": "ternary_complete"}[form]
                    d1, d2 = F.decode_evidence(ev, conv), F.decode_evidence(F.render(base, pair_form), conv)
                    assert d1 == d2, ("decode mismatch", k, arm)
                    eq_hashes[(k, arm)] = hashlib.sha256(json.dumps(d1, sort_keys=True).encode()).hexdigest()[:16]
                for pol in ("assert", "deny"):
                    rec_state = F.decode_evidence(ev, conv or "closed")["states"][cat] if conv else None
                    out.append({"pair_id": p["pair_id"], "k": k, "role": role, "category": cat, "polarity": pol,
                                "truth": F.implied_verdict(truth_state, pol), "arm": arm, "writer": writer,
                                "style": style, "evidence": ev, "convention": conv, "ternary": tern,
                                "record_implied": (F.implied_verdict(rec_state, pol, tern) if conv else None)})
    return out, eq_hashes, missing


def tasks_comp():
    return [{"pair_id": "synthetic", "k": "syn%03d" % i, "role": "synthetic", "category": c["target"],
             "polarity": c["polarity"], "truth": c["expected"], "arm": "COMP_" + c["encoding"], "writer": None,
             "style": None, "evidence": c["evidence"], "convention": "closed", "ternary": False,
             "record_implied": c["expected"]} for i, c in enumerate(F.comprehension_set())]


def parse_verdict(text, ternary):
    """Strict parser (protocol section 6): the whole reply must be one JSON object with a string "verdict" from the
    registered set and a string "reason"; anything else is unparsed. No fence or prose stripping."""
    try:
        o = json.loads(text.strip())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(o, dict) or not isinstance(o.get("verdict"), str) or not isinstance(o.get("reason"), str):
        return None
    v = o["verdict"].strip().lower()
    return v if v in (("supported", "contradicted", "unknown") if ternary else ("supported", "unsupported")) else None


def slot(t):
    return (t["pair_id"], t["k"], t["role"], t["polarity"], t["arm"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def identity(a, arms):
    return {"model": a.model, "writer": a.writer, "style": a.style, "arms": arms,
            "inputs": {"pool": sha(a.pool), "labels": sha(a.labels), "records": sha(a.records),
                       "d2_forms": sha(Path(F.__file__)), "d2_judge": sha(Path(__file__))}}


def verify(out, tasks, ident):
    """0 iff the finished run holds exactly the expected slots, for this judge / writer / style / arms / inputs."""
    try:
        done = json.loads((out / "DONE.json").read_text())
        raw = (out / "rows.jsonl").read_bytes()
    except FileNotFoundError as e:
        return "missing %s" % e.filename
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if hashlib.sha256(raw).hexdigest() != done.get("rows_sha256"):
        return "rows sha256 differs from the DONE marker"
    if done.get("identity") != ident:
        return "identity or inputs differ: %s vs %s" % (done.get("identity"), ident)
    want = {slot(t) for t in tasks}
    got = [slot(r) for r in rows]
    if len(got) != len(set(got)) or set(got) != want or len(rows) != len(tasks):
        return "slots differ: rows %d unique %d expected %d" % (len(got), len(set(got)), len(want))
    if any(r["judge"] != ident["model"] or r["writer"] != (ident["writer"] if r["pair_id"] != "synthetic" else None)
           or r["style"] != (ident["style"] if r["pair_id"] != "synthetic" else None) for r in rows):
        return "a row names another judge, writer, or style"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool"); ap.add_argument("--labels"); ap.add_argument("--records")
    ap.add_argument("--style"); ap.add_argument("--writer")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--verify-only", action="store_true", help="check a finished run against regenerated slots")
    a = ap.parse_args()
    assert a.base_url.startswith("http://127.0.0.1"), "local server only"
    arms = a.arms.split(",")
    if arms == ["COMP"]:
        tasks, eq, missing = tasks_comp(), {}, []
    else:
        pool = [json.loads(l) for l in open(a.pool) if l.strip()]
        labels = json.load(open(a.labels))
        records = [json.loads(l) for l in open(a.records) if l.strip()]
        tasks, eq, missing = tasks_study(pool, labels, records, a.style, a.writer, arms)
    out = Path(a.out)
    ident = identity(a, arms)
    if a.verify_only:
        err = verify(out, tasks, ident)
        print("VERIFY_OK %s slots=%d" % (out, len(tasks)) if err is None else "VERIFY_FAILED %s: %s" % (out, err))
        sys.exit(0 if err is None else 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.mkdir(exist_ok=False)               # atomic reservation BEFORE any call: reads are immutable
    except FileExistsError:
        raise SystemExit("OUTPUT_EXISTS %s (reads are immutable)" % out)
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=600)

    def one(t):
        claim = F.claim(t["category"], t["polarity"])
        msgs = F.judge_messages(t["evidence"], claim, convention=t["convention"], ternary=t["ternary"])
        text, err = "", None
        for attempt in range(2):
            try:
                r = client.chat.completions.create(model=a.model, messages=msgs, temperature=0.0, seed=20260930,
                                                   max_tokens=128, response_format={"type": "json_object"})
                text, err = r.choices[0].message.content or "", None
                break
            except Exception as e:  # noqa: BLE001
                err = "%s: %s" % (type(e).__name__, str(e)[:200]); time.sleep(2)
        verdict = parse_verdict(text, t["ternary"]) if err is None else None
        row = {k: v for k, v in t.items() if k != "evidence"}
        row.update({"judge": a.model, "claim": claim, "verdict": verdict, "ok": verdict is not None, "error": err,
                    "response": text, "evidence_sha": hashlib.sha256(t["evidence"].encode()).hexdigest()[:16]})
        return row

    rows = []
    with open(out / "rows.partial.jsonl", "w") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, t) for t in tasks]):   # persisted the moment each read finishes
            r = fut.result()
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    (out / "decode_equality.json").write_text(json.dumps({"%s|%s" % k: v for k, v in eq.items()}))
    (out / "missing_records.json").write_text(json.dumps(missing))
    os.replace(out / "rows.partial.jsonl", out / "rows.jsonl")
    (out / "DONE.json").write_text(json.dumps({"tasks": len(tasks), "rows": len(rows), "identity": ident,
                                               "rows_sha256": sha(out / "rows.jsonl")}))   # written last
    err = verify(out, tasks, ident)
    if err is not None:
        raise SystemExit("VERIFY_FAILED %s: %s" % (out, err))
    fails = sum(not r["ok"] for r in rows)
    comp = [r for r in rows if r["arm"].startswith("COMP")]
    if comp:
        for enc in ("COMP_complete", "COMP_sparse"):
            rs = [r for r in comp if r["arm"] == enc]
            acc = sum(r["verdict"] == r["record_implied"] for r in rs) / max(len(rs), 1)
            print("comprehension %s %s: %.1f%% of %d (pass >= 90%%)" % (a.model, enc, 100 * acc, len(rs)))
    print("DONE rows=%d failures=%d" % (len(rows), fails))


if __name__ == "__main__":
    main()
