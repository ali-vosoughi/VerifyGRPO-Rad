#!/usr/bin/env python3
"""Falsification controls on the evaluation-v2 metric (2026-09-25; red-team plan item 7, design v1 item 6).

Takes the cached step-0 record-arm rows, restricts them to the frozen image-necessary manifest, replaces
the writer's record with a CONTROL record, and asks the frozen base adjudicator to judge it against the
ORIGINAL claim exactly as the harness's sweep arm does (SYS_VERIFIER + A_DEFAULT_RECORD, payload = the entry
list, temperature 0). The rows then go through the unchanged cached-swap pass (rl_swap_adjudicate.py) and
directional scorer (rl_score_directional.py), so every control sits on the same population, donors, and
metric as a trained checkpoint read. Policies:
  step0_in        the step-0 writer's own rows, restricted to the manifest, NOT re-adjudicated: the step-0
                  point recomputed with the swap donors of the 238-pair file that checkpoint reads use
  all_absent      twelve entries, all absent
  all_present     twelve entries, all present
  top5            the five findings most often positive in the training-slice labels (validation images
                  excluded) present, the rest absent
  permute_cell    the step-0 writer's records deranged within (finding, direction, condition): the donor
                  image carries the same truth for the target finding, so a label-level record keeps Y
  permute_global  the step-0 writer's records deranged across all rows of the population: record content
                  is real but unrelated to the image, so G ~ B and I is the flip level from record noise
  oracle_labels   the record written from the image's own label vector (present = label 1, side "not
                  applicable", no box, no note): a PERFECT flags record, the ceiling of what any writer
                  can earn from this adjudicator on this metric (2026-09-25, red-team round 2 Q3)
  canon_readj     the rows' OWN records rewritten to the TRAINING serialization (lenient canonical: twelve
                  entries, unlisted = explicit absent) and re-adjudicated by the base judge: the train /
                  evaluation interface audit of red team round 2, finding 1 (2026-09-25). Only content
                  changes (missing -> explicit absent, repairs); JSON whitespace stays the harness's
  readj           the rows' OWN records re-adjudicated by whatever model is served (--model): the
                  adjudicator-transfer control (a policy trained against the frozen Qwen3-VL-8B judge is
                  re-judged by an independent family; gains that vanish there are judge exploits)
Constant policies give I = 0 by construction (one record on both images); they bound C_pres and FS.
Entries of control records: side "not applicable", box null, note "" (flags-only; the adjudicator reads
the flags at step 0). Exit 4 on zero rows.
"""
import argparse
import collections
import json
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(RAD / "code"))
sys.path.insert(0, str(RAD / "code" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from radagent_open.llm import LLM, parse_json  # noqa: E402
from radagent_open import sweep as sw  # noqa: E402
import verify_run as vr  # noqa: E402


_REPLY_DEFAULT = 'Reply as {"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}'
_REPLY_VARIANT = {
    "reason": 'Reply as {"reasoning": "<two or three sentences that check the sentence against the record>", "verdict": "supported" or "unsupported", "reason": "<one short sentence>"}',
    "cite": 'Reply as {"evidence": "<the record entry about the finding the sentence is about, quoted, or the words not listed if the record has no entry for it>", "verdict": "supported" or "unsupported", "reason": "<one short sentence>"}',
}


def _variant(msg):
    """Opt-in judge prompt variant (added 2026-09-30); unset env var = unchanged message."""
    v = os.environ.get("JUDGE_PROMPT_VARIANT", "")
    if not v:
        return msg
    assert v in _REPLY_VARIANT, v
    assert msg.count(_REPLY_DEFAULT) == 1, "reply line not found exactly once"
    return msg.replace(_REPLY_DEFAULT, _REPLY_VARIANT[v])


def _extra(rec, d):
    e = {k: d[k] for k in ("reasoning", "evidence") if k in d}
    if e:
        rec["judge_extra"] = e

def _convention(msg):
    """Opt-in explicit omission convention (added 2026-09-28); unset env var = unchanged message."""
    c = os.environ.get("JUDGE_CONVENTION", "")
    return msg + "\n\n" + c if c else msg

POLICIES = ("step0_in", "all_absent", "all_present", "top5", "permute_cell", "permute_global", "readj", "oracle_labels", "canon_readj", "side_neutral", "notes_removed", "flags_only", "hybrid_BB", "hybrid_BT", "hybrid_TB", "hybrid_TT", "order_only", "fill_only", "fill_target", "fill_other", "fill_unreported")


def constant_record(present):
    return {"record": [{"finding": f, "present": f in present, "side": "not applicable", "box": None, "note": ""}
                       for f in vr.FINDINGS], "other": ""}


def top5(path, exclude):
    cnt = collections.Counter(); n = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d["path"] in exclude:
            continue
        n += 1
        for f, v in zip(vr.FINDINGS, d["labels"]):
            cnt[f] += int(v == 1)
    return [f for f, _ in cnt.most_common(5)], {f: round(cnt[f] / n, 4) for f in vr.FINDINGS}, n


def sattolo(n, rng):
    """A uniformly random single-cycle permutation of range(n): no fixed points for n >= 2."""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="step-0 record-arm rows.jsonl (base writer)")
    ap.add_argument("--manifest", required=True, help="frozen image-necessary pair_id list")
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--base-rows", default="", help="hybrid_*: the base read rows")
    ap.add_argument("--trained-rows", default="", help="hybrid_*: the trained read rows")
    ap.add_argument("--train-labels", default="", help="jsonl of {path, labels[12]} for top5")
    ap.add_argument("--exclude-paths", default="", help="image paths excluded from the top5 prevalence count")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--oracle-labels", default="", help="jsonl of {path, labels[12]} for oracle_labels")
    ap.add_argument("--compact", action="store_true", help="judge the payload with compact JSON separators (the training reward's bytes) instead of the harness default")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pop = {l.strip() for l in open(args.manifest, encoding="utf-8") if l.strip()}
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["pair_id"] in pop and not r.get("claim_is_swapped")]
    meta = {"policy": args.policy, "rows_in": len(rows), "pairs": len({r["pair_id"] for r in rows}), "seed": args.seed}
    if args.policy == "step0_in":
        with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fails = sum(1 for r in rows if not r.get("ok"))
        (out / "control_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print("DONE rows=%d failures=%d elapsed=0.0s out=%s" % (len(rows), fails, out / "rows.jsonl"))
        sys.exit(0 if rows else 4)

    rng = random.Random(args.seed)
    new = {}          # index -> (record, source)
    if args.policy == "all_absent":
        rec = constant_record(set())
        new = {i: (rec, "constant") for i in range(len(rows))}
    elif args.policy == "all_present":
        rec = constant_record(set(vr.FINDINGS))
        new = {i: (rec, "constant") for i in range(len(rows))}
    elif args.policy == "top5":
        excl = {l.strip() for l in open(args.exclude_paths, encoding="utf-8") if l.strip()} if args.exclude_paths else set()
        top, prev, n = top5(args.train_labels, excl)
        meta.update({"top5": top, "prevalence": prev, "label_images": n, "excluded_images": len(excl)})
        rec = constant_record(set(top))
        new = {i: (rec, "constant") for i in range(len(rows))}
    elif args.policy == "oracle_labels":
        lab = {}
        for line in open(args.oracle_labels, encoding="utf-8"):
            if line.strip():
                d = json.loads(line); lab[d["path"]] = d["labels"]
        miss = 0
        for i, r in enumerate(rows):
            v = lab.get(r["image_path"])
            if v is None:
                miss += 1; continue
            new[i] = (constant_record({f for f, x in zip(vr.FINDINGS, v) if x == 1}), "labels")
        meta.update({"oracle_missing_labels": miss})
    elif args.policy == "notes_removed":
        # 2026-09-26, reader-reversal check: each record in lenient canonical form (same
        # twelve fields at every checkpoint) with every note emptied and "other" dropped; flags, sides and
        # boxes kept; re-judged against the original claims
        from rl_verl_reward_lenient import canonical_lenient
        for i, r in enumerate(rows):
            if r.get("ok") and isinstance(r.get("record"), dict):
                ok, canon, _ = canonical_lenient(r["record"])
                if ok:
                    c = json.loads(canon)
                    c["record"] = [dict(e, note="") for e in c["record"]]; c["other"] = ""
                    new[i] = (c, "notes_removed")
    elif args.policy.startswith("hybrid_"):
        # 2026-09-26 crossed flags x notes: each record takes its PRESENT FLAGS
        # from one read and its NOTES (and "other") from another, for the same image; sides "not applicable" and
        # boxes null in all four cells so that only flags and text vary. hybrid_XY: X = flag source, Y = note
        # source, B = base read, T = trained read. Records are controlled interventions, not faithful reports.
        from rl_verl_reward_lenient import canonical_lenient
        src = {}
        for tag, path in (("B", args.base_rows), ("T", args.trained_rows)):
            m = {}
            for line in open(path, encoding="utf-8"):
                r0 = json.loads(line)
                if not r0.get("claim_is_swapped") and r0.get("ok") and isinstance(r0.get("record"), dict):
                    ok, canon, _ = canonical_lenient(r0["record"])
                    if ok:
                        m[(r0["pair_id"], r0["condition"])] = json.loads(canon)
            src[tag] = m
        fx, nx = args.policy[-2], args.policy[-1]
        missing = 0
        for i, r in enumerate(rows):
            k = (r["pair_id"], r["condition"])
            if k not in src[fx] or k not in src[nx]:
                missing += 1; continue
            flags = {e["finding"]: e["present"] for e in src[fx][k]["record"]}
            notes = {e["finding"]: e["note"] for e in src[nx][k]["record"]}
            rec = {"record": [{"finding": f, "present": flags[f], "side": "not applicable", "box": None, "note": notes[f]}
                              for f in vr.FINDINGS], "other": src[nx][k].get("other", "")}
            new[i] = (rec, args.policy)
        meta.update({"hybrid_missing": missing, "flag_source": fx, "note_source": nx})
    elif args.policy in ("order_only", "fill_only"):
        # 2026-09-26 order x missingness (see patch docstring): repaired entries in canonical form, then either keep
        # the writer's omissions in the fixed order (order_only) or keep the writer's order and append the omitted
        # findings as explicit absent entries (fill_only)
        from rl_verl_reward_lenient import canonical_lenient
        for i, r in enumerate(rows):
            rec0 = r.get("record")
            if not (r.get("ok") and isinstance(rec0, dict)):
                continue
            ok, canon, _ = canonical_lenient(rec0)
            if not ok:
                continue
            c = json.loads(canon)
            by = {e["finding"]: e for e in c["record"]}
            listed = []
            for e in rec0.get("record", []) if isinstance(rec0.get("record"), list) else []:
                f = e.get("finding") if isinstance(e, dict) else None
                if f in by and f not in listed and isinstance(e.get("present"), (bool, str)):
                    listed.append(f)
            if args.policy == "order_only":
                entries = [by[f] for f in vr.FINDINGS if f in listed]
            else:
                entries = [by[f] for f in listed] + [by[f] for f in vr.FINDINGS if f not in listed]
            new[i] = ({"record": entries, "other": c.get("other", "")}, args.policy)
    elif args.policy in ("fill_target", "fill_other"):
        # 2026-09-26 target-fill x other-fill (see patch docstring); writer order kept, repairs as canonical_lenient
        from rl_verl_reward_lenient import canonical_lenient
        for i, r in enumerate(rows):
            rec0 = r.get("record")
            if not (r.get("ok") and isinstance(rec0, dict)):
                continue
            ok, canon, _ = canonical_lenient(rec0)
            if not ok:
                continue
            c = json.loads(canon)
            by = {e["finding"]: e for e in c["record"]}
            listed = []
            for e in rec0.get("record", []) if isinstance(rec0.get("record"), list) else []:
                f = e.get("finding") if isinstance(e, dict) else None
                if isinstance(f, str) and f in by and f not in listed and isinstance(e.get("present"), (bool, str)):
                    listed.append(f)
            omitted = [f for f in vr.FINDINGS if f not in listed]
            add = [f for f in omitted if (f == r["finding"]) == (args.policy == "fill_target")]
            new[i] = ({"record": [by[f] for f in listed] + [by[f] for f in add], "other": c.get("other", "")}, args.policy)
    elif args.policy == "fill_unreported":
        # 2026-09-27 exploratory: omissions made explicit as "not assessed" instead of absent (writer order kept)
        from rl_verl_reward_lenient import canonical_lenient
        for i, r in enumerate(rows):
            rec0 = r.get("record")
            if not (r.get("ok") and isinstance(rec0, dict)):
                continue
            ok, canon, _ = canonical_lenient(rec0)
            if not ok:
                continue
            c = json.loads(canon)
            by = {e["finding"]: e for e in c["record"]}
            listed = []
            for e in rec0.get("record", []) if isinstance(rec0.get("record"), list) else []:
                f = e.get("finding") if isinstance(e, dict) else None
                if isinstance(f, str) and f in by and f not in listed and isinstance(e.get("present"), (bool, str)):
                    listed.append(f)
            unrep = [{"finding": f, "present": "not assessed", "side": "not applicable", "box": None, "note": "not assessed"}
                     for f in vr.FINDINGS if f not in listed]
            new[i] = ({"record": [by[f] for f in listed] + unrep, "other": c.get("other", "")}, args.policy)
    elif args.policy == "flags_only":
        # 2026-09-26 payload factorial (replication protocol, secondary): each record in lenient canonical form
        # with only the present flags kept; side "not applicable", box null, note "" and "other" dropped, the
        # same entry format as oracle_labels, so the reader sees exactly what the flag reader sees
        from rl_verl_reward_lenient import canonical_lenient
        for i, r in enumerate(rows):
            if r.get("ok") and isinstance(r.get("record"), dict):
                ok, canon, _ = canonical_lenient(r["record"])
                if ok:
                    c = json.loads(canon)
                    c["record"] = [dict(e, side="not applicable", box=None, note="") for e in c["record"]]; c["other"] = ""
                    new[i] = (c, "flags_only")
    elif args.policy == "side_neutral":
        # 2026-09-26 dose-test audit: the rows' own records with every entry's side set to "not applicable"
        # (flags, boxes, notes unchanged), re-judged: isolates the laterality field's effect on verdicts
        for i, r in enumerate(rows):
            rec0 = r.get("record")
            if r.get("ok") and isinstance(rec0, dict) and isinstance(rec0.get("record"), list):
                rec1 = dict(rec0)
                rec1["record"] = [dict(e, side="not applicable") if isinstance(e, dict) else e for e in rec0["record"]]
                new[i] = (rec1, "side_neutral")
    elif args.policy == "canon_readj":
        from rl_verl_reward_lenient import canonical_lenient
        for i, r in enumerate(rows):
            if r.get("ok") and isinstance(r.get("record"), dict):
                ok, canon, _ = canonical_lenient(r["record"])
                if ok:
                    new[i] = (json.loads(canon), "canonical_lenient")
    elif args.policy == "readj":
        new = {i: (r["record"], "own") for i, r in enumerate(rows) if r.get("ok") and isinstance(r.get("record"), dict)}
    else:
        ok_idx = [i for i, r in enumerate(rows) if r.get("ok") and isinstance(r.get("record"), dict)]
        groups = collections.defaultdict(list)
        for i in ok_idx:
            key = (rows[i]["finding"], rows[i]["direction"], rows[i]["condition"]) if args.policy == "permute_cell" else "all"
            groups[key].append(i)
        singletons = 0
        for key in sorted(groups, key=str):
            g = groups[key]
            if len(g) < 2:
                singletons += len(g); continue
            perm = sattolo(len(g), rng)
            for a, b in zip(g, perm):
                donor = rows[g[b]]
                new[a] = (donor["record"], "%s|%s" % (donor["pair_id"], donor["condition"]))
        meta.update({"groups": len(groups), "singletons_dropped": singletons})

    pool = queue.Queue()
    for w in range(max(1, args.workers)):
        pool.put(LLM(model=args.model, base_url=args.base_url, run_dir=out / "_calls" / ("w%02d" % w), seed=args.seed))
    lock = threading.Lock(); fh = (out / "rows.jsonl").open("w", encoding="utf-8")
    state = {"done": 0, "ok": 0, "fail": 0}; t0 = time.time()

    def task(i):
        r = rows[i]
        rec = dict(r); rec["control"] = args.policy
        for k in ("verdict", "reason", "error"):
            rec.pop(k, None)
        if i not in new:
            rec.update({"ok": False, "error": "no control record (source row failed or singleton cell)", "record": None})
        else:
            record, source = new[i]
            rec["record"] = record; rec["record_source"] = source
            payload = json.dumps(record.get("record", record), separators=(",", ":")) if args.compact else json.dumps(record.get("record", record))
            llm = pool.get()
            try:
                text = llm.chat("adjudicator", sw.SYS_VERIFIER, _variant(_convention(sw.A_DEFAULT_RECORD.format(evidence=payload, claim=r["claim"]))), json_mode=True)
                d = parse_json(text)
                rec["verdict"] = str(d.get("verdict", "")).strip().lower(); rec["reason"] = d.get("reason", ""); rec["ok"] = True; _extra(rec, d)
            except Exception as exc:  # noqa: BLE001
                rec.update({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
            finally:
                pool.put(llm)
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["done"] += 1; state["ok" if rec.get("ok") else "fail"] += 1
            if state["done"] % 100 == 0:
                fh.flush(); print("ROWS %d/%d failures=%d elapsed=%.0fs" % (state["done"], len(rows), state["fail"], time.time() - t0), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(task, range(len(rows))))
    fh.close()
    (out / "control_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("META", json.dumps(meta))
    print("DONE rows=%d failures=%d elapsed=%.1fs out=%s" % (state["done"], state["fail"], time.time() - t0, out / "rows.jsonl"))
    sys.exit(0 if state["ok"] > 0 else 4)


if __name__ == "__main__":
    main()
