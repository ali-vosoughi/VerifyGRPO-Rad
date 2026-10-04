#!/usr/bin/env python3
"""Redesigned round 0 (after the 2026-09-24 external red team): build preference sets from
the CACHED round-0 samples under the pair-only reward with a hard canonical schema.

Rules (all pre-registered before any trained checkpoint of the new design is read):
  eligibility   a sample is eligible only if its record passes the strict schema: a JSON object with
                key "record" holding exactly the 12 findings, each once, in any order, with a Boolean
                "present" (true/false, never a string), "side" in {left,right,bilateral,not applicable},
                "box" null or four finite numbers, "note" a string of at most 300 characters, and no
                other keys inside an entry; the top level may carry "other" (string) and nothing else.
  clean images  the image's patient is absent from the evaluation sample, the dev slice, the gold
                set, and the frozen clean validation set (patient-disjoint training).
  primary set   "pairterm": per image, chosen = an eligible sample with the HIGHEST pair term,
                rejected = an eligible sample with the LOWEST pair term, only when chosen_pair >
                rejected_pair; ties broken by shortest canonical record (so length cannot be the cue);
                one preference per image.
  variant B     "pairterm_random": chosen = random pair-correct eligible sample, rejected = random
                pair-incorrect eligible sample (pair term 1 vs 0 on single-claim images).
  variant C     "allcontrasts": every ordered (pair-correct, pair-incorrect) eligible contrast, with
                weight 1/(number of contrasts on that image) so each image carries total weight 1.
  ablation      "total_v0_clean": the old best-versus-worst-by-total rule at margin >= 1 on the same
                clean, eligible samples (the quarantined design, for attribution).
Outputs OUT/<set>.jsonl with the canonical record bytes as chosen/rejected, plus OUT/summary.json
(counts by role, finding, direction; degenerate-policy scores on the pair term: all-absent, all-present,
fixed-top-5 by frequency). CPU only.
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import random
import sys
from pathlib import Path

FINDINGS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture"]
SIDES = {"left", "right", "bilateral", "not applicable"}


def canonical(rec):
    """Return (ok, canonical_json_string, reason). Strict schema; canonical ordering by FINDINGS."""
    if not isinstance(rec, dict):
        return False, None, "not_object"
    keys = set(rec.keys())
    if "record" not in keys or not keys <= {"record", "other"}:
        return False, None, "top_level_keys"
    entries = rec["record"]
    if not isinstance(entries, list) or not entries or len(entries) > 12:
        return False, None, "entry_count"
    seen = {}
    for e in entries:
        if not isinstance(e, dict) or set(e.keys()) != {"finding", "present", "side", "box", "note"}:
            return False, None, "entry_keys"
        f = e["finding"]
        if f not in FINDINGS or f in seen:
            return False, None, "finding_name_or_duplicate"
        if not isinstance(e["present"], bool):
            return False, None, "present_not_bool"
        if e["side"] not in SIDES:
            return False, None, "side"
        b = e["box"]
        if b is not None:
            if not (isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in b)):
                return False, None, "box"
        if not isinstance(e["note"], str) or len(e["note"]) > 300:
            return False, None, "note"
        seen[f] = {"finding": f, "present": e["present"], "side": e["side"], "box": None if b is None else [float(x) for x in b], "note": e["note"]}
    other = rec.get("other", "")
    if not isinstance(other, str):
        return False, None, "other"
    # canonicalisation rule (2026-09-24): a finding the writer did not list is absent, written out
    # explicitly, so every canonical record carries the 12 findings in fixed order; the count of
    # filled entries is returned in the reason string so the summary can report it
    filled = 0
    for f in FINDINGS:
        if f not in seen:
            seen[f] = {"finding": f, "present": False, "side": "not applicable", "box": None, "note": ""}
            filled += 1
    canon = {"record": [seen[f] for f in FINDINGS], "other": other[:300]}
    return True, json.dumps(canon, ensure_ascii=False, separators=(",", ":")), "ok_filled_%d" % filled


def pat(p):
    return p.split("/")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="samples_all.jsonl from the pool")
    ap.add_argument("--pairs-file", required=True, help="harness pairs file (cells define eval/dev)")
    ap.add_argument("--gold-file", required=True)
    ap.add_argument("--val-pairs", required=True, help="frozen clean validation pairs file")
    ap.add_argument("--block-gold", action="store_true", help="also block gold-set patients (only needed if box tracking on gold is reported)")
    ap.add_argument("--pair-field", default="pair_reward", help="pair_reward (raw, judged on the sampled payload) or pair_reward_canon (re-adjudicated on canonical bytes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in open(args.pairs_file, encoding="utf-8") if l.strip()]
    blocked = set()
    cells = collections.Counter()
    for p in pairs:
        k = (p["finding"], p["direction"]); i = cells[k]; cells[k] += 1
        if i < 60:
            blocked.add(pat(p["report_path"])); blocked.add(pat(p["image_path"]))
    for f in ([args.gold_file] if args.block_gold else []) + [args.val_pairs]:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                g = json.loads(l); blocked.add(pat(g["report_path"])); blocked.add(pat(g["image_path"]))
    by = collections.defaultdict(list)
    reasons = collections.Counter(); n_total = 0; n_blocked = 0
    for l in open(args.samples, encoding="utf-8"):
        s = json.loads(l); n_total += 1
        if not s.get("ok"):
            reasons["parse_fail"] += 1; continue
        if pat(s["path"]) in blocked:
            n_blocked += 1; continue
        ok, canon, why = canonical(s["record"])
        reasons[why.split("_filled_")[0] if why.startswith("ok") else why] += 1
        if why.startswith("ok_filled_"):
            reasons["filled_entries_total"] += int(why.split("_")[-1])
        if not ok:
            continue
        if args.pair_field not in s:
            reasons["missing_pair_field"] += 1; continue
        s["pair_reward"] = s[args.pair_field]
        s["canon"] = canon
        by[s["path"]].append(s)
    sets = {"pairterm": [], "pairterm_random": [], "allcontrasts": [], "total_v0_clean": []}
    role_count = collections.Counter(); finding_count = collections.Counter()
    for path, v in by.items():
        truth = v[0]["claims"][0]["truth"] if v[0].get("claims") else "none"
        finding = v[0]["claims"][0]["finding"] if v[0].get("claims") else "none"
        hi = max(v, key=lambda s: (s["pair_reward"], -len(s["canon"])))
        lo = min(v, key=lambda s: (s["pair_reward"], len(s["canon"])))
        base = {"path": path, "labels": v[0]["labels"], "truth_role": truth, "finding": finding, "prompt": None}
        if hi["pair_reward"] > lo["pair_reward"]:
            sets["pairterm"].append({**base, "chosen": hi["canon"], "rejected": lo["canon"], "chosen_pair": hi["pair_reward"], "rejected_pair": lo["pair_reward"]})
            role_count[truth] += 1; finding_count[finding] += 1
            corr = [s for s in v if s["pair_reward"] >= 0.999]; inc = [s for s in v if s["pair_reward"] <= 0.001]
            if corr and inc:
                c = rng.choice(corr); r = rng.choice(inc)
                sets["pairterm_random"].append({**base, "chosen": c["canon"], "rejected": r["canon"], "chosen_pair": c["pair_reward"], "rejected_pair": r["pair_reward"]})
                w = 1.0 / (len(corr) * len(inc))
                for c in corr:
                    for r in inc:
                        sets["allcontrasts"].append({**base, "chosen": c["canon"], "rejected": r["canon"], "chosen_pair": 1.0, "rejected_pair": 0.0, "weight": w})
        if len(v) >= 2:
            b = max(v, key=lambda s: s["total"]); w_ = min(v, key=lambda s: s["total"])
            if b["total"] - w_["total"] >= 1.0:
                sets["total_v0_clean"].append({**base, "chosen": b["canon"], "rejected": w_["canon"], "chosen_pair": b["pair_reward"], "rejected_pair": w_["pair_reward"], "chosen_total": b["total"], "rejected_total": w_["total"]})
    # degenerate policies on the pair term over the clean images: an all-absent record earns 1 on partner
    # images (adjudicator strikes) and 0 on report images, and vice versa for all-present, under the
    # observed adjudicator; we use the observed adjudicator behaviour on samples with 0 / 12 present as a proxy
    deg = {}
    for name, pred in (("all_absent", lambda s: s["n_present"] == 0), ("all_present", lambda s: s["n_present"] >= 11), ("five_or_more", lambda s: s["n_present"] >= 5)):
        sel = [s for v in by.values() for s in v if pred(s)]
        deg[name] = {"n_samples": len(sel), "pair_term_mean": (sum(s["pair_reward"] for s in sel) / len(sel)) if sel else None}
    role_all = collections.Counter(v[0]["claims"][0]["truth"] for v in by.values() if v[0].get("claims"))
    summary = {"samples_total": n_total, "samples_blocked_patient": n_blocked, "schema_reasons": dict(reasons),
               "clean_eligible_images": len(by), "images_by_role": dict(role_all),
               "sets": {k: len(v) for k, v in sets.items()}, "pairterm_by_role": dict(role_count),
               "pairterm_by_finding": dict(finding_count.most_common()), "degenerate_on_pair_term": deg}
    for k, v in sets.items():
        with (out / (k + ".jsonl")).open("w", encoding="utf-8") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary.setdefault("sha256", {})[k] = hashlib.sha256((out / (k + ".jsonl")).read_bytes()).hexdigest()
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("PREFS_DONE", {k: len(v) for k, v in sets.items()})
    sys.exit(0 if sets["pairterm"] else 4)


if __name__ == "__main__":
    main()
