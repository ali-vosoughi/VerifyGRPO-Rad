"""verl reward v1.2 candidate, LENIENT canonical record (2026-09-25).

Why: the v1 reward scores any record that fails the strict schema as 0, and the step-0 / step-35 group
decomposition (results/gvar) shows GRPO spending its gradient on that: invalid rollouts 8.4 -> 0.4 percent,
validity-only groups resolved to all-correct, content groups 18 -> 14, stuck-wrong images unchanged, and a
flat evaluation curve because greedy records were already ~99 percent valid. Here format carries no reward:
the record is REPAIRED to the canonical form without inventing content, then judged exactly as in v1.

Repair rules (content-preserving): top-level keys other than record / other are ignored; entries that are
not objects or name a finding outside the twelve (e.g. Support Devices, No Finding) are ignored; a
duplicated finding keeps its FIRST entry; extra entry keys are ignored; present must be a Boolean or the
string "true" / "false" (any other value drops the entry, i.e. absent, never present); side outside the four
values -> "not applicable"; a malformed box -> null; a non-string note -> "" and notes are cut at 300
characters; an empty or missing record list -> every finding absent. Unlisted findings are absent (as in v1).
A record that passes the strict schema gets byte-identical canonical output, so its reward is unchanged.
Only text with no JSON object scores 0. Same adjudicator call, cache, and claim handling as v1 (imported).
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_verl_reward as V  # noqa: E402
from rl_build_prefs_v1 import FINDINGS, SIDES  # noqa: E402


def canonical_lenient(rec):
    """Return (ok, canonical_json_string, n_repairs). ok is False only when rec is not an object."""
    if not isinstance(rec, dict):
        return False, None, 0
    repairs = 0
    entries = rec.get("record")
    if not isinstance(entries, list):
        entries = []; repairs += 1
    seen = {}
    for e in entries:
        if not isinstance(e, dict):
            repairs += 1; continue
        f = e.get("finding")
        if not isinstance(f, str) or f not in FINDINGS or f in seen:   # 2026-09-26 type guard (unhashable values)
            repairs += 1; continue
        p = e.get("present")
        if isinstance(p, bool):
            present = p
        elif isinstance(p, str) and p.strip().lower() in ("true", "false"):
            present = p.strip().lower() == "true"; repairs += 1
        else:
            repairs += 1; continue
        side = e.get("side")
        if not isinstance(side, str) or side not in SIDES:   # 2026-09-26 type guard (a list side crashed the w2 smoke)
            side = "not applicable"; repairs += 1
        b = e.get("box")
        if b is not None and not (isinstance(b, list) and len(b) == 4 and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in b)):
            b = None; repairs += 1
        note = e.get("note", "")
        if not isinstance(note, str):
            note = ""; repairs += 1
        if len(note) > 300:
            note = note[:300]; repairs += 1
        seen[f] = {"finding": f, "present": present, "side": side, "box": None if b is None else [float(x) for x in b], "note": note}
    other = rec.get("other", "")
    if not isinstance(other, str):
        other = ""; repairs += 1
    for f in FINDINGS:
        if f not in seen:
            seen[f] = {"finding": f, "present": False, "side": "not applicable", "box": None, "note": ""}
    canon = {"record": [seen[f] for f in FINDINGS], "other": other[:300]}
    return True, json.dumps(canon, ensure_ascii=False, separators=(",", ":")), repairs


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        rec = V._parse_json(solution_str)
    except Exception:  # noqa: BLE001
        return 0.0
    ok, canon, _ = canonical_lenient(rec)
    if not ok:
        return 0.0
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    claims = gt.get("claims", [])
    if not claims:
        return 0.0
    payload = json.dumps(json.loads(canon)["record"], separators=(",", ":"))
    sha = hashlib.sha256(canon.encode()).hexdigest()
    hits = sum(int(V._verdict(sha, payload, c["claim"]) == c["truth"]) for c in claims)
    return hits / len(claims)
