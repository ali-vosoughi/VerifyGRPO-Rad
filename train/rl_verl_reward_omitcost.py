"""OMISSION-COST reward (registered before any run): the logged lenient reward of the
replication (rl_verl_reward_logged.py) with ONE change. The judge still reads the canonical record exactly as before;
the returned score is max(0, hits / claims - 0.25 * m / 12), where m is the number of the twelve findings the writer
did NOT list as a valid entry in its raw record (valid = a finding name among the twelve with a Boolean present or
the string "true" / "false", first occurrence; the same validity rule canonical_lenient applies). Each log line gains
"missing" (m) and "content_score" (hits / claims) next to the returned "score".
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_verl_reward as V  # noqa: E402
from rl_verl_reward_lenient import canonical_lenient  # noqa: E402
from rl_build_prefs_v1 import FINDINGS  # noqa: E402
from rl_verl_reward_logged import _log  # noqa: E402

PENALTY = 0.25


def listed_findings(rec):
    out = []
    entries = rec.get("record") if isinstance(rec, dict) else None
    for e in entries if isinstance(entries, list) else []:
        if not isinstance(e, dict):
            continue
        f, p = e.get("finding"), e.get("present")
        ok_p = isinstance(p, bool) or (isinstance(p, str) and p.strip().lower() in ("true", "false"))
        if isinstance(f, str) and f in FINDINGS and f not in out and ok_p:
            out.append(f)
    return out


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    rsha = hashlib.sha256((solution_str or "").encode()).hexdigest()[:16]
    ei = extra_info if isinstance(extra_info, dict) else {}
    ident = {"path": ei.get("path"), "index": ei.get("index")}
    try:
        rec = V._parse_json(solution_str)
    except Exception:  # noqa: BLE001
        _log(dict(ident, t=time.time(), resp_sha=rsha, verdict="unparseable", score=0.0))
        return 0.0
    ok, canon, _ = canonical_lenient(rec)
    if not ok:
        _log(dict(ident, t=time.time(), resp_sha=rsha, verdict="not_object", score=0.0))
        return 0.0
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    claims = gt.get("claims", [])
    if not claims:
        return 0.0
    payload = json.dumps(json.loads(canon)["record"], separators=(",", ":"))
    sha = hashlib.sha256(canon.encode()).hexdigest()
    hits = 0; lines = []
    for c in claims:
        v = V._verdict(sha, payload, c["claim"])
        h = int(v == c["truth"]); hits += h
        lines.append(dict(ident, payload_sha=hashlib.sha256(payload.encode()).hexdigest(), canon_sha=sha,
                          claim=c["claim"], truth=c["truth"], verdict=v, hit=h))
    content = hits / len(claims)
    missing = len(FINDINGS) - len(listed_findings(rec))
    score = max(0.0, content - PENALTY * missing / len(FINDINGS))
    for i, l in enumerate(lines):
        l.update({"t": time.time(), "resp_sha": rsha, "score": score, "content_score": content, "missing": missing})
        if i == 0:
            l["payload"] = payload
        _log(l)
    return score
