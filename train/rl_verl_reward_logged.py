"""Dose test reward (2026-09-25): the v1.2 lenient reward, UNCHANGED in value, plus a log line per
judged claim so the exact judged payload and per-claim verdict of every training rollout are kept (red team
round 2 follow-up 1, question 5). Log: $REWARD_LOG_DIR/reward_<pid>.jsonl, one JSON object per claim:
{"t", "path", "index", "payload_sha" (exact judged bytes), "canon_sha" (canonical object), "payload",
"claim", "truth", "verdict", "hit", "score", "resp_sha"}. Unparseable text
logs one line with verdict "unparseable". Logging failures never change the returned reward.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_verl_reward as V  # noqa: E402
from rl_verl_reward_lenient import canonical_lenient  # noqa: E402

_LOG = None


def _log(obj):
    global _LOG
    d = os.environ.get("REWARD_LOG_DIR")
    if not d:
        return
    try:
        if _LOG is None:
            Path(d).mkdir(parents=True, exist_ok=True)
            _LOG = open(Path(d) / ("reward_%d.jsonl" % os.getpid()), "a", encoding="utf-8")
        _LOG.write(json.dumps(obj, ensure_ascii=False) + "\n"); _LOG.flush()
    except Exception:  # noqa: BLE001
        pass


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
    score = hits / len(claims)
    for i, l in enumerate(lines):
        l.update({"t": time.time(), "resp_sha": rsha, "score": score})
        if i == 0:
            l["payload"] = payload
        _log(l)
    return score
