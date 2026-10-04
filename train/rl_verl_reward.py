"""verl custom reward function for the claim-blind record (2026-09-24, design v1).

Reward per rollout = pair term only: the canonical record (strict schema; unlisted finding = absent)
is judged by the frozen base adjudicator, served by vLLM at ADJ_BASE_URL, against every claim
attached to the image; +1 per claim whose verdict matches the truth for THIS image, averaged.
A rollout that fails the canonical schema scores 0 (hard eligibility, not a bonus). No label term,
no positivity penalty. The claim never reaches the policy: it lives only in ground_truth, which the
reward function reads after the rollout is complete.

verl calls compute_score(data_source, solution_str, ground_truth, extra_info) per rollout; with
reward_model.reward_manager=prime the calls run in a process pool, so this module keeps no shared
state except an on-disk-free per-process LRU cache keyed by (record_sha256, claim).
Environment: ADJ_BASE_URL (http://127.0.0.1:PORT/v1), ADJ_MODEL (base model id), RADOPEN_ROOT.
"""
import functools
import hashlib
import json
import os
import sys
from pathlib import Path

RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(RAD / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from radagent_open import sweep as sw  # noqa: E402
from rl_build_prefs_v1 import canonical  # noqa: E402

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=os.environ["ADJ_BASE_URL"], api_key="EMPTY", timeout=120)
    return _client


def _parse_json(text):
    s = text.strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("no json")
    return json.loads(s[a:b + 1])


@functools.lru_cache(maxsize=200000)
def _verdict(record_sha, payload, claim):
    client = _get_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=os.environ.get("ADJ_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
                messages=[{"role": "system", "content": sw.SYS_VERIFIER},
                          {"role": "user", "content": sw.A_DEFAULT_RECORD.format(evidence=payload, claim=claim)}],
                temperature=0.0, seed=20260919, max_tokens=256, response_format={"type": "json_object"})
            d = _parse_json(resp.choices[0].message.content or "")
            v = str(d.get("verdict", "")).strip().lower()
            if v.startswith("supported"):
                return "supported"
            if v.startswith("unsupported"):
                return "unsupported"
            return "other"
        except Exception:  # noqa: BLE001
            if attempt == 2:
                return "error"
    return "error"


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        rec = _parse_json(solution_str)
    except Exception:  # noqa: BLE001
        return 0.0
    ok, canon, _ = canonical(rec)
    if not ok:
        return 0.0
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    claims = gt.get("claims", [])
    if not claims:
        return 0.0
    payload = json.dumps(json.loads(canon)["record"], separators=(",", ":"))
    sha = hashlib.sha256(canon.encode()).hexdigest()
    hits = 0
    for c in claims:
        hits += int(_verdict(sha, payload, c["claim"]) == c["truth"])
    return hits / len(claims)


def compute_score_flags_only(data_source, solution_str, ground_truth, extra_info=None):
    """Ablation: adjudicate a flags-only payload (finding + present), no side, box, or note."""
    try:
        rec = _parse_json(solution_str)
    except Exception:  # noqa: BLE001
        return 0.0
    ok, canon, _ = canonical(rec)
    if not ok:
        return 0.0
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    claims = gt.get("claims", [])
    if not claims:
        return 0.0
    payload = json.dumps([{"finding": e["finding"], "present": e["present"]} for e in json.loads(canon)["record"]], separators=(",", ":"))
    sha = hashlib.sha256(payload.encode()).hexdigest()
    return sum(int(_verdict(sha, payload, c["claim"]) == c["truth"]) for c in claims) / len(claims)
