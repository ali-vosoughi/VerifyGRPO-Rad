"""OpenAI-compatible chat client for a local vLLM server, with per-call disk logging.

Every call appends one JSON line to <run_dir>/calls.jsonl: role, model, messages (images as
sha256 + path, never the bytes), response text, latency, token usage, error. This is the audit
trail the paper's reproducibility gate reads.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}


def _assert_local(base_url: str) -> None:
    host = urlparse(base_url).hostname or ""
    if host not in _LOCAL_HOSTS and not host.startswith("10.") and not host.endswith(".local"):
        raise RuntimeError(f"refusing non-local endpoint {base_url}: no hosted APIs in recorded runs")


def _image_part(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    p = Path(path)
    raw = p.read_bytes()
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    part = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    meta = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()}
    return part, meta


@dataclass
class LLM:
    model: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    run_dir: Path = field(default_factory=lambda: Path("runs/_unnamed"))
    temperature: float = 0.0
    seed: int = 20260918
    timeout_s: int = 600
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        _assert_local(self.base_url)
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)
        self._log = self.run_dir / "calls.jsonl"

    def chat(self, role: str, system: str, user: str, images: list[str | Path] | None = None,
             json_mode: bool = False, max_tokens: int | None = None) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        image_meta: list[dict[str, str]] = []
        for img in images or []:
            part, meta = _image_part(img)
            content.append(part)
            image_meta.append(meta)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, temperature=self.temperature,
                                      seed=self.seed, max_tokens=max_tokens or self.max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.time()
        text, err, usage = "", None, None
        try:
            resp = self._client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            usage = resp.usage.model_dump() if resp.usage else None
        except Exception as e:  # logged, then re-raised so the pipeline records a failure
            err = f"{type(e).__name__}: {e}"
        rec = {"ts": time.time(), "role": role, "model": self.model, "latency_s": round(time.time() - t0, 3),
               "system": system, "user": user, "images": image_meta, "json_mode": json_mode,
               "response": text, "usage": usage, "error": err, "seed": self.seed, "temperature": self.temperature}
        with self._log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if err:
            raise RuntimeError(err)
        return text


def parse_json(text: str) -> Any:
    """Tolerant JSON extraction: strips code fences and leading prose."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(s[start:end + 1])
