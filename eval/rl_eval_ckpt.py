#!/usr/bin/env python3
"""S3: read one checkpoint with the harness instrument, unchanged (2026-09-23).

Thin wrapper over harness/code/scripts/verify_run.py. The only change is WHICH served model
answers each role: observer calls (the record writer) go to the LoRA adapter served by vLLM under
the name given in WRITER_MODEL, every other role (adjudicator, verifier, no-image verifier) goes to
the frozen base model in --model. The harness, its prompts, row builder, scorer, and per-call
logging are imported, never copied or edited. Usage is verify_run.py's, plus the environment
variable WRITER_MODEL (default: the base model, which reproduces the harness's prompted arm exactly and
is the step-0 point of the curve).
"""
import os
import sys
from pathlib import Path

RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(RAD / "code"))
sys.path.insert(0, str(RAD / "code" / "scripts"))
import radagent_open.llm as _llm  # noqa: E402
import verify_run as vr  # noqa: E402

WRITER_MODEL = os.environ.get("WRITER_MODEL")
if WRITER_MODEL is None:
    sys.exit("WRITER_MODEL must be set explicitly: the served LoRA name for a checkpoint read, or the literal "
             "string BASE for the prompted step-0 writer (fail-closed after the 2026-09-24 review)")
if WRITER_MODEL == "BASE":
    WRITER_MODEL = ""


class RoleLLM(_llm.LLM):
    """Routes the observer role to the writer (adapter) model, everything else to the base."""

    def chat(self, role, *a, **k):
        prev = self.model
        if WRITER_MODEL and role == "observer":
            self.model = WRITER_MODEL
        try:
            return super().chat(role, *a, **k)
        finally:
            self.model = prev


vr.LLM = RoleLLM
print("ROLE_ROUTING writer=%s base=%s" % (WRITER_MODEL or "(base)", "--model"), flush=True)
vr.main()
