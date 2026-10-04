#!/usr/bin/env python3
"""Seed averaging for the declared secondary analyses (written before any replicated
decomposition was computed; the extension the declaration allows: average each pair's G, B and FS over seeds
before the bootstrap, same arms and contrasts).

load_side(spec): spec is either one read [dir, proto, swap] or a list of reads [[dir, proto, swap], ...]; returns the
per-pair dict of rl_paired_delta.per_pair, averaged over the reads on the pairs present in every read.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_paired_delta import per_pair  # noqa: E402


def load_side(spec):
    if spec and isinstance(spec[0], (list, tuple)):
        reads = [per_pair(*s) for s in spec]
        common = set(reads[0]).intersection(*reads[1:])
        out = {}
        for p in common:
            out[p] = {k: sum(r[p][k] for r in reads) / len(reads) for k in ("G", "B", "FS")}
            out[p]["pres_n"] = sum(r[p]["pres_n"] for r in reads); out[p]["pres_flip"] = sum(r[p]["pres_flip"] for r in reads)
        return out
    return per_pair(*spec)


def load_arm(spec):
    """Old format: [a_dir, a_proto, a_swap, b_dir, b_proto, b_swap]; new format: {"a": read, "b": read or [reads]}."""
    if isinstance(spec, dict):
        return load_side(spec["a"]), load_side(spec["b"])
    return per_pair(*spec[:3]), per_pair(*spec[3:])
