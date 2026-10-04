#!/usr/bin/env python3
"""REPLICATION VERSION (seed-averaged via rl_seedavg.load_arm; otherwise identical). Payload DECOMPOSITION (2026-09-26 17:45; supersedes the notes contrast of rl_payload_contrast.py, which
compared RAW full records with CANONICAL ablated records and so mixed two changes). POST HOC on the discovery run;
DECLARED for the replication before any replicated record is read (project log 2026-09-26 17:45).

For each judge J, dY_J(arm) is the paired change from base to trained records under that arm. Arms: raw (the records
as the writer wrote them, the registered evaluation), canonical (the training serialization, every field kept),
notes_removed (canonical, notes and other emptied), flags_only (canonical, only the present flags). Contrasts:
  interface  X_J = dY_J(raw) - dY_J(canonical)
  notes      N_J = dY_J(canonical) - dY_J(notes_removed)
  rest       R_J = dY_J(notes_removed) - dY_J(flags_only)        (sides and boxes)
and the between-judge differences (independent minus training) of each, for Y and FS, on the pairs common to every
read, one patient-component bootstrap (2,000 draws, seed 20260925) recomputing everything on each draw.
Usage: rl_payload_contrast_v2.py ARMS.json OUTDIR
ARMS.json = {"training_judge": {"raw": [a_dir, a_proto, a_swap, b_dir, b_proto, b_swap], "canonical": [...], ...},
             "independent_judge": {...}}
"""
import os
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_paired_delta import per_pair, stats  # noqa: E402
from rl_payload_table import components  # noqa: E402
from rl_seedavg import load_arm  # noqa: E402  (seed-averaged reads; replication version, 2026-09-26 20:45)

JUDGES = ("training_judge", "independent_judge")
ARMS = ("raw", "canonical", "notes_removed", "flags_only")


def main():
    spec = json.load(open(sys.argv[1])); out = ROOT / sys.argv[2]; out.mkdir(parents=True, exist_ok=True)
    pop = [l.strip() for l in open(ROOT / spec.get("manifest", "results/val_clean/manifest_image_necessary.txt")) if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / spec.get("pairs", "results/val_clean/pairs_val_clean_in.jsonl")) if l.strip()}
    reads = {}
    for j in JUDGES:
        for a in ARMS:
            reads[(j, a)] = load_arm(spec[j][a])
    ids = [p for p in pop if all(p in A and p in B for A, B in reads.values())]
    units = components(ids, pairs)

    def q(sel):
        d = {(j, a, m): stats(B, sel)[m] - stats(A, sel)[m] for (j, a), (A, B) in reads.items() for m in ("Y", "FS")}
        o = {}
        for m in ("Y", "FS"):
            for j in JUDGES:
                for a in ARMS:
                    o["%s_%s_d_%s" % (j, m, a)] = d[(j, a, m)]
                o["%s_%s_interface" % (j, m)] = d[(j, "raw", m)] - d[(j, "canonical", m)]
                o["%s_%s_notes" % (j, m)] = d[(j, "canonical", m)] - d[(j, "notes_removed", m)]
                o["%s_%s_rest" % (j, m)] = d[(j, "notes_removed", m)] - d[(j, "flags_only", m)]
            for c in ("interface", "notes", "rest"):
                o["diff_%s_%s" % (m, c)] = o["independent_judge_%s_%s" % (m, c)] - o["training_judge_%s_%s" % (m, c)]
        return o

    point = q(ids)
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in q(sel).items():
            bo[k].append(v)
    res = {"pairs": len(ids), "units": len(units), "values": {}}
    lines = ["payload decomposition, %d pairs, %d components" % (len(ids), len(units))]
    for k, v0 in point.items():
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        res["values"][k] = {"point": round(v0, 4), "lo95": round(lo, 4), "hi95": round(hi, 4)}
        lines.append("  %-40s %+.4f [%+.4f, %+.4f]" % (k, v0, lo, hi))
    (out / "payload_decomposition.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "payload_decomposition.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
