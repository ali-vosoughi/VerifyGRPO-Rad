#!/usr/bin/env python3
"""Payload CONTRASTS (2026-09-26; POST HOC, requested by the outside figure review after the payload point
estimates were seen, and labelled as post hoc wherever reported).

Question: do the two judges' readings of the same notes differ beyond chance? For each judge J and payload arm A,
dY_J(A) = Y(trained records under A) - Y(base records under A) on the same pairs. The NOTES EFFECT of judge J is
N_J = dY_J(full) - dY_J(notes_removed): how much including the notes changes the training gain J registers; the
FLAGS EFFECT F_J = dY_J(full) - dY_J(flags_only). The between-judge contrasts are N_ind - N_train and F_ind - F_train.
All on the pairs common to every read involved, one patient-component bootstrap (2,000 draws, seed 20260925) in
which every quantity is recomputed on the same resample, so intervals are for the contrasts themselves.
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
from rl_payload_table import DEFAULT_ARMS, components  # noqa: E402

JUDGES = ("training_judge", "independent_judge")
ARMS = ("full", "notes_removed", "flags_only", "sides_neutral")


def main():
    out = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/analysis/payload_dose40")
    pop = [l.strip() for l in open(ROOT / "results/val_clean/manifest_image_necessary.txt") if l.strip()]
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "results/val_clean/pairs_val_clean_in.jsonl") if l.strip()}
    reads = {}
    for j in JUDGES:
        for a in ARMS:
            a_dir, a_proto, a_swap, b_dir, b_proto, b_swap = DEFAULT_ARMS[(j, a)]
            reads[(j, a)] = (per_pair(a_dir, a_proto, a_swap), per_pair(b_dir, b_proto, b_swap))
    ids = [p for p in pop if all(p in A and p in B for A, B in reads.values())]
    units = components(ids, pairs)

    def dy(sel):
        return {k: stats(B, sel)["Y"] - stats(A, sel)["Y"] for k, (A, B) in reads.items()}

    def contrasts(d):
        c = {}
        for j in JUDGES:
            c["N_" + j] = d[(j, "full")] - d[(j, "notes_removed")]
            c["F_" + j] = d[(j, "full")] - d[(j, "flags_only")]
            c["S_" + j] = d[(j, "full")] - d[(j, "sides_neutral")]
        c["N_ind_minus_train"] = c["N_independent_judge"] - c["N_training_judge"]
        c["F_ind_minus_train"] = c["F_independent_judge"] - c["F_training_judge"]
        return c

    point = contrasts(dy(ids))
    rng = random.Random(20260925); bo = defaultdict(list)
    for _ in range(2000):
        sel = [i for _ in units for i in units[rng.randrange(len(units))]]
        for k, v in contrasts(dy(sel)).items():
            bo[k].append(v)
    res = {"pairs": len(ids), "units": len(units), "post_hoc": True, "contrasts": {}}
    print("pairs common to all reads: %d, patient components %d (POST HOC contrasts)" % (len(ids), len(units)))
    for k in point:
        v = sorted(bo[k]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        p_le0 = sum(x <= 0 for x in v) / len(v); p_ge0 = sum(x >= 0 for x in v) / len(v)
        res["contrasts"][k] = {"point": round(point[k], 4), "lo": round(lo, 4), "hi": round(hi, 4),
                               "boot_frac_le0": round(p_le0, 4), "boot_frac_ge0": round(p_ge0, 4)}
        print("  %-22s %+.4f  [%+.4f, %+.4f]  frac<=0 %.3f  frac>=0 %.3f" % (k, point[k], lo, hi, p_le0, p_ge0))
    d0 = dy(ids)
    res["dY_common_pairs"] = {"%s|%s" % k: round(v, 4) for k, v in d0.items()}
    (out / "payload_contrasts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("WROTE", out / "payload_contrasts.json")


if __name__ == "__main__":
    main()
