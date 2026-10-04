#!/usr/bin/env python3
"""the training curve from checkpoint reads (2026-09-25). Figure 1's data, not the figure.

Collects results/eval2/<exp>_step_<N>/directional/directional.json for one run, puts the step-0 point
first (results/eval2/controls_v1/step0_in: same 238-pair manifest AND the same swap donors as the
checkpoint reads), and writes OUT/curve_<exp>.csv with every metric and its 95 percent pair-bootstrap
interval per step, plus the control floors (permute_global I, constant-policy C_pres) as reference rows.
Also prints the pre-registered reading per design v1 item 5:
  crossing candidate  d = I - C_pres with its lower bound above zero
  sustained           three consecutive saved checkpoints with lower bound above zero, and the last one
  stronger statement  G up and Y lower bound above zero with B not above step 0's upper bound
  FS guard            FS not above step 0's FS + 0.05 (point; the paired one-sided test comes later)
The bootstrap here is per checkpoint, not simultaneous; the simultaneous band for the GO gate is
computed once all seeds are read (never claim GO from this table alone).
"""
import os
import argparse
import csv
import json
import re
from pathlib import Path

KEYS = ("G", "B", "Y", "FS", "REC", "I", "C_pres", "C_flip_correct", "d", "Y_minus_Cpres")


def load(p):
    j = json.loads(Path(p).read_text())
    return j["point"], j["ci95"], j["scored"], j.get("manifest_sha256")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("CLAIMBLIND_ROOT", "."))
    ap.add_argument("--exp", required=True, help="verl experiment name, e.g. v1_paironly_seed2")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ev = Path(args.root) / "results" / "eval2"
    out = Path(args.out) if args.out else ev / "curves"; out.mkdir(parents=True, exist_ok=True)
    pts = []
    s0 = ev / "controls_v1" / "step0_in" / "directional" / "directional.json"
    if not s0.exists():
        raise SystemExit("NO_STEP0 %s (controls_v1 must finish first)" % s0)
    pts.append((0, *load(s0)))
    for d in ev.glob("%s_step_*/directional/directional.json" % args.exp):
        m = re.search(r"_step_(\d+)$", d.parent.parent.name)
        if m:
            pts.append((int(m.group(1)), *load(d)))
    pts.sort(key=lambda x: x[0])
    shas = {p[4] for p in pts}
    if len(shas) != 1:
        raise SystemExit("MANIFEST_MISMATCH %s" % shas)
    floors = {}
    for pol in ("permute_global", "permute_cell", "all_absent", "all_present", "top5"):
        f = ev / "controls_v1" / pol / "directional" / "directional.json"
        if f.exists():
            floors[pol] = load(f)
    rows = []
    for step, pt, ci, n, _ in pts:
        r = {"step": step, "scored": n}
        for k in KEYS:
            r[k] = pt.get(k)
            lo, hi = ci.get(k, [None, None])
            r[k + "_lo"], r[k + "_hi"] = lo, hi
        rows.append(r)
    for pol, (pt, ci, n, _) in floors.items():
        r = {"step": "control:" + pol, "scored": n}
        for k in KEYS:
            r[k] = pt.get(k)
            lo, hi = ci.get(k, [None, None])
            r[k + "_lo"], r[k + "_hi"] = lo, hi
        rows.append(r)
    cols = ["step", "scored"] + [c for k in KEYS for c in (k, k + "_lo", k + "_hi")]
    f = out / ("curve_%s.csv" % args.exp)
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    base = rows[0]
    fl = floors.get("permute_global")
    floor_I = fl[0]["I"] if fl else None
    print("%6s %5s %7s %7s %7s %7s %7s %7s %18s" % ("step", "n", "G", "B", "Y", "FS", "I", "C_pres", "d [lo, hi]"))
    run = 0; sustained_at = None
    for r in rows:
        if not isinstance(r["step"], int):
            continue
        above = r["d_lo"] is not None and r["d_lo"] > 0
        run = run + 1 if above else 0
        if run >= 3 and sustained_at is None:
            sustained_at = r["step"]
        print("%6s %5s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f [%.3f, %.3f]%s" % (
            r["step"], r["scored"], r["G"], r["B"], r["Y"], r["FS"], r["I"], r["C_pres"] or 0, r["d"] or 0,
            r["d_lo"] or 0, r["d_hi"] or 0, "  <- d lower bound > 0" if above else ""))
    last = [r for r in rows if isinstance(r["step"], int)][-1]
    print("CONTROL_FLOOR permute_global I=%s (a trained I at or below this is record noise, not image use)" % floor_I)
    print("SUSTAINED_CROSSING %s (three consecutive saved checkpoints; endpoint above: %s)" % (
        sustained_at, last["d_lo"] is not None and last["d_lo"] > 0))
    # pre-registered FS guard and Y target = CLAIM-FIRST on the same population (fixed 2026-09-25; the first
    # version compared against step 0); claim_first_cot is printed as the harder comparator
    refs = {}
    for proto in ("claim_first", "claim_first_cot"):
        rf = ev / "claimfirst_ref_v1" / proto / "directional" / "directional.json"
        if rf.exists():
            refs[proto] = load(rf)[0]
    if "claim_first" not in refs:
        raise SystemExit("NO_CLAIM_FIRST_REFERENCE (run code/rl_claimfirst_ref.sbatch)")
    cf = refs["claim_first"]
    guard = cf["FS"] + 0.05
    for r in rows:
        if isinstance(r["step"], int):
            r["gate_point"] = (r["Y"] > cf["Y"]) and (r["FS"] <= guard)
    print("REFERENCE claim_first Y %.3f FS %.3f (guard FS <= %.3f)%s" % (cf["Y"], cf["FS"], guard,
          "; claim_first_cot Y %.3f FS %.3f" % (refs["claim_first_cot"]["Y"], refs["claim_first_cot"]["FS"]) if "claim_first_cot" in refs else ""))
    print("GATE_POINT (Y above claim-first at FS within guard, point estimates only) at steps: %s" % (
        [r["step"] for r in rows if isinstance(r["step"], int) and r.get("gate_point")] or "none"))
    print("STRONGER last: G %.3f vs step0 %.3f; Y_lo %.3f > 0: %s; B %.3f <= step0 B_hi %.3f: %s; FS %.3f <= %.3f: %s" % (
        last["G"], base["G"], last["Y_lo"] or 0, (last["Y_lo"] or 0) > 0, last["B"], base["B_hi"], last["B"] <= base["B_hi"],
        last["FS"], guard, last["FS"] <= guard))
    print("CURVE", f)


if __name__ == "__main__":
    main()
