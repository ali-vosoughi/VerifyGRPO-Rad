#!/usr/bin/env python3
"""Dose test: the FROZEN decision metrics (2026-09-25; follow-up 1 Q2-Q3 + implementation review item 9).

For each probe measurement (rl_probe_sample.py --mode measure on data/dose_v1/probe.parquet, K16/T=1):
  probe_acc   mean over the two truth strata of the mean per-image fraction of rollouts judged correct
              (reward 1.0); unparseable / non-object rollouts count as INCORRECT, never dropped
  flag_acc    same weighting, fraction of rollouts whose lenient-canonical target flag equals the label;
              rollouts without a Boolean flag count as INCORRECT
Against the base measurement: delta and a 95 percent paired bootstrap over the 64 probe images (2,000 draws).
MOVEMENT = delta probe_acc >= +0.10 AND delta flag_acc > 0.
Extension choice at step 20 (arms given as name=dir): among arms with delta flag_acc > 0, the larger delta
probe_acc; tie -> the lower LR arm (3e-5); if no arm improves flag_acc, the larger delta probe_acc with a
REWARD-ONLY warning (the protocol then routes to a reader-exploitation audit, not a claim).
Also: retained exposures of each probe image in an arm's rollout dump (trainer.rollout_data_dir) and generated
exposures in its reward log; adapter displacement ||(alpha/r) B A||_F summed over LoRA modules.
"""
import os
import argparse
import glob
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))


def load_measure(d):
    gs = [json.loads(l) for f in sorted(Path(d).glob("groups_*.jsonl")) for l in open(f) if l.strip()]
    bad = [g for g in gs if not g.get("ok") or len(g.get("rollouts", [])) != 16]
    if len(gs) != 64 or bad:
        raise SystemExit("INCOMPLETE_MEASUREMENT %s groups=%d bad=%d" % (d, len(gs), len(bad)))
    return {g["path"]: g for g in gs}


def per_image(g):
    r = g["rollouts"]
    acc = sum(1.0 for x in r if x["reward"] >= 1.0) / len(r)
    fl = sum(1.0 for x in r if x.get("flag_correct") == 1) / len(r)
    return acc, fl


def balanced(m, paths):
    by = defaultdict(list)
    for p in paths:
        a, f = per_image(m[p]); by[m[p]["truth"]].append((a, f))
    acc = sum(sum(x[0] for x in v) / len(v) for v in by.values()) / len(by)
    fl = sum(sum(x[1] for x in v) / len(v) for v in by.values()) / len(by)
    return acc, fl


def delta_ci(base, cur, paths, boots=2000):
    rng = random.Random(20260925)
    strata = defaultdict(list)
    for p in paths:
        strata[base[p]["truth"]].append(p)
    out = {"acc": [], "flag": []}
    for _ in range(boots):
        sel = [s[rng.randrange(len(s))] for s in strata.values() for _ in s]   # stratified resample keeps 32 / 32
        a0, f0 = balanced(base, sel); a1, f1 = balanced(cur, sel)
        out["acc"].append(a1 - a0); out["flag"].append(f1 - f0)
    ci = {}
    for k, v in out.items():
        v.sort(); ci[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1])
    return ci


def displacement(adapter_dir):
    try:
        from safetensors.torch import load_file   # bf16 adapters: numpy cannot read bfloat16
    except Exception:  # noqa: BLE001
        return None
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scale = cfg.get("lora_alpha", 32) / cfg.get("r", 16)
    w = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    tot = 0.0
    for k, A in w.items():
        if "lora_A" in k:
            Bk = k.replace("lora_A", "lora_B")
            if Bk in w:
                d = scale * (w[Bk].float() @ A.float())
                tot += float((d ** 2).sum())
    return math.sqrt(tot)


def exposures(exp, probe):
    claim2path = {}
    for p in probe:
        claim2path[(p["claim"], p["truth"])] = p["path"]
    retained, generated = Counter(), Counter()
    for f in glob.glob(str(ROOT / "runs/rollouts" / exp / "*.jsonl")):
        for l in open(f):
            try:
                d = json.loads(l)
            except Exception:  # noqa: BLE001
                continue
            gt = d.get("gts") or d.get("ground_truth") or (d.get("reward_model") or {}).get("ground_truth")
            if isinstance(gt, str):
                try:
                    gt = json.loads(gt)
                except Exception:  # noqa: BLE001
                    gt = None
            if isinstance(gt, dict):
                for c in gt.get("claims", []):
                    k = (c.get("claim"), c.get("truth"))
                    if k in claim2path:
                        retained[claim2path[k]] += 1
    pset = {p["path"] for p in probe}
    for f in glob.glob(str(ROOT / "runs/reward_logs" / exp / "*.jsonl")):
        for l in open(f):
            try:
                d = json.loads(l)
            except Exception:  # noqa: BLE001
                continue
            if d.get("path") in pset and ("payload" in d or d.get("verdict") in ("unparseable", "not_object")):
                generated[d["path"]] += 1
    return retained, generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="measurement dir of the base writer")
    ap.add_argument("--arm", action="append", default=[], help="name=measure_dir[=exp[=adapter_dir]]")
    ap.add_argument("--manifest", default=str(ROOT / "data/dose_v1/probe_manifest.json"))
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    base = load_measure(args.base)
    paths = [p["path"] for p in man["probe"]]
    if set(paths) != set(base):
        raise SystemExit("PROBE_MISMATCH base")
    import pyarrow.parquet as pq
    probe_rows = pq.read_table(ROOT / "data/dose_v1/probe.parquet").to_pylist()
    probe = [{"path": r["extra_info"]["path"], "claim": json.loads(r["reward_model"]["ground_truth"])["claims"][0]["claim"],
              "truth": json.loads(r["reward_model"]["ground_truth"])["claims"][0]["truth"]} for r in probe_rows]
    a0, f0 = balanced(base, paths)
    print("BASE probe_acc %.4f flag_acc %.4f" % (a0, f0))
    res = []
    for spec in args.arm:
        parts = spec.split("=")
        name, mdir = parts[0], parts[1]
        cur = load_measure(mdir)
        if set(paths) != set(cur):
            raise SystemExit("PROBE_MISMATCH %s" % name)
        a1, f1 = balanced(cur, paths)
        ci = delta_ci(base, cur, paths)
        mv = (a1 - a0 >= 0.10) and (f1 - f0 > 0)
        row = {"arm": name, "probe_acc": a1, "flag_acc": f1, "d_acc": a1 - a0, "d_acc_ci": ci["acc"], "d_flag": f1 - f0,
               "d_flag_ci": ci["flag"], "MOVEMENT": mv}
        if len(parts) > 2:
            ret, gen = exposures(parts[2], probe)
            row["probe_retained_images"] = sum(1 for p in paths if ret[p] > 0)
            row["probe_retained_rollouts"] = sum(ret.values())
            row["probe_generated_images"] = sum(1 for p in paths if gen[p] > 0)
        if len(parts) > 3:
            row["adapter_displacement"] = displacement(parts[3])
        res.append(row)
        print("%-18s probe_acc %.4f (d %+.4f [%+.4f, %+.4f])  flag_acc %.4f (d %+.4f [%+.4f, %+.4f])  MOVEMENT=%s  %s" % (
            name, a1, a1 - a0, ci["acc"][0], ci["acc"][1], f1, f1 - f0, ci["flag"][0], ci["flag"][1], mv,
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k.startswith("probe_") or k == "adapter_displacement"}))
    if len(res) > 1:
        flagged = [r for r in res if r["d_flag"] > 0]
        pool = flagged or res
        best = max(pool, key=lambda r: (round(r["d_acc"], 9), "3e-5" in r["arm"]))
        print("EXTEND %s%s" % (best["arm"], "" if flagged else "  (REWARD-ONLY: no arm improved flag accuracy -> audit)"))
    Path(ROOT / "results/probe").mkdir(parents=True, exist_ok=True)
    print(json.dumps({"base": {"probe_acc": a0, "flag_acc": f0}, "arms": res}, indent=None)[:4000])


if __name__ == "__main__":
    main()
