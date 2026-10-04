"""Per-step training curves from the verl stdout logs (2026-09-28). Read-only; writes one small JSON.

Runs: registered replication (rep_s101/202/303) and the omission-cost arm (oc_s101/202/303).
Keys kept: reward mean/min/max, response length, entropy, KL loss, grad norm, filtered groups, step time,
plus any validation keys (val-*) if present.
"""
import json, re, glob, os, hashlib

ROOT = os.environ.get("CLAIMBLIND_ROOT", ".")
# verl stdout logs are logs/<job name>_<Slurm job id>.out (#SBATCH -o in rl_verl_grpo.sbatch); set the job ids of your runs
RUNS = {
    "rep_s101": "logs/rl_rep_s101_<JOBID>.out", "rep_s202": "logs/rl_rep_s202_<JOBID>.out",
    "rep_s303": "logs/rl_rep_s303_<JOBID>.out", "oc_s101": "logs/rl_oc_s101_<JOBID>.out",
    "oc_s202": "logs/rl_oc_s202_<JOBID>.out", "oc_s303": "logs/rl_oc_s303_<JOBID>.out",
}
KEEP = ["critic/score/mean", "critic/score/min", "critic/score/max", "response_length/mean",
        "response_length/clip_ratio", "actor/entropy", "actor/kl_loss", "actor/grad_norm",
        "actor/pg_clipfrac", "training/filter_groups/evicted_samples",
        "training/filter_groups/discarded_surplus_samples", "timing_s/step", "timing_s/gen"]
pat = re.compile(r"([A-Za-z_@/\-0-9]+):(-?[0-9.]+(?:e-?[0-9]+)?)")
out = {}
for run, rel in RUNS.items():
    p = os.path.join(ROOT, rel)
    raw = open(p, "rb").read()
    txt = raw.decode("utf-8", "replace")
    steps = {}
    for line in txt.splitlines():
        m = re.search(r"\bstep:(\d+) - ", line)
        if not m:
            continue
        d = dict(pat.findall(line[m.start():]))
        s = int(m.group(1))
        row = {k: float(d[k]) for k in KEEP if k in d}
        row.update({k: float(v) for k, v in d.items() if k.startswith("val")})
        steps[s] = row
    out[run] = {"log": rel, "sha256_16": hashlib.sha256(raw).hexdigest()[:16],
                "n_steps": len(steps), "steps": [dict(step=s, **steps[s]) for s in sorted(steps)]}
    print(run, len(steps), "first", steps[min(steps)].get("critic/score/mean"),
          "last", steps[max(steps)].get("critic/score/mean"),
          "val keys", sorted({k for r in steps.values() for k in r if k.startswith("val")})[:6])
os.makedirs(os.path.join(ROOT, "results/train_curves"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/train_curves/train_curves.json"), "w"), indent=1)
print("wrote results/train_curves/train_curves.json")
