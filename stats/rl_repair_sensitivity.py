#!/usr/bin/env python3
"""Sensitivity of the registered form contrast to the canonicalizer's repairs (2026-09-29, after rl_repair_audit.py).

Reruns the frozen decomposition (code/rl_payload_contrast_v2_rep.py, arms spec of REPLICATION_FINAL_decomp) twice:
(1) unchanged population, which must reproduce the registered numbers exactly, and (2) the same population without
every pair in which any validation record of the untrained writer or of runs 1-3 needs a content repair (dropped
entry, dropped box, reset side, truncated note, ...). Post hoc robustness check; no model is run.
"""
import os
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from rl_repair_audit import CONTENT, READS, audit  # noqa: E402

OUT = ROOT / "results/analysis/repair_audit"


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    bad_pairs, bad_images = set(), set()
    for writer, rel in READS["validation"].items():
        rows = [json.loads(l) for l in open(ROOT / rel / "rows.jsonl") if l.strip()]
        rec_by_img = {}
        for r in rows:
            rec_by_img.setdefault(r["image_path"], r.get("record"))
        hit = {img for img, rec in rec_by_img.items() if any(audit(rec)[0][k] for k in CONTENT)}
        bad_images |= hit
        bad_pairs |= {r["pair_id"] for r in rows if r["image_path"] in hit}
    man = ROOT / spec.get("manifest", "results/val_clean/manifest_image_necessary.txt")
    pop = [l.strip() for l in open(man) if l.strip()]
    keep = [p for p in pop if p not in bad_pairs]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest_no_repair.txt").write_text("\n".join(keep) + "\n")
    s2 = dict(spec); s2["manifest"] = str((OUT / "manifest_no_repair.txt").relative_to(ROOT))
    (OUT / "arms_decomp_no_repair.json").write_text(json.dumps(s2, indent=1))
    print("repaired images %d, pairs touched %d of %d in the manifest, kept %d" % (
        len(bad_images), len([p for p in pop if p in bad_pairs]), len(pop), len(keep)))
    for tag, sp in (("rerun_all", spec_path), ("no_repair", OUT / "arms_decomp_no_repair.json")):
        subprocess.run([sys.executable, str(ROOT / "code/rl_payload_contrast_v2_rep.py"), str(sp),
                        str((OUT / tag).relative_to(ROOT))], check=True, cwd=ROOT)
        v = json.load(open(next((OUT / tag).glob("*.json"))))
        vals = v.get("values", v)
        for k in ("diff_Y_interface", "training_judge_Y_d_raw", "training_judge_Y_d_canonical",
                  "independent_judge_Y_d_raw", "independent_judge_Y_d_canonical"):
            if k in vals:
                print("%-10s %-32s %s" % (tag, k, vals[k]))
        if "n_pairs" in v or "n" in v:
            print(tag, "n", v.get("n_pairs", v.get("n")))


if __name__ == "__main__":
    main()
