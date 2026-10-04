#!/usr/bin/env python3
"""Build the verl GRPO dataset (parquet) for the claim-blind record (2026-09-24).

One row per clean training image (patient-disjoint from eval, dev, and the clean validation set):
  data_source   "claimblind_cxr"
  prompt        [{"role": "system", "content": SYS_OBSERVER}, {"role": "user", "content": "<image>" + U_RECORD_GATED}]
                (the claim is NOT in the prompt; verl splits the user content on <image>)
  images        [{"bytes": <png bytes of the 1024-px cached image>, "path": <study path>}]
  ability       "claimblind_record"
  reward_model  {"style": "rule", "ground_truth": json({"claims": [{"claim", "truth", "finding", "direction", "role"}]})}
  extra_info    {"path", "labels", "n_claims", "role"}
Also writes a small held-out parquet (the first N_VAL clean images by seeded shuffle, removed from
train) for verl's own periodic test reward, and a manifest with sha256 of both files.
Runs on CPU under the verl environment (pyarrow) after the harness image cache (runs/image_cache,
1024-px PNGs written by verify_run.resolve_image) holds every training image.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

RAD = Path(os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"))
sys.path.insert(0, str(RAD / "code"))
from radagent_open import sweep as sw  # noqa: E402

FINDINGS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture"]
CACHE = RAD / "runs" / "image_cache"


def cache_path(study_path):
    key = study_path.rsplit(".", 1)[0]
    return CACHE / (key.replace("/", "__") + "_s1024.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(os.environ.get("CLAIMBLIND_ROOT", "."), "results/train_slice/clean_train_images.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("CLAIMBLIND_ROOT", "."), "data/verl_v1"))
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.images, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    prompt_text = "<image>" + sw.U_RECORD_GATED.format(findings=", ".join(FINDINGS))
    recs = []; missing = 0
    for i, r in enumerate(rows):
        p = cache_path(r["path"])
        if not p.exists():
            missing += 1; continue
        claims = [{"claim": c["claim"], "truth": c["truth"], "finding": c["finding"], "direction": c["direction"], "role": c["role"]} for c in r["claims"]]
        recs.append({
            "data_source": "claimblind_cxr",
            "prompt": [{"role": "system", "content": sw.SYS_OBSERVER}, {"role": "user", "content": prompt_text}],
            "images": [{"bytes": p.read_bytes(), "path": r["path"]}],
            "ability": "claimblind_record",
            "reward_model": {"style": "rule", "ground_truth": json.dumps({"claims": claims})},
            "extra_info": {"path": r["path"], "labels": r["labels"], "n_claims": len(claims), "role": claims[0]["truth"] if claims else "none", "index": i},
        })
    val, train = recs[:args.n_val], recs[args.n_val:]
    pd.DataFrame(train).to_parquet(out / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(out / "val.parquet", index=False)
    man = {"train_rows": len(train), "val_rows": len(val), "missing_images": missing, "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
           "train_sha256": hashlib.sha256((out / "train.parquet").read_bytes()).hexdigest(), "val_sha256": hashlib.sha256((out / "val.parquet").read_bytes()).hexdigest(),
           "roles_train": {k: sum(1 for r in train if r["extra_info"]["role"] == k) for k in ("supported", "unsupported")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1)); print("PREP_DONE")
    sys.exit(0 if train else 4)


if __name__ == "__main__":
    main()
