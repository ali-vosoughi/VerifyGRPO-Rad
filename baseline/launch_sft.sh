#!/bin/bash
# Direct-supervision baseline launcher (2026-09-26). Smoke (target mode, 16 examples, same one-GPU layout)
# gates three arms: sft_full_all (all 4501 training images, full labels), sft_full_matched and sft_target_matched
# (the 1331 images the discovery RL run TRAINED on: the distinct paths of its per-claim reward log, minus the 64
# in-training validation images that the same log also scores; correction logged 2026-09-26 14:30).
set -euo pipefail
ROOT=${CLAIMBLIND_ROOT:-.}
cd $ROOT
mkdir -p results/sft runs/sft
M0=results/sft/matched_paths_dose_v1_lr1e-4.txt
M=results/sft/matched_train_paths_dose_v1_lr1e-4.txt
if [ ! -s $M0 ]; then
  cat runs/reward_logs/dose_v1_lr1e-4/*.jsonl | python3 -c "
import sys, json
p = set()
for l in sys.stdin:
    try: p.add(json.loads(l)['path'])
    except Exception: pass
p.discard(None)
print('\n'.join(sorted(p)))" > $M0
fi
if [ ! -s $M ]; then
  python3 -c "
import pyarrow.parquet as pq
tr = {e['path'] for e in pq.read_table('data/dose_v1/train.parquet', columns=['extra_info']).column('extra_info').to_pylist()}
print('\n'.join(l.strip() for l in open('$M0') if l.strip() in tr))" > $M
fi
wc -l $M0 $M; sha256sum $M0 $M
S=code/rl_sft_direct.sbatch
SM=$(sbatch --parsable -J rl_sft_smoke --export=ALL,MODE=target,ARM=smoke_target,PATHS=$M,SMOKE=1 $S)
echo "smoke $SM"
sbatch --dependency=afterok:$SM --kill-on-invalid-dep=yes -J rl_sft_full_all --export=ALL,MODE=full,ARM=sft_full_all,PATHS= $S
sbatch --dependency=afterok:$SM --kill-on-invalid-dep=yes -J rl_sft_full_m --export=ALL,MODE=full,ARM=sft_full_matched,PATHS=$M $S
sbatch --dependency=afterok:$SM --kill-on-invalid-dep=yes -J rl_sft_target_m --export=ALL,MODE=target,ARM=sft_target_matched,PATHS=$M $S
