# Photograph study

[Back to the README](../README.md)

The study was prespecified and frozen under a hash manifest before any checklist or checker read of a study image.
Steps, from `$CLAIMBLIND_ROOT/domain2` after `setup_code_dir.sh`:

1. `python code/d2_build_pool.py 8 60 20` builds `pool_v8_cap60_n20/` from VQA v2 and the COCO instances.
2. `sbatch -p $PARTITION -q $QOS code/d2_screen.sbatch` runs the automated screen and writes `study/labels.json`, `study/screen.jsonl`, and
   `study/screen_pass.txt`.
3. `bash code/d2_freeze.sh` runs the self-tests and writes the manifest `study/FROZEN.sha256` and its digest.
4. `bash code/launch_study.sh DIGEST` runs both checklist models, then the two primary checkers, then the three
   secondary checkers (`code/d2_study.sbatch`, which verifies the manifest before every call).
5. `bash code/run_frozen_analysis.sh DIGEST` verifies every output and runs `d2_analyze.py` and `d2_report.py` once.

`photo_study/FROZEN_original.sha256` is the manifest the study froze (digest
`7053937df5257a29d97e9abc894f4d35e83f24b7fa2b24e485aac3becfdfdbea`, 24 files), with the cluster paths written as
`$CLAIMBLIND_ROOT` and `$RADOPEN_ROOT`. The released scripts differ from the frozen ones only in paths, site settings,
and the names of internal review steps in comments, so their hashes differ and a new run writes its own manifest.
`photo_study/protocol_registered.md` is the registered protocol with internal paths and review names replaced (its
first line gives the hash of the registered original). The engineering smoke, calibration, and convention-probe
scripts (`d2_smoke.sbatch`, `d2_make_smoke.py`, `d2_make_calib.py`, `d2_calib.sbatch`, `d2_conv_probe.py`,
`d2_conv_probe.sbatch`) are included because the frozen manifest lists them.
