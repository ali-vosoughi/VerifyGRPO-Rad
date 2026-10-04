# Reproducing the results

[Back to the README](../README.md)

**1. Data.**

* Pairs: `harness/code/scripts/index_images.py`, then `harness/code/scripts/build_pairs.py` (seed 20260919), or rebuild
  the shipped identifiers into the files the scripts read as described in `pairs/README.md`.
* Validation freeze and audits: `data/freeze_val_clean.py`, `data/rl_leakage_audit.py`, `data/rl_check_disjoint.py`
  (also run inside `forms/rl_controls.sbatch`), `data/rl_locked_val_overlap.py`.
* Training parquet: `train/rl_verl_prep.py` writes `data/verl_v1/` (shuffle seed 20260924, 64 images held out for
  verl's internal validation). The 64-image probe and `data/dose_v1/` of the learning-rate test come from
  `eval/rl_probe_sample.sbatch` (selection mode) and `train/rl_probe_select.py`. `train/rl_rep_parquet.py 101 202 303`
  writes the per-seed data order `data/rep_s<seed>/`.

**2. Training (GRPO).** Every arm runs `train/rl_verl_grpo.sbatch` on three GPUs: the last serves the frozen training
checker with vLLM, the other two run verl, and every saved checkpoint is merged to a PEFT adapter at the end of the job
(`train/rl_merge_ckpt.sbatch` merges one checkpoint on its own).

* Replication (the main results): `train/launch_rep.sh`, a 2-step smoke run, then seeds 101, 202, and 303, 40 steps
  each, reward `train/rl_verl_reward_logged.py` (the content reward of `train/rl_verl_reward_lenient.py` plus a
  per-claim log), 8 images per step, 16 checklists per image at temperature 1.0, mini-batch 4, learning rate 1e-4,
  LoRA rank 16 and alpha 32, KL loss 0.01 (`low_var_kl`), dynamic sampling (`algorithm.filter_groups`).
* Omission cost: `train/launch_oc.sh`, the same recipe with `train/rl_verl_reward_omitcost.py`.
* Second checklist model: `train/launch_w2.sh` (Qwen2.5-VL-7B-Instruct, seed 101).
* Direct supervision: `baseline/launch_sft.sh`, which runs `baseline/rl_sft_direct.sbatch` and
  `baseline/rl_sft_direct.py`.
* Development runs: the first design with the strict-schema reward `train/rl_verl_reward.py` (the default
  `REWARD_PATH` of `train/rl_verl_grpo.sbatch`; `TRAIN_BS=16 MINI_BS=8 ROLL_N=8`, temperature 0.8, learning rate
  1e-5), `train/launch_v11_pilot.sh` (dynamic sampling), `train/launch_v12_pilot.sh` (content reward at learning rate
  1e-5), and the prespecified learning-rate test `train/launch_dose.sh`.

**3. Writing and checking the checklists.**

* One checkpoint: `eval/rl_eval_v2.sbatch` writes the checklists with the adapter (`eval/rl_eval_ckpt.py` around the
  harness's `verify_run.py`, protocol `sweep_gated_default`, sharded by `eval/rl_shard_pairs.py`), runs the claim-swap
  pass on the saved checklists (`eval/rl_swap_adjudicate.py`), and scores the image-necessary manifest
  (`eval/rl_score_directional.py`). The untrained read (`ADAPTER=none RUN_NO_IMAGE=1`) also runs the no-image arm that
  defines the image-necessary manifest. Validation reads use `PAIRS=results/val_clean/pairs_val_clean_in.jsonl
  PER_CELL=200 CELL_SKIP=0` (`eval/launch_rep_reads.sh`); held-out test reads use
  `PAIRS=results/locked/pairs_e6_2026-09-19_0405.jsonl PER_CELL=40 CELL_SKIP=0` (as in `eval/launch_oc_locked.sh`).
* Independent checker: `eval/rl_xadj.sbatch` rereads the saved checklists with MedGemma-4B-it.
* Rule-based check: `forms/rl_det_reader.py` (no language model).
* Checklist formats: `forms/rl_controls.sbatch` with `POLICIES=readj+canon_readj+order_only+fill_only+...` rewrites every
  saved checklist (as generated, training format, fixed order only, absences only, target or other absences, not
  assessed) and rereads it. Other policies give the metric controls: constant checklists (`all_absent`, `all_present`,
  `top5`), permutations (`permute_cell`, `permute_global`), and checklists written from the labels (`oracle_labels`).
  `forms/rl_controls_multi.sbatch` runs several checklist sets in one allocation (`SPECS=rows:outdir,...`).
  `forms/launch_rep_payload.sh` and `forms/launch_payload.sh` submit the payload rereads of the replication and of the
  development run.
* Further arms: `eval/launch_oc_reads.sh` (omission cost), `eval/launch_sft_reads.sh` (direct supervision),
  `eval/launch_w2_reads.sh` (second checklist model), `eval/launch_reads.sh` and `eval/merge_and_read.sh` (development
  runs).
* Sentence-first references: `eval/rl_claimfirst_ref.sbatch` (validation) and `eval/rl_claimfirst_ref_locked.sbatch`
  (held-out test).
* Further checkers: `forms/rl_controls_multi.sbatch` with `MODEL=` set to each checker (`MAXLEN` 16384 for
  Qwen3-VL-30B-A3B, 8192 for Phi-4-mini, 4096 for OLMo-2) and `POLICIES=readj+canon_readj+order_only+fill_only`, over
  the untrained and the three trained checklist sets of each population (`PAIRS_REL`, `MANIFEST_REL`, and
  `NO_IMAGE_REL` switch to the held-out test set).
* Stated convention: the same rereads with `JUDGE_CONVENTION="Convention for this record: any of the twelve findings
  that the record does not list is absent."`
* Checker size and instruction: `eval/launch_judgevar.sh size|g27|prompt`; `eval/launch_judgevar.sh one ...` submits a
  single named read (Gemma-3-4B: `one rl_jsz_g4_val gpu:a6000:1 google/gemma-3-4b-it 8192
  results/eval2/judgesize/g4_val val "" ""`, and the same for `locked`).

**4. Statistics.** Every interval comes from a bootstrap over connected groups of patients (2,000 draws, seed 20260925)
on the pairs common to every read involved; trained reads are averaged over the three seeds per pair before resampling
(`stats/rl_seedavg.py`). Run the scripts from `$CLAIMBLIND_ROOT` with config paths such as `code/cfg_rep_final.json`;
output directories are relative to `$CLAIMBLIND_ROOT`.

| paper quantity | script | configs |
|---|---|---|
| prespecified endpoints, validation and held-out test | `stats/rl_rep_endpoints.py`; exact fractions `stats/rl_exact_table1.py` | `cfg_rep_final.json`, `cfg_locked_final.json` |
| between-checker difference in the format effect (prespecified) and the format table | `stats/rl_payload_contrast_v2_rep.py` | `arms_rep_decomp_final.json`; per run `arms_s101.json`, `arms_s202.json` |
| order and absences split | `stats/rl_orderfill_contrast_rep.py` (validation), `stats/rl_orderfill_contrast_locked.py` (held-out test) | `arms_rep_orderfill_final.json`, `arms_locked_orderfill.json` |
| target and other absences | `stats/rl_filltarget_contrast.py` | `arms_rep_filltarget_final.json`, `arms_filltarget_s101.json` |
| not assessed instead of absent | `stats/rl_fillunrep_contrast.py`, `stats/rl_fillunrep_by_direction.py` | `arms_rep_fillunrep.json` |
| five checkers | `stats/rl_orderfill_contrast_locked.py` | `arms_j3_<checker>_<pop>.json` (written by `stats/make_j3_arms.py`) |
| stated convention | `stats/rl_orderfill_contrast_locked.py` | `arms_conv_<checker>_<pop>.json`, `arms_convpair_<pop>.json` (`stats/make_conv_arms.py`) |
| checker size and instruction | `stats/run_judgevar_analysis.sh` (runs `stats/make_judgevar_arms.py`, `stats/rl_orderfill_contrast_locked.py`, `stats/rl_cite_evidence.py`, `stats/summarize_judgevar.py`); `stats/rl_cite_crosstab.py` | `arms_jsz_*.json`, `arms_jv_*.json`, `arms_jvpair_*.json` |
| acceptance before and after unmentioned findings are entered as absent | `stats/rl_omission_transitions.py` | reads the rereads above |
| omission cost | `stats/rl_oc_contrast.py`; its endpoints with `stats/rl_rep_endpoints.py`; target absences with `stats/rl_filltarget_contrast.py` | `cfg_oc_contrast.json`, `cfg_locked_oc_contrast.json`, `cfg_oc_final.json`, `cfg_locked_oc_final.json`, `arms_oc_filltarget.json`; code test `cfg_oc_codetest.json` |
| between-reader contrasts, breadth | `stats/rl_reader_contrast.py`, `stats/rl_breadth.py` | `cfg_rep_final.json`, `cfg_locked_final.json` |
| image specificity and flag rates | `stats/rl_fig3_data.py`, `stats/rl_det_checks.py` | reads |
| direct supervision, second checklist model | `stats/rl_rep_endpoints.py`; `stats/rl_interface_contrast.py` | `cfg_sft_*.json`, `cfg_w2.json`, `arms_w2_interface.json` |
| repair audit of the training format | `stats/rl_repair_audit.py`, `stats/rl_repair_sensitivity.py` (with `stats/rl_orderfill_contrast_rep_manifest.py`) | `arms_rep_decomp_final.json`, `arms_rep_orderfill_final.json` |
| training curves, exposure, findings listed | `stats/rl_train_curves.py`, `stats/rl_exposure.py`, `stats/rl_listed_hist.py`, `stats/rl_oc_contrast.py` | verl logs, reward logs, reads |
| reference verdicts and metric controls | `eval/rl_claimfirst_ref.sbatch`, `forms/rl_controls.sbatch`, `forms/rl_controls_summary.py` | reads |
| development analyses (Supplementary Information) | `stats/rl_curve.py`, `stats/rl_paired_delta.py`, `stats/rl_validity_trend.py`, `eval/rl_group_variance.sbatch` with `stats/rl_passk.py`, `stats/rl_probe_eval.py`, `stats/rl_dose40_audit.py`, `stats/rl_det_by_group.py`, `stats/rl_payload_table.py`, `stats/rl_payload_contrast.py`, `stats/rl_payload_contrast_v2.py`, `stats/rl_orderfill_contrast.py`, `stats/rl_hybrid_contrast.py` | `arms_dose40.json`, `arms_orderfill_dose40.json`, `arms_filltarget_dose40.json`, `cfg_rep_s101_interim.json`, `cfg_rep_s202_interim.json`, `endpoints_codetest.json` |

## Compute

A 40-step training run took 4.2 to 5.6 hours on three 48 GB A6000 GPUs (12.7 to 16.9 A6000 GPU-hours). A checklist
read of the 238 validation pairs took about 0.3 A6000 GPU-hours, and rereading saved checklists by another checker or in
another format took minutes on one GPU (A6000, A100, or RTX 4090). Including pilots and every read up to 29 September
2026, the radiology study used 279.6 A6000, 12.3 A100, and 10.2 RTX 4090 GPU-hours on one cluster and 10.5 RTX PRO 6000
Blackwell and 4.1 H100 GPU-hours on a second cluster for development pilots, check runs, and data staging; the
photograph study used 4.4 A6000 GPU-hours and the rereads by checker size and instruction 16.3 A6000 GPU-hours.
