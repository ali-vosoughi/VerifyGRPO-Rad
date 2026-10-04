# Unmentioned Checklist Findings Change How Reinforcement Learning Appears to Improve Chest Radiograph Report Checking

Code, configurations, prompts, and pair identifiers for the paper of the same title.

A model that checks radiology reports can be trained with reinforcement learning, but how much it appears to improve
depends on how its output is read. In this study a vision-language model sees a chest radiograph, never the report,
and fills in a checklist of 12 findings. A second, frozen language model then reads only the checklist and decides
whether a report sentence is supported. The checklist model is trained with group relative policy optimization (GRPO)
on matched pairs of CheXpert Plus studies, in which the same sentence agrees with the labels of one image and not the
other, and every saved checklist is evaluated by the checker used in training, by an independent medical checker, and
by a rule-based check. Findings that a checklist leaves unmentioned turn out to matter: entering them as absent, as the
training reward did, moves the two checkers' measured gains in opposite directions. This repository holds the code
that wrote and scored the checklists, trained the model, ran every checker and checklist format, and computed every
analysis in the paper, together with the prompts and the pair identifiers.

**Status:** submitted to npj Digital Medicine, 2026; preprint link to be added.

## Contents

| folder | what it holds |
|---|---|
| `train/` | GRPO launchers and the verl job (`rl_verl_grpo.sbatch`), the four reward functions and the strict record schema (`rl_build_prefs_v1.py`), the training parquet builders, the 64-image probe of the learning-rate test, checkpoint merging, a unit test of the reward's repairs |
| `eval/` | checklist writing and checking of one checkpoint (`rl_eval_v2.sbatch`), the claim-swap pass, the directional scorer, the independent checker (`rl_xadj.sbatch`), the sentence-first references, read launchers for every training arm, the checker-size and instruction rereads, the probe sampler and the reward-variance measurement of the development runs |
| `forms/` | the checklist formats and controls (`rl_control_records.py`, run by `rl_controls.sbatch` and `rl_controls_multi.sbatch`), the rule-based check (`rl_det_reader.py`), launchers of the format rereads |
| `stats/` | every analysis: endpoints, paired contrasts, the format and order-by-fill decompositions, the further checkers, the stated convention, checker size and instruction, acceptance with unmentioned findings entered as absent, omission cost, breadth, exposure, training curves, the repair audit, and the development analyses |
| `data/` | the validation freeze, leakage audits, the train and evaluation disjointness assertion, the patient-overlap check |
| `baseline/` | the direct-supervision baseline and its launcher |
| `configs/` | the JSON configs and arms files the analysis scripts take as arguments |
| `harness/` | the part of the radagent-open evaluation harness these scripts import: the language-model client and prompts (`harness/code/radagent_open/`), the row builder `verify_run.py`, the legacy scorer, the vLLM helper, the image indexer, and the pair builder (`harness/code/scripts/`) |
| `photo_study/` | the prespecified photograph study: pool builder, automated screen, checklist writers, checkers, checklist formats, the frozen analysis and its report, the freeze script, the launch and analysis runners, the pinned model revisions, the registered protocol, and the hash manifest of the freeze |
| `figures/` | builders of Figures 1 to 3 and Supplementary Figures S1 to S4 (they read analysis summaries only), and the rule-based pickers of the example pair and checklists |
| `pairs/` | pair and image identifiers only (see `pairs/README.md`) |

`setup_code_dir.sh` copies the scripts into the flat working layout they expect. `MANIFEST.tsv` lists every file with
its source in the study's code, the sha256 of that source, and the sha256 of the released copy.

Terms: the code keeps the names used while the study was developed. The **checklist model** is the *writer*, a **checklist** is a
*record*, a **checker** is a *judge* (the **training checker** is Qwen3-VL-8B-Instruct, the **independent checker** is
MedGemma-4B-it), the **rule-based check** is the *deterministic flag reader*, the **training format** is the
*canonical* or *reward-scoring* form (`canon_readj`), a checklist **as generated** is the *raw* form (`readj`), a
**false alarm** is a *false strike* (FS), the **held-out test set** is the *locked* set, and the photograph study is
*domain 2* (`d2_*`).

Prompts as run: the checklist model's messages are `SYS_OBSERVER` and `U_RECORD_GATED` (with `RECORD_JSON`) and the
checker's are `SYS_VERIFIER` and `A_DEFAULT_RECORD`, all in `harness/code/radagent_open/sweep.py` (protocol
`sweep_gated_default`); the sentence-first references are protocols of `harness/code/scripts/verify_run.py`; the
instruction variants are `_REPLY_VARIANT` in `forms/rl_control_records.py` and `eval/rl_swap_adjudicate.py`; the
photograph study's messages are `WRITER_SYSTEM`, `WRITER_USER`, `JUDGE_SYSTEM`, and `judge_messages` in
`photo_study/d2_forms.py`.

## Data access

Nothing from any dataset is redistributed here. Bring your own copies.

* **CheXpert Plus** (images, reports, and labels), from the Stanford Center for Artificial Intelligence in Medicine and
  Imaging under its research use agreement, dataset DOI 10.71718/6nvz-pm34. `pairs/` holds pair ids, CheXpert Plus
  relative image paths, finding names, sentence directions, split names, and sentence indices, with no report text and
  no label values. `pairs/README.md` explains how to rebuild the sentences and the label files the scripts read from
  your copy of the dataset.
* **COCO 2014** images and instance annotations (`annotations_trainval2014.zip`), and **VQA v2** questions, answers,
  and complementary pairs, for the photograph study. `pairs/photo_pool_pairs.jsonl` holds only the pair id, split,
  category, the two COCO image ids, and the two VQA question ids of each of the 516 pairs; `pairs/photo_images.txt`
  lists the 1,032 COCO image paths and `pairs/photo_screen_pass.txt` the 493 pairs that passed the automated screen.
  `photo_study/d2_build_pool.py 8 60 20` rebuilds the full pool from your copies; the sha256 of the pool the study used
  is in `photo_study/FROZEN_original.sha256`.

## Environment

Linux with Slurm and NVIDIA GPUs. Two Python 3.11.16 environments, pinned in the requirements files:

* `requirements-verl.txt` (`VERL_VENV`): training with verl 0.9.1, vLLM 0.24.0, PyTorch 2.11.0, Transformers 5.9.0,
  PEFT 0.21.0, Ray 2.58.0; also the parquet builders, the checkpoint merger, the direct-supervision baseline, and the
  CPU statistics.
* `requirements-harness.txt` (`HARNESS_VENV`): checklist writing and every checker read through a local
  OpenAI-compatible vLLM 0.29.0 server (PyTorch 2.13.0, Transformers 5.17.0, openai 3.19.0). The client refuses
  non-local endpoints.
* `requirements-figures.txt`: matplotlib, NumPy, and Pillow for the figure builders.

The versions were read from the environments that ran the study.

## Setup

```bash
export CLAIMBLIND_ROOT=/abs/path/workdir        # will hold code/, data/, results/, runs/, logs/, domain2/
bash setup_code_dir.sh                          # copies every script and config into $CLAIMBLIND_ROOT/code/
export RADOPEN_ROOT=/abs/path/to/repo/harness   # required: the harness and its data (below)
export VERL_VENV=/path/to/verl-env HARNESS_VENV=/path/to/harness-env
export PARTITION=<partition> QOS=<qos>          # used by the launchers
cd $CLAIMBLIND_ROOT                             # submit every job from here (logs go to logs/%x_%j.out)
```

`RADOPEN_ROOT` must be set. The study's scripts defaulted to the harness path on the cluster where they were
developed; the released scripts have no such default and stop with a message when it is unset (`HARNESS_ROOT` is
accepted as another name for it). Besides `code/`, the harness directory must hold `datasets/chexpert_plus/` (the
dataset files, see `pairs/README.md`), `results/e6_pairs/` (the pair build and the `image_index.tsv` written by
`harness/code/scripts/index_images.py`), `runs/image_cache/` (1024-pixel PNGs written on first use), and `hf_cache/`
(used as `HF_HOME`; every job runs offline). Set both roots to absolute paths. Relative paths inside the configs
(`results/...`, `data/...`, `runs/...`) resolve under `$CLAIMBLIND_ROOT`. Cluster settings (partition, QOS, account)
are passed on the sbatch command line; the `--gres` lines name the GPU types the runs used.

| variable | meaning |
|---|---|
| `CLAIMBLIND_ROOT` | working directory for code, data, results, runs, logs (default `.`; set it to an absolute path) |
| `RADOPEN_ROOT` | harness directory, required (`HARNESS_ROOT` is accepted as another name) |
| `VERL_VENV`, `HARNESS_VENV` | the two Python environments |
| `PARTITION`, `QOS` | Slurm partition and QOS used by the launchers |
| `VQA_ROOT` | photograph study: directory with the VQA v2 JSON files and the COCO `train2014/` and `val2014/` images; the COCO instance annotations go in `$CLAIMBLIND_ROOT/domain2/data/annotations/` |
| `FIG_ROOT` | figure builders: directory with `artifacts/` (inputs) and `figs/` (outputs) |
| `ADJ_BASE_URL`, `ADJ_MODEL` | set by `train/rl_verl_grpo.sbatch`: the local vLLM server of the training checker |
| `WRITER_MODEL` | set by the read jobs: `record` routes the checklist model to the served adapter, `BASE` to the base model |
| `REWARD_LOG_DIR` | per-claim reward log directory (logged and omission-cost rewards) |
| `JUDGE_CONVENTION` | optional sentence appended to the checker message (the stated-convention rereads); unset leaves the message unchanged |
| `JUDGE_PROMPT_VARIANT` | optional `cite` or `reason` reply line (the checker-instruction rereads); unset leaves the message unchanged |
| `VLLM_EXTRA` | extra vLLM server arguments for a reread (for example `--tensor-parallel-size 2`) |

## Models

Every job loads models by Hugging Face id from an offline cache that held one snapshot per model, at these revisions:

| model | role | revision |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | checklist model (with LoRA) and training checker | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| `google/medgemma-4b-it` | independent checker | `290cda5eeccbee130f987c4ad74a59ae6f196408` |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | further checker | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| `microsoft/Phi-4-mini-instruct` | further checker | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `allenai/OLMo-2-1124-7B-Instruct` | further checker | `470b1fba1ae01581f270116362ee4aa1b97f4c84` |
| `google/gemma-3-4b-it` | checker size series | `093f9f388b31de276ce2de164bdc2081324b9767` |
| `google/gemma-3-12b-it` | checker size series; photograph study checker | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| `google/gemma-3-27b-it` | checker size series | `005ad3404e59d6023443cb575daa05336842228a` |
| `Qwen/Qwen2.5-VL-7B-Instruct` | second checklist model; photograph study checklist model | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `HuggingFaceM4/Idefics3-8B-Llama3` | photograph study screen | `fddb4ff79181e55a994674777e06cd5456ce3dc3` |

Download each into `$RADOPEN_ROOT/hf_cache` at its revision, for example
`HF_HOME=$RADOPEN_ROOT/hf_cache huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
The Gemma and MedGemma models require accepting their terms on Hugging Face. The photograph study passes the pinned
revisions to vLLM itself (`photo_study/model_revisions.txt`).

The trained adapters are not released. For reference, the sha256 of `adapter_model.safetensors` at step 40 of the three
replication runs: seed 101 `d1d4116404d9b9f42ee0ee96358d6cdc8082b4b2c6e28efbe81395d6f7afe634`, seed 202
`eebf8f3ce37f91768b349349452ae30226d300ae83dbf76c13a12015496a2909`, seed 303
`f7bbf6840b6afb8eb0d5c57ef07044e1e54fa880a68947f5671d34b482cd6d09`.

## Reproducing the results, in order

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

## Photograph study

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

## Figures

The builders read analysis outputs copied under `$FIG_ROOT/artifacts/` and write `$FIG_ROOT/figs/`:

| input under `artifacts/` | written by |
|---|---|
| `replication/REPLICATION_FINAL_decomp_payload_decomposition.json` | `stats/rl_payload_contrast_v2_rep.py` with `arms_rep_decomp_final.json` (its `payload_decomposition.json`) |
| `locked_forms/LOCKED_orderfill_POSTHOC.json` | `stats/rl_orderfill_contrast_locked.py` with `arms_locked_orderfill.json` (its `orderfill.json`) |
| `replication/REPLICATION_FINAL_endpoints.json`, `replication/LOCKED_FINAL_endpoints.json` | `stats/rl_rep_endpoints.py` (its `endpoints.json`) |
| `fig3/fig3_data_replication.json` | `stats/rl_fig3_data.py` |
| `conv/<checker>_<pop>/orderfill.json` | `stats/rl_orderfill_contrast_locked.py` with the convention arms |
| `train_curves/train_curves.json` | `stats/rl_train_curves.py` |
| `listed_hist/listed_hist.json` | `stats/rl_listed_hist.py` |
| `flip_example/flip_pick.json`, `flip_example/flip_true.png` | `figures/rl_flip_pick.py` |
| `qual_examples/qual_examples.json` | `figures/rl_qual_examples.py` |
| `pair_example/pair.json`, `report.png`, `partner.png` | `figures/rl_teaser_pick.py` |

`figures/build_fig1_npj.py` (Figure 1), `figures/build_fig2_npj.py` (Figure 2), `figures/build_fig3_npj.py` (Figure 3),
and `figures/build_supp_figs.py` (Supplementary Figures S1 to S4). Figures 1 and 2 show CheXpert Plus radiographs and a
report sentence, so their inputs exist only where the dataset is available; the pickers write them from your copy.

## Compute

A 40-step training run took 4.2 to 5.6 hours on three 48 GB A6000 GPUs (12.7 to 16.9 A6000 GPU-hours). A checklist
read of the 238 validation pairs took about 0.3 A6000 GPU-hours, and rereading saved checklists by another checker or in
another format took minutes on one GPU (A6000, A100, or RTX 4090). Including pilots and every read up to 29 September
2026, the radiology study used 279.6 A6000, 12.3 A100, and 10.2 RTX 4090 GPU-hours on one cluster and 10.5 RTX PRO 6000
Blackwell and 4.1 H100 GPU-hours on a second cluster for development pilots, check runs, and data staging; the
photograph study used 4.4 A6000 GPU-hours and the rereads by checker size and instruction 16.3 A6000 GPU-hours.

## Not included

* Results files, saved checklists, verdicts, reward logs, rollouts, checkpoints, and adapters, including the
  photograph study's saved checklists, verdicts, screen outputs, and frozen analysis output.
* Any CheXpert Plus image, report text, or label; any COCO image or annotation; any VQA question or answer. The derived
  label files are rebuilt from your copies (`pairs/README.md`; `photo_study/d2_screen.py` for `labels.json`).
* Site-specific helpers of the study: scheduler accounting, model download and environment build scripts, one-off
  scripts that patched or generated files whose final form is released here, and superseded backups.

## Citation

```bibtex
@misc{vosoughi2026unmentioned,
  title  = {Unmentioned Checklist Findings Change How Reinforcement Learning Appears to Improve Chest Radiograph Report Checking},
  author = {Vosoughi, Ali and Kasturi, Akhil and Xu, Chenliang and Wismueller, Axel},
  year   = {2026},
  note   = {Submitted to npj Digital Medicine}
}
```

## License

MIT, see `LICENSE`. Datasets and model weights keep their own licenses and terms of use.
