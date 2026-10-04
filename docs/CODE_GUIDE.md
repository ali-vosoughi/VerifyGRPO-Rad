# Code guide

[Back to the README](../README.md)

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

`setup_code_dir.sh` copies the scripts into the flat working layout they expect. `docs/MANIFEST.tsv` lists every file with
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
