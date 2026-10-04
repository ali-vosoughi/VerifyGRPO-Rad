# Pair and image definitions (identifiers only)

These files define every population the paper reads, without redistributing CheXpert Plus content.
They carry study paths, pair identifiers, finding names, directions, split names, and sentence
indices. They carry no report text (no `target_sentence`, `report_findings`, or impression) and no
label values; the `context_vector` field of the original pair files (the label vector of the other
eleven findings, not read by any script here) is also omitted. Access to the images, reports, and
labels is governed by the CheXpert Plus research use agreement; obtain the dataset from its
distributor before rebuilding anything below.

## Files

| file | rows | what it is | rebuild into |
|---|---|---|---|
| `validation_pairs.jsonl` | 238 | validation population = the image-necessary manifest, in file order | `results/val_clean/pairs_val_clean_in.jsonl` |
| `validation_pool_pairs.jsonl` | 507 | the patient-disjoint validation pool written by `data/freeze_val_clean.py`; per the usage line of `eval/rl_eval_v2.sbatch`, the base read ran its no-image arm on this file and wrote the manifest from it | `results/val_clean/pairs_val_clean_2026-09-24.jsonl` |
| `validation_manifest_image_necessary.txt` | 238 | pair ids of the validation population | `results/val_clean/manifest_image_necessary.txt` |
| `validation_images.txt` | 958 | study paths of the validation pool (leakage checks, top-5 exclusion) | `results/val_clean/images.txt` |
| `locked_read_set.jsonl` | 548 | every pair a locked read processes: the first 40 pairs of each of the 16 (finding, direction) cells of the pair build, in build order (some cells hold fewer than 40) | `results/locked/pairs_e6_2026-09-19_0405.jsonl` |
| `locked_pairs.jsonl` | 274 | locked population = the locked image-necessary manifest (a subset of the read set, same order) | (convenience copy) |
| `locked_manifest_image_necessary.txt` | 274 | pair ids of the locked population | `results/locked/manifest_image_necessary.txt` |
| `selected_round0.txt` | 2,000 | training-slice images sampled in round 0, blocked from validation by `freeze_val_clean.py` | `results/train_slice/selected_round0.txt` |
| `training_images.jsonl` | 4,565 | every image of the GRPO parquet (4,501 train rows, 64 verl-internal validation rows) | see below |
| `training_claims.jsonl` | 4,565 | the reward claims of each training image, as pair identifiers | see below |

Pair rows have the fields `pair_id`, `finding`, `direction` (`false_finding`: the sentence asserts
the finding, which the report study has and the partner lacks; `missed_finding`: the sentence denies
it, and the partner has it), `report_path`, `report_split`, `image_path`, `image_split`,
`target_sentence_index`. The report study is the `true_image` condition, the partner the
`swapped_image` condition.

Why the locked read set and not only the 274 manifest pairs: the reads select pairs per cell
(`PER_CELL=40 CELL_SKIP=0`) and draw the claim-swap donors from all selected pairs with
`random.Random(20260919)` (`verify_run.build_rows`, reused by `rl_swap_adjudicate.py`). The original
locked file is the full pair build (4,423 rows); run on a file rebuilt from `locked_read_set.jsonl`,
the per-cell rule selects the same 548 rows in the same order, so the donors are unchanged. The one
difference is a descriptive count in `rl_exact_table1.py` ("locked patients (pair file)"), which
then counts the read set instead of the whole build. The validation reads use `PER_CELL=200` on the
238-row file, which selects every row.

## Rebuilding the sentences

For each pair row, take the CheXpert Plus table row whose `path_to_image` equals `report_path` and
its `section_findings` text. Then

```python
from build_pairs import sentences          # harness/code/scripts/build_pairs.py
target_sentence = sentences(section_findings)[target_sentence_index]
report_findings = section_findings          # stored in the original files; no script here reads it
```

`sentences()` joins hard-wrapped lines, strips a leading section heading, splits after `.`, `!`, or
`?`, collapses whitespace, and keeps pieces of five or more characters. Against the original files
this rule reproduces the stored `target_sentence` for every row of `validation_pairs.jsonl`,
`validation_pool_pairs.jsonl`, `locked_read_set.jsonl`, and for every training claim (0 mismatches).
Write each rebuilt row back with `target_sentence` (and `report_findings` if wanted) added, keeping
the row order of the shipped file.

The dataset fields the harness reads (`harness/code/scripts/build_pairs.py`, `index_images.py`,
`verify_run.py`): the table `df_chexpert_plus_240401.parquet` with `path_to_image`,
`frontal_lateral`, `ap_pa`, `section_findings`, `split`, `deid_patient_id`; the label file
`CheXpert_Labels/findings_fixed.json` with `path_to_image` and one value per finding; the PNG
archives under `files/PNG_compressed/`, which `index_images.py` matches by byte size to the
`file_name` and `size` columns of `tables/PNG_compressed.sample200.csv` and then indexes member by
member (paths relative to `$RADOPEN_ROOT/datasets/chexpert_plus/`).

## Label files the scripts expect (rebuild from the dataset)

* `results/labels_eval_dev_val.jsonl`: one line `{"path", "labels"}` per image of the evaluation,
  dev, and validation pairs; `labels` is the 12-finding vector in the order of `FINDINGS`
  (`build_pairs.py`). `build_pairs.py` builds pairs from the same kind of vector (the findings labels
  mapped with `norm_label`, uncertain studies dropped). In the runs the file held 0/1 values only,
  and on all 1,295 evaluation, dev, and validation pairs checked it agrees with the pair
  construction: the target finding is opposite on the two studies (1 on the report study for
  `false_finding`) and the other eleven findings equal the pair's context vector. Used for swap
  truth and the oracle control.
* `results/train_slice/images.jsonl`: training-slice images with `labels`; `forms/rl_controls.sbatch`
  reads only `path` and `labels` from it (top-5 prevalence control).
* `results/train_slice/clean_train_images.jsonl`, the input of `train/rl_verl_prep.py`: one line per
  training image, `{"path", "labels", "claims": [{"claim", "truth", "finding", "direction", "role"}]}`,
  where `claim` is the rebuilt `target_sentence` of the claim's `pair_id`, and the other claim fields
  come from `training_claims.jsonl`. Write the lines in `input_line` order (from
  `training_images.jsonl`); the script's shuffle (seed 20260924) then puts the 64
  `verl_internal_val` rows first (they become `val.parquet`) and the rest in the shipped `train` row
  order, provided every image is in the cache (the runs had none missing). The training labels held
  0/1 values only.

## Training image fields

`training_images.jsonl`: `split` (`train` or `verl_internal_val`), `row` (row in
`data/verl_v1/<split>.parquet`), `prep_index` (`extra_info.index`, the position after the prep
shuffle), `input_line` (0-based line in `clean_train_images.jsonl` that lands at `prep_index` under
`random.Random(20260924).shuffle`), `path`, `patient`, `study`, `probe_pos` (0 to 63 for the frozen
probe prefix of `data/dose_v1/probe.parquet` and `data/rep_s<seed>/`, else null).

`training_claims.jsonl`: `split`, `row`, `path`, and `claims`, each with `pair_id`, `finding`,
`direction`, `truth` (`supported` when the image is the pair's report study, `unsupported` when it is
the partner), `role`, `report_path`, `target_sentence_index`.

Checks run against the original files: the probe order in `data/dose_v1/probe.parquet` equals
`probe_pos` order; `train/rl_rep_parquet.py`'s rule applied to these rows reproduces the first three
shuffled paths recorded in the manifests of `data/rep_s101`, `rep_s202`, and `rep_s303`; every
training claim maps to exactly one pair id whose sentence equals the claim; `prep_index` runs
0 to 63 over the `verl_internal_val` rows and 64 to 4,564 over the `train` rows in row order.

## Photograph study (COCO 2014 and VQA v2, identifiers only)

| file | rows | what it is |
|---|---|---|
| `photo_pool_pairs.jsonl` | 516 | the frozen pool: `pair_id`, `split` (`train2014` or `val2014`), `category`, `yes_image_id` and `no_image_id` (COCO image ids of the image where the object is present and of the image where it is absent), `yes_question_id` and `no_question_id` (VQA v2 question ids) |
| `photo_images.txt` | 1,032 | COCO image paths of the pool, relative to `$VQA_ROOT` |
| `photo_screen_pass.txt` | 493 | pair ids that passed the automated screen (the secondary population) |

The question text and the answer counts of the study's pool file are left out. `photo_study/d2_build_pool.py 8 60 20` rebuilds the full pool from VQA v2 and the COCO 2014 instance annotations; the sha256 of the pool the study used is listed in `photo_study/FROZEN_original.sha256`.
