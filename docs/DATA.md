# Data

[Back to the README](../README.md)

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

## Not included

* Results files, saved checklists, verdicts, reward logs, rollouts, checkpoints, and adapters, including the
  photograph study's saved checklists, verdicts, screen outputs, and frozen analysis output.
* Any CheXpert Plus image, report text, or label; any COCO image or annotation; any VQA question or answer. The derived
  label files are rebuilt from your copies (`pairs/README.md`; `photo_study/d2_screen.py` for `labels.json`).
* Site-specific helpers of the study: scheduler accounting, model download and environment build scripts, one-off
  scripts that patched or generated files whose final form is released here, and superseded backups.
