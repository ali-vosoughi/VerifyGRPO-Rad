<!-- Released copy of the registered protocol of the photograph study. Internal file paths, the cluster name, and
the names of internal review steps are replaced by neutral words; nothing else is changed. The registered original
has sha256 a4f9582c46551a95bdd1783935c4c25d511d80392c907276ef38a7f15b4dabc6 (photo_study/FROZEN_original.sha256). -->
# Omission conventions on natural images: protocol for Experiments A and B (REGISTERED 2026-09-30 05:13 EDT)

Registered copy of the draft protocol, made at the freeze. Its sha256 is listed in
domain2/study/FROZEN.sha256, and the manifest digest is logged in the project log. Nothing below
changes except by a new, dated version logged before any study read.

Revision log: 2026-09-30 04:57 EDT, sixth pre-freeze check (both reviewers ACCEPT): D0, E, CLEAN, criteria and the
combined claim refer to the writer-complete pairs (section 7, missing data); a reply stopped at the token limit is never
a record.
2026-09-30 04:49 EDT, fifth pre-freeze check: R2 made diagnostic only (a cut-off record stays missing;
section 3).
2026-09-30 04:37 EDT, writer parsing (section 3): rules R1 (dict layout) and R2 (cut-off prefix), from the
writers-only check 23424 with the 2048-token limit (Qwen3-VL-8B 82/82 in both styles; Qwen2.5-VL-7B sparse 81/82 by a
repetition loop, checklist 79/82 by the dict layout); with the rules every calibration reply parses.
2026-09-30 04:00 EDT, after the fourth pre-freeze check (two outside reviews: every earlier item fixed, K5
accepted): calibration control cells rescored with missing verdicts as failures (164 expected per cell): K3 97.56%,
K4 97.56%, K5 98.78%, so K5 stands; writer max new tokens raised from 512 to 2048 (section 3).
2026-09-30 03:48 EDT, convention wording settled, still before any study read: a strict judge parser
was adopted (section 6), and a second probe (job 23386) scored K3 again with it on the same 8 smoke pairs: Gemma's
lowest cell moved from 93.8% to 84.4% at temperature 0. The smoke pairs are all person pairs (32 verdicts per cell,
1 verdict = 3.1%), too few to choose near 90%, and every failure of every wording was the same refusal ("the record
does not list any people, but it also does not explicitly state their absence"). The choice was therefore made on a
calibration set built for it, calib_v2 (d2_make_calib.py: 41 pairs outside the study pool and the smoke set, 17
categories, at most 6 per category, at least 6 of 10 opposite VQA answers plus COCO agreement; labels cannot move
comprehension or controls, whose expected verdict follows the record's stated target), with fresh sparse records from
both writers, under a rule logged before the run (job 23398): score = the lowest comprehension or control cell over
both primary judges; adopt the highest score; ties K5 > K3 > K4; no freeze below 90%. Scores: K3 98.8%, K4 98.8%,
K5 100%. K5 is adopted (section 5); it states the closed-world default outright and names no undisclosed list.
Round-3 plumbing: judge parser strict (the whole reply one JSON object with a string verdict from the registered set
and a string reason); every runner persists each answer the moment it completes; the study job re-derives the exact
expected slots, arm set, identity, and input hashes of every judge output before reuse and after creation
(d2_judge.py --verify-only), checks writer records against their DONE marker identity, and verifies the manifest digest
logged at the freeze plus every registered file before every study call.
2026-09-30 02:41 EDT, convention wording, still before any study read: the universe sentence added at
the red-team ("The record covers exactly these objects: <20 names>.") made both primary judges read the enumerated
names as the record's findings in out-of-pool smoke 23368 (comprehension on the sparse encoding 85.4% and 81.2%, down
from 100%; sparse target controls 59% to 75%, failures only on unlisted targets). Engineering probe 23375 (synthetic
comprehension set and out-of-pool smoke controls only; the encoding contrast E never computed) scored 4 candidate
wordings under a rule fixed before it ran: adopt the first of K1 (checklist named as possible objects, not findings),
K2 (claim object declared on the checklist, no enumeration), K3 (the prior rule plus the precedence sentence), K0
(the prior smoked wording) whose every comprehension and control cell reaches 90% for both primary judges. K1 failed
on Gemma (lowest cell 68.8%), K2 on Qwen3-VL-8B (84.4%); K3 passed (lowest cells 96.9% and 93.8%) and is adopted. The
judge therefore is not shown the enumerated object list; the writer prompt names the same 20 objects.
2026-09-30 02:22 EDT, after the recheck of those fixes (two outside reviews, still before any study read): every
interval, descriptive ones included, uses 10,000 draws; row accounting counts absent rows against the expected slots;
category-balanced estimates and screen exclusions by category and direction are computed in the frozen analysis; the
decomposition reports judge, reader, and residual on identical pairs; rows outside the 4 registered slots are errors;
runners reserve their output directory atomically, persist every answer as it arrives, and mark completion last;
writer records are checked for writer and style before any read; the 30B judge is refused unless it runs with
tensor parallel 2 on 2 allocated GPUs; the manifest must list every registered file and is re-verified before every
study call.
2026-09-30 02:05 EDT, after the pre-freeze red-team of this text and the code (two outside reviews, before any
writer record or judge read of a study image): the convention sentence now names the object universe and gives the
states and default precedence over notes; the target-control criterion is registered (section 7); missing data are
bounded over every pool pair with fills reversed for negative-weight terms; the analysis keys rows by writer style;
the label-error sensitivity is registered as a greedy diagnostic; the screen parser accepts only a bare yes or no; the
freeze covers a byte-level manifest and pinned model commits; outputs are immutable.
2026-09-30, after the acceptance consult: the full 2-rater human check is replaced by the registered
automated screen, the primary population is all 516 pool pairs, and a label-error sensitivity is added.
2026-09-29 23:22 EDT, after engineering smoke 23273 (out-of-pool pairs only): control records are
states only; parser wording matches the harness; judge context length set per model (OLMo-2 4,096 tokens).
2026-09-30 00:33 EDT: engineering smoke complete for all 5 judges and both writers (project log). Resources fixed: Qwen3-VL-30B-A3B
runs with tensor parallel 2 on 2 A6000; Gemma-3-12B revision 96b6f1eccf38110c56df3a15bffe176da04bfd80.

Status: DRAFT written 2026-09-29 after the lead author's decision (project log 22:49 EDT) adopting an outside-review
study plan and the earlier design consult. This file becomes the
registration after the automated screen has run: it is then hashed, the hash and the final sample size are logged in
the project log, and nothing below changes afterwards except by a new, dated version logged BEFORE any study read.
No writer or judge has read any study image or record when this draft is written.

## 1. Question

A (omission semantics). Does changing what an omission asserts (left blank, written "absent", written "not reported")
change the measured difference between 2 visual writers differently for 2 fixed judges?

B (equivalent encodings). With the meaning of the record held fixed by an explicit closed-world contract, does the
representation alone (every object's state written out, or only the present objects plus the declared default) change
that measured difference differently for the same 2 judges?

## 2. Pairs and human truth

- Pool (frozen, label-only): `domain2/pool_v8_cap60_n20/pool.jsonl` (sha256 fce65949c252eff1...), built by
  `domain2/code/d2_build_pool.py` (sha256 aa84052166423ffd..., args `8 60 20`): VQA v2 official complementary pairs
  (train2014 352 + val2014 164), literal unqualified existence questions, at least 8 of 10 opposite yes / no answers,
  COCO 2014 instances agree (crowd regions count as present), image-disjoint (seed 20260929), 20 categories capped at
  60 pairs each: person, clock, car, dog, bird, umbrella, boat, cat, banana, sink, stop sign, tv, bench, bicycle, knife,
  laptop, train, bus, mouse, chair. 516 pairs, 1,032 images.
- Truth (revised 2026-09-30 after the two outside reviews acceptance consult, before any study read; both advised that the full
  2-rater human check is not needed): each image's state comes from 2 separate human annotation sources that must agree,
  at least 8 of 10 VQA v2 answers and the COCO 2014 instance annotations (crowd regions count as present). This is
  described as agreement between separate annotation sources, not as independently verified truth.
- Automated screen (registered): HuggingFaceM4/Idefics3-8B-Llama3, whose language model (Llama 3) belongs to no writer or
  judge family in the study (InternVL2_5-8B, the first choice, failed to load its tokenizer in the smoke, 2026-09-30), answers `Is there a <name> in this image? Answer with yes or no.` for every pool image, temperature
  0, blind to labels, records, and verdicts (`domain2/code/d2_screen.py`). An answer parses only when the whole reply is
  yes or no, ignoring case, surrounding white space, and trailing full stops or exclamation marks; any other reply
  (for example "not sure") is unparsed. A pair passes when the answer is yes on the present image and no on the absent
  one; an unparsed answer fails. The screen runs once, before any writer record. Result (2026-09-30 01:51 EDT, before
  any writer record): 1,032 images, 0 unparsed, 97.8% agreement with VQA and COCO (14 disagreements on present images,
  9 on absent images); 493 of 516 pairs pass. Every reply was exactly "Yes." or "No.", so the full-match parser adopted
  afterwards gives the identical result.
- Populations: PRIMARY = all 516 pool pairs (the screen never removes a pair from the primary analysis, so no
  model-selected subset carries the claim); SECONDARY = the pairs that pass the screen. Both are reported, with the
  screen's agreement rate and exclusions by category and answer direction.
- Label-error sensitivity (registered diagnostic, not a decision rule): for D0 and E on the complete-case pairs, the
  expected estimate under random pair reversal at 2%, 5%, and 10% (a reversal swaps the true and false image and negates
  the pair's contribution, so the expected estimate shrinks by the factor 1 - 2p, checked by simulation), and a GREEDY
  count of worst-case reversals (largest same-sign pair contributions reversed first, starting from 0 reversals) until
  the estimate falls below the 5% threshold and until a normal-approximation 95% interval includes 0. It is reported as a
  greedy count, not as a proven minimum.
- Optional human spot check (never blocks a run): 1 rater may check 75 randomly chosen pairs (seed 20260930) on the check
  page, blind to all labels and outputs; its disagreement rate is reported as a spot check, not as certified label error.

## 3. Writers (inference only)

Qwen/Qwen2.5-VL-7B-Instruct (older) and Qwen/Qwen3-VL-8B-Instruct (newer), exact revisions pinned at freeze (section
8); vLLM,
temperature 0, max 2048 new tokens (raised from 512 before the freeze: on the calibration set 1 of 82 images per writer
was cut off at 512 tokens while listing many person instances, and the checklist style lists about 16 objects), JSON requested; identical prompts and decoding. Each image gets 1 record per writer
and style, generated once and reused for every claim, form, encoding, and judge. The writer never sees the question,
the target object, the claim, the answer, the pair, or the partner image.

System message (both styles):
`You are looking at a photograph and recording what you see. Answer only with the JSON object asked for. Do not add commentary.`

Sparse style (PRIMARY) user message, after the image:
`A photograph is attached. Record the objects from the list below that you can see in it, before any question about
it is asked. List an object only if at least one is visible. Objects to consider: <20 display names, fixed order>.
Reply as {"objects": [{"object": "<name from the list>", "note": "<where it is, one short phrase>"}], "other": "<anything notable not in the list, one sentence>"}`

Checklist style (SECONDARY, registered now, never substituted for the primary) user message, after the image:
`A photograph is attached. Record what it shows, object by object, before any question about it is asked. For each
object in the list below, say whether at least one is visible. Objects to cover: <20 display names>. Reply as
{"objects": [{"object": "<name from the list>", "present": true or false, "note": "<short>"}], "other": "<one sentence>"}`

Display names: the COCO names, except "tv" -> "TV or computer monitor" and "mouse" -> "computer mouse". Parsing uses the
radiology harness's JSON extraction (code fences and leading prose stripped, then strict JSON); entries whose object is not in the list are dropped and counted; a duplicate keeps its first entry. Two deterministic rules were added before the freeze (writers-only check 23424 on the calibration images: Qwen2.5-VL-7B wrote 3 of 82 checklist replies as a dictionary keyed by object name, and 1 of 82 sparse replies repeated one entry until the 2048-token limit): R1 reads the layout {"objects": {"<name>": {...}}} as the list [{"object": "<name>", ...}] in its written order; R2 is diagnostic only (revised at the fifth pre-freeze check): a reply the server reports as stopped at the token limit is never a record, even if part of it parses; its complete entries of the "objects" array before the cut (exactly one comma between entries, at least 1 entry) are stored as parsed_prefix, and the record itself stays missing, because a prefix is not a complete record. The counts by writer and style (list, dict, missing_stopped_at_limit, missing_unparsed) are reported. Replayed on all 328 calibration replies, R1 recovers the 3 dict-layout replies and leaves the 324 other records unchanged; the 1 looping reply stays missing. A
record that does not parse after 1 identical transport retry is "unparsed" and its image's pairs leave the common set
for that writer and style (counted and reported).

Decoded state vector per record: sparse style, listed = present, unlisted = omitted; checklist style, present true =
present, present false = absent, unlisted = omitted.

## 4. Claims

For the pair's category, 2 claims per image from frozen templates: assertion `There is a <name> in the image.` and
denial `There is no <name> in the image.` (article "an" before a vowel sound; `There is a TV or computer monitor in the
image.`). Truth follows the pool state (VQA and COCO in agreement). Each image carries both polarities, so each pair has a true and a false
image for each claim.

## 5. Forms and encodings (built by one renderer, fixed object order, claim never passed to it)

Common schema: `{"objects": [{"object": <name>, "state": "present" | "absent" | "not reported"}], "notes": {<name>: <note>}, "other": <text>}`;
`notes` holds the writer's notes for the objects it wrote, `other` its free text, both byte-identical across all forms
and encodings of a record.

Experiment A (default judge instruction, no convention sentence):
- unfilled: only the objects the writer wrote, with their decoded state;
- absent: all 20 objects, omitted ones written as "absent";
- not reported: all 20 objects, omitted ones written as "not reported".

Experiment B (closed-world contract, sparse-style records):
- complete: all 20 objects with "present" or "absent" (omitted decoded as absent);
- sparse: only the "present" objects, plus the declared default;
- both carry the same contract in the judge message: `Convention for this record: the listed states are
  authoritative for whether an object is in the photograph. Any object that the record does not list is absent from the
  photograph, even though the record does not say so explicitly. The listed states and this default take precedence
  over the notes and the other field.` (The precedence sentence was added at the pre-freeze red-team so that a writer's
  note or free text cannot override the stated state; the wording was chosen on the calibration set, see the revision
  log; an enumerated object universe was tried and removed.)
- a deterministic decoder recovers the state vector and the notes / other payload from each encoding; the run asserts
  equality and records the hash of the decoded object before any judge call.

Controls (primary judge pair, both B encodings):
- target-corrected: the decoded record with only the focal object's state set to the pool truth;
- target-reversed: the same with only the focal state negated;
- both control records carry STATES ONLY (notes and free text removed; engineering smoke 23273: the writer's own note on
  the focal object, e.g. "walking between two elephants", contradicts a reversed state and the judges followed the
  note), so the controls test whether a judge reads the stated fact;
- open-world ternary (secondary): complete ternary record vs sparse record, the same contract and precedence
  sentence with the default `Any object that the record does not list is not reported, so the record says nothing about
  whether it is in the photograph.` (not probed; secondary), judge asked for supported / contradicted / unknown relative to the
  record (acceptance = supported only); reported as semantic agreement and coverage, never called equivalent to B.

## 6. Judges (text only, never the image)

Qwen/Qwen3-VL-8B-Instruct (the paper's reward-judge base) and google/gemma-3-12b-it = the PRIMARY pair (Gemma chosen
before any read for continuity with the paper's independent judge MedGemma, a Gemma-family model; OLMo was the
consultants' first suggestion and stays secondary). Secondary panel: Qwen/Qwen3-VL-30B-A3B-Instruct,
microsoft/Phi-4-mini-instruct, allenai/OLMo-2-1124-7B-Instruct. No judge is replaced after any study output.
vLLM, temperature 0, max 128 new tokens, 1 identical transport retry, strict parse of
`{"verdict": "supported" or "unsupported", "reason": "..."}`: the whole reply must be one JSON object with a string
verdict from the registered set (case-insensitive) and a string reason; code fences, leading prose, a missing reason,
or any other verdict make the reply unparsed (the writer parser in section 3 stays the tolerant harness extraction).

System: `You are verifying a single sentence about a photograph. Sentences are sometimes true of the photograph and
sometimes not; both verdicts are common. Answer only with the JSON object asked for. Do not add commentary.`
User: `Here is a record of what a photograph shows, written before any sentence was made:\n{evidence}\n\n[convention
sentence, Experiment B and controls only]\n\nSentence about that photograph:\n{claim}\n\nUsing only the record, judge
whether the sentence is true.\nReply as {"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}`

Comprehension check BEFORE any study read (synthetic records, no study image): 48 balanced cases per encoding (present /
absent / omitted target x assertion / denial, random other objects), each judge; pass = at least 90% of the verdicts the
contract implies. A judge that fails is reported and its study results are labelled comprehension-failing; the primary
pair is not replaced.

## 7. Estimands and decision rules

Pair-level Youden index for writer w, judge j, form f: Y = mean over pairs and both polarities of [accept on the true
image] - [accept on the false image]. FS = rejection rate of true claims, reported by polarity, with false acceptance.

G_{j,f} = Y(Qwen3-VL-8B writer, j, f) - Y(Qwen2.5-VL-7B writer, j, f), the measured writer difference.

A (primary): H_j = G_{j,absent} - G_{j,unfilled}; D0 = H_{Qwen3-VL-8B judge} - H_{Gemma-3-12B judge}.
B (primary): E = (G_{Q8,sparse} - G_{Q8,complete}) - (G_{Gemma,sparse} - G_{Gemma,complete}).

Inference: pairs are the unit (image-disjoint), resampled whole with both images, polarities, writers, forms, encodings,
and judges together; 10,000 bootstrap draws, seed 20260930; two-sided 95% percentile intervals, for every interval the
analysis reports (primary, bounds, and descriptive). Pair weighting; assertion and denial averaged within a pair first;
category-balanced summaries secondary (each category weighted equally, pairs resampled within category).

Rows are identified by pair, writer, judge, writer style, form or encoding, polarity, and image; a duplicate is an
error. D0 and E use the sparse writer style; the checklist style is analysed and reported separately.

Decision rules (each primary): the interval excludes 0 AND |point| >= 5%. Criteria registered at the pre-freeze
red-team: comprehension, at least 90% of the 48 cases of each encoding for each primary judge (a missing verdict counts
as wrong); target controls, at least 90% agreement with the record-implied verdict in every primary judge x writer x
encoding (complete, sparse) x control (corrected, reversed) cell, over 4 verdicts per population pair (2 images x 2
polarities; a missing verdict or a missing record counts as a failure). The combined claim requires D0 and E to pass,
both to be clean (below), and every comprehension and control criterion to pass for the primary pair. Secondary results (other judges,
checklist style, per-judge H_j, polarity FS, omission exposure, writer accuracy against the human labels, the open-world
arm, category summaries) are descriptive unless a multiplicity procedure is added to a new version before reads.

Missing data (population rule accepted by both reviewers at the sixth pre-freeze check): the population is read from
the frozen pool (PRIMARY all 516 pair ids; SECONDARY the ids in screen_pass.txt), so a pair with no row at all is
counted, never dropped silently. D0, E, their bounds, CLEAN, the comprehension and control criteria, and the combined
claim refer to the WRITER-COMPLETE pairs of that population: the pairs whose 4 sparse writer records (both writers, both
images) exist. A writer failure happens before any claim, form, encoding, or judge exists and removes the pair from all
of them alike; such pairs are excluded, counted, and described by category and by writer, and the result is stated for
the writer-complete population, not for all 516 pairs. The point estimate and interval use the complete-case pairs
(every verdict the contrast needs parses); parse counts by writer, judge, style, form, and truth state are reported.
Bounds: every missing judge verdict of every writer-complete pair is set to its least (lower bound) and most (upper
bound) favourable value for the sign of the term it enters (for a term with a negative weight the fills are reversed),
and each bound gets its own 10,000-draw percentile interval. A result that passes is called clean only if the bound
adverse to its sign still passes with the same sign. The old worst-case fill over the whole population, writer-missing
pairs included, is reported as a diagnostic, not a criterion.

Decomposition (secondary; judge G, reader G, and their difference are computed on the identical pairs of each
judge and form): deterministic reader F reads the decoded target state (assert accepted iff present, deny
accepted iff absent, omitted = absent under the closed-world contract); G_{j,f} = [Y(Q3 writer, F) - Y(Q2.5 writer, F)] +
[R(Q3, j, f) - R(Q2.5, j, f)] with R = Y(j) - Y(F); the first term is called decoded target-state performance, never
perception.

Power (planning only, not a prediction): with up to 516 pairs, an assumed pair-level contrast SD of 35% gives about 4.3%
detectable at 80% power, 50% gives about 6.2%; at a true effect of exactly 5% the |point| >= 5% rule passes about half
the time.

## 8. Order of work

1. The automated screen (once, before any writer record); its agreement rate and the screened subset logged.
2. Freeze: a byte-level sha256 manifest (`domain2/study/FROZEN.sha256`) covers this file (copied as the registered
   version), the pool, the image list, labels.json, the screen outputs, every script and the job file, and
   `study/model_revisions.txt`; the study job verifies the whole manifest before any model loads and loads every model at
   its pinned commit (`--revision`); the manifest hash is logged in the project log. Writer and judge outputs are written
   once and never overwritten; a writer output is reused only if it holds the 1,032 pool images exactly once each.
   Pinned commits: Qwen3-VL-8B-Instruct 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b, Qwen2.5-VL-7B-Instruct
   cc594898137f460bfe9f0759e9844b3ce807cfb5, gemma-3-12b-it 96b6f1eccf38110c56df3a15bffe176da04bfd80,
   Qwen3-VL-30B-A3B-Instruct 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c (tensor parallel 2, 2 GPUs asserted),
   Phi-4-mini-instruct cfbefacb99257ffa30c83adab238a50856ac3083, OLMo-2-1124-7B-Instruct
   470b1fba1ae01581f270116362ee4aa1b97f4c84, screener Idefics3-8B-Llama3 fddb4ff79181e55a994674777e06cd5456ce3dc3.
3. Smoke on images OUTSIDE the pool (not study images): writer parse rate, round-trip decode equality for every encoding,
   the synthetic comprehension check, throughput and memory, a full miniature analysis on synthetic outcomes. The smoke
   never chooses a judge, style, category, or form.
4. Writer records for all study images (both writers, both styles).
5. Judge reads: primary pair first (A, B, controls), then the secondary panel and the checklist style.
6. The frozen analysis runs once; results reported under the decision rules; the result table below decides the paper.

| Result | Licensed paper |
|---|---|
| D0 and E pass, controls pass | Omission conventions distort visual-writer comparisons, including under equivalent decoded records |
| D0 passes, E does not | A broader omission-semantics case study; the "adding absence changes meaning" objection remains |
| E passes, D0 does not | An equivalent-encoding failure, with the second-domain omission replication reported as unsuccessful |
| Both fail | The radiology measurement paper as it stands, with this study reported as a bounded negative |
| Effect only in comprehension-failing reads | An instruction-following failure, not the stronger visual-evaluation claim |
