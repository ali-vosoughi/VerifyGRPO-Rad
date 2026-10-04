#!/usr/bin/env python3
"""Figure 2 for npj Digital Medicine (2026-10-03): the study design, drawn by script with no generated imagery.

Why redrawn: (1) Springer Nature's current AI policy does not permit visual content generated from text prompts
without verifiable source material, which the earlier robot panels were; (2) radiologist-readability reviews
(outside reviewers, 2026-10-03) asked for the matched pair to be shown and for plain labels.
Data: the matched pair, its sentence, and the pleural-effusion checklist entries before and after training come from
$FIG_ROOT/artifacts/pair_example/pair.json, written by figures/rl_teaser_pick.py (pair chosen by a rule declared 2026-09-26 13:45: the
first pleural-effusion pair whose run-1 step-40 checklists are right on both images under the rule-based check and
the independent checker). Images: artifacts/pair_example/report.png and partner.png (CheXpert Plus frontal studies).
Usage: FIG_ROOT=<dir> python build_fig2_npj.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

HERE = Path(os.environ.get("FIG_ROOT", "."))   # holds artifacts/ (inputs, read only) and figs/ (outputs)
SRC = HERE / "artifacts" / "pair_example"
OUT = HERE / "figs" / "fig2_method.pdf"

matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                            "pdf.fonttype": 42, "font.size": 7})
INK, GREY, BLUE, ORANGE, GREEN, TEAL = "#1f2933", "#5f6b76", "#1f5f99", "#c0561b", "#2e7d4f", "#1f8a80"
LIGHT = {"blue": "#e8f0f8", "orange": "#fbeee6", "green": "#e7f3ec", "grey": "#f1f3f5", "teal": "#e3f3f1"}

W, H = 7.0, 3.55
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fill, edge, lw=0.8, r=0.05):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, fc=fill, ec=edge, lw=lw))


def text(x, y, s, size=7, color=INK, weight="normal", ha="left", va="center", style="normal"):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, fontstyle=style)


def arrow(x0, y0, x1, y1, color=INK, lw=0.9, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 linestyle=ls, connectionstyle="arc3,rad=%g" % rad))


d = json.loads((SRC / "pair.json").read_text(encoding="utf-8"))
assert d["finding"] == "Pleural Effusion" and d["sentence"]
b, t = d["base"], d["trained_s101_step40"]
assert b["report"]["target_entry"]["present"] is False and b["partner"]["target_entry"]["present"] is False
assert t["report"]["target_entry"]["present"] is False and t["partner"]["target_entry"]["present"] is True
assert t["partner"]["target_entry"]["side"] == "bilateral"
assert b["report"]["truth"] == "supported" and b["partner"]["truth"] == "unsupported"

# ---------------- panel a ----------------
text(0.06, 3.44, "a", size=9, weight="bold"); text(0.22, 3.44, "Training the checklist model", size=8, weight="bold")

# matched pair block
box(0.06, 1.38, 2.36, 1.92, LIGHT["grey"], "#c9d0d6")
text(1.24, 3.17, "Matched pair from CheXpert Plus", size=7, weight="bold", ha="center")
text(1.24, 3.04, "report-derived labels agree on the other 11 findings", size=6.3, color=GREY, ha="center")
for k, (fn, lab, sub) in enumerate((("report.png", "Source study", "effusion label: absent"),
                                    ("partner.png", "Matched study", "effusion label: present"))):
    im = Image.open(SRC / fn).convert("L")
    # 2026-10-04: mask burned-in technologist initials ("LT/REY" on the source study, "NJ" on the matched study);
    # side and position markers ("L", "SEMI-UPRIGHT") stay. Each box is filled with the median of a ring around it.
    assert im.size == (512, 420)
    for mbox in {"report.png": [(426, 0, 478, 17)], "partner.png": [(377, 68, 396, 84)]}[fn]:
        x0, y0, x1, y1 = mbox
        ring = [im.getpixel((x, y)) for x in range(max(0, x0 - 6), min(512, x1 + 6)) for y in range(max(0, y0 - 6), min(420, y1 + 6))
                if not (x0 <= x < x1 and y0 <= y < y1)]
        im.paste(int(sorted(ring)[len(ring) // 2]), mbox)
    x0 = 0.16 + k * 1.14
    iax = fig.add_axes([x0 / W, 1.93 / H, 1.02 / W, 0.84 / H]); iax.imshow(im, cmap="gray", aspect="auto"); iax.axis("off")
    text(x0 + 0.51, 1.84, lab, size=6.8, weight="bold", ha="center")
    text(x0 + 0.51, 1.72, sub, size=6.3, color=GREY, ha="center")
box(0.14, 1.44, 2.20, 0.2, "white", ORANGE, lw=0.8, r=0.03)
text(1.24, 1.54, "Report sentence:  “%s”" % d["sentence"], size=6.6, color=ORANGE, ha="center", weight="bold")

# example entries before / after training
text(0.08, 1.24, "Selected example: pleural effusion entry", size=6.6, weight="bold")
text(0.08, 1.10, "untrained model", size=6.3, color=GREY)
text(0.98, 1.10, "absent", size=6.3); text(1.62, 1.10, "absent", size=6.3, color=ORANGE)
text(0.08, 0.97, "after training (run 1)", size=6.3, color=GREY)
text(0.98, 0.97, "absent", size=6.3); text(1.62, 0.97, "present, bilateral", size=6.3, color=GREEN)
text(0.98, 0.84, "source", size=5.8, color=GREY); text(1.62, 0.84, "matched", size=5.8, color=GREY)

# flow for one image at a time
fx = 2.62
box(fx, 2.66, 1.36, 0.62, LIGHT["blue"], BLUE)
text(fx + 0.68, 3.11, "Checklist model", size=7.2, weight="bold", color=BLUE, ha="center")
text(fx + 0.68, 2.96, "sees 1 radiograph", size=6.3, ha="center")
text(fx + 0.68, 2.84, "never sees the sentence", size=6.3, ha="center")
text(fx + 0.68, 2.72, "(updated in training)", size=6.0, color=GREY, ha="center")
arrow(2.42, 2.55, fx, 2.92)

box(fx, 1.70, 1.36, 0.78, "white", INK, lw=0.7)
text(fx + 0.68, 2.36, "Findings checklist", size=7, weight="bold", ha="center")
for i, s in enumerate(("up to 12 findings, each: present", "or absent, side, box, note;", "any finding may be left", "unmentioned")):
    text(fx + 0.68, 2.21 - 0.12 * i, s, size=6.2, ha="center")
arrow(fx + 0.68, 2.66, fx + 0.68, 2.48)

box(fx, 0.98, 1.36, 0.56, LIGHT["grey"], GREY, lw=0.7)
text(fx + 0.68, 1.41, "Training format", size=6.8, weight="bold", ha="center")
text(fx + 0.68, 1.27, "fixed finding order;", size=6.2, ha="center")
text(fx + 0.68, 1.15, "unmentioned findings", size=6.2, ha="center")
text(fx + 0.68, 1.04, "entered as absent", size=6.2, ha="center")
arrow(fx + 0.68, 1.70, fx + 0.68, 1.54)

cx = 4.18
box(cx, 0.98, 1.0, 0.92, LIGHT["grey"], INK, lw=0.7)
text(cx + 0.5, 1.76, "Training checker", size=7, weight="bold", ha="center")
for i, s in enumerate(("checking model:", "checklist + sentence,", "never the image", "(frozen)")):
    text(cx + 0.5, 1.61 - 0.12 * i, s, size=6.2, ha="center", color=(GREY if i == 3 else INK))
arrow(fx + 1.36, 1.26, cx, 1.30)
text(cx + 0.5, 2.22, "+ “%s”" % d["sentence"], size=6.4, color=ORANGE, ha="center", weight="bold")
text(cx + 0.5, 2.09, "same sentence, each image", size=5.7, color=GREY, ha="center")
arrow(cx + 0.5, 2.02, cx + 0.5, 1.90, color=ORANGE)

box(cx, 0.36, 1.0, 0.44, LIGHT["green"], GREEN, lw=0.8)
text(cx + 0.5, 0.67, "Reward = 1 if the verdict", size=6.2, ha="center", color=GREEN)
text(cx + 0.5, 0.55, "agrees with the report-", size=6.2, ha="center", color=GREEN)
text(cx + 0.5, 0.43, "derived label, else 0", size=6.2, ha="center", color=GREEN)
arrow(cx + 0.5, 0.98, cx + 0.5, 0.80)
text(cx + 0.44, 0.89, "supported / unsupported", size=5.8, color=GREY, ha="right")
ax.plot([cx, 2.51, 2.51], [0.58, 0.58, 2.78], color=GREEN, lw=0.9, ls="--")
arrow(2.51, 2.78, fx, 2.78, color=GREEN)
text(3.30, 0.47, "reinforcement learning (GRPO)", size=6.3, color=GREEN, ha="center")

# ---------------- panel b ----------------
bx = 5.36
ax.plot([bx - 0.08, bx - 0.08], [0.2, 3.4], color="#d5dbe0", lw=0.8)
text(bx, 3.44, "b", size=9, weight="bold"); text(bx + 0.16, 3.44, "Evaluating saved checklists", size=8, weight="bold")
box(bx, 2.88, 1.56, 0.38, "white", INK, lw=0.7)
text(bx + 0.78, 3.14, "Saved checklists", size=7, weight="bold", ha="center")
text(bx + 0.78, 2.98, "untrained and trained models", size=6.2, ha="center")
for k, (lab1, lab2) in enumerate((("As generated", "model's order; blanks kept"), ("Training format", "fixed order; blanks = absent"))):
    yy = 2.46 - k * 0.36
    box(bx, yy, 1.56, 0.30, LIGHT["grey"], GREY, lw=0.6)
    text(bx + 0.78, yy + 0.20, lab1, size=6.3, weight="bold", ha="center"); text(bx + 0.78, yy + 0.08, lab2, size=5.9, ha="center")
arrow(bx + 0.78, 2.88, bx + 0.78, 2.76)
text(bx + 0.78, 2.00, "+ 2 intermediate formats (post hoc split)", size=5.6, color=GREY, ha="center")
rows = (("Training checker", "Qwen3-VL-8B, gave the reward", LIGHT["grey"], INK),
        ("Independent checker", "MedGemma-4B, never in training", LIGHT["teal"], TEAL),
        ("Rule-based check", "present/absent mark; no language model", "white", GREY))
for k, (r1, r2, fill, edge) in enumerate(rows):
    yy = 1.50 - k * 0.46
    box(bx, yy, 1.56, 0.38, fill, edge, lw=0.7)
    text(bx + 0.78, yy + 0.25, r1, size=6.6, weight="bold", ha="center", color=(TEAL if k == 1 else INK))
    text(bx + 0.78, yy + 0.11, r2, size=5.8, ha="center")
arrow(bx + 0.78, 1.95, bx + 0.78, 1.88)
text(bx + 0.78, 0.40, "Measured against report-derived labels:", size=6.0, ha="center", weight="bold")
text(bx + 0.78, 0.28, "discrimination (Youden index)", size=6.0, ha="center")
text(bx + 0.78, 0.16, "and false-alarm rate", size=6.0, ha="center")

# overlap / bounds check on every text artist
fig.canvas.draw()
r = fig.canvas.get_renderer()
bad = []
for tx in ax.texts:
    bb = tx.get_window_extent(r)
    if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width or bb.y1 > fig.bbox.height:
        bad.append(tx.get_text())
assert not bad, "text off canvas: %s" % bad
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=400)
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("wrote", OUT)
