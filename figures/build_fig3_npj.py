#!/usr/bin/env python3
"""Figure 3 v3, the evidence, full width, 2 panels (2026-09-28, after a readability review). Readable in 10 seconds: each panel
title is the question it answers, series are labelled directly, light major grids sit behind the data, a one-line
reading aid says which way is better, and only the inferential contrast carries an interval.

(a) "Does the training format change the measured gain?" -- two slope facets (validation, registered; held-out test, post
    hoc): the seed-averaged change in Y from untrained to trained records, read in the original form and in the
    reward-scoring form, per judge; below, the between-judge difference in that response with its 95% interval.
(b) "Do the learned records discriminate the images?" -- untrained writer and runs 1 to 3: the rule-based reader's Y
    (squares) beside the full range of 200 shuffles within finding and sentence direction (grey bars, dark tick at the
    maximum).
Sources: artifacts/replication/REPLICATION_FINAL_decomp_payload_decomposition.json (validation, 237 pairs),
artifacts/locked_forms/LOCKED_orderfill_POSTHOC.json (held-out test, 270 pairs), artifacts/fig3/fig3_data_replication.json
(panel b, 238 pairs). The per-finding rates of v2 panel c move to supplementary Figure S4. Printed values: percent with
one decimal from the half-up three-decimal value. Reads $FIG_ROOT/artifacts/, writes $FIG_ROOT/figs/fig3_evidence.pdf
(+ .png).
"""
import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(os.environ.get("FIG_ROOT", "."))   # holds artifacts/ (inputs, read only) and figs/ (outputs)
ART = HERE / "artifacts"
OUT = HERE / "figs" / "fig3_evidence.pdf"
W, H = 6.875, 3.10
GREYJ, TEAL, SLATE = "#6b6f76", "#2a9d8f", "#34495e"
INK, SUB, GRID, ZERO, RANGE = "#20252b", "#5f6670", "#e3e7eb", "#8f969e", "#c9ced4"
FS, FT = 8, 9


def f3(x):
    q = Decimal(str(abs(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return float(q) if x >= 0 else -float(q)


def pc(x, n=None):
    """percent, one decimal; with n, the value is first recovered as the exact fraction k / n (0.0605 stored = 43/711
    = 0.06048 prints 6.0, not 6.1)."""
    if n:
        k = round(x * n)
        assert abs(k / n - x) < 6e-5, (x, n)
        q = (Decimal(k) / Decimal(n)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return "%.1f" % (float(q) * 100)
    return "%.1f" % (f3(x) * 100)


def box(x0, y0, x1, y1):
    return [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]


def style(ax, grid_axis="y"):
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("#7a8189")
    ax.tick_params(which="both", direction="out", length=2.5, width=0.6, colors=INK, labelsize=FS, top=False, right=False)
    ax.minorticks_off()
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42,
                         "font.size": FS, "axes.unicode_minus": True})
    dec = json.loads((ART / "replication" / "REPLICATION_FINAL_decomp_payload_decomposition.json").read_text(encoding="utf-8"))["values"]
    lk = json.loads((ART / "locked_forms" / "LOCKED_orderfill_POSTHOC.json").read_text(encoding="utf-8"))["values"]
    # Full-precision interval ends from the 2026-10-03 cluster recompute (audit_2026-10-03/fullprec_out.txt): the stored
    # 4-decimal values 0.1285 and 0.0196 would round twice; use the unrounded bootstrap ends for the printed labels.
    assert abs(lk["diff_Y_interface"]["hi95"] - 0.1285) < 1e-4 and abs(dec["diff_Y_interface"]["lo95"] - 0.0196) < 1e-4
    lk["diff_Y_interface"]["hi95"] = 0.128485
    dec["diff_Y_interface"]["lo95"] = 0.019608
    d = json.loads((ART / "fig3" / "fig3_data_replication.json").read_text(encoding="utf-8"))["reads"]
    fig = plt.figure(figsize=(W, H))

    # ---------------------------------------------------------------- (a) slopes + contrast strip
    fig.text(0.10 / W, (H - 0.13) / H, "a", fontsize=FT + 1, fontweight="bold", va="center")
    fig.text(0.26 / W, (H - 0.13) / H, "Does the training format change the measured gain?", fontsize=FT, fontweight="bold",
             va="center", color=INK)
    facets = ((0.62, 2.17, "Validation pairs, prespecified", dec, 711), (2.55, 4.10, "Held-out test pairs, post hoc", lk, 810))
    for i, (x0, x1, title, src, n3) in enumerate(facets):
        ax = fig.add_axes(box(x0, 1.18, x1, 2.62))
        style(ax, "y")
        for reader, col, mk, lab in (("training_judge", GREYJ, "o", "training checker"),
                                     ("independent_judge", TEAL, "^", "independent checker")):
            a, b = src["%s_Y_d_raw" % reader]["point"], src["%s_Y_d_canonical" % reader]["point"]
            ax.plot([0, 1], [a, b], color=col, lw=1.4, zorder=2)
            ax.plot([0, 1], [a, b], ls="none", marker=mk, ms=5.5 if mk == "^" else 5, mfc=col, mec=col, zorder=3)
            ax.text(-0.09, a, pc(a, n3), ha="right", va="center", fontsize=FS, color=col)
            ax.text(1.09, b, pc(b, n3), ha="left", va="center", fontsize=FS, color=col)
        if i == 0:   # compact coloured key in the empty upper part of the first facet, and the reading aid
            ax.plot([-0.33], [0.139], ls="none", marker="^", ms=5, mfc=TEAL, mec=TEAL, clip_on=False)
            ax.text(-0.25, 0.139, "independent checker", color=TEAL, fontsize=FS, va="center")
            ax.plot([-0.33], [0.124], ls="none", marker="o", ms=4.5, mfc=GREYJ, mec=GREYJ, clip_on=False)
            ax.text(-0.25, 0.124, "training checker", color=GREYJ, fontsize=FS, va="center")
        if i == 1:
            ax.text(1.38, 0.139, "↑ higher gain", color=SUB, fontsize=7.5, ha="right", va="center")
        ax.set_xlim(-0.42, 1.42); ax.set_ylim(0, 0.15)
        ax.set_yticks([0, 0.05, 0.10, 0.15])
        ax.set_yticklabels(["0", "5", "10", "15"] if i == 0 else [])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["as\ngenerated", "training\nformat"], fontsize=FS, linespacing=1.0)
        ax.set_title(title, fontsize=FS, color=INK, pad=3)
        if i == 0:
            ax.set_ylabel("Improvement in $Y$ from training (%)", fontsize=FS, color=INK, labelpad=2)
    # contrast strip
    ax = fig.add_axes(box(1.62, 0.38, 4.10, 0.86))
    style(ax, "x")
    rows = (("validation, prespecified", dec["diff_Y_interface"], 711), ("held-out test, post hoc", lk["diff_Y_interface"], 810))
    for j, (lab, v, n3) in enumerate(rows):
        y = 1 - j
        ax.plot([v["lo95"], v["hi95"]], [y, y], color=INK, lw=1.1, zorder=2)
        ax.plot([v["point"]], [y], ls="none", marker="D", ms=5, mfc=INK, mec=INK, zorder=3)
        ax.text(v["hi95"] + 0.004, y, "%s%% [%s%%, %s%%]" % (pc(v["point"], n3), pc(v["lo95"]), pc(v["hi95"])),
                ha="left", va="center", fontsize=FS, color=INK)
    ax.axvline(0, color=ZERO, lw=0.8, zorder=1)
    ax.set_ylim(-0.6, 1.6); ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
    ax.set_xlim(-0.02, 0.265); ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20]); ax.set_xticklabels(["0", "5", "10", "15", "20"])
    ax.set_xlabel("Between-checker contrast: independent minus training checker (%)", fontsize=FS, labelpad=1.5)

    # ---------------------------------------------------------------- (b) observed vs shuffles
    fig.text(4.34 / W, (H - 0.13) / H, "b", fontsize=FT + 1, fontweight="bold", va="center")
    fig.text(4.50 / W, (H - 0.13) / H, "Do checklists track image labels?", fontsize=FT, fontweight="bold",
             va="center", color=INK)
    ax = fig.add_axes(box(5.12, 0.62, 6.78, 2.45))
    style(ax, "x")
    reads = (("base", "untrained"), ("s101", "run 1"), ("s202", "run 2"), ("s303", "run 3"))
    for j, (k, lab) in enumerate(reads):
        y = len(reads) - 1 - j
        null = np.array(d[k]["null_Y"])
        lo, hi = float(null.min()), float(null.max())
        ax.plot([lo, hi], [y, y], color=RANGE, lw=5.5, solid_capstyle="butt", zorder=1)
        ax.plot([hi, hi], [y - 0.2, y + 0.2], color=SUB, lw=1.0, zorder=2)
        obs = d[k]["observed_Y"]
        ax.plot([obs], [y], ls="none", marker="s", ms=5.5, mfc=("white" if k == "base" else SLATE), mec=SLATE, mew=1.2, zorder=3)
        ax.text(obs, y + 0.33, pc(obs, 238), ha="center", va="bottom", fontsize=FS, color=SLATE)
    ax.axvline(0, color=ZERO, lw=0.8, zorder=0)
    ax.set_ylim(-0.6, len(reads) - 0.2)
    ax.set_yticks(range(len(reads))); ax.set_yticklabels([r[1] for r in reads][::-1], fontsize=FS)
    ax.set_xlim(-0.12, 0.21); ax.set_xticks([-0.1, 0, 0.1, 0.2]); ax.set_xticklabels(["−10", "0", "10", "20"])
    ax.set_xlabel("Rule-based check's $Y$ (%)", fontsize=FS, labelpad=1.5)
    fig.text(5.83 / W, 2.71 / H, "grey bar: range of 200 shuffles, not a\nconfidence interval; right of it = beyond all",
             ha="center", va="center", fontsize=7.5, color=SUB, linespacing=1.1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
