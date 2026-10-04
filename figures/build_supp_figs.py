#!/usr/bin/env python3
"""Supplementary figures S1-S4 in the readable style of Figure 3 v3 (2026-09-28): plain
panel questions, light major grids behind the data, bottom/left ticks only, direct labels, fixed reader colours and
shapes, every value in percent unless it is a count. Built only from artifacts; nothing is generated.

S1  training curves, 2 x 2: record length and KL term, standard reward vs omission penalty (artifacts/train_curves).
S2  results for each run: change in Y and in the false-strike rate per reader, runs 1-3 and their mean, validation and
    held-out test (artifacts/replication/{REPLICATION,LOCKED}_FINAL_endpoints.json).
S3  the 5 judges with and without the stated convention: form effect on the gain and effect of writing absences on
    false strikes, default instruction (open) vs convention (filled) (artifacts/conv/<judge>_<pop>/orderfill.json).
S4  what the writers put in their records: per-finding detection and false-alarm rates of the target flag, untrained
    vs mean of runs (artifacts/fig3/fig3_data_replication.json), and the distribution of findings listed per record
    (artifacts/listed_hist/listed_hist.json).
Writes figs/figS1_training.pdf, figS2_perrun.pdf, figS3_convention.pdf, figS4_records.pdf (+ .png).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(os.environ.get("FIG_ROOT", "."))   # holds artifacts/ (inputs, read only) and figs/ (outputs)
ART, FIGS = HERE / "artifacts", HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
W = 6.875
INK, SUB, GRID, ZERO = "#20252b", "#5f6670", "#e3e7eb", "#8f969e"
GREYJ, TEAL, SLATE = "#6b6f76", "#2a9d8f", "#34495e"
JUDGE = {"q8": ("training judge", GREYJ, "o"), "mg": ("independent judge", TEAL, "^"),
         "q30": ("Qwen3-VL-30B-A3B", "#7b5ea7", "D"), "olmo": ("OLMo-2-7B", "#4a7ab5", "p"),
         "phi": ("Phi-4-mini", "#8c6d46", "v")}
STD, PEN = "#252525", "#b96a20"
FS, FT = 8, 9


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42,
                         "font.size": FS, "axes.unicode_minus": True})


def style(ax, grid_axis="both"):
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("#7a8189")
    ax.tick_params(which="both", direction="out", length=2.5, width=0.6, colors=INK, labelsize=FS, top=False, right=False)
    ax.minorticks_off()
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def title(fig, x, y, letter, text, H):
    fig.text(x / W, y / H, letter, fontsize=FT + 1, fontweight="bold", va="center")
    fig.text((x + 0.16) / W, y / H, text, fontsize=FT, fontweight="bold", va="center", color=INK)


def box(x0, y0, x1, y1, H):
    return [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]


def pct_axis(ax, axis, ticks):
    lab = [("%g" % round(t * 100, 6)).replace("-", "−") for t in ticks]
    if axis == "x":
        ax.set_xticks(ticks); ax.set_xticklabels(lab)
    else:
        ax.set_yticks(ticks); ax.set_yticklabels(lab)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf"); fig.savefig(FIGS / f"{name}.png", dpi=300); plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------ S1
def smooth(y, k=5):
    pad = k // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(k) / k, mode="valid")


def fig_s1():
    d = json.loads((ART / "train_curves" / "train_curves.json").read_text(encoding="utf-8"))
    H = 3.45
    fig = plt.figure(figsize=(W, H))
    cols = (("response_length/mean", "How long are the records?", "record length (tokens)", (100, 900), [200, 400, 600, 800], 1.0),
            ("actor/kl_loss", "How far does the writer move from its start?", "KL term to the initial writer", (0, 0.09), [0, 0.03, 0.06, 0.09], 1.0))
    rows = (("rep", "standard reward", STD), ("oc", "reward with omission penalty", PEN))
    xs = ((0.62, 3.12), (4.04, 6.53))
    ys = ((1.92, 3.02), (0.42, 1.52))
    for ci, (key, q, ylab, ylim, yt, sc) in enumerate(cols):
        title(fig, xs[ci][0] - 0.52, H - 0.14, "ab"[ci], q, H)
        for ri, (arm, rlab, col) in enumerate(rows):
            ax = fig.add_axes(box(xs[ci][0], ys[ri][0], xs[ci][1], ys[ri][1], H))
            style(ax)
            ends = []
            for s, ls, run in (("s101", "-", "run 1"), ("s202", "--", "run 2"), ("s303", ":", "run 3")):
                st = [r for r in d[f"{arm}_{s}"]["steps"] if key in r]
                x = np.array([r["step"] for r in st]); y = smooth(np.array([r[key] for r in st]) * sc)
                ax.plot(x, y, color=col, lw=1.3, ls=ls, zorder=2)
                ends.append([y[-1], run, y[-1]])
            # direct labels at the line ends, pushed apart so that none overlaps (min gap = 9% of the axis range)
            gap = 0.09 * (ylim[1] - ylim[0])
            ends.sort(key=lambda e: e[0])
            for k in range(1, len(ends)):
                ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
            for yv, run, y0 in ends:   # a short leader line where a label was pushed off its line end (round 3, outside)
                if abs(yv - y0) > 0.01 * (ylim[1] - ylim[0]):
                    ax.plot([40.2, 40.9], [y0, yv], color=col, lw=0.5, clip_on=False, zorder=2)
                ax.text(41.1, yv, run, color=col, fontsize=7.5, va="center", ha="left")
            ax.set_xlim(0, 40); ax.set_xticks([0, 10, 20, 30, 40]); ax.set_ylim(*ylim); ax.set_yticks(yt)
            if ri == 1:
                ax.set_xlabel("training step", fontsize=FS, labelpad=1)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(ylab if ri == 0 else "", fontsize=FS, labelpad=2)
            ax.text(0.8, ylim[1] - (ylim[1] - ylim[0]) * 0.08, rlab, color=col, fontsize=FS, va="top", ha="left",
                    fontweight="bold")
    save(fig, "figS1_training")


# ------------------------------------------------------------------------------------------------ S2
def fig_s2():
    pops = (("Validation pairs, 237", json.loads((ART / "replication" / "REPLICATION_FINAL_endpoints.json").read_text(encoding="utf-8"))),
            ("Held-out test pairs, 273", json.loads((ART / "replication" / "LOCKED_FINAL_endpoints.json").read_text(encoding="utf-8"))))
    readers = (("flag_reader", "rule-based reader", SLATE, "s"), ("independent_judge", "independent judge", TEAL, "^"),
               ("training_judge", "training judge", GREYJ, "o"))
    mets = (("dY", "Did discrimination improve?", "change in $Y$ (%)", (-0.06, 0.26), [0, 0.1, 0.2], 0.05, "gain criterion"),
            ("dFS", "Did false strikes rise?", "change in false-strike rate (%)", (-0.10, 0.30), [-0.1, 0, 0.1, 0.2, 0.3], 0.05, "strike margin"))
    H = 4.65
    fig = plt.figure(figsize=(W, H))
    xs = ((1.55, 4.00), (4.30, 6.75))
    ys = ((2.74, 4.29), (0.57, 2.09))
    labels = []
    for (rk, rl, _, _) in readers:
        for lab in ("run 1", "run 2", "run 3", "mean"):
            labels.append(f"{rl}, {lab}")
    for mi, (mk_, q, xl, xlim, xt, crit, critlab) in enumerate(mets):
        title(fig, 0.08, ys[mi][1] + (0.24 if mi == 0 else 0.18), "ab"[mi], q, H)
        for pi, (pname, pdat) in enumerate(pops):
            ax = fig.add_axes(box(xs[pi][0], ys[mi][0], xs[pi][1], ys[mi][1], H))
            style(ax, "x")
            y = 0
            yt = []
            for (rk, rl, col, m) in readers:
                r = pdat["readers"][rk]
                for s in ("101", "202", "303"):
                    v = r[f"{mk_}_{s}"]
                    ax.plot([v["lo95"], v["hi95"]], [y, y], color=col, lw=0.9, alpha=0.8, zorder=2)
                    ax.plot([v["point"]], [y], ls="none", marker=m, ms=4, mfc="white", mec=col, mew=1.0, zorder=3)
                    yt.append(y); y -= 1
                v = r[mk_]
                if mk_ == "dY":
                    ax.plot([v["lo95"], v["hi95"]], [y, y], color=col, lw=1.6, zorder=2)
                else:
                    ax.plot([v["point"], v["upper95_one_sided"]], [y, y], color=col, lw=1.6, zorder=2)
                    ax.plot([v["upper95_one_sided"]] * 2, [y - 0.25, y + 0.25], color=col, lw=1.2, zorder=2)
                ax.plot([v["point"]], [y], ls="none", marker=m, ms=5.5, mfc=col, mec=col, zorder=3)
                yt.append(y); y -= 1.6
            ax.axvline(0, color=ZERO, lw=0.8, zorder=1)
            ax.axvline(crit, color="#c0392b", lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.set_xlim(*xlim); pct_axis(ax, "x", xt)
            ax.set_yticks(yt); ax.set_yticklabels(labels if pi == 0 else [], fontsize=7.5)
            ax.set_ylim(y + 1.0, 0.8)
            if mi == 0:
                ax.set_title(pname, fontsize=FS, color=INK, pad=3)
            ax.set_xlabel(xl, fontsize=FS, labelpad=1)
    fig.text(0.5, 0.08 / H, "open: each run with its 95% interval     filled: mean of the 3 runs     "
             "red dashed line: the registered 5% gain criterion (a) and false-strike margin (b)",
             ha="center", va="center", fontsize=7.5, color=SUB)
    save(fig, "figS2_perrun")


# ------------------------------------------------------------------------------------------------ S3
def fig_s3():
    order = ("q8", "q30", "mg", "olmo", "phi")
    H = 4.15
    fig = plt.figure(figsize=(W, H))
    mets = (("Y_interface", "Does the form change the measured gain?", "form effect on the gain (%)", 1.0, (-0.08, 0.10), [-0.05, 0, 0.05, 0.1]),
            ("FS_fill", "Does writing absences lower false strikes?", "reduction in the false-strike rise (%)", -1.0, (-0.02, 0.09), [0, 0.03, 0.06, 0.09]))
    xs = ((1.55, 4.00), (4.30, 6.75))
    ys = ((2.55, 3.72), (0.50, 1.67))
    for mi, (key, q, xl, sign, xlim, xt) in enumerate(mets):
        title(fig, 0.08, ys[mi][1] + 0.30, "ab"[mi], q, H)
        for pi, (pk, pname) in enumerate((("val", "Validation pairs"), ("locked", "Held-out test pairs"))):
            ax = fig.add_axes(box(xs[pi][0], ys[mi][0], xs[pi][1], ys[mi][1], H))
            style(ax, "x")
            for j, jk in enumerate(order):
                lab, col, m = JUDGE[jk]
                v = json.loads((ART / "conv" / f"{jk}_{pk}" / "orderfill.json").read_text(encoding="utf-8"))["values"]
                a = sign * v[f"training_judge_{key}"]["point"]      # default instruction
                b = sign * v[f"independent_judge_{key}"]["point"]   # stated convention
                y = -j
                ax.plot([a, b], [y, y], color=col, lw=1.0, zorder=2)
                ax.plot([a], [y], ls="none", marker=m, ms=5, mfc="white", mec=col, mew=1.1, zorder=3)
                ax.plot([b], [y], ls="none", marker=m, ms=5, mfc=col, mec=col, zorder=3)
            ax.axvline(0, color=ZERO, lw=0.8, zorder=1)
            ax.set_xlim(*xlim); pct_axis(ax, "x", xt)
            ax.set_yticks([-j for j in range(len(order))])
            ax.set_yticklabels([JUDGE[jk][0] for jk in order] if pi == 0 else [], fontsize=FS)
            ax.set_ylim(-len(order) + 0.4, 0.6)
            if mi == 0:
                ax.set_title(pname, fontsize=FS, color=INK, pad=3)
            ax.set_xlabel(xl, fontsize=FS, labelpad=1)
    fig.text(0.5, 0.06 / H, "open marker: default instruction      filled marker: instruction states that unlisted findings are absent",
             ha="center", va="center", fontsize=7.5, color=SUB)
    save(fig, "figS3_convention")


# ------------------------------------------------------------------------------------------------ S4
def fig_s4():
    d = json.loads((ART / "fig3" / "fig3_data_replication.json").read_text(encoding="utf-8"))["reads"]
    h = json.loads((ART / "listed_hist" / "listed_hist.json").read_text(encoding="utf-8"))
    SHORT = {"Pleural Effusion": "pleural effusion", "Consolidation": "consolidation", "Edema": "edema",
             "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Pneumothorax": "pneumothorax",
             "Fracture": "fracture", "Lung Lesion": "lung lesion"}
    H = 5.1
    fig = plt.figure(figsize=(W, H))
    # (a) per-finding target-flag rates
    title(fig, 0.08, H - 0.14, "a", "Which findings did the writer learn to flag?", H)
    base = d["base"]["per_finding"]
    order = sorted(base, key=lambda f: -base[f]["pos"])
    lanes = (("tpr", "flag set on report-positive images (%)", "detection, higher = better", 1.75, 4.05),
             ("fpr", "flag set on report-negative images (%)", "false alarms, lower = better", 4.40, 6.70))
    for li, (key, lab, head, x0, x1) in enumerate(lanes):
        ax = fig.add_axes(box(x0, 2.75, x1, 4.55, H))
        style(ax, "x")
        ax.set_title(head, fontsize=FS, color=INK, pad=3)
        for i, f in enumerate(order):
            b = base[f][key]; m = float(np.mean([d[s]["per_finding"][f][key] for s in ("s101", "s202", "s303")]))
            ax.plot([b, m], [i, i], color=SLATE, lw=1.1, zorder=2)
            ax.plot([b], [i], ls="none", marker="s", ms=5.5, mfc="white", mec=SLATE, mew=1.1, zorder=3)
            ax.plot([m], [i], ls="none", marker="s", ms=4.5, mfc=SLATE, mec=SLATE, zorder=4)
        ax.set_xlim(-0.04, 1.04); pct_axis(ax, "x", [0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(len(order) - 0.4, -0.6); ax.set_yticks(range(len(order)))
        ax.set_yticklabels(["%s (%d)" % (SHORT[f], base[f]["pos"]) for f in order] if li == 0 else [], fontsize=FS)
        ax.set_xlabel(lab, fontsize=FS, labelpad=1)
    # (b) findings listed per record
    title(fig, 0.08, 2.18, "b", "How many of the 12 findings does each writer list?", H)
    rows = (("base", "untrained writer"), ("rep_s101", "standard reward, run 1"), ("rep_s202", "standard reward, run 2"),
            ("rep_s303", "standard reward, run 3"), ("oc_s101", "omission penalty, run 1"),
            ("oc_s202", "omission penalty, run 2"), ("oc_s303", "omission penalty, run 3"))
    M = np.array([[h[k]["hist"][str(c)] / h[k]["n_records"] for c in range(13)] for k, _ in rows])
    ax = fig.add_axes(box(1.75, 0.42, 5.55, 1.98, H))
    ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for c in range(13):
            if M[i, c] >= 0.005:
                ax.text(c, i, "%d" % round(M[i, c] * 100), ha="center", va="center", fontsize=7,
                        color="white" if M[i, c] > 0.5 else INK)
    ax.set_xticks(range(13)); ax.set_xticklabels([str(c) for c in range(13)], fontsize=FS)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=FS)
    ax.set_xlabel("findings listed in a record (share of 476 validation records, %)", fontsize=FS, labelpad=1)
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("#7a8189")
    ax.tick_params(length=0)
    fig.text(5.60 / W, 2.02 / H, "mean", fontsize=FS, color=INK, ha="left", va="center", fontweight="bold")
    fig.text(6.02 / W, 2.02 / H, "with omissions", fontsize=FS, color=INK, ha="left", va="center", fontweight="bold")
    for i, (k, _) in enumerate(rows):
        yy = 1.98 - (i + 0.5) * (1.56 / len(rows))
        fig.text(5.60 / W, yy / H, "%.2f" % h[k]["mean"], fontsize=FS, color=INK, ha="left", va="center")
        fig.text(6.02 / W, yy / H, "%.0f%%" % (h[k]["share_omitting_any"] * 100), fontsize=FS, color=INK, ha="left", va="center")
    save(fig, "figS4_records")


if __name__ == "__main__":
    setup()
    fig_s1(); fig_s2(); fig_s3(); fig_s4()
