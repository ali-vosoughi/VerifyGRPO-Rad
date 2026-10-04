#!/usr/bin/env python3
"""Figure 1, the teaser, v7 (2026-09-29): one question told as a story, "Does a blank mean no?".

The figure maps the problem to something familiar and stays numerically precise, without naming runs, procedures,
or analysis stages. The reduction used here:
  1. a note-taker fills in a form about the image before seeing the question   (the claim-blind record)
  2. the form has blanks                                                          (omitted findings)
  3. a blank can be read as "not stated" or as "no"                              (original vs reward-scoring form)
  4. which reading a judge uses changes how much learning it measures            (the between-judge form contrast)
Data: step 1-3 are the record and cached verdicts/reasons in artifacts/flip_example/flip_pick.json (the training
judge; the independent judge reads this record the same way, qual_examples.json Q1); step 4 plots the registered
validation gains in Y from artifacts/replication/REPLICATION_FINAL_decomp_payload_decomposition.json (237 pairs,
3-run means, exact k/711 recovery, = Figure 3a left facet). Every string is data or a label; type >= 7 pt at the
3.25 in column width. Coordinates are inches from the bottom left.
Inputs are read from $FIG_ROOT/artifacts/ (see the README, Figures). Usage: FIG_ROOT=<dir> python build_fig1_npj.py [out.pdf]
"""
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(os.environ.get("FIG_ROOT", "."))   # holds artifacts/ (inputs, read only) and figs/ (outputs)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figs" / "fig1_teaser.pdf"
W, H = 3.25, 2.72
FS = 7.0
INK = "#20252b"
SUB = "#59636e"
REASON = "#4b5563"
GREEN = "#2a7d45"
ORANGE = "#c05a1c"
BLUE = "#2166ac"
GREYJ = "#6b6f76"        # the judge that trained the model (Figure 3 colour)
TEAL = "#2a9d8f"         # the second judge (Figure 3 colour)
CARD = "#f4f6f8"
EDGE = "#b9c1c9"
SYM = "DejaVu Sans"      # Arial has no check or cross glyph


def pc(x, n):
    """Percent with 1 decimal from the exact fraction k/n behind a rounded decimal (no double rounding)."""
    k = round(x * n)
    q = Fraction(k * 1000, n)
    t = int(q) + (1 if q - int(q) >= Fraction(1, 2) else 0)
    return "%d.%d" % (t // 10, t % 10)


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, W); self.ax.set_ylim(0, H); self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()
        self.boxes = []

    def text(self, x, y, s, check=True, **kw):
        kw.setdefault("fontsize", FS); kw.setdefault("color", INK); kw.setdefault("va", "center"); kw.setdefault("zorder", 4)
        t = self.ax.text(x, y, s, **kw)
        if check:   # every label is kept for the overlap check at the end
            self.boxes.append((s, t))
        return t

    def ext(self, t):
        b = t.get_window_extent(self.r)
        return b.x0 / self.fig.dpi, b.y0 / self.fig.dpi, b.x1 / self.fig.dpi, b.y1 / self.fig.dpi

    def run(self, x, y, pieces, gap=0.05):
        for s, kw in pieces:
            t = self.text(x, y, s, ha="left", **kw)
            x = self.ext(t)[2] + gap
        return x - gap

    def card(self, x0, y0, x1, y1, fc=CARD, ec=EDGE, lw=0.6):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.03",
                                         fc=fc, ec=ec, lw=lw, zorder=1))

    def blank(self, x, y, w=0.30, h=0.085, ec=ORANGE):
        self.ax.add_patch(Rectangle((x, y - h / 2), w, h, fc="white", ec=ec, lw=0.8, ls=(0, (2, 1.2)), zorder=3))

    def step(self, x, y, n):
        self.ax.add_patch(plt.Circle((x, y), 0.062, fc=INK, ec="none", zorder=3))
        self.text(x, y, str(n), check=False, fontsize=FS, color="white", ha="center", fontweight="bold", zorder=5)

    def mark(self, x, y, right):
        self.text(x, y, "✓" if right else "✗", fontfamily=SYM, fontsize=FS + 1, color=GREEN if right else ORANGE,
                  ha="left", fontweight="bold")

    def arrow(self, p0, p1, color=INK):
        self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=0.9, zorder=3))

    def check_overlaps(self):
        ex = [(s, self.ext(t)) for s, t in self.boxes]
        bad = []   # collected first, reported together
        for i in range(len(ex)):
            a = ex[i][1]
            if not (a[0] >= -0.001 and a[2] <= W + 0.001 and a[1] >= -0.001 and a[3] <= H + 0.001):
                bad.append(("off canvas", ex[i][0]))
            for j in range(i + 1, len(ex)):
                b = ex[j][1]
                if a[0] < b[2] - 0.005 and b[0] < a[2] - 0.005 and a[1] < b[3] - 0.005 and b[1] < a[3] - 0.005:
                    bad.append((ex[i][0], ex[j][0]))
        assert not bad, "text overlaps: %s" % bad


def main():
    plt.rcParams.update({"font.size": FS, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.it": "Arial:italic"})
    fx = json.loads((HERE / "artifacts" / "flip_example" / "flip_pick.json").read_text(encoding="utf-8"))
    dec = json.loads((HERE / "artifacts" / "replication" / "REPLICATION_FINAL_decomp_payload_decomposition.json")
                     .read_text(encoding="utf-8"))["values"]
    assert [e["finding"] for e in fx["raw_entries"]] == ["Enlarged Cardiomediastinum"] and fx["raw_entries"][0]["present"]
    assert "Pleural Effusion" in fx["omitted"] and len(fx["omitted"]) == 11
    assert fx["verdict_as_written"] == "unsupported" and fx["verdict_training_form"] == "supported"
    assert "does not mention pleural effusions" in fx["reason_as_written"]
    assert "marked as absent in the record" in fx["reason_training_form"]
    q1 = json.loads((HERE / "artifacts" / "qual_examples" / "qual_examples.json").read_text(encoding="utf-8"))["Q1"]
    assert q1["pair_id"] == fx["pair_id"]           # the same record, read by the second judge
    assert q1["independent_judge"]["as_written"]["verdict"] == "unsupported"
    assert q1["independent_judge"]["training_form"]["verdict"] == "supported"
    assert "does not mention pleural effusions" in q1["independent_judge"]["as_written"]["reason"]
    C = Canvas()

    # ---------------------------------------------------------------- title
    C.text(0.04, H - 0.085, "Does “not mentioned” mean “absent”?", fontsize=FS + 2, fontweight="bold")

    # ---------------------------------------------------------------- 1  the model writes its record
    y1 = H - 0.265
    C.step(0.10, y1, 1)
    C.text(0.20, y1, "An AI model sees only the X-ray and fills in a checklist", color=INK)
    img_top = y1 - 0.10
    IH = 0.50; IW = IH * 512 / 503
    a = C.fig.add_axes([0.04 / W, (img_top - IH) / H, IW / W, IH / H])
    a.imshow(np.asarray(Image.open(HERE / "artifacts" / "flip_example" / "flip_true.png").convert("L")), cmap="gray",
             interpolation="lanczos", aspect="auto")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(EDGE); s.set_linewidth(0.8)
    rx0, rx1 = 0.68, 3.21
    CH = IH + 0.12                               # the card also holds the record's free-text note (review 2026-09-29)
    C.arrow((0.04 + IW + 0.025, img_top - IH / 2), (rx0 - 0.025, img_top - IH / 2))
    C.card(rx0, img_top - CH, rx1, img_top, fc="white")
    rows = (("enlarged cardiomediastinum", "present"), ("pleural effusion", None), ("10 other findings", None))
    for i, (f, v) in enumerate(rows):
        yy = img_top - 0.105 - 0.145 * i
        C.text(rx0 + 0.07, yy, f, color=INK if v else (ORANGE if i == 1 else SUB), fontweight="bold" if i == 1 else "normal")
        if v:
            C.text(2.62, yy, v, color=INK)
        else:
            C.blank(2.62, yy, w=0.46, h=0.09)
    note = q1["record_as_written"]["other"]
    assert note == "No other abnormalities are visible in this radiograph."
    t = C.text(rx0 + 0.06, img_top - 0.105 - 0.145 * 3, "“%s”" % note, color=REASON, style="italic")
    assert C.ext(t)[2] <= rx1 - 0.02, "note runs past the card"

    # ---------------------------------------------------------------- 2  the claim
    y2 = img_top - CH - 0.15
    C.step(0.10, y2, 2)
    C.run(0.20, y2, [("Sentence", {}), ("“%s”" % fx["sentence"], {"fontsize": FS + 0.5, "style": "italic", "fontweight": "bold"})],
          gap=0.08)
    C.text(3.21, y2, "matches report-derived label", color=GREEN, ha="right")

    # ---------------------------------------------------------------- 3  two versions of the same record
    y3 = y2 - 0.165
    C.step(0.10, y3, 3)
    C.text(0.20, y3, "2 AI checkers read 2 versions of the same checklist", color=INK)
    ctop, cbot = y3 - 0.095, y3 - 0.575
    cards = ((0.04, 1.58, "as generated", ORANGE, None, "both checkers reject", ("“does not mention", "pleural effusions”")),
             (1.67, 3.21, "training format", BLUE, "fixed order; blanks = “absent”", "both checkers accept",
              ("“marked as absent”",)))
    for x0, x1, title, col, note, verdict, reason in cards:
        C.card(x0, cbot, x1, ctop, fc=CARD)
        yy = ctop - 0.085
        C.text(x0 + 0.07, yy, title, fontweight="bold", color=col)
        if note:
            yy -= 0.105
            C.text(x0 + 0.07, yy, note, color=SUB)
        yy -= 0.115
        C.text(x0 + 0.07, yy, verdict, fontweight="bold")
        for line in reason:
            yy -= 0.105
            C.text(x0 + 0.07, yy, line, color=REASON, style="italic")
        assert yy - 0.05 >= cbot, "card text runs below the card"

    # ---------------------------------------------------------------- 4  the measured gains move apart
    y4 = cbot - 0.13
    C.step(0.10, y4, 4)
    C.text(0.20, y4, "Gain from training in Y, over 237 validation pairs", color=INK)
    n = 711
    tr = (dec["training_judge_Y_d_raw"]["point"], dec["training_judge_Y_d_canonical"]["point"])
    ind = (dec["independent_judge_Y_d_raw"]["point"], dec["independent_judge_Y_d_canonical"]["point"])
    xa, xb, xc = 1.88, 2.62, 3.08                     # centres of the "blank" and "absent" numbers, and the arrow
    yh = y4 - 0.13
    C.text(xa, yh, "as generated", color=ORANGE, ha="center")
    C.text(xb, yh, "training format", color=BLUE, ha="center")
    for k, ((a0, a1), col, mk, lab) in enumerate(((tr, GREYJ, "o", "checker used in training"),
                                                   (ind, TEAL, "^", "independent medical checker"))):
        yy = yh - 0.16 - 0.165 * k
        C.ax.plot([0.26], [yy], ls="none", marker=mk, ms=4.5, mfc=col, mec=col, zorder=3)
        C.text(0.36, yy, lab, color=col)
        C.text(xa, yy, "%s%%" % pc(a0, n), color=col, ha="center", fontsize=FS + 1.5, fontweight="bold")
        C.arrow((xa + 0.25, yy), (xb - 0.25, yy), color=col)
        C.text(xb, yy, "%s%%" % pc(a1, n), color=col, ha="center", fontsize=FS + 1.5, fontweight="bold")
        C.text(xc, yy, "↑ up" if a1 > a0 else "↓ down", color=col, ha="center", fontweight="bold")
    C.text(3.21, 0.075, "Checklist format moves the measured gain.", fontsize=FS + 1.5, fontweight="bold", ha="right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    C.fig.savefig(OUT, dpi=400)
    C.fig.savefig(OUT.with_suffix(".png"), dpi=300)
    C.check_overlaps()
    print("wrote", OUT, "values", [pc(v, n) for v in tr + ind])


if __name__ == "__main__":
    main()
