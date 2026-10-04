#!/usr/bin/env python3
"""Second domain: turn the frozen analysis.json into (1) report.md, a plain summary that maps the result onto the
registered result table (protocol section 8), and (2) numbers.json, every number the paper may quote, as percent strings
with exact half-up rounding from the stored decimals. Reads only; never recomputes an estimate.
Usage: d2_report.py ANALYSIS_JSON OUT_DIR
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

Q8, GEM = "Qwen/Qwen3-VL-8B-Instruct", "google/gemma-3-12b-it"
SHORT = {Q8: "Qwen3-VL-8B", GEM: "Gemma-3-12B", "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B-A3B",
         "microsoft/Phi-4-mini-instruct": "Phi-4-mini", "allenai/OLMo-2-1124-7B-Instruct": "OLMo-2-7B",
         "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B"}


def pct(x, nd=1):
    if x is None:
        return "n/a"
    return str(Decimal(repr(float(x) * 100)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)) + "%"


def est(r):
    if not r or not r.get("n"):
        return "no pairs"
    return "%s [%s, %s] on %d pairs" % (pct(r["point"]), pct(r["lo95"]), pct(r["hi95"]), r["n"])


def licensed_row(res):
    d0, e, crit = res["D0"]["pass"], res["E"]["pass"], res["gates"]["all_pass"]
    if (d0 or e) and not crit:
        return 5, "Effect only in comprehension-failing reads: an instruction-following failure, not the stronger claim"
    if d0 and e:
        return 1, "Omission conventions distort visual-writer comparisons, including under equivalent decoded records"
    if d0:
        return 2, "A broader omission-semantics case study; the 'adding absence changes meaning' objection remains"
    if e:
        return 3, "An equivalent-encoding failure, with the second-domain omission replication reported as unsuccessful"
    return 4, "The radiology measurement paper as it stands, with this study reported as a bounded negative"


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    A = json.loads(src.read_text())
    out.mkdir(parents=True, exist_ok=True)
    nums, lines = {}, ["# Second-domain study: frozen analysis summary", "", "Source: `%s`" % src, ""]
    for popname, res in A.items():
        tag = "primary" if popname.startswith("primary") else "secondary"
        row, text = licensed_row(res)
        lines += ["## %s population (%s)" % (tag.capitalize(), popname), "",
                  "Pool pairs %d; writer-complete %d; writer-missing %d %s %s." % (
                      res["population_pairs"], res["writer_complete_pairs"], res["writer_missing_pairs"],
                      json.dumps(res["writer_missing_by_category"]),
                      json.dumps({SHORT.get(k, k): v for k, v in res["writer_missing_by_writer"].items()})), ""]
        for kind, label in (("D0", "D0 (omission forms: absent minus unfilled, Qwen3-VL-8B judge minus Gemma judge)"),
                            ("E", "E (equivalent encodings: sparse minus complete, same judge contrast)")):
            r = res[kind]
            lines.append("- %s: %s; registered pass %s; clean %s (adverse judge-verdict bound %s / %s); judge-verdict "
                         "missing pairs %d." % (label, est(r), r["pass"], r["clean"], est(r["bound_low"]),
                                                est(r["bound_high"]), r["missing_pairs"]))
            dg = r["diagnostic_worst_case_incl_writer_missing"]
            lines.append("  Diagnostic worst case incl. writer-missing pairs: low %s, high %s." % (est(dg["low"]),
                                                                                                   est(dg["high"])))
            ls = r["label_sensitivity"]
            if "random_reversal" in ls:
                lines.append("  Label sensitivity: expected point at 5%% random reversal %s; greedy reversals to below 5%%: "
                             "%s; to a normal interval including 0: %s." % (
                                 pct(ls["random_reversal"]["0.05"]["expected"]),
                                 ls["greedy_reversals_to_below_threshold"],
                                 ls["greedy_reversals_to_normal_interval_including_zero"]))
            for f in ("point", "lo95", "hi95"):
                nums["%s.%s.%s" % (tag, kind, f)] = pct(r.get(f))
            nums["%s.%s.n" % (tag, kind)] = str(r.get("n"))
            nums["%s.%s.pass" % (tag, kind)] = str(r["pass"])
            nums["%s.%s.clean" % (tag, kind)] = str(r["clean"])
        g = res["gates"]
        comp = {k: v["agreement"] for k, v in g.items() if k.startswith("comprehension")}
        ctrl = {k: v["agreement"] for k, v in g.items() if k.startswith("control")}
        lines += ["", "- Criteria all pass: %s. Comprehension lowest %s; controls lowest %s (%d cells)." % (
            g["all_pass"], pct(min(comp.values())), pct(min(ctrl.values())), len(ctrl)),
            "- Combined claim: %s." % res["combined_claim"],
            "- Registered result-table row: %d, %s." % (row, text), ""]
        nums["%s.combined_claim" % tag] = str(res["combined_claim"])
        nums["%s.result_row" % tag] = str(row)
        nums["%s.writer_complete_pairs" % tag] = str(res["writer_complete_pairs"])
        # per-judge H and encoding response, sparse style, all judges (descriptive)
        pj = res["descriptive"]["per_judge"]
        lines += ["Per-judge responses, sparse writer style (descriptive):", "",
                  "| Judge | H = G(absent) - G(unfilled) | G(sparse enc) - G(complete enc) |", "|---|---|---|"]
        judges = sorted({k.split("|")[0] for k in pj})
        for j in judges:
            h = pj.get("%s|sparse|H_absent_minus_unfilled" % j)
            b = pj.get("%s|sparse|sparse_minus_complete_encoding" % j)
            lines.append("| %s | %s | %s |" % (SHORT.get(j, j), est(h), est(b)))
            if h:
                nums["%s.H.%s" % (tag, SHORT.get(j, j))] = pct(h["point"])
        lines.append("")
        cb = res["descriptive"]["category"]
        lines.append("Category-balanced: D0 %s; E %s." % (est(cb["D0"]["category_balanced"]),
                                                          est(cb["E"]["category_balanced"])))
        lines.append("Writer parse formats: %s" % json.dumps(res["descriptive"]["writer_parse_formats"]))
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "numbers.json").write_text(json.dumps(nums, indent=1, sort_keys=True), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
