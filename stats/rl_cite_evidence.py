#!/usr/bin/env python3
"""Descriptive readout of the 'cite' judge-prompt variant (declared 2026-09-30 23:34 EDT, post hoc): what evidence does
the judge say it used? For every parsed verdict of the cite reads, the judge's "evidence" field is classed as
  not_listed : the judge says the record has no entry for the sentence's finding
  cited      : the judge quotes an entry
  other      : field missing or unclassifiable
and crossed with whether the record ACTUALLY lists the target finding (an entry whose finding matches), the form, the
verdict, and the truth of the claim (record pass: truth; swap pass: swap_truth). Also the false-strike proxy (true claim
judged unsupported) and false-acceptance proxy (false claim judged supported) by evidence class.
Usage: rl_cite_evidence.py OUTDIR  (reads results/eval2/judgeprompt/{q8,mg}_cite_{val,locked}/{base,s101,s202,s303}/{form}/{record,swapcached}/rows.jsonl)
"""
import os
import collections
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
FORMS = ("readj", "canon_readj", "order_only", "fill_only")
SETS = ("base", "s101", "s202", "s303")


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def target_listed(record, finding):
    rec = record.get("record", record) if isinstance(record, dict) else record
    if not isinstance(rec, list):
        return None
    f = norm(finding)
    return any(norm(e.get("finding", "")) == f for e in rec if isinstance(e, dict))


def classify(extra):
    if not isinstance(extra, dict) or "evidence" not in extra:
        return "other"
    e = str(extra["evidence"]).strip().lower()
    if not e:
        return "other"
    if "not listed" in e or e in ("none", "no entry", "not present in record", "not in record"):
        return "not_listed"
    return "cited"


def main():
    out = ROOT / sys.argv[1]; out.mkdir(parents=True, exist_ok=True)
    table = {}
    for J in ("q8", "mg"):
        for pop in ("val", "locked"):
            for st in SETS:
                for form in FORMS:
                    for pas, truth_key in (("record", "truth"), ("swapcached", "swap_truth")):
                        p = ROOT / f"results/eval2/judgeprompt/{J}_cite_{pop}/{st}/{form}/{pas}/rows.jsonl"
                        if not p.exists():
                            continue
                        c = collections.Counter(); cross = collections.Counter()
                        n = 0
                        for line in open(p, encoding="utf-8"):
                            if not line.strip():
                                continue
                            r = json.loads(line)
                            if not r.get("ok"):
                                c["unparsed"] += 1; continue
                            n += 1
                            cls = classify(r.get("judge_extra"))
                            listed = target_listed(r.get("record"), r.get("finding", ""))
                            truth = r.get(truth_key); verdict = r.get("verdict")
                            c[cls] += 1
                            c[f"{cls}|target_{'listed' if listed else 'omitted' if listed is False else 'unknown'}"] += 1
                            if truth in ("supported", "unsupported"):
                                fs = truth == "supported" and verdict == "unsupported"
                                fa = truth == "unsupported" and verdict == "supported"
                                cross[f"{cls}|n_truth_known"] += 1
                                cross[f"{cls}|false_strike"] += int(fs)
                                cross[f"{cls}|false_accept"] += int(fa)
                                cross[f"{cls}|{r.get('direction', '')}|verdict={verdict}|truth={truth}"] += 1
                        key = f"{J}|{pop}|{st}|{form}|{pas}"
                        table[key] = {"n_parsed": n, "classes": dict(c), "by_class": dict(cross)}
    (out / "cite_evidence.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    # compact markdown: per judge x pop x form (pooled over sets and passes)
    lines = ["| judge | pop | form | n | not_listed % | cited % | FS% when not_listed | FS% when cited | FA% when not_listed | FA% when cited |", "|---|---|---|---|---|---|---|---|---|---|"]
    agg = collections.defaultdict(collections.Counter)
    for key, v in table.items():
        J, pop, st, form, pas = key.split("|")
        a = agg[(J, pop, form)]
        a["n"] += v["n_parsed"]
        for k, x in v["classes"].items():
            a[k] += x
        for k, x in v["by_class"].items():
            a[k] += x

    def pct(num, den):
        return "n/a" if not den else f"{100.0 * num / den:.1f}"

    for (J, pop, form), a in sorted(agg.items()):
        n = a["n"] or 1
        lines.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            J, pop, form, a["n"], pct(a["not_listed"], n), pct(a["cited"], n),
            pct(a["not_listed|false_strike"], a["not_listed|n_truth_known"]), pct(a["cited|false_strike"], a["cited|n_truth_known"]),
            pct(a["not_listed|false_accept"], a["not_listed|n_truth_known"]), pct(a["cited|false_accept"], a["cited|n_truth_known"])))
    (out / "cite_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", out / "cite_evidence.json")


if __name__ == "__main__":
    main()
