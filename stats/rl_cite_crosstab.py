#!/usr/bin/env python3
"""Cite variant, by whether the record ACTUALLY lists the sentence's finding (descriptive; declared with the cite
readout 2026-09-30 23:34 EDT). Original form (readj) and the omissions-as-absent form (fill_only), both judges, both
populations, record + swap passes pooled. For rows whose target finding is omitted from the record: share the judge
says "not listed" vs quotes something, what it quotes (does the quote name the target finding? another finding?),
the verdict split by claim direction (assert vs deny, from the claim text) and truth. Samples printed.
Usage: rl_cite_crosstab.py OUTDIR
"""
import os
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAIMBLIND_ROOT", "."))
SETS = ("base", "s101", "s202", "s303")
NEG = re.compile(r"\b(no|not|without|absent|negative|resolved|clear|free of|unremarkable|normal)\b", re.I)


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def entries(record):
    rec = record.get("record", record) if isinstance(record, dict) else record
    return [e for e in rec if isinstance(e, dict)] if isinstance(rec, list) else []


def main():
    out = ROOT / sys.argv[1]; out.mkdir(parents=True, exist_ok=True)
    res = {}; samples = collections.defaultdict(list); rng = random.Random(20261001)
    for J in ("q8", "mg"):
        for pop in ("val", "locked"):
            for form in ("readj", "fill_only"):
                c = collections.Counter()
                for st in SETS:
                    for pas, tk in (("record", "truth"), ("swapcached", "swap_truth")):
                        p = ROOT / f"results/eval2/judgeprompt/{J}_cite_{pop}/{st}/{form}/{pas}/rows.jsonl"
                        if not p.exists():
                            continue
                        for line in open(p, encoding="utf-8"):
                            if not line.strip():
                                continue
                            r = json.loads(line)
                            if not r.get("ok"):
                                continue
                            ents = entries(r.get("record"))
                            tfind = r.get("donor_finding") if pas == "swapcached" else r.get("finding", "")
                            tgt = norm(tfind or "")
                            listed = any(norm(e.get("finding", "")) == tgt for e in ents)
                            ev = str((r.get("judge_extra") or {}).get("evidence", "")).strip()
                            nl = "not listed" in ev.lower()
                            names_target = (tgt in norm(ev)) if (ev and tgt) else False
                            nc, ne = norm(r.get("claim", "")), norm(ev)
                            quotes_claim = bool(ne) and len(ne) >= 12 and (ne in nc or nc in ne)
                            other_names = [e.get("finding") for e in ents if norm(e.get("finding", "")) and norm(e.get("finding", "")) != tgt and norm(e.get("finding", "")) in norm(ev)]
                            denial = bool(NEG.search(r.get("claim", "")))
                            truth = r.get(tk); v = r.get("verdict")
                            key = "listed" if listed else "omitted"
                            c[f"{key}|n"] += 1
                            c[f"{key}|says_not_listed"] += nl
                            c[f"{key}|quotes_claim"] += quotes_claim
                            if not nl:
                                c[f"{key}|quote_names_target"] += names_target
                                c[f"{key}|quote_names_other_only"] += (not names_target and bool(other_names))
                                c[f"{key}|quote_names_none"] += (not names_target and not other_names)
                            d = "deny" if denial else "assert"
                            c[f"{key}|{d}|n"] += 1
                            c[f"{key}|{d}|accept"] += v == "supported"
                            if truth == "supported":
                                c[f"{key}|{d}|true_n"] += 1; c[f"{key}|{d}|false_strike"] += v == "unsupported"
                            elif truth == "unsupported":
                                c[f"{key}|{d}|false_n"] += 1; c[f"{key}|{d}|false_accept"] += v == "supported"
                            if key == "omitted" and form == "readj" and len(samples[(J, pop)]) < 400:
                                samples[(J, pop)].append({"finding": tfind, "claim": r.get("claim"), "evidence": ev[:200],
                                                          "verdict": v, "truth": truth, "listed_findings": [e.get("finding") for e in ents]})
                res[f"{J}|{pop}|{form}"] = dict(c)
    (out / "cite_crosstab.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    def pc(a, b):
        return "n/a" if not b else f"{100.0 * a / b:.1f}"

    lines = ["| judge | pop | form | target omitted, share of rows | evidence = the sentence itself, omitted | evidence = the sentence itself, listed | says not listed when omitted | quote names target when omitted | quote names another finding only | says not listed when listed | accept assert when omitted | accept deny when omitted | false strike deny when omitted | false strike deny when listed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in res.items():
        J, pop, form = k.split("|")
        no, li = c.get("omitted|n", 0), c.get("listed|n", 0)
        q_no = no - c.get("omitted|says_not_listed", 0)
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            J, pop, form, pc(no, no + li), pc(c.get("omitted|quotes_claim", 0), no), pc(c.get("listed|quotes_claim", 0), li), pc(c.get("omitted|says_not_listed", 0), no),
            pc(c.get("omitted|quote_names_target", 0), q_no), pc(c.get("omitted|quote_names_other_only", 0), q_no),
            pc(c.get("listed|says_not_listed", 0), li),
            pc(c.get("omitted|assert|accept", 0), c.get("omitted|assert|n", 0)), pc(c.get("omitted|deny|accept", 0), c.get("omitted|deny|n", 0)),
            pc(c.get("omitted|deny|false_strike", 0), c.get("omitted|deny|true_n", 0)), pc(c.get("listed|deny|false_strike", 0), c.get("listed|deny|true_n", 0))))
    (out / "cite_crosstab.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for (J, pop), s in samples.items():
        if pop != "val":
            continue
        print(f"\n--- samples {J} {pop}, target omitted, original form ---")
        for x in rng.sample(s, min(6, len(s))):
            print(json.dumps(x, ensure_ascii=False)[:420])


if __name__ == "__main__":
    main()
