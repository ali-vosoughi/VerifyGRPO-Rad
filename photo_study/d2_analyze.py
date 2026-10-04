#!/usr/bin/env python3
"""Second domain: the frozen analysis (protocol section 7). Runs once on the judge rows. Version 3, after the pre-freeze
red-team and its recheck (outside reviewers, 2026-09-30).

Population: the frozen pool. PRIMARY = all 516 pool pairs; SECONDARY = the pairs in screen_pass.txt. Every population
pair is accounted for, and a pair with no row counts as missing. Rows are keyed by (pair, writer, judge, STYLE, arm,
polarity, role); a duplicate key or a role / polarity outside the 4 registered slots is an error. The primary contrasts
use the sparse style; checklist is reported apart.

Y(w, j, st, f) = mean over pairs of the mean over polarities of [accept on the true image] - [accept on the false
image]; G = Y(newer writer) - Y(older writer). D0 = H(q8) - H(gemma) with H = G(absent) - G(unfilled);
E = [G(q8, sparse enc) - G(q8, complete enc)] - [G(gemma, sparse enc) - G(gemma, complete enc)]. Complete-case estimate
on the pairs where every needed verdict parses. EVERY interval in this file uses 10,000 whole-pair bootstrap draws
(seed 20260930) and the two-sided 95% percentile interval; a primary passes when the interval excludes 0 and
|point| >= 5%. Missing-verdict bounds: every missing verdict of every population pair is filled with its least
(bound_low) and most (bound_high) favourable value FOR THE SIGN OF ITS TERM (a positive-weight term pushed down gets
true-image reject and false-image accept; a negative-weight term the reverse), each bound with its own interval. A
primary result is CLEAN only if the bound adverse to its sign preserves its decision (passes, same sign).
Criteria (registered): comprehension >= 90% per primary judge x encoding on the 48 synthetic cases of each encoding, and
target controls >= 90% agreement with the record-implied verdict per primary judge x writer x encoding x control (sparse
writer style), over the 4 slots of every population pair, a missing verdict or record counting as a failure. COMBINED
claim = D0 and E pass, both clean, every criterion passes. Label-error sensitivity (diagnostic, not a criterion):
expected estimate under random pair reversal at 2 / 5 / 10%, and a GREEDY count of worst-case reversals (largest
same-sign contributions first, starting at 0 reversals) until the point drops below 5% and until a normal-approximation
95% interval includes 0; a greedy count, not a proven minimum.
Usage: d2_analyze.py --judge-rows R1,R2,... --writer-rows W1,... --pool POOL --labels LABELS --screen-pass FILE
       --screen-rows SCREEN_JSONL --out DIR
       d2_analyze.py --selftest
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

NEW, OLD = "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"
Q8, GEM = "Qwen/Qwen3-VL-8B-Instruct", "google/gemma-3-12b-it"
PRIMARY_JUDGES = (Q8, GEM)
B, SEED, THRESH, GATE, N_COMP = 10000, 20260930, 0.05, 0.90, 48
POLS = (("assert", "yes", "no"), ("deny", "no", "yes"))    # polarity, true-image role, false-image role
CTRL_ARMS = ("C_corr_complete", "C_corr_sparse", "C_rev_complete", "C_rev_sparse")
ARMS_BY_STYLE = {"sparse": ("A_unfilled", "A_absent", "A_notreported", "B_complete", "B_sparse") + CTRL_ARMS
                 + ("T_complete", "T_sparse"),
                 "checklist": ("A_unfilled", "A_absent", "A_notreported")}


def index(rows):
    """acc[(pair, writer, judge, style, arm, polarity, role)] = 1 accept / 0 reject / None unparsed."""
    acc = {}
    for r in rows:
        if r["pair_id"] == "synthetic":
            continue
        if r["role"] not in ("yes", "no") or r["polarity"] not in ("assert", "deny"):
            raise ValueError("row outside the 4 registered slots %s %s %s" % (r["pair_id"], r["role"], r["polarity"]))
        key = (r["pair_id"], r["writer"], r["judge"], r["style"], r["arm"], r["polarity"], r["role"])
        if key in acc:
            raise ValueError("duplicate study row %s" % (key,))
        acc[key] = None if not r["ok"] else int(r["verdict"] == "supported")
    return acc


def pair_y(acc, pid, w, j, st, arm, fill=None):
    """Pair-level Youden contribution; fill = (value for a missing true-image verdict, for a missing false one)."""
    vals = []
    for pol, tr, fr in POLS:
        t, f = acc.get((pid, w, j, st, arm, pol, tr)), acc.get((pid, w, j, st, arm, pol, fr))
        if t is None or f is None:
            if fill is None:
                return None
            t = fill[0] if t is None else t
            f = fill[1] if f is None else f
        vals.append(t - f)
    return sum(vals) / len(vals)


def terms(kind, st="sparse"):
    """(weight, writer, judge, style, arm) of a primary contrast."""
    if kind == "D0":
        f1, f2 = "A_absent", "A_unfilled"
    elif kind == "E":
        f1, f2 = "B_sparse", "B_complete"
    else:
        raise ValueError(kind)
    return [(+1, NEW, Q8, st, f1), (-1, OLD, Q8, st, f1), (-1, NEW, Q8, st, f2), (+1, OLD, Q8, st, f2),
            (-1, NEW, GEM, st, f1), (+1, OLD, GEM, st, f1), (+1, NEW, GEM, st, f2), (-1, OLD, GEM, st, f2)]


def per_pair(acc, pids, tms, bound=None):
    """bound None = complete case; 'low' / 'high' = fill every missing verdict to push the contrast down / up."""
    out = {}
    for pid in pids:
        s, ok = 0.0, True
        for wgt, w, j, st, arm in tms:
            fill = None
            if bound is not None:
                down = (bound == "low") == (wgt > 0)        # this term's y must be pushed down
                fill = (0, 1) if down else (1, 0)
            y = pair_y(acc, pid, w, j, st, arm, fill)
            if y is None:
                ok = False
                break
            s += wgt * y
        if ok:
            out[pid] = s
    return out


def _pct(means, pt, n):
    means = np.sort(means)
    lo, hi = float(means[int(0.025 * B)]), float(means[int(0.975 * B) - 1])
    return {"n": n, "point": pt, "lo95": lo, "hi95": hi, "pass": bool((lo > 0 or hi < 0) and abs(pt) >= THRESH)}


def boot(vals, seed=SEED):
    """Whole-pair bootstrap, B draws, percentile interval."""
    v = np.asarray(list(vals.values()), dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0, "pass": False}
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for s in range(0, B, 500):
        m = min(500, B - s)
        means[s:s + m] = v[rng.integers(0, n, size=(m, n))].mean(axis=1)
    return _pct(means, float(v.mean()), n)


def boot_balanced(vals, cat, seed=SEED):
    """Category-balanced mean (each category weighted equally), bootstrap resampling pairs within category."""
    by = defaultdict(list)
    for pid, x in vals.items():
        by[cat[pid]].append(x)
    if not by:
        return {"n": 0, "pass": False}
    rng = np.random.default_rng(seed)
    means = np.zeros(B)
    for c in sorted(by):
        v = np.asarray(by[c], dtype=float)
        for s in range(0, B, 500):
            m = min(500, B - s)
            means[s:s + m] += v[rng.integers(0, len(v), size=(m, len(v)))].mean(axis=1)
    means /= len(by)
    pt = float(np.mean([np.mean(by[c]) for c in by]))
    r = _pct(means, pt, len(vals))
    r["categories"] = len(by)
    return r


def greedy_reversal(pp):
    """Label-error diagnostic. Reversing a pair (both labels wrong swaps the true and the false image) negates its
    contribution."""
    v = list(pp.values())
    n = len(v)
    if n < 2:
        return {"n_pairs": n, "note": "fewer than 2 pairs"}
    pt = sum(v) / n
    rng = random.Random(SEED)
    rand = {}
    for p in (0.02, 0.05, 0.10):
        sims = [sum((-x if rng.random() < p else x) for x in v) / n for _ in range(2000)]
        rand["%.2f" % p] = {"expected": (1 - 2 * p) * pt, "simulated_mean": sum(sims) / 2000}

    def state(xs):
        m = sum(xs) / n
        se = (sum((x - m) ** 2 for x in xs) / (n - 1) / n) ** 0.5
        return abs(m) < THRESH, (m - 1.96 * se) <= 0 <= (m + 1.96 * se)

    below, zero = state(v)
    to_thresh, to_zero = (0 if below else None), (0 if zero else None)
    sign = 1 if pt >= 0 else -1
    w = list(v)
    for k, i in enumerate(sorted(range(n), key=lambda i: -sign * v[i]), 1):
        if to_thresh is not None and to_zero is not None:
            break
        w[i] = -w[i]
        below, zero = state(w)
        if to_thresh is None and below:
            to_thresh = k
        if to_zero is None and zero:
            to_zero = k
    return {"n_pairs": n, "point": pt, "random_reversal": rand, "greedy_reversals_to_below_threshold": to_thresh,
            "greedy_reversals_to_normal_interval_including_zero": to_zero}


def gates(rows, pop):
    """Comprehension and control criteria. Study rows must have passed index() (4 valid slots, no duplicates)."""
    out, allpass = {}, True
    for j in PRIMARY_JUDGES:
        for enc in ("complete", "sparse"):
            rs = [r for r in rows if r["judge"] == j and r["arm"] == "COMP_" + enc]
            if len({r["k"] for r in rs}) != len(rs) or len(rs) > N_COMP:
                raise ValueError("duplicate or surplus comprehension rows %s %s" % (j, enc))
            agree = sum(r["verdict"] is not None and r["verdict"] == r["record_implied"] for r in rs) / N_COMP
            out["comprehension|%s|%s" % (j, enc)] = {"rows": len(rs), "expected": N_COMP, "agreement": agree,
                                                     "pass": agree >= GATE}
            allpass &= agree >= GATE
        for w in (NEW, OLD):
            for arm in CTRL_ARMS:
                rs = [r for r in rows if r["judge"] == j and r["writer"] == w and r["arm"] == arm
                      and r["style"] == "sparse" and r["pair_id"] in pop]
                slots = {(r["pair_id"], r["polarity"], r["role"]) for r in rs}
                if len(slots) != len(rs):
                    raise ValueError("duplicate control slots %s %s %s" % (j, w, arm))
                expected = 4 * len(pop)                          # 2 images x 2 polarities per population pair
                ok = sum(r["verdict"] is not None and r["verdict"] == r["record_implied"] for r in rs)
                agree = ok / expected if expected else 0.0
                out["control|%s|%s|%s" % (j, w, arm)] = {"rows": len(rs), "expected": expected, "agreement": agree,
                                                         "pass": agree >= GATE}
                allpass &= agree >= GATE
    out["all_pass"] = bool(allpass)
    return out


def image_keys(p):
    s = p["split"].replace("2014", "")
    return "%s_%d" % (s, p["yes_image_id"]), "%s_%d" % (s, p["no_image_id"])


def descriptive(rows, acc, pop, pool, writer_rows, labels, screen_rows=None):
    rep, popset = {}, set(pop)
    cat = {p["pair_id"]: p["category"] for p in pool}
    study = [r for r in rows if r["pair_id"] != "synthetic" and r["pair_id"] in popset]
    judges = sorted({r["judge"] for r in study} | set(PRIMARY_JUDGES))
    # row accounting against the expected slots: 2 rows per pair per truth state per (writer, judge, style, arm)
    pc = defaultdict(lambda: [0, 0])
    for r in study:
        c = pc[(r["writer"], r["judge"], r["style"], r["arm"], r["truth"])]
        c[0] += 1
        c[1] += int(not r["ok"])
    counts = {}
    for j in judges:
        for w in (NEW, OLD):
            for st, arms in ARMS_BY_STYLE.items():
                for arm in arms:
                    for truth in ("supported", "unsupported"):
                        got, bad = pc.get((w, j, st, arm, truth), (0, 0))
                        counts["%s|%s|%s|%s|%s" % (w, j, st, arm, truth)] = {
                            "expected": 2 * len(pop), "rows": got, "unparsed": bad, "absent_rows": 2 * len(pop) - got}
    rep["row_accounting"] = counts
    # per-judge H and encoding response, both writer styles; displayed Ys on exactly the pairs of G
    per = {}
    for j in judges:
        for st in ("sparse", "checklist"):
            for name, (f1, f2) in (("H_absent_minus_unfilled", ("A_absent", "A_unfilled")),
                                   ("sparse_minus_complete_encoding", ("B_sparse", "B_complete"))):
                pp = per_pair(acc, pop, [(+1, NEW, j, st, f1), (-1, OLD, j, st, f1), (-1, NEW, j, st, f2),
                                         (+1, OLD, j, st, f2)])
                if pp:
                    per["%s|%s|%s" % (j, st, name)] = boot(pp)
            for arm in ("A_unfilled", "A_absent", "A_notreported", "B_complete", "B_sparse"):
                g = per_pair(acc, pop, [(+1, NEW, j, st, arm), (-1, OLD, j, st, arm)])
                if not g:
                    continue
                yn = per_pair(acc, list(g), [(+1, NEW, j, st, arm)])
                yo = per_pair(acc, list(g), [(+1, OLD, j, st, arm)])
                per["%s|%s|%s|G" % (j, st, arm)] = {"n": len(g), "Y_new": sum(yn.values()) / len(yn),
                                                   "Y_old": sum(yo.values()) / len(yo), "G": boot(g)}
    rep["per_judge"] = per
    # false strikes (true claim rejected) and false acceptance (false claim accepted), by polarity
    fs = defaultdict(lambda: [0, 0, 0, 0])
    for r in study:
        if r["ternary"] or r["verdict"] not in ("supported", "unsupported"):
            continue
        c = fs["%s|%s|%s|%s|%s" % (r["writer"], r["judge"], r["style"], r["arm"], r["polarity"])]
        if r["truth"] == "supported":
            c[0] += 1; c[1] += int(r["verdict"] == "unsupported")
        else:
            c[2] += 1; c[3] += int(r["verdict"] == "supported")
    rep["polarity_rates"] = {k: {"false_strike": v[1] / v[0] if v[0] else None, "n_true": v[0],
                                 "false_acceptance": v[3] / v[2] if v[2] else None, "n_false": v[2]}
                             for k, v in sorted(fs.items())}
    # category summaries of the primary contrasts: per-category means and the category-balanced estimate
    cs = {}
    for kind in ("D0", "E"):
        pp = per_pair(acc, pop, terms(kind))
        by = defaultdict(list)
        for pid, v in pp.items():
            by[cat[pid]].append(v)
        cs[kind] = {"per_category": {c: {"n": len(v), "mean": sum(v) / len(v)} for c, v in sorted(by.items())},
                    "category_balanced": boot_balanced(pp, cat)}
    rep["category"] = cs
    # omission exposure and writer accuracy on the target object against the pool labels
    img = {}
    for p in pool:
        if p["pair_id"] in popset:
            ky, kn = image_keys(p)
            img[ky], img[kn] = p, p
    ox = defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # n present, omitted | present, n absent, omitted | absent, correct, unparsed
    fmt = defaultdict(lambda: defaultdict(int))     # writer-parser rule counts (list / dict layout R1 / cut-off prefix R2)
    for wr in writer_rows:
        p = img.get(wr["k"])
        if p is None:
            continue
        c = ox["%s|%s" % (wr["model"], wr["style"])]
        truth = labels[wr["k"]]
        f = wr["parsed"].get("format", "list") if wr["ok"] else (
            "missing_stopped_at_limit" if wr.get("finish_reason") == "length" else "missing_unparsed")
        fmt["%s|%s" % (wr["model"], wr["style"])][f] += 1
        if not wr["ok"]:
            c[5] += 1
            continue
        stt = wr["parsed"]["states"].get(p["category"])
        if truth == "present":
            c[0] += 1; c[1] += int(stt is None)
        else:
            c[2] += 1; c[3] += int(stt is None)
        c[4] += int((stt or "absent") == truth)          # closed-world decoding of the target
    rep["omission_and_accuracy"] = {k: {"omitted_when_present": v[1] / v[0] if v[0] else None,
                                        "omitted_when_absent": v[3] / v[2] if v[2] else None,
                                        "target_accuracy_closed_world": v[4] / (v[0] + v[2]) if v[0] + v[2] else None,
                                        "n_parsed_images": v[0] + v[2], "n_unparsed_images": v[5],
                                        "n_expected_images": 2 * len(pop)}
                                    for k, v in sorted(ox.items())}
    rep["writer_parse_formats"] = {k: dict(v) for k, v in sorted(fmt.items())}
    # decomposition: a deterministic reader F that accepts exactly what the sparse record implies (closed world);
    # judge G, reader G, and their residual on the IDENTICAL pairs of each judge x arm
    det = {}
    for w in (NEW, OLD):
        recs = {wr["k"]: wr for wr in writer_rows if wr["model"] == w and wr["style"] == "sparse"}
        det[w] = {}
        for p in pool:
            if p["pair_id"] not in popset:
                continue
            ky, kn = image_keys(p)
            ry, rn = recs.get(ky), recs.get(kn)
            if not (ry and rn and ry["ok"] and rn["ok"]):
                continue
            py = ry["parsed"]["states"].get(p["category"]) == "present"
            pn = rn["parsed"]["states"].get(p["category"]) == "present"
            det[w][p["pair_id"]] = ((int(py) - int(pn)) + (int(not pn) - int(not py))) / 2
    common = sorted(set(det[NEW]) & set(det[OLD]))
    gf = {pid: det[NEW][pid] - det[OLD][pid] for pid in common}
    dec = {}
    for j in PRIMARY_JUDGES:
        for arm in ("A_unfilled", "A_absent", "B_complete", "B_sparse"):
            g = per_pair(acc, common, [(+1, NEW, j, "sparse", arm), (-1, OLD, j, "sparse", arm)])
            if not g:
                continue
            gr = {pid: gf[pid] for pid in g}
            res = {pid: g[pid] - gr[pid] for pid in g}
            gj_b, gr_b, res_b = boot(g), boot(gr), boot(res)
            assert abs(gj_b["point"] - gr_b["point"] - res_b["point"]) < 1e-9, "decomposition must add up"
            dec["%s|%s" % (j, arm)] = {"n": len(g), "G_judge": gj_b, "G_deterministic_reader": gr_b,
                                       "residual_judge_minus_reader": res_b}
    rep["decomposition"] = dec
    # open-world ternary arm: agreement with the record-implied verdict, coverage (share not answered unknown), and the
    # paired agreement between the complete and the sparse ternary encodings on the same slot
    tern, tv = defaultdict(lambda: [0, 0, 0]), {}
    for r in study:
        if r["arm"] in ("T_complete", "T_sparse"):
            c = tern["%s|%s|%s|%s" % (r["writer"], r["judge"], r["style"], r["arm"])]
            c[0] += 1
            c[1] += int(r["verdict"] is not None and r["verdict"] == r["record_implied"])
            c[2] += int(r["verdict"] in ("supported", "contradicted"))
            tv[(r["pair_id"], r["writer"], r["judge"], r["style"], r["polarity"], r["role"], r["arm"])] = r["verdict"]
    paired = defaultdict(lambda: [0, 0])
    for (pid, w, j, st, pol, role, arm), v in tv.items():
        if arm != "T_complete":
            continue
        v2 = tv.get((pid, w, j, st, pol, role, "T_sparse"))
        if v is None or v2 is None:
            continue
        c = paired["%s|%s|%s" % (w, j, st)]
        c[0] += 1
        c[1] += int(v == v2)
    rep["ternary"] = {"per_arm": {k: {"rows": v[0], "agreement_with_record": v[1] / v[0], "coverage": v[2] / v[0]}
                                  for k, v in sorted(tern.items())},
                      "complete_vs_sparse_paired": {k: {"slots": v[0], "same_verdict": v[1] / v[0]}
                                                    for k, v in sorted(paired.items())}}
    # screen exclusions within this population, by category and direction
    if screen_rows is not None:
        by = defaultdict(dict)
        for r in screen_rows:
            if r["pair_id"] in popset:
                by[r["pair_id"]][r["role"]] = r["answer"]
        fail = defaultdict(lambda: defaultdict(int))
        for pid, d in by.items():
            dirs = []
            if d.get("yes") is None or d.get("no") is None:
                dirs.append("unparsed")
            if d.get("yes") == "no":
                dirs.append("present_image_answered_no")
            if d.get("no") == "yes":
                dirs.append("absent_image_answered_yes")
            for x in dirs:
                fail[cat[pid]][x] += 1
        n_fail = sum(1 for d in by.values() if not (d.get("yes") == "yes" and d.get("no") == "no"))
        rep["screen_exclusions"] = {"pairs": len(by), "pairs_failing": n_fail,
                                    "by_category_and_direction": {c: dict(v) for c, v in sorted(fail.items())}}
    return rep


def writer_complete_pairs(pool, writer_rows, style="sparse"):
    """Pairs whose records by BOTH writers, for BOTH images, exist in the given style (parsed, not a cut-off prefix)."""
    ok = {(wr["model"], wr["k"]) for wr in writer_rows if wr["style"] == style and wr["ok"]}
    out = set()
    for p in pool:
        ky, kn = image_keys(p)
        if all((w, k) in ok for w in (NEW, OLD) for k in (ky, kn)):
            out.add(p["pair_id"])
    return out


def analyse(rows, pool, pop, writer_rows, labels, screen_rows=None):
    acc = index(rows)
    res = {"population_pairs": len(pop)}
    wok = writer_complete_pairs(pool, writer_rows)
    pop_w = [pid for pid in pop if pid in wok]          # registered population of D0, E, CLEAN, criteria (round 6)
    res["writer_complete_pairs"] = len(pop_w)
    res["writer_missing_pairs"] = len(pop) - len(pop_w)
    cat = {p["pair_id"]: p["category"] for p in pool}
    have = {(wr["model"], wr["k"]) for wr in writer_rows if wr["style"] == "sparse" and wr["ok"]}
    by_cat, by_writer = defaultdict(int), defaultdict(int)
    for p in pool:
        if p["pair_id"] in set(pop) - wok:
            by_cat[p["category"]] += 1
            for w in (NEW, OLD):
                if any((w, k) not in have for k in image_keys(p)):
                    by_writer[w] += 1
    res["writer_missing_by_category"], res["writer_missing_by_writer"] = dict(sorted(by_cat.items())), dict(by_writer)
    for kind in ("D0", "E"):
        tm = terms(kind)
        cc = per_pair(acc, pop_w, tm)
        main = boot(cc)
        lo, hi = boot(per_pair(acc, pop_w, tm, "low")), boot(per_pair(acc, pop_w, tm, "high"))
        main["complete_case_pairs"] = len(cc)
        main["missing_pairs"] = len(pop_w) - len(cc)            # judge-verdict missingness within the population
        main["bound_low"], main["bound_high"] = lo, hi          # CLEAN criterion: judge-verdict missingness
        main["diagnostic_worst_case_incl_writer_missing"] = {  # reported, not a criterion
            "low": boot(per_pair(acc, pop, tm, "low")), "high": boot(per_pair(acc, pop, tm, "high"))}
        if main["pass"]:
            adverse = lo if main["point"] > 0 else hi
            main["clean"] = bool(adverse["pass"] and (adverse["point"] > 0) == (main["point"] > 0))
        else:
            main["clean"] = False
        main["label_sensitivity"] = greedy_reversal(cc)
        res[kind] = main
    res["gates"] = gates(rows, set(pop_w))
    res["combined_claim"] = bool(res["D0"]["pass"] and res["E"]["pass"] and res["D0"]["clean"] and res["E"]["clean"]
                                 and res["gates"]["all_pass"])
    res["descriptive"] = descriptive(rows, acc, pop, pool, writer_rows, labels, screen_rows)
    return res


def selftest():
    """Planted effect: judge Q8 credits the newer writer 10% more under 'absent' (D0 truth +20% in Y units); E truth 0;
    controls 97% faithful; comprehension 100%; 1 pool pair has no rows at all; checklist rows share every key but style."""
    rng = random.Random(1)
    pool, labels, rows, wrows, screen = [], {}, [], [], []
    cats = ("dog", "cat", "car")
    for i in range(1000):
        p = {"pair_id": "p%d" % i, "category": cats[i % 3], "split": "val2014", "yes_image_id": 2 * i,
             "no_image_id": 2 * i + 1}
        pool.append(p)
        ky, kn = image_keys(p)
        labels[ky], labels[kn] = "present", "absent"
        screen += [{"pair_id": p["pair_id"], "role": "yes", "answer": "yes" if i % 50 else "no"},
                   {"pair_id": p["pair_id"], "role": "no", "answer": "no"}]
        for w in (NEW, OLD):
            for k, present in ((ky, True), (kn, False)):
                for st in ("sparse", "checklist"):
                    states = {p["category"]: "present"} if (present and rng.random() < 0.8) else {}
                    wrows.append({"k": k, "model": w, "style": st, "ok": True, "parsed": {"states": states}})
            for j in (Q8, GEM):
                for st in ("sparse", "checklist"):
                    arms = ("A_unfilled", "A_absent") if st == "checklist" else ("A_unfilled", "A_absent", "B_complete",
                                                                               "B_sparse") + CTRL_ARMS
                    for arm in arms:
                        base = 0.70 if w == NEW else 0.60
                        if j == Q8 and arm == "A_absent" and w == NEW:
                            base += 0.10 if st == "sparse" else -0.30   # checklist must not leak into the primary
                        for pol, tr, fr in POLS:
                            for role in ("yes", "no"):
                                truth = "supported" if role == tr else "unsupported"
                                v = "supported" if rng.random() < (base if role == tr else 1 - base) else "unsupported"
                                implied = None
                                if arm.startswith("C_"):
                                    implied = truth if arm.startswith("C_corr") else (
                                        "unsupported" if truth == "supported" else "supported")
                                    v = implied if rng.random() < 0.97 else ("supported" if implied == "unsupported"
                                                                            else "unsupported")
                                ok = rng.random() > 0.01
                                rows.append({"pair_id": p["pair_id"], "k": ky if role == "yes" else kn, "writer": w,
                                             "judge": j, "style": st, "arm": arm, "polarity": pol, "role": role,
                                             "truth": truth, "verdict": v if ok else None, "ok": ok, "ternary": False,
                                             "record_implied": implied})
    for j in (Q8, GEM):
        for enc in ("complete", "sparse"):
            for c in range(N_COMP):
                rows.append({"pair_id": "synthetic", "k": "syn%03d" % c, "writer": None, "judge": j, "style": None,
                             "arm": "COMP_" + enc, "polarity": "assert", "role": "synthetic", "truth": "supported",
                             "verdict": "supported", "ok": True, "ternary": False, "record_implied": "supported"})
    pool.append({"pair_id": "p_no_rows", "category": "dog", "split": "val2014", "yes_image_id": 99998,
                 "no_image_id": 99999})
    pop = [p["pair_id"] for p in pool]
    res = analyse(rows, pool, pop, wrows, labels, screen)
    d0, e = res["D0"], res["E"]
    print("selftest D0 %.3f [%.3f, %.3f] pass=%s clean=%s | bounds %.3f .. %.3f | E %.3f pass=%s | gates %s | combined %s"
          % (d0["point"], d0["lo95"], d0["hi95"], d0["pass"], d0["clean"], d0["bound_low"]["point"],
             d0["bound_high"]["point"], e["point"], e["pass"], res["gates"]["all_pass"], res["combined_claim"]))
    assert d0["pass"] and d0["lo95"] <= 0.20 <= d0["hi95"], "D0 must recover the planted +20% from sparse rows only"
    assert d0["bound_low"]["point"] <= d0["point"] <= d0["bound_high"]["point"]
    assert d0["bound_low"]["n"] == len(pop) - 1 == len(pop) - res["writer_missing_pairs"], "bounds: writer-complete pairs"
    assert d0["diagnostic_worst_case_incl_writer_missing"]["low"]["n"] == len(pop)
    assert res["writer_missing_by_category"] == {"dog": 1} and res["writer_missing_by_writer"] == {NEW: 1, OLD: 1}
    assert e["lo95"] <= 0 <= e["hi95"] and not e["pass"] and not res["combined_claim"]
    assert res["gates"]["all_pass"], "97% faithful controls and 100% comprehension pass the 90% criteria"
    ls = d0["label_sensitivity"]
    assert ls["greedy_reversals_to_below_threshold"] >= 1
    assert abs(ls["random_reversal"]["0.05"]["simulated_mean"] - ls["random_reversal"]["0.05"]["expected"]) < 0.01
    assert greedy_reversal({"a": 0.01, "b": -0.01})["greedy_reversals_to_below_threshold"] == 0, "k = 0 is evaluated"
    assert greedy_reversal({"a": 0.3})["n_pairs"] == 1
    for bad in ([rows[0]], [dict(rows[0], role="maybe")]):
        try:
            index(rows + bad)
            raise AssertionError("invalid row not caught")
        except ValueError:
            pass
    g2 = gates([r for r in rows if not (r["arm"] == "C_rev_sparse" and r["judge"] == GEM and r["pair_id"] < "p2")],
               set(pop))
    assert not g2["all_pass"], "dropped control rows count as failures"
    dd = res["descriptive"]
    x = dd["per_judge"]["%s|sparse|A_absent|G" % Q8]
    assert x["n"] == x["G"]["n"] and abs(x["Y_new"] - x["Y_old"] - x["G"]["point"]) < 1e-9, "Ys on the pairs of G"
    ra = dd["row_accounting"]["%s|%s|sparse|A_absent|supported" % (NEW, Q8)]
    assert ra["expected"] == 2 * len(pop) and ra["absent_rows"] == 2, "the pair with no rows is counted as absent"
    assert dd["screen_exclusions"]["pairs_failing"] == 20
    assert dd["category"]["D0"]["category_balanced"]["categories"] == 3
    dk = dd["decomposition"]["%s|A_absent" % Q8]
    assert dk["n"] == dk["G_judge"]["n"] == dk["G_deterministic_reader"]["n"] == dk["residual_judge_minus_reader"]["n"]
    assert dd["writer_parse_formats"]
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-rows"); ap.add_argument("--writer-rows"); ap.add_argument("--pool")
    ap.add_argument("--labels"); ap.add_argument("--screen-pass"); ap.add_argument("--screen-rows")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--expect-pairs", type=int, default=516, help="0 only for the out-of-pool engineering smoke")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    rows = []
    for p in a.judge_rows.split(","):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    wrows = []
    for p in a.writer_rows.split(","):
        wrows += [json.loads(l) for l in open(p) if l.strip()]
    pool = [json.loads(l) for l in open(a.pool) if l.strip()]
    labels = json.load(open(a.labels))
    keep = {l.strip() for l in open(a.screen_pass) if l.strip()}
    screen = [json.loads(l) for l in open(a.screen_rows) if l.strip()] if a.screen_rows else None
    pop_all = [p["pair_id"] for p in pool]
    assert len(set(pop_all)) == len(pop_all), "pool pair ids are unique"
    assert a.expect_pairs == 0 or len(pop_all) == a.expect_pairs, "the frozen pool has 516 pairs"
    assert a.expect_pairs == 0 or screen is not None, "the study analysis reads the screen rows"
    assert keep <= set(pop_all), "screen mask must be a subset of the pool"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)                             # atomic reservation: the frozen analysis runs once
    res = {"primary_all_pool_pairs": analyse(rows, pool, pop_all, wrows, labels, screen),
           "secondary_screened_pairs": analyse(rows, pool, [pid for pid in pop_all if pid in keep], wrows, labels,
                                               screen)}
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    for pop_name, r in res.items():
        for k in ("D0", "E"):
            x = r[k]
            print(pop_name, k, {f: x.get(f) for f in ("n", "point", "lo95", "hi95", "pass", "clean", "missing_pairs")},
                  "bounds", round(x["bound_low"].get("point", float("nan")), 4),
                  round(x["bound_high"].get("point", float("nan")), 4))
        print(pop_name, "criteria all pass:", r["gates"]["all_pass"], "| combined claim:", r["combined_claim"])


if __name__ == "__main__":
    main()
