#!/usr/bin/env python3
"""Second domain: records, forms, encodings, claims, and judge messages (protocol DRAFT 2026-09-29, sections 3-6).

Pure functions, no model. The renderer never receives the claim. `python d2_forms.py --selftest` runs the checks the
protocol requires before any read: round-trip decode equality of the 2 closed-world encodings, the controls change only
the target state, claim templates, parser behaviour, and the synthetic comprehension set's contract-implied verdicts.
"""
import json
import random
import re
import sys

CATS = ["person", "clock", "car", "dog", "bird", "umbrella", "boat", "cat", "banana", "sink", "stop sign", "tv",
        "bench", "bicycle", "knife", "laptop", "train", "bus", "mouse", "chair"]
DISPLAY = {c: c for c in CATS}
DISPLAY.update({"tv": "TV or computer monitor", "mouse": "computer mouse"})
BY_DISPLAY = {v: k for k, v in DISPLAY.items()}
AN = {"umbrella"}                                     # display names that take "an"

WRITER_SYSTEM = ("You are looking at a photograph and recording what you see. Answer only with the JSON object asked "
                 "for. Do not add commentary.")
_LIST = ", ".join(DISPLAY[c] for c in CATS)
WRITER_USER = {
    "sparse": ("A photograph is attached. Record the objects from the list below that you can see in it, before any "
               "question about it is asked. List an object only if at least one is visible. Objects to consider: "
               + _LIST + ".\nReply as {\"objects\": [{\"object\": \"<name from the list>\", \"note\": \"<where it is, "
               "one short phrase>\"}], \"other\": \"<anything notable not in the list, one sentence>\"}"),
    "checklist": ("A photograph is attached. Record what it shows, object by object, before any question about it is "
                  "asked. For each object in the list below, say whether at least one is visible. Objects to cover: "
                  + _LIST + ".\nReply as {\"objects\": [{\"object\": \"<name from the list>\", \"present\": true or "
                  "false, \"note\": \"<short>\"}], \"other\": \"<one sentence>\"}"),
}
JUDGE_SYSTEM = ("You are verifying a single sentence about a photograph. Sentences are sometimes true of the photograph "
                "and sometimes not; both verdicts are common. Answer only with the JSON object asked for. Do not add "
                "commentary.")
_UNIVERSE = ("The record covers exactly these objects: " + _LIST + ". ")
_PRECEDENCE = " The listed states and this default take precedence over the notes and the other field."
CONVENTION = {   # history (all 2026-09-30, before any study read): the red-team's enumerated universe sentence (_UNIVERSE,
                 # kept above for the record) made both primary judges read the names as findings (smoke 23368); probe
                 # 23375 adopted K3 on 8 person-only smoke pairs; probe 23386 showed that too few to choose (K3 Gemma
                 # 93.8% -> 84.4% at temperature 0); calibration 23398 on calib_v2 (41 out-of-pool pairs, 17 categories,
                 # rule logged before the run) adopted K5: lowest comprehension/control cell 100% for both primary judges
                 # (K3 98.8%, K4 98.8%). K5 states the closed-world default outright and names no undisclosed list; the
                 # precedence sentence keeps states + default authoritative over notes and free text in both encodings.
    "closed": ("Convention for this record: the listed states are authoritative for whether an object is in the "
               "photograph. Any object that the record does not list is absent from the photograph, even though the "
               "record does not say so explicitly." + _PRECEDENCE),
    "open": ("Convention for this record: the listed states are authoritative for whether an object is in the "
             "photograph. Any object that the record does not list is not reported, so the record says nothing about "
             "whether it is in the photograph." + _PRECEDENCE),
}
TERNARY_REPLY = ("Using only the record, judge whether the sentence is supported, contradicted, or unknown from the "
                 "record.\nReply as {\"verdict\": \"supported\" or \"contradicted\" or \"unknown\", \"reason\": \"<one "
                 "short sentence>\"}")
BINARY_REPLY = ("Using only the record, judge whether the sentence is true.\nReply as {\"verdict\": \"supported\" or "
                "\"unsupported\", \"reason\": \"<one short sentence>\"}")


def claim(cat, polarity):
    n = DISPLAY[cat]
    art = "an" if cat in AN else "a"
    return ("There is %s %s in the image." % (art, n)) if polarity == "assert" else ("There is no %s in the image." % n)


def parse_writer(obj, style, fmt="list"):
    """Writer JSON (already json.loads'ed) -> {"states": {cat: "present"|"absent"}, "notes": {cat: str}, "other": str,
    "dropped": n, "format": fmt}. Unknown objects are dropped and counted; a duplicate keeps its first entry.
    Rule R1 (pre-freeze, writers-only check 23424): the dict layout {"objects": {"<name>": {...} or "<note>"}} is read as
    the list [{"object": "<name>", ...}] in its written order and flagged format "dict"."""
    if isinstance(obj, dict) and isinstance(obj.get("objects"), dict):
        items = []
        for name, e in obj["objects"].items():
            items.append(dict(e, object=name) if isinstance(e, dict) else {"object": name, "note": e})
        obj, fmt = dict(obj, objects=items), "dict"
    if not isinstance(obj, dict) or not isinstance(obj.get("objects"), list):
        return None
    states, notes, dropped = {}, {}, 0
    for e in obj["objects"]:
        if not isinstance(e, dict):
            dropped += 1; continue
        name = e.get("object")
        cat = BY_DISPLAY.get(name) or (name if name in CATS else None)
        if cat is None or cat in states:
            dropped += 1; continue
        if style == "sparse":
            states[cat] = "present"
        else:
            p = e.get("present")
            if not isinstance(p, bool):
                dropped += 1; continue
            states[cat] = "present" if p else "absent"
        note = e.get("note", "")
        if isinstance(note, str) and note:
            notes[cat] = note
    other = obj.get("other", "")
    return {"states": states, "notes": notes, "other": other if isinstance(other, str) else "", "dropped": dropped,
            "format": fmt}


def repair_cut_off(text, style):
    """Rule R2 (pre-freeze, writers-only check 23424; revised after round 5): DIAGNOSTICS ONLY. Called only when the whole
    reply fails to parse AND the server stopped at the token limit. If the reply opens an "objects" array, return the
    complete entries before the cut (json raw_decode one entry at a time; exactly one comma between entries; at least 1
    entry), flagged "cut_off_prefix". The writer record itself stays missing: a prefix is not a complete record."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    m = re.match(r'\s*\{\s*"objects"\s*:\s*\[', s)
    if not m:
        return None
    dec, i, items = json.JSONDecoder(), m.end(), []
    sep, ws = re.compile(r"\s*,\s*"), re.compile(r"\s*")
    while True:
        nxt = (sep if items else ws).match(s, i)
        if nxt is None:
            break
        i = nxt.end()
        try:
            e, i = dec.raw_decode(s, i)
        except ValueError:
            break
        items.append(e)
    if not items:
        return None
    return parse_writer({"objects": items, "other": ""}, style, fmt="cut_off_prefix")


def decoded(rec, world):
    """Full state vector over CATS: omitted objects become 'absent' (closed) or 'not reported' (open)."""
    fill = "absent" if world == "closed" else "not reported"
    return {c: rec["states"].get(c, fill) for c in CATS}


def _payload(objects, rec):
    notes = {DISPLAY[c]: rec["notes"][c] for c in CATS if c in rec["notes"]}
    return json.dumps({"objects": objects, "notes": notes, "other": rec["other"]}, ensure_ascii=False)


def render(rec, form):
    """Forms (A): unfilled, absent, not_reported. Encodings (B, closed world): complete, sparse. Open world: ternary_complete,
    ternary_sparse. The claim is never an argument."""
    if form == "unfilled":
        objs = [{"object": DISPLAY[c], "state": rec["states"][c]} for c in CATS if c in rec["states"]]
    elif form in ("absent", "complete"):
        objs = [{"object": DISPLAY[c], "state": s} for c, s in decoded(rec, "closed").items()]
    elif form in ("not_reported", "ternary_complete"):
        objs = [{"object": DISPLAY[c], "state": s} for c, s in decoded(rec, "open").items()]
    elif form == "sparse":
        objs = [{"object": DISPLAY[c], "state": "present"} for c, s in decoded(rec, "closed").items() if s == "present"]
    elif form == "ternary_sparse":
        objs = [{"object": DISPLAY[c], "state": s} for c, s in decoded(rec, "open").items() if s != "not reported"]
    else:
        raise ValueError(form)
    return _payload(objs, rec)


def decode_evidence(evidence, world):
    """Inverse of render for the contract encodings: state vector + notes + other."""
    o = json.loads(evidence)
    fill = "absent" if world == "closed" else "not reported"
    st = {c: fill for c in CATS}
    for e in o["objects"]:
        st[BY_DISPLAY[e["object"]]] = e["state"]
    return {"states": st, "notes": o["notes"], "other": o["other"]}


def with_target(rec, cat, state, strip_text=True):
    """Control records: the focal object's state is set (target-corrected / target-reversed). With strip_text (the
    default, engineering smoke 23273, 2026-09-29) the notes and free text are removed, because the writer's own note on
    the focal object ("walking between two elephants") contradicts a reversed state and the judges followed the note;
    the controls then test whether a judge reads the stated fact, on states-only records."""
    r = {"states": dict(decoded(rec, "closed")), "notes": {} if strip_text else dict(rec["notes"]),
         "other": "" if strip_text else rec["other"], "dropped": rec.get("dropped", 0)}
    r["states"][cat] = state
    return r


def judge_messages(evidence, claim_text, convention=None, ternary=False):
    parts = ["Here is a record of what a photograph shows, written before any sentence was made:", evidence, ""]
    if convention:
        parts += [CONVENTION[convention], ""]
    parts += ["Sentence about that photograph:", claim_text, "", TERNARY_REPLY if ternary else BINARY_REPLY]
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


def implied_verdict(state, polarity, ternary=False):
    """The verdict a contract-following judge must give for the focal object's state."""
    if ternary:
        if state == "not reported":
            return "unknown"
        return "supported" if (state == "present") == (polarity == "assert") else "contradicted"
    present = state == "present"
    return "supported" if present == (polarity == "assert") else "unsupported"


def comprehension_set(seed=20260930, n_per_cell=8):
    """48 balanced synthetic cases per closed-world encoding: target present / written absent / omitted x assert / deny
    x 8 random backgrounds. For the complete encoding an omitted target is rendered as absent (same decoded meaning)."""
    rng = random.Random(seed)
    cases = []
    for target_state in ("present", "absent", "omitted"):
        for pol in ("assert", "deny"):
            for _ in range(n_per_cell):
                cat = rng.choice(CATS)
                states = {c: "present" for c in rng.sample([c for c in CATS if c != cat], rng.randint(0, 4))}
                if target_state != "omitted":
                    states[cat] = target_state
                rec = {"states": states, "notes": {c: "in the scene" for c in states if states[c] == "present"},
                       "other": "An ordinary scene.", "dropped": 0}
                truth_state = "present" if target_state == "present" else "absent"
                for enc in ("complete", "sparse"):
                    cases.append({"encoding": enc, "target": cat, "polarity": pol, "target_state": target_state,
                                  "evidence": render(rec, enc), "claim": claim(cat, pol),
                                  "expected": implied_verdict(truth_state, pol)})
    return cases


def selftest():
    rng = random.Random(7)
    for _ in range(2000):
        states = {c: rng.choice(["present", "absent"]) for c in rng.sample(CATS, rng.randint(0, 20))}
        rec = {"states": states, "notes": {c: "n%d" % i for i, c in enumerate(states) if rng.random() < 0.5},
               "other": rng.choice(["", "A street.", "No people visible."]), "dropped": 0}
        a, b = decode_evidence(render(rec, "complete"), "closed"), decode_evidence(render(rec, "sparse"), "closed")
        assert a == b, "closed-world encodings must decode identically"
        assert a["states"] == decoded(rec, "closed")
        ta, tb = decode_evidence(render(rec, "ternary_complete"), "open"), decode_evidence(render(rec, "ternary_sparse"), "open")
        assert ta == tb, "open-world encodings must decode identically"
        cat = rng.choice(CATS)
        for st in ("present", "absent"):
            r2 = with_target(rec, cat, st)
            d0, d1 = decoded(rec, "closed"), decoded(r2, "closed")
            assert all(d0[c] == d1[c] for c in CATS if c != cat) and d1[cat] == st, "controls change only the target state"
            assert r2["notes"] == {} and r2["other"] == "", "control records are states only"
            r3 = with_target(rec, cat, st, strip_text=False)
            assert r3["notes"] == rec["notes"] and r3["other"] == rec["other"]
    assert claim("umbrella", "assert") == "There is an umbrella in the image."
    assert claim("tv", "deny") == "There is no TV or computer monitor in the image."
    assert claim("mouse", "assert") == "There is a computer mouse in the image."
    p = parse_writer({"objects": [{"object": "dog", "note": "left"}, {"object": "dog"}, {"object": "horse"},
                                  {"object": "TV or computer monitor", "note": "wall"}], "other": "x"}, "sparse")
    assert p["states"] == {"dog": "present", "tv": "present"} and p["dropped"] == 2
    p = parse_writer({"objects": [{"object": "cat", "present": False}, {"object": "car", "present": "yes"}]}, "checklist")
    assert p["states"] == {"cat": "absent"} and p["dropped"] == 1
    assert parse_writer({"record": []}, "sparse") is None
    cs = comprehension_set()
    assert len(cs) == 96 and sum(c["encoding"] == "sparse" for c in cs) == 48
    for c in cs:
        d = decode_evidence(c["evidence"], "closed")
        st = d["states"][c["target"]]
        assert implied_verdict(st, c["polarity"]) == c["expected"]
    exp = [c["expected"] for c in cs if c["encoding"] == "sparse"]
    assert exp.count("supported") == 24 and exp.count("unsupported") == 24
    msgs = judge_messages(render(rec, "sparse"), claim("dog", "assert"), convention="closed")
    assert CONVENTION["closed"] in msgs[1]["content"] and "dog" in msgs[1]["content"]
    print("selftest OK: 2000 random records round-trip, controls, claims, parser, 96 comprehension cases balanced")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
