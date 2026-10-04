"""Claim-blind design sweep (2026-09-22): record elicitations x adjudication rules.

The pre-registered arms (schema, prose, capped-at-two, zoom records, one adjudicator) failed to
beat claim-first at a matched false-strike rate. This module holds the wider design space the
sweep explores on a held-out dev slice of pairs (cells 41 to 60, never part of any evaluation
sample). Selection happens in scripts/sweep_select.py under a rule fixed before the runs: the arm
with the highest Youden index on the image-necessary dev pairs whose false-strike rate is within
five points of claim-first's, and the winner is then run once on the evaluation sample.

Record variants (the call that sees the image and never the claim)
  schema     the pre-registered finding checklist, absent by default
  prose      free description, no checklist
  proselat   description region by region with explicit side and zone, normal stated as normal
  sparse1    checklist capped at one finding
  sparse3    checklist capped at three findings
  gated      checklist in which a finding may be marked present only with a named visible sign
  zoom       schema plus one native-resolution crop request

Adjudication variants (the call that sees the record and the claim, never the image)
  default    the pre-registered rule
  strict     supported only when the record explicitly agrees with the sentence's polarity
  lenient    unsupported only when the record explicitly contradicts the sentence
  polarity   an assertion needs the record to affirm it; a denial fails only if the record affirms
             the finding (the asymmetric rule the 2026-09-20 readings suggested)

Names used by the harness: sweep_<record>_<adjudicator>, e.g. sweep_gated_polarity.
"""
from __future__ import annotations

import json
from typing import Any

SWEEP_VERSION = "s1"

RECORD_VARIANTS = ("schema", "prose", "proselat", "sparse1", "sparse3", "gated", "zoom")
ADJ_VARIANTS = ("default", "strict", "lenient", "polarity")

SYS_VERIFIER = (
    "You are verifying a single sentence taken from a chest radiograph report. "
    "Sentences are sometimes true of the radiograph and sometimes not; both verdicts are common. "
    "Answer only with the JSON object asked for. Do not add commentary."
)
SYS_OBSERVER = (
    "You are reading a chest radiograph and recording what you see. "
    "Answer only with the JSON object asked for. Do not add commentary."
)

RECORD_JSON = (
    'Reply as {{"record": [{{"finding": "<name>", "present": true or false, '
    '"side": "left" or "right" or "bilateral" or "not applicable", '
    '"box": [x1, y1, x2, y2] or null, "note": "<short>"}}], '
    '"other": "<anything notable not in the list, one sentence>"}}'
)

U_RECORD = (
    "A chest radiograph is attached. Record what it shows, finding by finding, before any claim "
    "about it is made. Most chest radiographs show none or only one of the findings listed, so "
    "the default for every finding is present false; mark a finding present only when you see "
    "clear evidence of it in this image, and say what that evidence is in the note. For each "
    "finding state whether it is present, which side, and the region you would point to.\n"
    "Findings to cover: {findings}.\n" + RECORD_JSON
)
U_RECORD_ZOOM = U_RECORD.rstrip(")") + (
    '\nIf one region needs a closer look before you can fill the record, name it as '
    '"zoom": [x1, y1, x2, y2]; otherwise set "zoom" to null.'
)
U_RECORD_CAP = (
    "A chest radiograph is attached. Record what it shows before any claim about it is made. Most "
    "chest radiographs show none or only one abnormality, so the default for every finding is "
    "present false. You may mark AT MOST {cap} present, and only the ones you are most "
    "confident about; give the region for each. Everything else must be present false.\n"
    "Findings to cover: {findings}.\n" + RECORD_JSON
)
U_RECORD_GATED = (
    "A chest radiograph is attached. Record what it shows, finding by finding, before any claim "
    "about it is made. A finding may be marked present ONLY if you can name a specific sign of it "
    "that is visible in this image, for example blunting of a costophrenic angle, a visible pleural "
    "line with no lung markings beyond it, a cardiac silhouette wider than half the thorax, or an "
    "air bronchogram inside an opacity. Write that sign in the note. If you cannot name such a sign "
    "for this image, the finding is absent. When in doubt, absent. Most radiographs show none or "
    "one of the findings listed.\n"
    "Findings to cover: {findings}.\n" + RECORD_JSON
)
U_PROSE = (
    "A chest radiograph is attached. Describe what it shows in plain prose, as a radiologist would "
    "dictate the findings section, before any claim about it is made. Describe the heart size, the "
    "mediastinum, the lungs, the pleural spaces, and the bones, naming only abnormalities you can "
    "actually see; a normal structure is described as normal. Four to seven sentences, no lists.\n"
    'Reply as {{"description": "<the prose>"}}'
)
U_PROSE_LAT = (
    "A chest radiograph is attached. Describe it region by region, before any claim about it is "
    "made: the heart and mediastinum; the right lung by upper, mid, and lower zone; the left lung "
    "by upper, mid, and lower zone; the right and left pleural spaces and costophrenic angles; the "
    "bones and any devices. For each region either say it is normal or name the abnormality you "
    "see, with its side and zone. Name only what you can see in this image. Five to nine "
    "sentences, no lists.\n"
    'Reply as {{"description": "<the prose>"}}'
)

# adjudicators; {evidence} is the record (JSON) or the description (prose)
A_DEFAULT_RECORD = (
    "Here is a record of what a chest radiograph shows, written before any claim was made:\n"
    "{evidence}\n\nSentence from the report of that radiograph:\n{claim}\n\n"
    "Using only the record, judge whether the sentence is true.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}}'
)
A_DEFAULT_PROSE = (
    "Here is a description of a chest radiograph, written before any claim was made:\n"
    "{evidence}\n\nSentence from the report of that radiograph:\n{claim}\n\n"
    "Using only the description, judge whether the sentence is true. If the description does not "
    "mention what the sentence asserts, the sentence is unsupported.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}}'
)
A_STRICT = (
    "Here is {kind} of a chest radiograph, written before any claim was made:\n"
    "{evidence}\n\nSentence from the report of that radiograph:\n{claim}\n\n"
    "Judge the sentence from {kind} alone, by this rule. If the sentence says a finding is "
    "present, it is supported only if {kind} explicitly says that finding is present, on the same "
    "side if the sentence names a side; otherwise it is unsupported. If the sentence says a finding "
    "is absent, it is supported only if {kind} explicitly says that finding is absent or that the "
    "region is normal; otherwise it is unsupported. A finding {kind} does not mention counts as not "
    "confirmed.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}}'
)
A_LENIENT = (
    "Here is {kind} of a chest radiograph, written before any claim was made:\n"
    "{evidence}\n\nSentence from the report of that radiograph:\n{claim}\n\n"
    "Judge the sentence from {kind} alone, by this rule. The sentence is unsupported only if {kind} "
    "explicitly contradicts it, that is, the sentence says a finding is present and {kind} says it "
    "is absent or the region is normal, or the sentence says a finding is absent and {kind} says it "
    "is present. If {kind} does not mention the finding, the sentence is supported.\n"
    'Reply as {{"verdict": "supported" or "unsupported", "reason": "<one short sentence>"}}'
)
A_POLARITY = (
    "Here is {kind} of a chest radiograph, written before any claim was made:\n"
    "{evidence}\n\nSentence from the report of that radiograph:\n{claim}\n\n"
    "First decide whether the sentence asserts that a finding is present or denies one. Then judge "
    "it from {kind} alone, by this rule. An assertion is supported only if {kind} explicitly says "
    "that finding is present, on the same side if the sentence names a side; otherwise "
    "unsupported. A denial is unsupported only if {kind} explicitly says that finding is present; "
    "if {kind} says it is absent or does not mention it, the denial is supported.\n"
    'Reply as {{"polarity": "asserts" or "denies", "verdict": "supported" or "unsupported", '
    '"reason": "<one short sentence>"}}'
)


def parse_name(protocol: str) -> tuple[str, str]:
    """sweep_<record>_<adjudicator> -> (record, adjudicator)."""
    parts = protocol.split("_")
    if len(parts) != 3 or parts[0] != "sweep":
        raise ValueError("bad sweep protocol name %s" % protocol)
    rec, adj = parts[1], parts[2]
    if rec not in RECORD_VARIANTS or adj not in ADJ_VARIANTS:
        raise ValueError("unknown sweep variant in %s" % protocol)
    return rec, adj


def is_prose(rec: str) -> bool:
    return rec in ("prose", "proselat")


def make_record(llm, parse_json, rec: str, images: list, findings: list[str]) -> dict[str, Any]:
    """One observer call. Returns {"record": <stored object>, "zoom": ..., "payload": <text for
    the adjudicator>, "kind": "the record" | "the description"}."""
    fl = ", ".join(findings)
    if rec == "prose":
        text = llm.chat("observer", SYS_OBSERVER, U_PROSE, images=images, json_mode=True, max_tokens=768)
        d = parse_json(text)
        desc = d.get("description", text) if isinstance(d, dict) else text
        return {"record": {"description": desc}, "zoom": None, "payload": desc, "kind": "the description"}
    if rec == "proselat":
        text = llm.chat("observer", SYS_OBSERVER, U_PROSE_LAT, images=images, json_mode=True, max_tokens=900)
        d = parse_json(text)
        desc = d.get("description", text) if isinstance(d, dict) else text
        return {"record": {"description": desc}, "zoom": None, "payload": desc, "kind": "the description"}
    if rec == "schema":
        tpl = U_RECORD.format(findings=fl)
    elif rec == "zoom":
        tpl = U_RECORD_ZOOM.format(findings=fl)
    elif rec == "gated":
        tpl = U_RECORD_GATED.format(findings=fl)
    elif rec == "sparse1":
        tpl = U_RECORD_CAP.format(cap="ONE finding", findings=fl)
    elif rec == "sparse3":
        tpl = U_RECORD_CAP.format(cap="THREE findings", findings=fl)
    else:
        raise ValueError("unknown record variant %s" % rec)
    text = llm.chat("observer", SYS_OBSERVER, tpl, images=images, json_mode=True, max_tokens=1536)
    record = parse_json(text)
    if not isinstance(record, dict):
        raise ValueError("record is not an object")
    payload = json.dumps(record.get("record", record))
    return {"record": record, "zoom": record.get("zoom"), "payload": payload, "kind": "the record"}


def adjudicate(llm, parse_json, adj: str, made: dict[str, Any], claim: str) -> dict[str, Any]:
    """One text-only adjudicator call against a made record."""
    kind = made["kind"]
    ev = made["payload"]
    if adj == "default":
        tpl = A_DEFAULT_PROSE if kind == "the description" else A_DEFAULT_RECORD
        user = tpl.format(evidence=ev, claim=claim)
    elif adj == "strict":
        user = A_STRICT.format(kind=kind, evidence=ev, claim=claim)
    elif adj == "lenient":
        user = A_LENIENT.format(kind=kind, evidence=ev, claim=claim)
    elif adj == "polarity":
        user = A_POLARITY.format(kind=kind, evidence=ev, claim=claim)
    else:
        raise ValueError("unknown adjudicator %s" % adj)
    text = llm.chat("adjudicator", SYS_VERIFIER, user, json_mode=True)
    data = parse_json(text)
    return data if isinstance(data, dict) else {}


def box_for(made: dict[str, Any], finding: str):
    rec = made.get("record")
    if isinstance(rec, dict):
        for entry in rec.get("record", []) or []:
            if isinstance(entry, dict) and entry.get("finding") == finding:
                return entry.get("box")
    return None


def run_protocol(llm, parse_json, protocol: str, row: dict, images: list, findings: list[str]) -> tuple[dict, dict]:
    """For verify_run.py: returns (out_fields, data) like the built-in branches."""
    rec, adj = parse_name(protocol)
    made = make_record(llm, parse_json, rec, images, findings)
    data = adjudicate(llm, parse_json, adj, made, row["claim"])
    out = {"record": made["record"], "observations": None, "box": box_for(made, row["finding"]),
           "zoom": made["zoom"], "sweep_version": SWEEP_VERSION}
    if "polarity" in data:
        out["observations"] = data.get("polarity")
    return out, data
