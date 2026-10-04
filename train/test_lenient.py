"""Unit tests for canonical_lenient (no adjudicator needed). Run from $CLAIMBLIND_ROOT/code."""
import json
import sys

sys.path.insert(0, ".")
from rl_build_prefs_v1 import FINDINGS, canonical
from rl_verl_reward_lenient import canonical_lenient

ok = 0


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok += int(bool(cond))


valid = {"record": [{"finding": "Pleural Effusion", "present": True, "side": "left", "box": [1, 2, 3, 4], "note": "blunted angle"},
                    {"finding": "Cardiomegaly", "present": False, "side": "not applicable", "box": None, "note": ""}], "other": "pacemaker"}
s_ok, s_canon, _ = canonical(valid)
l_ok, l_canon, rep = canonical_lenient(valid)
check("valid record: identical bytes to strict", s_ok and l_ok and s_canon == l_canon and rep == 0)

extra = {"record": valid["record"] + [{"finding": "Support Devices", "present": True, "side": "right", "box": None, "note": "line"}] * 11}
check("strict rejects >12 entries", canonical(extra)[0] is False)
l_ok, l_canon, rep = canonical_lenient(extra)
check("lenient repairs >12 entries to the valid record's content", l_ok and json.loads(l_canon)["record"] == json.loads(s_canon)["record"])

badbox = {"record": [{"finding": "Pleural Effusion", "present": True, "side": "left", "box": [1, 2, 3], "note": "x"}], "other": ""}
check("strict rejects bad box", canonical(badbox)[0] is False)
l = json.loads(canonical_lenient(badbox)[1])["record"]
pe = [e for e in l if e["finding"] == "Pleural Effusion"][0]
check("lenient keeps presence, nulls the box", pe["present"] is True and pe["box"] is None and pe["side"] == "left")

dup = {"record": [{"finding": "Edema", "present": True, "side": "bilateral", "box": None, "note": "a"},
                  {"finding": "Edema", "present": False, "side": "not applicable", "box": None, "note": "b"}]}
e = [x for x in json.loads(canonical_lenient(dup)[1])["record"] if x["finding"] == "Edema"][0]
check("duplicate keeps FIRST entry", e["present"] is True and e["note"] == "a")

strbool = {"record": [{"finding": "Edema", "present": "false", "side": "x", "box": None, "note": 5}]}
e = [x for x in json.loads(canonical_lenient(strbool)[1])["record"] if x["finding"] == "Edema"][0]
check("string 'false' -> absent (never bool('false') = True); bad side and note repaired",
      e["present"] is False and e["side"] == "not applicable" and e["note"] == "")

junk = {"record": [{"finding": "Edema", "present": "maybe", "side": "left", "box": None, "note": ""}]}
e = [x for x in json.loads(canonical_lenient(junk)[1])["record"] if x["finding"] == "Edema"][0]
check("unreadable present -> absent", e["present"] is False)

check("empty record -> all 12 absent", all(not x["present"] for x in json.loads(canonical_lenient({"record": []})[1])["record"])
      and len(json.loads(canonical_lenient({"record": []})[1])["record"]) == 12)
check("non-object -> not ok", canonical_lenient([1, 2])[0] is False)
print("TESTS %d/10" % ok)
sys.exit(0 if ok == 10 else 1)
