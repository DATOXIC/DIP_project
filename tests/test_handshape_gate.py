# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamic_dtw_engine import MatchCandidate, MatchResult  # noqa: E402
from handshape_gate import (WordShape, apply_shape_gate, boundary_group_evidence, group_of,  # noqa: E402
                            learn_word_table, to_group_probs)

TABLE = {
    "PLEASE": WordShape({"open": 0.9}, {"open"}, True),
    "SORRY": WordShape({"fist": 0.85}, {"fist"}, True),
    "HUNGRY": WordShape({"fist": 0.4, "open": 0.2}, {"fist", "open"}, False),
}


def make(status, words):
    cands = [MatchCandidate(w, w, 1.0 + 0.05 * i, 0.5) for i, w in enumerate(words)]
    return MatchResult(status, words[0], words[0], 1.0, 0.5, cands)


def test_groups():
    assert group_of("d") == group_of("1") == "point"
    assert group_of("a") == group_of("s") == "fist"
    assert group_of("c") == "c"


def test_removes_conflicting_rival_and_promotes():
    r = apply_shape_gate(make("UNCERTAIN", ["PLEASE", "SORRY"]), {"open": 0.9, "fist": 0.1}, TABLE)
    assert r.top_word == "PLEASE" and r.status == "ACCEPTED" and [c.word_key for c in r.candidates] == ["PLEASE"]


def test_changes_top_when_top_conflicts_and_stays_uncertain():
    r = apply_shape_gate(make("UNCERTAIN", ["SORRY", "PLEASE"]), {"open": 0.9, "fist": 0.1}, TABLE)
    assert r.top_word == "PLEASE" and r.status == "UNCERTAIN"


def test_accepted_untouched_unless_hard():
    res = make("ACCEPTED", ["SORRY", "PLEASE"])
    ev = {"open": 0.9, "fist": 0.1}
    assert apply_shape_gate(res, ev, TABLE) is res
    assert apply_shape_gate(res, ev, TABLE, gate_accepted=True).top_word == "PLEASE"


def test_weak_or_missing_evidence_no_change():
    res = make("UNCERTAIN", ["PLEASE", "SORRY"])
    assert apply_shape_gate(res, {"open": 0.4, "fist": 0.35}, TABLE) is res
    assert apply_shape_gate(res, None, TABLE) is res
    assert apply_shape_gate(res, {}, TABLE) is res


def test_unreliable_word_never_removed():
    res = make("UNCERTAIN", ["HUNGRY", "PLEASE"])
    assert apply_shape_gate(res, {"c": 0.9}, TABLE).top_word in ("HUNGRY", "PLEASE")
    r2 = apply_shape_gate(make("UNCERTAIN", ["HUNGRY", "SORRY"]), {"open": 0.9}, TABLE)
    assert "HUNGRY" in [c.word_key for c in r2.candidates]


def test_all_conflicting_returns_unchanged():
    res = make("UNCERTAIN", ["SORRY", "SORRY"])
    assert apply_shape_gate(res, {"open": 0.95}, TABLE) is res


def test_evidence_and_table_learning():
    classes = ["a", "b", "c", "s"]
    probs = np.tile(np.array([0.3, 0.0, 0.0, 0.7], dtype=np.float32), (20, 1))
    ev = boundary_group_evidence(probs, classes)
    assert abs(ev["fist"] - 1.0) < 1e-5
    t = learn_word_table({"X": [ev] * 6, "Y": [{"open": 0.5, "c": 0.5}] * 6})
    assert t["X"].reliable and t["X"].allowed == {"fist"}
    assert not t["Y"].reliable
