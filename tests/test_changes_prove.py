"""Narrows / widens proved over a rule's text (Consistency(prove=True), the smt extra): known narrowing and widening
pairs, a construct outside the subset → "unknown" and the probes measure instead, and a clear message without z3."""
from __future__ import annotations

import sys
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import Change, Consistency, Pipeline, prove_direction

LIMIT = 1000


def ok_old(amount, country):
    return amount < LIMIT and country in {"DE", "FR"}


def ok_new(amount, country):
    return amount < LIMIT + 500 and country in {"DE", "FR"}


PAIRS = [   # (in force, candidate, proved)
    ("def ok(amount): return amount < 1000", "def ok(amount): return amount < 1500", "widens"),
    ("def ok(amount): return amount < 1000", "def ok(amount): return amount <= 999", "narrows"),
    ("lambda age, country: age >= 18 and country in ('DE', 'FR')",
     "lambda age, country: age >= 18 and country in ('DE', 'FR', 'IT')", "widens"),
    ("lambda age, country: age >= 18 and country in ('DE', 'FR')",
     "lambda age, country: age >= 21 and country not in ('IT',)", "both"),
    ("tier != 'vip' or waiting_hours < 4", "tier != 'vip' or waiting_hours < 3", "narrows"),
    ("def f(x, y):\n    if x > 2 * y:\n        return None\n    return 'pay'",
     "def f(x, y):\n    z = 2 * y\n    return 'pay' if not x > z else None", "same"),
    ("def f(score):\n    if score < 0.5:\n        return 'abstain'\n    return 'yes'",
     "def f(score):\n    if score < 0.3:\n        return 'abstain'\n    return 'yes'", "widens"),
    (ok_old, ok_new, "widens"),
]


@pytest.mark.parametrize("before, after, proved", PAIRS)
def test_known_pairs_are_proved(before, after, proved):
    pytest.importorskip("z3")
    r = prove_direction(before, after)
    assert r["proved"] == proved, r
    for k in ("widens", "narrows"):                     # a witness is an input that shows it
        if proved in (k, "both"):
            assert f"witness_{k}" in r


def test_a_witness_really_shows_the_widening():
    pytest.importorskip("z3")
    r = prove_direction(ok_old, ok_new)
    w = r["witness_widens"]
    assert ok_new(**w) and not ok_old(**w)
    assert r["sorts"] == {"amount": "Real", "country": "String"}


def test_integer_facts_and_outcomes():
    pytest.importorskip("z3")
    assert prove_direction("x > 3", "x >= 4")["proved"] == "narrows"                       # over reals: 3.5
    assert prove_direction("x > 3", "x >= 4", types={"x": int})["proved"] == "same"        # over integers
    prio = ("def priority(urgency, anger):\n    score = 2 * urgency + anger\n"
            "    return 'high' if score >= {t} else 'normal' if score >= 1 else 'low'")
    a, b = prio.replace("{t}", "3"), prio.replace("{t}", "2")
    assert prove_direction(a, b, outcome="high")["proved"] == "widens"
    assert prove_direction(a, b, outcome="normal")["proved"] == "narrows"
    assert prove_direction(a, b, outcome="low")["proved"] == "same"


@pytest.mark.parametrize("after, why", [
    ("lambda amount, items: len(items) > 3", "Call"),
    ("lambda amount, x: amount < limit", "not an input"),
    ("def f(amount):\n    for i in range(3):\n        pass\n    return True", "For"),
    ("lambda amount, d: d['k'] > amount", "Subscript"),
])
def test_an_unsupported_construct_is_unknown(after, why):
    pytest.importorskip("z3")
    r = prove_direction("lambda amount: amount < 10", after)
    assert r["proved"] == "unknown" and why in r["reason"], r


def test_without_z3_the_message_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "z3", None)                        # import z3 → ImportError
    with pytest.raises(ImportError, match=r"solvi\[smt\]"):
        prove_direction("x > 1", "x > 2")
    with pytest.raises(ImportError, match=r"solvi\[smt\]"):
        Consistency(prove=True, before="x > 1")
    Consistency()                                                       # the default does not need it


# ─────────────────────────────────────────────── in the pipeline
class Threshold:
    """A target that answers: below the threshold it hands over (None)."""
    artifact = "update"

    def __init__(self, t=0.5):
        self.t = t

    def answerer(self, change, ctx):
        t = self.t if change is None else change.payload["t"]
        return lambda x: None if x["score"] < t else "yes"

    def promote(self, change, measured):
        before, self.t = self.t, change.payload["t"]
        return {"t": self.t, "before": before}

    def rollback(self, ref, by=None, why=None):
        self.t = ref["before"]


RULE = "def answer(score):\n    return None if score < {t} else 'yes'"
OUTSIDE = [{"id": f"o{i}", "score": s} for i, s in enumerate((0.1, 0.2, 0.3, 0.4))]


def move(t, code=None):
    return Change("outcome", artifact="update", payload={"t": t, "fingerprint": f"t={t}"}, proposer="self",
                  code=RULE.format(t=t) if code is None else code)


def pipe_of(tmp_path, stages, name="c.jsonl"):
    return Pipeline(list(stages), str(tmp_path / name), targets=[Threshold(0.5)], approval={"update": "gate"},
                    refusals=OUTSIDE[:1])


def consistency_of(res):
    return next(v for v in res.verdicts if v.stage == "consistency").measured


def test_a_widening_proved_over_the_text_is_held_for_a_person(tmp_path):
    pytest.importorskip("z3")
    claim = lambda change, ctx: "narrows"                                          # noqa: E731
    res = pipe_of(tmp_path, [Consistency(classify=claim, prove=True, before=RULE.format(t=0.5))]).propose(move(0.3))
    assert res.status == "held" and res.stage == "approval"
    m = consistency_of(res)
    assert m["direction"] == "widens" and m["claimed"] == "narrows" and m["proof"]["proved"] == "widens"
    assert 0.3 <= m["proof"]["witness_widens"]["score"] < 0.5


def test_a_narrowing_proved_over_the_text_is_promoted_without_probes(tmp_path):
    pytest.importorskip("z3")

    def probes(change, ctx):
        raise AssertionError("a decided proof does not ask the probes")
    res = pipe_of(tmp_path, [Consistency(prove=True, before=RULE.format(t=0.5), probes=probes)]).propose(move(0.7))
    assert res.promoted
    m = consistency_of(res)
    assert m["direction"] == "narrows" and m["proof"]["proved"] == "narrows" and "probes" not in m


def test_an_unknown_proof_falls_back_to_the_probes(tmp_path):
    pytest.importorskip("z3")
    opaque = "def answer(score):\n    return None if round(score, 1) < 0.3 else 'yes'"
    res = pipe_of(tmp_path, [Consistency(prove=True, before=RULE.format(t=0.5), probes=OUTSIDE)]).propose(
        move(0.3, code=opaque))
    m = consistency_of(res)
    assert m["proof"]["proved"] == "unknown" and "Call" in m["proof"]["reason"]
    assert m["direction"] == "widens" and m["widened"] == 2 and m["probes"] == 4      # measured instead
    assert res.status == "held"


def test_prove_as_a_callback_gives_both_rules(tmp_path):
    pytest.importorskip("z3")
    pair = lambda change, ctx: (RULE.format(t=0.5), RULE.format(t=change.payload["t"]))   # noqa: E731
    res = pipe_of(tmp_path, [Consistency(prove=pair)]).propose(move(0.5, code=""))
    assert consistency_of(res)["proof"]["proved"] == "same" and consistency_of(res)["direction"] == "same"
