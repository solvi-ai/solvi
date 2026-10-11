"""solvi.experimental.relations.RulesFirst: rules first, a model second. The composition table answers when it derives
(the model is never called); when the table refuses ("disagree", "underivable") the fallback answers the question
rendered in words from the chains, among the given options; with no fallback the refusal stays a refusal. Every answer
says which path answered, is journal-ready (`to_record`) and replays from its record — the rules path by deriving
again, the model path from the recorded output, never calling the model."""
from __future__ import annotations

import json
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import (CompositionTable, Derivation, RulesFirst, RulesFirstAnswer,
                                              decider_fallback, render_chains)

# a planted algebra: a ∘ b → c, c ∘ a → b, b ∘ a → a; a ∘ a → c added in DISAGREE so that (a, b, a) has two bracketings
# giving different relations; (b, b) has no cell (underivable).
PLANTED = [(("a", "b"), "c")] * 2 + [(("c", "a"), "b")] * 2 + [(("b", "a"), "a")] * 2
DISAGREE = PLANTED + [(("a", "a"), "c")] * 2
OPTIONS = ["a", "b", "c"]


class Stub:
    """A counting fallback: answers `answer` with `confidence`, records every call."""

    def __init__(self, answer="b", confidence=0.7):
        self.answer, self.confidence, self.calls = answer, confidence, []

    def __call__(self, question, options):
        self.calls.append((question, list(options)))
        return self.answer, self.confidence


def tripwire(question, options):
    raise AssertionError("the model was called")


@pytest.fixture
def table():
    return CompositionTable().fit(DISAGREE)


def test_derived_answers_by_rules_and_never_calls_the_model(table):
    stub = Stub()
    rf = RulesFirst(table, fallback=stub)
    a = rf.answer(("a", "b"), OPTIONS)
    assert isinstance(a, RulesFirstAnswer)
    assert (a.value, a.who, a.reason) == ("c", "rules", "derived")
    assert a.derivation.relation == "c" and [(c.left, c.right) for c in a.derivation.cells] == [("a", "b")]
    assert a.model_output is None and a.confidence is None          # rules give no probability
    assert stub.calls == []


def test_disagree_goes_to_the_model_once_with_the_rendered_question(table):
    stub = Stub("b", 0.7)
    rf = RulesFirst(table, fallback=stub)
    a = rf.answer(("a", "b", "a"), OPTIONS)
    assert (a.value, a.who, a.reason) == ("b", "model", "disagree")
    assert a.derivation.reason == "disagree" and a.derivation.candidates == {"b": 1, "c": 1}
    assert len(stub.calls) == 1
    q, opts = stub.calls[0]
    assert q == render_chains([("a", "b", "a")]) == a.question
    assert opts == OPTIONS and a.confidence == 0.7
    assert a.model_output == {"answer": "b", "confidence": 0.7}


def test_underivable_goes_to_the_model(table):
    stub = Stub("a", None)
    a = RulesFirst(table, fallback=stub).answer(("b", "b"), OPTIONS)
    assert (a.value, a.who, a.reason, a.confidence) == ("a", "model", "underivable", None)
    assert len(stub.calls) == 1


def test_no_fallback_keeps_the_refusal_with_its_reason(table):
    rf = RulesFirst(table)
    a = rf.answer(("a", "b", "a"), OPTIONS)
    assert (a.value, a.who, a.reason) == (None, None, "disagree")
    b = rf.answer(("b", "b"), OPTIONS)
    assert (b.value, b.who, b.reason) == (None, None, "underivable")
    assert not a.answered and not b.answered


def test_a_model_answer_outside_the_options_is_not_an_answer(table):
    stub = Stub("zebra", 0.9)
    a = RulesFirst(table, fallback=stub).answer(("b", "b"), OPTIONS)
    assert a.value is None and a.who == "model" and a.reason == "underivable"
    assert "not one of the options" in a.note and a.model_output["answer"] == "zebra"


def test_a_model_that_returns_none_refuses(table):
    a = RulesFirst(table, fallback=lambda q, o: None).answer(("b", "b"), OPTIONS)
    assert a.value is None and a.who == "model" and "refused" in a.note


def test_a_plain_string_from_the_fallback_is_an_answer_without_confidence(table):
    a = RulesFirst(table, fallback=lambda q, o: "c").answer(("b", "b"), OPTIONS)
    assert (a.value, a.confidence) == ("c", None)


def test_several_chains_of_one_query_are_pooled(table):
    stub = Stub()
    rf = RulesFirst(table, fallback=stub)
    a = rf.answer([("a", "b"), ("a", "b")], OPTIONS)            # two paths, both a ∘ b → c
    assert (a.value, a.who) == ("c", "rules") and a.chains == (("a", "b"), ("a", "b"))
    b = rf.answer([("a", "b"), ("b", "a")], OPTIONS)            # c vs a: the pooled bracketings disagree
    assert (b.who, b.reason) == ("model", "disagree") and len(stub.calls) == 1


def test_a_callable_rules_source(table):
    def rules(chains):
        return Derivation(tuple(chains), "c", "derived", {"c": 1})
    a = RulesFirst(rules, fallback=tripwire).answer(("x", "y"), OPTIONS)
    assert (a.value, a.who) == ("c", "rules")


def test_to_record_is_plain_json_and_names_the_path(table):
    rf = RulesFirst(table, fallback=Stub("b", 0.7))
    r1 = rf.answer(("a", "b"), OPTIONS).to_record()
    r2 = rf.answer(("a", "b", "a"), OPTIONS).to_record()
    for r in (r1, r2):
        assert json.loads(json.dumps(r)) == r
        assert r["by"] == "relations:RulesFirst" and r["options"] == OPTIONS
    assert r1["who"] == "rules" and r1["derivation"]["cells"] == [["a", "b", "c"]] and r1["model_output"] is None
    assert r2["who"] == "model" and r2["model_output"] == {"answer": "b", "confidence": 0.7}
    assert r2["question"] == render_chains([("a", "b", "a")]) and r2["derivation"]["reason"] == "disagree"


def test_replay_from_the_record_never_calls_the_model(table):
    rf = RulesFirst(table, fallback=Stub("b", 0.7))
    recs = [rf.answer(c, OPTIONS).to_record() for c in (("a", "b"), ("a", "b", "a"), ("b", "b"))]
    again = RulesFirst(table, fallback=tripwire)
    for r in recs:
        rep = again.replay(r)
        assert rep.ok and rep.from_record, rep.problems
    rep = RulesFirst(table).replay(recs[1])                     # no fallback at all: still replays from the record
    assert rep.ok


def test_replay_catches_an_edited_record_and_a_changed_table(table):
    rf = RulesFirst(table, fallback=Stub("b", 0.7))
    r = rf.answer(("a", "b", "a"), OPTIONS).to_record()
    bad = dict(r, value="c")
    assert not rf.replay(bad).ok
    bad = dict(r, question="something else")
    assert not rf.replay(bad).ok
    r_rules = rf.answer(("a", "b"), OPTIONS).to_record()
    other = CompositionTable().fit([(("a", "b"), "a")] * 2)     # a ∘ b → a now
    rep = RulesFirst(other).replay(r_rules)
    assert not rep.ok and any("derive" in p for p in rep.problems)
    rep = RulesFirst(other).replay(r)                           # the table derives (a, b, a) differently now
    assert not rep.ok


def test_stats_count_paths_per_reason(table):
    rf = RulesFirst(table, fallback=Stub("b"))
    for c in (("a", "b"), ("c", "a"), ("a", "b", "a"), ("b", "b")):
        rf.answer(c, OPTIONS)
    s = rf.stats
    assert (s["rules"], s["model"], s["refused"]) == (2, 2, 0)
    assert s["by_reason"] == {"derived": {"rules": 2}, "disagree": {"model": 1}, "underivable": {"model": 1}}
    plain = RulesFirst(table)
    plain.answer(("b", "b"), OPTIONS)
    assert plain.stats["refused"] == 1 and plain.stats["by_reason"] == {"underivable": {"refused": 1}}
    s["rules"] = 99                                             # a copy: the counters are not changed from outside
    assert rf.stats["rules"] == 2


def test_deterministic(table):
    a = RulesFirst(table, fallback=Stub("b", 0.7))
    b = RulesFirst(table, fallback=Stub("b", 0.7))
    for c in (("a", "b"), ("a", "b", "a"), ("b", "b")):
        assert a.answer(c, OPTIONS).to_record() == b.answer(c, OPTIONS).to_record()


def test_render_default_on_a_three_chain():
    assert render_chains([("father", "sister", "son")]) == \
        "B is the father of A. C is the sister of B. D is the son of C. What is D to A?"
    two = render_chains([("father", "sister"), ("mother", "sister")])
    assert two == "B is the father of A. D is the sister of B. C is the mother of A. D is the sister of C. What is D to A?"


def test_a_custom_render_is_used(table):
    stub = Stub()
    rf = RulesFirst(table, fallback=stub, render=lambda chains, head, tail: f"{len(chains)} chain(s) from {head}")
    a = rf.answer(("b", "b"), OPTIONS)
    assert stub.calls[0][0] == "1 chain(s) from A" == a.question


def test_rules_answers_have_no_question_rendered(table):
    calls = []

    def render(chains, head, tail):
        calls.append(chains)
        return "q"
    RulesFirst(table, fallback=Stub(), render=render).answer(("a", "b"), OPTIONS)
    assert calls == []


def test_bad_arguments(table):
    with pytest.raises(TypeError):
        RulesFirst(42)
    with pytest.raises(ValueError):
        RulesFirst(table).answer([], OPTIONS)


def test_decider_fallback_wraps_a_decide_model():
    from solvi.core.catalog import Decision

    class FakeModel:
        def __init__(self):
            self.calls = []

        def decide(self, text, task, options, **kw):
            self.calls.append((text, task, list(options)))
            return Decision("b", {"a": 0.1, "b": 0.6, "c": 0.3})

    m = FakeModel()
    fb = decider_fallback(m, task="Which relation?")
    ans, conf, raw = fb("a question", OPTIONS)
    assert (ans, conf) == ("b", 0.6) and raw == {"probs": {"a": 0.1, "b": 0.6, "c": 0.3}}
    assert m.calls == [("a question", "Which relation?", OPTIONS)]
    table = CompositionTable().fit(DISAGREE)
    a = RulesFirst(table, fallback=fb).answer(("b", "b"), OPTIONS)
    assert a.value == "b" and a.model_output["raw"] == {"probs": {"a": 0.1, "b": 0.6, "c": 0.3}}
    assert RulesFirst(table, fallback=tripwire).replay(a.to_record()).ok
