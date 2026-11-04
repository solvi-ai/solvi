"""Explicit cause priority is separate from answer-option strictness and scheduling order."""
import warnings

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.provenance import catalog_fingerprint, part_fingerprint


def alternative(replacement: str) -> str:
    return replacement


def make(*, priority=10, second=0, combine="first", then="allow", cache=None):
    cat = Catalog(combine=combine)

    @cat.check(hard=True, then={"q": "deny"}, precedence=second)
    def earlier(low: bool) -> bool:
        return low

    @cat.check(hard=True, then={"q": then}, precedence=priority)
    def important(high: bool) -> bool:
        return high

    @cat.rule("q")
    def decide(payload: str) -> str:
        return payload

    return System(cat, [Question("q", "?", Answer.choice(["allow", "deny"]))], cache=cache)


FACTS = {"low": False, "high": False, "payload": "allow", "replacement": "allow"}


@pytest.mark.parametrize("combine", ["first", "strictest"])
@pytest.mark.parametrize("then,want", [("allow", "allow"), (alternative, "allow"), (None, None)])
def test_priority_wins_over_option_order_and_learned_schedule(combine, then, want):
    s = make(combine=combine, then=then)
    for learned in (False, True):
        if learned:
            s.learn_order([dict(FACTS, high=True)] * 10)
        for early in (True, False):
            r = s.ask(FACTS, store=False, early_exit=early)
            assert r["q"].answer == want and r["q"].source == "important"
            assert r["q"].status == ("abstain" if want is None else "forced")
            assert r.trace.replay(s, r.flow)["ok"]
            assert {x.name for x in r.trace.records} >= {"earlier", "important"}


@pytest.mark.parametrize("priority,second,combine,want", [(0, None, "first", "deny"),
    (3, 3, "strictest", "deny"), (-5, -10, "strictest", "allow")])
def test_equal_priorities_retain_combine_and_negative_priorities_are_ordered(priority, second, combine, want):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert make(priority=priority, second=second, combine=combine).ask(FACTS, store=False)["q"].answer == want


@pytest.mark.parametrize("bad", [True, False, 1.0, float("nan"), float("inf"), "1", [], {}])
def test_invalid_precedence_fails_before_catalog_mutation(bad):
    c = Catalog()
    with pytest.raises(ValueError, match="precedence"):
        c.check(hard=True, precedence=bad)
    assert not c.parts


def test_priority_only_applies_to_hard_checks():
    with pytest.raises(ValueError, match="hard=True"):
        Catalog().check(precedence=0)


def test_priority_change_invalidates_cache_fingerprints_and_old_trace_replay():
    s = make(cache=8)
    old = s.ask(FACTS, store=False)
    assert old["q"].answer == "allow"
    fp = catalog_fingerprint(s.catalog)["fp"]
    s.catalog.parts["important"].precedence = -1
    assert catalog_fingerprint(s.catalog)["fp"] != fp
    assert s.ask(FACTS, store=False)["q"].answer == "deny"
    assert not old.trace.replay(s, old.flow)["ok"]


def test_zero_and_unspecified_priorities_have_identical_fingerprints():
    assert part_fingerprint(make(priority=None).catalog.parts["important"]) == part_fingerprint(
        make(priority=0).catalog.parts["important"])


def test_renaming_preserves_priority_and_selection():
    s = make().rename(inputs={"high": "higher"}, answers={"q": "renamed"})
    assert s.catalog.parts["important"].precedence == 10
    assert s.ask({k: v for k, v in {**FACTS, "higher": False}.items() if k != "high"}, store=False)["renamed"].answer == "allow"


def test_unavailable_hard_check_prevents_permissive_priority_selection():
    s = make()
    r = s.ask({"low": False, "payload": "allow"}, store=False)
    assert r["q"].status == "abstain" and r["q"].answer is None


def test_a_passing_high_priority_check_does_not_override_a_false_lower_priority_check():
    s = make()
    assert s.ask(dict(FACTS, high=True), store=False)["q"].answer == "deny"


@pytest.mark.parametrize("early", [True, False])
def test_unavailable_check_reason_is_preserved_with_priority(early):
    s = make()

    def broken(high: bool) -> bool:
        raise RuntimeError("required source offline")

    s.catalog.parts["important"].func = broken
    r = s.ask(FACTS, store=False, early_exit=early)
    assert r["q"].status == "abstain"
    assert "required source offline" in r["q"].why


def test_hard_check_without_then_can_win_as_abstention():
    s = make()
    c = s.catalog

    @c.check(hard=True, precedence=20)
    def no_reason(extra: bool) -> bool:
        return extra

    q = Question("q", "?", Answer.choice(["allow", "deny"]), requires=["no_reason"])
    s = System(c, [q])
    r = s.ask({**FACTS, "extra": False}, store=False)
    assert r["q"].status == "abstain" and r["q"].source == "no_reason"
    assert r.trace.replay(s, r.flow)["ok"]
