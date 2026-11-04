"""Stateless comparator diagnostic; unchanged acceptance and arithmetic."""

import pytest
from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


def setup(null="base"):
    g = Grammar([Atom("new", lambda s: s["day"] >= 30)], k=1)
    b = Buffer(g)
    for day in range(60):
        b.add(Transition({"day": day}, "x", "A" if day < 30 else ("C" if day % 9 == 4 else "B"), group=day))
    return LawSet(g, base=lambda s, a: "A", null=null, min_precision=0.85), b, Candidate("x", ("new",), "B")


def test_base_warning_stateless_and_arithmetic_unchanged():
    ls, b, c = setup()
    before = (b.tested, dict(ls.cost))
    w = ls.why_not(c, b)
    assert "including rows matching the candidate" in w["null_warning"]
    assert "may mix regimes" in w["null_warning"]
    assert (b.tested, ls.cost) == before and ls.why_not(c, b) == w
    assert (w["support"], w["hits"], w["null"]) == (30, 26, pytest.approx(26 / 60, abs=5e-5))
    assert w["passes"] == ls.passes(30, 26, tested=w["tested"], null=ls.base_rate(b, "x", "B"))
    assert len(ls.test([c], b)) == int(w["passes"])
    assert "null_warning" in ls.why_not(c, b)  # early active return still explains comparator


def test_numeric_null_no_warning_same_fixed_null_after_more_rows():
    ls, b, c = setup(0.05)
    w = ls.why_not(c, b)
    assert "null_warning" not in w and w["null"] == 0.05
    b.add(Transition({"day": 60}, "x", "B", group=60))
    later = ls.why_not(c, b)
    assert later["null"] == 0.05 and "null_warning" not in later
    assert (w["support"], w["hits"], w["need"]) == (30, 26, 5)


def test_excluded_group_and_empty_buffer_remain_readonly():
    ls, b, c = setup()
    w = ls.why_not(c, b, exclude=59)
    assert (w["support"], w["hits"]) == (29, 25)
    assert "group exclusion" in w["null_warning"] and b.tested == 0
    empty = Buffer(ls.grammar)
    w = ls.why_not(c, empty)
    assert w["support"] == 0 and "null_warning" in w and empty.tested == 0
