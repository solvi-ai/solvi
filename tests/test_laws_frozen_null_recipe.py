"""S234 fixed pre-change alternative: deterministic API/arithmetic proof, no coverage evaluation."""
import pytest

from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


def grammar():
    return Grammar([Atom("since:30", lambda s: s["day"] >= 30, feature="since")], k=1)


def lawset(g, null="base"):
    return LawSet(g, base=lambda s, a: "A", min_precision=0.85, retire_below=0.7,
                  min_support=2, null=null, delta=0.05, space=2)


def test_original_regime_rows_change_base_null_but_not_frozen_prechange_null():
    g = grammar()
    pre, growing = Buffer(g), Buffer(g)
    for day in range(30):
        row = Transition({"day": day}, "x", "A", group=day)
        pre.add(row)
        growing.add(row)
    adaptive = lawset(g)
    frozen_value = adaptive.base_rate(pre, "x", "B")
    assert frozen_value == adaptive.NULL_FLOOR == 0.05
    fixed = lawset(g, null=frozen_value)
    candidate = Candidate("x", ("since:30",), "B")
    trail = []
    for day in range(30, 60):
        obs = "C" if day % 9 == 4 else "B"  # exact S234 original observation rule after day30
        growing.add(Transition({"day": day}, "x", obs, group=day))
        if day in (35, 45, 57):
            before = growing.tested
            a, f = adaptive.why_not(candidate, growing), fixed.why_not(candidate, growing)
            assert growing.tested == before  # read-only diagnostics do not spend candidate budget
            assert f["null"] == 0.05 and fixed.null == frozen_value
            assert a["null"] == round(adaptive.base_rate(growing, "x", "B"), 4)
            assert (a["support"], a["hits"], a["net"]) == (f["support"], f["hits"], f["net"])
            assert a["support"] == day - 29  # condition/action applicability, not all-buffer rows
            assert a["need"] >= f["need"]
            trail.append(a["null"])
    assert trail[0] < trail[1] < trail[2]
    assert adaptive.base_rate(pre, "x", "B") == frozen_value  # original baseline buffer untouched


def test_frozen_rate_is_action_effect_and_exclusion_specific_not_candidate_condition_rate():
    g = grammar()
    pre = Buffer(g)
    for day in range(10):
        pre.add(Transition({"day": day}, "x", "B" if day < 2 else "A", group=day))
        pre.add(Transition({"day": day}, "other", "B", group=day + 100))
    ls = lawset(g)
    assert ls.base_rate(pre, "x", "B") == pytest.approx(0.2)
    assert ls.base_rate(pre, "x", "B", exclude=0) == pytest.approx(1 / 9)
    assert ls.base_rate(pre, "other", "B") == ls.min_precision  # existing cap, not an exact empirical probability
    candidate = Candidate("x", ("since:30",), "B")
    assert ls.score(candidate, pre)[:2] == (0, 0)
    fixed = lawset(g, null=ls.base_rate(pre, "x", "B"))
    pre.add(Transition({"day": 30}, "x", "B", group=30))
    assert ls.base_rate(pre, "x", "B") == pytest.approx(3 / 11)
    assert fixed.why_not(candidate, pre)["null"] == 0.2


def test_known_numeric_null_need_and_acceptance_arithmetic_stay_fixed():
    ls = lawset(grammar(), null=0.25)
    needed = ls.need_at(tested=1, null=0.25, precision=1.0)
    assert needed is not None and needed >= ls.min_support
    assert ls.passes(needed, needed, tested=1)
    assert not ls.passes(needed - 1, needed - 1, tested=1)
    assert not ls.passes(needed, 0, tested=1)
    assert ls.need_at(tested=1, null=0.25, precision=0.25) is None
    # Freezing a null does not remove the separate observed-precision requirement.
    assert not ls.passes(100, 84, tested=1)
