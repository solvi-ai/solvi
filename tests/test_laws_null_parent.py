"""solvi.experimental.laws: LawSet(null="parent") — an exception to an active law is tested against the effect's rate
on the rows of the law it refines, not against the effect's share over every row of the action.

Synthetic worlds only: every law is known, so a false accept is known exactly."""
from __future__ import annotations

import itertools
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


# ───────────────────────────────────────────── a known exception: q → block, except q & e → pass
def exception_world(n_groups):
    """Per group: one exception row (q & e → pass), four parent rows (q & ¬e → block) and four rows without q (three
    pass, the one with e blocks, so `e` alone is not a law). The base rate of pass is 4/9."""
    g = Grammar([Atom("q", lambda s: s["q"], feature="q"), Atom("e", lambda s: s["e"], feature="e"),
                 Atom("d", lambda s: s["d"], feature="d")], k=3)
    b = Buffer(g)
    for gi in range(n_groups):
        rows = [({"q": True, "e": True, "d": gi % 2 == 0}, "pass")]
        rows += [({"q": True, "e": False, "d": j % 2 == 0}, "block") for j in range(4)]
        rows += [({"q": False, "e": j == 0, "d": j % 2 == 0}, "block" if j == 0 else "pass") for j in range(4)]
        for j, (s, o) in enumerate(rows):
            b.add(Transition(s, "go", o, group=f"g{gi}", id=f"{gi}.{j}"))
    return g, b


def all_conditions(effect="pass"):
    return [Candidate("go", c, effect) for r in (1, 2, 3) for c in itertools.combinations(("q", "e", "d"), r)]


def groups_needed(null):
    """The fewest supporting groups (the proposing group held out) at which the exception is accepted."""
    for n in range(2, 40):
        g, b = exception_world(n + 1)
        ls = LawSet(g, null=null)
        parent = ls.add(Candidate("go", ("q",), "block"))
        got = ls.test(all_conditions(), b, exclude=f"g{n}")
        if got:
            return n, got[0], parent
    return None, None, None


def test_an_exception_needs_fewer_supporting_groups_against_its_parent():
    nb, law_b, _ = groups_needed("base")
    np_, law_p, parent = groups_needed("parent")
    assert law_b.condition == law_p.condition == ("q", "e")
    assert (nb, np_) == (8, 5)                         # base: null 4/9; parent: min_support binds
    assert law_b.null == pytest.approx(4 / 9) and law_b.parent is None
    assert law_p.null == LawSet.NULL_FLOOR and law_p.parent == parent.id
    assert law_p.need < law_b.need


def test_parent_of_picks_the_most_specific_refined_law_with_another_effect():
    g, b = exception_world(6)
    ls = LawSet(g, null="parent")
    general = ls.add(Candidate("go", (), "pass"))
    q = ls.add(Candidate("go", ("q",), "block"))
    ls.add(Candidate("go", ("q", "d"), "pass"))          # not a subset of (q, e)
    assert ls.parent_of(Candidate("go", ("q", "e"), "pass")) is q
    assert ls.parent_of(Candidate("go", ("q", "e"), "block")) is general     # same effect as q: q is skipped
    assert ls.parent_of(Candidate("go", ("q",), "block")) is general         # a proper subset only
    assert ls.parent_of(Candidate("stay", ("q", "e"), "pass")) is None       # another action
    null, par = ls.parent_rate(b, Candidate("go", ("q", "e"), "pass"))
    assert par is q and null == pytest.approx(max((0 + 1) / (24 + 2), LawSet.NULL_FLOOR))
    assert ls.why_not(Candidate("go", ("q", "e"), "pass"), b)["null"] == round(null, 4)


def test_without_a_parent_the_parent_null_is_the_base_null():
    for n in (4, 8, 12):
        g, b = exception_world(n)
        base, parent = LawSet(g, null="base"), LawSet(g, null="parent")
        for c in all_conditions() + all_conditions("block"):
            assert parent.null_for(b, c, "g0") == base.null_for(b, c, "g0") == base.base_rate(b, "go", c.effect, "g0")
        got_b = [x.text() for x in base.test(all_conditions(), b, exclude="g0", accept=3)]
        g2, b2 = exception_world(n)
        got_p = [x.text() for x in LawSet(g2, null="parent").test(all_conditions(), b2, exclude="g0", accept=3)]
        assert got_b == got_p


def test_the_default_null_is_unchanged_and_the_option_is_validated():
    g, _ = exception_world(1)
    assert LawSet(g).null == "base"
    assert LawSet(g, null="parent").null == "parent"
    with pytest.raises(ValueError):
        LawSet(g, null="parents")
    with pytest.raises(ValueError):
        LawSet(g, null="parent").passes(10, 10, 5)                # an estimated null must be passed in


# ───────────────────────────────────────────── a false exception: noise in the parent's rows
M = 12


def noisy_world(seed, p, n_groups=10, per=8):
    """q → block with `p` of q's rows passing at random (no exception exists); without q, pass at 85%."""
    rng = random.Random(seed)
    g = Grammar([Atom("q", lambda s: s["q"], feature="q")]
                + [Atom(f"a{i}", (lambda i: lambda s: s["a"][i])(i), feature=f"a{i}") for i in range(M)], k=3)
    b = Buffer(g)
    for gi in range(n_groups):
        for j in range(per):
            q = rng.random() < 0.5
            a = [rng.random() < 0.25 for _ in range(M)]
            eff = ("pass" if rng.random() < p else "block") if q else ("pass" if rng.random() < 0.85 else "block")
            b.add(Transition({"q": q, "a": a}, "go", eff, group=f"g{gi}", id=f"{gi}.{j}"))
    return g, b


def exceptions():
    out = [Candidate("go", ("q", f"a{i}"), "pass") for i in range(M)]
    return out + [Candidate("go", ("q", f"a{i}", f"a{j}"), "pass") for i, j in itertools.combinations(range(M), 2)]


@pytest.mark.parametrize("p, min_precision, learned", [(0.08, 0.9, True), (0.3, 0.6, False), (0.3, 0.6, True)])
def test_a_false_exception_is_rarely_accepted(p, min_precision, learned):
    """78 candidate exceptions on 80 rows, no true exception; the parent added by hand or learned on the same buffer.
    The stated bound is delta = 0.1 per buffer; over 1,000 seeds: 0 of 715, 2 of 1,000 and 1 of 905 buffers (the base
    null: 0 in each)."""
    trials = accepts = 0
    for seed in range(100):
        g, b = noisy_world(seed, p)
        ls = LawSet(g, null="parent", min_precision=min_precision, retire_below=min(0.8, min_precision))
        par = Candidate("go", ("q",), "block")
        if learned:
            if not ls.test([par], b, exclude="g0"):
                continue
        else:
            ls.add(par)
        trials += 1
        accepts += bool(ls.test(exceptions(), b, exclude="g0"))
    assert trials >= 60
    assert accepts <= 0.1 * trials
