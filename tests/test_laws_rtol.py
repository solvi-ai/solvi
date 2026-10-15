"""solvi.experimental.laws: a relative tolerance (rtol=) next to tol= / same= in split, discriminate and
admit_by_probes, and tol / rtol / same on LawSet itself, so step, test, observe and check compare a prediction with an
observation within it (a numeric effect measured with relative noise). Synthetic worlds: the law is known."""
from __future__ import annotations

import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition, discriminate, split
    from solvi.experimental.laws._probes import sameness

ALWAYS = lambda s, act: True                                             # noqa: E731


def test_sameness_rtol_is_relative_and_symmetric():
    eq = sameness(rtol=1e-6)
    assert eq(1e6, 1e6 + 0.5) and eq(1e6 + 0.5, 1e6)                      # 0.5 on a million: within 1e-6
    assert not eq(1.0, 1.5)
    assert eq((1e6, "a"), (1e6 + 0.5, "a")) and not eq((1e6, "a"), (1e6 + 0.5, "b"))
    both = sameness(tol=1e-3, rtol=1e-6)                                  # |a − b| ≤ tol + rtol·max(|a|, |b|)
    assert both(0.0, 5e-4) and both(1e6, 1e6 + 1.0) and not both(1e6, 1e6 + 2.0)
    assert sameness() is None                                             # the default: exact equality
    with pytest.raises(ValueError):
        sameness(rtol=1e-6, same=lambda a, b: True)
    with pytest.raises(ValueError):
        sameness(rtol=-1.0)


def test_split_and_discriminate_take_rtol():
    # predictions over six orders of magnitude: one absolute tol either merges the small ones or splits the large ones
    probes = [(10.0 ** k, "y") for k in range(-3, 4)]
    a = lambda s, act: 3.0 * s                                           # noqa: E731
    b = lambda s, act: 3.0 * s * (1 + 1e-9)                              # the same law up to rounding
    c = lambda s, act: 3.003 * s                                         # another law, 0.1 % off
    assert discriminate(a, b, ALWAYS, probes, rtol=1e-6) is None
    assert discriminate(a, c, ALWAYS, probes, rtol=1e-6) is not None
    assert discriminate(a, c, ALWAYS, probes, tol=1e-2) is not None      # absolute: told apart only at large s
    pr = split([a, b, c], ALWAYS, probes, rtol=1e-6)
    assert pr is not None and pr.score == 1


def scale(noise, seed=0):
    g = Grammar([Atom("box", lambda s: True, feature="box")], k=1)
    cands = [Candidate("y", ("box",), c) for c in (1.0, 2.0, 3.0)]
    probes = [(10.0 ** (3 + i / 20), "y") for i in range(40)]            # 1e3 … 1e5
    rng = random.Random(seed)
    ask = lambda s, act: 2.0 * s * (1 + noise * rng.gauss(0, 1))         # noqa: E731
    return g, cands, probes, ask


def test_admit_by_probes_with_rtol_admits_through_relative_noise():
    g, cands, probes, ask = scale(1e-6)
    absolute = LawSet(g, apply=lambda e, s: e * s, base=lambda s, a: 0.0).admit_by_probes(
        cands, ask, probes=probes, tol=1e-4)
    assert absolute.status == "undetermined"                             # the noise is 2e-3 … 0.2 > tol
    adm = LawSet(g, apply=lambda e, s: e * s, base=lambda s, a: 0.0).admit_by_probes(
        cands, ask, probes=probes, rtol=1e-4)
    assert adm.status == "accepted" and adm.law.effect == 2.0
    # the set's own comparison is the default for admit_by_probes
    own = LawSet(g, apply=lambda e, s: e * s, base=lambda s, a: 0.0, rtol=1e-4).admit_by_probes(
        cands, ask, probes=probes)
    assert own.status == "accepted" and own.law.effect == 2.0


def noisy_world(rtol=None, n=60, noise=1e-7):
    """y = 2.5·x measured with relative noise; the effect proposed from a row is obs / x rounded to 4 digits."""
    g = Grammar([Atom("box", lambda s: True, feature="box")], k=1)
    rng = random.Random(1)
    kw = {} if rtol is None else {"rtol": rtol}
    ls = LawSet(g, apply=lambda e, s: e * s["x"], base=lambda s, a: 0.0,
                effects=lambda s, obs: [round(obs / s["x"], 4)], min_support=3, **kw)
    b = Buffer(g)
    rows = []
    for i in range(n):
        x = 10.0 ** rng.uniform(0, 5)
        rows.append(Transition({"x": x}, "y", 2.5 * x * (1 + noise * rng.gauss(0, 1)), group=f"g{i % 6}"))
    return ls, b, rows


def test_lawset_rtol_learns_and_keeps_a_law_over_noisy_measurements():
    exact, b, rows = noisy_world()
    for t in rows:
        exact.step(t, b, remember=True)
    assert exact.laws == []                                              # exact ==: every row is a mismatch
    ls, b, rows = noisy_world(rtol=1e-5)
    steps = [ls.step(t, b, remember=True) for t in rows]
    assert [law.effect for law in ls.laws] == [2.5]
    first = next(i for i, s in enumerate(steps) if s.accepted)
    assert not any(s.mismatch for s in steps[first + 1:])                    # no mismatch once the law is in
    assert ls.retired == [] and ls.check(b) == []                        # its running precision stays 1
    law = ls.laws[0]
    assert law.hits_after == law.n and law.held[0] == law.held[1]


def test_lawset_comparison_arguments_are_checked():
    g = Grammar([Atom("box", lambda s: True, feature="box")], k=1)
    with pytest.raises(ValueError):
        LawSet(g, tol=1e-3, same=lambda a, b: a == b)
    assert LawSet(g).same is None                                        # default: exact equality, as before


def test_conflict_uses_the_sets_comparison():
    g = Grammar([Atom("a", lambda s: True, feature="a"), Atom("b", lambda s: True, feature="b")], k=1)
    ls = LawSet(g, rtol=1e-6, disagree="abstain")
    ls.add(Candidate("y", ("a",), 1.0))
    ls.add(Candidate("y", ("b",), 1.0 + 1e-9))
    assert ls.conflict({}, "y") is None                                   # within rtol: they agree
    exact = LawSet(g, disagree="abstain")
    exact.add(Candidate("y", ("a",), 1.0))
    exact.add(Candidate("y", ("b",), 1.0 + 1e-9))
    assert exact.conflict({}, "y") is not None
