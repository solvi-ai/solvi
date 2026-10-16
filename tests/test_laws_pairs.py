"""solvi.experimental.laws: admission by probes over law sets with up to two new laws (max_new=2).

Synthetic worlds only: every law is known, so a wrong admission is known exactly."""
from __future__ import annotations

import itertools
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition

NF = 10
STATES = list(itertools.product((0, 1), repeat=NF))


def bits(k=2):
    return Grammar([Atom(f"x{i}", (lambda s, i=i: s[i] == 1), feature=f"f{i}") for i in range(NF)], k=k)


def wrong(g, laws, world):
    return any(law.holds(g.true(s, "go")) and world(s, "go") != law.effect for law in laws for s in STATES)


def or_trials(seeds, **kw):
    world = lambda s, a: bool(s[0] or s[1])                              # noqa: E731 — two laws are missing
    acc = bad = both = 0
    for seed in seeds:
        rng = random.Random(seed)
        g = bits()
        ls = LawSet(g, base=lambda s, a: False)
        s0 = tuple([1, 1] + [rng.randint(0, 1) for _ in range(NF - 2)])
        pool = [(rng.choice(STATES), "go") for _ in range(200)]
        adm = ls.admit_by_probes(ls.mismatch(s0, "go", True), world, probes=pool, budget=8, seed=seed, **kw)
        if adm.status == "accepted":
            acc += 1
            bad += wrong(g, adm.laws, world)
            both += all(ls.predict(s, "go") == world(s, "go") for s in STATES)
        else:
            assert adm.law is None and adm.laws == [] and ls.laws == []
    return acc, bad, both


def test_with_two_laws_missing_no_wrong_law_is_admitted_on_the_known_trials():
    acc, bad, both = or_trials(range(100), max_new=2)
    assert bad == 0                                                      # the witness guard: 6 of 100
    assert acc >= 60                                                     # measured: 68 (13 with both laws)


def test_the_lone_survivor_the_witnesses_covered_is_refuted_by_the_pair():
    g = Grammar([Atom(f"x{i}", (lambda s, i=i: s[i] == 1), feature=f"f{i}") for i in range(3)], k=1)
    world = lambda s, a: bool(s[0] or s[1])                              # noqa: E731
    cands = [Candidate("go", ("x0",), True), Candidate("go", ("x1",), True), Candidate("go", ("x2",), True)]
    b = Buffer(g)
    b.extend([Transition((0, 1, 1), "go", True, group="log"), Transition((1, 0, 1), "go", True, group="log")])
    pool = [((0, 0, 1), "go"), ((1, 1, 0), "go"), ((1, 0, 0), "go"), ((0, 1, 0), "go")]
    ls = LawSet(g, base=lambda s, a: False)
    adm = ls.admit_by_probes(cands, world, probes=pool, buffer=b, budget=4, max_new=2)
    assert adm.status == "accepted"
    assert sorted(law.condition for law in adm.laws) == [("x0",), ("x1",)]
    assert adm.law is adm.laws[0] and len(ls.laws) == 2


def test_one_missing_law_is_admitted_alone():
    g = bits()
    world = lambda s, a: bool(s[3] and s[5])                             # noqa: E731
    rng = random.Random(3)
    s0 = tuple(1 if i in (3, 5) else rng.randint(0, 1) for i in range(NF))
    pool = [(s, "go") for s in rng.sample(STATES, 200)]
    ls = LawSet(g, base=lambda s, a: False)
    adm = ls.admit_by_probes(ls.mismatch(s0, "go", True), world, probes=pool, budget=12, max_new=2)
    assert adm.status == "accepted" and [law.condition for law in adm.laws] == [("x3", "x5")]


def test_budget_zero_and_the_null_alive_are_undetermined():
    g = bits()
    ls = LawSet(g, base=lambda s, a: False)
    s0 = (1,) * NF
    pool = [(s, "go") for s in STATES[:50]]
    adm = ls.admit_by_probes(ls.mismatch(s0, "go", True), lambda s, a: bool(s[0]), probes=pool, budget=0, max_new=2)
    assert adm.status == "undetermined" and adm.laws == [] and ls.laws == []


def test_the_smallest_hypothesis_is_accepted_where_every_other_agrees_with_it_where_it_speaks():
    g = Grammar([Atom(f"x{i}", (lambda s, i=i: s[i] == 1), feature=f"f{i}") for i in range(2)], k=1)
    world = lambda s, a: "A" if s[0] else ("B" if s[1] else "-")         # noqa: E731 — x0 first where both hold
    cands = [Candidate("go", ("x0",), "A"), Candidate("go", ("x1",), "B")]
    pool = [((a, b), "go") for a in (0, 1) for b in (0, 1)]
    for ties in ("earlier", "newer"):                                    # under "newer" the pair says B at (1, 1):
        adm = LawSet(g, base=lambda s, a: "-", ties=ties).admit_by_probes(cands, world, probes=pool, budget=4,
                                                                          max_new=2)
        assert adm.status == "accepted" and [law.condition for law in adm.laws] == [("x0",)]
        assert ((1, 1)) in [t.state for t in adm.asked]                  # asked there; x0 alone is right


def test_max_new_arguments():
    ls = LawSet(bits(), base=lambda s, a: False)
    with pytest.raises(ValueError):
        ls.admit_by_probes([], lambda s, a: False, probes=[], max_new=3)
    with pytest.raises(ValueError):
        ls.admit_by_probes([], lambda s, a: False, probes=[], max_new=2, order="random")
    one = ls.admit_by_probes([Candidate("go", ("x0",), True)], lambda s, a: bool(s[0]),
                             probes=[(s, "go") for s in STATES[::16]], budget=8, max_new=1)
    assert one.status == "accepted" and one.laws == [one.law]
