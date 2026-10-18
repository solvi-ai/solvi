"""solvi.experimental.laws: small gaps an agent met — `Grammar(min_k=1)` (no empty condition), an `Unknown` probe
answer (refutes nothing, counts as spent) and `admit_by_probes(siblings="undetermined")`."""
from __future__ import annotations

import itertools
import pickle
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, Buffer, CallableProposer, Candidate, Enumerate, Grammar, LawSet,
                                         RandomProposer, Transition, Unknown)

F = 6
ALL = [tuple(bits) for bits in itertools.product((False, True), repeat=F)]


def grammar(k=2, min_k=0):
    atoms = []
    for i in range(F):
        atoms.append(Atom(f"x{i}", (lambda s, i=i: s[i]), feature=f"x{i}"))
        atoms.append(Atom(f"~x{i}", (lambda s, i=i: not s[i]), feature=f"x{i}"))
    return Grammar(atoms, k=k, min_k=min_k)


def off(s, a):
    return "off"


def world(s, a):
    return "on" if s[0] else "off"


# ───────────────────────────────────────────── min_k
def test_min_k_drops_the_empty_condition_everywhere():
    g0, g1 = grammar(), grammar(min_k=1)
    true = g1.true(ALL[5], "do")
    assert () in g0.conditions(true) and () not in g1.conditions(true)
    assert set(g1.conditions(true)) == set(g0.conditions(true)) - {()}
    assert g1.sizes(true)[0] == 0 and g1.sizes(true)[1:] == g0.sizes(true)[1:]
    with pytest.raises(ValueError):
        g1.check(())
    assert g1.check(("x0",)) == ("x0",) and g0.check(()) == ()
    rng = random.Random(0)
    assert all(len(g1.random_condition(rng, 0)) >= 1 for _ in range(50))
    with pytest.raises(ValueError):
        grammar(k=2, min_k=3)
    with pytest.raises(ValueError):
        grammar(min_k=-1)


def test_min_k_in_the_law_set():
    g = grammar(min_k=1)
    ls = LawSet(g, base=off)
    cands = ls.mismatch(ALL[-1], "do", "on")
    assert cands and all(len(c.condition) >= 1 for c in cands)
    assert len(cands) == len(Enumerate().propose(ls, ALL[-1], "do", "on"))
    assert all(len(c.condition) >= 1 for c in RandomProposer(seed=1).propose(ls, ALL[-1], "do", "on"))
    p = CallableProposer(lambda s, a, o, true, g: [(), ("x0",)])
    assert [c.condition for c in p.propose(ls, ALL[-1], "do", "on")] == [("x0",)] and p.dropped == 1
    # every row passes: without min_k the empty condition would be the best law
    b = Buffer(g)
    for i, s in enumerate(ALL):
        b.add(Transition(s, "do", "on", group=i % 4, id=str(i)))
    assert LawSet(grammar(), base=off).test([Candidate("do", (), "on")], b, exclude=0)
    assert LawSet(g, base=off).test([Candidate("do", (), "on")], b, exclude=0) == []
    adm = LawSet(g, base=off).admit_by_probes([Candidate("do", (), "on"), Candidate("do", ("x0",), "on")], world,
                                              probes=[(s, "do") for s in ALL], budget=10)
    assert adm.status == "accepted" and adm.law.condition == ("x0",)


# ───────────────────────────────────────────── Unknown
def test_unknown_refutes_nothing_and_counts_as_spent():
    g = grammar(k=1)
    probes = [(s, "do") for s in ALL]
    cands = Enumerate().propose(LawSet(g, base=off), tuple([True] * F), "do", "on")
    ls = LawSet(g, base=off)
    adm = ls.admit_by_probes(cands, lambda s, a: Unknown, probes=probes, budget=5)
    assert adm.status == "undetermined" and adm.asked == [] and len(adm.unknown) == 5
    assert adm.refuted == 0 and ls.cost["probes"] == 5 and "budget" in adm.why
    assert all(t.observed is Unknown for t in adm.unknown)
    calls = []

    def flaky(s, a):                         # the first two probes go wrong; the rest answer
        calls.append(s)
        return Unknown if len(calls) <= 2 else world(s, a)
    ls = LawSet(g, base=off)
    adm = ls.admit_by_probes(cands, flaky, probes=probes, budget=30)
    assert adm.status == "accepted" and adm.law.condition == ("x0",) and len(adm.unknown) == 2
    assert ls.cost["probes"] == len(adm.asked) + 2
    assert {t.state for t in adm.unknown}.isdisjoint({t.state for t in adm.asked})   # not asked again
    assert adm.law.support == sum(1 for t in adm.asked if t.state[0])                 # Unknown supports nothing
    assert repr(Unknown) == "Unknown" and pickle.loads(pickle.dumps(Unknown)) is Unknown


def test_unknown_with_noise_and_with_two_laws():
    g = grammar(k=1)
    ls = LawSet(g, base=off)
    adm = ls.admit_by_probes([Candidate("do", ("x0",), "on"), Candidate("do", ("x1",), "on")],
                             lambda s, a: Unknown, probes=[(s, "do") for s in ALL], budget=6, noise=0.05)
    assert adm.status == "undetermined" and len(adm.unknown) == 6 and adm.asked == [] and ls.cost["probes"] == 6
    g2 = Grammar([Atom(f"x{i}", (lambda s, i=i: s[i] == 1), feature=f"f{i}") for i in range(3)], k=1)
    ls = LawSet(g2, base=lambda s, a: False)
    adm = ls.admit_by_probes([Candidate("go", (f"x{i}",), True) for i in range(3)], lambda s, a: Unknown,
                             probes=[((0, 0, 1), "go"), ((1, 1, 0), "go"), ((1, 0, 0), "go")], budget=3, max_new=2)
    assert adm.status == "undetermined" and len(adm.unknown) == 3 and ls.cost["probes"] == 3 and ls.laws == []


# ───────────────────────────────────────────── siblings
def twin_setup():
    """x0 and x1 are equal on every probe of the pool: the class left holds ("x0",) and ("x1",)."""
    g = grammar(k=1)
    cands = [Candidate("do", ("x0",), "on"), Candidate("do", ("x1",), "on"), Candidate("do", ("x2",), "on")]
    probes = [(s, "do") for s in ALL if s[0] == s[1]]
    return g, cands, probes


def test_siblings_undetermined_refuses_a_class_of_equal_length_conditions():
    g, cands, probes = twin_setup()
    ls = LawSet(g, base=off)
    adm = ls.admit_by_probes(cands, world, probes=probes, budget=20)
    assert adm.status == "accepted" and adm.law.condition == ("x0",)          # the default: order decides
    ls = LawSet(g, base=off)
    adm = ls.admit_by_probes(cands, world, probes=probes, budget=20, siblings="undetermined")
    assert adm.status == "undetermined" and ls.laws == [] and "('x1',)" in adm.why
    assert [[c.condition for c in cl] for cl in adm.classes] == [[("x0",), ("x1",)]]
    ls = LawSet(g, base=off)                       # a pool that tells them apart: accepted as before
    adm = ls.admit_by_probes(cands, world, probes=[(s, "do") for s in ALL], budget=20, siblings="undetermined")
    assert adm.status == "accepted" and adm.law.condition == ("x0",)


def test_siblings_of_another_length_do_not_refuse():
    g = grammar(k=2)
    cands = [Candidate("do", ("x0",), "on"), Candidate("do", ("x0", "x1"), "on")]
    probes = [(s, "do") for s in ALL if s[1]]         # x1 never varies: the longer one is in the class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adm = LawSet(g, base=off).admit_by_probes(cands, world, probes=probes, budget=20, siblings="undetermined")
    assert adm.status == "accepted" and adm.law.condition == ("x0",) and adm.untested == ["x1"]


def test_siblings_with_noise_and_validation():
    g, cands, probes = twin_setup()
    adm = LawSet(g, base=off).admit_by_probes(cands, world, probes=probes, budget=20, noise=0.05)
    assert adm.status == "accepted" and adm.law.condition == ("x0",)
    adm = LawSet(g, base=off).admit_by_probes(cands, world, probes=probes, budget=20, noise=0.05,
                                              siblings="undetermined")
    assert adm.status == "undetermined" and "cannot tell apart" in adm.why
    with pytest.raises(ValueError):
        LawSet(g, base=off).admit_by_probes(cands, world, probes=probes, siblings="first")
    with pytest.raises(ValueError):
        LawSet(g, base=off).admit_by_probes(cands, world, probes=probes, siblings="undetermined", max_new=2)
