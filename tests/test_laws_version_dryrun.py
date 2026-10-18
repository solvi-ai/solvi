"""solvi.experimental.laws: `LawSet.version` (a public counter of law-set changes), pickle-safe caches, and the dry
run `admit_by_probes(..., commit=False)`."""
from __future__ import annotations

import itertools
import pickle
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Enumerate, Grammar, LawSet, Transition

F = 6


class _x:
    """s[i] — a picklable atom function."""
    def __init__(self, i):
        self.i = i

    def __call__(self, s):
        return s[self.i]


class _y(_x):
    def __call__(self, s):
        return not s[self.i]


def same(u, v):
    return u == v


def grammar(k=2):
    atoms = []
    for i in range(F):
        atoms.append(Atom(f"x{i}", _x(i), feature=f"x{i}"))
        atoms.append(Atom(f"~x{i}", _y(i), feature=f"x{i}"))
    return Grammar(atoms, k=k)


ALL = [tuple(bits) for bits in itertools.product((False, True), repeat=F)]


def world(s, a):
    return "on" if s[0] else "off"


def off(s, a):
    return "off"


def buffer_of(g, rule=world, groups=4):
    b = Buffer(g)
    for i, s in enumerate(ALL):
        b.add(Transition(s, "do", rule(s, "do"), group=i % groups, id=f"r{i}"))
    return b


# ───────────────────────────────────────────── version
def test_version_moves_on_every_admit_retire_suspend_and_withdraw():
    g = grammar()
    ls = LawSet(g, base=off, on_fall="suspend", suspended_cap=1)
    v = ls.version
    assert v == 0
    law = ls.add(Candidate("do", ("x0",), "on"))
    assert ls.version == v + 1
    ls.suspend(law)
    assert ls.version == v + 2
    ls.reactivate(law)
    assert ls.version == v + 3
    ls.retire(law)
    assert ls.version == v + 4 and law.status == "retired"
    a = ls.add(Candidate("do", ("x1",), "on"))
    b = ls.add(Candidate("do", ("x2",), "on"))
    ls.suspend(a)
    v = ls.version
    ls.suspend(b)                                  # cap 1: `a` is withdrawn from suspended into retired
    assert ls.version == v + 2 and a.status == "retired" and ls.suspended == [b]
    v = ls.version
    ls.retire(b)                                   # a suspended law retired
    assert ls.version == v + 1


def test_version_moves_with_the_test_and_the_check_and_not_with_predictions():
    g = grammar(k=1)
    b = buffer_of(g)
    ls = LawSet(g, base=off)
    v = ls.version
    ls.predict(ALL[1], "do")
    ls.score(Candidate("do", ("x0",), "on"), b)
    assert ls.version == v
    got = ls.test([Candidate("do", ("x0",), "on")], b, exclude=0)
    assert got and ls.version == v + 1
    wrong = ls.add(Candidate("do", ("x1",), "on"))           # wrong wherever x1 and not x0
    v = ls.version
    assert ls.check(b) == [wrong] and ls.version == v + 1


# ───────────────────────────────────────────── pickling
def test_buffer_cache_is_not_pickled_and_answers_stay_right():
    g = grammar()
    b = buffer_of(g)
    ls = LawSet(g, base=off)
    ls.add(Candidate("do", ("x0",), "on"))
    before = ls.score(Candidate("do", ("x1",), "on"), b)
    assert b._cache                                         # id()-keyed entries exist
    assert any(isinstance(x, int) for ck in b._cache for part in ck[1:] if isinstance(part, tuple) for x in part)
    b2 = pickle.loads(pickle.dumps(b))
    assert b2._cache == {} and len(b2) == len(b)
    ls2 = pickle.loads(pickle.dumps(ls))
    assert ls2.version == ls.version and ls2._memo == {} and ls2._memo_at is None
    assert ls2.score(Candidate("do", ("x1",), "on"), b2) == before
    assert ls2._base_key == ("base", id(ls2.base))


def test_a_pickled_pair_keeps_its_answers_and_its_cache_keys_point_at_live_objects():
    g = grammar()
    b = buffer_of(g)
    ls = LawSet(g, base=off, memo=True, same=same)
    ls.add(Candidate("do", ("x0",), "on"))
    cands = [Candidate("do", c, "on") for c in (("x1",), ("x0", "x1"), ("~x2",))]
    want = [ls.score(c, b) for c in cands]
    ls2, b2 = pickle.loads(pickle.dumps((ls, b)))
    assert b2._cache == {} and ls2._memo == {}
    assert [ls2.score(c, b2) for c in cands] == want
    assert ls2._cmp == ("same", id(ls2.same))
    assert all(ck[1][1] == id(ls2.apply) for ck in b2._cache if ck[1][0] == "effect")


# ───────────────────────────────────────────── the dry run
def test_a_dry_run_returns_the_decision_and_leaves_the_set_as_it_was():
    g = grammar(k=2)
    ls = LawSet(g, base=off)
    cands = Enumerate().propose(ls, tuple([True] * F), "do", "on")
    probes = [(s, "do") for s in ALL]
    v, seq, acc = ls.version, ls._seq, ls.cost["accepted"]
    dry = ls.admit_by_probes(cands, world, probes=probes, budget=30, commit=False)
    assert dry.status == "accepted" and dry.law.condition == ("x0",) and dry.law.status == "proposed"
    assert ls.laws == [] and ls.retired == [] and ls.version == v and ls._seq == seq and ls.cost["accepted"] == acc
    assert ls.cost["probes"] == len(dry.asked) > 0            # the probes were asked: they are spent
    real = ls.admit_by_probes(cands, world, probes=probes, budget=30)
    assert real.status == "accepted" and real.law.condition == dry.law.condition and real.law.id == dry.law.id
    assert [x.id for x in real.asked] == [x.id for x in dry.asked] and real.law.status == "active"
    assert ls.laws == [real.law] and ls.version == v + 1


def test_refusing_a_dry_run_records_no_retirement():
    g = grammar(k=2)
    ls = LawSet(g, base=off)
    keep = ls.add(Candidate("do", ("x5",), "off"))
    cands = Enumerate().propose(ls, tuple([True] * F), "do", "on")
    dry = ls.admit_by_probes(cands, world, probes=[(s, "do") for s in ALL], budget=30, commit=False)
    assert dry.status == "accepted"
    assert ls.laws == [keep] and ls.retired == [] and [x.status for x in ls.all()] == ["active"]


def test_an_undetermined_dry_run_is_the_same_as_a_real_one():
    g = grammar(k=2)
    ls = LawSet(g, base=off)
    cands = Enumerate().propose(ls, tuple([True] * F), "do", "on")
    probes = [(s, "do") for s in ALL]
    dry = ls.admit_by_probes(cands, world, probes=probes, budget=0, commit=False)
    real = ls.admit_by_probes(cands, world, probes=probes, budget=0)
    assert dry.status == real.status == "undetermined" and dry.why == real.why and dry.law is None
    assert ls.laws == []


def test_a_dry_run_leaves_a_suspended_law_where_it_was():
    g = grammar(k=1)
    ls = LawSet(g, base=off, on_fall="suspend")
    old = ls.add(Candidate("do", ("x0",), "on"))
    ls.suspend(old, why="fell")
    v = ls.version
    adm = ls.admit_by_probes([Candidate("do", ("x0",), "on")], world, probes=[(s, "do") for s in ALL], budget=10,
                             commit=False)
    assert adm.status == "accepted"
    assert ls.suspended == [old] and old.status == "suspended" and old.why == "fell" and ls.retired == []
    assert ls.version == v and ls.laws == []


def test_a_dry_run_of_two_laws_and_with_noise():
    g = Grammar([Atom(f"x{i}", _x(i), feature=f"f{i}") for i in range(3)], k=1)
    or_world = lambda s, a: bool(s[0] or s[1])                          # noqa: E731
    cands = [Candidate("go", (f"x{i}",), True) for i in range(3)]
    b = Buffer(g)
    b.extend([Transition((0, 1, 1), "go", True, group="log"), Transition((1, 0, 1), "go", True, group="log")])
    pool = [((0, 0, 1), "go"), ((1, 1, 0), "go"), ((1, 0, 0), "go"), ((0, 1, 0), "go")]
    ls = LawSet(g, base=lambda s, a: False)
    adm = ls.admit_by_probes(cands, or_world, probes=pool, buffer=b, budget=4, max_new=2, commit=False)
    assert adm.status == "accepted" and len(adm.laws) == 2 and all(x.status == "proposed" for x in adm.laws)
    assert ls.laws == [] and ls.version == 0
    g = grammar(k=1)
    ls = LawSet(g, base=off)
    adm = ls.admit_by_probes([Candidate("do", ("x0",), "on"), Candidate("do", ("x1",), "on")], world,
                             probes=[(s, "do") for s in ALL], budget=20, noise=0.05, commit=False)
    assert adm.status == "accepted" and adm.law.condition == ("x0",) and ls.laws == [] and ls.version == 0
