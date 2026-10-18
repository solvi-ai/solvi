"""solvi.experimental.laws: Grammar(implies=) and coverage over a second buffer (admit_by_probes(coverage="varied",
coverage_on=)).

Known answers on threshold atoms and a small "near the table" world; a false-accept simulation on null buffers with a
declared implication chain (no law exists: any acceptance is false)."""
from __future__ import annotations

import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


def thresholds(n=1, top=3, implies=True):
    """Counters c0..c{n-1}; atoms c{i}>=t (t = 1..top), each its own feature so a condition may hold several."""
    atoms, imp = [], {}
    for i in range(n):
        for t in range(1, top + 1):
            atoms.append(Atom(f"c{i}>={t}", (lambda s, i=i, t=t: s[i] >= t), feature=f"c{i}>={t}"))
            if t > 1:
                imp[f"c{i}>={t}"] = [f"c{i}>={t - 1}"]
    return Grammar(atoms, k=3, implies=imp if implies else None)


def test_implied_is_transitive_and_conditions_never_hold_an_atom_and_one_it_implies():
    g = thresholds()
    assert g.implied(("c0>=2",)) == {"c0>=1"}
    assert g.implied(("c0>=3",)) == {"c0>=2", "c0>=1"}
    assert g.implied(("c0>=3", "c0>=2")) == {"c0>=1"}
    conds = g.conditions(["c0>=1", "c0>=2", "c0>=3"])
    assert conds == [(), ("c0>=1",), ("c0>=2",), ("c0>=3",)]
    plain = thresholds(implies=False)
    assert len(plain.conditions(["c0>=1", "c0>=2", "c0>=3"])) == 8
    assert not plain.has_implies and plain.implied(("c0>=3",)) == frozenset()


def test_implies_as_a_function_and_on_add():
    g = Grammar([Atom("prop:red", lambda s: s["red"]), Atom("obj", lambda s: s["obj"])], k=2,
                implies=lambda name: ["obj"] if name.startswith("prop:") else [])
    assert g.implied(("prop:red",)) == {"obj"}
    g.add(Atom("pair:A|B", lambda s: s.get("ab", False)), implies=["pair:A|A"])
    g.add(Atom("pair:A|A", lambda s: s.get("aa", False)))
    assert g.implied(("pair:A|B",)) == {"pair:A|A"}
    assert ("obj", "prop:red") not in g.conditions(["obj", "prop:red"])
    with pytest.raises(ValueError):
        Grammar([], implies=3)


def test_check_implies_names_the_row_that_breaks_a_declaration_and_the_buffer_warns_once():
    g = thresholds()
    b = Buffer(g)
    b.add(Transition((2,), "do", 0, id="ok-1"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # a planted contradiction: a state where c0>=2 holds and c0>=1 does not (the atoms are evaluated as given)
        bad = Grammar([Atom("c0>=1", lambda s: s[0] >= 1, feature="a"), Atom("c0>=2", lambda s: s[1], feature="b")],
                      implies={"c0>=2": ["c0>=1"]})
        b2 = Buffer(bad)
        b2.add(Transition((0, True), "do", 0, id="bad-1"))
        b2.add(Transition((0, True), "do", 0, id="bad-2"))
        b2.add(Transition((2, True), "do", 0, id="fine"))
    msgs = [str(x.message) for x in w if "breaks the declared implication" in str(x.message)]
    assert len(msgs) == 1 and "bad-1" in msgs[0]
    assert bad.check_implies(b2) == [("c0>=2", "c0>=1", ["bad-1", "bad-2"])]
    assert g.check_implies(b) == []


# ───────────────────────────────────────────── coverage: the pool never saw "near" vary
def table_world(implies=None):
    g = Grammar([Atom("wood", lambda s: s["wood"]), Atom("near", lambda s: s["near"]),
                 Atom("x", lambda s: s["x"])], k=2, implies=implies)
    world = lambda s, a: "pickaxe" if s["wood"] and s["near"] else "nothing"
    return g, world


def states(near=None, rng=None, n=40):
    rng = rng or random.Random(0)
    out = []
    for _ in range(n):
        out.append({"wood": rng.random() < 0.5, "near": rng.random() < 0.5 if near is None else near,
                    "x": rng.random() < 0.5})
    return out


def admit(g, world, cands, *, coverage, coverage_on=None, held_out=None):
    ls = LawSet(g, base=lambda s, a: "nothing")
    pool = [(s, "make") for s in states(near=True)]        # every probe next to the table
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ls.admit_by_probes(cands, world, probes=pool, budget=8, coverage=coverage, coverage_on=coverage_on,
                                  held_out=held_out)


def logged(g, world, near=None, group="h"):
    b = Buffer(g)
    for i, s in enumerate(states(near=near, rng=random.Random(5), n=60)):
        b.add(Transition(s, "make", world(s, "make"), group=group, id=f"{group}-{i}"))
    return b


CANDS = [Candidate("make", ("wood",), "pickaxe"), Candidate("make", ("near", "wood"), "pickaxe")]


def test_varied_refuses_when_the_atom_varied_on_the_second_buffer_and_accepts_when_it_never_did():
    g, world = table_world()
    a = admit(g, world, CANDS, coverage="undetermined")
    assert a.status == "undetermined" and a.untested == ["near"]          # today: refused, the pool cannot tell
    a = admit(g, world, CANDS, coverage="varied", coverage_on=logged(g, world))
    assert a.status == "undetermined" and a.untested == ["near"] and "coverage rows" in a.why
    a = admit(g, world, CANDS, coverage="varied", coverage_on=[logged(g, world, near=True)])
    assert a.status == "accepted" and a.law.condition == ("wood",) and a.untested == ["near"]
    # held_out= is read as a coverage buffer too (and still checked as before)
    a = admit(g, world, CANDS, coverage="varied", held_out=logged(g, world))
    assert a.status == "undetermined"


def test_an_implied_atom_is_never_listed_and_does_not_block():
    # a world where wood ⇒ near by construction (wood is only ever seen next to the table)
    g, _ = table_world(implies={"wood": ["near"]})
    world = lambda s, a: "pickaxe" if s["wood"] else "nothing"
    a = admit(g, world, CANDS, coverage="undetermined")
    assert a.status == "accepted" and a.untested == [] and a.law.condition == ("wood",)
    g0, _ = table_world()
    a0 = admit(g0, world, CANDS, coverage="undetermined")
    assert a0.status == "undetermined" and a0.untested == ["near"]        # without the declaration: refused for ever


def test_coverage_validation():
    g, world = table_world()
    with pytest.raises(ValueError):
        admit(g, world, CANDS, coverage="sometimes")
    with pytest.raises(ValueError):
        admit(g, world, CANDS, coverage="varied", coverage_on=[3])


def test_coverage_rows_are_read_not_tested():
    g, world = table_world()
    b = logged(g, world)
    admit(g, world, CANDS, coverage="varied", coverage_on=b)
    assert b.tested == 0


# ───────────────────────────────────────────── false accepts on null buffers with a declared implication chain
def null_buffer(rng, g, rows=1500, n=8, groups=10):
    b = Buffer(g)
    for r in range(rows):
        s = tuple(rng.choice((0, 0, 0, 1, 2, 3)) for _ in range(n))
        b.add(Transition(s, "do", "E" if rng.random() < 0.7 else "N", group=r % groups, id=f"r{r}"))
    return b


def test_false_accepts_on_null_buffers_stay_below_delta_and_n_is_smaller():
    rng = random.Random(3)
    out = {}
    for implies in (False, True):
        g = thresholds(n=8, implies=implies)
        names = [a.name for a in g.atoms]
        conds = [c for c in g.conditions(names) if len(c) <= 2]
        acc, tested = 0, 0
        r2 = random.Random(11)
        for _ in range(100):
            b = null_buffer(r2, g)
            ls = LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9)
            acc += bool(ls.test([Candidate("do", c, "E") for c in conds], b))
            tested = b.tested
        out[implies] = (acc, tested)
    assert out[True][0] / 100 <= 0.1 and out[False][0] / 100 <= 0.1        # measured: 0 of 100 either way
    assert out[True][1] < out[False][1]                                    # fewer candidates, a smaller N (277 vs 301)
