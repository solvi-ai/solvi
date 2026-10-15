"""solvi.experimental.laws: fixes to admission and retirement — precision counted where a law would apply, retirements
inside the test reported, the sized N counted over distinct candidates, the exact acceptance, the tie order between
laws of equal length, a tolerance for numeric predictions, and the probe admission's guard against a wrong lone class.

Synthetic worlds only: every law is known, so a false accept is known exactly."""
from __future__ import annotations

import itertools
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


# ───────────────────────────────────────────── precision on the rows where the law would apply
def hills():
    g = Grammar([Atom("hills", lambda s: s["t"] == "H", feature="t"), Atom("center", lambda s: s["c"], feature="c")],
                k=2)
    b = Buffer(g)
    for grp in ("A", "B"):
        b.extend(Transition({"t": "H", "c": False, "i": i}, "food", 1, group=grp, id=f"{grp}d{i}") for i in range(45))
        b.extend(Transition({"t": "H", "c": True, "i": i}, "food", 2, group=grp, id=f"{grp}e{i}") for i in range(10))
    return g, b


def test_a_default_is_judged_on_the_rows_its_exceptions_leave_it():
    g, b = hills()
    ls = LawSet(g, base=lambda s, a: 0, sized=False)
    ls.add(Candidate("food", ("center", "hills"), 2))                 # the exception, already known
    c = Candidate("food", ("hills",), 1)                               # the default
    assert ls.score(c, b, exclude="A") == (45, 45, 45)                 # was (55, 45, 45): precision 0.818
    got = ls.test([c], b, exclude="A")
    assert [law.condition for law in got] == [("hills",)]
    law = got[0]
    assert (law.support, law.hits, law.precision) == (45, 45, 1.0)
    assert sorted(law.sources) == sorted(f"Bd{i}" for i in range(45))  # the rows it applies on, not the exception's
    old = LawSet(g, base=lambda s, a: 0, sized=False, support_on="matched")
    old.add(Candidate("food", ("center", "hills"), 2))
    assert old.score(c, b, exclude="A") == (55, 45, 45) and old.test([c], b, exclude="A") == []
    with pytest.raises(ValueError):
        LawSet(g, support_on="rows")


def test_acceptance_precision_is_the_first_checks_precision():
    g, b = hills()
    ls = LawSet(g, base=lambda s, a: 0, sized=False)
    ls.add(Candidate("food", ("center", "hills"), 2))
    law = ls.test([Candidate("food", ("hills",), 1)], b, exclude="A")[0]
    ls.check(b, "food")
    assert law.status == "active" and law.held == (law.hits, law.support)


def test_readmission_waits_for_the_buffer_to_grow_and_an_observe_retirement_is_stamped_at_the_next_test():
    g = Grammar([Atom("a", lambda s: s["a"], feature="a")], k=1)
    b = Buffer(g)
    for grp in range(3):
        b.extend(Transition({"a": True, "i": i}, "go", 1, group=grp) for i in range(10))
    ls = LawSet(g, base=lambda s, a: 0, sized=False, readmit="grown")
    c = Candidate("go", ("a",), 1)
    law = ls.test([c], b, exclude=0)[0]
    for i in range(8):                                                 # the stream says the law is wrong now
        ls.observe(Transition({"a": True, "i": 100 + i}, "go", 0, group="s"))
    assert law.status == "retired"
    assert ls.test([c], b, exclude=0) == [] and ls.refused_readmit == 1    # same buffer: the same evidence
    b.add(Transition({"a": True, "i": 200}, "go", 1, group=1))
    assert [x.condition for x in ls.test([c], b, exclude=0)] == [("a",)]   # the buffer grew: tested again
    plain = LawSet(g, base=lambda s, a: 0, sized=False)
    plain.test([c], b, exclude=0)
    for i in range(8):
        plain.observe(Transition({"a": True, "i": 100 + i}, "go", 0, group="s"))
    assert plain.test([c], b, exclude=0)                               # the default re-admits at once, as before
    with pytest.raises(ValueError):
        LawSet(g, readmit="never")


# ───────────────────────────────────────────── retirements inside the test reach the step
def churn():
    g = Grammar([Atom("a", lambda s: s["a"], feature="a")], k=1)
    b = Buffer(g)
    rng = random.Random(0)
    for grp in range(4):
        for i in range(50):
            b.add(Transition({"a": True, "i": i}, "go", 1 if rng.random() < 0.77 else 0, group=grp))
    return g, b


def test_a_law_its_own_check_retires_is_in_step_retired_and_the_bars_are_warned():
    g, b = churn()
    with pytest.warns(UserWarning, match="retire_below"):
        ls = LawSet(g, base=lambda s, a: 0, sized=False, min_precision=0.75, retire_below=0.8)
    steps = [ls.step(Transition({"a": True, "i": -j}, "go", 1, group=f"new{j}"), b) for j in range(3)]
    assert [len(st.accepted) for st in steps] == [1, 1, 1]
    assert [len(st.retired) for st in steps] == [1, 1, 1]                # was [0, 0, 0]
    assert steps[0].retired[0] is steps[0].accepted[0] and steps[0].accepted[0].status == "retired"
    assert ls.last_retired == steps[-1].retired
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LawSet(g, min_precision=0.9, retire_below=0.8)                   # the defaults: no warning


# ───────────────────────────────────────────── the sized N over distinct candidates; the exact acceptance
NF = 10
STATES = list(itertools.product((0, 1), repeat=NF))


def bits(k=2):
    return Grammar([Atom(f"x{i}", (lambda s, i=i: s[i] == 1), feature=f"f{i}") for i in range(NF)], k=k)


def test_the_same_candidates_rescored_on_the_same_buffer_do_not_raise_n():
    g = bits()
    b = Buffer(g)
    rng = random.Random(1)
    for i in range(300):
        s = rng.choice(STATES)
        b.add(Transition(s, "go", s[3] == 1 and s[4] == 1, group=i % 3))
    cands = LawSet(g, base=lambda s, a: False).mismatch(tuple([1] * NF), "go", True)
    Ns, costs = [], []
    for _ in range(3):
        ls = LawSet(g, base=lambda s, a: False)
        ls.test(cands, b, exclude=0)
        Ns.append(b.tested)
        costs.append(ls.cost["tests"])
    assert Ns == [len(cands)] * 3                                        # was 56, 112, 168
    assert costs == [len(cands)] * 3                                     # the cost still counts every evaluation
    ls = LawSet(g, base=lambda s, a: False)
    ls.test(cands + cands, b, exclude=0)
    assert b.tested == len(cands)                                        # a repeat within one call counts once
    ls.test([Candidate("go", ("x9",), False)], b, exclude=0)
    assert b.tested == len(cands) + 1


def few_shot(seed):
    rng = random.Random(seed)
    g = bits()
    b = Buffer(g)
    for grp in range(3):
        for _ in range(6):
            s = rng.choice(STATES)
            b.add(Transition(s, "go", s[3] == 1, group=grp))
    return g, b


@pytest.mark.parametrize("kw, learned", [({}, 0), ({"acceptance": "exact", "min_support": 2}, 20)])
def test_few_shot_an_exact_law_is_learned_only_with_the_exact_acceptance(kw, learned):
    ok = 0
    for seed in range(20):
        g, b = few_shot(seed)
        ls = LawSet(g, base=lambda s, a: False, **kw)
        for t in b.rows("go"):
            ls.step(t, b)
        ok += any(law.condition == ("x3",) and law.effect is True for law in ls.laws)
    assert ok == learned


def test_the_exact_acceptance_refuses_any_error_and_asks_the_confirm_hook():
    g, b = few_shot(0)
    ls = LawSet(g, base=lambda s, a: False, acceptance="exact", min_support=2)
    assert ls.acceptance == "exact" and not ls.sized
    assert ls.passes(5, 5, 10_000) and not ls.passes(50, 49, 10) and not ls.passes(1, 1, 1)
    seen = []
    no = LawSet(g, base=lambda s, a: False, acceptance="exact", min_support=2,
                confirm=lambda c: seen.append(c) or False)
    for t in b.rows("go"):
        no.step(t, b)
    assert no.laws == [] and seen                                        # every passing candidate was asked, refused
    yes = LawSet(g, base=lambda s, a: False, acceptance="exact", min_support=2,
                 confirm=lambda c: c.condition == ("x3",))
    for t in b.rows("go"):
        yes.step(t, b)
    assert [law.condition for law in yes.laws] == [("x3",)]
    law = yes.laws[0]
    assert law.precision == 1.0 and law.level is None and law.null is None
    assert LawSet(g).acceptance == "sized" and LawSet(g, sized=False).acceptance == "fixed"
    with pytest.raises(ValueError):
        LawSet(g, acceptance="loose")


# ───────────────────────────────────────────── the tie order between laws of equal length
def wood():
    return Grammar([Atom("w2", lambda s: s["wood"] >= 2, feature="w2"),
                    Atom("w3", lambda s: s["wood"] >= 3, feature="w3")], k=1)


def test_ties_newer_lets_a_newer_law_of_equal_length_speak():
    old = LawSet(wood(), base=lambda s, a: None)
    new = LawSet(wood(), base=lambda s, a: None, ties="newer")
    for ls in (old, new):
        ls.add(Candidate("table", ("w2",), "none"))
        ls.add(Candidate("table", ("w3",), "table"))
    assert old.predict({"wood": 3}, "table") == "none"                   # the earlier law (the default, unchanged)
    assert new.predict({"wood": 3}, "table") == "table"
    assert new.predict({"wood": 2}, "table") == "none"
    assert [law.condition for law in new.laws] == [("w3",), ("w2",)]
    h = LawSet(wood(), base=lambda s, a: None, ties="newer")
    h.add(Candidate("table", ("w2",), "none"))
    assert h.hypothesis(Candidate("table", ("w3",), "table"))({"wood": 3}, "table") == "table"
    with pytest.raises(ValueError):
        LawSet(wood(), ties="random")


def test_ties_newer_scores_a_candidate_on_the_rows_it_would_take_from_an_equal_law():
    g = wood()
    b = Buffer(g)
    for i in range(30):
        w = 2 + i % 2
        b.add(Transition({"wood": w, "i": i}, "table", "table" if w >= 3 else "none", group=i % 3))
    c = Candidate("table", ("w3",), "table")
    for ties, expect in (("earlier", (0, 0, 0)), ("newer", (15, 15, 15))):
        ls = LawSet(g, base=lambda s, a: None, ties=ties, sized=False)
        ls.add(Candidate("table", ("w2",), "none"))
        assert ls.score(c, b) == expect
    ls = LawSet(g, base=lambda s, a: None, ties="newer")
    ls.add(Candidate("table", ("w2",), "none"))
    adm = ls.admit_by_probes([c], lambda s, a: "table" if s["wood"] >= 3 else "none",
                             probes=[({"wood": w}, "table") for w in range(5)])
    assert adm.status == "accepted" and ls.predict({"wood": 4}, "table") == "table"


# ───────────────────────────────────────────── a tolerance for numeric predictions
def test_discriminate_and_split_with_a_tolerance_ignore_float_rounding():
    from solvi.experimental.laws import discriminate, split
    a = lambda s, act: s * 0.1 * 10                                      # noqa: E731
    b = lambda s, act: s                                                 # noqa: E731
    probes = [(0.1 + 0.1 * i, "y") for i in range(30)]
    always = lambda s, act: True                                         # noqa: E731
    assert discriminate(a, b, always, probes) is not None                # exact: 0.10000000000000002 vs 0.1
    assert discriminate(a, b, always, probes, tol=1e-9) is None
    assert discriminate(a, b, always, probes, same=lambda x, y: round(x, 6) == round(y, 6)) is None
    c = lambda s, act: 2 * s                                             # noqa: E731
    pr = split([a, b, c], always, probes, tol=1e-9)
    assert pr is not None and pr.score == 1                              # a and b are one group, c the other
    assert split([(lambda s, act: (s, "x")), (lambda s, act: (s + 1e-12, "x"))], always, probes, tol=1e-9) is None
    with pytest.raises(ValueError):
        split([a, b], always, probes, tol=1e-9, same=lambda x, y: True)


def scale():
    g = Grammar([Atom("box", lambda s: True, feature="box")], k=1)
    cands = [Candidate("y", ("box",), c) for c in (1.0, 2.0, 3.0)]
    probes = [(0.5 + 0.1 * i, "y") for i in range(40)]
    return g, cands, probes


@pytest.mark.parametrize("noise", [0.0, 1e-6])
def test_admit_by_probes_with_a_tolerance_admits_through_measurement_noise(noise):
    g, cands, probes = scale()
    rng = random.Random(0)
    ask = lambda s, act: 2.0 * s * (1 + noise * rng.gauss(0, 1))         # noqa: E731
    exact = LawSet(g, apply=lambda e, s: e * s, base=lambda s, a: 0.0).admit_by_probes(cands, ask, probes=probes)
    assert exact.status == ("accepted" if noise == 0 else "undetermined")   # exact equality: noise refutes all
    ls = LawSet(g, apply=lambda e, s: e * s, base=lambda s, a: 0.0)
    adm = ls.admit_by_probes(cands, ask, probes=probes, tol=1e-4)
    assert adm.status == "accepted" and adm.law.effect == 2.0 and adm.refuted == 2
    assert adm.law.hits == adm.law.support                               # hits counted within the tolerance
    off = LawSet(g, apply=lambda e, s: e * s, base=lambda s, a: 0.0).admit_by_probes(
        cands, lambda s, act: 2.5 * s, probes=probes, tol=1e-4)
    assert off.status == "undetermined"                                  # outside the tolerance: refuted
