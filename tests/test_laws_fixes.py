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
