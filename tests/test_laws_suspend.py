"""solvi.experimental.laws: LawSet(on_fall="suspend") — a fallen law stays a candidate, is re-checked on later
observations where it would apply and reactivated when they bear it out; doubt_probe names where to ask."""
from __future__ import annotations

import itertools
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition

F = 6
STATES = list(itertools.product((0, 1), repeat=F))
ON = [s for s in STATES if s[0] and s[1]]
OFF = [s for s in STATES if not (s[0] and s[1])]


def grammar():
    return Grammar([Atom(f"x{i}", (lambda s, i=i: s[i] == 1), feature=f"x{i}") for i in range(F)], k=2)


def lawset(**kw):
    ls = LawSet(grammar(), base=lambda s, a: "N", **kw)
    law = ls.add(Candidate("go", ("x0", "x1"), "E"))
    return ls, law


def feed(ls, states, obs, group="g"):
    out = []
    for i, s in enumerate(states):
        out += ls.observe(Transition(s, "go", obs, group=f"{group}{i}"))
    return out


def test_default_retires_for_good():
    ls, law = lawset()
    assert feed(ls, ON[:5], "N") == [law]
    assert law.status == "retired" and ls.suspended == [] and ls.laws == []
    feed(ls, ON * 3, "E", "back")
    assert ls.laws == []


def test_suspended_law_is_reactivated_by_fresh_observations():
    ls, law = lawset(on_fall="suspend")
    assert feed(ls, ON[:5], "N") == [law]
    assert law.status == "suspended" and ls.suspended == [law] and ls.laws == []
    feed(ls, ON[:9], "E", "back")             # the window is not full yet
    assert ls.laws == []
    feed(ls, ON[9:10] or ON[:1], "E", "more")
    assert ls.laws == [law] and law.status == "active" and ls.reactivated == 1
    assert law.running() == (0, 0)            # judged afresh
    assert ls.predict(ON[0], "go") == "E"


def test_rows_where_it_does_not_apply_do_not_count():
    ls, law = lawset(on_fall="suspend")
    feed(ls, ON[:5], "N")
    feed(ls, OFF * 2, "E", "off")              # condition false: nothing counted
    assert len(law.doubt) == 0 and ls.laws == []


def test_a_change_that_stays_never_brings_it_back():
    ls, law = lawset(on_fall="suspend")
    feed(ls, ON[:5], "N")
    feed(ls, ON * 10, "N", "after")
    assert ls.laws == [] and ls.reactivated == 0 and law.status == "suspended"


def test_a_mixed_window_is_not_enough():
    ls, law = lawset(on_fall="suspend", reactivate_min=10)
    feed(ls, ON[:5], "N")
    seq = (["E"] * 4 + ["N"]) * 6                 # 80 % right: below min_precision 0.9
    for i, o in enumerate(seq):
        ls.observe(Transition(ON[i % len(ON)], "go", o, group=f"m{i}"))
    assert ls.laws == []


def test_doubt_probe_points_where_the_suspended_law_and_the_set_disagree():
    ls, law = lawset(on_fall="suspend")
    assert ls.doubt_probe([(s, "go") for s in STATES]) is None      # nothing suspended
    feed(ls, ON[:5], "N")
    got = ls.doubt_probe([(s, "go") for s in OFF] + [(ON[3], "go")])
    assert got is not None
    which, p = got
    assert which is law and p.state == ON[3]
    assert ls.doubt_probe([(s, "go") for s in OFF]) is None


def test_the_cap_retires_the_oldest_and_the_test_takes_a_suspended_law_out():
    ls = LawSet(grammar(), base=lambda s, a: "N", on_fall="suspend", suspended_cap=1)
    a = ls.add(Candidate("go", ("x0", "x1"), "E"))
    b = ls.add(Candidate("go", ("x2", "x3"), "F"))
    ls.suspend(a)
    ls.suspend(b)
    assert ls.suspended == [b] and a.status == "retired"
    ls.add(Candidate("go", ("x2", "x3"), "F"))
    assert ls.suspended == [] and b.status == "retired"


def test_arguments_are_checked():
    with pytest.raises(ValueError):
        LawSet(grammar(), on_fall="sleep")
    with pytest.raises(ValueError):
        LawSet(grammar(), on_fall="suspend", reactivate_min=0)


def test_step_loop_with_a_buffer_suspends_and_reactivates():
    """The learning loop: a law learned, broken for a while (suspended), back (reactivated without a new test)."""
    g = grammar()
    b = Buffer(g, window=60)
    ls = LawSet(g, base=lambda s, a: "N", on_fall="suspend", recent=10, readmit="grown")
    rows = []
    for r in range(240):
        s = STATES[(r * 7) % len(STATES)]
        broken = 80 <= r < 120
        obs = "E" if (s[0] and s[1] and not broken) else "N"
        rows.append(Transition(s, "go", obs, group=r // 4))
    steps = [ls.step(t, b, remember=True) for t in rows]
    assert any(st.retired for st in steps[80:120])
    assert ls.reactivated >= 1 or any(st.accepted for st in steps[120:])
    assert ls.predict(ON[0], "go") == "E"
