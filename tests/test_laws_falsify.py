"""admit_by_probes(falsify="leftover"): the leftover budget spent on probes where the accepted law applies.

Synthetic boolean worlds with positive atoms only: "x0 and not x1" cannot be expressed, so the one candidate "x0"
survives the split phase as a wrong law."""
from __future__ import annotations

import itertools
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


N = 4
STATES = [tuple(bits) for bits in itertools.product((False, True), repeat=N)]


def grammar():
    return Grammar([Atom(f"x{i}", (lambda s, i=i: s[i])) for i in range(N)], k=2)


def lawset(g):
    return LawSet(g, base=lambda s, a: "N")


def wrong_world(s, a):
    return "E" if s[0] and not s[1] else "N"          # needs a negation the grammar lacks


def right_world(s, a):
    return "E" if s[0] else "N"


def setup(world, pool):
    """One candidate "x0 -> E"; a logged row where x0 holds and x1 does not refutes the set without it, so one class
    is left before any probe: the whole budget is leftover."""
    g = grammar()
    b = Buffer(g)
    s = (True, False, False, False)
    b.add(Transition(s, "do", world(s, "do"), group=0, id="r0"))
    return g, b, [Candidate("do", ("x0",), "E")], [(st, "do") for st in pool]


def run(world, pool, **kw):
    g, b, cands, probes = setup(world, pool)
    ls = lawset(g)
    adm = ls.admit_by_probes(cands, world, probes=probes, budget=8, buffer=b, **kw)
    return ls, adm


def test_default_is_unchanged_and_admits_the_wrong_law():
    ls0, a0 = run(wrong_world, STATES)
    ls1, a1 = run(wrong_world, STATES, falsify=None)
    assert a0.status == a1.status == "accepted"
    assert a0.law.condition == a1.law.condition == ("x0",)
    assert a0.why == a1.why and a0.asked == a1.asked == []
    assert ls0.cost == ls1.cost and len(ls0.laws) == len(ls1.laws) == 1
    assert any(wrong_world(s, "do") != "E" for s in STATES if s[0])        # the admitted law is wrong somewhere


def test_leftover_refutes_the_wrong_law_the_default_admits():
    ls, adm = run(wrong_world, STATES, falsify="leftover")
    assert adm.status == "undetermined" and adm.law is None
    assert "falsified by probe" in adm.why
    assert ls.laws == [] and ls.cost["accepted"] == 0
    assert 1 <= len(adm.asked) <= 8 and ls.cost["probes"] == len(adm.asked)
    assert all(t.id.startswith("falsify-") and t.state[0] for t in adm.asked)
    assert adm.asked[-1].observed == "N"                                    # the refuting answer is the last one


def test_a_sound_law_is_still_admitted_and_spends_the_leftover():
    ls, adm = run(right_world, STATES, falsify="leftover")
    assert adm.status == "accepted" and adm.law.condition == ("x0",)
    eligible = sum(1 for s in STATES if s[0])
    assert len(adm.asked) == min(8, eligible) and ls.cost["probes"] == len(adm.asked)
    assert "none refuted" in adm.why
    ls0, adm0 = run(right_world, STATES)
    assert adm.law.support == adm0.law.support                             # committed as without the option


def test_a_starved_pool_still_admits_the_wrong_law():
    pool = [s for s in STATES if not s[1]]                                  # no state where the law is wrong
    ls, adm = run(wrong_world, pool, falsify="leftover")
    assert adm.status == "accepted" and adm.law.condition == ("x0",)
    eligible = sum(1 for s in pool if s[0])
    assert len(adm.asked) == eligible < 8
    assert f"{eligible} of {eligible} eligible" in adm.why


def test_the_dry_run_and_the_seed_repeat_the_decision():
    _, a = run(wrong_world, STATES, falsify="leftover", seed=3)
    ls, b = run(wrong_world, STATES, falsify="leftover", seed=3, commit=False)
    assert (a.status, [t.id for t in a.asked]) == (b.status, [t.id for t in b.asked]) and ls.laws == []


def test_unsafe_probes_are_not_asked():
    _, adm = run(wrong_world, STATES, falsify="leftover", safe=lambda s, a: not s[1])
    assert adm.status == "accepted" and all(not t.state[1] for t in adm.asked)


@pytest.mark.parametrize("bad", ["reserve", "", True, 1])
def test_an_invalid_value_raises(bad):
    with pytest.raises(ValueError):
        run(wrong_world, STATES, falsify=bad)


def test_noise_and_two_laws_refuse_it():
    with pytest.raises(ValueError):
        run(wrong_world, STATES, falsify="leftover", noise=0.1)
    with pytest.raises(ValueError):
        run(wrong_world, STATES, falsify="leftover", max_new=2)
