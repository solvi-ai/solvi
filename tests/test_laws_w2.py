"""solvi.experimental.laws and .skills, wave 2: acceptance sized by the candidates tested, admission by discriminating
probes, abstention when consistent laws disagree, and the search-cost record.

Synthetic worlds only: every law and every base rate is known, so a false accept is known exactly."""
from __future__ import annotations

import itertools
import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Admission, Atom, Buffer, Candidate, Enumerate, Grammar, LawSet, Transition,
                                         Undetermined, sized_level, sized_need)
    from solvi.experimental.skills import Certificate, Skill, SkillLearner, SkillSet


# ───────────────────────────────────────────── W2-1: the acceptance sized by the candidates tested
def sparse_grammar(n=40, k=2):
    return Grammar([Atom(f"f{i}", (lambda s, i=i: s[i]), feature=f"f{i}") for i in range(n)], k=k)


def noise_buffer(rng, g, rows=2000, n=40, p_atom=0.06, p_effect=0.7, law=None, groups=10):
    """Rows whose effect "E" is drawn at p_effect independently of every atom (no law exists), or, with law=(i, j),
    "E" exactly where atoms i and j both hold."""
    b = Buffer(g)
    for r in range(rows):
        s = tuple(rng.random() < p_atom for _ in range(n))
        if law is None:
            obs = "E" if rng.random() < p_effect else "N"
        else:
            obs = "E" if all(s[i] for i in law) else "N"
        b.add(Transition(s, "do", obs, group=r % groups, id=f"r{r}"))
    return b


def every_condition(g, k):
    names = [a.name for a in g.atoms]
    return [c for size in range(k + 1) for c in itertools.combinations(names, size)]


def false_accept_rate(sized, trials, seed=0):
    rng = random.Random(seed)
    g = sparse_grammar()
    conds = every_condition(g, 2)
    accepted = 0
    for _ in range(trials):
        b = noise_buffer(rng, g)
        ls = LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9, sized=sized)
        got = ls.test([Candidate("do", c, "E") for c in conds], b)
        accepted += bool(got)                         # no law exists: any acceptance is false
    return accepted / trials, len(conds)


def test_the_sized_rule_needs_more_support_the_more_candidates_were_tested():
    assert sized_need(1, 0.1, 0.9) == math.ceil(math.log(1 / sized_level(1, 0.1)) / math.log(1 / 0.9))
    assert sized_need(10, 0.1, 0.9) < sized_need(1000, 0.1, 0.9) < sized_need(100_000, 0.1, 0.9)
    assert sized_level(1000, 0.1) == pytest.approx(0.1 / (1000 * 1001))
    assert sized_level(1000, 0.1, space=5000) == pytest.approx(0.1 / 5000)   # a declared space: Bonferroni
    assert sized_level(9000, 0.1, space=5000) == pytest.approx(0.1 / (9000 * 9001))   # beyond it: anytime


def test_at_the_old_thresholds_random_candidates_are_accepted_and_the_sized_rule_stops_them():
    old, n = false_accept_rate(sized=False, trials=20)
    new, _ = false_accept_rate(sized=True, trials=20)
    assert n == 821
    assert old >= 0.5                                  # measured: 1.00 (20 of 20 buffers)
    assert new <= 0.1                                  # measured: 0.00


def test_the_sized_rule_still_accepts_a_real_law_with_the_support_it_needs():
    rng = random.Random(1)
    g = sparse_grammar(n=40, k=2)
    b = noise_buffer(rng, g, rows=4000, p_atom=0.25, law=(0, 1))
    ls = LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9)
    conds = every_condition(g, 2)
    got = ls.test([Candidate("do", c, "E") for c in conds], b)
    assert len(got) == 1 and got[0].condition == ("f0", "f1")
    law = got[0]
    assert law.tested == len(conds) and law.need == sized_need(law.tested, 0.1, 0.9)
    assert law.level == pytest.approx(sized_level(law.tested, 0.1)) and law.support >= law.need
    assert b.tested == len(conds)                       # counted on the buffer


def test_the_count_accumulates_on_the_buffer_and_sized_false_keeps_the_old_rule():
    rng = random.Random(2)
    g = sparse_grammar(n=10, k=1)
    b = noise_buffer(rng, g, rows=200, n=10, p_atom=0.5, law=(3,))
    ls = LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9)
    first = ls.test([Candidate("do", ("f0",), "E")], b)
    assert first == [] and b.tested == 1
    got = ls.test([Candidate("do", ("f3",), "E")], b)
    assert got and got[0].tested == 2
    old = LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9, sized=False)
    small = noise_buffer(rng, g, rows=30, n=10, p_atom=0.5, law=(3,))
    got = old.test([Candidate("do", ("f3",), "E")], small)
    assert got and got[0].level is None and got[0].tested == 1   # fixed thresholds: support ≥ 5 is enough


def test_the_sized_rule_with_errors_needs_a_binomial_tail_below_the_level():
    ls = LawSet(sparse_grammar(n=2, k=1), min_support=5, min_precision=0.9)
    lvl = sized_level(10, ls.delta)
    need = sized_need(10, ls.delta, 0.9)
    assert ls.passes(need, need, 10) and not ls.passes(need - 1, need - 1, 10)
    assert not ls.passes(need, need - 1, 10)            # one error at the bare minimum: the tail is above the level
    s = 4 * need
    assert ls.passes(s, s - 1, 10)                       # with more support one error passes
    tail = sum(math.comb(s, j) * 0.9 ** j * 0.1 ** (s - j) for j in range(s - 1, s + 1))
    assert tail <= lvl


def test_min_precision_one_uses_the_null_precision():
    ls = LawSet(sparse_grammar(n=2, k=1), min_support=1, min_precision=1.0)
    assert ls.null_precision == 0.95
    assert ls.passes(sized_need(5, 0.1, 0.95), sized_need(5, 0.1, 0.95), 5)


def test_the_search_cost_is_recorded_on_the_set_and_per_step():
    g = sparse_grammar(n=6, k=2)
    rng = random.Random(3)
    b = noise_buffer(rng, g, rows=600, n=6, p_atom=0.5, law=(0,))
    ls = LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9)
    s = tuple([True] + [False] * 5)
    st = ls.step(Transition(s, "do", "E", group="new"), b)
    assert st.candidates > 0 and st.tested == st.candidates
    c = ls.cost
    assert c["candidates"] == st.candidates and c["tests"] == st.tested and c["probes"] == 0 and c["simulated"] == 0
    assert c["accepted"] == len(st.accepted)


# ───────────────────────────────────────────── W2-9: admission by discriminating probes
F = 10


def binary_grammar(k=3):
    atoms = []
    for i in range(F):
        atoms.append(Atom(f"x{i}", (lambda s, i=i: s[i]), feature=f"x{i}"))
        atoms.append(Atom(f"~x{i}", (lambda s, i=i: not s[i]), feature=f"x{i}"))
    return Grammar(atoms, k=k)


ALL = [tuple(bits) for bits in itertools.product((False, True), repeat=F)]


def world_of(truth):
    def world(s, a):
        return "on" if all((s[int(x.lstrip("~x"))] is not x.startswith("~")) for x in truth) else "off"
    return world


def holds(cond, s):
    return all((s[int(x.lstrip("~x"))] is not x.startswith("~")) for x in cond)


def admission_trial(rng, *, order, unique, budget=8, pool=200):
    g = binary_grammar()
    feats = rng.sample(range(F), 2)
    truth = tuple(sorted(("~" if rng.random() < 0.5 else "") + f"x{i}" for i in feats))
    world = world_of(truth)
    s0 = next(s for s in rng.sample(ALL, len(ALL)) if holds(truth, s))
    ls = LawSet(g, base=lambda s, a: "off")
    cands = Enumerate().propose(ls, s0, "do", "on")
    probes = [(s, "do") for s in rng.sample(ALL, pool)]
    adm = ls.admit_by_probes(cands, world, probes=probes, budget=budget, order=order, unique=unique,
                             seed=rng.randrange(10 ** 9))
    if adm.status != "accepted":
        return "undetermined", adm
    wrong = any(holds(adm.law.condition, s) != holds(truth, s) for s in ALL)
    return ("false" if wrong else "right"), adm


def rates(order, unique, trials, seed=7):
    rng = random.Random(seed)
    out = {"right": 0, "false": 0, "undetermined": 0}
    asked = 0
    for _ in range(trials):
        r, adm = admission_trial(rng, order=order, unique=unique)
        out[r] += 1
        asked += len(adm.asked)
    return {k: v / trials for k, v in out.items()}, asked / trials


def test_active_probes_falsely_accept_less_than_random_held_out_at_the_same_number_of_checks():
    rnd, rnd_asked = rates("random", False, 150)
    act, act_asked = rates("split", True, 150)
    assert rnd_asked == 8 and act_asked <= 8
    assert rnd["false"] >= 0.2                           # measured: see the docstring of LawSet.admit_by_probes
    assert act["false"] < rnd["false"] and act["false"] <= 0.05


def test_two_laws_that_only_unsafe_probes_tell_apart_stay_undetermined():
    g = binary_grammar(k=1)
    ls = LawSet(g, base=lambda s, a: "off")
    a, b = Candidate("do", ("x0",), "on"), Candidate("do", ("x1",), "on")
    world = world_of(("x0",))
    probes = [(s, "do") for s in ALL]
    safe = lambda s, act: s[0] == s[1]                   # noqa: E731 — x0 and x1 agree wherever we may ask
    adm = ls.admit_by_probes([a, b], world, probes=probes, safe=safe, budget=20)
    assert adm.status == "undetermined" and adm.law is None and ls.laws == []
    assert sorted(len(c) for c in adm.classes) == [1, 1]
    assert "unsafe" in adm.why or "no safe probe" in adm.why


def test_a_budget_too_small_leaves_the_classes_undetermined_and_a_unique_class_is_accepted_shortest():
    g = binary_grammar(k=2)
    ls = LawSet(g, base=lambda s, a: "off")
    world = world_of(("x0",))
    s0 = tuple([True] * F)
    cands = Enumerate().propose(ls, s0, "do", "on")
    probes = [(s, "do") for s in ALL]
    adm0 = ls.admit_by_probes(cands, world, probes=probes, budget=0)
    assert adm0.status == "undetermined" and len(adm0.classes) > 1 and ls.laws == []
    adm = ls.admit_by_probes(cands, world, probes=probes, budget=30)
    assert adm.status == "accepted" and adm.law.condition == ("x0",) and len(adm.classes) == 1
    assert adm.law.admitted == "probes" and adm.law.tested == len(cands) and len(adm.asked) <= 30
    assert ls.cost["probes"] == len(adm.asked) and ls.laws == [adm.law]
    assert isinstance(adm, Admission)


def test_logged_rows_filter_the_candidates_before_any_probe():
    g = binary_grammar(k=1)
    ls = LawSet(g, base=lambda s, a: "off")
    world = world_of(("x0",))
    b = Buffer(g)
    for i, s in enumerate(random.Random(0).sample(ALL, 300)):
        b.add(Transition(s, "do", world(s, "do"), group=i % 3, id=f"l{i}"))
    cands = Enumerate().propose(ls, tuple([True] * F), "do", "on")
    adm = ls.admit_by_probes(cands, world, probes=[(s, "do") for s in ALL], budget=0, buffer=b)
    assert adm.status == "accepted" and adm.law.condition == ("x0",) and adm.asked == []


def test_no_survivor_is_undetermined():
    g = binary_grammar(k=1)
    ls = LawSet(g, base=lambda s, a: "off")
    adm = ls.admit_by_probes([Candidate("do", ("x0",), "on")], lambda s, a: "off", probes=[(ALL[-1], "do")], budget=3)
    assert adm.status == "undetermined" and adm.classes == [] and "no candidate" in adm.why


# ───────────────────────────────────────────── W2-2: abstain when consistent laws disagree
def door():
    g = Grammar([Atom("red", lambda s: s["red"], feature="colour"),
                 Atom("big", lambda s: s["big"], feature="size"),
                 Atom("wet", lambda s: s["wet"], feature="wet")], k=3)
    return g


STATES = [{"red": r, "big": b, "wet": w} for r in (False, True) for b in (False, True) for w in (False, True)]


def test_abstain_fires_exactly_where_two_incomparable_laws_disagree():
    ls = LawSet(door(), base=lambda s, a: "base")
    ls.add(Candidate("go", ("red",), "A"))
    ls.add(Candidate("go", ("big",), "B"))
    ls.add(Candidate("go", ("red", "wet"), "A"))           # agrees with its own wider law
    ls.add(Candidate("go", ("big", "red", "wet"), "C"))    # an exception more specific than both: resolves them
    seen = {}
    for s in STATES:
        p = ls.predict(s, "go", disagree="abstain")
        seen[(s["red"], s["big"], s["wet"])] = p
        first = ls.predict(s, "go")
        if isinstance(p, Undetermined):
            assert s["red"] and s["big"] and not s["wet"]
            assert {law.effect for law in p.laws} == {"A", "B"} and set(p.predictions) == {"A", "B"}
            assert first == "A"                              # the default: most specific first (ties: the earlier)
        else:
            assert p == first
    assert sum(isinstance(p, Undetermined) for p in seen.values()) == 1
    assert seen[(True, True, True)] == "C" and seen[(True, False, False)] == "A" and seen[(False, True, True)] == "B"
    assert ls.abstained == 1
    assert ls.predict(STATES[0], "go", disagree="abstain") == "base"


def test_abstain_as_the_set_default_and_the_loop_still_learns_most_specific_first():
    ls = LawSet(door(), base=lambda s, a: "base", disagree="abstain")
    ls.add(Candidate("go", ("red",), "A"))
    ls.add(Candidate("go", ("big",), "B"))
    s = {"red": True, "big": True, "wet": False}
    assert isinstance(ls.predict(s, "go"), Undetermined)
    assert ls.predict(s, "go", disagree="first") == "A"
    assert not ls.mismatch(s, "go", "A")                    # the learning loop keeps most-specific-first
    with pytest.raises(ValueError):
        LawSet(door(), disagree="vote")


def test_skillset_match_stays_most_specific_first_different_sequences_are_not_a_disagreement():
    """Two certified skills of one goal are both verified plans: a different sequence says nothing about the world,
    so a skill set never abstains (it would only on conflicting predicted end states, which skills do not carry)."""
    g = door()
    with pytest.raises(TypeError):
        SkillSet(g, certificate=None, disagree="abstain")
    ks = SkillSet(g, certificate=None)
    a = ks.add(Skill(("red",), ("left",), "exit"))
    b = ks.add(Skill(("big",), ("right",), "exit"))
    c = ks.add(Skill(("big", "wet"), ("right",), "exit"))
    with pytest.raises(TypeError):
        ks.match({"red": True, "big": True, "wet": False}, "exit", disagree="abstain")
    assert ks.match({"red": True, "big": True, "wet": False}, "exit") is a     # ties: the earlier skill
    assert ks.match({"red": True, "big": True, "wet": True}, "exit") is c      # the most specific
    assert ks.match({"red": False, "big": True, "wet": False}, "exit") is b
    assert not hasattr(ks, "abstained") and "abstained" not in ks.report()


# ───────────────────────────────────────────── the search cost of a skill learner
class Corridor:
    def __init__(self):
        from solvi.core import Outcome
        self.Outcome = Outcome
        self.reset(0)

    def reset(self, seed=None):
        self.pos = 0
        return {"pos": 0}

    def actions(self, state):
        return ["left", "right"]

    def step(self, action):
        to = self.pos + (1 if action == "right" else -1)
        if not 0 <= to < 6:
            return self.Outcome({"pos": self.pos}, accepted=False)
        self.pos = to
        return self.Outcome({"pos": to}, accepted=True, done=to == 3)


def test_skill_learner_episodes_record_their_search_cost():
    g = Grammar([Atom(f"pos={i}", (lambda s, i=i: s["pos"] == i), feature="pos") for i in range(6)])
    L = SkillLearner(Corridor(), g, {"door": lambda s: s["pos"] == 3}, horizon=6, budget=1000,
                     certificate=Certificate(0.5, 0.5))
    e1 = L.run(seed=0, steps=20)
    c = e1["cost"]
    assert c["candidates"] > 0 and c["tests"] >= 1 and c["probes"] == 0 and c["simulated"] == e1["simulated"]
    assert L.report()["totals"]["cost"]["simulated"] >= c["simulated"]
