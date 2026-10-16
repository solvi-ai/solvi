"""solvi.experimental.laws: probes under measurement noise — `split(score="eig", noise=ε)` and
`admit_by_probes(..., noise=ε)` (hypotheses weighted by likelihood instead of dropped by one answer)."""
from __future__ import annotations

import itertools
import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Enumerate, Grammar, LawSet, eig, split

F = 6
STATES = [tuple(b) for b in itertools.product((False, True), repeat=F)]


def grammar():
    atoms = []
    for i in range(F):
        atoms.append(Atom(f"x{i}", (lambda s, i=i: s[i]), feature=f"x{i}"))
        atoms.append(Atom(f"~x{i}", (lambda s, i=i: not s[i]), feature=f"x{i}"))
    return Grammar(atoms, k=2)


def holds(cond, s):
    return all((s[int(x.lstrip("~x"))] is not x.startswith("~")) for x in cond)


def world(seed):
    rng = random.Random(seed)
    feats = rng.sample(range(F), 2)
    truth = tuple(sorted(("~" if rng.random() < 0.5 else "") + f"x{i}" for i in feats))
    s0 = next(s for s in rng.sample(STATES, len(STATES)) if holds(truth, s))
    return truth, s0


def h2(e):
    return -e * math.log2(e) - (1 - e) * math.log2(1 - e)


def test_eig_of_an_even_binary_split_is_one_bit_less_the_noise_entropy():
    assert eig([0.5, 0.5], 0.0) == pytest.approx(1.0)
    assert eig([0.5, 0.5], 0.1) == pytest.approx(1 - h2(0.1))
    assert eig([1.0], 0.1) == 0.0
    assert eig([1.0, 0.0], 0.1) == pytest.approx(0.0, abs=1e-12)
    assert eig([0.9, 0.1], 0.1) < eig([0.5, 0.5], 0.1)


def test_split_eig_follows_the_weights():
    hyps = [lambda s, a: s == 0, lambda s, a: s == 1, lambda s, a: False]
    probes = [(0, "a"), (1, "a")]
    # equal weights: both probes split one hypothesis off — the first is taken
    p = split(hyps, lambda s, a: True, probes, score="eig", noise=0.1)
    assert p.state == 0 and p.info == pytest.approx(eig([1 / 3, 2 / 3], 0.1))
    # the second hypothesis heavy: the probe that separates it is worth more
    p = split(hyps, lambda s, a: True, probes, score="eig", noise=0.1, weights=[0.1, 0.8, 0.1])
    assert p.state == 1


def test_split_eig_with_no_noise_and_equal_weights_is_the_split_entropy():
    hyps = [lambda s, a: s % 2, lambda s, a: s % 3, lambda s, a: 0, lambda s, a: s]
    probes = [(s, "a") for s in range(6)]
    a = split(hyps, lambda s, a: True, probes)
    b = split(hyps, lambda s, a: True, probes, score="eig", noise=0.0)
    assert b.info == pytest.approx(max(split(hyps, lambda s, a: True, [p]).info for p in probes
                                       if split(hyps, lambda s, a: True, [p]) is not None))
    assert a.info <= b.info + 1e-12


def test_split_arguments_are_checked():
    hyps = [lambda s, a: s, lambda s, a: 0]
    with pytest.raises(ValueError):
        split(hyps, lambda s, a: True, [(1, "a")], score="bits")
    with pytest.raises(ValueError):
        split(hyps, lambda s, a: True, [(1, "a")], score="eig", noise=0.5)
    with pytest.raises(ValueError):
        split(hyps, lambda s, a: True, [(1, "a")], noise=0.1)
    with pytest.raises(ValueError):
        split(hyps, lambda s, a: True, [(1, "a")], score="eig", weights=[1.0])


def run(seed, noise_world, noise_arg, budget, level=0.95):
    truth, s0 = world(seed)
    rng = random.Random(seed + 1000)

    def ask(s, a):
        on = holds(truth, s)
        if rng.random() < noise_world:
            on = not on
        return "on" if on else "off"

    ls = LawSet(grammar(), base=lambda s, a: "off")
    cands = Enumerate().propose(ls, s0, "do", "on")
    pool = [(s, "do") for s in STATES]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adm = ls.admit_by_probes(cands, ask, probes=pool, budget=budget, noise=noise_arg, level=level)
    if adm.status != "accepted":
        return "undetermined", len(adm.asked)
    ok = all(holds(adm.law.condition, s) == holds(truth, s) for s in STATES)
    return ("right" if ok else "wrong"), len(adm.asked)


def test_noisy_admission_admits_the_right_law_where_one_answer_refutes_it():
    out = [run(seed, 0.1, 0.1, 40)[0] for seed in range(20)]
    base = [run(seed, 0.1, None, 40)[0] for seed in range(20)]
    assert out.count("right") >= 16 and out.count("wrong") <= 1
    assert base.count("right") < out.count("right")


def test_noisy_admission_without_noise_in_the_world_stops_early_and_is_right():
    res = [run(seed, 0.0, 0.1, 40) for seed in range(10)]
    assert all(r == "right" for r, _ in res)
    assert max(n for _, n in res) <= 40


def test_a_small_budget_is_undetermined_not_a_guess():
    res = [run(seed, 0.2, 0.2, 2)[0] for seed in range(10)]
    assert res.count("wrong") == 0 and res.count("undetermined") >= 8


def test_noise_arguments_are_checked():
    ls = LawSet(grammar(), base=lambda s, a: "off")
    pool = [(s, "do") for s in STATES]
    with pytest.raises(ValueError):
        ls.admit_by_probes([], lambda s, a: "off", probes=pool, noise=0.6)
    with pytest.raises(ValueError):
        ls.admit_by_probes([], lambda s, a: "off", probes=pool, noise=0.1, level=1.0)
    with pytest.raises(ValueError):
        ls.admit_by_probes([], lambda s, a: "off", probes=pool, noise=0.1, max_new=2)


def test_an_answer_no_hypothesis_predicts_moves_nothing():
    truth, s0 = world(3)
    ls = LawSet(grammar(), base=lambda s, a: "off")
    cands = Enumerate().propose(ls, s0, "do", "on")
    adm = ls.admit_by_probes(cands, lambda s, a: "???", probes=[(s, "do") for s in STATES], budget=5, noise=0.1)
    assert adm.status == "undetermined" and len(adm.asked) == 5
