"""solvi.experimental.laws: `admit_by_probes(..., noise=ε)` weighs only the offered hypotheses against each other
(a closed world). When the true law is not among the candidates, the least-bad wrong one used to win the weight and be
accepted although it disagrees with far more answers than ε allows. The fit check (`fit_level`, `fit_support`) refuses
such a law: "undetermined", with a reason that names the fit."""
from __future__ import annotations

import itertools
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Enumerate, Grammar, LawSet, Transition

F = 6
STATES = [tuple(b) for b in itertools.product((False, True), repeat=F)]
TRUTH = ("x0", "x1")                                  # the hidden law: do → on where x0 ∧ x1, else off


def grammar():
    atoms = []
    for i in range(F):
        atoms.append(Atom(f"x{i}", (lambda s, i=i: s[i]), feature=f"x{i}"))
        atoms.append(Atom(f"~x{i}", (lambda s, i=i: not s[i]), feature=f"x{i}"))
    return Grammar(atoms, k=2)


def holds(cond, s):
    return all((s[int(x.lstrip("~x"))] is not x.startswith("~")) for x in cond)


def setup(offer_truth, noise_world=0.0, seed=0):
    rng = random.Random(seed)

    def ask(s, a):
        on = holds(TRUTH, s)
        if rng.random() < noise_world:
            on = not on
        return "on" if on else "off"

    ls = LawSet(grammar(), base=lambda s, a: "off")
    s0 = next(s for s in STATES if holds(TRUTH, s))
    cands = Enumerate().propose(ls, s0, "do", "on")
    if not offer_truth:
        cands = [c for c in cands if tuple(sorted(c.condition)) != TRUTH]
    return ls, cands, ask, [(s, "do") for s in STATES]


def admit(ls, cands, ask, pool, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ls.admit_by_probes(cands, ask, probes=pool, noise=0.05, **kw)


def right(adm):
    return adm.status == "accepted" and all(holds(adm.law.condition, s) == holds(TRUTH, s) for s in STATES)


def logged(ls, ask, n=200, seed=1):
    buf = Buffer(ls.grammar)
    rng = random.Random(seed)
    for i in range(n):
        s = rng.choice(STATES)
        buf.add(Transition(s, "do", ask(s, "do"), group=f"g{i % 5}", id=f"r{i}"))
    return buf


@pytest.mark.parametrize("budget", [0, 8])
def test_true_law_not_offered_is_undetermined_not_the_least_bad_wrong_law(budget):
    # the logged rows alone push a law wrong on about half of the rows where it fires to the top of the weight
    ls, cands, ask, pool = setup(offer_truth=False)
    buf = logged(ls, ask)
    adm = admit(ls, cands, ask, pool, budget=budget, buffer=buf)
    assert adm.status == "undetermined" and adm.law is None and not ls.laws
    assert "does not fit the noise" in adm.why
    # without the fit check (fit_level=None, as in 1.2.0) the same call accepted it
    ls2, cands2, ask2, _ = setup(offer_truth=False)
    old = admit(ls2, cands2, ask2, pool, budget=budget, buffer=logged(ls2, ask2), fit_level=None)
    assert old.status == "accepted" and not right(old)
    assert (old.law.support - old.law.hits) / old.law.support > 0.3      # precision far below 1 − ε


def test_too_little_support_is_undetermined():
    ls, cands, ask, pool = setup(offer_truth=False)
    adm = admit(ls, cands, ask, pool, budget=0, buffer=logged(ls, ask), fit_support=10_000)
    assert adm.status == "undetermined" and "too little support" in adm.why


@pytest.mark.parametrize("with_buffer", [False, True])
def test_true_law_offered_is_still_accepted(with_buffer):
    for seed in range(10):
        ls, cands, ask, pool = setup(offer_truth=True, noise_world=0.05, seed=seed)
        adm = admit(ls, cands, ask, pool, budget=40, buffer=logged(ls, ask, seed=seed) if with_buffer else None)
        ls2, cands2, ask2, _ = setup(offer_truth=True, noise_world=0.05, seed=seed)
        old = admit(ls2, cands2, ask2, pool, budget=40, fit_level=None,
                    buffer=logged(ls2, ask2, seed=seed) if with_buffer else None)
        assert (adm.status, adm.law and adm.law.condition, len(adm.asked)) == \
            (old.status, old.law and old.law.condition, len(old.asked))
        assert right(adm)


def test_fit_arguments_are_checked():
    ls, cands, ask, pool = setup(offer_truth=True)
    with pytest.raises(ValueError):
        admit(ls, cands, ask, pool, fit_level=1.0)
    with pytest.raises(ValueError):
        admit(ls, cands, ask, pool, fit_support=-1)
