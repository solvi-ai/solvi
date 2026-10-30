"""Laws fixes from the 2026-10-30 external handoff: the falsified set is LawSet state (S303 — a candidate refuted
where it applies by a leftover probe is not re-proposed or re-probed until the buffer grows), `mismatch(…, buffer=)`
carries the negations="refine" candidates into `admit_by_probes` (S305), `residual_check(dof_sigma=)` tests an
estimated σ̂ at the F quantile instead of the known-σ χ² (S306), and the docstring carries the measured number of a
wrong law that fits within the noise (S307)."""
from __future__ import annotations

import itertools
import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, Buffer, Candidate, Grammar, LawSet, Transition, residual_check)


# ───────────────────────────────────────────── S303: the falsified law is remembered
N4 = 4
STATES4 = [tuple(bits) for bits in itertools.product((False, True), repeat=N4)]


def wrong_world(s, a):
    return "E" if s[0] and not s[1] else "N"          # needs a negation the grammar lacks


def setup4():
    g = Grammar([Atom(f"x{i}", (lambda s, i=i: s[i])) for i in range(N4)], k=2)
    b = Buffer(g)
    s = (True, False, False, False)
    b.add(Transition(s, "do", wrong_world(s, "do"), group=0, id="r0"))
    return g, b, [Candidate("do", ("x0",), "E")], [(st, "do") for st in STATES4]


def test_a_falsified_law_is_not_re_probed_until_the_buffer_grows():
    g, b, cands, probes = setup4()
    ls = LawSet(g, base=lambda s, a: "N")
    first = ls.admit_by_probes(cands, wrong_world, probes=probes, budget=8, buffer=b, falsify="leftover")
    assert first.status == "undetermined" and "falsified by probe" in first.why
    spent, asked = ls.cost["probes"], len(first.asked)
    assert asked > 0

    second = ls.admit_by_probes(cands, wrong_world, probes=probes, budget=8, buffer=b, falsify="leftover")
    assert second.status == "undetermined" and second.law is None and second.asked == []
    assert "every candidate was falsified by a leftover probe" in second.why
    assert "the buffer has not grown since" in second.why and f"({len(b)} rows)" in second.why
    assert ls.cost["probes"] == spent and ls.laws == [] and ls.refused_falsified == 1

    wn = ls.why_not(cands[0], b)
    assert wn["failed"] == "falsified" and "falsified by a leftover probe" in wn["reason"]

    b.add(Transition((True, True, False, False), "do", "N", group=1, id="r1"))    # the buffer grows
    wn = ls.why_not(cands[0], b)
    assert wn["failed"] != "falsified"                                          # the falsification no longer blocks
    third = ls.admit_by_probes(cands, wrong_world, probes=probes, budget=8, buffer=b, falsify="leftover")
    assert third.status == "undetermined" and "every candidate was falsified" not in third.why
    assert third.asked == []                                                     # the new row refutes it, no probe needed


def test_a_falsified_law_is_not_re_proposed_by_mismatch():
    g, b, cands, probes = setup4()
    ls = LawSet(g, base=lambda s, a: "N")
    ls.admit_by_probes(cands, wrong_world, probes=probes, budget=8, buffer=b, falsify="leftover")
    again = ls.mismatch((True, False, False, False), "do", "E", buffer=b, exclude=0)
    assert all(c.condition != ("x0",) for c in again) and ls.refused_falsified == 1  # dropped, not re-proposed


def test_every_candidate_is_falsified_at_most_once_without_a_buffer():
    # the handoff's repro world: no buffer, so nothing persists between calls but the set's own memory
    states = [tuple(bits) for bits in itertools.product((0, 1), repeat=6)]
    truth = lambda s, a: "fire" if s[0] and not s[1] else "idle"
    g = Grammar([Atom(f"x{i}", (lambda s, i=i: s[i] == 1), feature=f"x{i}") for i in range(6)], k=3)
    ls = LawSet(g, base=lambda s, a: "idle")
    pool, s0 = [(s, "act") for s in states], (1, 0, 1, 1, 1, 1)
    fell = []
    for _ in range(16):
        adm = ls.admit_by_probes(ls.mismatch(s0, "act", "fire"), truth, probes=pool, budget=20, falsify="leftover")
        if "falsified by probe" in adm.why:
            fell.append(adm.why.split("the accepted law ")[1].split(" is")[0])
    assert fell and len(set(fell)) == len(fell)                                  # no candidate falsified twice
    assert len(set(fell)) == len(fell[:11])                                      # the falsifications stop: 11 laws, then no new one
    assert ls.refused_falsified >= len(set(fell))


def test_the_asked_rows_workaround_still_refutes_everything():
    # passing the asked rows back as buffer= remains the wider workaround: it refutes the split phase's refutations too
    g, b, cands, probes = setup4()
    ls = LawSet(g, base=lambda s, a: "N")
    adm = ls.admit_by_probes(cands, wrong_world, probes=probes, budget=8, falsify="leftover")
    assert adm.status == "undetermined" and adm.asked
    mem = Buffer(g)
    mem.extend(adm.asked)
    again = ls.admit_by_probes(cands, wrong_world, probes=probes, budget=8, buffer=mem, falsify="leftover")
    assert again.status == "undetermined" and again.asked == [] and "no candidate survived" in again.why


# ───────────────────────────────────────────── S305: negations="refine" reaches admit_by_probes
def refine_world():
    """5 bits, truth x0 ∧ ¬x2 → fire, the grammar knows negations: the true law is expressible only refined."""
    states = list(itertools.product((0, 1), repeat=5))
    truth = lambda s, a: "fire" if s[0] and not s[2] else "idle"
    g = Grammar([Atom(f"x{i}", (lambda s, i=i: s[i] == 1), feature=f"x{i}") for i in range(5)], k=2,
                negations="refine")
    buf = Buffer(g)
    buf.extend([Transition(s, "act", truth(s, "act"), group=f"g{i % 4}") for i, s in enumerate(states)])
    return states, truth, g, buf


def test_mismatch_without_a_buffer_still_proposes_the_positive_conditions_only():
    states, truth, g, buf = refine_world()
    ls = LawSet(g, base=lambda s, a: "idle")
    plain = ls.mismatch((1, 0, 0, 1, 1), "act", "fire")
    assert plain and not any("not:" in x for c in plain for x in c.condition)


def test_mismatch_with_the_buffer_proposes_the_refinements_and_admits_the_true_law():
    states, truth, g, buf = refine_world()
    ls = LawSet(g, base=lambda s, a: "idle")
    cands = ls.mismatch((1, 0, 0, 1, 1), "act", "fire", buffer=buf)
    negated = [c.condition for c in cands if any("not:" in x for x in c.condition)]
    assert ("not:x2", "x0") in negated and ("not:x2",) in negated
    adm = ls.admit_by_probes(cands, truth, probes=[(s, "act") for s in states], budget=20, buffer=buf)
    assert adm.status == "accepted" and adm.law is not None
    assert adm.law.condition == ("not:x2", "x0")                                  # x0 unless x2: the hidden law
    assert all(ls.predict(s, "act") == truth(s, "act") for s in states)


def test_admit_by_probes_cannot_add_the_refinements_on_its_own():
    # its buffer= arrives after the candidates were computed: the positive candidates alone stay undetermined
    states, truth, g, buf = refine_world()
    ls = LawSet(g, base=lambda s, a: "idle")
    plain = ls.mismatch((1, 0, 0, 1, 1), "act", "fire")
    adm = ls.admit_by_probes(plain, truth, probes=[(s, "act") for s in states], budget=20, buffer=buf)
    assert adm.status == "undetermined"


# ───────────────────────────────────────────── S306: an estimated σ̂ is tested at the F quantile
def sigma_hat_trials(trials, seed=0):
    """The handoff's stand: the TRUE law y = 2x at σ = 1 %, σ̂ from 15 repetitions of 3 (10 dof) → per trial the
    residuals in units of the known σ and in units of σ̂."""
    rng = random.Random(seed)
    known, estimated = [], []
    for _ in range(trials):
        xs = [rng.uniform(1, 5) for _ in range(60)]
        ys = [2 * x * (1 + 0.01 * rng.gauss(0, 1)) for x in xs]
        reps = [[1 + 0.01 * rng.gauss(0, 1) for _ in range(3)] for _ in range(5)]
        s_hat = math.sqrt(sum((r - sum(p) / 3) ** 2 for p in reps for r in p) / 10)
        c = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        rel = [(y - c * x) / (c * x) for x, y in zip(xs, ys)]
        known.append([r / 0.01 for r in rel])
        estimated.append([r / s_hat for r in rel])
    return known, estimated


def test_an_estimated_sigma_is_refused_far_above_nominal_without_the_option():
    known, estimated = sigma_hat_trials(800)
    known_refused = sum(not residual_check(lambda c: (r, 1))(None) for r in known) / 800
    assert known_refused <= 0.015                                                 # the known σ: at the nominal 0.005
    plain = sum(not residual_check(lambda c: (r, 1))(None) for r in estimated) / 800
    assert plain > 0.10                                                           # the handoff measured 0.2315 at 2000


def test_dof_sigma_tests_the_estimated_sigma_at_the_nominal_level():
    _, estimated = sigma_hat_trials(800)
    refused = sum(not residual_check(lambda c: (r, 1), dof_sigma=10)(None) for r in estimated) / 800
    assert refused <= 0.02                                                            # nominal 0.005 (measured 0.0055 at 2000)


def test_dof_sigma_defaults_and_validation():
    check = residual_check(lambda c: ([0.1, -0.2, 0.3], 1))
    assert check.dof_sigma is None and residual_check(lambda c: None, dof_sigma=7).dof_sigma == 7
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            residual_check(lambda c: None, dof_sigma=bad)


# ───────────────────────────────────────────── S307: a wrong law that fits within the noise passes
def test_a_wrong_formula_within_the_noise_passes_and_the_docstring_says_so():
    # truth y = x(1 + 0.05x) at 10 % noise; the wrong y = c·x, c refitted per trial, σ known — passes ≈ 96 % of trials
    rng, passed, trials = random.Random(0), 0, 600
    for _ in range(trials):
        xs = [rng.uniform(1, 2) for _ in range(60)]
        ys = [x * (1 + 0.05 * x) * (1 + 0.1 * rng.gauss(0, 1)) for x in xs]
        w = [1 / (y * y) for y in ys]
        c = sum(wi * x * y for wi, x, y in zip(w, xs, ys)) / sum(wi * x * x for wi, x in zip(w, xs))
        rel = [(y - c * x) / (c * x) / 0.1 for x, y in zip(xs, ys)]
        passed += residual_check(lambda cand: (rel, 1))(None)
    assert passed / trials >= 0.90                                                    # the χ² alone is no guarantee (0.961 at 1000)
    assert "0.961" in residual_check.__doc__                                          # the measured number is documented
    assert "dof_sigma" in residual_check.__doc__
