"""solvi.experimental.changes.SliceGuard: a change is judged per slice, not on the aggregate (harm in a 3 % slice under
cover of an overall improvement passes an aggregate gate), and against the ORIGINAL version as well as the one in force
(a chain of small changes, each under the tolerance and not significant on its own — "salami" — adds up; the
comparison with the original sees the sum)."""
from __future__ import annotations

import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import Change, Pipeline, SliceGuard, StageVerdict

SLICES = {"main": 300, "rare": 40}
NOISE = 0.10


class Shifts:
    """A toy target: per-slice loss shifts that accumulate with every promoted change (payload {"shift": {slice: d}})."""
    artifact = "update"

    def __init__(self):
        self.shift = {s: 0.0 for s in SLICES}

    def promote(self, change, measured):
        for s, d in change.payload["shift"].items():
            self.shift[s] += d
        return dict(change.payload["shift"])

    def rollback(self, ref, by=None, why=None):
        for s, d in ref.items():
            self.shift[s] -= d


def base(s):
    r = random.Random(s)
    return [r.random() for _ in range(SLICES[s])]


def measure_of(target):
    """Paired per-slice losses: the version in force and the candidate on the same cases; the candidate's per-case
    difference is its shift plus zero-mean noise (seeded by the change)."""
    def measure(change, ctx):
        out = {}
        for s in SLICES:
            r = random.Random(f"{change.fingerprint()}/{s}")
            now = [b + target.shift[s] for b in base(s)]
            if s not in change.payload["shift"]:          # a slice the change does not touch stays as it is
                out[s] = (now, list(now))
                continue
            d = change.payload["shift"][s]
            out[s] = (now, [x + d + r.gauss(0, NOISE) for x in now])
        return out
    return measure


class Aggregate:
    """An aggregate gate: the mean loss over every case is not higher."""
    name = "aggregate"

    def __init__(self, measure):
        self.measure = measure

    def run(self, change, ctx):
        now, cand = [], []
        for a, b in self.measure(change, ctx).values():
            now += a
            cand += b
        d = sum(cand) / len(cand) - sum(now) / len(now)
        return StageVerdict("admit" if d <= 0.0 else "refuse", f"aggregate {d:+.4f}", {"d": d})


def shift(i, **d):
    return Change("outcome", artifact="update", payload={"shift": d, "fingerprint": f"step{i}:{sorted(d.items())}"},
                  proposer="tuner", reason=f"step {i}")


def pipe_of(target, *stages):
    return Pipeline(list(stages), None, targets=[target], approval={"update": "gate"})


def test_harm_in_a_slice_under_an_overall_improvement_passes_the_aggregate_and_is_refused_per_slice():
    ch = shift(0, main=-0.05, rare=+0.10)            # better on 300 cases, worse by 0.10 on 40
    t = Shifts()
    assert pipe_of(t, Aggregate(measure_of(t))).propose(ch).promoted
    t = Shifts()
    res = pipe_of(t, Aggregate(measure_of(t)), SliceGuard(measure_of(t), tol=0.03)).propose(ch)
    assert res.status == "rejected" and res.stage == "slices" and "rare" in res.reason and "main" not in res.reason
    assert t.shift["rare"] == 0.0


def test_salami_passes_one_by_one_against_the_version_in_force_and_is_caught_against_the_original():
    steps = [shift(i, rare=0.012) for i in range(8)]  # each: under tol, rarely significant on 40 noisy cases
    t = Shifts()
    p = pipe_of(t, SliceGuard(measure_of(t), tol=0.03, original=False))
    through = sum(p.propose(c).promoted for c in steps)
    assert through >= 7 and t.shift["rare"] >= 0.08  # the chain is in: +0.084 on "rare", 2.8× the tolerance

    t = Shifts()
    p = pipe_of(t, SliceGuard(measure_of(t), tol=0.03))   # the original pinned at the guard's first run
    out = [p.propose(c) for c in steps]
    first = next(i for i, r in enumerate(out) if not r.promoted)
    assert out[first].stage == "slices" and "vs original" in out[first].reason
    assert not any(r.promoted for r in out[-4:])      # the chain stays stopped: the sum is what is judged
    assert t.shift["rare"] <= 0.03 + 0.012 + 1e-9
    m = next(v for v in out[first].verdicts if v.stage == "slices").measured
    assert m["rare"]["original"]["d"] > 0.03 and m["rare"]["in_force"]["d"] < 0.03


def test_the_original_can_be_given_and_must_be_measured_on_the_same_cases():
    t = Shifts()
    orig = {s: base(s) for s in SLICES}
    t.shift["rare"] = 0.05                            # already drifted before the guard was built
    res = pipe_of(t, SliceGuard(measure_of(t), tol=0.03, original=orig)).propose(shift(0, rare=0.0))
    assert res.stage == "slices" and "rare" in res.reason and "vs original" in res.reason
    bad = pipe_of(Shifts(), SliceGuard(measure_of(Shifts()), original={"rare": [0.0] * 5}))
    res = bad.propose(shift(0, rare=0.0))
    assert res.status == "error" and "same cases" in res.reason


def test_small_or_missing_slices_hold_by_default_and_on_small_chooses():
    def few(ch, ctx):
        return {"rare": ([0.0] * 5, [0.0] * 5)}
    assert pipe_of(Shifts(), SliceGuard(few, min_n=30)).propose(shift(0)).status == "held"
    assert pipe_of(Shifts(), SliceGuard(few, min_n=30, on_small="refuse")).propose(shift(0)).status == "rejected"
    assert pipe_of(Shifts(), SliceGuard(few, min_n=30, on_small="admit")).propose(shift(0)).promoted
    # a slice the original had and the measurement lost is unmeasured, not passed
    g = SliceGuard(few, min_n=5, original={"rare": [0.0] * 5, "gone": [0.0] * 50})
    res = pipe_of(Shifts(), g).propose(shift(0))
    assert res.status == "held" and "gone" in res.reason


def test_the_significance_test_is_corrected_for_the_number_of_slices():
    g1 = SliceGuard(lambda ch, ctx: {}, alpha=0.05)
    assert g1.z(1) < g1.z(10)                         # Bonferroni: a stricter z with more comparisons
    # an exactly constant positive difference has no noise to hide in: refused even under tol
    flat = SliceGuard(lambda ch, ctx: {"a": ([0.0] * 50, [0.01] * 50)}, tol=0.03, original=False)
    assert pipe_of(Shifts(), flat).propose(shift(0)).stage == "slices"
    # equal is fine
    same = SliceGuard(lambda ch, ctx: {"a": ([0.0] * 50, [0.0] * 50)}, tol=0.03)
    assert pipe_of(Shifts(), same).propose(shift(0)).promoted


def test_arguments_are_checked():
    with pytest.raises(ValueError):
        SliceGuard(lambda ch, ctx: {}, on_small="maybe")
    with pytest.raises(ValueError):
        SliceGuard(lambda ch, ctx: {}, alpha=1.5)
    with pytest.raises(ValueError):
        SliceGuard(lambda ch, ctx: {}, tol=-0.1)
