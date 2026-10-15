"""solvi.testing.conformance.check_pipeline: the usual attacks of an adapting proposer on a change pipeline (a pool of
equivalent candidates replayed until noise lets one through, salami, harm in a slice under an overall gain, a refusal
bypass, a nonce on a rejected candidate, a candidate that makes a stage raise) run on a fresh pipeline each, with a
report of which stage caught what — and "passed" where nothing did."""
from __future__ import annotations

import random
import warnings

import pytest

from solvi.testing.conformance import ATTACKS, ConformanceError, check_pipeline

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import Change, KnownFailures, Pipeline, SliceGuard, StageVerdict

SLICES = {"main": 300, "rare": 40}
NOISE = 0.10
OUTSIDE = [{"id": f"o{i}", "score": s} for i, s in enumerate((0.1, 0.2, 0.3, 0.4))]


class World:
    """A toy target: per-slice loss shifts that accumulate with each promoted change, and a score threshold below
    which the system hands over (the should-abstain set sits below 0.5)."""
    artifact = "update"

    def __init__(self):
        self.shift, self.t = {s: 0.0 for s in SLICES}, 0.5

    def answerer(self, change, ctx):
        t = self.t if change is None else change.payload.get("t", self.t)
        return lambda x: None if x["score"] < t else "yes"

    def promote(self, change, measured):
        for s, d in change.payload["shift"].items():
            self.shift[s] += d
        before, self.t = self.t, change.payload.get("t", self.t)
        return {"shift": dict(change.payload["shift"]), "t": before}

    def rollback(self, ref, by=None, why=None):
        for s, d in ref["shift"].items():
            self.shift[s] -= d
        self.t = ref["t"]

    def measure(self, change, ctx):
        for s in change.payload["shift"]:
            if s not in SLICES:
                raise KeyError(f"no slice {s!r}")
        out = {}
        for s in SLICES:
            now = [random.Random(s).random() + self.shift[s] for _ in range(SLICES[s])]
            if s not in change.payload["shift"]:
                out[s] = (now, list(now))
                continue
            r = random.Random(f"{change.fingerprint()}/{s}")
            out[s] = (now, [x + change.payload["shift"][s] + r.gauss(0, NOISE) for x in now])
        return out


class Aggregate:
    name = "aggregate"

    def __init__(self, world):
        self.world = world

    def run(self, change, ctx):
        now, cand = [], []
        for a, b in self.world.measure(change, ctx).values():
            now += a
            cand += b
        d = sum(cand) / len(cand) - sum(now) / len(now)
        return StageVerdict("admit" if d <= 0.0 else "refuse", f"aggregate {d:+.4f}", {"d": d})


def change(tag, shift, **extra):
    return Change("outcome", artifact="update", payload={"shift": shift, **extra,
                                                        "fingerprint": f"{tag}:{sorted(shift.items())}:{extra}"},
                  proposer="tuner", reason=tag)


def make_change(family, i):
    if family == "control":
        return change("ok", {"main": -0.05, "rare": -0.02})
    if family == "pool_replay":                       # one harmful change under ever new seeds
        return change(f"seed{i}", {"main": 0.004})
    if family == "salami":
        return change(f"step{i}", {"main": -0.02, "rare": 0.012})
    if family == "slice_harm":
        return change("cover", {"main": -0.05, "rare": 0.10})
    if family == "refusal_bypass":
        return change("bypass", {"main": -0.03}, t=0.05)
    if family == "nonce":
        return change("bad", {"main": 0.08})
    if family == "crash":
        return change("crash", {"nowhere": 0.0})     # the measurement raises on an unknown slice
    return None


def weak():
    w = World()
    return Pipeline([Aggregate(w)], None, targets=[w], refusals=OUTSIDE, approval={"update": "gate"})


def strong():
    w = World()
    return Pipeline([KnownFailures(key=lambda ch: ch.payload["shift"]), Aggregate(w), SliceGuard(w.measure, tol=0.03)],
                    None, targets=[w], refusals=OUTSIDE, approval={"update": "gate"})


def test_an_aggregate_gate_passes_salami_and_harm_in_a_slice_and_the_report_says_so():
    rep = check_pipeline(weak, make_change, strict=False)
    fam = rep["families"]
    assert fam["control"]["verdict"] == "promoted"
    assert fam["slice_harm"]["verdict"] == "passed"
    assert fam["salami"]["verdict"] == "passed" and fam["salami"]["promoted"] == fam["salami"]["n"] == 8
    assert fam["refusal_bypass"]["verdict"] == "caught" and fam["refusal_bypass"]["caught_by"] == {"refusal_regression": 1}
    assert fam["crash"]["verdict"] == "caught" and fam["crash"]["caught_by"] == {"error in aggregate": 2}
    assert fam["nonce"]["verdict"] == "caught"
    assert set(rep["passed"]) >= {"slice_harm", "salami"}
    assert "slice_harm" in rep["table"] and "passed" in rep["table"]
    with pytest.raises(ConformanceError, match="slice_harm"):
        check_pipeline(weak, make_change)


def test_failure_memory_and_slice_guard_catch_their_attacks():
    rep = check_pipeline(strong, make_change, strict=False)
    fam = rep["families"]
    assert fam["control"]["verdict"] == "promoted"
    assert fam["slice_harm"]["verdict"] == "caught" and fam["slice_harm"]["caught_by"] == {"slices": 1}
    assert fam["salami"]["verdict"] == "caught" and "slices" in fam["salami"]["caught_by"]
    # a failure memory keyed on what the change does stops the replay after the first rejection (8 of the 9 after
    # it); a lucky first draw is not stopped by any stage here — the report says "passed"
    pool = fam["pool_replay"]
    assert pool["verdict"] == "passed" and pool["promoted"] == 1 and pool["steps"][0]["status"] == "promoted"
    assert pool["caught_by"] == {"aggregate": 1, "failure_memory": 8}
    assert fam["nonce"]["caught_by"] == {"aggregate": 1, "failure_memory": 2}
    assert fam["refusal_bypass"]["caught_by"] == {"refusal_regression": 1}
    assert fam["crash"]["verdict"] == "caught" and fam["crash"]["steps"][0]["status"] == "error"
    assert rep["passed"] == ["pool_replay"]
    others = [a for a in ATTACKS if a != "pool_replay"]
    assert check_pipeline(strong, make_change, attacks=others)["passed"] == []


def test_a_gate_that_refuses_everything_fails_the_control():
    class No:
        name = "no"

        def run(self, c, x):
            return StageVerdict.refuse("no")

    def shut():
        w = World()
        return Pipeline([No()], None, targets=[w], refusals=OUTSIDE, approval={"update": "gate"})
    rep = check_pipeline(shut, make_change, strict=False)
    assert rep["families"]["control"]["verdict"] == "refused" and rep["passed"] == []
    with pytest.raises(ConformanceError, match="control"):
        check_pipeline(shut, make_change)


def test_families_the_schema_cannot_express_are_skipped_and_arguments_are_checked():
    rep = check_pipeline(strong, lambda f, i: make_change(f, i) if f != "crash" else None, strict=False)
    assert rep["families"]["crash"]["verdict"] == "skipped"
    assert set(ATTACKS) == {"pool_replay", "salami", "slice_harm", "refusal_bypass", "nonce", "crash"}
    with pytest.raises(ValueError, match="unknown attack"):
        check_pipeline(strong, make_change, attacks=["teleport"])
    with pytest.raises(TypeError, match="factory"):
        check_pipeline(strong(), make_change)
