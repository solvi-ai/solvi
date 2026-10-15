"""solvi.experimental.changes: revert(version) undoes one version (rollback(v) undoes every later version) and
ShipWatch, a post-ship watch in units of the promise — per slice a test martingale on the promoted change's alone
outcomes against "this slice fails at most its promise"; when one slice's evidence reaches slices/alpha the version is
reverted (alone) and the event journaled."""
from __future__ import annotations

import random
import warnings

import pytest

from solvi.core.knowledge import KnowledgeStore

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import Change, KnownFailures, Pipeline, ShipWatch, Static


def fact(i):
    return Change("person", payload={"kind": "fact", "body": {"s": f"thing{i}", "r": "size", "o": i}}, proposer="ann")


def pipeline(ks=None, stages=None):
    ks = KnowledgeStore() if ks is None else ks
    return Pipeline(stages or [Static()], None, store=ks, approval={"fact": "gate"}), ks


def promoted(pipe, n):
    out = []
    for i in range(n):
        r = pipe.propose(fact(i))
        assert r.promoted, str(r)
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ revert
def test_revert_undoes_one_version_and_keeps_the_later_ones():
    pipe, ks = pipeline()
    r1, r2, r3 = promoted(pipe, 3)
    assert pipe.revert(r2.version, by="ann", why="slice harm") == [r2.version]
    assert [v["version"] for v in pipe.versions() if v["in_force"]] == [r1.version, r3.version]
    assert pipe.current == r3.version
    assert ks.status(r2.ref) == "retracted" and ks.status(r1.ref) == ks.status(r3.ref) == "active"
    rec = pipe.history("rollback")[-1]
    assert rec["undone"] == [r2.version] and rec["revert"] == r2.version and rec["failed"] is True
    assert pipe.journal.verify()["ok"]
    # rollback still works on what is left
    assert pipe.rollback(0) == [r3.version, r1.version] and pipe.current == 0


def test_a_reverted_change_with_a_reason_is_a_known_failure():
    pipe, _ = pipeline(stages=[KnownFailures(), Static()])
    r1, r2 = promoted(pipe, 2)
    pipe.revert(r1.version, why="harm")
    again = pipe.propose(fact(0))
    assert again.status == "rejected" and again.stage == "failure_memory"


def test_revert_of_a_version_not_in_force_is_an_error():
    pipe, _ = pipeline()
    r1, r2 = promoted(pipe, 2)
    pipe.revert(r1.version)
    with pytest.raises(KeyError):
        pipe.revert(r1.version)
    with pytest.raises(KeyError):
        pipe.revert(0)


class Snap:
    """A snapshot target: rollback(ref) restores the state before the change (all later state with it)."""
    artifact = "update"

    def __init__(self):
        self.state = None

    def promote(self, change, summary):
        before, self.state = self.state, change.payload["fingerprint"]
        return {"before": before}

    def rollback(self, ref, by=None, why=None):
        self.state = ref["before"]


def test_revert_refuses_a_snapshot_version_out_of_the_middle():
    t = Snap()
    pipe = Pipeline([], None, targets=[t], approval={("update", "person"): "gate"})
    a = pipe.propose(Change("person", artifact="update", payload={"fingerprint": "a"}))
    b = pipe.propose(Change("person", artifact="update", payload={"fingerprint": "b"}))
    with pytest.raises(ValueError, match="snapshot"):
        pipe.revert(a.version)
    assert pipe.revert(b.version) == [b.version] and t.state == "a" and pipe.current == a.version


# ------------------------------------------------------------------------------------------------ ShipWatch
SLICES = {"a": 0.60, "b": 0.25, "c": 0.10, "rare": 0.05}


def stream(rates, seed):
    rng = random.Random(seed)
    names, w = list(SLICES), list(SLICES.values())
    while True:
        s = rng.choices(names, w)[0]
        yield s, rng.random() >= rates[s]


def test_harm_in_a_small_slice_reverts_the_watched_version_alone():
    pipe, ks = pipeline()
    r1, r2 = promoted(pipe, 2)                  # r1 shipped and watched; r2 a later independent change
    w = ShipWatch(pipe, r1.version, promise={s: 0.10 for s in SLICES}, alpha=0.05, horizon=8000)
    rates = {"a": 0.05, "b": 0.05, "c": 0.05, "rare": 0.40}
    ev = None
    for n, (s, ok) in enumerate(stream(rates, 1)):
        ev = w.observe(ok, slice=s)
        if ev:
            break
    assert ev["event"] == "reverted" and ev["slice"] == "rare" and ev["undone"] == [r1.version]
    assert not w.active and pipe.current == r2.version
    assert [v["version"] for v in pipe.versions() if v["in_force"]] == [r2.version]
    obs = pipe.history("observed")[-1]
    assert obs["observed"]["event"] == "reverted" and obs["change"] == r1.change_id and obs["by"] == "ship_watch"
    assert w.observe(True, slice="a") is None   # an ended watch ignores outcomes


def test_a_clean_rollout_graduates_at_the_horizon():
    pipe, _ = pipeline()
    (r1,) = promoted(pipe, 1)
    w = ShipWatch(pipe, r1.version, promise=0.10, slices=list(SLICES), horizon=500)
    ev = None
    for s, ok in stream({k: 0.05 for k in SLICES}, 2):
        ev = w.observe(ok, slice=s) or ev
        if not w.active:
            break
    assert ev["event"] == "graduated" and ev["observations"] == 500 and pipe.current == r1.version
    assert pipe.history("observed")[-1]["observed"]["event"] == "graduated"
    st = ev["slices"]["rare"]
    assert set(st) >= {"promise", "outcomes", "failures", "evidence", "threshold"}


def test_supervised_outcomes_are_recorded_but_not_bet_on():
    pipe, _ = pipeline()
    (r1,) = promoted(pipe, 1)
    w = ShipWatch(pipe, r1.version, promise=0.10, slices=["x"], horizon=10 ** 6)
    for _ in range(200):
        assert w.observe(False, slice="x", alone=False) is None
    st = w.status()["x"]
    assert st["outcomes"] == 0 and st["supervised"] == 200 and st["evidence"] == 0.0


def test_slices_and_promise_are_checked():
    pipe, _ = pipeline()
    (r1,) = promoted(pipe, 1)
    with pytest.raises(ValueError):
        ShipWatch(pipe, r1.version, promise=0.10)              # a single promise needs slices=
    with pytest.raises(ValueError):
        ShipWatch(pipe, r1.version, promise={"a": 1.0})
    with pytest.raises(ValueError):
        ShipWatch(pipe, r1.version, promise=0.1, slices=["a"], alpha=0.0)
    with pytest.raises(KeyError):
        ShipWatch(pipe, 99, promise=0.1, slices=["a"])          # not a version in force
    w = ShipWatch(pipe, r1.version, promise=0.1, slices=["a"])
    with pytest.raises(KeyError):
        w.observe(True, slice="b")                             # undeclared slice: no alpha was spent on it


def test_a_watch_whose_version_was_rolled_back_elsewhere_ends_without_reverting():
    pipe, _ = pipeline()
    r1, r2 = promoted(pipe, 2)
    w = ShipWatch(pipe, r2.version, promise=0.10, slices=["x"], horizon=10 ** 6)
    pipe.rollback(r1.version)
    ev = None
    for _ in range(100):
        ev = w.observe(False, slice="x")
        if ev:
            break
    assert ev["event"] == "ended" and pipe.current == r1.version and not w.active


def test_the_pipeline_hands_out_a_watch():
    pipe, _ = pipeline()
    (r1,) = promoted(pipe, 1)
    w = pipe.watch(r1.version, promise=0.2, slices=["x"])
    assert isinstance(w, ShipWatch) and w.version == r1.version and w.active


def test_false_reverts_at_the_promise_are_at_most_alpha():
    """Ville: every slice failing exactly at its promise (the worst clean case), few reverts over a long watch."""
    pipe, _ = pipeline()
    (r1,) = promoted(pipe, 1)
    hits = 0
    for seed in range(200):
        w = ShipWatch(pipe, r1.version, promise=0.10, slices=list(SLICES), alpha=0.05, horizon=600, act=False)
        for s, ok in stream({k: 0.10 for k in SLICES}, 1000 + seed):
            ev = w.observe(ok, slice=s)
            if ev:
                hits += ev["event"] == "reverted"
                break
    assert pipe.current == r1.version and hits / 200 <= 0.05 + 0.035
