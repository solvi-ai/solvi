"""solvi.experimental.changes: the lifecycle of a candidate — a stage error is not a rejection, a rollback with a reason
is a failure, the failure memory's identity, context and expiry, the in-memory journal and the cached version state."""
from __future__ import annotations

import warnings

import pytest

from solvi.core.knowledge import KnowledgeStore

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import Change, KnownFailures, Pipeline, StageVerdict, hazard_body


class Target:
    """An "update" target that keeps the state it was given (and the rollbacks it was asked for)."""
    artifact = "update"

    def __init__(self):
        self.state, self.log = None, []

    def promote(self, change, summary):
        self.state = change.payload["fingerprint"]
        self.log.append(self.state)
        return self.state

    def rollback(self, ref, by=None, why=None):
        self.log.append(("undo", ref))


A = {("update", "person"): "gate"}


def update(fp, **extra):
    return Change("person", artifact="update", payload={"fingerprint": fp, **extra})


class Deny:
    name = "deny"

    def run(self, c, x):
        return StageVerdict.refuse("bad")


# ───────────────────────────────────────────── a stage that raises is an error, not a rejection
def test_a_stage_that_raises_is_an_error_and_the_failure_memory_does_not_remember_it(tmp_path):
    class Flaky:
        name = "tests"
        n = 0

        def run(self, c, x):
            Flaky.n += 1
            if Flaky.n == 1:
                raise TypeError("unhashable type: 'list'")
            return StageVerdict.admit()
    pipe = Pipeline([KnownFailures(), Flaky()], str(tmp_path / "b.jsonl"), targets=[Target()], approval=A)
    a = pipe.propose(update("s1"))
    assert a.status == "error" and a.stage == "tests" and "TypeError" in a.reason and not a.promoted
    assert "error" in str(a) and pipe.history("rejected") == []
    err = pipe.history("error")
    assert len(err) == 1 and err[0]["stage"] == "tests" and err[0]["fingerprint"] == update("s1").fingerprint()
    b = pipe.propose(update("s1"))                    # the stage is fixed: the same candidate runs again
    assert b.promoted and b.version == 1, str(b)


def test_raise_stage_errors_lets_the_exception_through(tmp_path):
    class Broken:
        name = "tests"

        def run(self, c, x):
            raise KeyError("x")
    pipe = Pipeline([Broken()], str(tmp_path / "e.jsonl"), targets=[Target()], approval=A, raise_stage_errors=True)
    with pytest.raises(KeyError):
        pipe.propose(update("s1"))
    assert pipe.history("error")[0]["stage"] == "tests"          # journaled before it is raised


# ───────────────────────────────────────────── a rollback with a reason is a failure
def test_a_rollback_with_a_reason_counts_as_a_failure(tmp_path):
    pipe = Pipeline([KnownFailures()], str(tmp_path / "c.jsonl"), targets=[Target()], approval=A)
    a = pipe.propose(update("s1"))
    pipe.observe(a.change_id, {"gain": -1.0})
    assert pipe.rollback(0, why="observed loss") == [1]
    b = pipe.propose(update("s1"))
    assert b.status == "rejected" and b.stage == "failure_memory" and "observed loss" in b.reason, str(b)
    assert pipe.propose(update("s2")).promoted                     # another candidate is not touched


def test_a_rollback_without_a_reason_or_with_failed_false_is_not_a_failure(tmp_path):
    pipe = Pipeline([KnownFailures()], str(tmp_path / "c.jsonl"), targets=[Target()], approval=A)
    pipe.propose(update("s1"))
    pipe.rollback(0)
    assert pipe.propose(update("s1")).promoted
    pipe.rollback(0, why="moving to another branch", failed=False)
    assert pipe.propose(update("s1")).promoted
    assert KnownFailures(rollbacks=False).name == "failure_memory"
    pipe.rollback(0, why="observed loss")
    p2 = Pipeline([KnownFailures(rollbacks=False)], pipe.journal.storage, targets=[Target()], approval=A)
    assert p2.propose(update("s1")).promoted                       # rollbacks=False: only rejections count


# ───────────────────────────────────────────── the in-memory journal and the cached version state
def test_the_in_memory_journal_is_a_conforming_storage_and_none_or_memory_selects_it(tmp_path):
    from solvi.experimental.changes import MemoryStorage
    from solvi.testing.conformance import check_storage
    assert check_storage(MemoryStorage)["verify"]
    for where in (None, ":memory:"):
        pipe = Pipeline([KnownFailures()], where, targets=[Target()], approval=A)
        assert isinstance(pipe.journal.storage, MemoryStorage)
        assert pipe.propose(update("s1")).promoted and pipe.current == 1 and pipe.journal.verify()["ok"]
    assert not list(tmp_path.iterdir()) and not (tmp_path.parent / ":memory:").exists()


def test_proposals_do_not_reread_the_journal(tmp_path):
    from solvi.experimental.changes import MemoryStorage

    class Counting(MemoryStorage):
        reads = 0

        def _raw(self, snap=None):
            Counting.reads += 1
            return super()._raw(snap)
    pipe = Pipeline([KnownFailures()], Counting(), targets=[Target()], approval=A)
    for i in range(60):
        assert pipe.propose(update(f"x{i}")).promoted
    assert pipe.current == 60 and len(pipe.versions()) == 60
    assert Counting.reads <= 2, Counting.reads                      # read once, then kept up to date on write


def test_the_cache_follows_another_writer_on_the_same_file(tmp_path):
    path = str(tmp_path / "shared.jsonl")
    a = Pipeline([KnownFailures()], path, targets=[Target()], approval=A)
    b = Pipeline([KnownFailures()], path, targets=[Target()], approval=A)
    assert a.propose(update("s1")).version == 1 and a.current == 1
    assert b.current == 1 and b.propose(update("s2")).version == 2
    assert a.current == 2 and a.propose(update("s3")).version == 3
    a2 = Pipeline([KnownFailures(), Deny()], path, targets=[Target()], approval=A)
    assert a2.propose(update("s9")).stage == "deny"
    assert b.propose(update("s9")).stage == "failure_memory"       # b sees a2's rejection
    assert b.rollback(1) == [3, 2] and a.current == 1


# ───────────────────────────────────────────── EvaluatorLock.create
def test_evaluator_lock_create_takes_a_seed_list_and_refuses_a_bad_n_clearly(tmp_path):
    import json
    from solvi.experimental.changes import EvaluatorLock
    pin = EvaluatorLock.create(tmp_path / "a.json", [3, 1, 2])
    assert json.loads((tmp_path / "a.json").read_text())["seeds"] == [1, 2, 3] and len(pin) == 64
    EvaluatorLock.create(tmp_path / "b.json", seeds=range(10))
    assert json.loads((tmp_path / "b.json").read_text())["n"] == 10
    for bad, err in ((0, ValueError), (-3, ValueError), (2.5, TypeError), (True, TypeError), ("7", TypeError),
                     ([1, 1], ValueError), ([1, "x"], TypeError), ([], ValueError)):
        with pytest.raises(err, match="seeds|n is"):
            EvaluatorLock.create(tmp_path / "c.json", bad)
    assert not (tmp_path / "c.json").exists()
    with pytest.raises(TypeError, match="not both"):
        EvaluatorLock.create(tmp_path / "c.json", 5, seeds=[1])


# ───────────────────────────────────────────── the failure memory's identity: an extra body key is not a new candidate
def test_an_extra_key_in_a_rejected_rule_body_does_not_make_it_new(tmp_path):
    pipe = Pipeline([KnownFailures(), Deny()], None, store=KnowledgeStore(), approval={"rule": "gate"})
    b = hazard_body("lava", {"next": "lava"}, ["step"])
    rule = lambda body, **kw: Change("person", payload={"kind": "rule", "body": body, **kw})  # noqa: E731
    assert pipe.propose(rule(b)).stage == "deny"
    assert pipe.propose(rule(b)).stage == "failure_memory"
    r = pipe.propose(rule({**b, "nonce": 7}))
    assert r.stage == "failure_memory" and "same subject" in r.reason, str(r)
    assert pipe.propose(rule(hazard_body("lava", {"next": "lava"}, ["step"], why="again"))).stage == "failure_memory"
    # what the rule says is its identity: another condition, another scope, another kind is a new candidate
    assert pipe.propose(rule(hazard_body("lava", {"next": "pit"}, ["step"]))).stage == "deny"
    assert pipe.propose(rule(b, scope={"x": 1})).stage == "deny"
    assert pipe.propose(Change("person", payload={"kind": "fact", "body": b})).stage == "deny"


# ───────────────────────────────────────────── the failure memory's context and expiry; held changes expire
def test_known_failures_with_a_context_and_a_ttl(tmp_path):
    from solvi.experimental.changes import MemoryStorage
    now = [1000.0]
    clock = lambda: now[0]  # noqa: E731
    tests = {"version": 1}
    kf = KnownFailures(context=lambda ch: tests["version"], ttl=100, clock=clock)
    pipe = Pipeline([kf, Deny()], MemoryStorage(clock=clock), targets=[Target()], approval=A)
    assert pipe.propose(update("s1")).stage == "deny"
    now[0] += 50
    r = pipe.propose(update("s1"))
    assert r.stage == "failure_memory" and "50s ago" in r.reason, str(r)
    now[0] += 40                                    # a refusal by the memory does not renew the failure
    assert pipe.propose(update("s1")).stage == "failure_memory"
    now[0] += 20                                    # 110 s after the rejection: expired, the stages run again
    assert pipe.propose(update("s1")).stage == "deny"
    tests["version"] = 2                            # another context (the tests changed): not a known failure
    assert pipe.propose(update("s1")).stage == "deny"
    tests["version"] = 1
    assert pipe.propose(update("s1")).stage == "failure_memory"
    # a key of the caller's: two candidates with the same key are one
    kf2 = KnownFailures(key=lambda ch: ch.payload["fingerprint"][0])
    p2 = Pipeline([kf2, Deny()], None, targets=[Target()], approval=A)
    assert p2.propose(update("s1")).stage == "deny" and p2.propose(update("s2")).stage == "failure_memory"
    assert p2.propose(update("t1")).stage == "deny"
    with pytest.raises(ValueError):
        KnownFailures(ttl=0)


def test_held_changes_expire(tmp_path):
    from solvi.experimental.changes import MemoryStorage
    now = [0.0]
    clock = lambda: now[0]  # noqa: E731
    pipe = Pipeline([KnownFailures()], MemoryStorage(clock=clock), targets=[Target()],
                    approval={("update", "person"): "person"})
    a = pipe.propose(update("s1"))
    now[0] = 50
    b = pipe.propose(update("s2"))
    assert a.status == b.status == "held" and pipe.held() == sorted([a.change_id, b.change_id])
    now[0] = 120
    out = pipe.expire_held(100, now=now[0])
    assert [(r.change_id, r.status) for r in out] == [(a.change_id, "rejected")] and pipe.held() == [b.change_id]
    assert pipe.history("ruling")[-1]["by"] == "expiry"
    again = pipe.propose(update("s1"))       # an expired change is a rejected one (advisory under level "person")
    assert again.verdicts[0].stage == "failure_memory" and not again.verdicts[0].ok and "expiry" in again.verdicts[0].reason
    assert pipe.rule(again.change_id, False, by="ann").status == "rejected"
    out = pipe.expire_held(10, approve=True, by="ops", now=now[0])
    assert [r.status for r in out] == ["promoted"] and pipe.held() == []
