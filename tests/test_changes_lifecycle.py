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
