import pytest
from solvi.experimental.changes import (
    Change,
    ChangeJournal,
    Context,
    Economics,
    ProposerHistory,
    Pipeline,
    StageVerdict,
)


def change():
    return Change("person", payload={"kind": "fact", "body": {"s": "a", "r": "b", "o": 1}}, proposer="p")


def ctx(j):
    return Context(None, None, j, "candidate")


def test_economics_stops_before_expensive_callback_and_fingerprints_utility():
    calls = []

    class Expensive:
        name = "expensive"

        def run(self, c, x):
            calls.append(1)
            return StageVerdict.refuse("bad")

    e = Economics(lambda c, x: dict(gain=3, loss=1), costs={"expensive": 2}, units="USD")
    p = Pipeline([e, Expensive()], storage=None)
    assert p.propose(change()).status == "rejected" and calls == []
    a = e.run(change(), ctx(p.journal)).measured["policy_fingerprint"]
    e2 = Economics(lambda c, x: dict(gain=5, loss=1), costs={"expensive": 2}, units="USD")
    assert e2.run(change(), ctx(p.journal)).measured["policy_fingerprint"] != a
    assert Pipeline([e2, Expensive()], storage=None).propose(change()).status == "rejected" and calls == [1]


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_invalid_cost(value):
    with pytest.raises(ValueError):
        Economics(lambda c, x: dict(gain=1, loss=0), costs={"s": value}, units="u")


def test_labelled_history_restart_expiry_and_no_admission_as_truth(tmp_path):
    path = tmp_path / "history.jsonl"
    j = ChangeJournal(path)
    clock = [10]
    h = ProposerHistory(window=5, clock=lambda: clock[0])
    assert h.run(change(), ctx(j)).ok
    j.write("promoted", proposer="p")
    assert h.run(change(), ctx(j)).measured["labels"] == 0
    h.label(j, proposer="p", loss=1, observed_at=9, evidence="independent:test1", event_id="e1")
    assert not h.run(change(), ctx(ChangeJournal(path))).ok
    with pytest.raises(ValueError):
        h.label(j, proposer="p", loss=0, observed_at=11, evidence="x", event_id="future")
    with pytest.raises(ValueError):
        h.label(j, proposer="p", loss=0, observed_at=9, evidence="x", event_id="e1")
    clock[0] = 20
    assert h.run(change(), ctx(j)).ok  # expired bad evidence does not cause eternal cold-start refusal
    assert ProposerHistory(window=5, unknown="hold", clock=lambda: 20).run(change(), ctx(j)).decision == "hold"


def test_good_history_never_bypasses_fresh_bad_candidate():
    j = ChangeJournal(None)
    h = ProposerHistory(window=5, clock=lambda: 10)
    h.label(j, proposer="p", loss=0, observed_at=9, evidence="external", event_id="e")
    calls = []

    class Hard:
        name = "hard"

        def run(self, c, x):
            calls.append(1)
            return StageVerdict.refuse("fresh bad candidate")

    p = Pipeline([h, Hard()], storage=j.storage)
    assert p.propose(change()).status == "rejected" and calls == [1]


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_atomic_concurrent_labels_unique_across_backend_instances(tmp_path, backend):
    import concurrent.futures
    import threading
    from solvi.core.store import JSONLStorage, SQLiteStorage

    path = tmp_path / ("labels.jsonl" if backend == "jsonl" else "labels.db")
    cls = JSONLStorage if backend == "jsonl" else SQLiteStorage
    journals = [ChangeJournal(cls(path)) for _ in range(2)]
    h = ProposerHistory(window=5, min_labels=2, unknown="hold", clock=lambda: 10)
    barrier = threading.Barrier(2)

    def label(j):
        barrier.wait()
        try:
            h.label(j, proposer="p", loss=0, observed_at=9, evidence="one", event_id="one")
            return "accepted"
        except ValueError:
            return "duplicate"

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(label, journals))
    assert results.count("accepted") == 1
    reopened = ChangeJournal(cls(path))
    assert len(reopened.records("proposer_outcome")) == 1
    assert h.run(change(), ctx(reopened)).decision == "hold"
    before = len(reopened.records())
    with reopened.storage.batch():
        with pytest.raises(ValueError):
            h.label(reopened, proposer="p", loss=0, observed_at=9, evidence="two", event_id="two")
    assert len(reopened.records()) == before


def test_legacy_identical_duplicate_counted_once_conflict_fails_closed():
    j = ChangeJournal(None)
    h = ProposerHistory(window=5, min_labels=2, unknown="hold", clock=lambda: 10)
    row = dict(proposer="p", loss=0, observed_at=9, labelled_at=10, evidence="one", event_id="one")
    j.write("proposer_outcome", **row)
    j.write("proposer_outcome", **row)
    assert h.run(change(), ctx(j)).measured["labels"] == 1
    assert h.run(change(), ctx(j)).decision == "hold"
    j.write("proposer_outcome", **dict(row, loss=1))
    with pytest.raises(ValueError, match="conflicting"):
        h.run(change(), ctx(j))


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_event_uniqueness_across_processes(tmp_path, backend):
    import multiprocessing
    from solvi.core.store import JSONLStorage, SQLiteStorage

    mp = multiprocessing.get_context("fork")
    barrier = mp.Barrier(2)
    queue = mp.Queue()
    path = str(tmp_path / ("process.jsonl" if backend == "jsonl" else "process.db"))
    cls = JSONLStorage if backend == "jsonl" else SQLiteStorage
    cls(path)  # initialize schema/file before workers

    def worker():
        j = ChangeJournal(cls(path))
        h = ProposerHistory(window=5, clock=lambda: 10)
        barrier.wait(timeout=5)
        try:
            h.label(j, proposer="p", loss=0, observed_at=9, evidence="same", event_id="same")
            queue.put("accepted")
        except ValueError:
            queue.put("duplicate")

    processes = [mp.Process(target=worker) for _ in range(2)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert results.count("accepted") == 1
    assert len(ChangeJournal(cls(path)).records("proposer_outcome")) == 1
