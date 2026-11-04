import concurrent.futures
from types import SimpleNamespace

import pytest

from solvi.core.store import JSONLStorage, SQLiteStorage
from solvi.experimental.changes import FreshEnvironments, Change, Pipeline
from solvi.experimental.changes._journal import ChangeJournal


def world_id(seed):
    return f"world:{seed}"


def candidate_id(change, ctx):
    return change


def stage(ids=("a", "b"), select=(1, 2), confirm=(3, 4), calls=None):
    def agent(change, ctx):
        if calls is not None:
            calls.append(change)
        return change is not None

    s = FreshEnvironments(
        lambda: None,
        agent,
        select=select,
        confirm=confirm,
        boot=20,
        world_registry="env-v1",
        candidate_ids=list(ids),
        candidate_id=candidate_id,
        world_id=world_id,
    )
    s._play = lambda policy, seeds: [
        dict(seed=x, failed=not policy, errors=0, actions=["safe"], progress=1) for x in seeds
    ]
    return s


def context(store):
    return SimpleNamespace(journal=ChangeJournal(store), change_id="change")


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_reservation_persists_reopens_and_refuses_reuse_before_agent(tmp_path, backend):
    path = tmp_path / ("r.jsonl" if backend == "jsonl" else "r.db")
    factory = lambda: JSONLStorage(path, fsync=True) if backend == "jsonl" else SQLiteStorage(path)
    ctx = context(factory())
    calls = []
    s = stage(calls=calls)
    verdict = s.run("a", ctx)
    assert verdict.ok and calls
    measurement = verdict.measured["world_registry"]
    assert measurement["allocated_alpha"] == 0.025 and measurement["interval_level"] == 0.975
    calls.clear()
    reopened = context(factory())
    assert not s.run("b", reopened).ok and calls == []
    assert not stage(select=(5, 6), confirm=(7, 8), calls=calls).run("a", reopened).ok and calls == []
    assert len(reopened.journal.records("fresh_world_reservation")) == 1
    assert reopened.journal.verify()["ok"]


def test_claim_is_atomic_between_two_journals_and_crash_burns_worlds(tmp_path):
    path = tmp_path / "atomic.jsonl"

    def reserve(cid):
        s = stage()
        return s._reserve_worlds(cid, context(JSONLStorage(path, fsync=True)), [1, 2], [3, 4])

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        out = list(pool.map(reserve, ["a", "b"]))
    assert sum(record is not None for record, reason in out) == 1
    assert sum(reason is not None for record, reason in out) == 1
    # No evaluation followed the successful reservation; restart still refuses reuse.
    assert reserve("b")[1] is not None


def test_pipeline_refuses_without_play_and_registry_config_is_frozen(tmp_path):
    calls = []
    s = stage(calls=calls)
    storage = JSONLStorage(tmp_path / "pipe.jsonl", fsync=True)
    # Reserve the worlds independently first, then the real Pipeline must refuse before agent.
    assert s._reserve_worlds("a", context(storage), [1, 2], [3, 4])[0]
    s.candidate_id = lambda change, ctx: "b"
    pipe = Pipeline([s], storage)
    r = pipe.propose(Change("person", payload={"kind": "fact", "body": {"s": "a", "r": "b", "o": 1}}))
    assert r.status == "rejected" and calls == []


def test_nondurable_and_unknown_candidates_refuse_before_eval(tmp_path):
    calls = []
    s = stage(calls=calls)
    with pytest.raises(ValueError, match="fsync"):
        s.run("a", context(JSONLStorage(tmp_path / "unsafe.jsonl")))
    assert not s.run("unknown", context(JSONLStorage(tmp_path / "safe.jsonl", fsync=True))).ok
    assert calls == []


def test_jsonl_atomic_recheck_between_processes(tmp_path):
    import multiprocessing

    mp = multiprocessing.get_context("fork")
    path = tmp_path / "processes.jsonl"
    start = mp.Event()
    queue = mp.Queue()

    def worker(cid):
        start.wait(3)
        record, reason = stage()._reserve_worlds(cid, context(JSONLStorage(path, fsync=True)), [1, 2], [3, 4])
        queue.put((record is not None, reason))

    children = [mp.Process(target=worker, args=(cid,)) for cid in ["a", "b"]]
    try:
        for child in children:
            child.start()
        start.set()
        results = [queue.get(timeout=5) for _ in children]
        for child in children:
            child.join(timeout=5)
        assert all(child.exitcode == 0 for child in children)
        assert sum(ok for ok, reason in results) == 1
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()


def test_nominal_family_level_is_passed_to_all_intervals(tmp_path, monkeypatch):
    from solvi.experimental.changes import _paired

    original = _paired.paired_bootstrap
    levels = []

    def track(*args, **kw):
        levels.append(kw.get("level"))
        return original(*args, **kw)

    monkeypatch.setattr(_paired, "paired_bootstrap", track)
    assert stage().run("a", context(JSONLStorage(tmp_path / "levels.jsonl", fsync=True))).ok
    assert levels == [0.975, 0.975, 0.975]


@pytest.mark.parametrize("alpha", [1e-20, 5e-324])
def test_unrepresentable_family_allocation_is_rejected_at_construction(alpha):
    calls = []

    def never(*args):
        calls.append(args)
        raise AssertionError("callback must not run")

    with pytest.raises(ValueError, match="representable"):
        FreshEnvironments(
            never,
            never,
            select=never,
            confirm=never,
            world_registry="alpha",
            candidate_ids=["a", "b"],
            candidate_id=never,
            world_id=never,
            alpha=alpha,
        )
    assert calls == []


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_outer_batch_refuses_before_all_callbacks_and_cannot_rollback_evaluation(tmp_path, backend):
    path = tmp_path / ("outer.jsonl" if backend == "jsonl" else "outer.db")
    storage = JSONLStorage(path, fsync=True) if backend == "jsonl" else SQLiteStorage(path)
    s = stage()
    calls = []

    def never(*args):
        calls.append(args)
        raise AssertionError("callback must not run")

    s.agent = s.select = s.confirm = s.candidate_id = s.world_id = never
    with pytest.raises(RuntimeError, match="outer rollback"):
        with storage.batch():
            with pytest.raises(ValueError, match="active storage batch"):
                s.run("a", context(storage))
            assert calls == []
            raise RuntimeError("outer rollback")
    reopened = JSONLStorage(path, fsync=True) if backend == "jsonl" else SQLiteStorage(path)
    assert context(reopened).journal.records("fresh_world_reservation") == []
    # No evaluation happened in the rejected outer transaction; a normal durable attempt is allowed.
    assert stage().run("a", context(reopened)).ok


@pytest.mark.parametrize("path", [":memory:", ""])
def test_ephemeral_sqlite_cannot_evaluate_or_reopen_as_fresh_registry(path):
    storage = SQLiteStorage(path)
    calls = []
    s = stage()

    def never(*args):
        calls.append(args)
        raise AssertionError("callback must not run")

    s.agent = s.select = s.confirm = s.candidate_id = s.world_id = never
    try:
        with pytest.raises(ValueError, match="persistent file"):
            s.run("a", context(storage))
        assert calls == [] and context(storage).journal.records("fresh_world_reservation") == []
    finally:
        storage.close()
    reopened = SQLiteStorage(path)
    try:
        with pytest.raises(ValueError, match="persistent file"):
            s.run("a", context(reopened))
        assert calls == []
    finally:
        reopened.close()


@pytest.mark.parametrize("pragma", ["synchronous=OFF", "synchronous=NORMAL", "journal_mode=OFF", "journal_mode=MEMORY"])
def test_explicitly_disabled_sqlite_durability_refuses_before_callbacks(tmp_path, pragma):
    storage = SQLiteStorage(tmp_path / "disabled.db")
    storage.db.execute("PRAGMA " + pragma)
    calls = []
    s = stage(calls=calls)

    def never(*args):
        calls.append(args)
        raise AssertionError("callback must not run")

    s.select = s.confirm = s.candidate_id = s.world_id = never
    try:
        with pytest.raises(ValueError, match="durable journaling"):
            s.run("a", context(storage))
        assert calls == [] and context(storage).journal.records("fresh_world_reservation") == []
    finally:
        storage.close()
