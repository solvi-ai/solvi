"""Compact records written by solvi 1.0.0 (tests/fixtures/store_1_0_0: make.py ran with solvi 1.0.0 from PyPI) load,
verify and replay with this version as they did then: 1.0.0 did not keep the record a hard check's `then` function
writes, so a decision a `then` function answered is reported "not_kept" (not verified, never passed), the others replay;
the same decisions stored compact now keep that record and replay (1.0.1)."""
import shutil
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "fixtures" / "store_1_0_0"


@pytest.fixture
def task():
    sys.path.insert(0, str(DATA))
    try:
        import task
        yield task
    finally:
        sys.path.remove(str(DATA))
        sys.modules.pop("task", None)


@pytest.fixture
def copy(tmp_path):
    """The fixture files in a fresh folder: replay never writes, but a store opened for it may (keep the originals)."""
    for f in DATA.iterdir():
        if f.name.startswith("decisions."):
            shutil.copy(f, tmp_path / f.name)
    return tmp_path


@pytest.mark.parametrize("name", ["decisions.jsonl", "decisions.db"])
def test_compact_records_of_1_0_0_verify_and_replay_as_then(task, copy, name):
    from solvi.core.store import open_storage
    store = open_storage(str(copy / name))
    system = task.system()
    store.catalog = system
    v = store.verify()
    assert v["ok"] and v["count"] == 4 and not v["problems"]
    stored = list(store.iter())
    assert [s.compact for s in stored] == [True] * 4
    for s, st in zip(stored, task.STATES):                   # the stored answers are the ones given now
        new = system.ask(st, store=False)
        assert {q: [a.answer, a.status] for q, a in new.results.items()} == \
            {q: [a[0], a[2]] for q, a in s.data["answers"].items()}
        assert not any(r["kind"] == "then" for r in s.data["compact"]["trace"]["records"])   # 1.0.0 did not keep it
    bad = store.replay_all(system)
    assert [b["seq"] for b in bad] == [i for i, t in enumerate(task.THEN_FUNCTION) if t]
    for b in bad:                                            # honestly unchecked, not passed and not "damaged"
        assert b["kinds"] == {"not_kept": 1} and b["record"] == "compact"
        assert b["summary"].startswith("not verified")
        assert b["mismatches"][0][1] == "then:step"
    store.close()


def test_the_same_decisions_stored_compact_now_keep_the_then_record_and_replay(task, tmp_path):
    from solvi.core.store import JSONLStorage
    store = JSONLStorage(tmp_path / "now.jsonl", record="compact")
    system = task.system(store)
    for st in task.STATES:
        system.ask(st)
    kept = [[r["name"] for r in s.data["compact"]["trace"]["records"] if r["kind"] == "then"] for s in store.iter()]
    assert kept == [["then:step"] if t else [] for t in task.THEN_FUNCTION]
    assert store.verify()["ok"] and store.replay_all(system) == []
