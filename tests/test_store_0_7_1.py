"""Stores written by solvi 0.7.1 (tests/fixtures/store_0_7_1: make.py ran with the v0.7.1 sources) load, verify and replay
with this version — the renames of 0.8 change no stored key, hash or fingerprint."""
import shutil
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "fixtures" / "store_0_7_1"


@pytest.fixture
def task(tmp_path):
    sys.path.insert(0, str(DATA))
    try:
        import task
        yield task
    finally:
        sys.path.remove(str(DATA))
        sys.modules.pop("task", None)


@pytest.mark.parametrize("name", ["decisions.jsonl", "decisions.db"])
def test_a_store_written_by_0_7_1_verifies_loads_and_replays(task, tmp_path, name):
    from solvi.cli import main
    from solvi.storage import open_storage
    for f in DATA.glob("decisions*"):
        shutil.copy(f, tmp_path / f.name)                    # replay never writes, but keep the fixtures untouched
    store = open_storage(str(tmp_path / name))
    system = task.system()
    store.catalog = system
    v = store.verify()
    assert v["ok"] and v["count"] == 5 and not v["problems"]
    stored = list(store.iter())
    assert len(stored) == 4
    for s, st in zip(stored, task.STATES):
        old = store.get(s.id, system)
        new = system.ask(st, store=False)
        assert {q: (r.answer, r.status) for q, r in old.results.items()} == \
            {q: (r.answer, r.status) for q, r in new.results.items()}
        assert old.trace.replay(system)["ok"]
    assert store.replay_all(system) == []
    assert [c["answer"] for c in store.corrections()] == ["software"]
    assert main(["replay", str(tmp_path / name), "--system", f"{DATA / 'task.py'}:system"]) == 0
