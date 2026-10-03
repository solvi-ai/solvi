"""Files written by solvi 0.9.0 (tests/fixtures/store_0_9_0: make.py ran with the v0.9.0 sources) load, verify and replay
with this version, and every fingerprint 0.9.0 computed is the one computed now — the code ones included: the 1.0 layout
moves modules, and a stored decision must still name the system that made it (solvi._deprecate.MOVED keeps the 0.9
module names in the fingerprints; tests/test_golden_fingerprints.py pins many more)."""
import json
import shutil
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "fixtures" / "store_0_9_0"


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
        if f.name.startswith(("decisions.", "dispatch.")):
            shutil.copy(f, tmp_path / f.name)
    return tmp_path


def test_every_fingerprint_of_0_9_0_holds(task):
    fp = json.loads((DATA / "fingerprints.json").read_text())
    assert task.category_part().fingerprint() == fp["part"]
    part = task.category_part(DATA / "calibration.json")              # strict: it checks question, model and adaptation
    assert part.fingerprint() == fp["calibrated_part"]
    assert part.min_confidence == pytest.approx(json.loads((DATA / "calibration.json").read_text())["escalate_below"])
    assert task.system(calibration=DATA / "calibration.json").fingerprint() == fp["system"]
    assert task.dispatch().system.fingerprint() == fp["dispatch_system"]


@pytest.mark.parametrize("name", ["decisions.jsonl", "decisions.db"])
def test_a_store_written_by_0_9_0_verifies_loads_and_replays(task, copy, name):
    from solvi.cli import main
    from solvi.storage import open_storage
    store = open_storage(str(copy / name))
    system = task.system(calibration=DATA / "calibration.json")
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
        assert old.trace.fingerprint == new.trace.fingerprint            # the same system, as far as a replay can tell
        assert old.trace.replay(system)["ok"]
    old = store.get(stored[2].id, system)
    assert old.results["receipt_total"].answer == 40.0 and old.results["channels"].status == "ok"
    assert store.replay_all(system) == []
    assert [c["answer"] for c in store.corrections()] == ["software"]
    store.close()
    assert main(["replay", str(copy / name), "--system", f"{DATA / 'task.py'}:calibrated"]) == 0
    # without the calibration file the part is another one, and replay says so (status 1)
    assert main(["replay", str(copy / name), "--system", f"{DATA / 'task.py'}:system"]) == 1


def test_the_dispatch_store_of_solvi_build_written_by_0_9_0_verifies_and_replays(task, copy):
    from solvi.storage import open_storage
    store = open_storage(str(copy / "dispatch.jsonl"))
    assert store.verify()["ok"]
    store.close()
    auto = task.dispatch(copy / "dispatch.jsonl")
    stored = list(auto.dispatcher.stored())
    assert len(stored) == 30 and {d.by for d in stored} == {"s1", "s2"}
    assert auto.replay_all() == []
    again = [auto.dispatcher.ask(st) for st in task.ASKED]                 # the same answers by the same paths
    assert [(d.answer, d.by) for d in again] == [(d.answer, d.by) for d in stored]
    auto.storage.close()


def test_the_stores_replay_after_every_module_moved_with_the_table(task, copy):
    """Every solvi class and function moved to another module, MOVED mapping each back: what 0.9.0 stored replays."""
    from test_golden_fingerprints import _relocated
    with _relocated(table=True):
        from solvi.storage import open_storage
        system = task.system(calibration=DATA / "calibration.json")
        store = open_storage(str(copy / "decisions.jsonl"))
        assert store.verify()["ok"] and store.replay_all(system) == []
        for s, st in zip(store.iter(), task.STATES):                   # and the stored ones name this very system
            old = store.get(s.id, system).trace
            assert old.fingerprint == system.ask(st, store=False).trace.fingerprint
            assert old.replay(system)["catalog"] == "same"
        store.close()
        auto = task.dispatch(copy / "dispatch.jsonl")
        assert auto.replay_all() == []
        auto.storage.close()
