"""Files written by solvi 0.8.0 (tests/fixtures/store_0_8_0: make.py ran with the v0.8.0 sources) load, verify and replay
with this version: the stores, the calibration file and the fingerprints — the names removed in 0.9 change no stored key,
hash or fingerprint. The one documented difference (CHANGELOG 0.9): the code fingerprints of a catalog no longer depend
on the Python version, so they differ once from the ones 0.8.0 computed; the stored decisions still replay."""
import json
import shutil
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "fixtures" / "store_0_8_0"


@pytest.fixture
def task():
    sys.path.insert(0, str(DATA))
    try:
        import task
        yield task
    finally:
        sys.path.remove(str(DATA))
        sys.modules.pop("task", None)


def test_a_calibration_file_written_by_0_8_0_loads_and_gives_the_same_part(task):
    fp = json.loads((DATA / "fingerprints.json").read_text())
    assert task.category_part().fingerprint() == fp["part"]
    part = task.category_part(DATA / "calibration.json")              # strict: it checks question, model and adaptation
    assert part.fingerprint() == fp["calibrated_part"]
    assert part.min_confidence == pytest.approx(json.loads((DATA / "calibration.json").read_text())["escalate_below"])


def test_the_fingerprints_of_0_8_0_hold_except_the_code_ones(task):
    fp = json.loads((DATA / "fingerprints.json").read_text())["system"]
    now = task.system(calibration=DATA / "calibration.json").fingerprint()
    assert now["models"] == fp["models"] and now["questions"] == fp["questions"]
    assert now["parts"]["answer:category"] == fp["parts"]["answer:category"]     # a decision part: no code hashed
    # the code fingerprints (a rule, a function, a check, and so the catalog) changed once in 0.9: see the module doc
    assert {k for k in fp["parts"] if now["parts"][k] != fp["parts"][k]} == {"answer:approve", "days_old", "within_limit"}
    assert now["catalog"] != fp["catalog"]


@pytest.mark.parametrize("name", ["decisions.jsonl", "decisions.db"])
def test_a_store_written_by_0_8_0_verifies_loads_and_replays(task, tmp_path, name):
    from solvi.cli import main
    from solvi.storage import open_storage
    for f in DATA.glob("decisions*"):
        shutil.copy(f, tmp_path / f.name)                    # replay never writes, but keep the fixtures untouched
    store = open_storage(str(tmp_path / name))
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
        assert old.trace.replay(system)["ok"]
    assert [store.get(s.id, system).results["category"].status for s in stored] == \
        ["abstain", "abstain", "ok", "abstain"]               # the calibrated threshold (0.995) answers one alone
    assert store.replay_all(system) == []
    assert [c["answer"] for c in store.corrections()] == ["software"]
    store.close()
    assert main(["replay", str(tmp_path / name), "--system", f"{DATA / 'task.py'}:calibrated"]) == 0
    # without the calibration file the part is another one, and replay says so (status 1)
    assert main(["replay", str(tmp_path / name), "--system", f"{DATA / 'task.py'}:system"]) == 1
