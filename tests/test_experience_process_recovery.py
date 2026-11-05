"""Actual process death across independent SQLite effect/journal boundaries."""

import os
import subprocess
import sys
import time

import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience import ExperienceJournal, ExperienceRunner, WorkBudget
from solvi.experimental.experience.durable import SQLiteEnvironment

CASES = {"one": {"family": "resource", "x": 8, "y": 2}, "two": {"family": "workflow", "x": 3, "y": 5}}

CHILD = r"""
from pathlib import Path
import sys,time
from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience import ExperienceJournal,ExperienceRunner,WorkBudget
from solvi.experimental.experience.durable import SQLiteEnvironment

directory,stage=Path(sys.argv[1]),sys.argv[2]
cases={"one":{"family":"resource","x":8,"y":2},"two":{"family":"workflow","x":3,"y":5}}
environment=SQLiteEnvironment(directory/"environment.db",cases)
store=KnowledgeStore(str(directory/"knowledge.db"))
journal=ExperienceJournal(store,"process-crash",{})

def halt():
    (directory/"marker").write_text(stage)
    while True:
        time.sleep(.05)

execute=environment.execute
def effect(task,kind,payload,attempt):
    if task=="one" and kind=="act" and stage=="before_effect":
        halt()
    result=execute(task,kind,payload,attempt)
    if task=="one" and kind=="act" and stage=="after_effect":
        halt()
    return result
environment.execute=effect
append=journal.append
def record(kind,payload):
    result=append(kind,payload)
    if (stage=="after_receipt" and kind=="receipt" and
            payload["state"]["pending"]["kind"]=="act"):
        halt()
    return result
journal.append=record
add=store.add
def add_fact(kind,body,*args,**kwargs):
    result=add(kind,body,*args,**kwargs)
    if stage=="after_fact" and kind=="fact":
        halt()
    return result
store.add=add_fact
runner=ExperienceRunner(environment,journal,WorkBudget(10000),mode="fixed")
runner.run(["one","two"])
"""


@pytest.mark.parametrize("stage", ["before_effect", "after_effect", "after_receipt", "after_fact"])
@pytest.mark.skipif(os.name != "posix", reason="SIGKILL fault injection requires POSIX")
def test_sigkill_resume_preserves_effects_queue_facts_and_budget(tmp_path, stage):
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    log = crashed / "child.log"
    with log.open("w") as stream:
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD, str(crashed), stage],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        try:
            deadline = time.monotonic() + 15
            while not (crashed / "marker").exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (crashed / "marker").exists(), log.read_text()
            child.kill()  # Only the child this test created; actual SIGKILL.
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    assert child.returncode == -9
    environment = SQLiteEnvironment(crashed / "environment.db", CASES)
    store = KnowledgeStore(str(crashed / "knowledge.db"))
    journal = ExperienceJournal(store, "process-crash", {})
    restored = ExperienceRunner.restore(journal.latest(), environment=environment, journal=journal)
    results = restored.run()

    uninterrupted = tmp_path / "uninterrupted"
    uninterrupted.mkdir()
    reference_env = SQLiteEnvironment(uninterrupted / "environment.db", CASES)
    reference_store = KnowledgeStore(str(uninterrupted / "knowledge.db"))
    reference = ExperienceRunner(
        reference_env, ExperienceJournal(reference_store, "process-crash", {}), WorkBudget(10000), mode="fixed"
    )
    expected = reference.run(["one", "two"])
    assert results == expected and all(r["ok"] for r in results)
    assert environment.state == reference_env.state
    assert all(v["effects"] == 1 for v in environment.state.values())
    assert restored.budget.snapshot() == reference.budget.snapshot()
    assert restored.events == reference.events and restored.cursor == 2
    facts = [item for item in store.items.values() if item.kind == "fact"]
    assert len(facts) == 2 and all(len(item.asserts) == 1 for item in facts)
    assert journal.verify()
