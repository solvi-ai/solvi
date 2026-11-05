"""Real SIGKILL at the outer curriculum/runner learning transaction edges."""

import json
import os
import subprocess
import sys
import time

import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience.competences import CompetenceRegistry
from solvi.experimental.experience.curriculum import AutoCurriculum
from solvi.experimental.experience.durable import SQLiteEnvironment
from solvi.experimental.experience.journal import ExperienceJournal
from solvi.experimental.experience.programs import Grammar


CASES = {str(i): {"family": "resource", "x": i + 6, "y": i % 3 + 1} for i in range(10)}

CHILD = r"""
import json,sys,time
from pathlib import Path
from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.competences import CompetenceRegistry
from solvi.experimental.experience.curriculum import AutoCurriculum
from solvi.experimental.experience.durable import SQLiteEnvironment
from solvi.experimental.experience.journal import ExperienceJournal
from solvi.experimental.experience.loop import ExperienceRunner
from solvi.experimental.experience.programs import Grammar

directory,stage=Path(sys.argv[1]),sys.argv[2]
cases={str(i):{"family":"resource","x":i+6,"y":i%3+1} for i in range(10)}
environment=SQLiteEnvironment(directory/"environment.db",cases)
store=KnowledgeStore(str(directory/"knowledge.db"))
journal=ExperienceJournal(store,"curriculum-process",{})
registry=CompetenceRegistry(Grammar(("x","y"),operators=("sub",),max_candidates=100),store=store)
runner=ExperienceRunner(environment,journal,WorkBudget(100000),registry,mode="fixed")
controller=AutoCurriculum(runner,list(cases),max_attempts=1)

def halt():
    (directory/"marker").write_text(json.dumps({"stage":stage,"spent":runner.budget.spent}))
    while True:
        time.sleep(.05)

append=journal.append
def record(kind,payload):
    result=append(kind,payload)
    if (stage=="controller_intent" and kind=="curriculum"
            and payload["controller_state"]["learning"] is not None):
        halt()
    return result
journal.append=record
learn=runner.learn
def learn_then_halt(*args,**kwargs):
    result=learn(*args,**kwargs)
    if stage=="learn_completed":
        assert result is not None and result.accepted
        halt()
    return result
runner.learn=learn_then_halt
controller.run()
raise AssertionError("fault boundary was not reached")
"""


@pytest.mark.parametrize("stage", ["controller_intent", "learn_completed"])
@pytest.mark.skipif(os.name != "posix", reason="SIGKILL injection requires POSIX")
def test_curriculum_sigkill_learning_edges_resume_once(tmp_path, stage):
    log = tmp_path / "child.log"
    with log.open("w") as stream:
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD, str(tmp_path), stage],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        try:
            # This guards a fault boundary, not a runtime target. Durable fsyncs
            # can exceed 30 seconds on a busy shared host; never skip the marker.
            deadline = time.monotonic() + 120
            while not (tmp_path / "marker").exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (tmp_path / "marker").exists(), log.read_text()
            child.kill()  # Kill only the subprocess created by this test.
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    assert child.returncode == -9
    stopped = json.loads((tmp_path / "marker").read_text())

    # All host objects below are new instances; nothing is recovered from the
    # killed process's Python objects or from a pre-kill controller snapshot.
    environment = SQLiteEnvironment(tmp_path / "environment.db", CASES)
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    journal = ExperienceJournal(store, "curriculum-process", {})
    records_before = journal.records()
    controller_state = [e["payload"]["controller_state"] for e in records_before if e["kind"] == "curriculum"][-1]
    assert controller_state["learning"] is not None
    intents_before = [e for e in records_before if e["kind"] == "learning_intent"]
    assert len(intents_before) == (1 if stage == "learn_completed" else 0)
    registry = CompetenceRegistry(Grammar(("x", "y"), operators=("sub",), max_candidates=100), store=store)
    restored = AutoCurriculum.restore(
        {"controller": controller_state, "runner": journal.latest()},
        environment=environment,
        journal=journal,
        registry=registry,
    )
    assert restored.runner.budget.spent >= stopped["spent"]
    results = restored.run()
    assert restored.state["status"] == "done"
    assert len(results) == len(CASES) and all(result["ok"] for result in results)
    assert all(state["effects"] == 1 for state in environment.state.values())
    assert len(restored.state["seen"]) == len(CASES)
    assert restored.runner.budget.spent >= stopped["spent"]
    assert restored.state["attempts"] == 1

    records = journal.records()
    intents = [e for e in records if e["kind"] == "learning_intent"]
    assert len(intents) == 1
    learning = intents[0]["payload"]["state"]["learning"]
    proposal = {e["task_id"] for e in learning["proposal"]}
    admission = {e["task_id"] for e in learning["admission"]}
    assert proposal == {"0", "1", "2", "3"}
    assert admission == {"4", "5", "6", "7"}
    assert proposal.isdisjoint(admission)
    assignments = restored.state["assignments"]
    assert [assignments[str(i)]["role"] for i in range(10)] == ["proposal"] * 4 + ["admission"] * 4 + ["evaluation"] * 2
    assert all(assignments[str(i)]["used_as"] == "evaluation" for i in (8, 9))
    assert all(result["competence_id"] is not None for result in results[8:])
    # No later controller write may turn an evaluation assignment into training.
    for event in records:
        if event["kind"] == "curriculum":
            for task_id in ("8", "9"):
                assignment = event["payload"]["controller_state"]["assignments"].get(task_id)
                if assignment is not None:
                    assert assignment["role"] == "evaluation"
                    assert assignment.get("used_as", "evaluation") == "evaluation"
    ledgers = [
        e["payload"]["state"]["ledger"]["spent"]
        for e in records
        if isinstance(e["payload"].get("state"), dict) and "ledger" in e["payload"]["state"]
    ]
    assert ledgers == sorted(ledgers)
    assert journal.verify()
