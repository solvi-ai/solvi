"""Ten explicit before/after append faults across checked-learning recovery."""

import json
import hashlib
from pathlib import Path

import pytest

from test_experience_checked_process import CASES, fresh, read_audit
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.curriculum import AutoCurriculum
from solvi.experimental.experience.loop import ExperienceRunner


BOUNDARIES = (
    "learning_reserved",
    "domain_read_intent",
    "domain_read_receipt",
    "learning_committed",
    "controller_save_after_learn",
)


@pytest.fixture(scope="module", autouse=True)
def require_local_runtime(tmp_path_factory):
    from solvi.experimental.experience import curriculum, domain_checks, loop, programs

    source = Path(__file__).resolve().parents[1] / "src/solvi/experimental/experience"
    provenance = {}
    for module in (curriculum, domain_checks, loop, programs):
        path = Path(module.__file__).resolve()
        assert path == source / path.name, "fault matrix imported a different worktree"
        provenance[path.name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path_factory.getbasetemp() / "RUNTIME.json").write_text(json.dumps(provenance, indent=2))


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("side", ("before", "after"))
def test_checked_durable_append_before_after_matrix(tmp_path, boundary, side):
    environment, store, journal, registry, contracts = fresh(tmp_path, create=True)
    runner = ExperienceRunner(
        environment,
        journal,
        WorkBudget(100000),
        registry,
        mode="fixed",
        checked_domains=contracts,
        checked_store_policy="evidence_only",
    )
    controller = AutoCurriculum(runner, list(CASES), max_attempts=1)
    append = journal.append
    learn = runner.learn
    reached = []
    completed_learn = []

    def learned(*args, **kwargs):
        result = learn(*args, **kwargs)
        assert result is not None and result.accepted
        completed_learn.append(True)
        return result

    runner.learn = learned

    def fault(kind, payload):
        target = kind == boundary
        if boundary == "controller_save_after_learn":
            target = kind == "curriculum" and completed_learn and payload["controller_state"]["learning"] is None
        if target and not reached:
            # The stop marker belongs to the evaluator, never actor state.
            marker = dict(
                boundary=boundary,
                side=side,
                kind=kind,
                spent=runner.budget.spent,
                counts=runner.budget.counts,
                oracle_reads=len(read_audit(tmp_path)),
            )
            reached.append(marker)
            if side == "after":
                append(kind, payload)
            (tmp_path / "injected-fault.json").write_text(json.dumps(marker, sort_keys=True))
            raise RuntimeError("injected append boundary " + boundary + ":" + side)
        return append(kind, payload)

    journal.append = fault
    with pytest.raises(RuntimeError, match="injected append boundary"):
        controller.run()
    assert len(reached) == 1
    stopped = reached[0]
    before_reads = read_audit(tmp_path)
    if boundary in ("learning_reserved", "domain_read_intent"):
        assert len(before_reads) == 0
    elif boundary == "domain_read_receipt":
        assert len(before_reads) == 1
    else:
        assert len(before_reads) == 10

    # Drop every actor object and restore from actual persistent SQLite state.
    environment, store, journal, registry, contracts = fresh(tmp_path)
    records = journal.records()
    controller_states = [e["payload"]["controller_state"] for e in records if e["kind"] == "curriculum"]
    assert controller_states
    latest = journal.latest()
    restored = AutoCurriculum.restore(
        {"controller": controller_states[-1], "runner": latest},
        environment=environment,
        journal=journal,
        registry=registry,
        checked_domains=contracts,
        checked_store_policy="evidence_only",
    )
    assert restored.runner.budget.spent >= stopped["spent"]
    for category, count in stopped["counts"].items():
        assert restored.runner.budget.counts.get(category, 0) >= count
    results = restored.run(max_steps=1000)
    assert restored.state["status"] == "done"
    assert len(results) == 10 and all(row["ok"] for row in results)
    assert len(restored.state["seen"]) == 10
    assert all(state["effects"] == 1 for state in environment.state.values())
    assert restored.state["attempts"] == 1 and restored.state["learning"] is None
    assert restored.runner.learning is None
    assert read_audit(tmp_path) == before_reads
    assert sum(event["kind"] == "learning_intent" for event in journal.records()) == 1

    committed = boundary == "controller_save_after_learn" or (boundary == "learning_committed" and side == "after")
    packets = list(restored.state["packets"].values())
    competences = restored.runner.registry.snapshot()["competences"]
    if committed:
        assert packets and all(packet["transfer"] for packet in packets)
        assert competences and all(record["status"] == "active" for record in competences.values())
        assert all(row["competence_id"] is not None for row in results[8:])
    else:
        assert not competences
        assert all(
            not packet["transfer"] and not packet["proposal"] and not packet["admission"] and packet["generation"] == 1
            for packet in packets
        )
        assert all(row["competence_id"] is None for row in results[8:])

    ledger_spent = []
    for event in journal.records():
        payload = event["payload"]
        if event["kind"] == "debit":
            ledger_spent.append(payload["ledger"]["spent"])
        elif "state" in payload:
            ledger_spent.append(payload["state"]["ledger"]["spent"])
    assert ledger_spent == sorted(ledger_spent)
    assert restored.runner.budget.spent >= stopped["spent"]
    assert journal.verify()
    # A second complete reopen is stable and cannot repeat effects/oracle reads.
    again_env, _, again_journal, again_registry, again_contracts = fresh(tmp_path)
    state = [e["payload"]["controller_state"] for e in again_journal.records() if e["kind"] == "curriculum"][-1]
    again = AutoCurriculum.restore(
        {"controller": state, "runner": again_journal.latest()},
        environment=again_env,
        journal=again_journal,
        registry=again_registry,
        checked_domains=again_contracts,
        checked_store_policy="evidence_only",
    )
    assert again.run() == results
    assert read_audit(tmp_path) == before_reads
    assert all(state["effects"] == 1 for state in again_env.state.values())
