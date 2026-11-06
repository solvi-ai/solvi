"""Verified finite computations: paid effects, revisions and durable recovery."""
import copy

import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.computation import (
    ComputationPlan, ComputeNode, ComputeOperator, VerifiedComputation,
)
from solvi.experimental.experience.journal import ExperienceJournal


def add(name="add", revision="1", execute=None, verify=None):
    return ComputeOperator(name, revision, execute or (lambda xs: sum(xs)),
                           verify or (lambda xs, value: value == sum(xs)), 2, 3)


def plan():
    return ComputationPlan(("x", "y"), (
        ComputeNode("a", ("x",), ("add",)),
        ComputeNode("b", ("y",), ("add",)),
        ComputeNode("out", ("a", "b"), ("add",)),
    ), ("out",))


@pytest.mark.parametrize("policy", ["fifo", "cheap", "critical"])
def test_verified_outputs_and_equal_accounting(policy):
    budget = WorkBudget(100)
    session = VerifiedComputation(plan(), [add()], budget, inputs={"x": 2, "y": 3})
    assert session.run(policy).status == "complete"
    assert dict(session.outputs) == {"out": 5}
    assert budget.spent == 16 + 3 * 15 + 3 * 7
    with pytest.raises(TypeError):
        session.values["out"] = 100
    state = session.snapshot()
    state["cache"]["out"]["value"] = 99
    assert session.outputs["out"] == 5


def test_budget_reserves_verification_before_execution():
    calls = []
    op = add(execute=lambda xs: calls.append(xs) or sum(xs))
    p = ComputationPlan(("x",), (ComputeNode("out", ("x",), ("add",)),), ("out",))
    session = VerifiedComputation(p, [op], WorkBudget(10), inputs={"x": 1})
    assert session.step().status == "budget"  # setup2 + controller2 leaves6, needs7
    assert not calls and not session.outputs


def test_rejected_operator_tries_distinct_alternative_and_never_leaks():
    bad = add("bad", execute=lambda xs: 999)
    p = ComputationPlan(("x",), (ComputeNode("out", ("x",), ("bad", "add")),), ("out",))
    session = VerifiedComputation(p, [bad, add()], WorkBudget(100), inputs={"x": 4})
    report = session.step()
    assert report.status == "rejected" and report.value is None
    assert not session.values
    assert session.run().status == "complete"
    assert session.outputs["out"] == 4
    only_bad = ComputationPlan(("x",), (ComputeNode("out", ("x",), ("bad",)),), ("out",))
    failed = VerifiedComputation(only_bad, [bad], WorkBudget(100), inputs={"x": 2})
    assert failed.run().status == "rejected"
    assert failed.budget.spent < 100


@pytest.mark.parametrize("op", [add(execute=lambda xs: 1.0), add(verify=lambda xs, value: 1)])
def test_strict_callback_types_are_rejected(op):
    session = VerifiedComputation(plan(), [op], WorkBudget(100), inputs={"x": 2, "y": 3})
    assert session.step().status == "rejected"
    assert not session.values


def test_selective_cache_reuse_and_revision_updates():
    calls = []
    op = add(execute=lambda xs: calls.append(xs) or sum(xs))
    session = VerifiedComputation(plan(), [op], WorkBudget(500), inputs={"x": 2, "y": 3})
    session.run()
    session.update_inputs({"x": 9}, "v2")
    assert dict(session.values) == {"b": 3}
    session.run()
    assert calls.count((3,)) == 1 and session.outputs["out"] == 12
    with pytest.raises(ValueError, match="revision"):
        session.update_operators([add(execute=lambda xs: sum(xs))])
    session.update_operators([add(revision="2")])
    assert not session.values
    session.run()
    assert session.outputs["out"] == 12


def test_missing_input_and_observation_pending(tmp_path):
    journal = ExperienceJournal(KnowledgeStore(str(tmp_path / "state.db")), "compute", {})
    session = VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal)
    assert session.step().missing == ("x", "y")
    seen = []
    def reader():
        seen.append(journal.latest()["state"]["pending"]["kind"])
        return 5
    assert session.observe("x", reader, 4, "sensor1") == 5
    assert seen == ["observation"] and not session.outputs
    session.observe("y", lambda: 6, 4, "sensor1")
    session.run()
    assert session.outputs["out"] == 11


def test_cold_journal_reopen_restores_ledger_and_configuration(tmp_path):
    path = str(tmp_path / "state.db")
    journal = ExperienceJournal(KnowledgeStore(path), "compute", {})
    session = VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal, inputs={"x": 2, "y": 3})
    session.step()
    spent = session.budget.spent
    reopened = ExperienceJournal(KnowledgeStore(path), "compute", {})
    restored = VerifiedComputation(plan(), [add()], WorkBudget(300), journal=reopened)
    assert restored.budget.spent == spent + 16 + 5 and dict(restored.values) == {"a": 2}
    restored.run()
    assert restored.outputs["out"] == 5
    with pytest.raises(ValueError, match="configuration"):
        VerifiedComputation(plan(), [add(revision="changed")], WorkBudget(300), journal=reopened)


def test_callback_exception_is_durable_and_unknown_pending_never_reexecutes(tmp_path):
    path = str(tmp_path / "pending.db")
    journal = ExperienceJournal(KnowledgeStore(path), "compute", {})
    calls = []
    def broken(xs):
        calls.append(xs)
        assert journal.latest()["state"]["pending"]["kind"] == "compute"
        raise KeyboardInterrupt("crash")
    session = VerifiedComputation(plan(), [add(execute=broken)], WorkBudget(300), journal=journal, inputs={"x": 2, "y": 3})
    with pytest.raises(KeyboardInterrupt, match="crash"):
        session.step()
    reopened = ExperienceJournal(KnowledgeStore(path), "compute", {})
    with pytest.raises(RuntimeError, match="pending"):
        VerifiedComputation(plan(), [add(execute=broken)], WorkBudget(300), journal=reopened)
    assert calls == [(2,)]


def test_sink_failure_stops_before_callbacks():
    calls = []
    budget = WorkBudget(100)
    session = VerifiedComputation(plan(), [add(execute=lambda xs: calls.append(xs) or sum(xs))], budget, inputs={"x": 2})
    def fail(snapshot):
        raise OSError("sink failed")
    budget.on_charge = fail
    with pytest.raises(OSError):
        session.step()
    with pytest.raises(RuntimeError):
        session.step()
    assert not calls


def test_tampered_journal_fails_before_callback():
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "compute", {})
    session = VerifiedComputation(plan(), [add()], WorkBudget(100), journal=journal, inputs={"x": 2, "y": 3})
    store.journal[-1]["experience"]["payload"]["state"]["inputs"]["x"]["value"] = 999
    with pytest.raises(ValueError):
        session.step()
    with pytest.raises(RuntimeError):
        _ = session.values


def test_irrelevant_nodes_are_skipped_for_every_policy():
    def forbidden(xs):
        raise AssertionError("irrelevant callback")
    p = ComputationPlan(("x", "unused"), (
        ComputeNode("irrelevant", ("unused",), ("bad",)),
        ComputeNode("out", ("x",), ("add",)),
    ), ("out",))
    for policy in ("fifo", "cheap", "critical"):
        session = VerifiedComputation(p, [add(), add("bad", execute=forbidden)], WorkBudget(100), inputs={"x": True})
        assert session.run(policy).status == "complete"
        assert "irrelevant" not in session.values


def test_plan_validation_and_snapshot_bad_derived_key():
    with pytest.raises(ValueError):
        ComputationPlan(("x",), (ComputeNode("out", ("later",), ("add",)),), ("out",))
    with pytest.raises(ValueError):
        ComputationPlan(("x",), (ComputeNode("out", ("x",), ("add",)),), ("x",))
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "compute", {})
    session = VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal, inputs={"x": 2, "y": 3})
    session.run()
    state = copy.deepcopy(session.snapshot())
    state["cache"]["out"]["key"] = "wrong"
    journal.checkpoint(state)
    with pytest.raises(ValueError, match="derived"):
        VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal)


def test_typed_source_revision_equality_does_not_reuse_int_for_bool():
    op = add(execute=lambda xs: type(xs[0]) is bool,
             verify=lambda xs, value: type(value) is bool and value == (type(xs[0]) is bool))
    p = ComputationPlan(("x",), (ComputeNode("out", ("x",), ("add",)),), ("out",))
    session = VerifiedComputation(p, [op], WorkBudget(300), inputs={"x": 1})
    session.run()
    assert session.outputs["out"] is False
    session.update_inputs({"x": True}, "initial")
    assert not session.outputs
    session.run()
    assert session.outputs["out"] is True


def test_reentry_from_charge_sink_is_blocked_before_mutation():
    session = VerifiedComputation(plan(), [add()], WorkBudget(300), inputs={"x": 2, "y": 3})
    errors = []
    def sink(snapshot):
        try:
            session.update_inputs({"x": 99}, "reentrant")
        except RuntimeError as error:
            errors.append(str(error))
    session.budget.on_charge = sink
    session.run()
    assert errors and session.outputs["out"] == 5


def test_ledger_history_rollback_rejected():
    journal = ExperienceJournal(KnowledgeStore(), "compute", {})
    session = VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal, inputs={"x": 2, "y": 3})
    old = session.snapshot()
    session.run()
    journal.checkpoint(old)
    with pytest.raises(ValueError, match="nonmonotonic"):
        VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal)


def test_restored_tampered_value_is_independently_reverified():
    journal = ExperienceJournal(KnowledgeStore(), "compute", {})
    session = VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal, inputs={"x": 2, "y": 3})
    session.run()
    changed = session.snapshot()
    changed["cache"]["out"]["value"] = 777
    journal.checkpoint(changed)
    with pytest.raises(ValueError, match="independent verification"):
        VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal)


def test_liveness_revocation_invalidates_descendants_and_preserves_disjoint():
    active = [True]
    op = ComputeOperator("live", "1", lambda xs: sum(xs), lambda xs, value: value == sum(xs),
                         2, 3, is_valid=lambda: active[0])
    p = ComputationPlan(("x", "y"), (
        ComputeNode("a", ("x",), ("live", "add")),
        ComputeNode("b", ("y",), ("add",)),
        ComputeNode("out", ("a", "b"), ("add",)),
    ), ("out",))
    session = VerifiedComputation(p, [op, add()], WorkBudget(1000), inputs={"x": 2, "y": 3})
    session.run()
    active[0] = False
    assert dict(session.values) == {"b": 3}
    assert not session.outputs


def test_affordable_alternative_is_not_hidden_by_expensive_first():
    p = ComputationPlan(("x",), (ComputeNode("out", ("x",), ("huge", "add")),), ("out",))
    huge = ComputeOperator("huge", "1", lambda xs: 999, lambda xs, value: False, 1000, 1000)
    for policy in ("fifo", "cheap", "critical"):
        session = VerifiedComputation(p, [huge, add()], WorkBudget(100), inputs={"x": 2})
        assert session.run(policy).status == "complete"
        assert session.outputs["out"] == 2


def test_final_checkpoint_sink_failure_hides_verified_but_unsaved_value(monkeypatch):
    journal = ExperienceJournal(KnowledgeStore(), "compute", {})
    session = VerifiedComputation(plan(), [add()], WorkBudget(300), journal=journal, inputs={"x": 2, "y": 3})
    original = journal.checkpoint
    def failing(state):
        if state["pending"] is None:
            raise OSError("result write failed")
        return original(state)
    monkeypatch.setattr(journal, "checkpoint", failing)
    with pytest.raises(OSError):
        session.step()
    with pytest.raises(RuntimeError):
        _ = session.outputs
    with pytest.raises(RuntimeError):
        _ = session.values


def test_budgetstop_raised_by_sink_is_fatal():
    from solvi.experimental.experience._common import BudgetStop
    session = VerifiedComputation(plan(), [add()], WorkBudget(300), inputs={"x": 2})
    def sink(snapshot):
        raise BudgetStop("sink refuses")
    session.budget.on_charge = sink
    with pytest.raises(RuntimeError, match="sink failed"):
        session.step()
    with pytest.raises(RuntimeError):
        session.step()
