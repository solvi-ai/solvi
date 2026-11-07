import copy

import pytest

from solvi.experimental.experience._common import BudgetStop, WorkBudget, digest
from solvi.experimental.experience.cooperation import (
    ConstraintSolver, CooperativeSearch, Evidence, FiniteProblem, ObjectiveSolver,
    finite_reference,
)


def problem(revision="1", feasible=None):
    return FiniteProblem({"a": {"cost": 0, "ok": False},
                          "b": {"cost": 3, "ok": True},
                          "c": {"cost": 8, "ok": True}},
                         {"allowed": feasible or (lambda x: x["ok"])},
                         lambda x: x["cost"], revision=revision, budget=WorkBudget(1000))


def exclusion(id="e", **kwargs):
    return Evidence(id, "constraint", "1", "exclusion", ("a",), "allowed", **kwargs)


def test_actual_exchange_changes_next_solver_input_and_cost():
    p = problem()
    shared = CooperativeSearch(p, WorkBudget(100))
    observed = []
    objective = ObjectiveSolver(p)

    def observe(residual, budget):
        observed.append(residual.candidates)
        return objective(residual, budget)

    result = shared.run([ConstraintSolver(p), observe])
    assert observed == [("b", "c")]
    assert result["status"] == "unknown"
    assert shared.residual().incumbent == "b"
    shared_objective_cost = shared.budget.counts["cooperation.objective"]
    independent = CooperativeSearch(p, WorkBudget(100))
    independent.run([ObjectiveSolver(p)])
    assert independent.budget.counts["cooperation.objective"] == 4
    assert shared_objective_cost == 3
    assert shared.certify() == {"status": "optimal", "solution": (3, "b")}


def test_checked_bound_and_counterexample_are_not_fabricated():
    s = CooperativeSearch(problem(), WorkBudget(100))
    s.submit(Evidence("lb", "bound", "1", "lower_bound", value=3))
    assert s.residual().lower_bound == 3
    with pytest.raises(ValueError, match="rejected"):
        s.submit(Evidence("bad", "liar", "1", "lower_bound", value=4))
    with pytest.raises(ValueError, match="rejected"):
        s.submit(Evidence("bad2", "liar", "1", "exclusion", ("b",), "allowed"))
    with pytest.raises(ValueError, match="rejected"):
        s.submit(Evidence("bad3", "liar", "1", "incumbent", ("a",), 0))
    with pytest.raises(ValueError, match="rejected"):
        s.submit(Evidence("bad4", "liar", "1", "incumbent", ("b",), 2))


def test_revision_dedup_dependencies_and_transitive_revocation():
    s = CooperativeSearch(problem(), WorkBudget(100))
    s.submit(exclusion())
    s.submit(Evidence("i", "objective", "1", "incumbent", ("b",), 3, ("e",)))
    s.submit(Evidence("l", "bound", "1", "lower_bound", value=0, dependencies=("i",)))
    with pytest.raises(ValueError):
        s.submit(exclusion())
    with pytest.raises(ValueError):
        s.submit(Evidence("x", "a", "old", "lower_bound", value=0))
    with pytest.raises(ValueError):
        s.submit(exclusion("cycle", dependencies=("cycle",)))
    assert s.revoke("e") == ("e", "i", "l")
    assert s.residual().candidates == ("a", "b", "c")
    assert s.residual().incumbent is None
    with pytest.raises(ValueError):
        s.submit(exclusion("new", dependencies=("i",)))


def test_snapshot_replays_host_checker_and_pays():
    s = CooperativeSearch(problem(), WorkBudget(300))
    s.submit(exclusion())
    s.revoke("e")
    s.submit(exclusion("e2"))
    expected = s.residual()
    before_spent = s.budget.spent
    restored = CooperativeSearch.restore(s.snapshot(), problem(), s.budget, authority=s.authority)
    assert restored.residual() == expected
    assert restored.budget.spent > before_spent
    assert restored.budget.counts["cooperation.replay"] == 3
    with pytest.raises(ValueError, match="stale"):
        s.residual()
    snapshot = restored.snapshot()
    with pytest.raises(ValueError):
        CooperativeSearch.restore(snapshot, problem("2"), restored.budget, authority=restored.authority)
    with pytest.raises(ValueError, match="mismatch"):
        CooperativeSearch.restore(snapshot, problem(feasible=lambda x: True), restored.budget, authority=restored.authority)
    forged = copy.deepcopy(snapshot)
    forged["events"][0]["accept"]["targets"] = ["b"]
    forged["digest"] = digest({k: v for k, v in forged.items() if k != "digest"})
    with pytest.raises(ValueError, match="untrusted"):
        CooperativeSearch.restore(forged, problem(), restored.budget, authority=restored.authority)


def test_budget_exhaustion_never_proves_and_replay_fails_closed():
    s = CooperativeSearch(problem(), WorkBudget(1))
    assert s.run([ConstraintSolver(s.problem)])["status"] == "unknown"
    with pytest.raises(BudgetStop):
        s.residual()
    assert s.budget.spent <= 1
    assert s.certify()["status"] == "unknown"
    s = CooperativeSearch(problem(), WorkBudget(20))
    s.submit(exclusion())
    snapshot = s.snapshot()
    s.budget.charge("consume", s.budget.remaining)
    with pytest.raises(BudgetStop):
        CooperativeSearch.restore(snapshot, problem(), s.budget, authority=s.authority)


def test_typed_attrs_and_host_outputs_and_unknown_targets():
    with pytest.raises(ValueError):
        FiniteProblem({"x": {"bad": 1.2}}, {}, lambda x: 0, revision="1", budget=WorkBudget(1000))
    with pytest.raises(ValueError, match="bool"):
        problem(feasible=lambda x: 1)
    s = CooperativeSearch(problem(), WorkBudget(100))
    with pytest.raises(ValueError):
        s.submit(Evidence("x", "x", "1", "exclusion", ("missing",), "allowed"))
    # Mutation of returned attrs never changes host task.
    attrs = s.problem.attributes("b")
    attrs["cost"] = -100
    assert finite_reference(s.problem, s.budget) == (3, "b")


def test_exhaustive_oracle_against_manual_domains():
    for n in range(1, 12):
        p = FiniteProblem({str(i): {"x": i} for i in range(n)},
                          {"even": lambda a: a["x"] % 2 == 0},
                          lambda a: -a["x"], revision=str(n), budget=WorkBudget(1000))
        expected = min((-i, str(i)) for i in range(n) if i % 2 == 0)
        assert finite_reference(p, WorkBudget(100)) == expected
    empty = FiniteProblem({}, {}, lambda x: 0, revision="empty", budget=WorkBudget(1000))
    assert CooperativeSearch(empty, WorkBudget(10)).certify()["status"] == "infeasible"


def test_authority_rollback_refund_transfer_and_trusted_crossprocess():
    from solvi.experimental.experience.cooperation import CooperativeAuthority
    s = CooperativeSearch(problem(), WorkBudget(1000))
    s.submit(exclusion())
    old = s.snapshot()
    s.revoke("e")
    with pytest.raises(ValueError, match="stale"):
        CooperativeSearch.restore(old, problem(), s.budget, authority=s.authority)
    current = s.snapshot()
    old_ledger = WorkBudget.restore(s.budget.snapshot())
    s.budget.charge("unrelated_shared_work", 50)
    with pytest.raises(ValueError, match="refunded"):
        CooperativeSearch.restore(current, problem(), old_ledger, authority=s.authority)
    # Trusted host persists the CURRENT anchor and ledger, not an old payload.
    anchor = s.authority.export_trusted_state()
    authority = CooperativeAuthority.from_trusted_state(anchor)
    ledger = WorkBudget.restore(s.budget.snapshot())
    restored = CooperativeSearch.restore(current, problem(), ledger, authority=authority)
    assert restored.residual().candidates == ("a", "b", "c")
    with pytest.raises(ValueError):
        CooperativeSearch.restore(current, problem(), ledger, authority=authority)


def test_evidence_subclass_cannot_change_stored_certificate():
    class Counterfeit(Evidence):
        def data(self):
            return {**super().data(), "targets": ["c"], "value": -100}
    s = CooperativeSearch(problem(), WorkBudget(100))
    with pytest.raises(ValueError, match="exact Evidence"):
        s.submit(Counterfeit("fake", "evil", "1", "incumbent", ("b",), 3))
    assert s.residual().incumbent is None


def test_shared_ledger_many_categories_and_problem_binding():
    s = CooperativeSearch(problem(), WorkBudget(1000))
    for i in range(70):
        s.budget.charge(f"other-module-{i}")
    with pytest.raises(AttributeError):
        s.problem = problem("other")
    restored = CooperativeSearch.restore(s.snapshot(), s.problem, s.budget, authority=s.authority)
    assert restored.residual().candidates == ("a", "b", "c")


def test_reentrant_revoke_during_replay_cannot_be_overwritten():
    s = CooperativeSearch(problem(), WorkBudget(1000))
    s.submit(exclusion())
    snapshot = s.snapshot()
    triggered = False

    def sink(ledger):
        nonlocal triggered
        if not triggered and ledger["counts"].get("cooperation.replay", 0):
            triggered = True
            s.revoke("e")

    s.budget.on_charge = sink
    with pytest.raises(ValueError, match="changed during replay"):
        CooperativeSearch.restore(snapshot, s.problem, s.budget, authority=s.authority)
    assert triggered
    assert s.residual().candidates == ("a", "b", "c")
