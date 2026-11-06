import pytest

from solvi.experimental.experience._common import BudgetStop, WorkBudget, digest
from solvi.experimental.experience.competences import CompetenceRegistry, ExternalExample
from solvi.experimental.experience.computation import (
    ComputationPlan, ComputeNode, ComputeOperator, VerifiedComputation,
)
from solvi.experimental.experience.domain_checks import DomainReceipt, FiniteDomain
from solvi.experimental.experience.operation_learning import OperationCatalog
from solvi.experimental.experience.programs import Grammar, Program


def example(n, x, split="proposal", expected=None):
    return ExternalExample(n, n, "environment", "m", "schema", "s", {"x": x},
                           x + 1 if expected is None else expected, split)


class Oracle:
    source_id, source_revision = "audit", "r1"

    def __init__(self, wrong=False):
        self.calls, self.wrong = 0, wrong

    def read(self, request_id, source_revision, point):
        self.calls += 1
        return DomainReceipt(request_id, "oracle:" + request_id, self.source_id,
                             source_revision, digest(point), point["x"] + (2 if self.wrong else 1))


def setup():
    budget = WorkBudget(100000)
    grammar = Grammar(("x",))
    registry = CompetenceRegistry(grammar)
    catalog = OperationCatalog(registry)
    program = Program(("add", ("input", "x"), ("const", 1)))
    key = catalog.record(program, example("p0", 0), budget)
    catalog.record(program, example("p1", 1), budget)
    domain = FiniteDomain("s", "m", "schema", grammar, "audit", "r1",
                          [{"x": x} for x in range(6)], budget=budget)
    return budget, registry, catalog, program, key, domain


def acquired(verify=None):
    budget, registry, catalog, program, key, domain = setup()
    oracle = Oracle()
    admitted = catalog.acquire(key, [example("a0", 2, "admission"), example("a1", 3, "admission")],
                               domain, oracle, budget)
    assert admitted.accepted and oracle.calls == 6
    op = catalog.operation(key, verify=verify or (lambda values, out: out == values[0] + 1),
                           verify_revision="independent-v1", budget=budget)
    return budget, registry, catalog, program, key, domain, op


def test_reuses_new_binding_and_composition_with_full_primitive_cost():
    budget, _, _, program, _, _, op = acquired()
    direct = WorkBudget(100)
    assert program.evaluate({"x": 4}, direct, Grammar(("x",))) == 5
    assert op.cost >= direct.spent
    plan = ComputationPlan(("z",), (ComputeNode("first", ("z",), (op.name,)),
                                    ComputeNode("second", ("first",), (op.name,))), ("second",))
    session = VerifiedComputation(plan, [op], budget, inputs={"z": 4})
    session.run()
    assert dict(session.outputs) == {"second": 6}
    assert budget.counts["operation_compile"] >= program.nodes


def test_counterexample_revokes_every_previously_built_callable():
    _, registry, catalog, _, key, _, op = acquired(lambda _values, _out: False)
    assert not op.verify((4,), 5)
    with pytest.raises(ValueError, match="revoked"):
        op.execute((4,))
    assert next(iter(registry.snapshot()["competences"].values()))["status"] == "revoked"
    with pytest.raises(ValueError, match="revoked"):
        catalog.operation(key, verify=lambda _v, _p: True, verify_revision="v2",
                          budget=WorkBudget(100)).execute((4,))


def test_rejects_outside_domain_and_strict_type_confusion():
    *_, op = acquired()
    for values in [(99,), (True,), [4], ()]:
        with pytest.raises(ValueError):
            op.execute(values)


def test_independent_domain_failure_never_produces_operation():
    budget, _, catalog, _, key, domain = setup()
    result = catalog.acquire(key, [example("a0", 2, "admission"), example("a1", 3, "admission")],
                             domain, Oracle(wrong=True), budget)
    assert not result.accepted
    with pytest.raises(ValueError, match="independently admitted"):
        catalog.operation(key, verify=lambda _v, _o: True, verify_revision="v1", budget=budget)


def test_proposal_api_does_not_fabricate_dependencies_and_restores():
    budget, registry, _, program, _, _ = setup()
    candidate = registry.propose_program(program, [example("other", 5)], budget)
    assert candidate.dependencies == ()
    restored = CompetenceRegistry.restore(registry.snapshot(), grammar=registry.grammar)
    assert restored.snapshot() == registry.snapshot()
    with pytest.raises(ValueError):
        registry.propose_program(program, [example("wrong", 5, expected=10)], budget)
    with pytest.raises(ValueError):
        registry.propose_program(Program(("const", 99)), [example("badconst", 5, expected=99)], budget)
    with pytest.raises(ValueError):
        registry.propose_program(program, [example("eval", 5, "evaluation")], budget)
    with pytest.raises(BudgetStop):
        registry.propose_program(program, [example("empty-budget", 5)], WorkBudget(0))


def test_distinct_trace_bindings_and_split_separation():
    budget, _, catalog, program, key, domain = setup()
    with pytest.raises(ValueError, match="distinct"):
        catalog.record(program, example("same", 1), budget)
    with pytest.raises(ValueError, match="split"):
        catalog.acquire(key, [example("p0", 0), example("p1", 1)], domain, Oracle(), budget)


def test_live_verifier_revision_changes_operator_revision():
    budget, _, catalog, _, key, _, first = acquired()
    second = catalog.operation(key, verify=lambda v, p: p == v[0] + 1,
                               verify_revision="independent-v2", budget=budget)
    assert first.revision != second.revision


def test_acquisition_and_lookup_cannot_run_on_empty_budget():
    _, _, catalog, _, key, domain = setup()
    with pytest.raises(BudgetStop):
        catalog.acquire(key, [], domain, Oracle(), WorkBudget(0))
    with pytest.raises(BudgetStop):
        catalog.operation(key, verify=lambda _v, _o: True, verify_revision="v1", budget=WorkBudget(0))


def test_revocation_removes_previously_cached_descendants():
    budget, registry, _, _, _, _, op = acquired()
    plan = ComputationPlan(("z",), (ComputeNode("first", ("z",), (op.name,)),
                                    ComputeNode("second", ("first",), (op.name,))), ("second",))
    session = VerifiedComputation(plan, [op], budget, inputs={"z": 4})
    session.run()
    assert session.outputs["second"] == 6
    cid = next(iter(registry.snapshot()["competences"]))
    registry.invalidate(cid, "counterexample", "outside-revocation")
    assert not session.outputs and not session.values


def test_verifier_must_be_strict_bool_and_rechecks_live_admission():
    *_, op = acquired(lambda _v, _p: 1)
    with pytest.raises(ValueError, match="strict bool"):
        op.verify((4,), 5)
    assert op.is_valid()
    budget, registry, catalog, _, key, _, _ = acquired()

    def revoke(_v, _p):
        cid = next(iter(registry.snapshot()["competences"]))
        registry.invalidate(cid, "counterexample", "during-verify")
        return True

    op = catalog.operation(key, verify=revoke, verify_revision="revoking-v1", budget=budget)
    with pytest.raises(ValueError, match="revoked"):
        op.verify((4,), 5)


def test_extracts_real_accepted_sessions_then_reuses_new_composition():
    budget = WorkBudget(100000)
    grammar = Grammar(("x",))
    catalog = OperationCatalog(CompetenceRegistry(grammar))
    program = Program(("add", ("input", "x"), ("const", 1)))
    direct = ComputeOperator("direct", "r1", lambda v: v[0] + 1,
                             lambda v, p: p - v[0] == 1, cost=4, verify_cost=2)
    plan = ComputationPlan(("source",), (ComputeNode("accepted", ("source",), (direct.name,)),),
                           ("accepted",))
    sessions = []
    for value in (0, 1):
        session = VerifiedComputation(plan, [direct], budget, inputs={"source": value})
        session.run()
        keys = catalog.record_session(session, {"accepted": program}, budget,
                                      scope="s", mechanism="m", schema="schema", task_prefix="training")
        assert keys == (program.fingerprint,)
        sessions.append(session)
    key = keys[0]
    before = budget.spent
    assert catalog.record_session(sessions[0], {"accepted": program}, budget,
                                  scope="s", mechanism="m", schema="schema", task_prefix="training") == (key,)
    assert budget.spent > before
    assert len(catalog._templates[key]["candidate"].proposal_examples) == 2
    domain = FiniteDomain("s", "m", "schema", grammar, "audit", "r1",
                          [{"x": x} for x in range(6)], budget=budget)
    assert catalog.acquire(key, [example("a0", 2, "admission"), example("a1", 3, "admission")],
                           domain, Oracle(), budget).accepted
    learned = catalog.operation(key, verify=lambda v, p: p - v[0] == 1,
                                verify_revision="independent-v1", budget=budget)
    composition = ComputationPlan(("source",), (
        ComputeNode("one", ("source",), (learned.name,)),
        ComputeNode("two", ("one",), (learned.name,))), ("two",))
    new_session = VerifiedComputation(composition, [learned], budget, inputs={"source": 4})
    new_session.run()
    assert new_session.outputs["two"] == 6
    assert budget.counts["operation_trace_port"] >= 3


def test_session_extraction_skips_unverified_output_and_prepays():
    budget = WorkBudget(100000)
    catalog = OperationCatalog(CompetenceRegistry(Grammar(("x",))))
    program = Program(("add", ("input", "x"), ("const", 1)))
    wrong = ComputeOperator("wrong", "r1", lambda v: v[0] + 2, lambda _v, _p: False)
    plan = ComputationPlan(("source",), (ComputeNode("rejected", ("source",), (wrong.name,)),),
                           ("rejected",))
    session = VerifiedComputation(plan, [wrong], budget, inputs={"source": 0})
    session.run()
    assert catalog.record_session(session, {"rejected": program}, budget, scope="s",
                                  mechanism="m", schema="schema", task_prefix="bad") == ()
    assert not catalog._templates
    with pytest.raises(BudgetStop):
        catalog.record_session(session, {"rejected": program}, WorkBudget(0), scope="s",
                               mechanism="m", schema="schema", task_prefix="bad")
