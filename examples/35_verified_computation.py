"""Three connected experimental interfaces; host checks remain a trust boundary."""
from solvi.experimental.experience import (
    CompetenceRegistry, ComputationPlan, ComputeNode, ComputeOperator, DomainReceipt,
    ExternalExample, FiniteDomain, Grammar, OperationCatalog, Program,
    VerifiedComputation, WorkBudget,
)
from solvi.experimental.experience._common import digest


class IndependentOracle:
    source_id, source_revision = "audit", "independent-integer-spec-v1"

    def read(self, request_id, source_revision, point):
        # Independent of both the Program evaluator and the compiled callable.
        expected = sum((point["x"], 1))
        return DomainReceipt(request_id, "event:" + request_id, self.source_id,
                             source_revision, digest(point), expected)


def main():
    budget = WorkBudget(100000)
    program = Program(("add", ("input", "x"), ("const", 1)))
    grammar = Grammar(("x",))
    # Callback fees are declared host prices, not a CPU or wall-time quota.
    producer = ComputeOperator("successor", "v1", lambda values: values[0] + 1,
                               lambda values, out: type(out) is int and out - values[0] == 1,
                               cost=program.nodes + 1, verify_cost=4)
    catalog = OperationCatalog(CompetenceRegistry(grammar))
    training_plan = ComputationPlan(("source",),
                                    (ComputeNode("accepted", ("source",), (producer.name,)),),
                                    ("accepted",))
    for x in (0, 1):
        session = VerifiedComputation(training_plan, [producer], budget, inputs={"source": x})
        assert session.run().status == "complete"
        keys = catalog.record_session(session, {"accepted": program}, budget,
                                      scope="successor", mechanism="v1", schema="integer",
                                      task_prefix=f"training-{x}")
    domain = FiniteDomain("successor", "v1", "integer", grammar,
                          IndependentOracle.source_id, IndependentOracle.source_revision,
                          [{"x": x} for x in range(8)], budget=budget)
    admission = [ExternalExample(f"admit-{x}", f"admit-task-{x}", "audit", "v1",
                                 "integer", "successor", {"x": x}, x + 1, "admission")
                 for x in (2, 3)]
    assert catalog.acquire(keys[0], admission, domain, IndependentOracle(), budget).accepted
    learned = catalog.operation(keys[0], verify=producer.verify,
                                verify_revision="independent-v1", verify_cost=4, budget=budget)
    plan = ComputationPlan(("a", "b"), (
        ComputeNode("first", ("a",), (learned.name,)),
        ComputeNode("composed", ("first",), (learned.name,)),
        ComputeNode("unrelated", ("b",), (producer.name,)),
    ), ("composed", "unrelated"))
    session = VerifiedComputation(plan, [producer, learned], budget, inputs={"a": 4, "b": 2})
    session.run(policy="critical")
    assert dict(session.outputs) == {"composed": 6, "unrelated": 3}
    session.update_inputs({"a": 5}, "source-v2")
    assert dict(session.values) == {"unrelated": 3}
    session.run(policy="critical")
    assert dict(session.outputs) == {"composed": 7, "unrelated": 3}
    print("verified outputs:", dict(session.outputs))
    print("complete work ledger:", budget.spent, budget.counts)


if __name__ == "__main__":
    main()
