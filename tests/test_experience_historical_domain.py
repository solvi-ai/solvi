"""Historical finite-domain regression; fixtures never rerun historical trials."""

import itertools
import json
from pathlib import Path

import pytest

from solvi.experimental.experience._common import WorkBudget, canonical, digest
from solvi.experimental.experience.competences import Candidate, CompetenceRegistry, ExternalExample
from solvi.experimental.experience.domain_checks import DomainReceipt, FiniteDomain
from solvi.experimental.experience.programs import Grammar, Program


BAD_RESOURCE = ("sub", ("sub", ("input", "x"), ("input", "y")), ("add", ("const", 1), ("const", 1)))


class Oracle:
    source_id = "audit"
    source_revision = "fixture-v1"

    def __init__(self, truth):
        self.truth = truth
        self.points = []

    def read(self, request_id, revision, point):
        self.points.append(point.copy())
        return DomainReceipt(
            request_id, "oracle:" + request_id, self.source_id, revision, digest(point), self.truth(point)
        )


def freeze_validated(grammar, candidate):
    """Host fixture import through the public strict snapshot validator.

    This is a forced diagnostic candidate, never a claim of autonomous search.
    No internal candidate dictionary is assigned on the live registry.
    """
    state = CompetenceRegistry(grammar).snapshot()
    state["candidates"] = {candidate.fingerprint: candidate.to_dict()}
    state["events"] = {row.event_id: row.to_dict() for row in candidate.proposal_examples}
    state["tasks"] = {row.task_id: [row.split, row.event_id] for row in candidate.proposal_examples}
    state["digest"] = digest({k: v for k, v in state.items() if k != "digest"})
    return CompetenceRegistry.restore(state, grammar=grammar)


def compare_admission(grammar, candidate, admission, points, truth, *, minimum):
    budget = WorkBudget(1000000)
    for row in (*candidate.proposal_examples, *admission):
        budget.charge("fixture_label", 8)
        assert type(row.expected) is type(truth(row.inputs)) and row.expected == truth(row.inputs)
        assert candidate.program.evaluate(row.inputs, budget, grammar) == row.expected
    empirical = freeze_validated(grammar, candidate)
    assert empirical.admit(candidate, admission, budget, min_examples=minimum).accepted
    checked = freeze_validated(grammar, candidate)
    domain = FiniteDomain(
        candidate.scope,
        candidate.mechanism,
        candidate.schema,
        grammar,
        "audit",
        "fixture-v1",
        points,
        budget=budget,
        input_types={name: "bool" if name == "expired" else "int" for name in grammar.inputs},
        result_type="bool" if candidate.scope == "eligibility" else "int",
    )
    oracle = Oracle(truth)
    result = checked.admit_checked(candidate, admission, budget, domain=domain, oracle=oracle, min_examples=minimum)
    assert not result.accepted
    proof = checked.snapshot()["checked"][candidate.fingerprint]["result"]
    assert proof["status"] == "rejected" and proof["reason"] == "counterexample"
    assert not checked.snapshot()["competences"]
    witness = proof["witness"]
    assert canonical(witness["actual"]) != canonical(witness["expected"])
    assert (
        checked.predict(
            oracle.points[-1],
            task_id="post-rejection",
            scope=candidate.scope,
            mechanism=candidate.mechanism,
            schema=candidate.schema,
            domain=domain,
            budget=budget,
        ).status
        == "unknown"
    )
    return oracle.points[-1], witness, budget


def test_actual_historical_16_plus_16_admits_but_full_435_rejects_3_0():
    fixture = json.loads((Path(__file__).parent / "fixtures/experience_historical_resource_v2.json").read_text())
    assert fixture["provenance"]["trial_key"] == "cf3b6bffe02feda9eb4f809ac8560dbfa85620f3dcb4ca2f2947b0ff54d66a64"
    assert fixture["provenance"]["line"] == 20511 and fixture["provenance"]["seed"] == 1101
    candidate = Candidate.from_dict(fixture["candidate"])
    grammar = Grammar(("x", "y"), operators=("sub", "max", "add"))
    assert candidate.grammar_fingerprint == grammar.fingerprint
    assert candidate.program.ast == BAD_RESOURCE
    assert candidate.fingerprint == fixture["provenance"]["competence_id"]
    admission = tuple(ExternalExample.from_dict(row) for row in fixture["admission"])
    assert len(candidate.proposal_examples) == len(admission) == 16
    assert len({row.event_id for row in (*candidate.proposal_examples, *admission)}) == 32
    points = [{"x": x, "y": y} for x in range(32) for y in range(32) if x >= y + 3]
    assert len(points) == 435
    truth = lambda p: p["x"] - p["y"] - 1 - int(p["x"] >= 4)
    point, witness, budget = compare_admission(grammar, candidate, admission, points, truth, minimum=16)
    assert point == {"x": 3, "y": 0}
    assert witness["actual"] == 1 and witness["expected"] == 2
    assert budget.counts["domain_oracle"] > 0


@pytest.mark.parametrize("family", ["inventory", "eligibility", "schedule"])
def test_forced_wrong_family_candidate_passes_empirical_then_fails_domain(family):
    if family == "inventory":
        names = ("stock", "reserved", "damaged")
        axes = (range(5),) * 3
        ast = ("input", "stock")
        wrong = lambda p: p["stock"]
        truth = lambda p: max(0, p["stock"] - p["reserved"] - p["damaged"])
        operators = ("sub", "max", "add")
    elif family == "eligibility":
        names = ("requested", "available", "expired")
        axes = (range(5), range(5), (False, True))
        ast = ("const", False)
        wrong = lambda p: False
        truth = lambda p: p["requested"] < p["available"] and not p["expired"]
        operators = ("lt", "not", "and", "or", "eq")
    else:
        names = ("a", "b", "c", "d")
        axes = (range(4),) * 4
        ast = ("add", ("input", "a"), ("input", "b"))
        wrong = lambda p: p["a"] + p["b"]
        truth = lambda p: max(p["a"] + p["b"], p["c"]) + p["d"]
        operators = ("add", "max", "sub")
    grammar = Grammar(
        names, constants=(0, 1), operators=operators, max_depth=4, max_nodes=7, max_candidates=10000, max_abs=32
    )
    points = [dict(zip(names, values)) for values in itertools.product(*axes)]
    matching = [p for p in points if canonical(wrong(p)) == canonical(truth(p))]
    if family == "inventory":
        assert len(matching) == 29
    correct = sorted(
        matching,
        key=lambda p: canonical([p[name] for name in names]),
    )[:16]
    assert len(correct) == 16
    rows = [
        ExternalExample(
            f"{family}:event:{i}",
            f"{family}:task:{i}",
            "environment",
            "v1",
            "scalar",
            family,
            point,
            truth(point),
            "proposal" if i < 8 else "admission",
        )
        for i, point in enumerate(correct)
    ]
    candidate = Candidate(Program(ast), tuple(rows[:8]), family, "v1", "scalar", grammar.fingerprint)
    assert candidate.program.ast == Program(ast).ast
    compare_admission(grammar, candidate, tuple(rows[8:]), points, truth, minimum=8)
