"""Independent scorer fixtures; no learning actor or oracle executes here."""

from pathlib import Path
import sys
import hashlib
import json

import pytest

from solvi.experimental.experience._common import WorkBudget, digest
from solvi.experimental.experience.domain_checks import FiniteDomain
from solvi.experimental.experience.programs import Grammar, Program

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import refinement_score as score  # noqa: E402
from refinement_tasks import Problem  # noqa: E402


@pytest.fixture
def case(monkeypatch):
    problem = Problem(
        "test_only",
        "numeric_tabular",
        Grammar(("x",)),
        tuple({"x": i} for i in range(32)),
        lambda p: p["x"],
        Program(("input", "x")),
        "Identity fixture",
    )
    monkeypatch.setattr(score, "tasks", lambda: (problem,))
    monkeypatch.setattr(score, "SEEDS", {"test": [0]})
    monkeypatch.setattr(score, "ARMS", ("FORMULA",))
    points = score.partition(problem, 0)["transfer"]
    budget = WorkBudget(100000)
    for kind, n in [("program_node", 14), ("program_input", 14), ("transfer_effect", 112)]:
        budget.charge(kind, n)
    row = dict(
        task="test_only",
        family="numeric_tabular",
        arm="FORMULA",
        seed=0,
        status="known_formula",
        budget=budget.snapshot(),
        learning=None,
        external_reads=0,
        predictions=[dict(point=p, value=p["x"], expected=p["x"], outcome="correct") for p in points],
    )
    return problem, row


def test_formula_fixture_scores_without_actors(tmp_path, case):
    _, row = case
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = score.summarize(path, split="test")
    assert result["trials"] == 1 and result["arms"]["FORMULA"]["correct"] == 14
    assert (
        result["arms"]["FORMULA"]["units"] == 140
        and result["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "damage",
    ["missing", "duplicate", "family", "ledger", "value_type", "typed_gold", "status", "point", "arm", "reads"],
)
def test_corrupt_or_incomplete_raw_is_rejected(tmp_path, case, damage):
    _, row = case
    rows = [row]
    if damage == "missing":
        rows = []
    elif damage == "duplicate":
        rows *= 2
    elif damage == "family":
        row["family"] = "unassigned"
    elif damage == "ledger":
        row["budget"]["spent"] -= 1
    elif damage == "value_type":
        row["predictions"][0]["value"] = False
    elif damage == "typed_gold":
        row["predictions"][0]["expected"] = True
    elif damage == "status":
        row["predictions"][0]["outcome"] = "wrong"
    elif damage == "point":
        row["predictions"].reverse()
    elif damage == "arm":
        row["arm"] = "DEV_ONLY"
    elif damage == "reads":
        row["external_reads"] = 14
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        score.summarize(path, split="test")


def test_verify_reexecutes_ast_and_binds_receipts(case):
    problem, _ = case
    work = WorkBudget(10**6)
    rows = score.partition(problem, 0)["verification"]
    domain = FiniteDomain("s", "m", "schema", problem.grammar, "source", "v1", rows, budget=work)
    requests = {}
    actuals = []
    evidence = []
    for i in range(len(rows)):
        point = domain.point_dict(i)
        rid = f"t:round:1:{i}"
        expected = point["x"]
        receipt = dict(request_id=rid, input_digest=digest(point), expected=expected)
        requests[rid] = receipt
        actuals.append(expected)
        evidence.append([digest(point), expected, receipt])
    program = problem.formula
    check = dict(
        status="verified",
        reason="complete_finite_domain",
        program_fingerprint=program.fingerprint,
        grammar_fingerprint=problem.grammar.fingerprint,
        domain_fingerprint=domain.fingerprint,
        checked_count=8,
        receipts=list(requests.values()),
        actuals=actuals,
        proof_id=digest(
            dict(
                program=program.fingerprint,
                grammar=problem.grammar.fingerprint,
                domain=domain.fingerprint,
                evidence=evidence,
            )
        ),
    )
    state = dict(
        config={"session_id": "t"},
        rounds=1,
        result={"program": program.to_dict()},
        candidate=program.to_dict(),
        receipts=requests,
        checks=[check],
    )
    assert score._verified(state, problem, domain, work) == program
    wrong = Program(("const", 0))
    state["candidate"] = state["result"]["program"] = wrong.to_dict()
    check["program_fingerprint"] = wrong.fingerprint
    with pytest.raises(ValueError, match="verification AST"):
        score._verified(state, problem, domain, work)
