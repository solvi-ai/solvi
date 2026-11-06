"""Tiny checks; intentionally not executed before protocol freeze."""

import copy
import pytest
from benchmarks.experience_transfer import partitions, run_trial
from benchmarks._experience_scorer import ANCHORS, expected, hard_violation, score


def test_partition_disjointness():
    for family in ("resource", "workflow"):
        parts = partitions(1, family, "final")
        seen = set()
        for name in ("proposal", "admission", "dev", "id", "encoding"):
            pairs = {(c["x"], c["y"]) for c in parts[name].values()}
            assert not seen & pairs
            seen |= pairs
        prefix = list(parts["ood"].values())[:32]
        suffix = list(parts["ood"].values())[32:]
        assert not seen & {(c["x"], c["y"]) for c in prefix}
        assert [(c["x"], c["y"]) for c in suffix] == [(c["x"], c["y"]) for c in list(parts["id"].values())[:96]]
        if family == "workflow":
            for packet in (
                list(parts["proposal"].values()),
                list(parts["admission"].values()),
                prefix[:16],
                prefix[16:],
            ):
                assert any(c["x"] < c["y"] for c in packet)
                assert any(c["x"] > c["y"] for c in packet)
        assert set(partitions(1, family, "dev")) == {"proposal", "admission", "dev"}


@pytest.mark.parametrize("family", ["resource", "workflow"])
def test_hand_anchors_and_mutants(family):
    for x, y, mechanism, answer in ANCHORS[family]:
        case = dict(family=family, x=x, y=y, mechanism=mechanism)
        assert expected(case) == answer
        state = (
            dict(available=answer - 1, reserved=1, effects=1)
            if family == "resource"
            else dict(clock=answer, completed=True, stages_completed=[True, True], effects=1)
        )
        receipt = dict(kind="act", value=dict(answer=answer, ok=True, refused=False, state=state))
        assert score(case, answer, receipt, composite=False)
        assert not score(case, answer + 1, receipt, composite=False)
        mutant = copy.deepcopy(receipt)
        if family == "resource":
            mutant["value"]["state"]["available"] += 1
        else:
            mutant["value"]["state"]["stages_completed"][1] = False
        assert not score(case, answer, mutant, composite=False)
        if mechanism == "v2" and x >= 4 and y > 0:
            old = dict(case, mechanism="v1")
            assert not score(old, answer, receipt, composite=False)


@pytest.mark.parametrize("family", ["resource", "workflow"])
def test_tiny_formula_executes_real_effect(family):
    result = run_trial(1, family, "FORMULA_CACHE", 65536, 8, tasks=2)
    assert result["success"] == 2
    assert all(r["receipt"]["value"]["state"]["effects"] == 1 for r in result["rows"])
    assert result["weighted_total"] == sum(result["weighted_vector"].values())


def test_budget_failures_stay_in_denominator():
    result = run_trial(1, "resource", "FIXED", 0, 8, tasks=3)
    assert result["denominator"] == 3 and result["success"] == 0


def test_composite_gate():
    from benchmarks.experience_transfer import is_composite
    from solvi.experimental.experience.programs import Program

    assert not is_composite(Program(["sub", ["input", "x"], ["input", "y"]]))
    assert is_composite(Program(["sub", ["sub", ["input", "x"], ["input", "y"]], ["const", 1]]))


@pytest.mark.parametrize("price", [1, 8, 32])
def test_exact_probe_prices_do_not_round_cheap_observations_to_zero(price):
    from benchmarks.experience_transfer import Ledger

    b = Ledger(100, price)
    for kind in ("observe_measure", "observe_diagnose", "observe_noise", "action", "primitive"):
        b.charge(kind)
    assert b.counts == dict(
        observe_measure=8 * price, observe_diagnose=4 * price, observe_noise=price, action=64, primitive=8
    )
    assert b.limit == 800 and b.spent == 13 * price + 72


@pytest.mark.parametrize("field", ["answer", "effects", "reserved"])
def test_scorer_rejects_bool_int_receipt_mutants(field):
    case = dict(family="resource", x=3, y=1, mechanism="v1")  # composite answer = 1
    receipt = dict(
        kind="act", value=dict(answer=1, ok=True, refused=False, state=dict(available=0, reserved=1, effects=1))
    )
    assert score(case, 1, receipt)
    if field == "answer":
        receipt["value"]["answer"] = True
    else:
        receipt["value"]["state"][field] = True
    assert not score(case, 1, receipt)


def test_scorer_rejects_integer_completion_flags_and_audits_refusal_separately():
    case = dict(family="workflow", x=0, y=0, mechanism="v1")
    receipt = dict(
        kind="act",
        value=dict(
            answer=1,
            ok=True,
            refused=False,
            state=dict(clock=1, completed=True, stages_completed=[True, True], effects=1),
        ),
    )
    assert score(case, 1, receipt)
    receipt["value"]["state"]["completed"] = 1
    assert not score(case, 1, receipt) and hard_violation(case, receipt)
    receipt["value"] = dict(
        answer=1,
        ok=False,
        refused=True,
        state=dict(clock=0, completed=False, stages_completed=[False, False], effects=0),
    )
    assert not score(case, 0, receipt) and not hard_violation(case, receipt)


def test_completed_episode_checkpoint_resume(tmp_path):
    import json

    path = tmp_path / "trial.json"
    first = run_trial(1, "resource", "FORMULA_CACHE", 65536, 8, tasks=2, checkpoint_path=path)
    resumed = run_trial(
        1, "resource", "FORMULA_CACHE", 65536, 8, tasks=2, checkpoint=json.loads(path.read_text()), checkpoint_path=path
    )
    assert resumed["rows"] == first["rows"]
    assert resumed["weighted_total"] == first["weighted_total"]
    assert resumed["registry"] == first["registry"]


def test_receipt_survives_revision_budget_stop():
    import random
    from types import SimpleNamespace
    from benchmarks.experience_transfer import CompositeEnvironment, Ledger, solve_task
    from solvi.experimental.experience._common import BudgetStop

    class Registry:
        def predict(self, *args, **kwargs):
            return SimpleNamespace(status="applied", value=2, application_id="app", competence_id="op")

        def observe(self, *args, **kwargs):
            raise BudgetStop("revision cost")

    env = CompositeEnvironment({"task": dict(family="resource", x=7, y=4)})
    row = solve_task(env, "task", "FULL", Ledger(100, 8), Registry(), {}, random.Random(1))
    assert row["receipt"]["value"]["ok"] is True
    assert row["proposed"] == 2 and row["revision_pending"] is True


def test_inflight_checkpoint_is_fail_closed():
    from benchmarks.experience_transfer import SPEC

    checkpoint = dict(config=[SPEC, 1, "resource", "FULL", 65536, 8, "dev", 2], stage="learning_inflight")
    with pytest.raises(ValueError, match="unresolved partial"):
        run_trial(1, "resource", "FULL", 65536, 8, tasks=2, checkpoint=checkpoint)


def test_teacher_external_examples_keep_distinct_training_splits():
    from benchmarks.experience_transfer import CompositeEnvironment, Ledger, teacher_task
    from solvi.experimental.experience.competences import CompetenceRegistry
    from solvi.experimental.experience.programs import Grammar

    registry = CompetenceRegistry(Grammar(("x", "y"), operators=("sub", "max", "add")))
    env = CompositeEnvironment(
        {
            "proposal_task": dict(family="workflow", x=3, y=7, mechanism="v2"),
            "admission_task": dict(family="workflow", x=8, y=2, mechanism="v2"),
        }
    )
    budget = Ledger(1000, 8)
    proposal = teacher_task(env, "proposal_task", 0, budget, registry)
    admission = teacher_task(env, "admission_task", 16, budget, registry)
    assert proposal["adaptation_example"]["split"] == "proposal"
    assert admission["adaptation_example"]["split"] == "admission"
    assert proposal["proposed"] == 11 and admission["proposed"] == 11
    assert proposal["receipt"]["value"]["ok"] and admission["receipt"]["value"]["ok"]
    assert budget.raw["observation"] == 2 and budget.raw["action"] == 2


def test_ood_proposer_sees_only_fresh_proposal_prefix_and_suffix_is_frozen(monkeypatch):
    from solvi.experimental.experience.competences import CompetenceRegistry

    seen_proposal, seen_admission = [], []
    propose, admit = CompetenceRegistry.propose, CompetenceRegistry.admit

    def proposal_spy(self, examples, budget):
        rows = tuple(examples)
        seen_proposal.append([(e.task_id, e.split) for e in rows])
        return propose(self, rows, budget)

    def admission_spy(self, candidate, examples, budget, **kwargs):
        rows = tuple(examples)
        before = candidate.fingerprint
        seen_admission.append([(e.task_id, e.split) for e in rows])
        result = admit(self, candidate, rows, budget, **kwargs)
        assert candidate.fingerprint == before
        return result

    monkeypatch.setattr(CompetenceRegistry, "propose", proposal_spy)
    monkeypatch.setattr(CompetenceRegistry, "admit", admission_spy)
    result = run_trial(1, "workflow", "FULL", 262144, 8, stratum="ood", tasks=33)
    assert result["learning"]["record"]["accepted"]
    assert result["adaptation_learning"]["record"]["accepted"]
    assert len(seen_proposal) == len(seen_admission) == 2
    assert all(len(batch) == 16 and all(split == "proposal" for _, split in batch) for batch in seen_proposal)
    assert all(len(batch) == 16 and all(split == "admission" for _, split in batch) for batch in seen_admission)
    assert [tid for tid, _ in seen_proposal[1]] == [f"1:workflow:ood:{i}" for i in range(16)]
    assert [tid for tid, _ in seen_admission[1]] == [f"1:workflow:ood:{i}" for i in range(16, 32)]
    assert all(
        "1:workflow:ood:32" != tid
        for batches in (seen_proposal, seen_admission)
        for batch in batches
        for tid, _ in batch
    )
    assert result["rows"][-1]["phase"] == "suffix" and result["rows"][-1]["applied"]


def test_seed_shards_are_disjoint_complete_and_keep_order():
    from benchmarks.experience_transfer import SEEDS, shard_seeds

    shards = [shard_seeds(SEEDS, i, 3) for i in range(3)]
    assert sorted(seed for shard in shards for seed in shard) == list(SEEDS)
    assert all(len(shard) == 4 for shard in shards)
    assert shard_seeds(SEEDS) == SEEDS
    for index, count in [(-1, 3), (3, 3), (0, 0), (0, 13), (True, 3)]:
        with pytest.raises(ValueError, match="invalid seed shard"):
            shard_seeds(SEEDS, index, count)


@pytest.mark.parametrize("family,mechanism,primitives", [
    ("resource", "v1", 2), ("resource", "v2", 4),
    ("workflow", "v1", 2), ("workflow", "v2", 4),
])
def test_formula_debits_every_executed_primitive_before_work(family, mechanism, primitives):
    """Count Python arithmetic independently; cache hits must execute no formula."""
    from types import SimpleNamespace
    from benchmarks.experience_transfer import Ledger, solve_task

    budget = Ledger(65536, 8)
    operations = []

    def operation(name):
        operations.append(name)
        assert len(operations) <= budget.raw.get("formula_primitive", 0), "unpaid formula operation"

    class CountedInt(int):
        def __sub__(self, other):
            operation("sub")
            return CountedInt(int(self) - int(other))

        def __add__(self, other):
            operation("add")
            return CountedInt(int(self) + int(other))

        def __ge__(self, other):
            operation("ge")
            return int(self) >= int(other)

        def __gt__(self, other):
            operation("gt")
            return int(self) > int(other)

    task = SimpleNamespace(scope="composite:" + family, mechanism=mechanism,
                           inputs={"x": CountedInt(8), "y": CountedInt(1)})
    receipt = SimpleNamespace(value={"ok": True}, to_dict=lambda: {"kind": "act"})
    environment = SimpleNamespace(task=lambda task_id: task, execute=lambda *args: receipt)
    cache = {}
    first = solve_task(environment, "first", "FORMULA_CACHE", budget, None, cache, None)
    assert len(operations) == budget.raw["formula_primitive"] == primitives
    assert int(first["proposed"]) == (5 if family == "resource" and mechanism == "v2"
                                     else 6 if family == "resource" else 10 if mechanism == "v2" else 9)
    solve_task(environment, "second", "FORMULA_CACHE", budget, None, cache, None)
    assert len(operations) == budget.raw["formula_primitive"] == primitives
    assert budget.raw["cache_lookup"] == 2
    assert budget.raw["cache_fill"] == 1
