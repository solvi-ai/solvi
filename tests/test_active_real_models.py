"""Known synthetic CART fixtures; no dataset or comparative run is loaded."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import random
import sys

import pytest
from solvi.experimental.experience._common import BudgetStop, WorkBudget, digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from active_real_models import Forest, Node, Tree, fit_forest  # noqa: E402


def rows(points):
    return [
        dict(id=str(i), inputs={f"f{f}": v for f, v in enumerate(values)}, label=label)
        for i, (values, label) in enumerate(points)
    ]


def threshold_rows():
    return rows([((0, 0), False), ((7, 7), True)] * 16)


def test_tie_breaks_exact_costs_and_ordered_bootstrap_proof():
    data = threshold_rows()
    ledger = WorkBudget(10**6)
    forest = fit_forest(data, ledger, seed=11)
    assert ledger.counts == dict(
        tree_fit_validate=64,
        forest_bootstrap=96,
        tree_fit_node=9,
        tree_fit_labels=192,
        tree_fit_hist=192,
        tree_fit_candidates=3 * 14 * 7 * 2,
        tree_fit_route=96,
    )
    for j, tree in enumerate(forest.trees):
        rng = random.Random(11 + j)
        expected = [rng.randrange(len(data)) for _ in data]
        assert tree.bootstrap == tuple(data[i]["id"] for i in expected)
        # f1 gives the same fractions as f0; cuts 1 through 7 give the same partition.
        assert (tree.root.feature, tree.root.threshold) == ("f0", 1)
        assert tree.root.n == 32 and tree.root.pos == sum(data[i]["label"] for i in expected)
        assert tree.root.left.n == 32 - tree.root.pos and tree.root.left.pos == 0
    pred = WorkBudget(100)
    assert forest.predict({"f0": 0, "f1": 7}, pred) is False
    assert pred.counts == {"forest_vote": 3, "tree_predict_node": 6, "tree_predict_input": 3}
    with pytest.raises(FrozenInstanceError):
        forest.seed = 12


def test_nonlinear_xor_and_stable_round_trip():
    data = rows([((a, b), bool(a) ^ bool(b)) for _ in range(24) for a in [0, 1] for b in [0, 1]])
    forest = fit_forest(data, WorkBudget(10**6), seed=13)
    restored = Forest.from_dict(forest.to_dict())
    assert forest == restored == fit_forest(data, WorkBudget(10**6), seed=13)
    assert fit_forest(data, WorkBudget(10**6), seed=14).fingerprint != forest.fingerprint
    for a in range(2):
        for b in range(2):
            assert restored.predict(dict(f0=a, f1=b), WorkBudget(100)) is (bool(a) ^ bool(b))


def test_noisy_coincident_rows_do_not_authorize_zero_gain_and_tied_leaf_is_false():
    data = rows([((1,), v) for v in (True, False)] * 16)
    forest = fit_forest(data, WorkBudget(10**6), seed=4)
    for tree in forest.trees:
        n, pos = tree.leaf({"f0": 1}, WorkBudget(1))
        assert tree.root.feature is None and n == 32 and pos == sum(data[int(rid)]["label"] for rid in tree.bootstrap)
    assert Tree(Node(2, 1), ("a", "b")).evaluate({}, WorkBudget(1)) is False


def test_min_leaf_and_depth_limits_apply_to_bootstrap_population():
    data = threshold_rows()
    for options in [dict(max_depth=0), dict(min_leaf=17)]:
        ledger = WorkBudget(10**6)
        forest = fit_forest(data, ledger, seed=5, **options)
        assert all(t.root.feature is None for t in forest.trees)
        assert "tree_fit_hist" not in ledger.counts
    data = rows([((int(i == 0),), i == 0) for i in range(32)])
    model = fit_forest(data, WorkBudget(10**6), seed=5)
    for tree in model.trees:
        minority = tree.bootstrap.count("0")
        assert (tree.root.feature is not None) == (2 <= minority <= 30)


@pytest.mark.parametrize(
    "damage", ["as_int", "bool_feature", "float", "missing", "unknown_field", "duplicate_id", "bin8"]
)
def test_strict_training_schema(damage):
    data = threshold_rows()
    if damage == "as_int":
        data[1]["label"] = 1
    elif damage == "bool_feature":
        data[1]["inputs"]["f1"] = True
    elif damage == "float":
        data[1]["inputs"]["f1"] = 2.0
    elif damage == "missing":
        data[1]["inputs"].pop("f0")
    elif damage == "unknown_field":
        data[1]["expected"] = True
    elif damage == "duplicate_id":
        data[1]["id"] = data[0]["id"]
    elif damage == "bin8":
        data[1]["inputs"]["f0"] = 8
    ledger = WorkBudget(1000)
    with pytest.raises(ValueError):
        fit_forest(data, ledger, seed=0)
    assert ledger.counts == {"tree_fit_validate": 64}


@pytest.mark.parametrize("options", [{"seed": True}, {"seed": 0, "max_depth": 5}, {"seed": 0, "min_leaf": 1}])
def test_strict_model_limits(options):
    with pytest.raises(ValueError):
        fit_forest(threshold_rows(), WorkBudget(10000), **options)


def test_partial_fit_fails_before_overspend_and_sink_failure_keeps_identity():
    data = threshold_rows()
    ledger = WorkBudget(64 + 32 + 1 + 32)  # room for schema, first sample and root labels only
    with pytest.raises(BudgetStop):
        fit_forest(data, ledger, seed=11)
    assert ledger.spent == ledger.limit and "tree_fit_hist" not in ledger.counts
    error = BudgetStop("sink after debit")

    def sink(snapshot):
        if "tree_fit_candidates" in snapshot["counts"]:
            raise error

    ledger = WorkBudget(10**6, on_charge=sink)
    with pytest.raises(BudgetStop) as caught:
        fit_forest(data, ledger, seed=11)
    assert caught.value is error and ledger.counts["tree_fit_candidates"] == 14 * 7 * 2


def test_unlabeled_prediction_validates_paid_branches_and_stops_on_budget():
    tree = fit_forest(threshold_rows(), WorkBudget(10**6), seed=11).trees[0]
    for inputs in [{"f0": True}, {"f0": 8}, {}, {"f0": 1, "label": False}]:
        ledger = WorkBudget(100)
        with pytest.raises(ValueError):
            tree.evaluate(inputs, ledger)
        assert ledger.counts["tree_predict_node"] == 1
    ledger = WorkBudget(2)
    with pytest.raises(BudgetStop):
        tree.leaf({"f0": 1}, ledger)
    assert ledger.counts == {"tree_predict_node": 1, "tree_predict_input": 1}


@pytest.mark.parametrize(
    "damage", ["unknown_key", "hash", "negative_count", "bad_threshold", "bool_version", "bad_bootstrap"]
)
def test_snapshot_requires_digest_and_structural_validity(damage):
    snap = deepcopy(fit_forest(threshold_rows(), WorkBudget(10**6), seed=11).to_dict())
    if damage == "unknown_key":
        snap["labels"] = []
    elif damage == "hash":
        snap["fingerprint"] = "wrong"
    elif damage == "negative_count":
        snap["trees"][0]["root"]["left"]["n"] = -1
    elif damage == "bad_threshold":
        snap["trees"][0]["root"]["threshold"] = False
    elif damage == "bool_version":
        snap["v"] = True
    elif damage == "bad_bootstrap":
        snap["trees"][0]["bootstrap"].pop()
    if damage != "hash":
        snap["fingerprint"] = digest({k: v for k, v in snap.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        Forest.from_dict(snap)


def test_each_root_matches_direct_fraction_reference_on_its_recorded_sample():
    from fractions import Fraction

    data = rows([((i % 8, (3 * i + i // 8) % 8), i % 5 in (1, 4) or i % 8 < 2) for i in range(64)])
    model = fit_forest(data, WorkBudget(10**6), seed=42, max_depth=1)

    def purity(samples):
        positives = sum(row["label"] for row in samples)
        return Fraction(positives**2 + (len(samples) - positives) ** 2, len(samples))

    for tree in model.trees:
        sample = [data[int(rid)] for rid in tree.bootstrap]
        best = purity(sample)
        winner = (None, None)
        for feature in ("f0", "f1"):
            for threshold in range(1, 8):
                low = [row for row in sample if row["inputs"][feature] < threshold]
                high = [row for row in sample if row["inputs"][feature] >= threshold]
                if min(len(low), len(high)) >= 2 and purity(low) + purity(high) > best:
                    best, winner = purity(low) + purity(high), (feature, threshold)
        assert (tree.root.feature, tree.root.threshold) == winner


@pytest.mark.parametrize("data", [[], threshold_rows() * 9, rows([(tuple(range(129)), False)])])
def test_bounds_checked_before_fit_blocks(data):
    budget = WorkBudget(10000)
    with pytest.raises(ValueError):
        fit_forest(data, budget, seed=1)
    assert budget.spent == 0


def test_unpaid_or_failed_vote_cannot_begin_tree_predictions():
    forest = fit_forest(threshold_rows(), WorkBudget(10**6), seed=11)
    insufficient = WorkBudget(2)
    with pytest.raises(BudgetStop):
        forest.predict({"f0": 0, "f1": 0}, insufficient)
    assert insufficient.spent == 0 and insufficient.counts == {}

    error = ValueError("vote persistence failed")

    def sink(snapshot):
        assert snapshot["counts"] == {"forest_vote": 3}
        raise error

    durable = WorkBudget(100, on_charge=sink)
    with pytest.raises(ValueError) as caught:
        forest.predict({"f0": 0, "f1": 0}, durable)
    assert caught.value is error and durable.counts == {"forest_vote": 3}
