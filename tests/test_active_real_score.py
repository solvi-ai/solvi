"""Engineering-only synthetic inputs; these tests never load source archives."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import active_real_score as score  # noqa: E402
import active_real_trial as actor  # noqa: E402


@pytest.fixture
def data(monkeypatch):
    def rows(start, count):
        return [
            dict(id=i, inputs={"f0": i % 8, "f1": (i // 8) % 8}, label=(i % 8 < 3) ^ (i % 7 == 1))
            for i in range(start, start + count)
        ]

    result = dict(
        name="banknote",
        seed=5901,
        split="dev",
        source_sha="synthetic",
        pool=rows(0, 256),
        test=rows(256, 512),
        feature_names=["first", "second"],
        group_ids=dict(pool=list(range(256)), test=list(range(256, 768)), holdout=list(range(768, 1280))),
        preprocess_costs={"prep": 100},
        thresholds={},
        rawcounts={},
        quantized_test_rows_matching_pool=512,
    )
    monkeypatch.setattr(score, "prepare_dataset", lambda *args: result)
    return result


def simulate(data, arm):
    return actor.real_trial("banknote", None, 5901, "dev", arm, 128, prepared=data)


@pytest.mark.parametrize("arm", actor.ARMS)
def test_all_finite_reference_policies(data, arm):
    row = simulate(data, arm)
    result = score.audit_real(row, None)
    assert result["assigned"] == 512 and result["correct"] + result["wrong"] == 512
    assert result["fits"] == (1 if arm in ("FULL_LABEL", "RANDOM") else 9)
    assert result["labels"] == (256 if arm == "FULL_LABEL" else 48)
    assert result["units"] == row["budget"]["spent"] and result["auditorwork"] > 0


@pytest.mark.parametrize(
    "damage", ["fingerprint", "correct", "receipt", "foreign_id", "identity", "trace", "frozen", "bootstrap", "labels"]
)
def test_forged_records_do_not_score(data, damage):
    row = simulate(data, "RANDOM")
    if damage == "fingerprint":
        row["fits"][0]["model"]["fingerprint"] = "wrong"
    elif damage == "correct":
        row["predictions"][0]["value"] = int(row["predictions"][0]["value"])
    elif damage == "receipt":
        row["receipts"][0]["expected"] = int(row["receipts"][0]["expected"])
    elif damage == "foreign_id":
        row["receipts"][0]["id"] = str(data["test"][0]["id"])
    elif damage == "identity":
        row["price"] = True
    elif damage == "trace":
        row["trace"][0][1] += 1
    elif damage == "frozen":
        row["frozen"]["model"] = None
    elif damage == "bootstrap":
        row["fits"][0]["model"]["trees"][0]["bootstrap"].reverse()
    elif damage == "labels":
        row["receipts"].pop()
    with pytest.raises(ValueError):
        score.audit_real(row, None)


def test_decision_and_cache_provenance(data):
    row = simulate(data, "QBC_CAPPED")
    for key in ("decision", "hit", "test_input"):
        changed = deepcopy(row)
        if key == "decision":
            changed["decisions"][0]["fallback"] = 1
        elif key == "hit":
            changed["predict_calls"][0]["hit"] = not changed["predict_calls"][0]["hit"]
        else:
            changed["predict_calls"][0]["point_digest"] = "not in the pool"
        with pytest.raises(ValueError):
            score.audit_real(changed, None)


def test_partial_fit_preserves_only_paid_trace_and_no_frozen_model(data, monkeypatch):
    for module in (actor, score):
        monkeypatch.setattr(module, "LIMIT", 45000)
    row = simulate(data, "FULL_LABEL")
    assert len(row["fits"]) == 0 and row["frozen"]["reason"] == "budget" and len(row["receipts"]) == 256
    result = score.audit_real(row, None)
    assert result["fits"] == 0 and result["abstain"] == 512 and result["ba"] == [0, 1]


def test_interrupted_full_scan_retains_first_model_and_logged_partial_calls(data, monkeypatch):
    for module in (actor, score):
        monkeypatch.setattr(module, "LIMIT", 9500)
    row = simulate(data, "QBC_FULL")
    assert len(row["fits"]) == 1 and row["frozen"]["reason"] == "budget"
    assert row["predict_calls"] and row["frozen"]["model"] == row["fits"][-1]["model"]
    result = score.audit_real(row, None)
    assert result["fits"] == 1 and result["budget"] > 0
    assert result["budget"] + result["abstain"] + result["correct"] + result["wrong"] == 512


def test_missing_class_is_undefined_without_resampling(data):
    for row in data["test"]:
        row["label"] = True
    record = simulate(data, "FULL_LABEL")
    assert score.audit_real(record, None)["ba"] is None
    record["unexpected_gold"] = {}
    with pytest.raises(ValueError, match="schema"):
        score.audit_real(record, None)


def test_full_scan_charges_only_metadata_before_forecasts(data):
    row = simulate(data, "QBC_FULL")
    first = row["decisions"][0]
    remaining = len(data["pool"]) - first["query"]
    before = first["before"]["counts"].get("observation_plan", 0)
    # Four setup debits, metadata per option, three forecasts, final selection pass.
    assert first["after"]["counts"]["observation_plan"] - before == 4 + 5 * remaining

    trace = row["trace"]
    start = next(i for i, (kind, _, _) in enumerate(trace) if kind == "observation_plan")
    assert trace[start] == ["observation_plan", 1, 4 + remaining + 1]
    # Reintroduce the old width-dependent charge for metadata. Replay rejects
    # it at the first such debit, before either a forecast or a ledger check.
    trace[start : start + 1] = [
        ["observation_plan", 1, 4],
        ["observation_plan", len(data["feature_names"]) + 2, remaining],
        ["observation_plan", 1, 1],
    ]
    with pytest.raises(ValueError, match="expected observation_plan/1"):
        score.audit_real(row, None)
