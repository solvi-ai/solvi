import json
import math
import statistics

import pytest
from benchmarks._experience_aggregate import (
    ARMS,
    FAMILIES,
    SEEDS,
    SPEC,
    TCRIT,
    choose_comparators,
    collect,
    finalaggregate,
    paired_interval,
)


def trial(seed, family, arm, stratum="id", success=128, cost=100):
    denominator = 64 if stratum == "dev" else 128
    return dict(
        spec=SPEC,
        seed=seed,
        family=family,
        arm=arm,
        stratum=stratum,
        limit=65536,
        observation_price=8,
        denominator=denominator,
        success=min(success, denominator),
        applications=32 if arm == "FULL" else 0,
        weighted_total=8 * cost,
        raw_vector={"compute": cost},
        weighted_vector={"compute": 8 * cost},
        accounting_scale=8,
        exact_accounting_total={"numerator": 8 * cost, "denominator": 8},
        hard_violations=0,
        learning={
            "record": {"accepted": True},
            "candidate": {"program": {"ast": ["sub", ["sub", ["input", "x"], ["input", "y"]], ["const", 1]]}},
        },
    )


def dev():
    return [
        trial(s, f, a, "dev", cost=10 if a == "FORMULA_CACHE" else 100)
        for s in SEEDS
        for f in FAMILIES
        for a in ARMS[:5]
    ]


def final():
    return [
        trial(
            s,
            f,
            a,
            success=120 if a in ("FIXED", "RANDOM") else 128,
            cost=50 if a == "FULL" else 10 if a == "FORMULA_CACHE" else 100,
        )
        for s in SEEDS
        for f in FAMILIES
        for a in ARMS
    ]


def test_interval_uses_twelve_seed_means():
    values = list(range(12))
    result = paired_interval(values)
    assert result["mean"] == 5.5
    assert result["lower"] == pytest.approx(5.5 - TCRIT * statistics.stdev(values) / math.sqrt(12))
    with pytest.raises(ValueError):
        paired_interval(values * 128)


def test_stronger_formula_is_locked_and_negative():
    lock = choose_comparators(dev())
    assert lock["comparators"]["resource"]["strong"] == "FORMULA_CACHE"
    result = finalaggregate(final(), lock)
    assert result["families"]["resource"]["overall"]["relative_cost_saving"]["mean"] == -4
    assert result["scientific_verdict"] == "NOT_CONFIRMED"
    assert not result["complete_grid"]


def test_missing_dev_seed_and_final_arm_fail_closed():
    with pytest.raises(ValueError):
        choose_comparators(dev()[1:])
    with pytest.raises(ValueError):
        finalaggregate([r for r in final() if r["arm"] != "NO_CALL"], choose_comparators(dev()))


def test_duplicate_semantic_keys_rejected(tmp_path):
    row = trial(1101, "resource", "FULL")
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps({"kind": "trial", "key": key, "result": row}) for key in ("one", "two")))
    with pytest.raises(ValueError):
        collect([path])


def test_zero_control_cost_undefined_and_negative_cost_rejected():
    rows = final()
    for row in rows:
        if row["arm"] == "NO_CALL":
            row.update(
                weighted_total=0,
                raw_vector={},
                weighted_vector={},
                exact_accounting_total={"numerator": 0, "denominator": 8},
            )
    result = finalaggregate(rows, choose_comparators(dev()))
    assert result["families"]["resource"]["H_operation"]["relative_cost_saving"] is None
    rows[0]["weighted_total"] = -1
    with pytest.raises(ValueError):
        finalaggregate(rows, choose_comparators(dev()))


def test_missing_safety_or_learning_cannot_pass():
    rows = final()
    for row in rows:
        row.pop("hard_violations")
        row.pop("learning")
    result = finalaggregate(rows, choose_comparators(dev()))
    assert result["safety_status"] == "safety_not_audited"
    assert not result["families"]["resource"]["H_operation"]["passed"]


def test_cost_vector_integrity():
    rows = dev()
    rows[0]["raw_vector"] = {"action": 100}
    with pytest.raises(ValueError):
        choose_comparators(rows)


def test_paired_savings_not_ratio_of_averages():
    rows = final()
    for row in rows:
        if row["arm"] in ("FULL", "NO_CALL"):
            cost = 10 if row["seed"] % 2 else 100
            if row["arm"] == "FULL":
                cost = 5 if row["seed"] % 2 else 100
            row.update(
                weighted_total=8 * cost,
                raw_vector={"compute": cost},
                weighted_vector={"compute": 8 * cost},
                exact_accounting_total={"numerator": 8 * cost, "denominator": 8},
            )
    result = finalaggregate(rows, choose_comparators(dev()))
    assert result["families"]["resource"]["H_operation"]["relative_cost_saving"]["mean"] == 0.25


def test_partial_grid_cannot_promote_primary_pass():
    development = dev()
    for row in development:
        row["success"] = 64 if row["arm"] == "ACTIVE" else 60
    lock = choose_comparators(development)
    result = finalaggregate(final(), lock)
    assert all(test["passed"] for family in result["families"].values() for test in family.values())
    assert result["scientific_verdict"] == "NOT_CONFIRMED"
    assert result["actual_trials"] == 168
    assert result["missing_trials"] == 4368
    assert not result["dev_complete_grid"]


def test_exact_eighth_prices_and_budget_validation():
    from benchmarks._experience_aggregate import _validate

    row = trial(1101, "resource", "FULL")
    row["raw_vector"] = {
        name: 1 for name in ("compute", "action", "observation", "observe_measure", "observe_diagnose", "observe_noise")
    }
    row["weighted_vector"] = dict(
        compute=8, action=64, observation=64, observe_measure=64, observe_diagnose=32, observe_noise=8
    )
    row["weighted_total"] = 240
    row["exact_accounting_total"]["numerator"] = 240
    assert _validate([row]) == [row]
    row["weighted_total"] = row["limit"] * 8 + 8
    row["raw_vector"] = {"compute": row["weighted_total"] // 8}
    row["weighted_vector"] = {"compute": row["weighted_total"]}
    row["exact_accounting_total"]["numerator"] = row["weighted_total"]
    with pytest.raises(ValueError, match="budget"):
        _validate([row])


@pytest.mark.parametrize(
    "field,value",
    [("seed", 1101.0), ("limit", 65536.0), ("observation_price", 8.0), ("denominator", 1), ("hard_violations", False)],
)
def test_strict_coordinate_and_count_types(field, value):
    from benchmarks._experience_aggregate import _validate

    row = trial(1101, "resource", "FULL")
    row[field] = value
    with pytest.raises(ValueError):
        _validate([row])


def test_zero_difference_has_zero_lower_bound_and_no_info_pass():
    assert paired_interval([0] * 12)["lower"] == 0
    rows = final()
    for row in rows:
        row["success"] = 128
    result = finalaggregate(rows, choose_comparators(dev()))
    assert not result["families"]["resource"]["H_info"]["passed"]


def test_cli_partial_dev_lock_rejected_for_final(tmp_path):
    from benchmarks._experience_aggregate import main

    path = tmp_path / "dev.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in dev()))
    main(["--dev", str(path), "--output", str(tmp_path / "out")])
    lock_path = tmp_path / "out" / "comparator_lock.json"
    lock = json.loads(lock_path.read_text())
    assert lock["dev_raw_files_sha256"]
    assert lock["bound_files_sha256"]
    with pytest.raises(ValueError, match="1512"):
        main(
            ["--final", str(tmp_path / "not_read.jsonl"), "--lock", str(lock_path), "--output", str(tmp_path / "final")]
        )


def test_episode_identity_flags_and_counts():
    from benchmarks._experience_aggregate import _validate

    row = trial(1101, "resource", "SOLVER")
    row["rows"] = [dict(task_id=str(i), success=True, applied=False, hard_violation=False) for i in range(128)]
    assert _validate([row])
    row["rows"][0]["success"] = 1
    with pytest.raises(ValueError, match="boolean"):
        _validate([row])
    row["rows"][0]["success"] = True
    row["rows"][0]["task_id"] = "1"
    with pytest.raises(ValueError, match="duplicate"):
        _validate([row])
