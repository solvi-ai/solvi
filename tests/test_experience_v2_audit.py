import copy
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from experience_v2_audit import audit_trial, evaluate, truth
from experience_v2_strategies import run_trial


@pytest.mark.parametrize("wire", [False, True])
def test_independent_audit_complete_checked_control(tmp_path, wire):
    result = run_trial(tmp_path / "trial", 123, "eligibility", "FORMULA_CHECKED", wire=wire)
    audit = audit_trial(tmp_path / "trial", result)
    assert not audit["errors"], audit["errors"]
    assert (audit["correct_R"], audit["correct_H"]) == (24, 8)
    assert audit["useful_R"] == audit["useful_H"] == 0
    assert audit["verification_work"] > 50


def test_forged_flags_do_not_create_credit(tmp_path):
    result = run_trial(tmp_path / "trial", 123, "inventory", "FIXED_MEASURE")
    for entry in result["transfers"]:
        entry["useful"] = True
        entry["receipt"]["value"]["ok"] = False
    audit = audit_trial(tmp_path / "trial", result)
    assert audit["errors"]
    assert audit["useful_R"] == audit["useful_H"] == 0


def test_wrong_recorded_ast_and_duplicate_effect_are_errors(tmp_path):
    result = run_trial(tmp_path / "trial", 123, "eligibility", "FORMULA_CHECKED")
    forged = copy.deepcopy(result)
    candidate = next(iter(forged["registry"]["candidates"].values()))
    candidate["program"]["ast"] = ["const", False]
    audit = audit_trial(tmp_path / "trial", forged)
    assert audit["errors"]
    with sqlite3.connect(tmp_path / "trial" / "environment.db") as conn:
        conn.execute("UPDATE tabular_tasks SET effects=2 WHERE effects=1")
    assert any("effects" in e for e in audit_trial(tmp_path / "trial", result)["errors"])


def test_rejected_and_budget_trials_stay_in_denominator_without_errors(tmp_path):
    for arm, limit in [("NEW_EMPIRICAL", 200000), ("NEW_CHECKED", 3)]:
        path = tmp_path / arm
        result = run_trial(path, 123, "eligibility", arm, limit=limit)
        assert result["status"] in ("rejected", "budget")
        audit = audit_trial(path, result)
        assert not audit["errors"], audit["errors"]
        assert audit["correct_R"] == audit["correct_H"] == 0
        assert audit["useful_R"] == audit["useful_H"] == 0


def test_simulators_and_strict_eager_interpreter():
    assert truth("inventory", [4, 1, 2]) == 1
    assert truth("eligibility", [0, 1, False]) is True
    assert truth("schedule", [0, 0, 3, 1]) == 4
    grammar = dict(inputs=["x"], constants=[0, 1], operators=["if", "add"], max_depth=4, max_nodes=10, max_abs=1)
    with pytest.raises(ValueError, match="overflow"):
        evaluate(["if", ["const", True], ["const", 0], ["add", ["const", 1], ["const", 1]]], {}, grammar, [0])
    with pytest.raises(ValueError, match="scalar type"):
        evaluate(["add", ["const", True], ["const", 0]], {}, grammar, [0])


def test_claimed_useful_flags_never_credit_fixed_control(tmp_path):
    result = run_trial(tmp_path / "trial", 123, "schedule", "FIXED_MEASURE")
    for entry in result["transfers"]:
        entry["useful"] = True
    audit = audit_trial(tmp_path / "trial", result)
    assert not audit["errors"], audit["errors"]
    assert (audit["correct_R"], audit["correct_H"]) == (24, 8)
    assert audit["useful_R"] == audit["useful_H"] == 0


def test_formula_metadata_spoof_cannot_create_learned_credit(tmp_path):
    result = run_trial(tmp_path / "trial", 123, "eligibility", "FORMULA_CHECKED")
    result["arm"] = "NEW_CHECKED"
    result["search"]["reason"] = "found"
    audit = audit_trial(tmp_path / "trial", result)
    assert audit["errors"]
    assert audit["useful_R"] == audit["useful_H"] == 0


def test_duplicate_cross_task_receipt_cannot_credit_twice(tmp_path):
    result = run_trial(tmp_path / "trial", 123, "eligibility", "FORMULA_CHECKED")
    result["transfers"][1]["receipt"] = copy.deepcopy(result["transfers"][0]["receipt"])
    audit = audit_trial(tmp_path / "trial", result)
    assert any("cross-task" in error for error in audit["errors"])
    assert audit["correct_R"] + audit["correct_H"] == 31


def test_postcommit_budget_is_reconciled_and_physically_counted(tmp_path):
    result = run_trial(tmp_path / "trial", 123, "eligibility", "FORMULA", limit=20)
    assert result["status"] == "budget"
    committed = [e for e in result["transfers"] if e["status"] == "committed_unlinked"]
    assert len(committed) == 1
    assert committed[0]["receipt"] and not committed[0]["useful"]
    audit = audit_trial(tmp_path / "trial", result)
    assert not audit["errors"], audit["errors"]
    assert audit["correct_R"] + audit["correct_H"] == 1
    assert audit["useful_R"] + audit["useful_H"] == 0
    committed[0]["receipt"] = None
    omitted = audit_trial(tmp_path / "trial", result)
    assert any("bijection" in e for e in omitted["errors"])


def test_real_paid_new_search_trace_audits_even_when_admission_rejects(tmp_path):
    result = run_trial(tmp_path / "trial", 123, "eligibility", "NEW_CHECKED")
    assert result["search"]["reason"] == "found"
    audit = audit_trial(tmp_path / "trial", result)
    assert not audit["errors"], audit["errors"]
