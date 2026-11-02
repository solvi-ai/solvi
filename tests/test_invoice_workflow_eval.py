import json
import math

import pytest

from benchmarks.invoice_workflow_eval import evaluate, load_rows, metrics, select_threshold, upper_error


def row(i, split="calibration", answer="approve", confidence=1.0, group=None, cost=1.0):
    return {"id": str(i), "group": group or f"g{i}", "split": split, "gold": "approve",
            "arms": {a: {"answer": answer, "confidence": confidence, "latency_s": 2., "cost_usd": cost}
                     for a in ("direct", "plain", "solvi")}}


def write(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_one_sided_exact_known_values():
    assert upper_error(0, 10, .05) == pytest.approx(1 - .05 ** .1)
    assert upper_error(10, 10, .05) == 1
    assert upper_error(0, 0, .05) is None


def test_abstentions_cost_and_empty_answers():
    rows = [row(1, answer=None, cost=2), row(2, answer="", cost=3), row(3, answer="reject", cost=4)]
    m = metrics(rows, "solvi", .5, .05)
    assert (m["answered"], m["errors"], m["coverage"], m["joint_risk"]) == (1, 1, 1 / 3, 1 / 3)
    assert m["total_cost_usd"] == 9
    empty = metrics(rows[:2], "solvi", .5, .05)
    assert empty["conditional_error"] is empty["upper_error"] is None


def test_validation_rejects_leakage_duplicates_and_nan(tmp_path):
    a, b = row(1), row(2, "eval")
    b["group"] = a["group"]
    with pytest.raises(ValueError, match="group overlap"):
        load_rows(write(tmp_path, [a, b]))
    b["group"], b["id"] = "other", a["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        load_rows(write(tmp_path, [a, b]))
    b["id"], b["texthash"], a["texthash"] = "2", "same", "same"
    with pytest.raises(ValueError, match="texthash overlap"):
        load_rows(write(tmp_path, [a, b]))
    del b["texthash"]
    b["arms"]["solvi"]["confidence"] = math.nan
    with pytest.raises(ValueError, match="confidence"):
        load_rows(write(tmp_path, [a, b]))


def test_selection_uses_calibration_only_and_eval_gate():
    calibration = [row(i) for i in range(160)] + [row(160, answer="reject", confidence=.5)]
    ev = [row(i, "eval", answer="reject") for i in range(200, 270)]
    assert select_threshold(calibration, "solvi") == .7
    report = evaluate(calibration + ev, "hash", bootstrap_samples=20)
    assert report["arms"]["solvi"]["threshold"] == .7
    assert not report["arms"]["solvi"]["error_gate_passed"]
    assert not report["paired_coverage_vs_baseline"]["plain"]["available"]


def test_empty_selection_fails_gate():
    report = evaluate([row(1, answer=None), row(2, "eval", answer=None)], "hash", bootstrap_samples=20)
    m = report["arms"]["solvi"]
    assert m["threshold"] is None and not m["error_gate_passed"]
    assert m["total_cost_usd"] == 1


def test_paired_bootstrap_and_deterministic_cli(tmp_path):
    import subprocess
    import sys

    rows = [row(i) for i in range(320)] + [row(i, "eval", group=f"eval{i // 2}") for i in range(400, 640)]
    for r in rows:
        r["arms"]["plain"]["answer"] = None if int(r["id"]) % 2 else "approve"
    report = evaluate(rows, "hash", bootstrap_samples=100, seed=141)
    comparison = report["paired_coverage_vs_baseline"]["plain"]
    assert comparison["available"] and comparison["coverage_difference"] == .5
    assert comparison["lower_one_sided_bound"] == .5
    assert report["manifest"]["within_group_non_independence_possible"]
    source = write(tmp_path, rows)
    out = tmp_path / "report.json"
    cmd = [sys.executable, "benchmarks/invoice_workflow_eval.py", "--rows", str(source), "--out", str(out)]
    subprocess.run(cmd, check=True)
    first = out.read_bytes()
    subprocess.run(cmd, check=True)
    assert out.read_bytes() == first
    assert len(json.loads(first)["manifest"]["input_sha256"]) == 64


def test_uncertified_raw_metrics_and_primary_acceptance():
    calibration = [row(i, answer="reject") for i in range(3)]
    evaluation = [row(10, "eval", answer="approve", group="same"),
                  row(11, "eval", answer=None, group="same")]
    report = evaluate(calibration + evaluation, "hash", bootstrap_samples=10)
    arm = report["arms"]["solvi"]
    assert arm["threshold"] is None
    raw = arm["ungated_secondary"]
    assert raw["accuracy_on_answered"] == 1
    assert raw["accuracy_all_attempts"] == .5
    assert not raw["certified"]
    assert not report["manifest"]["inferential_validity"]
    assert report["manifest"]["independence_assumption"] == "unverified"
    assert not report["primary_acceptance"]["passed"]
    assert any("1000" in reason for reason in report["primary_acceptance"]["reasons"])
    assert any("share" in reason for reason in report["primary_acceptance"]["reasons"])


def test_variant_slices_preserve_failures_and_costs():
    a, b = row(10, "eval", answer=None, cost=2), row(11, "eval", cost=3)
    a["variants"] = {"number_format": "german"}
    b["variants"] = {"number_format": "english"}
    report = evaluate([row(1), a, b], "hash", bootstrap_samples=10)
    slices = report["arms"]["solvi"]["variant_slices"]["number_format"]
    assert slices["german"]["interpretation"] == "descriptive_only"
    assert slices["german"]["ungated_secondary"]["answered"] == 0
    assert slices["german"]["ungated_secondary"]["total_cost_usd"] == 2
    assert slices["english"]["ungated_secondary"]["answered"] == 1
