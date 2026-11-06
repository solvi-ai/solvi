import hashlib
import importlib.util
import os
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "experience_v2_bench", Path(__file__).parents[1] / "benchmarks/experience_v2.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_roles_and_wire_sql_known_answers():
    for family, size in [("inventory", 125), ("eligibility", 50), ("schedule", 256)]:
        assert len(b.points(family)) == size
        r = b.roles(1, family)
        assert list(map(len, r.values())) == [8, 8, 24, 8]
        flat = [b.canonical(row) for group in r.values() for row in group]
        assert len(set(flat)) == 48
        c, w = b.Wire(1, family), b.Wire(1, family, True)
        row = r["proposal"][0]
        assert list(w.encode(row)) == list(reversed(["w" + str(i) for i in range(len(row))]))
        assert w.decode(w.encode(row)) == row
        assert w.identity("task", "a") != c.identity("task", "a")
        assert w.decode_receipt(w.receipt(b.sql_value(family, row))) == b.sql_value(family, row)
    assert b.sql_value("inventory", [4, 1, 2]) == 1
    assert b.sql_value("eligibility", [0, 1, False]) is True
    assert b.sql_value("schedule", [0, 0, 3, 1]) == 4


def test_nested_atomic_caps_and_log_resume(tmp_path):
    parent = b.WorkBudget(10)
    child = b.NestedBudget(parent, 6)
    child.charge("work", 5)
    with pytest.raises(b.BudgetStop):
        child.charge("work", 2)
    assert parent.spent == child.spent == 5
    parent.charge("other", 3)
    assert child.remaining == 0  # parent debit during child is inside its cap
    path = tmp_path / "trial.jsonl"
    log = b.TrialLog(path, {"source": "pinned"})
    log.append("engineering", "completed", {"spent": 8})
    resumed = b.TrialLog(path, {"source": "pinned"}, resume=True)
    assert resumed.finished == {"engineering"}
    with pytest.raises(ValueError):
        resumed.append("engineering", "completed", {})
    with pytest.raises(ValueError):
        b.TrialLog(path, {"source": "changed"}, resume=True)
    with pytest.raises(ValueError):
        b.require_engineering_seed(8101)


def test_s1_reference_small_and_exact_baseline_loading():
    baseline = os.environ.get("SOLVI_EXPERIENCE_V2_BASELINE")
    if baseline is None:
        pytest.skip("historical source requires SOLVI_EXPERIENCE_V2_BASELINE")
    root = Path(baseline)
    hashes = {
        leaf: hashlib.sha256((root / "src/solvi/experimental/experience" / (leaf + ".py")).read_bytes()).hexdigest()
        for leaf in ("programs", "acquisition")
    }
    old = b.load_baseline(root, hashes)
    paired = b.compare_s1(
        1,
        "threshold",
        (5, 1, 2, 1, 1, 5),
        1,
        old["acquisition"],
        {"old": b.WorkBudget(200000), "new": b.WorkBudget(200000)},
    )
    for result in paired.values():
        assert result["status"] == result["reference"]["status"]
        assert result["name"] == result["reference"]["name"]
        assert result["cost"] == result["reference"]["cost"]
    search = b.compare_s2(1, "inventory", old["programs"], limit=200)
    assert set(search) == {"old", "new"}
    assert all(r["spent"] <= 200 and r["status"] in ("completed", "budget") for r in search.values())
    hashes["programs"] = "0" * 64
    with pytest.raises(ValueError):
        b.load_baseline(root, hashes)
