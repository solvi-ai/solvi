"""benchmarks/changes_credit.py: the change pipeline on the credit policy. The full run needs the German Credit
applications (benchmarks/tasks/fetch.sh credit, then prepare.py) and takes minutes; here: without them it says how to
get them and exits 0, and its fixed inputs are in the repository."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_without_the_applications_it_says_how_to_fetch_them(tmp_path):
    env = {**os.environ, "STAND_DATA": str(tmp_path / "nothing"), "PYTHONWARNINGS": "ignore"}
    p = subprocess.run([sys.executable, str(ROOT / "benchmarks" / "changes_credit.py")], env=env, capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert "fetch.sh credit" in p.stdout and "prepare.py" in p.stdout


def test_the_policy_and_the_corrections_are_in_the_repository():
    policy = (ROOT / "benchmarks" / "tasks" / "credit" / "POLICY_v1_v2.md").read_text()
    assert "## Version 2" in policy and "Proposed rule `young`" in policy
    cor = json.loads((ROOT / "benchmarks" / "tasks" / "credit" / "corrections.json").read_text())
    assert len(cor["corrections"]) == 20 and cor["counts"]["expected"] == {
        "refused by impact": 8, "held for the person, then promoted after the ruling": 7, "admitted (promoted)": 5}
