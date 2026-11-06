import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import experience_v2_summary as summary
from experience_v2 import TrialLog, digest


def test_incomplete_or_corrupted_log_cannot_confirm_effect(tmp_path):
    manifest = {"pinned": "test"}
    TrialLog(tmp_path / "trials.jsonl", {"phase": "smoke", "manifest": digest(manifest)})
    (tmp_path / "trials.jsonl").write_text("")
    result = summary.summarize(tmp_path, "smoke", manifest)
    assert result["observed"] == 0 and result["expected"] == 54
    assert result["Hcost"] == result["Huseful"] == "INDETERMINATE"
    (tmp_path / "trials.jsonl").write_text(json.dumps({"hash": "forged"}) + "\n")
    with pytest.raises((ValueError, KeyError)):
        summary.summarize(tmp_path, "smoke", manifest)


def test_wire_ast_normalization_preserves_strict_constants():
    plain = {"ast": ["and", ["lt", ["input", "requested"], ["input", "available"]], ["not", ["input", "expired"]]]}
    wire = {"ast": ["and", ["lt", ["input", "w0"], ["input", "w1"]], ["not", ["input", "w2"]]]}
    assert summary.normalized_ast(plain, False) == summary.normalized_ast(wire, True)
    assert summary.canonical(summary.normalized_ast(["const", False], False)) != summary.canonical(
        summary.normalized_ast(["const", 0], False)
    )


def test_exact_tenth_bar_and_invalid_work_profile():
    assert summary.tenth_reduction(10, 9)
    assert not summary.tenth_reduction(100, 91)
    assert not summary.tenth_reduction(10, -1)
    assert not summary.tenth_reduction(True, 0)
    fake = {"cells": [{"cell": "A1", "pair": {}}] * 6}
    assert summary.validate_s1(fake) == ["invalid S1 cells/order"]
    cells = []
    for index, label in enumerate(("A1", "A2", "A3", "B1", "B2", "B3")):
        cells.append(
            {
                "cell": label,
                "pair": {
                    version: dict(start=index, end=index + 1, spent=1, remaining=min(99999, 200000 - index - 1))
                    for version in ("old", "new")
                },
            }
        )
    profile = dict(cells=cells, ledgers={version: dict(spent=6, counts={"work": 6}) for version in ("old", "new")})
    assert not summary.validate_s1(profile)
    profile["cells"][0]["pair"]["new"]["spent"] = -1
    assert summary.validate_s1(profile)


def test_standalone_summary_pins_local_lane_before_editable_install_without_pythonpath(tmp_path):
    import hashlib
    import os
    import subprocess

    lane = Path(__file__).resolve().parents[1]
    manifest = {'source': {'src/solvi/experimental/experience/programs.py': 'a' * 64}}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / 'trials.jsonl').write_text('')
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, str(lane / 'benchmarks/experience_v2_summary.py'),
                                '--out', str(tmp_path), '--phase', 'smoke', '--manifest', str(manifest_path)],
                               env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / 'summary.json').read_text())
    provenance = report['scorer_provenance']
    assert Path(provenance['solvi_origin']) == lane / 'src/solvi/__init__.py'
    assert provenance['actor_engine_sha256'] == 'a' * 64
    for relative, checksum in provenance['source_hashes'].items():
        assert checksum == hashlib.sha256((lane / relative).read_bytes()).hexdigest()
