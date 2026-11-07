"""All four directions share one real paid workflow."""
import runpy
from pathlib import Path


def test_end_to_end_four_directions():
    example = Path(__file__).resolve().parents[1] / "examples" / "40_decision_workflow.py"
    result = runpy.run_path(str(example))["main"]()
    assert result["taught_feasibility"] == {"economy": False, "balanced": True, "large": True}
    assert result["verified_solution"]["solution"] == (-19, "large")
    assert result["next_learned_proposal"] == "large"
    assert result["observed_external_outcome"] == {"action": "large", "demand": 7, "reward": 9}
    assert result["work"]["spent"] > 0
