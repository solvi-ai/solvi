"""What 1.0 removed (CHANGELOG, 1.0.0 "Removed"): importing a removed module raises ModuleNotFoundError, the removed
methods and names are gone, and no doc page, nav entry or extra is left behind."""
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GONE = ["solvi.many", "solvi.otel", "solvi.pytest_plugin", "solvi.counterfactual", "solvi.segment_model",
        "solvi.aliases", "solvi.agents.pydantic_ai", "solvi.agents.langgraph", "solvi.agents.openai_agents"]


@pytest.mark.parametrize("module", GONE)
def test_the_module_is_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_the_names_are_gone():
    from solvi import strategy
    from solvi.system import Response
    assert not hasattr(Response, "counterfactual") and not hasattr(strategy, "ModelStrategist")


def test_nothing_is_left_behind():
    nav = (ROOT / "mkdocs.yml").read_text()
    index = (ROOT / "docs" / "api" / "index.md").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()
    for mod in GONE:
        name = mod.split(".", 1)[1]
        assert not (ROOT / "docs" / "api" / f"{name}.md").exists(), mod
        assert f"{mod}:" not in nav and f"[`{mod}`]" not in index, mod
    assert "pytest11" not in pyproject
    for extra in ("otel", "pydantic-ai", "langgraph", "openai-agents"):
        assert f"\n{extra} = " not in pyproject, extra
    assert not (ROOT / "spaces" / "documents-server").exists() and not (ROOT / "tools" / "smoke_decide.py").exists()
    assert "## 1.0.0 — unreleased" in (ROOT / "CHANGELOG.md").read_text()
