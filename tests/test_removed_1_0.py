"""What 1.0 removed (CHANGELOG, 1.0.0 "Removed"): importing a removed module raises ModuleNotFoundError, the removed
methods and names are gone, and no doc page, nav entry or extra is left behind."""
import importlib
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GONE = ["solvi.many", "solvi.otel", "solvi.pytest_plugin", "solvi.segment_model",
        "solvi.aliases", "solvi.agents.pydantic_ai", "solvi.agents.langgraph",
        "solvi.agents.openai_agents", "solvi.extract_multi", "solvi.core.extract.multi"]


UNDER_OLD = [m for m in GONE if m.startswith("solvi.agents.")]      # removed from a 0.9 package that moved


@pytest.mark.parametrize("module", GONE)
def test_the_module_is_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("parent_loaded", [False, True])
@pytest.mark.parametrize("module", UNDER_OLD)
def test_a_removed_submodule_of_a_moved_package_is_gone_either_way(module, parent_loaded, monkeypatch):
    """The same ModuleNotFoundError whether or not the old parent (solvi.agents) was imported before, and whether or
    not its warning is an error: the removed child is refused before the parent is imported."""
    if parent_loaded:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module("solvi.agents")
    else:
        monkeypatch.delitem(sys.modules, "solvi.agents", raising=False)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("error")
        with pytest.raises(ModuleNotFoundError, match="removed in 1.0") as exc:
            importlib.import_module(module)
    assert exc.value.name == module and not seen
    assert parent_loaded or "solvi.agents" not in sys.modules


@pytest.mark.parametrize("module", UNDER_OLD)
def test_a_removed_submodule_is_gone_in_a_fresh_process(module):
    code = (f"import warnings; warnings.simplefilter('error')\n"     # solvi itself not imported yet
            f"try:\n    import {module}\nexcept ModuleNotFoundError as e:\n    print('gone', e.name)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == f"gone {module}", out.stderr[-2000:]


def test_the_names_are_gone():
    from solvi.core.plan import cost as strategy
    from solvi.core.system import Response
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
    assert "## 1.0.0 — " in (ROOT / "CHANGELOG.md").read_text()
