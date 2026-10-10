"""The MCP proxy with a guard graduated into solvi.serve in 1.1 (measured against the real filesystem server: the X2
run, its script benchmarks/mcp_proxy_filesystem.py): solvi.serve exports it, `solvi serve --guard --upstream` imports
nothing experimental, and the old path solvi.experimental.mcp re-exports the same objects with the experimental warning
for one release."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("CONTEXT_CHARS", "CONTEXT_MESSAGES", "Proxy", "run_proxy", "Upstream", "UpstreamError")


def test_solvi_serve_exports_the_proxy():
    import solvi.serve as serve
    for name in NAMES:
        assert name in serve.__all__ and getattr(serve, name) is not None, name
    assert serve.run_proxy.__module__ == "solvi._mcp_proxy"


def test_the_old_path_re_exports_the_same_objects_with_the_experimental_warning():
    import solvi.experimental
    import solvi.serve as serve
    from solvi.core import ExperimentalWarning
    sys.modules.pop("solvi.experimental.mcp", None)
    vars(solvi.experimental).pop("mcp", None)
    with pytest.warns(ExperimentalWarning, match="graduated in 1.1.*solvi.serve"):
        old = importlib.import_module("solvi.experimental.mcp")
    for name in NAMES:
        assert getattr(old, name) is getattr(serve, name), name
    assert "mcp" not in solvi.experimental.STATUS


def test_serve_with_a_guard_and_an_upstream_loads_nothing_experimental():
    src = (ROOT / "src" / "solvi" / "serve.py").read_text() + (ROOT / "src" / "solvi" / "_mcp_proxy.py").read_text()
    assert "experimental" not in src.replace("experimental pieces", "")
    code = ("import sys, warnings\nwarnings.simplefilter('error')\nimport solvi.serve\n"
            "from solvi.serve import run_proxy, Proxy\n"
            "assert not [m for m in sys.modules if m.startswith('solvi.experimental.')]\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]


def test_the_docs_moved_with_it():
    exp = (ROOT / "docs" / "experimental.md").read_text()
    assert "## `mcp`" not in exp
    guide = (ROOT / "docs" / "guide.md").read_text()
    serving = guide.split("## Serving: HTTP, MCP and System One")[1].split("\n## ")[0]
    assert "### An MCP proxy" in serving
    nav = (ROOT / "mkdocs.yml").read_text()
    assert "solvi.experimental.mcp: api/mcp.md" not in nav
    from solvi.core.provenance import fp_module
    assert fp_module("solvi._mcp_proxy", "Proxy") == "solvi.agents.mcp"
