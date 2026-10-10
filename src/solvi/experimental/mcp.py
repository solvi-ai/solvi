"""Graduated in 1.1: the MCP proxy with a guard is solvi.serve's (`run_proxy`, `Proxy`, `Upstream`, `UpstreamError`).
This path re-exports every name, as the same objects, with the experimental warning, for one release: it is removed in
1.2. Import solvi.serve."""
# ruff: noqa: F405 — __all__ names what the star import re-exports
from __future__ import annotations

from .._deprecate import _warn_from_caller
from .._mcp_proxy import *  # noqa: F401,F403 — the graduated names
from ..core.catalog import ExperimentalWarning

__all__ = ["CONTEXT_CHARS", "CONTEXT_MESSAGES", "Proxy", "run_proxy", "Upstream", "UpstreamError"]   # the stable module's names

_warn_from_caller(f"{__name__} is experimental and graduated in 1.1: import the proxy from solvi.serve; this path is "
                  "removed in 1.2", ExperimentalWarning, skip=1)
