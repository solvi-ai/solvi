"""Graduated in 1.1: counterfactual explanations are the stable solvi.core.explain (`search`, `Counterfactual`,
`Counterfactuals`, `Change`, `Rerun`). This path re-exports every name, as the same objects, with the experimental
warning (its removal, planned for 1.2, is now in a future release). Import solvi.core.explain."""
# ruff: noqa: F405 — __all__ names what the star import re-exports
from __future__ import annotations

from .._deprecate import _warn_from_caller
from ..core.catalog import ExperimentalWarning
from ..core.explain import *  # noqa: F401,F403 — the graduated names

__all__ = ["Change", "Counterfactual", "Counterfactuals", "Rerun", "search"]   # the stable module's names

_warn_from_caller(f"{__name__} is experimental and graduated in 1.1: import solvi.core.explain; this path is removed in "
                  "a future release", ExperimentalWarning, skip=1)
