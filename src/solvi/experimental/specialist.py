"""Graduated in 1.1: the specialist contract is the stable solvi.core.specialist (its runs on the core TraceStorage).
This path re-exports every name, as the same objects, with the experimental warning, for one release: it is removed in
1.2. Import solvi.core.specialist."""
from __future__ import annotations

from .._deprecate import _warn_from_caller
from ..core.catalog import ExperimentalWarning
from ..core import specialist as _stable
from ..core.specialist import *  # noqa: F401,F403 — the graduated names

__all__ = list(_stable.__all__)

_warn_from_caller(f"{__name__} is experimental and graduated in 1.1: import solvi.core.specialist; this path is "
                  "removed in 1.2", ExperimentalWarning, skip=1)
