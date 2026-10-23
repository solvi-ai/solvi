"""Graduated in 1.1: the specialist contract is the stable solvi.core.specialist (its runs on the core TraceStorage).
This path re-exports every name, as the same objects, with the experimental warning (its removal, planned for 1.2,
is now in a future release). Import solvi.core.specialist."""
# ruff: noqa: F405 — __all__ names what the star import re-exports
from __future__ import annotations

from .._deprecate import _warn_from_caller
from ..core.catalog import ExperimentalWarning
from ..core.specialist import *  # noqa: F401,F403 — the graduated names

__all__ = ["Agreed", "align", "Alignment", "BLOCKED", "box_iou", "canonical", "CHANGED", "Checked", "DROPPED",
           "interval_overlap", "Issue", "KIND", "Reading", "Replay", "replay_runs", "Run", "SEVERITIES", "sha256",
           "Specialist", "Trace", "WARNING"]   # the stable module's names

_warn_from_caller(f"{__name__} is experimental and graduated in 1.1: import solvi.core.specialist; this path is "
                  "removed in a future release", ExperimentalWarning, skip=1)
