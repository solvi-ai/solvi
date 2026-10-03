"""The hash chain of a trace: a record added after the flow's steps (an answer head's decision, a guarantee's check, the
plan), chained to the last one. Moved out of solvi.core.system in 1.0 (System re-exports it as `_append`), so that the
guarantees below the System add their records without importing it."""
from __future__ import annotations

from .runtime import vhash


def append(trace, rec, n_steps):
    """Add a record after the flow's steps (an answer head's decision), chained to the last one."""
    last = trace.records[-1] if trace.records else None
    rec.step = max(n_steps, last.step if last else 0) + 1
    rec.prev = last.hash if last else trace.init_hash
    rec.hash = vhash(rec.body())
    trace.records.append(rec)


__all__ = ["append"]
