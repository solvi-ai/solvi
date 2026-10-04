"""The Monitor protocol: something that watches the stream of decisions and says when it has changed.

    from solvi.core import Monitor

The built-in is `solvi.core.guarantees.drift.DriftMonitor`; `solvi.testing.conformance.check_monitor` checks a
monitor of your own."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Monitor(Protocol):
    """A watcher of the decision stream.

    You implement: `observe(decision, label=None, question=None)` → a report dict with at least "drift" (bool: the
    stream has changed) and "why" (a list of reasons, empty when nothing is flagged) — `decision` is a Response (with
    question=), a Result, a Decision or a dict; `flagged` → whether the last observation flagged a change;
    `reset()` → the stream starts again (the window and the flag are cleared; what was calibrated is kept).

    You get for free: `Dispatcher(monitor=...)` feeds it every System 1 response and turns its flag into a "drift"
    signal (the slow path checks System 1's answers until `reset_drift()`), recorded with every decision and taken as
    recorded on replay; the system report (`solvi.core.store.sysreport.system_report(monitor=...)`) runs a fresh one
    over a stored stream and says where it flagged and why.

    Stability: stable. `DriftMonitor` is stable to use, provisional to subclass."""

    def observe(self, decision: Any, label: Any = None, question: str | None = None) -> dict: ...

    @property
    def flagged(self) -> bool: ...

    def reset(self) -> None: ...


__all__ = ["Monitor"]
