"""Strict portable values and a shared work ledger; no model or external effect."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping

from ...core.costs import BudgetStop


def _check(value, depth=0):
    if depth > 64:
        raise ValueError("value exceeds the portable nesting limit")
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is list:
        for item in value:
            _check(item, depth + 1)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _check(item, depth + 1)
        return
    raise ValueError("experience values must be JSON null, bool, int, string, list or string-keyed dict")


def canonical(value) -> str:
    """Exact typed comparison: bool and int differ; floats/NaN are unsupported."""
    _check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def portable(value):
    return json.loads(canonical(value))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value, name="identifier") -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def natural(value, name="count", *, positive=False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


class WorkBudget:
    """Integer work units shared by reasoning, observations, search and execution.

    The host declares the units, including observation prices. Counts retain the
    category vector; they are not dollars or a promise about wall time. A sink
    runs after debit and before its caller executes the paid operation. A
    durable host can bind this sink to KnowledgeStore.note. Sink failure stops
    the caller; recovery must reconcile the recorded ledger, never reset it.
    """

    def __init__(self, limit: int, *, on_charge: Callable[[dict], None] | None = None):
        self.limit = natural(limit, "limit")
        if on_charge is not None and not callable(on_charge):
            raise TypeError("on_charge must be callable")
        self._counts: dict[str, int] = {}
        self._spent = 0
        self.on_charge = on_charge

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return self.limit - self._spent

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def charge(self, kind: str, units: int = 1) -> None:
        identifier(kind, "cost kind")
        natural(units, "work units")
        if units > self.remaining:
            raise BudgetStop(f"{kind}: {units} units exceed remaining {self.remaining}")
        if not units:
            return
        self._counts[kind] = self._counts.get(kind, 0) + units
        self._spent += units
        if self.on_charge is not None:
            self.on_charge(self.snapshot())

    def snapshot(self) -> dict:
        body = {"v": 1, "limit": self.limit, "spent": self.spent, "counts": self.counts}
        return {**body, "digest": digest(body)}

    @classmethod
    def restore(cls, snapshot: Mapping, *, on_charge=None):
        s = portable(dict(snapshot))
        if set(s) != {"v", "limit", "spent", "counts", "digest"} or type(s["v"]) is not int or s["v"] != 1:
            raise ValueError("unsupported work ledger")
        body = {k: v for k, v in s.items() if k != "digest"}
        if s["digest"] != digest(body) or type(s["counts"]) is not dict:
            raise ValueError("corrupt work ledger")
        ledger = cls(s["limit"], on_charge=on_charge)
        for name, units in s["counts"].items():
            identifier(name, "cost kind")
            natural(units, "recorded units", positive=True)
        natural(s["spent"], "spent")
        if sum(s["counts"].values()) != s["spent"] or s["spent"] > ledger.limit:
            raise ValueError("inconsistent work ledger")
        ledger._counts = dict(s["counts"])
        ledger._spent = s["spent"]
        return ledger
