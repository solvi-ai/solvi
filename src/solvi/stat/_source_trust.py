"""Time-local descriptive source reliability, not a truth certificate or e-process.

The caller authenticates source identities and supplies independently observed loss.
An assertion is a particular statement at a particular time; a repeat is a NEW
accountable assertion. Labels never propagate to ancestors or descendants. The
window uses assertion time, so delayed labels cannot rejuvenate old evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

__all__ = ["SourceTrust", "SourceOpinion"]


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _key(value):
    if not isinstance(value, str) or not value:
        raise ValueError("identifiers must be nonempty strings")
    return value


@dataclass(frozen=True)
class SourceOpinion:
    source: str
    role: str
    now: float
    support: str
    reliability: float
    n: int
    pending: int
    expired: int
    mean_loss: float | None
    last_asserted_at: float | None
    staleness: float | None


class SourceTrust:
    """Explicit sliding time window with a fixed Beta prior and bounded losses.

    `support` is 'unknown' until `min_labels` recent labels exist, otherwise
    'supported'; neither value certifies truth. Counts refer to assertions, not
    independent samples. A prolific repeater can supply correlated assertions.
    Times are caller-defined finite monotone known times in consistent units.
    """

    def __init__(self, *, window: float, prior=(1.0, 1.0), min_labels: int = 1):
        window = _number(window, "window")
        if window <= 0:
            raise ValueError("window must be positive")
        if not isinstance(prior, (list, tuple)) or len(prior) != 2:
            raise ValueError("prior must contain two positive parameters")
        prior = tuple(_number(x, "prior") for x in prior)
        if min(prior) <= 0:
            raise ValueError("prior must be positive")
        if type(min_labels) is not int or min_labels < 1:
            raise ValueError("min_labels must be a positive integer")
        self.window, self.prior, self.min_labels = window, prior, min_labels
        self._clock = None
        self._assertions = {}
        self._labels = {}

    def _time(self, now):
        now = _number(now, "now")
        if self._clock is not None and now < self._clock:
            raise ValueError("known time cannot go backwards")
        return now

    def assert_statement(self, assertion_id: str, *, statement: str, source: str,
                         asserted_at: float, now: float, parent: str | None = None):
        """Register an origin or repeat; parent must name the SAME statement.

        A changed or time-indexed proposition needs its own statement identifier.
        Registration alone incurs no loss. No signature implies correctness.
        """
        assertion_id, statement, source = map(_key, (assertion_id, statement, source))
        now, asserted_at = self._time(now), _number(asserted_at, "asserted_at")
        if assertion_id in self._assertions or asserted_at > now:
            raise ValueError("duplicate assertion or assertion from the future")
        if parent is not None:
            _key(parent)
            ancestor = self._assertions.get(parent)
            if ancestor is None or ancestor['statement'] != statement or ancestor['asserted_at'] > asserted_at:
                raise ValueError("repeat requires an earlier parent of the same statement")
        row = dict(statement=statement, source=source, asserted_at=asserted_at, known_at=now, parent=parent)
        self._assertions[assertion_id] = row
        self._clock = now

    def label(self, assertion_id: str, *, loss: float, now: float):
        """Attach one independently observed loss to exactly one accountable assertion."""
        assertion_id = _key(assertion_id)
        now, loss = self._time(now), _number(loss, "loss")
        if not 0 <= loss <= 1:
            raise ValueError("loss must be in [0, 1]")
        if assertion_id not in self._assertions or assertion_id in self._labels:
            raise ValueError("unknown or already labeled assertion")
        self._labels[assertion_id] = dict(loss=loss, known_at=now)
        self._clock = now

    def opinion(self, source: str, *, now: float, role: str = 'all') -> SourceOpinion:
        """Read without advancing the registry clock; role separates origin/repeater."""
        source, now = _key(source), self._time(now)
        if role not in ('all', 'origin', 'repeater'):
            raise ValueError("role must be all, origin or repeater")
        rows = [(k, v) for k, v in self._assertions.items() if v['source'] == source
                and (role == 'all' or (v['parent'] is None) == (role == 'origin'))]
        recent = [(k, v) for k, v in rows if now - self.window < v['asserted_at'] <= now]
        losses = [self._labels[k]['loss'] for k, _ in recent if k in self._labels]
        n, total = len(losses), math.fsum(losses)
        last = max((v['asserted_at'] for _, v in rows), default=None)
        a, b = self.prior
        scale = max(a, b, n, 1.0)
        reliability = (a / scale + (n - total) / scale) / (a / scale + b / scale + n / scale)
        return SourceOpinion(source, role, now, 'supported' if n >= self.min_labels else 'unknown',
                             reliability, n, len(recent) - n,
                             len(rows) - len(recent), total / n if n else None,
                             last, now - last if last is not None else None)

    def snapshot(self) -> dict:
        # JSON roundtrip also detaches all caller-visible containers.
        return json.loads(json.dumps(dict(version=1, window=self.window, prior=self.prior,
                         min_labels=self.min_labels, clock=self._clock,
                         assertions=self._assertions, labels=self._labels), allow_nan=False))

    @classmethod
    def restore(cls, state: dict) -> SourceTrust:
        """Validate a complete snapshot before returning a fresh registry."""
        if not isinstance(state, dict) or set(state) != {'version', 'window', 'prior', 'min_labels', 'clock', 'assertions', 'labels'} or type(state['version']) is not int or state['version'] != 1:
            raise ValueError("invalid snapshot schema")
        obj = cls(window=state['window'], prior=state['prior'], min_labels=state['min_labels'])
        if not isinstance(state['assertions'], dict) or not isinstance(state['labels'], dict):
            raise ValueError("invalid snapshot records")
        events = []
        for key, row in state['assertions'].items():
            if not isinstance(row, dict) or set(row) != {'statement', 'source', 'asserted_at', 'known_at', 'parent'}:
                raise ValueError("invalid assertion schema")
            events.append((_number(row['known_at'], 'known_at'), 0, key, row))
        for key, row in state['labels'].items():
            if not isinstance(row, dict) or set(row) != {'loss', 'known_at'}:
                raise ValueError("invalid label schema")
            events.append((_number(row['known_at'], 'known_at'), 1, key, row))
        # Parent dependencies determine ties, independent of JSON object ordering.
        pending = sorted(events, key=lambda e: (e[0], e[1]))
        while pending:
            now = pending[0][0]
            group = [event for event in pending if event[0] == now]
            pending = [event for event in pending if event[0] != now]
            assertions = [event for event in group if event[1] == 0]
            while assertions:
                ready = [event for event in assertions if event[3]['parent'] is None
                         or event[3]['parent'] in obj._assertions]
                if not ready:
                    raise ValueError("missing, future or cyclic parent")
                for event in ready:
                    _, _, key, row = event
                    obj.assert_statement(key, statement=row['statement'], source=row['source'],
                                         asserted_at=row['asserted_at'], parent=row['parent'], now=now)
                    assertions.remove(event)
            for _, kind, key, row in group:
                if kind:
                    obj.label(key, loss=row['loss'], now=now)
        clock = state['clock']
        if clock is None:
            if events:
                raise ValueError("missing snapshot clock")
        elif obj._time(clock) != obj._clock:
            raise ValueError("snapshot clock differs from last event")
        return obj

    def fingerprint(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()
