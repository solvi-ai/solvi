"""Bounded packed observations and paid, feedback-driven stratification.

Snapshots bind ordinary host data, not authenticated observations. No oracle is
called and no label is acquired by these objects.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ._common import identifier

__all__ = ["PackedObservationFrame", "PackedStratifiedSampler"]


def _bin(value):
    if type(value) is not int or not 0 <= value <= 7:
        raise ValueError("strict integer bins 0..7 required")
    return value


@dataclass(frozen=True, init=False)
class PackedObservationFrame:
    names: tuple
    features: tuple
    values: tuple
    source: str
    index: Mapping[str, int]

    def __init__(self, records, budget, *, source):
        budget.charge("packed_frame_meta")
        if type(records) not in (tuple, list) or not 1 <= len(records) <= 128:
            raise ValueError("1..128 bounded records required")
        n = len(records)
        budget.charge("packed_frame_meta", 2 * n)
        identifier(source, "source")
        first = records[0]
        if type(first) not in (tuple, list) or len(first) != 2 or type(first[1]) is not dict:
            raise ValueError("(name, dict) records required")
        p = len(first[1])
        if not 1 <= p <= 128:
            raise ValueError("1..128 features required")
        budget.charge("packed_frame_schema", p)
        budget.charge("packed_frame_fields", 2 * n * p)
        features = tuple(f"f{i}" for i in range(p))
        names, rows, index = [], [], {}
        for record in records:
            if type(record) not in (tuple, list) or len(record) != 2:
                raise ValueError("(name, dict) records required")
            name, row = record
            identifier(name, "record name")
            if name in index or type(row) is not dict or len(row) != p:
                raise ValueError("unique names and matching schema required")
            values = tuple(_bin(row.get(feature)) for feature in features)
            index[name] = len(names)
            names.append(name)
            rows.append(values)
        for key, value in (
            ("names", tuple(names)),
            ("features", features),
            ("values", tuple(rows)),
            ("source", source),
            ("index", MappingProxyType(index)),
        ):
            object.__setattr__(self, key, value)


class PackedStratifiedSampler:
    """Fixed pilot cuts, bounded queues, and true caller-supplied feedback."""

    def __init__(self, frame, pilot, budget, *, strategy="round_robin", cut_mode="all", max_steps=8):
        budget.charge("packed_validate")
        if type(frame) is not PackedObservationFrame:
            raise ValueError("PackedObservationFrame required")
        if type(pilot) not in (tuple, list) or not 1 <= len(pilot) <= 64:
            raise ValueError("1..64 bounded pilot labels required")
        if strategy not in ("round_robin", "adaptive") or cut_mode not in ("all", "median"):
            raise ValueError("unknown strategy or cut mode")
        n, p = len(pilot), len(frame.features)
        q = len(frame.names) - n
        if type(max_steps) is not int or not 1 <= max_steps <= 16 or not max_steps <= q <= 64:
            raise ValueError("require max_steps 1..16 <= candidates 1..64")
        budget.charge("packed_pilot", n)
        labels = {}
        for item in pilot:
            if type(item) not in (tuple, list) or len(item) != 2:
                raise ValueError("(name, bool) pilot required")
            name, label = item
            identifier(name)
            if type(label) is not bool or name not in frame.index or name in labels:
                raise ValueError("unique in-frame pilot IDs with strict bool labels required")
            labels[name] = label
        self._frame, self._pilot = frame, tuple(labels.items())
        self._strategy, self._cut_mode, self._max_steps = strategy, cut_mode, max_steps
        budget.charge("packed_hist", n * p + 8 * p)
        hist, pos = [[0] * 8 for _ in range(p)], [[0] * 8 for _ in range(p)]
        for name, label in self._pilot:
            for i, value in enumerate(frame.values[frame.index[name]]):
                hist[i][value] += 1
                pos[i][value] += label
        budget.charge("packed_cuts", (98 if cut_mode == "all" else 16) * p)
        scores = []
        for i in range(p):
            median, cumulative = 0, 0
            for b in range(8):
                cumulative += hist[i][b]
                if cumulative > n // 2:
                    median = b
                    break
            best = None
            left = lp = 0
            for cut in range(1, 8):
                left += hist[i][cut - 1]
                lp += pos[i][cut - 1]
                if cut_mode == "median" and cut != median:
                    continue
                right, rp = n - left, sum(pos[i]) - lp
                if min(left, right) < 4:
                    continue
                num = abs((lp + 1) * (right + 2) - (rp + 1) * (left + 2))
                den = (left + 2) * (right + 2)
                if num and (best is None or num * best[3] > best[2] * den):
                    best = (i, cut, num, den)
            if best is not None:
                scores.append(best)
        budget.charge("packed_top2", 4 * p)
        selected = []
        if any(labels.values()) and not all(labels.values()):
            for score in scores:
                at = 0
                while at < len(selected) and (
                    selected[at][2] * score[3] > score[2] * selected[at][3]
                    or selected[at][2] * score[3] == score[2] * selected[at][3]
                    and selected[at][0] < score[0]
                ):
                    at += 1
                selected.insert(at, score)
                del selected[2:]
        self._features = tuple(frame.features[s[0]] for s in selected)
        self._thresholds = tuple(s[1] for s in selected)
        k = len(selected)
        budget.charge("packed_cells", 2 * (n + q) * k if k else n + q)
        budget.charge("packed_queues", n + q)
        self._queues = [[] for _ in range(1 << k)]
        self._counts, self._positives = [0] * (1 << k), [0] * (1 << k)
        for name, row in zip(frame.names, frame.values):
            code = sum((row[s[0]] >= s[1]) << bit for bit, s in enumerate(selected))
            if name in labels:
                self._counts[code] += 1
                self._positives[code] += labels[name]
            else:
                self._queues[code].append(name)
        self._positions = [0] * len(self._queues)
        self._cursor, self._pending, self._pending_cell = 0, None, None
        self._received = []

    frame = property(lambda self: self._frame)
    pilot = property(lambda self: self._pilot)
    cursor = property(lambda self: self._cursor)
    pending = property(lambda self: self._pending)
    received = property(lambda self: tuple(self._received))
    queues = property(lambda self: tuple(tuple(queue) for queue in self._queues))
    counts = property(lambda self: tuple(self._counts))
    positives = property(lambda self: tuple(self._positives))
    positions = property(lambda self: tuple(self._positions))
    features = property(lambda self: self._features)
    thresholds = property(lambda self: self._thresholds)
    strategy = property(lambda self: self._strategy)
    cut_mode = property(lambda self: self._cut_mode)
    max_steps = property(lambda self: self._max_steps)
    reason = property(lambda self: "stratified" if self._features else "random_prefix")

    def choose(self, budget):
        budget.charge("packed_choose", 8 * len(self._queues) + 1)
        if self._pending is not None:
            raise ValueError("observe pending choice first")
        if len(self._received) >= self._max_steps:
            return None
        chosen, numerator, denominator = None, 0, 1
        for offset in range(len(self._queues)):
            code = (self._cursor + offset) % len(self._queues) if self._strategy == "round_robin" else offset
            remaining = len(self._queues[code]) - self._positions[code]
            if not remaining:
                continue
            if self._strategy == "round_robin":
                chosen = code
                break
            n, a = self._counts[code], self._positives[code]
            num, den = remaining * (a + 1) * (n - a + 1), (n + 2) ** 2 * (n + 1)
            if chosen is None or num * denominator > numerator * den:
                chosen, numerator, denominator = code, num, den
        if chosen is None:
            return None
        self._pending_cell = chosen
        self._pending = self._queues[chosen][self._positions[chosen]]
        return self._pending

    def observe(self, name, label, budget):
        budget.charge("packed_observe", 6)
        if self._pending is None or type(name) is not str or name != self._pending or type(label) is not bool:
            raise ValueError("strict bool feedback matching pending ID required")
        code = self._pending_cell
        if (
            type(code) is not int
            or not 0 <= code < len(self._queues)
            or not 0 <= self._positions[code] < len(self._queues[code])
            or self._queues[code][self._positions[code]] != name
        ):
            raise ValueError("pending queue mismatch")
        self._counts[code] += 1
        self._positives[code] += label
        self._positions[code] += 1
        self._cursor = (code + 1) % len(self._queues)
        self._received.append((name, label))
        self._pending, self._pending_cell = None, None

    def to_dict(self):
        """Native IO snapshot; serialized lists are accepted by restore."""
        return dict(
            source=self._frame.source,
            names=self._frame.names,
            values=self._frame.values,
            features=self._frame.features,
            pilot=self._pilot,
            strategy=self._strategy,
            cut_mode=self._cut_mode,
            max_steps=self._max_steps,
            received=tuple(self._received),
            pending=self._pending,
        )

    @classmethod
    def restore(cls, frame, saved, budget):
        budget.charge("packed_validate")
        if (
            type(frame) is not PackedObservationFrame
            or type(saved) is not dict
            or len(saved) != 10
            or saved.keys()
            != {
                "source",
                "names",
                "values",
                "features",
                "pilot",
                "strategy",
                "cut_mode",
                "max_steps",
                "received",
                "pending",
            }
        ):
            raise ValueError("frame and snapshot dictionary required")
        budget.charge("packed_frame_match", len(frame.names) * (len(frame.features) + 1))

        def matches(value, expected):
            return (
                type(value) in (tuple, list)
                and len(value) == len(expected)
                and all(
                    matches(a, b) if type(b) is tuple else type(a) is type(b) and a == b
                    for a, b in zip(value, expected)
                )
            )

        if (
            type(saved.get("source")) is not str
            or saved["source"] != frame.source
            or any(not matches(saved.get(key), getattr(frame, key)) for key in ("names", "values", "features"))
        ):
            raise ValueError("snapshot frame mismatch")
        received = saved["received"]
        if type(received) not in (tuple, list) or len(received) > 16:
            raise ValueError("bounded received history required")
        result = cls(
            frame,
            saved["pilot"],
            budget,
            strategy=saved["strategy"],
            cut_mode=saved["cut_mode"],
            max_steps=saved["max_steps"],
        )
        for item in received:
            if type(item) not in (tuple, list) or len(item) != 2 or result.choose(budget) != item[0]:
                raise ValueError("snapshot replay mismatch")
            result.observe(item[0], item[1], budget)
        pending = saved.get("pending")
        if pending is not None and (type(pending) is not str or result.choose(budget) != pending):
            raise ValueError("snapshot pending mismatch")
        return result
