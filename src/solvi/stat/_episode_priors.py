"""Explicit hierarchical backoff across past episodes; descriptive Beta point model."""

from dataclasses import dataclass
import hashlib
import json
import math

__all__ = ["EpisodePrior", "EpisodePriors"]


def _key(value):
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty stable identifiers required")
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric values required")
    return float(value)


@dataclass(frozen=True)
class EpisodePrior:
    prior: tuple
    level: str
    record_ids: tuple
    episodes: tuple
    context: str
    source_type: str
    source: str
    role: str
    excluded_episode: str
    as_of: float
    fingerprint: str


class EpisodePriors:
    """Source → type → context → base backoff, within explicit context and role.

    Owner supplies independently labelled past-episode losses, identities, roles
    and episode end times. Not automatic classes or an exact hierarchical Bayes posterior.
    """

    def __init__(self, *, window, base=(1.0, 1.0), max_strength=20.0, min_records=1):
        self.window, self.max_strength = _number(window), _number(max_strength)
        if not isinstance(base, (tuple, list)) or len(base) != 2:
            raise ValueError("positive Beta base required")
        self.base = tuple(_number(x) for x in base)
        if (
            self.window <= 0
            or min(self.base) <= 0
            or not math.isfinite(sum(self.base))
            or self.max_strength < sum(self.base)
            or type(min_records) is not int
            or min_records < 1
        ):
            raise ValueError("invalid window, strength or minimum records")
        self.min_records = min_records
        self._records = {}

    def record(
        self, record_id, *, episode, context, source_type, source, role, loss, asserted_at, ended_at, known_at, now
    ):
        names = dict(
            record_id=record_id, episode=episode, context=context, source_type=source_type, source=source, role=role
        )
        for value in names.values():
            _key(value)
        loss, asserted_at, ended_at, known_at, now = map(_number, (loss, asserted_at, ended_at, known_at, now))
        if record_id in self._records or not 0 <= loss <= 1 or not asserted_at <= ended_at <= known_at <= now:
            raise ValueError("unique independently labelled past-episode record required")
        row = dict(
            episode=episode,
            context=context,
            source_type=source_type,
            source=source,
            role=role,
            loss=loss,
            asserted_at=asserted_at,
            ended_at=ended_at,
            known_at=known_at,
        )
        self._records[record_id] = row

    def get(self, *, current_episode, context, source_type, source, role, now):
        for value in (current_episode, context, source_type, source, role):
            _key(value)
        now = _number(now)
        eligible = [
            (k, r)
            for k, r in sorted(self._records.items())
            if r["episode"] != current_episode
            and r["context"] == context
            and r["role"] == role
            and now - self.window <= r["asserted_at"] <= now
            and r["ended_at"] <= now
            and r["known_at"] <= now
        ]
        selected = []
        level = "base"
        for name, rows in [
            ("source", [(k, r) for k, r in eligible if r["source_type"] == source_type and r["source"] == source]),
            ("type", [(k, r) for k, r in eligible if r["source_type"] == source_type]),
            ("context", eligible),
        ]:
            if len(rows) >= self.min_records:
                selected, level = rows, name
                break
        alpha = self.base[0] + math.fsum(1 - r["loss"] for _, r in selected)
        beta = self.base[1] + math.fsum(r["loss"] for _, r in selected)
        strength = alpha + beta
        if strength > self.max_strength:
            alpha, beta = (alpha / strength * self.max_strength, beta / strength * self.max_strength)
        if min(alpha, beta) <= 0 or not math.isfinite(alpha + beta):
            raise ValueError("prior not representable")
        ids = tuple(k for k, _ in selected)
        episodes = tuple(sorted({r["episode"] for _, r in selected}))
        provenance = dict(
            snapshot=self.fingerprint(),
            prior=[alpha, beta],
            level=level,
            record_ids=ids,
            context=context,
            source_type=source_type,
            source=source,
            role=role,
            current_episode=current_episode,
            now=now,
        )
        fp = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return EpisodePrior(
            (alpha, beta), level, ids, episodes, context, source_type, source, role, current_episode, now, fp
        )

    def snapshot(self):
        return json.loads(
            json.dumps(
                dict(
                    version=1,
                    window=self.window,
                    base=self.base,
                    max_strength=self.max_strength,
                    min_records=self.min_records,
                    records=self._records,
                ),
                allow_nan=False,
            )
        )

    @classmethod
    def restore(cls, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "window", "base", "max_strength", "min_records", "records"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or not isinstance(state["records"], dict)
        ):
            raise ValueError("invalid episode-prior snapshot")
        obj = cls(
            window=state["window"],
            base=state["base"],
            max_strength=state["max_strength"],
            min_records=state["min_records"],
        )
        keys = {"episode", "context", "source_type", "source", "role", "loss", "asserted_at", "ended_at", "known_at"}
        for key, row in state["records"].items():
            if not isinstance(row, dict) or set(row) != keys:
                raise ValueError("invalid episode record schema")
            obj.record(key, **row, now=row["known_at"])
        return obj

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
