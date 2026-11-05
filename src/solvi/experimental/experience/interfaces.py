"""Host-owned finite environment protocol. Receipts are evidence, not model output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ._common import identifier, natural, portable


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    field: str
    cost: int = 8

    def __post_init__(self):
        identifier(self.name, "probe name")
        identifier(self.field, "probe field")
        natural(self.cost, "probe cost", positive=True)

    def to_dict(self):
        return {"name": self.name, "field": self.field, "cost": self.cost}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    scope: str
    mechanism: str
    schema: str
    inputs: dict
    worlds: tuple
    probes: tuple[ProbeSpec, ...]
    target: str = "answer"
    action_cost: int = 8

    def __post_init__(self):
        for name in ("task_id", "scope", "mechanism", "schema", "target"):
            identifier(getattr(self, name), name)
        if type(self.inputs) is not dict:
            raise ValueError("public inputs must be a portable dict")
        object.__setattr__(self, "inputs", portable(self.inputs))
        object.__setattr__(self, "worlds", tuple(portable(list(self.worlds))))
        object.__setattr__(self, "probes", tuple(self.probes))
        if any(not isinstance(p, ProbeSpec) for p in self.probes):
            raise ValueError("task probes must be ProbeSpec values")
        if len({p.name for p in self.probes}) != len(self.probes):
            raise ValueError("task probes have duplicate names")
        natural(self.action_cost, "action cost", positive=True)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "scope": self.scope,
            "mechanism": self.mechanism,
            "schema": self.schema,
            "inputs": portable(self.inputs),
            "worlds": portable(list(self.worlds)),
            "probes": [p.to_dict() for p in self.probes],
            "target": self.target,
            "action_cost": self.action_cost,
        }

    @classmethod
    def from_dict(cls, value):
        v = portable(value)
        expected = {"task_id", "scope", "mechanism", "schema", "inputs", "worlds", "probes", "target", "action_cost"}
        if type(v) is not dict or set(v) != expected:
            raise ValueError("unsupported task specification")
        v["probes"] = tuple(ProbeSpec(**p) for p in v["probes"])
        v["worlds"] = tuple(v["worlds"])
        return cls(**v)


@dataclass(frozen=True)
class Receipt:
    event_id: str
    attempt_id: str
    task_id: str
    kind: str
    source: str
    mechanism: str
    value: dict

    def __post_init__(self):
        for name in ("event_id", "attempt_id", "task_id", "source", "mechanism"):
            identifier(getattr(self, name), name)
        if self.kind not in ("observe", "act") or type(self.value) is not dict:
            raise ValueError("receipt requires an external observe/act result dict")
        object.__setattr__(self, "value", portable(self.value))

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "attempt_id": self.attempt_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "source": self.source,
            "mechanism": self.mechanism,
            "value": portable(self.value),
        }

    @classmethod
    def from_dict(cls, value):
        v = portable(value)
        if type(v) is not dict or set(v) != {
            "event_id",
            "attempt_id",
            "task_id",
            "kind",
            "source",
            "mechanism",
            "value",
        }:
            raise ValueError("unsupported external receipt")
        return cls(**v)


class Environment(Protocol):
    """Trusted host adapter; identity is not network authentication.

    execute must be idempotent by attempt ID, or reconcile must settle whether
    its effect happened. Unknown outcomes must not be retried blindly. A
    serializable receipt is not proof of truth against a hostile host.

    observe payload: {"probe": name, "field": field}; receipt value:
    {"field": field, "value": measured_value}.
    act payload: {"answer": value}; receipt value:
    {"answer": independently_observed_correct_answer, "ok": bool, ...}.
    """

    identity: str

    def task(self, task_id: str) -> TaskSpec: ...

    def execute(self, task_id: str, kind: str, payload: dict, attempt_id: str) -> Receipt | None: ...

    def reconcile(self, attempt_id: str) -> Receipt | None: ...


__all__ = ["Environment", "ProbeSpec", "Receipt", "TaskSpec"]
