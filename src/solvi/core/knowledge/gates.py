"""Write gates: what lets a proposed item into the knowledge store, and what lets a behaviour-changing one become a fact.

    class MyGate:                                  # the WriteGate protocol
        name = "mine"
        def admit(self, store, item, shadow) -> Verdict: ...

    KnowledgeStore(gates=[SourceGate(), ConsistencyGate(), MyGate()])

`item` is the proposal as a dict (KnowledgeStore.proposal: id, kind, body, scope, source, by, derived_from, key, ...);
`shadow` is whatever the writer passed to `add(..., shadow=)` — for a rule, skill or action, a shadow measurement.
Every verdict is recorded in the journal: a refusal as a "refused" entry, a held behaviour-changing item as "held", a
promotion with every gate's verdict.

Built in (stable):
  SourceGate        the source check: "person", "outcome", "spec", or "verified" — a System 2 answer that answered
                    alone under its guarantee, named by `of=` and checked in the store's TraceStorage. Never the system's
                    own unverified answer. It runs first in every store, whatever gates are given (by construction).
  ConsistencyGate   premises must exist and be present (not retracted or refuted), and a derived item may not hold
                    another value on its own premise's key (it would refute its support). Contradictions between items
                    are not a refusal: the store resolves them (outcome > person > spec > verified; else a dispute).

The shadow gate — measure a candidate on a shadow of the system (held-out gain, honesty, size of the change) before it
changes System 1's behaviour — is experimental: solvi.experimental.learning.ShadowGate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..sources import UntrustedLabel, check_source

SOURCES = ("outcome", "person", "spec", "verified")
RANK = {"outcome": 4, "person": 3, "spec": 2, "verified": 1}     # who wins a contradiction (else: a dispute)
SOLVI_SOURCE = {"person": "human", "outcome": "outcome", "spec": "rule", "verified": "verified"}   # check_source's names


@dataclass
class Verdict:
    """A write gate's answer about one proposed item: admitted or not, why, and what it measured (recorded)."""
    admit: bool
    reason: str = ""
    measured: dict | None = None
    gate: str = ""

    def to_dict(self):
        return {"gate": self.gate, "admit": self.admit, "reason": self.reason, "measured": self.measured}


@runtime_checkable
class WriteGate(Protocol):
    """A write gate of the knowledge store.

    You implement: `admit(store, item, shadow) → Verdict(admit, reason, measured)` — `item` is the proposal
    (KnowledgeStore.proposal), `shadow` what the writer passed to `add(..., shadow=)`.

    You get for free: it runs after the built-in source check on every proposed item; its verdict is journaled; a
    refusal keeps a fact out, a hold keeps a rule, skill or action a hypothesis.

    Stability: stable."""

    def admit(self, store, item, shadow) -> Verdict: ...


class SourceGate:
    """The source check (see the module docstring)."""
    name = "source"

    def admit(self, store, item, shadow=None):
        src = item.get("source")
        try:
            if src not in RANK:
                raise UntrustedLabel(f"source {src!r} is not one of {sorted(RANK)}: knowledge comes from a person, an "
                                     "outcome, a written spec, or a verified System 2 answer — never the system's own "
                                     "unverified answer")
            check_source(SOLVI_SOURCE[src], accept=("verified",) if src == "verified" else ())
            if src == "verified":
                self._vouched(store, item)
        except UntrustedLabel as e:
            return Verdict(False, str(e), gate=self.name)
        return Verdict(True, gate=self.name)

    @staticmethod
    def _vouched(store, item):
        of, body = item.get("of"), item.get("body")
        if of is None:
            raise UntrustedLabel("a verified item names the System 2 decision it comes from: of=<its stored id>")
        if store.storage is None:
            raise UntrustedLabel("a verified item is checked against its decision in the store's TraceStorage: open the "
                                 "KnowledgeStore on the decisions' storage")
        if not (isinstance(body, dict) and "r" in body and "o" in body):
            raise UntrustedLabel("a verified item is a fact {s, r, o} whose relation is the question and whose object "
                                 "is the decision's answer")
        store.storage._check_vouched(body["r"], body["o"], of)


class ConsistencyGate:
    """Premises exist and are present; a derived item is not self-defeating (see the module docstring)."""
    name = "consistency"

    def admit(self, store, item, shadow=None):
        just = item.get("derived_from") or ()
        missing = [p for p in just if p not in store.items]
        if missing:
            return Verdict(False, f"premises never written: {missing}", gate=self.name)
        gone = [p for p in just if store.status(p) in ("retracted", "refuted")]
        if gone:
            return Verdict(False, f"rests on items that are {', '.join(sorted({store.status(p) for p in gone}))}: {gone}",
                           gate=self.name)
        key = item.get("key")
        if key is not None and just:
            v = (item.get("body") or {}).get("o")
            for p in sorted(store.upstream(just)):
                q = store.items[p]
                if q.key == key and q.value != v:
                    return Verdict(False, f"self-defeating: premise {p} holds {q.value!r} on its key", gate=self.name)
        return Verdict(True, gate=self.name)


def default_gates():
    """The gates a KnowledgeStore uses when none are given: [SourceGate(), ConsistencyGate()]."""
    return [SourceGate(), ConsistencyGate()]


def with_source_gate(gates):
    """The gates with a SourceGate first (added when missing): the source check is never left out."""
    gates = list(gates)
    if not any(isinstance(g, SourceGate) for g in gates):
        gates.insert(0, SourceGate())
    for g in gates:
        if not callable(getattr(g, "admit", None)):
            raise TypeError(f"a write gate has admit(store, item, shadow) → Verdict; {g!r} has none")
    return gates


__all__ = ["ConsistencyGate", "default_gates", "RANK", "SOURCES", "SourceGate", "Verdict", "with_source_gate", "WriteGate"]
