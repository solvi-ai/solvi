"""Experience notes on KnowledgeStore's chain; never executes an environment.

Ordinary append/checkpoint fingerprint the checked prefix and fully verify any
new suffix. debit() is reserved for trusted pure computation: it only checks the
last known head/count, then durably writes the ledger without scanning history.
It can temporarily miss old payload mutations; the next ordinary event,
checkpoint, records, latest or verify validates the entire fast suffix before
the host may cause an external effect. Unexpected growth also verifies fully.
This assumes no out-of-band backend writes between full checks; a head-only
guard cannot detect edits to old in-memory payloads.
A checkpoint is not an external anchor against a fully rewritten source chain.
Hosts must serialize writers sharing a cycle/store.
"""

from __future__ import annotations

import hashlib
import json

from ...core.knowledge import KnowledgeStore
from ._common import WorkBudget, digest, identifier, natural, portable

_VERSION = 1
_FIELDS = {"version", "cycle", "config", "seq", "kind", "payload"}


def _signature(records):
    # Store envelopes have their own JSON contract, including floating time.
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class ExperienceJournal:
    def __init__(self, store: KnowledgeStore, cycle_id: str, config: dict):
        if not isinstance(store, KnowledgeStore):
            raise TypeError("store must be KnowledgeStore")
        if type(config) is not dict:
            raise ValueError("config must be a portable dict")
        self.store = store
        self.cycle_id = identifier(cycle_id, "cycle_id")
        self._config = digest(config)
        self._count = 0
        self._signature = _signature([])
        self._events = []
        self._validate()

    def _scan(self):
        cycles = {}
        events = []
        for cursor, record in enumerate(self.store.journal):
            if "experience" not in record:
                continue
            e = portable(record["experience"])
            if record.get("op") != "note" or type(e) is not dict or set(e) != _FIELDS:
                raise ValueError("malformed experience event")
            if type(e["version"]) is not int or e["version"] != _VERSION:
                raise ValueError("unsupported experience version")
            cycle = identifier(e["cycle"], "cycle")
            identifier(e["kind"], "event kind")
            fp = identifier(e["config"], "config fingerprint")
            if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
                raise ValueError("malformed config fingerprint")
            if type(e["payload"]) is not dict:
                raise ValueError("event payload must be a dict")
            seq = natural(e["seq"], "event sequence")
            previous_fp, expected = cycles.get(cycle, (fp, 0))
            if fp != previous_fp or seq != expected:
                raise ValueError("conflicting config or nonmonotonic experience sequence")
            cycles[cycle] = (fp, seq + 1)
            if e["kind"] == "checkpoint":
                p = e["payload"]
                if set(p) != {"version", "cursor", "anchor", "state", "digest"}:
                    raise ValueError("malformed checkpoint")
                if type(p["version"]) is not int or p["version"] != _VERSION:
                    raise ValueError("unsupported checkpoint version")
                if type(p["cursor"]) is not int or p["cursor"] != cursor:
                    raise ValueError("checkpoint cursor mismatch")
                anchor = self.store.journal[cursor - 1]["hash"] if cursor else ""
                if p["anchor"] != anchor or p["digest"] != digest(p["state"]):
                    raise ValueError("checkpoint anchor or digest mismatch")
                if type(p["state"]) is not dict:
                    raise ValueError("checkpoint state must be a dict")
            if cycle == self.cycle_id:
                if fp != self._config:
                    raise ValueError("incompatible experience configuration")
                events.append(e)
        return events

    def _unchanged_prefix(self):
        if len(self.store.journal) < self._count:
            raise ValueError("experience journal was truncated")
        if _signature(self.store.journal[: self._count]) != self._signature:
            raise ValueError("previously checked journal was modified")

    def _remember(self):
        self._count = len(self.store.journal)
        self._signature = _signature(self.store.journal)
        self._fast_head = self.store.head()

    def _check_fast_anchor(self):
        head = getattr(self, "_fast_head", None)
        if head is None:
            return
        count = head["count"]
        if len(self.store.journal) < count:
            raise ValueError("experience journal was truncated")
        if count and self.store.journal[count - 1]["hash"] != head["hash"]:
            raise ValueError("last known experience anchor was modified")

    def _validate(self):
        self._check_fast_anchor()
        self._unchanged_prefix()
        if not self.store.verify():
            raise ValueError("corrupt knowledge journal")
        self._events = self._scan()
        self._remember()

    def _guard(self):
        self._check_fast_anchor()
        self._unchanged_prefix()
        if self.store.head()["count"] != self._count:
            self._validate()

    def append(self, kind: str, payload: dict):
        identifier(kind, "event kind")
        if kind == "checkpoint":
            raise ValueError("use checkpoint() for checkpoint events")
        return self._append(kind, payload)

    def debit(self, ledger_snapshot: dict):
        """Durable budget event for a trusted interval of pure computation.

        Cost is independent of journal length when the known head is unchanged
        (ledger validation/copying still scales with its cost-category count).
        The host MUST use ordinary append/checkpoint before an external effect.
        This deliberately postpones history validation to that boundary.
        """
        if type(ledger_snapshot) is not dict:
            raise ValueError("ledger snapshot must be a portable dict")
        ledger = WorkBudget.restore(ledger_snapshot).snapshot()
        if self.store.head() != self._fast_head:
            self._validate()
        event = {
            "version": _VERSION,
            "cycle": self.cycle_id,
            "config": self._config,
            "seq": len(self._events),
            "kind": "debit",
            "payload": {"ledger": ledger},
        }
        self.store.note(experience=event)
        self._events.append(portable(event))
        self._fast_head = self.store.head()
        return portable(event)

    def _append(self, kind, payload):
        if type(payload) is not dict:
            raise ValueError("event payload must be a portable dict")
        payload = portable(payload)
        self._guard()
        event = {
            "version": _VERSION,
            "cycle": self.cycle_id,
            "config": self._config,
            "seq": len(self._events),
            "kind": kind,
            "payload": payload,
        }
        self.store.note(experience=event)
        self._events.append(portable(event))
        self._remember()
        return portable(event)

    def records(self):
        self._validate()
        return portable(self._events)

    def checkpoint(self, state: dict):
        if type(state) is not dict:
            raise ValueError("checkpoint state must be a portable dict")
        state = portable(state)
        self._guard()
        head = self.store.head()
        return self._append(
            "checkpoint",
            {
                "version": _VERSION,
                "cursor": head["count"],
                "anchor": head["hash"],
                "state": state,
                "digest": digest(state),
            },
        )

    def latest(self):
        self._validate()
        for event in reversed(self._events):
            if event["kind"] == "checkpoint":
                return portable(event["payload"])
        return None

    def verify(self):
        """Return True after full validation; raise ValueError on corruption."""
        self._validate()
        return True
