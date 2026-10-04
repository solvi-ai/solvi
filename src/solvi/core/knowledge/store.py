"""The knowledge store: what a system has learned, from whom, and what rests on it — a hash-chained journal of items.

    from solvi.core.knowledge import KnowledgeStore
    ks = KnowledgeStore("knowledge.jsonl")              # any TraceStorage backend (a path, a TraceStorage) or None
    a = ks.add("fact", {"s": "abt:123", "r": "same_as", "o": "buy:456"}, source="person", by="ann")
    b = ks.add("fact", {"s": "abt:123", "r": "brand", "o": "acme"}, source="outcome", derived_from=[a])
    ks.retract(a, by="bob", why="wrong label")          # b goes with it; what a refuted or disputed comes back
    ks.verify(); ks.fingerprint(); ks.rebuild(skip={a}).fingerprint()   # equal: a retraction is exact
    ks.snapshot(about="abt:123")                        # what a decision is given: facts, hints, the fingerprint

An item is one of five kinds — fact (subject, relation, object), rule, skill, action (an action model's preconditions
and effects), episode (one finished case) — with a source: "person", "outcome" (an observation: what really happened),
"spec" (a written specification) or "verified" (a System 2 answer given alone under its guarantee, named by its stored
decision `of=`). The system's own unverified answers are never admitted: the source check is part of the store, not a
setting (SourceGate; solvi.core.sources.check_source).

State is a function of the live journal entries (JTMS-style): an item is present while it has a live assertion whose
premises (`derived_from`) are present and not refuted. Per key (scope, subject, relation) the live assertions are walked
in time order: the same value confirms; an outcome refutes the current value (the world changed); a higher-ranked source
(outcome > person > spec > verified) refutes a lower one; `supersedes` (a producer's own new version) refutes the version
it names; any other contradiction makes every side "disputed" and asks a person — one question per dispute EVENT (the
journal position it opened at): the same two items disputed again after a resolution are asked again. `resolve(key,
value, by=)` is the person's answer. A disputed item is never a fact.

Statuses: "active" (a fact System 1 may answer on), "hypothesis" (a hint: a behaviour-changing item not yet promoted
by the write gates, a derived item on a premise that is not active, or an item under a pending flag), "disputed",
"refuted", "expired" (its reconfirm_after ran out on the store's clock, or retired) and "retracted".

Staleness. `flag(scope=..., why=...)` (a drift or open-set flag) or `reconfirm(id)` marks items for re-confirmation: an
item learned before the flag is a hint, not a fact, until a newer observation or person confirms it or the flag ends
(`end_flag`). A flag is read where staleness is asked (`stale(id)`), not inferred from the item's status. `rollback(id)`
(supervision undoing an update) restores the previous version as a hint, never as a fact, while a flag covering the
rolled-back version or its scope is pending (a rollback during a pending drift flag once restored a stale threshold as
a fact on a shifted stream, and a stale check that read the item status missed it). `reconfirm_after` per
(kind, source) (RECONFIRM_DEFAULTS, the store's clock advanced by `tick`) expires items that weaker evidence keeps.

Every write — add, promote, refuse, retract, flag, resolve, tick, use — is a journal entry: in a TraceStorage (records of
kind "knowledge", in the same hash chain as the decisions when it is the decisions' store) or in memory with its own hash
chain. `verify()` checks the chain and that the state is what the journal gives; `rebuild(upto=n, skip=ids)` gives the
store as it was at entry n, or as if `ids` had never been written; a store opened on a backend is replayed from its
journal (an entry edited by hand breaks the chain). `snapshot(...)` returns what a decision reads, with the store's
fingerprint and journal position, so every decision that is given it records the knowledge it used and replays;
`redecide(storage, retracted, decide)` re-runs the decisions that rested on retracted items on their re-built snapshots
and splits them into "answer changes" (to a reviewer) and "justification only" (recorded).

Measured by benchmarks/knowledge/retraction.py: RETRACTION_DOC. Cost grows with the connected component a write
touches (premise edges and shared keys); the store has been measured to 10,000 items, not 10⁵–10⁶."""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from contextlib import contextmanager

from ..runtime import vhash
from .gates import RANK, SOURCES, Verdict, default_gates, with_source_gate

KINDS = ("fact", "rule", "skill", "action", "episode")
NEEDS_PROMOTION = ("rule", "skill", "action")         # behaviour-changing kinds: a hypothesis until the gates promote it
STATUSES = ("active", "hypothesis", "disputed", "refuted", "expired", "retracted")
LIFTS_FLAG = ("outcome", "person")                    # a newer assertion from these confirms an item under a flag
# reconfirm_after per (kind, source), in the store's clock units (decisions on a stream, moves in an environment); None:
# no clock — the item stays until an observation, a person, a flag or a retraction ends it. Why: an observation is
# refuted by the next contradicting one, so it needs no clock; a System 2
# answer vouched for only by its guarantee is weaker evidence; an S1 version is a fact only while an audit renews it.
RECONFIRM_DEFAULTS = {
    ("fact", "outcome"): None,
    ("fact", "person"): None,
    ("fact", "spec"): None,
    ("fact", "verified"): 2000,
    ("rule", "person"): 1000,
    ("skill", "outcome"): None,
    ("action", "outcome"): None,
    ("episode", "outcome"): None,
}
_KNOWLEDGE = "knowledge"                              # the record kind of journal entries in a TraceStorage


class _Item:
    __slots__ = ("id", "kind", "body", "scope", "key", "value", "asserts", "promoted", "retracted", "retired",
                 "reconfirm_after", "uses")

    def __init__(self, iid, kind, body, scope, key, value, reconfirm_after):
        self.id, self.kind, self.body, self.scope, self.key, self.value = iid, kind, body, scope, key, value
        self.asserts = []           # [{"t", "clock", "source", "by", "just", "supersedes", "evidence", "valid", "level"}]
        self.promoted = []          # [t]
        self.retracted = []         # [t]
        self.retired = []           # [t]
        self.reconfirm_after = reconfirm_after
        self.uses = [0, 0]          # [used, used and wrong]


def plain_json(x):
    """A body, scope or evidence as plain JSON (tuples become lists), so an id is the same after a reload."""
    return json.loads(json.dumps(x, ensure_ascii=False, sort_keys=True))


def key_of(body, scope):
    """Items whose body has a subject "s" and a relation "r" have a key: one value per (scope, s, r) at a time."""
    if isinstance(body, dict) and "s" in body and "r" in body:
        s = body["s"] if isinstance(body["s"], str) else json.dumps(body["s"], sort_keys=True, ensure_ascii=False)
        return (json.dumps(scope, sort_keys=True), s, str(body["r"]))
    return None


def item_id(kind, body, scope):
    """The content id of an item: the same knowledge written twice is one item."""
    return vhash({"kind": kind, "body": body, "scope": scope})[:24]


def scope_matches(scope, pattern):
    """Does an item's scope fall under a flag's scope (every key of `pattern` has the same value in `scope`)?"""
    return all(scope.get(k) == v for k, v in (pattern or {}).items())


class SelfDefeating(ValueError):
    """A derived item whose key is a key of one of its own premises with another value: it would refute its support."""


class _MemoryJournal:
    """The journal of a store kept in memory: entries chained by vhash (as WorldMap's journal)."""

    def __init__(self):
        self.entries, self.prev = [], ""

    def append(self, body):
        rec = dict(body, prev=self.prev)
        rec["hash"] = vhash(rec)
        self.prev = rec["hash"]
        self.entries.append(rec)
        return rec

    def verify(self):
        prev = ""
        for r in self.entries:
            if r.get("prev") != prev or vhash({k: v for k, v in r.items() if k != "hash"}) != r.get("hash"):
                return False
            prev = r["hash"]
        return True


class KnowledgeStore:
    """See the module docstring.

    storage: where the journal is kept — None (in memory), a path (solvi.core.store.open_storage: .jsonl, .db, .duckdb,
    postgresql://) or a TraceStorage (the decisions' store itself, so knowledge and decisions share one hash chain); an
    existing journal there is replayed. gates: the write gates (solvi.core.knowledge.gates; default SourceGate and
    ConsistencyGate); the source check runs first whatever is given. defaults: reconfirm_after per (kind, source)."""

    def __init__(self, storage=None, *, gates=None, defaults=None, _replay=True):
        from ..store import open_storage
        self.storage = open_storage(storage)
        self.gates = with_source_gate(gates if gates is not None else default_gates())
        self.defaults = dict(RECONFIRM_DEFAULTS if defaults is None else defaults)
        self._lock = threading.RLock()
        self._mem = _MemoryJournal() if self.storage is None else None
        self.journal = []                            # the entries in order, each with its position "n"
        self.items: dict[str, _Item] = {}
        self.by_key = defaultdict(set)
        self.dependents = defaultdict(set)
        self.resolves = defaultdict(list)            # key → [(t, value, by)]
        self.flags = {}                              # t → {"ids", "scope", "key", "why", "by", "parent", "ended"}
        self.clock = 0
        self.state = {}                              # id → resolved record (present items)
        self.questions = []                          # person questions (disputes), one per dispute event
        self._open_disputes = set()
        self._batch, self._touched, self._tick_dirty = 0, set(), False
        self._fp = None
        if self.storage is not None and _replay:
            with self.batch():
                for s in self.storage.iter(_KNOWLEDGE):
                    r = dict(s.data)
                    self.journal.append(r)
                    self._apply(r)

    # ------------------------------------------------------------------------------------------------ the journal
    def _write(self, op, **data):
        body = {"kind": _KNOWLEDGE, "n": len(self.journal), "op": op, "clock": self.clock, **data}
        if self._mem is not None:
            rec = self._mem.append(body)
        else:
            from ..store import FORMAT
            rec = self.storage._append({"v": FORMAT, **body})
        self.journal.append(rec)
        self._fp = None
        return rec

    def __len__(self):
        """The number of journal entries."""
        return len(self.journal)

    def head(self):
        """{"count", "hash"}: the number of journal entries and the last entry's hash."""
        return {"count": len(self.journal), "hash": self.journal[-1]["hash"] if self.journal else ""}

    def verify(self):
        """Is the journal's hash chain whole (in memory: its own chain; on a backend: the backend's verify()), and is the
        state the one the journal gives (the store rebuilt from it has the same fingerprint)?"""
        if self._mem is not None:
            if not self._mem.verify():
                return False
        elif not self.storage.verify()["ok"]:
            return False
        return self.rebuild().fingerprint() == self.fingerprint()

    def rebuild(self, upto=None, skip=()):
        """The store the journal gives, in memory: entries applied in order (upto: the first `upto` — the store as it
        was then); `skip`: item ids whose own entries (assertions, promotions, retractions) are left out — the store as if
        they had never been written. Premises naming a skipped item stay in the entries derived from it (dead)."""
        skip = set(skip)
        ks = KnowledgeStore(None, gates=self.gates, defaults=self.defaults, _replay=False)
        entries = self.journal if upto is None else self.journal[:upto]
        with ks.batch():
            for r in entries:
                if r.get("item") in skip:
                    continue
                ks.journal.append(r)
                ks._apply(r)
        return ks

    def _apply(self, r):
        """Apply one journal entry to the raw data (rebuild and replay: no new entry is written)."""
        op, t = r["op"], r["n"]
        self.clock = r.get("clock", self.clock)
        if op == "assert":
            self._raw_assert(r, t)
        elif op == "promote":
            if r["item"] in self.items:
                self.items[r["item"]].promoted.append(t)
                self._touch(r["item"])
        elif op == "retract":
            if r["item"] in self.items:
                self.items[r["item"]].retracted.append(t)
                self._touch(r["item"])
        elif op == "retire":
            if r["item"] in self.items:
                self.items[r["item"]].retired.append(t)
                self._touch(r["item"])
        elif op == "resolve":
            k = tuple(r["key"])
            self.resolves[k].append((t, r["value"], r.get("by")))
            for i in self.by_key.get(k, ()):
                self._touch(i)
        elif op == "flag":
            self.flags[t] = {"ids": r.get("ids"), "scope": r.get("scope"), "key": r.get("key"), "why": r.get("why"),
                             "by": r.get("by"), "parent": r.get("parent"), "ended": None}
            self._touch_flag(t)
        elif op == "end_flag":
            f = self.flags.get(r["flag"])
            if f is not None and f["ended"] is None:
                f["ended"] = t
                self._touch_flag(r["flag"])
        elif op == "tick":
            self.clock = r["clock"] + r["k"]
            self._tick_dirty = True
            if not self._batch:
                self._flush()
        elif op == "use":
            for i in r.get("ids") or ():
                it = self.items.get(i)
                if it is not None:
                    it.uses[0] += 1
                    it.uses[1] += bool(r.get("wrong"))
        elif op in ("refused", "held", "note", "observe", "agenda", "rollback"):
            pass
        else:
            raise ValueError(f"unknown knowledge journal op {op!r}")

    # ------------------------------------------------------------------------------------------------ writes
    @contextmanager
    def batch(self):
        """Several writes, one resolution at the end (a consolidation writes many items)."""
        with self._lock:
            self._batch += 1
            try:
                yield self
            finally:
                self._batch -= 1
                if not self._batch:
                    self._flush()

    def _touch(self, iid):
        self._touched.add(iid)
        if not self._batch:
            self._flush()

    def _raw_assert(self, r, t):
        iid = r["item"]
        it = self.items.get(iid)
        if it is None:
            key = key_of(r["body"], r["scope"])
            value = r["body"].get("o") if isinstance(r["body"], dict) else None
            it = _Item(iid, r["item_kind"], r["body"], r["scope"], key, value, r.get("reconfirm_after"))
            self.items[iid] = it
            if key is not None:
                self.by_key[key].add(iid)
        it.asserts.append({"t": t, "clock": r.get("clock", 0), "source": r["source"], "by": r.get("by"),
                           "just": tuple(r.get("just") or ()), "supersedes": r.get("supersedes"),
                           "evidence": r.get("evidence"), "valid": r.get("valid"), "level": r.get("level"),
                           "of": r.get("of")})
        for p in it.asserts[-1]["just"]:
            self.dependents[p].add(iid)
        self._touch(iid)

    def proposal(self, kind, body, scope=None, *, source, by=None, evidence=None, derived_from=(), supersedes=None,
                 reconfirm_after=None, valid=None, level=None, of=None):
        """A proposed item as the write gates see it (a dict with the record's fields; nothing is written)."""
        if kind not in KINDS:
            raise ValueError(f"kind is one of {KINDS}, not {kind!r}")
        body, scope = plain_json(body), plain_json(scope or {})
        if reconfirm_after is None:
            reconfirm_after = self.defaults.get((kind, source))
        return {"id": item_id(kind, body, scope), "kind": kind, "body": body, "scope": scope, "source": source,
                "by": None if by is None else str(by), "evidence": plain_json(evidence) if evidence is not None else None,
                "derived_from": sorted(set(derived_from)), "supersedes": supersedes, "reconfirm_after": reconfirm_after,
                "valid": plain_json(list(valid)) if valid is not None else None, "level": level,
                "of": None if of is None else str(of), "key": key_of(body, scope)}

    def add(self, kind, body, scope=None, *, source, by=None, evidence=None, derived_from=(), supersedes=None,
            reconfirm_after=None, valid=None, level=None, of=None, shadow=None):
        """Propose an item → its id, or None when a gate refused it (a "refused" entry in the journal says why).

        kind: one of KINDS; body: plain JSON (a fact: {"s", "r", "o"}); scope: a dict (a question, an environment);
        source: "person" | "outcome" | "spec" | "verified" (anything else is refused: never the system's own answers);
        by: who; evidence: stored ids, quotes, steps; derived_from: premise item ids (the retraction cascade follows them);
        supersedes: the id of the producer's previous version; reconfirm_after: clock units until it needs re-confirmation
        (default RECONFIRM_DEFAULTS); valid: (since, until) in the world, as you count time; level: a guarantee level
        (for "verified"); of: the stored decision a "verified" item is (checked in the store's TraceStorage); shadow:
        passed to the write gates (a shadow measurement for a behaviour-changing item).

        A fact or an episode the gates admit is present at once; a rule, skill or action the gates admit is promoted (its
        verdicts recorded); one a later gate holds stays a hypothesis (a "held" entry says why)."""
        prop = self.proposal(kind, body, scope, source=source, by=by, evidence=evidence, derived_from=derived_from,
                             supersedes=supersedes, reconfirm_after=reconfirm_after, valid=valid, level=level, of=of)
        with self._lock:
            verdicts = []
            for g in self.gates:
                v = g.admit(self, prop, shadow)
                v.gate = v.gate or getattr(g, "name", type(g).__name__)
                verdicts.append(v)
                if not v.admit and (v.gate == "source" or kind not in NEEDS_PROMOTION or v.gate == "consistency"):
                    self._write("refused", item=prop["id"], item_kind=kind, source=source, by=prop["by"],
                                why=f"{v.gate}: {v.reason}")
                    return None
            rec = self._write("assert", item=prop["id"], item_kind=kind, body=prop["body"], scope=prop["scope"],
                              source=source, by=prop["by"], evidence=prop["evidence"], just=prop["derived_from"],
                              supersedes=supersedes, reconfirm_after=prop["reconfirm_after"], valid=prop["valid"],
                              level=level, of=prop["of"])
            with self.batch():
                self._raw_assert(rec, rec["n"])
                if kind in NEEDS_PROMOTION:
                    held = [v for v in verdicts if not v.admit]
                    if held:
                        self._write("held", item=prop["id"], verdicts=[v.to_dict() for v in verdicts])
                    else:
                        self.promote(prop["id"], measured=[v.to_dict() for v in verdicts])
            return prop["id"]

    def promote(self, iid, measured=None):
        """A behaviour-changing item passed its gates (or a person promotes it): it becomes a fact (measurements kept)."""
        with self._lock:
            rec = self._write("promote", item=iid, measured=measured)
            self._apply(rec)

    def retract(self, iid, *, by=None, why=None):
        """Take an item back, with everything derived from it (the cascade over derived_from); what it refuted or
        disputed comes back. Nothing is deleted: the journal keeps the entry and the chain stays whole.
        → {id: (status before, status after)} of every item whose status changed."""
        with self._lock:
            if iid not in self.items:
                raise KeyError(f"no item {iid!r}")
            before = self._statuses(self.component([iid]))
            rec = self._write("retract", item=iid, by=None if by is None else str(by), why=why)
            self._apply(rec)
            after = self._statuses(before)
            return {i: (before[i], after[i]) for i in before if before[i] != after[i]}

    def rollback(self, iid, *, by=None, why=None):
        """Supervision undoes an update: retract `iid`; the version it superseded or refuted comes back — as a fact, or
        only as a hint while a flag covering `iid` (or its scope) is pending: a stale version is never restored as a fact.
        → the status changes (as retract)."""
        with self._lock:
            if iid not in self.items:
                raise KeyError(f"no item {iid!r}")
            it = self.items[iid]
            pending = [t for t in self._pending_flags() if self._flag_covers(t, iid, ignore_age=True)]
            prev = sorted(self._previous_versions(iid))
            with self.batch():
                self._write("rollback", item=iid, previous=prev, pending_flags=pending, by=by, why=why)
                if pending and prev:          # the restored version stays a hint while the flag is pending
                    rec = self._write("flag", ids=prev, scope=None, key=None, parent=pending[-1], by=by,
                                      why=f"restored by the rollback of {iid} during pending flag {pending[-1]}")
                    self._apply(rec)
                before = self._statuses(self.component([iid] + prev))
                rec = self._write("retract", item=iid, by=None if by is None else str(by), why=why or "rollback")
                self._apply(rec)
            del it
            after = self._statuses(before)
            return {i: (before[i], after[i]) for i in before if before[i] != after[i]}

    def _previous_versions(self, iid):
        it = self.items[iid]
        out = {a["supersedes"] for a in it.asserts if a.get("supersedes") in self.items}
        st = self.state.get(iid) or {}
        if it.key is not None:
            out |= {j for j in self.by_key[it.key] if j != iid and iid in ((self.state.get(j) or {}).get("refuted_by") or ())}
        del st
        return out

    def retire(self, iid, *, why=None):
        """The item's producer no longer stands behind it (no longer compiled, no longer offered): expired."""
        with self._lock:
            rec = self._write("retire", item=iid, why=why)
            self._apply(rec)

    def reconfirm(self, iid, *, why=None, by=None):
        """Mark one item for re-confirmation (a flag on it alone) → the flag's id."""
        return self.flag(ids=[iid], why=why, by=by)

    def flag(self, *, scope=None, ids=None, key=None, why=None, by=None):
        """A drift / open-set flag: the items it covers — `ids`, every item whose scope falls under `scope`, or the
        items on `key` — learned before it are hints, not facts, until a newer observation or a person confirms each
        (or end_flag). → the flag's id (its journal position)."""
        if scope is None and ids is None and key is None:
            raise ValueError("a flag covers ids=, scope= or key=")
        with self._lock:
            rec = self._write("flag", ids=None if ids is None else sorted(ids), scope=plain_json(scope) if scope else None,
                              key=None if key is None else list(key), parent=None, by=by, why=why)
            self._apply(rec)
            return rec["n"]

    def end_flag(self, flag, *, by=None, why=None):
        """The flag is resolved (re-confirmation done, a person cleared it): its items are facts again if nothing else
        holds them back."""
        with self._lock:
            if flag not in self.flags:
                raise KeyError(f"no flag {flag!r}")
            rec = self._write("end_flag", flag=flag, by=by, why=why)
            self._apply(rec)

    def resolve(self, key, value, *, by):
        """A person's answer to a dispute on `key` (an item's key, as disputes() lists it): `value` wins, the other
        sides are refuted."""
        with self._lock:
            rec = self._write("resolve", key=list(key), value=plain_json(value), by=str(by))
            self._apply(rec)

    def tick(self, k=1):
        """The store's clock moved by k (decisions on a stream, moves in an environment): reconfirm_after counts it."""
        with self._lock:
            rec = self._write("tick", k=int(k))
            self._apply(rec)

    def used(self, ids, *, wrong=False):
        """Decisions used these items (wrong=True: and were found wrong): counts kept in each item's confidence."""
        with self._lock:
            rec = self._write("use", ids=sorted(set(ids)), wrong=bool(wrong))
            self._apply(rec)

    def note(self, **data):
        """A journal entry that changes nothing (a gate's rejected measurement, a reason): recorded and chained."""
        with self._lock:
            return self._write("note", **plain_json(data))

    # ------------------------------------------------------------------------------------------------ resolution
    def upstream(self, ids):
        """Items whose state can change the state of `ids`: premises (transitively) and same-key competitors."""
        seen, todo = set(), list(ids)
        while todo:
            i = todo.pop()
            if i in seen or i not in self.items:
                continue
            seen.add(i)
            it = self.items[i]
            for a in it.asserts:
                todo.extend(a["just"])
            if it.key is not None:
                todo.extend(self.by_key[it.key])
        return seen

    def component(self, ids):
        """The connected component(s) of `ids`: premise edges both ways and shared keys."""
        seen, todo = set(), list(ids)
        while todo:
            i = todo.pop()
            if i in seen or i not in self.items:
                continue
            seen.add(i)
            it = self.items[i]
            for a in it.asserts:
                todo.extend(a["just"])
            todo.extend(self.dependents.get(i, ()))
            if it.key is not None:
                todo.extend(self.by_key[it.key])
        return seen

    def _statuses(self, ids):
        return {i: self.status(i) for i in ids}

    def _pending_flags(self):
        out = []
        for t, f in self.flags.items():
            p = f
            while p is not None and p["ended"] is None:
                p = self.flags.get(p["parent"]) if p["parent"] is not None else None
                if p is None:
                    out.append(t)
                    break
        return sorted(out)

    def _flag_covers(self, t, iid, ignore_age=False):
        f, it = self.flags[t], self.items.get(iid)
        if it is None:
            return False
        hit = (f["ids"] is not None and iid in f["ids"]) or (f["scope"] is not None and scope_matches(it.scope, f["scope"])) \
            or (f["key"] is not None and it.key is not None and list(it.key) == list(f["key"]))
        if not hit:
            return False
        if ignore_age:
            return True
        live = [a for a in it.asserts if a["t"] > max(it.retracted, default=-1)]
        if not live or min(a["t"] for a in live) > t:
            return False                                  # learned after the flag: not covered
        return not any(a["t"] > t and a["source"] in LIFTS_FLAG for a in live)

    def _touch_flag(self, t):
        f = self.flags[t]
        if f["ids"]:
            self._touched |= {i for i in f["ids"] if i in self.items}
        if f["scope"] is not None or f["key"] is not None:
            self._touched |= {i for i in self.items if self._flag_covers(t, i, ignore_age=True)}
        for c, g in self.flags.items():                   # children of this flag (rollbacks under it)
            if g["parent"] == t and g["ids"]:
                self._touched |= {i for i in g["ids"] if i in self.items}
        if not self._batch:
            self._flush()

    def _flush(self):
        if self._tick_dirty:                              # expiry depends on the clock
            self._touched |= {i for i, it in self.items.items() if it.reconfirm_after}
            self._tick_dirty = False
        if not self._touched:
            return
        comp = self.component(self._touched)
        self._touched = set()
        pending = self._pending_flags()
        flagged = {i for i in comp if any(self._flag_covers(t, i) for t in pending)}
        new = _resolve(self.items, comp, self.by_key, self.resolves, self.clock, flagged)
        for i in comp:
            if i in new:
                self.state[i] = new[i]
            else:
                self.state.pop(i, None)
        self._fp = None
        self._update_questions(comp)

    def _update_questions(self, comp):
        keys = {self.items[i].key for i in comp if self.items[i].key is not None}
        for k in sorted(keys):
            disp = sorted(i for i in self.by_key[k] if (self.state.get(i) or {}).get("status") == "disputed")
            if not disp:
                continue
            sig = (k, max(self.state[i].get("dispute_at") or 0 for i in disp))   # the dispute EVENT, not its content
            if sig not in self._open_disputes:
                self._open_disputes.add(sig)
                self.questions.append({"question": f"dispute@{sig[1]}", "key": list(k), "items": disp,
                                       "opened_at": sig[1], "asked_at": len(self.journal)})

    # ------------------------------------------------------------------------------------------------ reading
    def status(self, iid):
        """The item's status (STATUSES); "retracted" for an item with no live support (retracted, or derived from one)."""
        st = self.state.get(iid)
        if st is not None:
            return st["status"]
        return "retracted" if iid in self.items else None

    def stale(self, iid):
        """Must this item be re-confirmed before System 1 answers on it alone? Reads the pending flags (and the clock's
        expiry) — not the status, which a later write could have set back."""
        it = self.items.get(iid)
        if it is None:
            return False
        if any(self._flag_covers(t, iid) for t in self._pending_flags()):
            return True
        live = [a for a in it.asserts if a["t"] > max(it.retracted, default=-1)]
        return bool(it.reconfirm_after and live and self.clock - max(a["clock"] for a in live) > it.reconfirm_after)

    def usable(self, iid):
        """Is the item a fact System 1 may answer on alone: active and not stale?"""
        return self.status(iid) == "active" and not self.stale(iid)

    def item(self, iid):
        """The record of one item (the schema of DESIGN_KM §4) → dict, or None."""
        it = self.items.get(iid)
        if it is None:
            return None
        st = self.state.get(iid) or {}
        last = it.asserts[-1] if it.asserts else {}
        first = it.asserts[0] if it.asserts else {}
        version = 1
        if it.key is not None:
            order = sorted(self.by_key[it.key], key=lambda j: (self.items[j].asserts[0]["t"], j))
            version = order.index(iid) + 1
        return {"id": iid, "kind": it.kind, "body": it.body, "scope": it.scope, "source": last.get("source"),
                "by": last.get("by"), "evidence": [a.get("evidence") for a in it.asserts if a.get("evidence") is not None],
                "derived_from": sorted({p for a in it.asserts for p in a["just"]}),
                "confidence": {"n_confirm": st.get("n_confirm", 0), "n_refute": len(st.get("refuted_by") or ()),
                               "n_used": it.uses[0], "n_used_wrong": it.uses[1], "level": last.get("level")},
                "status": self.status(iid), "stale": self.stale(iid), "version": version,
                "supersedes": last.get("supersedes"),
                "valid": {"world": last.get("valid") or [None, None],
                          "known": [st.get("learned_at", first.get("t")), st.get("invalidated_at")],
                          "reconfirm_after": it.reconfirm_after},
                "of": last.get("of")}

    def find(self, *, kind=None, s=None, r=None, scope=None, status=None):
        """Records of the items that match every filter given (status: one or a tuple of STATUSES)."""
        sts = (status,) if isinstance(status, str) else status
        out = []
        for iid in sorted(self.items):
            it = self.items[iid]
            b = it.body if isinstance(it.body, dict) else {}
            if kind is not None and it.kind != kind:
                continue
            if s is not None and b.get("s") != s:
                continue
            if r is not None and b.get("r") != r:
                continue
            if scope is not None and not scope_matches(it.scope, scope):
                continue
            if sts is not None and self.status(iid) not in sts:
                continue
            out.append(self.item(iid))
        return out

    def active(self, key):
        """The active item on a key, or None."""
        for i in sorted(self.by_key.get(tuple(key), ())):
            if self.status(i) == "active":
                return i
        return None

    def disputes(self):
        """The open disputes (person questions not resolved yet) → [{"question", "key", "items", "opened_at"}]."""
        return [q for q in self.questions
                if any((self.state.get(i) or {}).get("status") == "disputed"
                       and (self.state[i].get("dispute_at") == q["opened_at"]) for i in q["items"])]

    def snapshot(self, *, about=None, relation=None, scope=None, kinds=None):
        """What a decision is given, as one plain fact: the items that match (`about`: a subject or a list of them;
        `relation`; `scope`; `kinds`, default every kind) — "facts" (active and not stale) and "hints" (hypotheses,
        expired or stale items, with why) — the ids they rest on, the store's fingerprint and journal position, and the
        query (so redecide can rebuild it). Disputed, refuted and retracted items are not given."""
        q = {"about": about, "relation": relation, "scope": scope, "kinds": None if kinds is None else list(kinds)}
        subjects = None if about is None else ({about} if isinstance(about, str) else set(about))
        facts, hints = [], []
        for iid in sorted(self.state):
            it = self.items[iid]
            b = it.body if isinstance(it.body, dict) else {}
            if kinds is not None and it.kind not in kinds:
                continue
            if subjects is not None and b.get("s") not in subjects:
                continue
            if relation is not None and b.get("r") != relation:
                continue
            if scope is not None and not scope_matches(it.scope, scope):
                continue
            st = self.status(iid)
            row = {"id": iid, "kind": it.kind, "body": it.body, "source": it.asserts[-1]["source"]}
            if st == "active" and not self.stale(iid):
                facts.append(row)
            elif st in ("active", "hypothesis", "expired"):
                hints.append({**row, "why": "stale" if st == "active" else st})
        ids = [x["id"] for x in facts + hints]
        return {"at": len(self.journal), "fp": self.fingerprint(), "query": q, "facts": facts, "hints": hints,
                "rests_on": sorted(self.upstream(ids)) if ids else []}

    def fingerprint(self):
        """A hash of what the store believes: every present item with its status, rank, support and time range."""
        if self._fp is None:
            rows = []
            for i in sorted(self.state):
                st, it = self.state[i], self.items[i]
                rows.append([i, it.kind, it.body, it.scope, st["status"], st["rank"], st["n_confirm"], st["refuted_by"],
                             st["learned_at"], st["invalidated_at"]])
            self._fp = vhash({"items": rows})[:16]
        return self._fp

    def counts(self):
        """{status: number of items} (absent items counted as "retracted")."""
        out = defaultdict(int)
        for i in self.items:
            out[self.status(i)] += 1
        return dict(out)

    def report(self):
        """What was learned and from whom, what is refuted, retracted, expired, disputed, flagged → a dict."""
        by_source = defaultdict(int)
        for it in self.items.values():
            if it.asserts:
                by_source[it.asserts[-1]["source"]] += 1
        return {"items": len(self.items), "journal": len(self.journal), "fingerprint": self.fingerprint(),
                "status": self.counts(), "by_source": dict(by_source),
                "refused": sum(r["op"] == "refused" for r in self.journal),
                "disputes_open": len(self.disputes()), "questions_asked": len(self.questions),
                "flags_pending": [{"flag": t, "why": self.flags[t]["why"]} for t in self._pending_flags()]}

    # ------------------------------------------------------------------------------------------------ re-deciding
    def redecide(self, storage, retracted, decide, *, fact="knowledge"):
        """The stored decisions that rested on retracted items, re-run on their re-built snapshots.

        storage: the decisions' TraceStorage; retracted: the retracted item ids; decide: init_state → {question: answer}
        (a function, or a System: its ask, not stored); fact: the given fact the snapshot was passed as. A decision is
        listed when its recorded snapshot changes once the retracted items' entries are left out of the journal up to
        its position; each is re-run on the old and the new snapshot. → (answer_changes, justification_only): lists of
        {"id", "old", "new"} — the first goes to a reviewer, the second is recorded."""
        retracted = set(retracted)
        run = _decider(decide)
        rebuilt, changes, same = {}, [], []
        for s in storage.iter():
            init = s.input or {}
            snap = init.get(fact)
            if not isinstance(snap, dict) or "at" not in snap or not retracted & set(snap.get("rests_on") or ()):
                continue
            at = int(snap["at"])
            if at not in rebuilt:
                rebuilt[at] = self.rebuild(upto=at, skip=retracted)
            new_snap = rebuilt[at].snapshot(**{k: v for k, v in (snap.get("query") or {}).items()})
            if _content(new_snap) == _content(snap):
                continue
            old, new = run(init), run({**init, fact: new_snap})
            (changes if old != new else same).append({"id": s.id, "old": old, "new": new})
        return changes, same


def _content(snap):
    return {"facts": snap.get("facts"), "hints": snap.get("hints")}


def _decider(decide):
    if callable(getattr(decide, "ask", None)):
        system = decide

        def run(init):
            res = system.ask(init, store=False)
            return {q: (r.answer if isinstance(r.answer, (str, int, float, bool, type(None))) else repr(r.answer), r.status)
                    for q, r in sorted(res.results.items())}
        return run
    return decide


# ---------------------------------------------------------------------------------------------------- the resolver
def _resolve(items, comp, by_key, resolves, clock, flagged, max_iter=200):
    """The state of the items in `comp` (a union of components) from their raw entries → {id: record} for present
    items. Fixed point of: liveness of justifications ↔ per-key walk. Deterministic; components do not interact."""
    order = sorted(comp, key=lambda i: (min((a["t"] for a in items[i].asserts), default=0), i))

    def live_asserts(it):
        cut = max(it.retracted, default=-1)
        return [a for a in it.asserts if a["t"] > cut]

    status = {i: "active" for i in order if live_asserts(items[i])}
    keys = sorted({items[i].key for i in order if items[i].key is not None})
    live, keyed, rank = {}, {}, {}
    for _ in range(max_iter):
        live, rank, arank = {}, {}, {}
        for i in order:                                 # liveness of each assertion under the current premise statuses
            la = [a for a in live_asserts(items[i]) if all(status.get(p) not in (None, "refuted") for p in a["just"])]
            if not la:
                continue
            live[i] = la
            r = 0
            for a in la:
                ra = RANK[a["source"]]
                for p in a["just"]:
                    ra = min(ra, rank.get(p, 0))
                arank[(i, a["t"])] = ra
                r = max(r, ra)
            rank[i] = r
        keyed = {}
        for k in keys:
            ids = [i for i in by_key[k] if i in live]
            keyed.update(_walk_key(ids, live, arank, {i: items[i].value for i in ids}, resolves.get(k, ())))
        new = {}
        for i in order:
            if i not in live:
                continue
            it = items[i]
            kr = keyed.get(i)
            if kr is not None and kr[0] in ("refuted", "disputed"):
                new[i] = kr[0]
                continue
            last = max(a["t"] for a in live[i])
            if it.retired and max(it.retired) > last:
                new[i] = "expired"
                continue
            s = "active"
            if it.kind in NEEDS_PROMOTION and not any(p > max(it.retracted, default=-1) for p in it.promoted):
                s = "hypothesis"
            if i in flagged:
                s = "hypothesis"
            if it.reconfirm_after and clock - max(a["clock"] for a in live[i]) > it.reconfirm_after:
                s = "expired"
            if s == "active" and not any(all(status.get(p) == "active" for p in a["just"]) for a in live[i]):
                s = "hypothesis"                        # a derived item is a fact only on active premises
            new[i] = s
        if new == status:
            break
        status = new
    else:
        raise RuntimeError(f"resolution did not converge on a component of {len(comp)} items")
    rec = {}
    for i, s in status.items():
        la = live[i]
        kr = keyed.get(i) or ("", None, [], None)
        rec[i] = {"status": s, "rank": rank[i], "n_confirm": len(la) - 1, "refuted_by": kr[2],
                  "learned_at": min(a["t"] for a in la), "invalidated_at": kr[1] if s == "refuted" else None,
                  "last_support": max(a["t"] for a in la), "dispute_at": kr[3] if s == "disputed" else None}
    return rec


def _walk_key(ids, live, arank, values, res):
    """One key's live assertions in time order → {id: (state, refuted_at, refuted_by, dispute_opened_at)}; state "" =
    holds the key."""
    if not ids:
        return {}
    ev = [(a["t"], 0, i, a) for i in ids for a in live[i]] + [(t, 1, v, by) for t, v, by in res]
    ev.sort(key=lambda e: (e[0], e[1], str(e[2])))
    holders, hval, hrank, disputed, refuted = set(), None, 0, set(), {}
    dopen = None                                        # the journal position the current dispute opened at
    hold_set = False
    for e in ev:
        if e[1] == 1:                                   # a person's answer to a dispute: that value wins
            v = e[2]
            if disputed:
                win = {i for i in disputed if values[i] == v}
                for i in disputed - win:
                    refuted[i] = (e[0], ["resolve"])
                holders, hval, hold_set = win, v, bool(win)
                hrank = max((arank[(i, a["t"])] for i in win for a in live[i] if a["t"] <= e[0]), default=0)
                disputed = set()
            continue
        t, _, i, a = e
        r, v = arank[(i, t)], values[i]
        if disputed:
            if a["source"] == "outcome":                # an observation ends a dispute
                for j in disputed:
                    if values[j] != v:
                        refuted[j] = (t, [i])
                holders = {j for j in disputed if values[j] == v} | {i}
                hval, hrank, disputed, hold_set = v, r, set(), True
                for j in holders:
                    refuted.pop(j, None)
            else:
                disputed.add(i)
                refuted.pop(i, None)
            continue
        if not hold_set:
            holders, hval, hrank, hold_set = {i}, v, r, True
            refuted.pop(i, None)
            continue
        if v == hval:                                   # a confirmation
            holders.add(i)
            hrank = max(hrank, r)
            refuted.pop(i, None)
            continue
        if a.get("supersedes") in holders or a["source"] == "outcome" or r > hrank:
            for h in holders:
                refuted[h] = (t, [i])
            holders, hval, hrank = {i}, v, r
            refuted.pop(i, None)
        else:                                           # equal or higher rank on the other side: a person decides
            disputed = holders | {i}
            dopen = t
            holders, hval, hrank, hold_set = set(), None, 0, False
            refuted.pop(i, None)
    out = {}
    for i in ids:
        if i in disputed:
            out[i] = ("disputed", None, [], dopen)
        elif i in holders:
            out[i] = ("", None, [], None)
        elif i in refuted:
            out[i] = ("refuted", refuted[i][0], refuted[i][1], None)
        else:
            out[i] = ("", None, [], None)
    return out


__all__ = ["item_id", "key_of", "KINDS", "KnowledgeStore", "NEEDS_PROMOTION", "RANK", "RECONFIRM_DEFAULTS",
           "scope_matches", "SelfDefeating", "SOURCES", "STATUSES", "Verdict"]
