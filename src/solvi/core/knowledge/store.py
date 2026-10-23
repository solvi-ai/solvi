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

State is a function of the live journal entries (JTMS-style; `valid=` is valid time and the journal position is
transaction time, as in a bitemporal store): an item is present while it has a live assertion whose
premises (`derived_from`) are present and not refuted. Per key (scope, subject, relation) the live assertions are walked
in time order: the same value confirms; an outcome refutes the current value (the world changed) — unless the outcome
assertion is `inferred` (a consequence computed from premises, not an observation: it competes by rank, so two inferences
of different values from equal-rank premises are a dispute); a higher-ranked source
(outcome > person > spec > verified) refutes a lower one; `supersedes` (a producer's own new version) refutes the version
it names; any other contradiction makes every side "disputed" and asks a person — one question per dispute EVENT (the
journal position it opened at): the same two items disputed again after a resolution are asked again. `resolve(key,
value, by=)` is the person's answer. A disputed item is never a fact.

Validity. `valid=(since, until)` is when a value holds in the world (half-open, None unbounded — a missing since is
the beginning of time, and an assertion without `valid` on a key where others carry one is the beginning-of-time
version, open-ended; ints, ISO date strings or datetime.date, which is kept as its ISO text). On a key where some
assertion carries one, values compete only when their intervals overlap (a dose label for 2023 and another from
mid-2024 are both facts, for their times). Versions of different sinces are versions in time, not a contradiction:
the later since closes the earlier version at that since and the earlier stays active (its effective interval ends
there; a bounded later version gives the earlier value back after its until; a retracted later version re-opens what
it closed), whatever order they were written in. Validity is per assertion: a value told for two periods (12, then 24,
then 12 again) holds in both under one id; `valid(id)` is the latest assertion's interval, `periods(id)` the effective
intervals and `item(id)["valid"]` lists every claim. A version is one item's claims of one since: when they give
different untils, the latest claim's until is the version's (a correction, as one validity per item was), so (45, 55)
then (45, None) reads open from 45 and the reverse order reads bounded at 55. The same since from two writers (two missing sinces too) is not an
observation of a change — rank decides, equal ranks dispute; that holds while any other writer still asserts the old
value, even when the new value's writer had asserted it too. A writer's later value at one since replaces its own
earlier one there (it no longer asserts the old value), so two writers who both move from one value to the same other
value leave that value holding. The winner takes only the overlap: a loser whose interval
reaches beyond it is closed there and holds outside (refuted only when the winner covers it whole), and a dispute is
contested only on the overlap — outside it each side holds; a person's answer (`resolve`) closes the loser the same
way. An item pending re-test takes no side on its key (no dispute, no refutation, no closing). `usable(id, valid_at=)`
and `snapshot(valid_at=)` read the effective intervals (a snapshot row carries "periods" when they are not the item's
own `valid`; `Knowledge.value(valid_at=)` reads them). An item derived from a premise disputed only on an overlap is a
hypothesis as a whole, but at a time where every premise of one of its derivations holds it is usable and a fact. Keys whose assertions carry no `valid` resolve as described
above.

Statuses: "active" (a fact System 1 may answer on), "hypothesis" (a hint: a behaviour-changing item not yet promoted
by the write gates, a derived item on a premise that is not active, an item under a pending flag, or one pending a
re-test), "disputed",
"refuted", "expired" (its reconfirm_after ran out on the store's clock, or retired) and "retracted".

Re-test. Content written again after it was retracted — the same item id, since an id is its content — is not revived
by the new assertion: it is a hypothesis "pending re-test" until `retest(id, ok=True)` (a person or a test confirms
it; ok=False retracts it again), and so is every item that rested on it when it was retracted, each until its own
re-test. `pending()` lists the items waiting, `item(id)["pending"]` says why.

The guard on content told with `valid`. Every retraction (plain, `only_from`, `assertion=`, `retract_source`, a failed
re-test, a rollback) withdraws its assertions' content (the key's value) on their validity. A later assertion of the
same content on an overlapping validity, whatever its item id, is a copy: it waits for a re-test (the item is pending;
when the item has other claims they still hold, and a failed re-test takes back only the copies) — unless writers it
was not withdrawn from still hold that content wherever the copy could hold; such a copy is admitted and marked
"unchecked" in `item(id)["valid"]["claims"]`, and it goes pending once those holders are cut. A passed re-test vouches
for the item's assertions before it and clears the withdrawals of its content. Derived and inferred assertions follow
their premises instead. Items derived after the re-test are fresh; an item a rule only re-infers (`inferred=True`) follows its
premises, and so does an inferred item that rested on one: once the premise passes its re-test it is a fact again
without a re-test of its own. `pending_retest(id)` reads it.

Derived validity. An item with `derived_from` and no `valid` of its own gets the intersection of its premises'
validity (each premise's span over its live claims); when they never hold together, or their times do not compare
(an int and an ISO date), it is refused with the reason in the journal. It holds at a time (`usable(id, valid_at=)`,
`snapshot(valid_at=)`, `periods(id)`) only where, for one of its derivations, every premise holds — a premise closed by
a later version no longer carries it; an item with its own `valid` over premises whose times do not compare reads its
own `valid` only.

Staleness. `flag(scope=..., why=...)` (a drift or open-set flag) or `reconfirm(id)` marks items for re-confirmation: an
item learned before the flag is a hint, not a fact, until a newer observation or person confirms it or the flag ends
(`end_flag`). A flag is read where staleness is asked (`stale(id)`), not inferred from the item's status. `rollback(id)`
(supervision undoing an update) restores the previous version as a hint, never as a fact, while a flag covering the
rolled-back version or its scope is pending (a rollback during a pending drift flag once restored a stale threshold as
a fact on a shifted stream, and a stale check that read the item status missed it). `reconfirm_after` per
(kind, source) (RECONFIRM_DEFAULTS, the store's clock advanced by `tick`) expires items that weaker evidence keeps. A tick
re-resolves only the items whose expiry can flip between the old and the new clock (an index of their deadlines).

Every write — add, promote, refuse, retract, flag, resolve, tick, use — is a journal entry: in a TraceStorage (records of
kind "knowledge", in the same hash chain as the decisions when it is the decisions' store) or in memory with its own hash
chain. `verify()` checks the chain and that the state is what the journal gives; `rebuild(upto=n, skip=ids)` gives the
store as it was at entry n, or as if `ids` had never been written; a store opened on a backend is replayed from its
journal (an entry edited by hand breaks the chain). `snapshot(...)` returns what a decision reads, with the store's
fingerprint and journal position, so every decision that is given it records the knowledge it used and replays;
`redecide(storage, retracted, decide)` re-runs the decisions that rested on retracted items on their re-built snapshots
and splits them into "answer changes" (to a reviewer) and "justification only" (recorded).

Measured by benchmarks/knowledge/retraction.py: 1,000 of 1,000 random retractions exact on a 10,000-item synthetic store (the store after a retraction has the fingerprint of the store rebuilt without it). Cost grows with the connected component a write
touches (premise edges and shared keys); the store has been measured to 10,000 items, not 10⁵–10⁶."""
from __future__ import annotations

import datetime as _dt
import json
import threading
from bisect import bisect_left, insort
from collections import defaultdict
from contextlib import contextmanager

from ..runtime import vhash
from .gates import RANK, SOURCES, Verdict, WriterGate, default_gates, with_source_gate

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
                 "reconfirm_after", "uses", "cuts", "acuts", "rechecked")

    def __init__(self, iid, kind, body, scope, key, value, reconfirm_after):
        self.id, self.kind, self.body, self.scope, self.key, self.value = iid, kind, body, scope, key, value
        self.asserts = []           # [{"t", "clock", "source", "by", "just", "supersedes", "evidence", "valid", "level",
        #                               "of", "inferred"}]
        self.promoted = []          # [t]
        self.retracted = []         # [t]
        self.cuts = []              # [(t, writer)]: one writer's assertions taken back (retract(..., only_from=))
        self.acuts = []             # [(t, n)]: one assertion taken back (retract(..., assertion=n))
        self.rechecked = []         # [t]: passed re-tests (retest(id, ok=True))
        self.retired = []           # [t]
        self.reconfirm_after = reconfirm_after
        self.uses = [0, 0]          # [used, used and wrong]


def _acut(it, a):
    """The journal position assertion `a` was taken back at by its writer's or its own cut, or None."""
    return min([tc for tc, w in it.cuts if tc > a["t"] and w == a["by"]]
               + [tc for tc, n in it.acuts if n == a["t"] and tc > a["t"]], default=None)


def _cut(it):
    """The last journal position the item's whole support was taken back at (a retraction, or a writer's or an
    assertion's retraction that left no other assertion standing), or -1."""
    cut = max(it.retracted, default=-1)
    if it.acuts:
        for tc in sorted({tc for tc, _ in it.cuts} | {tc for tc, _ in it.acuts}):
            if tc > cut and all((x := _acut(it, a)) is not None and x <= tc for a in it.asserts if cut < a["t"] < tc):
                cut = tc
        return cut
    for tc, _ in sorted(it.cuts):
        if tc > cut and not any(cut < a["t"] < tc and not any(a["t"] < t2 <= tc and w2 == a["by"] for t2, w2 in it.cuts)
                                for a in it.asserts):
            cut = tc
    return cut


def _live(it):
    """The item's assertions still standing: after its last retraction, and not taken back with their writer's or on
    their own."""
    if it.acuts:
        cut = _cut(it)
        return [a for a in it.asserts if a["t"] > cut and _acut(it, a) is None]
    if not it.cuts:
        cut = max(it.retracted, default=-1)
        return [a for a in it.asserts if a["t"] > cut]
    cut = _cut(it)
    return [a for a in it.asserts if a["t"] > cut and not any(tc > a["t"] and w == a["by"] for tc, w in it.cuts)]


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


def _time(x):
    """A world time as the store keeps it: a datetime.date or datetime.datetime as its ISO text (so it is plain JSON
    and compares with ISO date strings), anything else as it is."""
    return x.isoformat() if isinstance(x, (_dt.date, _dt.datetime)) else x


def _interval(valid):
    """A `valid` as (since, until); None (no valid given) is unbounded on both sides."""
    if not valid:
        return (None, None)
    return (valid[0], valid[1] if len(valid) > 1 else None)


def _lo_lt(a, b):
    """since a < since b, None the beginning of time (below every time)."""
    if a is None:
        return b is not None
    return b is not None and a < b


def _hi_lt(a, b):
    """until a < until b, None the end of time (above every time)."""
    if b is None:
        return a is not None
    return a is not None and a < b


def _same_since(a, b):
    """Do two sinces compete as one (equal, both missing, or times that do not compare)? Different sinces are two
    versions in time: the later closes the earlier, neither refutes the other."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return a == b or not (a < b or b < a)
    except TypeError:
        return True


def _ivs_and(xs, ys):
    """The intersection of two sets of half-open intervals ([[since, until], ...]; None: no limit on that side)."""
    out = []
    for a1, b1 in xs:
        for a2, b2 in ys:
            lo = a2 if _lo_lt(a1, a2) else a1
            hi = b1 if _hi_lt(b1, b2) else b2
            if lo is None or hi is None or lo < hi:
                out.append([lo, hi])
    return _ivs_or(out)


def _ivs_or(xs):
    """The union of half-open intervals, merged and in order."""
    xs = sorted(([a, b] for a, b in xs), key=lambda v: (v[0] is not None, v[0] if v[0] is not None else 0))
    out = []
    for a, b in xs:
        if out and (out[-1][1] is None or not _lo_lt(out[-1][1], a)):     # touches or overlaps the last one
            if _hi_lt(out[-1][1], b):
                out[-1][1] = b
        else:
            out.append([a, b])
    return out


def _segments(ivs):
    """The elementary segments of the line cut at every since and until of `ivs` → [(lo, hi)] (None: unbounded); an
    interval covers a segment whole or not at all. Raises TypeError when the times do not compare."""
    pts = sorted({x for a, b in ivs for x in (a, b) if x is not None})
    if not pts:
        return [(None, None)]
    return [(None, pts[0])] + list(zip(pts, pts[1:])) + [(pts[-1], None)]


def _covers_seg(iv, seg):
    (a, b), (lo, hi) = iv, seg
    return (a is None or (lo is not None and not lo < a)) and (b is None or (hi is not None and not b < hi))


def _effective(units):
    """Effective intervals on one key with `valid`: at each world time the value of the in-play versions that cover it
    with the latest since holds (a version closes an earlier one at its own since; a bounded one gives the earlier value
    back after its until); a version that took another's overlap (a same-since contest it won) beats it there; two
    values left at one time are contested there. units: [(unit, item id, value (JSON text), (since, until), lost_to
    (units))] → {item id: {"periods": [[lo, hi], ...], "contested": [[lo, hi], ...]}}, or None when the times on the
    key do not compare."""
    try:
        units = [u for u in units if u[3][0] is None or u[3][1] is None or u[3][0] < u[3][1]]   # empty: never holds
        order = sorted(range(len(units)), key=lambda k: (units[k][3][0] is not None, units[k][3][0]
                                                          if units[k][3][0] is not None else 0))
        rank = {k: n for n, k in enumerate(order)}      # position by since (None first)
        starts, ends = defaultdict(list), defaultdict(list)
        active = []                                     # the covering versions' positions, in since order
        for k, u in enumerate(units):
            if u[3][0] is None:
                insort(active, rank[k])
            else:
                starts[u[3][0]].append(rank[k])
            if u[3][1] is not None:
                ends[u[3][1]].append(rank[k])
        out = {u[1]: {"periods": [], "contested": []} for u in units}
        for seg in _segments([u[3] for u in units]):
            if seg[0] is not None:
                for n in ends.get(seg[0], ()):
                    if n in active:
                        active.remove(n)
                for n in starts.get(seg[0], ()):
                    insort(active, n)
            if not active:
                continue
            top = units[order[active[-1]]][3][0]
            best = []
            for n in reversed(active):
                u = units[order[n]]
                if _lo_lt(u[3][0], top):
                    break
                best.append(u)
            names = {u[0] for u in best}
            best = [u for u in best if not (u[4] & names)]
            vals = {u[2] for u in best}
            for u in best:
                out[u[1]]["periods" if len(vals) == 1 else "contested"].append([seg[0], seg[1]])
        return {i: {k: _ivs_or(v) for k, v in r.items()} for i, r in out.items()}
    except TypeError:
        return None


def _overlap(v1, v2):
    """Do two half-open validity intervals [since, until) share a moment? Times that do not compare overlap (the
    conservative answer: the values compete)."""
    (a1, b1), (a2, b2) = _interval(v1), _interval(v2)
    try:
        return (b2 is None or a1 is None or a1 < b2) and (b1 is None or a2 is None or a2 < b1)
    except TypeError:
        return True


def covers(valid, at):
    """Is the world time `at` inside an item's validity [since, until)? (None: unbounded; no valid: always.) A
    datetime.date is read as its ISO text."""
    since, until = _interval(valid)
    at = _time(at)
    try:
        return (since is None or since <= at) and (until is None or at < until)
    except TypeError:
        raise ValueError(f"valid_at={at!r} does not compare with the item's valid {list(valid)!r}") from None


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
    ConsistencyGate); the source check runs first whatever is given. defaults: reconfirm_after per (kind, source).
    writers: {writer: source (or sources), ...} — opt-in: a source (a rank) is then taken only from a writer (`by=`)
    registered with it; any other item is refused with the reason (gates.WriterGate, run right after the source
    check). None (default): any writer may declare any source, as before."""

    def __init__(self, storage=None, *, gates=None, defaults=None, writers=None, _replay=True):
        from ..store import open_storage
        self.storage = open_storage(storage)
        self.gates = with_source_gate(gates if gates is not None else default_gates())
        if writers is not None:
            self.gates.insert(1, WriterGate(writers))       # right after the source check
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
        self._expiry = []                            # sorted (clock an item's expiry can flip at, id): the tick index
        self._resolved_clock = 0                     # the clock the state was last resolved at
        self._fp = None
        self._taints = {}                            # id → the journal position its support was last cut at (memo)
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
            from ..store import _stamp
            rec = self._mem.append(_stamp(body))     # run_context: the in-memory journal carries the run too
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
                if r.get("retest"):
                    self.items[r["item"]].rechecked.append(t)
                self._touch(r["item"])
        elif op == "retract_assertions":                 # its own op: a store that knows only "retract" refuses
            if r["item"] in self.items:                  # it loudly instead of reading a whole retraction
                self.items[r["item"]].acuts.extend((t, n) for n in r["assertions"])
                self._touch(r["item"])
        elif op == "retract":
            if r["item"] in self.items:
                if r.get("assertions") is not None:
                    self.items[r["item"]].acuts.extend((t, n) for n in r["assertions"])
                elif r.get("only_from") is not None:
                    self.items[r["item"]].cuts.append((t, r["only_from"]))
                else:
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
        elif op in ("refused", "held", "note", "observe", "agenda", "rollback", "forget"):
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
                           "of": r.get("of"), "inferred": bool(r.get("inferred")),
                           "consulted": tuple(r.get("consulted") or ())})
        for p in it.asserts[-1]["just"] + it.asserts[-1]["consulted"]:     # a consulted item's change re-resolves it
            self.dependents[p].add(iid)
        if it.reconfirm_after:                          # this assertion's clock is a possible last support: past
            insort(self._expiry, (it.asserts[-1]["clock"] + it.reconfirm_after, iid))   # it, the item expires
        self._touch(iid)

    def proposal(self, kind, body, scope=None, *, source, by=None, evidence=None, derived_from=(), supersedes=None,
                 reconfirm_after=None, valid=None, level=None, of=None, inferred=False, consulted=()):
        """A proposed item as the write gates see it (a dict with the record's fields; nothing is written)."""
        if kind not in KINDS:
            raise ValueError(f"kind is one of {KINDS}, not {kind!r}")
        if inferred and not derived_from:
            raise ValueError("an inferred assertion names the premises it is computed from (derived_from=)")
        consulted = sorted(set(consulted) - set(derived_from))
        unknown = [c for c in consulted if c not in self.items]
        if unknown:
            raise ValueError(f"consulted items not in the store: {unknown}")
        body, scope = plain_json(body), plain_json(scope or {})
        if reconfirm_after is None:
            reconfirm_after = self.defaults.get((kind, source))
        return {"id": item_id(kind, body, scope), "kind": kind, "body": body, "scope": scope, "source": source,
                "by": None if by is None else str(by), "evidence": plain_json(evidence) if evidence is not None else None,
                "derived_from": sorted(set(derived_from)), "supersedes": supersedes, "reconfirm_after": reconfirm_after,
                "valid": plain_json([_time(x) for x in valid]) if valid is not None else None, "level": level,
                "of": None if of is None else str(of), "key": key_of(body, scope), "inferred": bool(inferred),
                "consulted": consulted}

    def add(self, kind, body, scope=None, *, source, by=None, evidence=None, derived_from=(), supersedes=None,
            reconfirm_after=None, valid=None, level=None, of=None, shadow=None, inferred=False, consulted=()):
        """Propose an item → its id, or None when a gate refused it (a "refused" entry in the journal says why).

        kind: one of KINDS; body: plain JSON (a fact: {"s", "r", "o"}); scope: a dict (a question, an environment);
        source: "person" | "outcome" | "spec" | "verified" (anything else is refused: never the system's own answers);
        by: who; evidence: stored ids, quotes, steps; derived_from: premise item ids (the retraction cascade follows them);
        supersedes: the id of the producer's previous version; reconfirm_after: clock units until it needs re-confirmation
        (default RECONFIRM_DEFAULTS); valid: (since, until) in the world, as you count time; level: a guarantee level
        (for "verified"); of: the stored decision a "verified" item is (checked in the store's TraceStorage); shadow:
        passed to the write gates (a shadow measurement for a behaviour-changing item); inferred: the assertion is a
        consequence computed from `derived_from` (a rule applied to facts), not an observation — an inferred outcome
        does not refute the current value as an observation does: it competes by rank, and equal ranks are a dispute.
        consulted: item ids the conclusion looked at besides its premises — the evidence for the alternatives it did not
        choose (opt-in). Retracting one does not retract this item (its premises stand) but marks it for review: a
        hypothesis pending re-test (`retest(id, ok=True)`), as an item resting on a retracted premise; `item(id)
        ["consulted"]` lists them. Such a store differs from its rebuild without the retracted item until the re-test.

        A fact or an episode the gates admit is present at once; a rule, skill or action the gates admit is promoted (its
        verdicts recorded); one a later gate holds stays a hypothesis (a "held" entry says why)."""
        with self._lock:
            if valid is None and derived_from:              # a derived item holds where all its premises hold
                valid, why = self._derived_valid(derived_from)
                if why is not None:
                    prop = self.proposal(kind, body, scope, source=source, by=by, derived_from=derived_from)
                    self._write("refused", item=prop["id"], item_kind=kind, source=source, by=prop["by"],
                                why=f"validity: {why}")
                    return None
        prop = self.proposal(kind, body, scope, source=source, by=by, evidence=evidence, derived_from=derived_from,
                             supersedes=supersedes, reconfirm_after=reconfirm_after, valid=valid, level=level, of=of,
                             inferred=inferred, consulted=consulted)
        with self._lock:
            verdicts = []
            for g in self.gates:
                v = g.admit(self, prop, shadow)
                v.gate = v.gate or getattr(g, "name", type(g).__name__)
                verdicts.append(v)
                if not v.admit and (v.gate in ("source", "writers", "consistency") or kind not in NEEDS_PROMOTION):
                    self._write("refused", item=prop["id"], item_kind=kind, source=source, by=prop["by"],
                                why=f"{v.gate}: {v.reason}")
                    return None
            rec = self._write("assert", item=prop["id"], item_kind=kind, body=prop["body"], scope=prop["scope"],
                              source=source, by=prop["by"], evidence=prop["evidence"], just=prop["derived_from"],
                              supersedes=supersedes, reconfirm_after=prop["reconfirm_after"], valid=prop["valid"],
                              level=level, of=prop["of"], **({"inferred": True} if inferred else {}),
                              **({"consulted": prop["consulted"]} if prop["consulted"] else {}))
            with self.batch():
                self._raw_assert(rec, rec["n"])
                if kind in NEEDS_PROMOTION:
                    if self._needs_retest(prop["id"]):
                        verdicts.append(Verdict(False, "written again after a retraction: pending re-test "
                                                       "(retest(id, ok=True))", gate="retest"))
                    held = [v for v in verdicts if not v.admit]
                    if held:
                        self._write("held", item=prop["id"], verdicts=[v.to_dict() for v in verdicts])
                    else:
                        self.promote(prop["id"], measured=[v.to_dict() for v in verdicts])
            return prop["id"]

    def _derived_valid(self, premises):
        """The validity of an item derived from `premises` that gives none of its own: the intersection of the premises'
        validity (each premise's span: from its earliest live since to its latest until) → (valid or None, None), or
        (None, why) when they never hold together or their times do not compare (an int and an ISO date). None when no
        premise carries `valid` (the item is as before)."""
        ivs = []
        for p in sorted(set(premises)):
            it = self.items.get(p)
            if it is None:
                continue
            claims = [_interval(a["valid"]) for a in (_live(it) or it.asserts) if a.get("valid") is not None]
            if not claims:
                continue
            lo, hi = claims[0]
            for a, b in claims[1:]:
                lo = a if _lo_lt(a, lo) else lo
                hi = b if _hi_lt(hi, b) else hi
            ivs.append([[lo, hi]])
        if not ivs:
            return None, None
        acc = ivs[0]
        try:
            for x in ivs[1:]:
                acc = _ivs_and(acc, x)
        except TypeError:
            return None, f"the premises' validity times do not compare: {ivs}"
        if not acc:
            return None, "the premises' validity intervals do not intersect: " + ", ".join(
                f"{p} {self.valid(p)}" for p in sorted(set(premises)) if self.valid(p) is not None)
        return [acc[0][0], acc[0][1]], None

    def promote(self, iid, measured=None):
        """A behaviour-changing item passed its gates (or a person promotes it): it becomes a fact (measurements kept)."""
        with self._lock:
            rec = self._write("promote", item=iid, measured=measured)
            self._apply(rec)

    def retract(self, iid, *, by=None, why=None, only_from=None, assertion=None):
        """Take an item back, with everything derived from it (the cascade over derived_from); what it refuted or
        disputed comes back. Nothing is deleted: the journal keeps the entry and the chain stays whole.
        only_from: a writer (the `by` of its assertions) — take back only that writer's assertions of the item: the same
        content told by others stays (one item for both, since an id is its content); the item goes only when no other
        assertion is left. only_from=(writer, evidence): only that writer's assertions given with this evidence (one of
        two publications of the same content). assertion: the journal position of one standing assertion of the item
        (`item(id)["valid"]["claims"]` lists them as "at"; a list takes several) — only it is taken back (a journal
        entry of its own, "retract_assertions"; a duplicate taken back while its writer's other claim of the content
        stands is not a withdrawal of that content). → {id: (status before, status after)} of every item whose status
        changed."""
        with self._lock:
            if iid not in self.items:
                raise KeyError(f"no item {iid!r}")
            if only_from is not None and assertion is not None:
                raise ValueError("give only_from= or assertion=, not both")
            extra = {}
            live = _live(self.items[iid])
            if isinstance(only_from, (tuple, list)):
                if len(only_from) != 2:
                    raise ValueError("only_from=(writer, evidence) names one writer and the evidence it gave")
                w, ev = str(only_from[0]), plain_json(only_from[1])
                assertion = [a["t"] for a in live if a["by"] == w and a.get("evidence") == ev]
                if not assertion:
                    raise ValueError(f"item {iid!r} has no standing assertion from {w!r} with evidence {ev!r}")
                only_from = None
            if assertion is not None:
                ns = sorted({int(n) for n in (assertion if isinstance(assertion, (list, tuple)) else [assertion])})
                missing = [n for n in ns if not any(a["t"] == n for a in live)]
                if missing or not ns:
                    raise ValueError(f"item {iid!r} has no standing assertion at journal entry {missing or ns}")
                extra = {"assertions": ns}
            elif only_from is not None:
                only_from = str(only_from)
                if not any(a["by"] == only_from for a in live):
                    raise ValueError(f"item {iid!r} has no standing assertion from {only_from!r}")
                extra = {"only_from": only_from}
            self._settle()
            before = self._statuses(self.component([iid]))
            rec = self._write("retract_assertions" if "assertions" in extra else "retract", item=iid,
                              by=None if by is None else str(by), why=why, **extra)
            self._apply(rec)
            after = self._statuses(before)
            return {i: (before[i], after[i]) for i in before if before[i] != after[i]}

    def retract_source(self, writer, *, by=None, why=None):
        """Take back every standing assertion from `writer` (an exposed source): items only it told go with their
        derivations; items others told too stay on their support. → {id: (status before, status after)}."""
        writer = str(writer)
        with self._lock:
            self._settle()
            hit = sorted(i for i, it in self.items.items() if any(a["by"] == writer for a in _live(it)))
            before = self._statuses(self.component(hit)) if hit else {}
            with self.batch():
                for i in hit:
                    rec = self._write("retract", item=i, by=None if by is None else str(by), why=why, only_from=writer)
                    self._apply(rec)
            after = self._statuses(before)
            return {i: (before[i], after[i]) for i in before if before[i] != after[i]}

    def forget(self, writer, storage=None, decide=None, *, by=None, why=None, fact="knowledge"):
        """Forget a source in one call: what `writer` told is taken back, the stored decisions that rested on it are
        re-decided on the journal as it is now, and one "forget" journal entry records what was taken back and which
        decisions changed. The three steps are the public calls

            changed = store.retract_source(writer, by=by, why=why)          # its items, and what was derived from them
            gone = [i for i, (_, now) in changed.items() if now != "active"]
            answer_changes, same = store.redecide(storage, gone, decide, fact=fact, upto="now")

        then the entry. storage: the decisions' TraceStorage (None: only the retraction and the entry); decide: what
        re-decides an input (a System, or init_state → {question: answer}; needed with storage); fact: the given fact
        the snapshots were passed as. storage may be a path (opened as `System(storage=)` opens one, before anything is
        written). Nothing is erased: the journal keeps the source's entries (retracted) and the
        decision records keep their inputs — erasing personal data in a record is `TraceStorage.redact(id)`, kept
        explicit (`where_is` finds the records holding a value). → {"source", "retracted" ({id: (status before, status
        after)}), "decisions" (the ids of the stored decisions whose snapshot changed), "changed" / "same" (as
        redecide: {"id", "old", "new"}; the first are for a reviewer), "entry" (the forget entry's journal position)}."""
        if storage is not None and decide is None:
            raise ValueError("forget(writer, storage, decide): give decide= (a System, or init_state → {question: "
                             "answer}) to re-decide the stored decisions, or no storage")
        storage = _storage(storage)                     # a path is opened before anything is written: a storage that
        writer = str(writer)                            # cannot be opened leaves the journal as it was
        retracted = self.retract_source(writer, by=by, why=why)
        gone = sorted(i for i, (_, now) in retracted.items() if now != "active")
        changed, same = ([], []) if storage is None or not gone else self.redecide(storage, gone, decide, fact=fact,
                                                                                   upto="now")
        with self._lock:
            rec = self._write("forget", source=writer, by=None if by is None else str(by), why=why, retracted=gone,
                              changed=[c["id"] for c in changed], same=[c["id"] for c in same])
            self._apply(rec)
        return {"source": writer, "retracted": retracted, "decisions": [c["id"] for c in changed + same],
                "changed": changed, "same": same, "entry": rec["n"]}

    def retest(self, iid, *, ok, by=None, why=None):
        """The re-test of an item pending one (re-asserted after a retraction, resting on one that was, or a copy of
        withdrawn content): ok=True — it is a fact again (a "promote" entry; its premises must be active too); ok=False —
        it is retracted again (only its pending copies when other claims of it hold)."""
        with self._lock:
            if iid not in self.items:
                raise KeyError(f"no item {iid!r}")
            if not ok:
                self._settle()
                st = self.state.get(iid) or {}
                copies = st.get("pending_claims") or []
                if copies and any(a["t"] not in copies for a in _live(self.items[iid])):
                    return self.retract(iid, by=by, why=why or "failed re-test", assertion=copies)  # only the copies
                return self.retract(iid, by=by, why=why or "failed re-test")
            rec = self._write("promote", item=iid, measured=None, retest=True, by=None if by is None else str(by),
                              why=why)
            self._apply(rec)

    def pending_retest(self, iid):
        """Is the item waiting for a re-test (see the module docstring)?"""
        self._settle()
        return bool((self.state.get(iid) or {}).get("pending_retest"))

    def pending(self):
        """The items waiting for a re-test (`retest(id, ok=)`), sorted; `item(id)["pending"]` says why."""
        self._settle()
        return sorted(i for i, st in self.state.items() if st.get("pending_retest"))

    def _needs_retest(self, iid):
        it = self.items.get(iid)
        if it is None:
            return False
        t = _taint(self.items, iid, self._taints)
        return t is not None and not any(p > t for p in it.promoted)

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
            mine = [_interval(a["valid"]) for a in _live(it) if a.get("valid") is not None]
            for j in self.by_key[it.key] if mine else ():     # the versions it closed (an earlier since it overlaps)
                jt = self.items[j]
                if j == iid or jt.value == it.value:
                    continue
                for b in _live(jt):
                    jv = _interval(b.get("valid"))
                    try:
                        if any(_overlap(jv, m) and _lo_lt(jv[0], m[0]) for m in mine):
                            out.add(j)
                            break
                    except TypeError:
                        continue
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

    def downstream(self, ids):
        """Items whose state a change to `ids` can change: `ids`, what is derived from them (transitively) and same-key
        competitors — what a write re-resolves (the rest keeps its state)."""
        seen, todo = set(), list(ids)
        while todo:
            i = todo.pop()
            if i in seen or i not in self.items:
                continue
            seen.add(i)
            todo.extend(self.dependents.get(i, ()))
            k = self.items[i].key
            if k is not None:
                todo.extend(self.by_key.get(k, ()))
        return seen

    def _statuses(self, ids):
        return {i: self._status(i) for i in ids}

    def _status(self, iid):                              # the resolved state as it stands (no flush)
        st = self.state.get(iid)
        if st is not None:
            return st["status"]
        return "retracted" if iid in self.items else None

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
        live = _live(it)
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

    def _expiring(self, c0, c1):
        """Items whose expiry can differ between clocks c0 and c1: an item expires once the clock is past the clock of
        its last live support plus its reconfirm_after, and every assertion's such deadline is in the index (a superset
        of the deadlines resolution can use), so only items with a deadline between the two clocks can flip (a margin of
        one clock unit on each side keeps it exact under float rounding)."""
        lo, hi = (c0, c1) if c0 <= c1 else (c1, c0)
        a, b = bisect_left(self._expiry, (lo - 1,)), bisect_left(self._expiry, (hi + 1,))
        return {i for _, i in self._expiry[a:b]}

    def _flush(self):
        if self._tick_dirty or self.clock != self._resolved_clock:    # expiry depends on the clock: the index names
            self._touched |= self._expiring(self._resolved_clock, self.clock)   # the items whose expiry can flip
            self._resolved_clock, self._tick_dirty = self.clock, False
        if not self._touched:
            return
        comp = self.downstream(self._touched)           # upstream items keep their state: nothing here changes it
        self._touched = set()
        for i in comp:
            self._taints.pop(i, None)
        pending = self._pending_flags()
        flagged = {i for i in comp if any(self._flag_covers(t, i) for t in pending)}
        new = _resolve(self.items, comp, self.by_key, self.resolves, self.clock, flagged, taints=self._taints,
                       state=self.state)
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
        """The item's status (STATUSES); "retracted" for an item with no live support (retracted, or derived from one).
        Inside a batch the writes so far are resolved first: an item added in the batch reads as what it is, not as
        absent (a premise added earlier in the same batch passes the consistency gate)."""
        self._settle()
        return self._status(iid)

    def _settle(self):                                   # resolve the writes a batch has deferred, before a read
        if self._touched or self._tick_dirty:
            with self._lock:
                self._flush()

    def stale(self, iid):
        """Must this item be re-confirmed before System 1 answers on it alone? Reads the pending flags (and the clock's
        expiry) — not the status, which a later write could have set back."""
        it = self.items.get(iid)
        if it is None:
            return False
        if any(self._flag_covers(t, iid) for t in self._pending_flags()):
            return True
        live = _live(it)
        return bool(it.reconfirm_after and live and self.clock - max(a["clock"] for a in live) > it.reconfirm_after)

    def usable(self, iid, *, valid_at=None):
        """Is the item a fact System 1 may answer on alone: active and not stale? valid_at: a world time — and does the
        item hold then: inside its effective intervals (`periods(id)`; an item without `valid` always is), and, for a
        derived item, where its premises hold. A disputed item is usable at a time outside the contested overlap; so
        is an item derived from it, where for one of its derivations every premise is usable then."""
        if valid_at is None:
            return self.status(iid) == "active" and not self.stale(iid)
        st = self.status(iid)
        rec = self.state.get(iid) or {}
        if st == "disputed":
            st = "active" if "periods" in rec else st
        scoped = st == "hypothesis" and rec.get("premise_scoped")
        if scoped:
            st = "active"
        if not (st == "active" and not self.stale(iid)):
            return False
        if scoped and not any(a["just"] and all(self.usable(p, valid_at=valid_at) for p in a["just"])
                              for a in _live(self.items[iid])):
            return False
        return self._holds_at(iid, valid_at)

    def periods(self, iid):
        """The world times the item holds as its key's value → [[since, until], ...] (half-open, None: unbounded), or
        None for an item that holds at any time (no `valid` on it or under it). On a key with `valid` a version holds
        from its since until a later version's since (or its own until; a bounded later version gives it back after
        the bounded one ends), outside any contested overlap; versions pending a re-test take no part. A derived item
        holds where its own validity and, for one of its derivations, every premise holds."""
        self._settle()
        return self._periods(iid, set(), {})

    def _periods(self, iid, seen, memo=None):
        it, st = self.items.get(iid), self.state.get(iid) or {}
        if it is None or iid in seen:
            return None
        if memo is not None and iid in memo:
            return memo[iid]
        seen = seen | {iid}
        if "periods" in st:
            own = st["periods"]
        else:
            v = self.valid(iid)
            own = None if v is None else [list(_interval(v))]
        live = _live(it)
        pre = None
        try:
            if live and all(a["just"] for a in live):      # every support is a derivation: it holds where they hold
                pre = []
                for a in live:
                    ps = [self._periods(p, seen, memo) for p in a["just"]]
                    ps = [x for x in ps if x is not None]
                    if not ps:
                        pre = None
                        break
                    acc = ps[0]
                    for x in ps[1:]:
                        acc = _ivs_and(acc, x)
                    pre = _ivs_or(pre + acc)
            out = pre if own is None else own if pre is None else _ivs_and(own, pre)
        except TypeError:                                  # premises whose times do not compare (an int and an ISO
            out = own                                      # date): this item reads its own validity, nothing else
        if memo is not None:
            memo[iid] = out
        return out

    def _holds_at(self, iid, at):
        try:
            p = self._periods(iid, set(), {})
        except TypeError:                                # times that do not compare: the item's own `valid`
            return covers(self.valid(iid), at)
        return p is None or any(covers(iv, at) for iv in p)

    def valid(self, iid):
        """The item's validity in the world: the `valid` of its latest live assertion, or None (no valid given)."""
        it = self.items.get(iid)
        if it is None:
            return None
        live = _live(it) or it.asserts
        return live[-1].get("valid") if live else None

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
                **({"consulted": cs} if (cs := sorted({p for a in it.asserts for p in a.get("consulted", ())})) else {}),
                "confidence": {"n_confirm": st.get("n_confirm", 0), "n_refute": len(st.get("refuted_by") or ()),
                               "n_used": it.uses[0], "n_used_wrong": it.uses[1], "level": last.get("level")},
                "status": self.status(iid), "stale": self.stale(iid), "version": version,
                "supersedes": last.get("supersedes"),
                "valid": {"world": last.get("valid") or [None, None],
                          "known": [st.get("learned_at", first.get("t")), st.get("invalidated_at")],
                          "reconfirm_after": it.reconfirm_after, **self._claims(it)},
                "of": last.get("of"), **({"pending": st.get("pending_why")} if st.get("pending_retest") else {})}

    def _claims(self, it):
        """{"claims": every live assertion's validity, "effective": where the item holds} for an item told with
        `valid`; {} otherwise (the record is as before)."""
        claims = [a for a in _live(it) if a.get("valid") is not None]
        if not claims:
            return {}
        try:
            eff = self._periods(it.id, set())
        except TypeError:
            eff = None
        st = self.state.get(it.id) or {}
        pend, unch = set(st.get("pending_claims") or ()), set(st.get("unchecked_claims") or ())
        return {"claims": [{"valid": a["valid"], "by": a["by"], "at": a["t"],
                            **({"pending": True} if a["t"] in pend else {}),
                            **({"unchecked": True} if a["t"] in unch else {})} for a in claims], "effective": eff}

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

    def snapshot(self, *, about=None, relation=None, scope=None, kinds=None, valid_at=None):
        """What a decision is given, as one plain fact: the items that match (`about`: a subject or a list of them;
        `relation`; `scope`; `kinds`, default every kind) — "facts" (active and not stale) and "hints" (hypotheses,
        expired or stale items, with why) — the ids they rest on, the store's fingerprint and journal position, and the
        query (so redecide can rebuild it). Disputed, refuted and retracted items are not given. valid_at: a world time —
        an item whose `valid` does not cover it is a hint ("not valid at ..."), not a fact. A row of an item with a `valid`
        carries it ("valid": [since, until])."""
        self._settle()
        valid_at = _time(valid_at)
        q = {"about": about, "relation": relation, "scope": scope, "kinds": None if kinds is None else list(kinds)}
        if valid_at is not None:
            q["valid_at"] = valid_at
        subjects = None if about is None else ({about} if isinstance(about, str) else set(about))
        facts, hints, memo = [], [], {}
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
            v = self.valid(iid)
            if v is not None:                            # the validity a reader filters on (Knowledge.value(valid_at=))
                row["valid"] = v
            p = self._periods(iid, set(), memo) if v is not None or "periods" in self.state[iid] or \
                any(a["just"] for a in it.asserts) else None
            if p is not None and p != [list(_interval(v))]:
                row["periods"] = p                       # where it holds, when that is not its own `valid`
            if st == "hypothesis" and self.state[iid].get("premise_scoped") and p is not None:
                if valid_at is None or self.usable(iid, valid_at=valid_at):    # derived from such a premise: a fact
                    if not self.stale(iid):                                    # where its premises hold
                        facts.append(row if "periods" in row else {**row, "periods": p})
                    continue
            if st == "disputed":                         # a dispute scoped to an overlap: a fact outside it
                if p is None or "periods" not in self.state[iid] or not p:
                    continue
                if valid_at is not None and not any(covers(iv, valid_at) for iv in p):
                    continue
                if not self.stale(iid):
                    facts.append(row)
                continue
            if st == "active" and valid_at is not None and not (any(covers(iv, valid_at) for iv in p) if p is not None
                                                                else covers(v, valid_at)):
                hints.append({**row, "why": f"not valid at {valid_at}"})
            elif st == "active" and not self.stale(iid):
                facts.append(row)
            elif st in ("active", "hypothesis", "expired"):
                why = "stale" if st == "active" else st
                hints.append({**row, "why": "pending retest" if self.state[iid].get("pending_retest") else why})
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
    def redecide(self, storage, retracted, decide, *, fact="knowledge", upto="decision"):
        """The stored decisions that rested on retracted items, re-run on their re-built snapshots.

        storage: the decisions' TraceStorage; retracted: the retracted item ids; decide: init_state → {question: answer}
        (a function, or a System: its ask, not stored); fact: the given fact the snapshot was passed as. A decision is
        listed when its recorded snapshot changes once the retracted items' entries are left out of the journal up to
        its position; each is re-run on the old and the new snapshot. upto: "decision" (default) — the journal up to the
        decision's position, so the new snapshot is what the decision would have read without the retracted items;
        "now" — the whole journal without them, so a correction told after the decision (a wrong dose retracted and
        the right one told) is in the new snapshot and the new answer is the corrected one, not "unknown".
        → (answer_changes, justification_only): lists of {"id", "old", "new"} — the first goes to a reviewer, the second
        is recorded."""
        if upto not in ("decision", "now"):
            raise ValueError(f'upto is "decision" or "now", not {upto!r}')
        retracted = set(retracted)
        storage = _storage(storage)
        run = _decider(decide)
        rebuilt, changes, same = {}, [], []
        for s in storage.iter():
            init = s.input or {}
            snap = init.get(fact)
            if not isinstance(snap, dict) or "at" not in snap or not retracted & set(snap.get("rests_on") or ()):
                continue
            at = None if upto == "now" else int(snap["at"])
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


def _storage(where):
    """The decisions' storage: a TraceStorage (anything with iter()) as it is, a path opened (open_storage)."""
    if where is None or hasattr(where, "iter"):
        return where
    from ..store import open_storage
    return open_storage(where)


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
def _taint(items, iid, memo):
    """The journal position the item's support was last cut at — its own retraction, or, when every assertion since
    then rests on a premise cut after it was written, that premise's cut — or None. An item retracted and then only
    re-inferred (`inferred` assertions: a rule recomputing it from premises) is not cut by its own retraction. Pure in the journal (`memo` keeps it
    between resolutions); an item with one uncut justification left is not cut (so a retraction stays exact: the store
    after it is the store rebuilt without the retracted item)."""
    if iid in memo:
        return memo[iid]
    memo[iid] = None                                    # a premise cycle reads as uncut
    it = items.get(iid)
    if it is None:
        return None
    cut = _cut(it)
    cut = None if cut < 0 else cut
    live = _live(it)
    via, whole, seen = [], bool(live), []
    for a in live:                                      # an inferred assertion follows its premises: a premise's cut
        ta = max((tp for p in a["just"] if (tp := _taint(items, p, memo)) is not None and tp > a["t"]   # that its own
                  and not (a.get("inferred") and any(q > tp for q in items[p].promoted))), default=None)  # re-test cleared
        #                                                                                          does not hold it back
        if ta is None:
            whole = False
        else:
            via.append(ta)
        seen += [tp for p in a.get("consulted", ()) if (tp := _taint(items, p, memo)) is not None and tp > a["t"]]
    if cut is not None and live and all(a.get("inferred") for a in live):
        cut = None                                      # recomputed from premises: their own cuts decide (above)
    # a consulted item (an alternative's evidence) cut after the assertion: any one marks it for review
    memo[iid] = max(([cut] if cut is not None else []) + (via if whole else []) + seen, default=None)
    return memo[iid]


def _guard(items, ids):
    """The revival guard on one key: a later assertion with `valid` of content (the key's value) that was withdrawn
    from a writer on an overlapping validity — by any retraction (plain, only_from, an assertion, retract_source, a
    failed re-test, a rollback), on any item — is a copy. A copy is pending a re-test unless writers the content was
    not withdrawn from still hold it, in their uncopied assertions, wherever the copy could hold (outside the spans
    where a later version holds anyway); once they no longer do, it is pending then. A passed re-test of an item vouches
    for its assertions written before it and clears the withdrawals of its content before it. Derived and inferred
    assertions are not copies (they follow their premises). → ({(id, assertion position): why} pending,
    {(id, assertion position)} copies admitted while held)."""
    wd = []
    vj = {i: json.dumps(items[i].value, sort_keys=True) for i in ids}
    for i in ids:
        it = items[i]
        for a in it.asserts:
            if a.get("valid") is None or a["just"]:
                continue
            tc = min([x for x in it.retracted if x > a["t"]], default=None)
            x = _acut(it, a)
            own = min([tx for tx, n in it.acuts if n == a["t"] and tx > a["t"]], default=None)
            if x is not None and x == own and _kept(items, ids, vj, i, a, x):
                x = min([tw for tw, w in it.cuts if tw > a["t"] and w == a["by"]], default=None)  # a duplicate taken
            if x is not None and (tc is None or x < tc):                       # back: its writer still stands behind
                tc = x                                                         # the content (not a withdrawal)
            if tc is not None:
                wd.append((tc, i, a))
    if not wd:
        return {}, {}
    clears = defaultdict(list)
    for i in ids:
        clears[vj[i]] += items[i].rechecked
    clean, copies = [], []
    for i in ids:
        for a in _live(items[i]):
            if a.get("valid") is None or a["just"]:
                continue
            iv = _interval(a["valid"])
            w = [(tc, b) for tc, j, b in wd if vj[j] == vj[i] and tc < a["t"] and _overlap(_interval(b["valid"]), iv)
                 and not any(tc < tp <= a["t"] for tp in clears[vj[i]])]
            if not w or any(tp > a["t"] for tp in items[i].rechecked):
                clean.append((i, a))
            else:
                copies.append((i, a, w))
    pending, held = {}, {}
    for i, a, w in copies:
        ws = sorted({str(b["by"]) for _, b in w})
        base = [(vj[j], _interval(b["valid"]), (j, json.dumps(_interval(b["valid"])[0], sort_keys=True)))
                for j, b in clean if not (vj[j] == vj[i] and str(b["by"]) in ws)]
        why = (f"re-published content withdrawn from {', '.join(ws)} (journal entry {max(tc for tc, _ in w)}) and "
               f"held by no other writer")
        if _held(_interval(a["valid"]), vj[i], base):
            held[(i, a["t"])] = (_interval(a["valid"]), vj[i], base, why)
        else:
            pending[(i, a["t"])] = why
    return pending, held


def _kept(items, ids, vj, i, a, tc):
    """Does a's writer still stand behind a's content on a's validity after a's own cut at `tc`: a live assertion of
    the same content from the same writer, written before the cut, whose interval covers a's (a duplicate taken back)?"""
    iv = _interval(a["valid"])
    return any(b["by"] == a["by"] and b["t"] < tc and b.get("valid") is not None and not b["just"]
               and _inside(iv, [_interval(b["valid"])])
               for j in ids if vj[j] == vj[i] for b in _live(items[j]))


def _held(iv, value, base, lost=None):
    """Do the uncopied assertions `base` ([(value, interval, version)]) hold `value` wherever an assertion of it on `iv`
    could hold (not where a version with a later since holds)? lost: after the walk, {version in play: the versions
    that took its overlap} — then `base` holds only the versions in play, a version that lost the segment to another
    at the top since does not count, and the segment is held only when `value` is the one value left there (a
    contested segment, or one the content lost, is not held)."""
    try:
        for seg in _segments([iv] + [b[1] for b in base]):
            if not _covers_seg(iv, seg):
                continue
            cov = [b for b in base if _covers_seg(b[1], seg)]
            if not cov:
                return False
            top = cov[0][1][0]
            for b in cov[1:]:
                if _lo_lt(top, b[1][0]):
                    top = b[1][0]
            if _lo_lt(iv[0], top):
                continue                                # a later version holds here: the copy could not
            best = [b for b in cov if not _lo_lt(b[1][0], top)]
            if lost is None:
                if value not in {b[0] for b in best}:
                    return False
                continue
            names = {b[2] for b in best}
            if {b[0] for b in best if not (lost[b[2]] & names)} != {value}:
                return False
        return True
    except TypeError:
        return False


def _retest_why(items, iid, t, memo):
    """Why an item waits for a re-test (its support cut at journal entry t) → a sentence for item(id)["pending"]."""
    it = items[iid]
    if _cut(it) == t:
        return f"written again after its retraction at journal entry {t}"
    for a in _live(it):
        for p in a["just"]:
            if _taint(items, p, memo) == t:
                return f"rests on {p}, whose support was cut at journal entry {t}"
        for p in a.get("consulted", ()):
            if _taint(items, p, memo) == t:
                return f"consulted {p}, whose support was cut at journal entry {t}"
    return f"its support was cut at journal entry {t}"


def _resolve(items, comp, by_key, resolves, clock, flagged, max_iter=200, taints=None, state=None):
    """The state of the items in `comp` (closed under dependents and shared keys) from their raw entries → {id:
    record} for present items. Fixed point of: liveness of justifications ↔ per-key walk. Deterministic; premises
    outside `comp` are read from `state` (their resolved records), which nothing in `comp` can change."""
    order = sorted(comp, key=lambda i: (min((a["t"] for a in items[i].asserts), default=0), i))
    fixed = {p: state[p] for i in order for a in items[i].asserts for p in a["just"]
             if p not in comp and p in (state or {})}
    taints = {} if taints is None else taints
    retest, why = set(), {}
    for i in order:
        t = _taint(items, i, taints)
        if t is not None and not any(p > t for p in items[i].promoted):
            retest.add(i)
            why[i] = _retest_why(items, i, t, taints)
    pclaims, hinfo = {}, {}                             # the revival guard: copies of withdrawn content
    for k in sorted({items[i].key for i in order if items[i].key is not None}):
        p, h = _guard(items, sorted(by_key[k]))
        pclaims.update(p)
        hinfo.update(h)
    retest0 = set(retest)
    keys = sorted({items[i].key for i in order if items[i].key is not None})
    while True:                                         # a copy admitted as held is re-checked after the walk
        status, live, keyed, rank, units, retest, partial, on_disputed = _resolve_walk(
            items, order, keys, by_key, resolves, clock, flagged, max_iter, fixed, comp, retest0, why, pclaims)
        inplay = {u[0]: u[4] for k in keys for u in units.get(k, ())}
        bad = {c: h[3] for c, h in hinfo.items()
               if not _held(h[0], h[1], [b for b in h[2] if b[2] in inplay], inplay)}
        if not bad:                                     # its holders hold in the walk: refuted, disputed or closed
            break                                       # versions do not (the copy goes pending)
        for c, w in bad.items():
            pclaims[c] = w
            del hinfo[c]
    held = set(hinfo)
    eff = {}
    for k in keys:                                      # effective intervals on the keys with `valid`
        if units.get(k):
            eff.update(_effective(units[k]) or {})
    scoped = set()                                      # derived items held up only by a dispute scoped to an
    grew = True                                         # overlap: each holds where its premises hold (usable /
    while grew:                                         # snapshot at a time read the premises' periods there)
        grew = False
        for i in order:
            if i in on_disputed and i not in scoped and any(
                    all(status.get(p) == "active" or p in scoped or (fixed[p].get("premise_scoped") if p in fixed else False)
                        or (status.get(p) == "disputed" and (p in eff or "periods" in (fixed.get(p) or {})))
                        for p in a["just"]) for a in live[i]):
                scoped.add(i)
                grew = True
    rec = {}
    for i, s in status.items():
        if i in fixed:
            continue
        la = live[i]
        kr = keyed.get(i) or ("", None, [], None)
        rec[i] = {"status": s, "rank": rank[i], "n_confirm": len(la) - 1, "refuted_by": kr[2],
                  "learned_at": min(a["t"] for a in la), "invalidated_at": kr[1] if s == "refuted" else None,
                  "last_support": max(a["t"] for a in la), "dispute_at": kr[3] if s == "disputed" else None,
                  **({"pending_retest": True, "pending_why": why[i]} if i in retest or i in partial else {})}
        if i in partial:                                # copies waiting for a re-test; the item's other claims hold
            rec[i]["pending_claims"] = partial[i]
        if any(j == i for j, _ in held):
            rec[i]["unchecked_claims"] = sorted(t for j, t in held if j == i)
        if i in scoped:                                 # not in the fingerprint either
            rec[i]["premise_scoped"] = True
        if i in eff:                                    # not in the fingerprint: a function of the assertions
            rec[i]["periods"] = eff[i]["periods"]
            if eff[i]["contested"]:
                rec[i]["contested"] = eff[i]["contested"]
    return rec


def _resolve_walk(items, order, keys, by_key, resolves, clock, flagged, max_iter, fixed, comp, retest0, why, pclaims):
    """The fixed point of _resolve for a given set of pending copies → (status, live, keyed, rank, units, retest,
    partial)."""
    retest = set(retest0)
    partial = defaultdict(list)
    for (i, t), w in sorted(pclaims.items()):
        if i in items and i not in retest:
            partial[i].append(t)
            why.setdefault(i, w)
    for i in list(partial):
        if not any((i, a["t"]) not in pclaims for a in _live(items[i])):
            retest.add(i)                               # nothing but pending copies: the item waits as a whole

    live_asserts = _live

    status = {**{p: f["status"] for p, f in fixed.items()}, **{i: "active" for i in order if live_asserts(items[i])}}
    live, keyed, rank, units, on_disputed = {}, {}, {}, {}, set()
    for _ in range(max_iter):
        prev_rank = rank                                # a premise later in `order` is read from the last round
        live, rank, arank = {}, {p: f["rank"] for p, f in fixed.items()}, {}
        for i in order:                                 # liveness of each assertion under the current premise statuses
            la = [a for a in live_asserts(items[i]) if all(status.get(p) not in (None, "refuted") for p in a["just"])]
            if not la:
                continue
            live[i] = la
            r = 0
            for a in la:
                ra = RANK[a["source"]]
                for p in a["just"]:
                    ra = min(ra, rank.get(p, prev_rank.get(p, 0)))
                arank[(i, a["t"])] = ra
                r = max(r, ra)
            rank[i] = r
        keyed, units = {}, {}
        for k in keys:                                  # an item pending re-test is a hint: it takes no side on its key
            ids = [i for i in by_key[k] if i in live and i not in retest]
            units[k] = []
            wl = live if not partial else {i: [a for a in live[i] if (i, a["t"]) not in pclaims] or live[i] for i in ids}
            keyed.update(_walk_key(ids, wl, arank, {i: items[i].value for i in ids}, resolves.get(k, ()), units[k]))
        new = {p: f["status"] for p, f in fixed.items()}
        on_disputed = set()                             # a hypothesis only because a premise is disputed
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
            if i in flagged or i in retest:
                s = "hypothesis"
            if it.reconfirm_after and clock - max(a["clock"] for a in live[i]) > it.reconfirm_after:
                s = "expired"
            if s == "active" and not any(all(status.get(p) == "active" for p in a["just"]) for a in live[i]):
                s = "hypothesis"                        # a derived item is a fact only on active premises
                on_disputed.add(i)
            new[i] = s
        if new == status and rank == prev_rank:
            break
        status = new
    else:
        raise RuntimeError(f"resolution did not converge on a component of {len(comp)} items")
    return status, live, keyed, rank, units, retest, partial, on_disputed


def _walk_key(ids, live, arank, values, res, units=None):
    """One key's live assertions in time order → {id: (state, refuted_at, refuted_by, dispute_opened_at)}; state "" =
    holds the key."""
    if not ids:
        return {}
    if any(a.get("valid") for i in ids for a in live[i]):
        return _walk_key_valid(ids, live, arank, values, res, units)
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
        observed = a["source"] == "outcome" and not a.get("inferred")
        if disputed:
            if observed:                                # an observation ends a dispute
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
        if a.get("supersedes") in holders or observed or r > hrank:
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


def _beats(a, r, iv, by, holder):
    """+1: assertion `a` (rank r, interval iv, writer by) refutes the holder; -1: the holder refutes it; 0: a dispute.
    Only versions of one since compete (two missing sinces are one since; different sinces are versions in time,
    see _walk_key_valid). The holder's writers: every writer whose live assertion of that version has been seen."""
    hiv, hr, hby, hj = holder
    if a.get("supersedes") == hj:
        return 1
    same_since = iv[0] == hiv[0] and bool(set(hby) - {by})   # another writer holds it from the same since
    if a["source"] == "outcome" and not a.get("inferred") and not same_since:
        return 1
    return 1 if r > hr else 0


def _inside(iv, ivs):
    """Is the interval iv inside the union of ivs?"""
    try:
        return _ivs_and([list(iv)], _ivs_or(ivs)) == _ivs_or([list(iv)])
    except TypeError:
        return True


def _time_kind(x):
    return "number" if isinstance(x, (int, float)) and not isinstance(x, bool) else type(x).__name__


def _walk_key_valid(ids, live, arank, values, res, units=None):
    """_walk_key on a key where assertions carry `valid` (see the module docstring) → the same {id: (state,
    refuted_at, refuted_by, dispute_opened_at)}. A version is an item's assertions of one since (an item told for two
    periods, A-B-A, has two). Versions of different sinces never compete: the later closes the earlier at its since
    (_effective). Versions of one since that overlap with different values compete as before (rank, supersedes, an
    observation; equal ranks dispute) — the loser is refuted when the winner covers its whole interval, otherwise it
    is closed only on the overlap and holds outside it; a person's answer to a dispute does the same. units: a list
    that receives the versions still in play, for _effective."""
    ev = [(a["t"], 0, i, a) for i in ids for a in live[i]] + [(t, 1, v, by) for t, v, by in res]
    ev.sort(key=lambda e: (e[0], e[1], str(e[2])))
    seen = {}                                           # version → (interval, rank, writers, item id): its latest
    writers = defaultdict(set)                          # assertion's interval and rank, every writer seen asserting it
    lost = defaultdict(set)                             # version → the versions that took its overlap
    holders, disputed, refuted, dopen = set(), set(), {}, None
    kinds = {_time_kind(_interval(e[3].get("valid"))[0]) for e in ev if e[1] == 0}
    mixed = len(kinds - {"NoneType"}) > 1              # times of several types: every version is a candidate
    by_since = defaultdict(set)                         # since → the versions seen with it (rival candidates)

    def rival(u, v):
        return values[u[0]] != values[v[0]] and _overlap(seen[u][0], seen[v][0]) and \
            _same_since(seen[u][0][0], seen[v][0][0])
    for e in ev:
        if e[1] == 1:                                   # a person's answer to a dispute: that value wins
            win = {u for u in disputed if values[u[0]] == e[2]}
            for u in sorted(disputed - win):
                if _inside(seen[u][0], [seen[w][0] for w in win]) or not win:
                    refuted[u] = (e[0], ["resolve"])
                else:                                   # closed on the overlap only
                    lost[u] |= win
                    holders.add(u)
            holders |= win
            disputed = set()
            continue
        t, _, i, a = e
        iv, r = _interval(a.get("valid")), arank[(i, t)]
        u = (i, json.dumps(iv[0], sort_keys=True))
        writers[u].add(a.get("by"))
        seen[u] = (iv, r, frozenset(writers[u]), i)
        holders.discard(u)
        disputed.discard(u)
        refuted.pop(u, None)
        lost.pop(u, None)
        sk = repr(float(iv[0])) if isinstance(iv[0], (int, float)) and not isinstance(iv[0], bool) else u[1]
        if a.get("by") is not None:                     # a writer's later value at one since replaces its own
            for v in by_since[sk]:                      # earlier one: it no longer holds that version
                if v != u and values[v[0]] != values[i] and a["by"] in writers[v]:
                    writers[v].discard(a["by"])
                    seen[v] = (seen[v][0], seen[v][1], frozenset(writers[v]), seen[v][3])
        by_since[sk].add(u)
        cands = (holders | disputed) if mixed else [v for v in by_since[sk] if v in holders or v in disputed]
        rivals = sorted(v for v in cands if rival(u, v))
        verdict = {v: _beats(a, r, iv, a.get("by"), seen[v]) for v in rivals}
        beaten = [v for v in rivals if verdict[v] < 0]
        if beaten:
            if _inside(iv, [seen[v][0] for v in beaten]):
                refuted[u] = (t, sorted({v[0] for v in beaten}))
                continue
            lost[u] |= set(beaten)
        for v in rivals:
            if verdict[v] > 0:
                if _inside(seen[v][0], [iv]):
                    holders.discard(v)
                    disputed.discard(v)
                    refuted[v] = (t, [i])
                else:
                    lost[v].add(u)
        tie = [v for v in rivals if verdict[v] == 0]
        if tie:
            holders.difference_update(tie)
            disputed |= set(tie) | {u}
            dopen = t
        else:
            holders.add(u)
        for v in sorted(disputed):                      # a side whose every rival is gone holds again
            if not any(rival(v, k) for k in disputed if k != v):
                disputed.discard(v)
                holders.add(v)
    if units is not None:
        units.extend((u, seen[u][3], json.dumps(values[u[0]], sort_keys=True), seen[u][0], frozenset(lost[u]))
                     for u in sorted(holders | disputed))
    by_item = defaultdict(list)
    for u in seen:
        by_item[u[0]].append(u)
    out = {}
    for i in ids:
        us = by_item.get(i, ())
        if any(u in disputed for u in us):
            out[i] = ("disputed", None, [], dopen)
        elif any(u in holders for u in us) or not any(u in refuted for u in us):
            out[i] = ("", None, [], None)
        else:
            last = max((u for u in us if u in refuted), key=lambda u: refuted[u][0])
            out[i] = ("refuted", refuted[last][0], refuted[last][1], None)
    return out


__all__ = ["covers", "item_id", "key_of", "KINDS", "KnowledgeStore", "NEEDS_PROMOTION", "RANK", "RECONFIRM_DEFAULTS",
           "scope_matches", "SelfDefeating", "SOURCES", "STATUSES", "Verdict"]
