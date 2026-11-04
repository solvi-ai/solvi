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
earlier one there (it no longer asserts the old value) — blindly: after an erratum withdraws one of its claims, the
writer's next value at that since replaces every value it had there, including ones the erratum did not correct; two writers who both move from one value to the same other
value leave that value holding. The winner takes only the overlap: a loser whose interval
reaches beyond it is closed there and holds outside (refuted only when the winner covers it whole), and a dispute is
contested only on the overlap — outside it each side holds; a person's answer (`resolve`) closes the loser the same
way. An item pending re-test takes no side on its key (no dispute, no refutation, no closing). `usable(id, valid_at=)`
and `snapshot(valid_at=)` read the effective intervals (a snapshot row carries "periods" when they are not the item's
own `valid`; `Knowledge.value(valid_at=)` reads them). An item derived from a premise disputed only on an overlap is a
hypothesis as a whole, but at a time where every premise of one of its derivations holds it is usable and a fact. Keys whose assertions carry no `valid` resolve as described
above.
A circular derivation (a ← p, b ← a, then a ← b) is accepted (with a warning at the write), but a chain back
through the item itself is no support: retracting the loop's outside premise taints every item in it, so such
content cannot keep itself alive past a retraction.

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
import copy
import json
import threading
import warnings
from bisect import bisect_left, insort
from collections import defaultdict
from contextlib import contextmanager

from ..runtime import vhash
from . import _merkle
from .gates import RANK, SOURCES, Verdict, WriterGate, SourceGate, ConsistencyGate, default_gates, with_source_gate

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
                 "reconfirm_after", "uses", "cuts", "acuts", "rechecked", "lc", "lv", "cutv")

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
        self.lc = None              # the raw-entry lengths `lv`/`cutv` were computed at (S266: a read of a key with
        self.lv = []                # cuts used to rescan the item's whole history on every write)
        self.cutv = -1


def _acut(it, a):
    """The journal position assertion `a` was taken back at by its writer's or its own cut, or None."""
    return min([tc for tc, w in it.cuts if tc > a["t"] and w == a["by"]]
               + [tc for tc, n in it.acuts if n == a["t"] and tc > a["t"]], default=None)


def _cut(it):
    """The last journal position this item's whole support was taken back at (a retraction, or a writer's or an
    assertion's retraction that left no other assertion standing), or -1. With `_live`, kept on the item and moved
    forward by the takedown events (S266)."""
    _refresh_live(it)
    return it.cutv


def _live(it):
    """The item's assertions still standing: after its last retraction, and not taken back with their writer's or on
    their own. Kept on the item and moved forward by the takedown events (S266): an assertion stands while no
    retraction, no cut of its writer and no cut of its own was written after it — the filters read only raw
    positions, so the events apply in any order and once each."""
    _refresh_live(it)
    return it.lv


def _refresh_live(it):
    """Move the item's cached `_live` list and `_cut` position up to its raw entries: the takedown events are
    replayed in journal order, each dropping the assertions it reaches, and an event before which nothing stands
    any more is a position the whole support was taken back at (a retraction always is: it reaches every assertion
    written before it)."""
    counts = (len(it.asserts), len(it.retracted), len(it.cuts), len(it.acuts))
    if it.lc == counts:
        return
    na, nr, nc, nx = (0, 0, 0, 0) if it.lc is None else it.lc
    live = list(it.asserts) if it.lc is None else it.lv + list(it.asserts[na:])
    cut = -1 if it.lc is None else it.cutv
    events = ([("retract", t, None) for t in it.retracted[nr:]]
              + [("cut", t, w) for t, w in it.cuts[nc:]]
              + [("acut", t, n) for t, n in it.acuts[nx:]])
    events.sort(key=lambda e: e[1])
    for kind, t, x in events:
        live = [a for a in live if (a["t"] >= t if kind == "retract" else
                                    a["t"] >= t or a["by"] != x if kind == "cut" else a["t"] != x)]
        if not any(a["t"] < t for a in live):
            cut = t
    it.lv, it.cutv, it.lc = live, cut, counts


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


def _ivs_minus(xs, ys):
    """The half-open intervals of `xs` minus those of `ys` (a later version closes an earlier one on its own span;
    a bounded one gives the earlier value back after it)."""
    out = []
    for seg in _segments(xs + ys):
        if any(_covers_seg(x, seg) for x in xs) and not any(_covers_seg(y, seg) for y in ys):
            out.append(list(seg))
    return out


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
    check). None (default): any writer may declare any source, as before. now: a world clock, opt-in — a callable
    returning the current world time (no arguments), or a fixed time; with it `usable(id)` and `usable` on derived
    items read the effective intervals at that time instead of status alone, so a version closed by a later one
    (or an item derived from it) is not usable with no `valid_at` given. None (default): as before — `usable(id)`
    with no `valid_at` reads status and staleness only. revival_window: optional inclusive protection
    after withdrawal, in logical tick units. New direct dated claims after the window may vouch
    anew; old held claims never become clean merely because time passes. None preserves the
    original overlap-based guard. Reopen persistent stores with the same explicit policy."""

    def __init__(self, storage=None, *, gates=None, defaults=None, writers=None, reassert_writers=(), now=None, revival_window=None, _replay=True):
        from ..store import open_storage
        self.storage = open_storage(storage)
        self.gates = with_source_gate(gates if gates is not None else default_gates())
        if writers is not None:
            self.gates.insert(1, WriterGate(writers))       # right after the source check
        self.reassert_writers = frozenset(str(w) for w in reassert_writers)
        if self.reassert_writers and (writers is None or not self.reassert_writers <= set(map(str, writers))):
            raise ValueError("reassert_writers must be registered writers")
        self.defaults = dict(RECONFIRM_DEFAULTS if defaults is None else defaults)
        if revival_window is not None:
            import math
            if type(revival_window) not in (int, float):
                raise ValueError("revival_window must be a finite nonnegative number or None")
            try:
                revival_window = float(revival_window)
            except OverflowError:
                raise ValueError("revival_window is not representable") from None
            if not math.isfinite(revival_window) or revival_window < 0:
                raise ValueError("revival_window must be finite and nonnegative")
        self._revival_window = revival_window
        self._entry_clocks = {}
        self.now = now                                   # a world clock for usable() with no valid_at (see usable)
        self._lock = threading.RLock()
        self._mem = _MemoryJournal() if self.storage is None else None
        self.journal = []                            # the entries in order, each with its position "n"
        self._next_n = 0                             # logical positions may have historical holes/legacy notes
        self.items: dict[str, _Item] = {}
        self.by_key = defaultdict(set)
        self._by_subject = defaultdict(set)
        self._by_relation = defaultdict(set)
        self.dependents = defaultdict(set)
        self.resolves = defaultdict(list)            # key → [(t, value, by)]
        self.flags = {}                              # t → {"ids", "scope", "key", "why", "by", "parent", "ended"}
        self.clock = 0
        self.state = {}                              # id → resolved record (present items)
        self._period_epoch, self._period_cache = 0, {}  # immutable top-level read results only
        self.questions = []                          # person questions (disputes), one per dispute event
        self._open_disputes = set()
        self._batch, self._touched, self._tick_dirty = 0, set(), False
        self._expiry = []                            # sorted (clock an item's expiry can flip at, id): the tick index
        self._resolved_clock = 0                     # the clock the state was last resolved at
        self._fp = None
        self._fingerprint_root = None
        self._fingerprint_rows = {}
        self._taints = {}                            # id → the journal position its support was last cut at (memo)
        self._append_chains = {}                     # proven plain open-ended succession keys only
        self._append_epoch = 0                      # invalidate witnesses without walking unrelated keys
        self._append_pair = None
        self._guards = {}                            # key → _GuardKey: the revival guard's per-key cache (S266)
        if self.storage is not None and _replay:
            with self.batch():
                for s in self.storage.iter(_KNOWLEDGE):
                    r = dict(s.data)
                    self.journal.append(r)
                    self._apply(r)

    # ------------------------------------------------------------------------------------------------ the journal
    def _write(self, op, **data):
        body = {"kind": _KNOWLEDGE, "n": self._next_n,
                "op": op, "clock": self.clock, **data}
        if self._mem is not None:
            from ..store import _stamp
            rec = self._mem.append(_stamp(body))     # run_context: the in-memory journal carries the run too
        else:
            from ..store import FORMAT
            rec = self.storage._append({"v": FORMAT, **body})
        self.journal.append(rec)
        self._next_n += 1
        self._fp = None
        return rec

    def __len__(self):
        """The number of journal entries."""
        return len(self.journal)

    def head(self):
        """{"count", "hash"}: the number of journal entries and the last entry's hash."""
        return {"count": len(self.journal), "hash": self.journal[-1]["hash"] if self.journal else ""}

    @property
    def revival_window(self):
        """Optional inclusive withdrawal window in owner-declared logical clock units."""
        return self._revival_window

    def verify(self):
        """Is the journal's hash chain whole (in memory: its own chain; on a backend: the backend's verify()), and is the
        state the one the journal gives (the store rebuilt from it has the same fingerprint)?"""
        with self._lock:
            if not self._journal_whole():
                return False
            # Verification replays recorded verdicts; it never invokes write
            # gates or exposes a writable copy. Locks/resources owned by custom
            # gates therefore need not be cloneable for this read-only query.
            ks = KnowledgeStore(None, gates=self.gates, defaults=self.defaults, now=self.now, revival_window=self.revival_window, _replay=False)
            ks.reassert_writers = self.reassert_writers
            self._replay_entries(ks, copy.deepcopy(self.journal))
            return ks.fingerprint() == self.fingerprint()

    def _journal_whole(self):
        """Validate source records without calling rebuild/verify recursively."""
        if self._mem is not None:
            return self._mem.verify() and self.journal == self._mem.entries
        if not self.storage.verify()["ok"]:
            return False
        stored = [dict(s.data) for s in self.storage.iter(_KNOWLEDGE)]
        return len(stored) >= len(self.journal) and self.journal == stored[:len(self.journal)]

    def rebuild(self, upto=None, skip=()):
        """The store the journal gives, in memory: entries applied in order (upto: the first `upto` — the store as it
        was then); `skip`: item ids whose own entries (assertions, promotions, retractions) are left out — the store as if
        they had never been written. Premises naming a skipped item stay in the entries derived from it (dead).
        The result is independently writable, with its own hash chain, copied journal data, write gates, defaults,
        world clock and reassert capabilities. Logical entry positions are retained for assertion proofs. Uncopyable
        policy objects fail before replay. Functions used as callbacks retain their normal Python closure semantics;
        external state accessed by a callback is not snapshotted. Stored verified decisions are not copied into this
        in-memory reconstruction: new verified writes still require a decision in the destination TraceStorage."""
        skip = set(skip)
        with self._lock:
            if not self._journal_whole():
                raise ValueError("cannot reconstruct a corrupt knowledge journal")
            # Snapshot validated data before policy deepcopy callbacks can run.
            entries = copy.deepcopy(self.journal if upto is None else self.journal[:upto])
            try:
                gates, defaults, now = copy.deepcopy((self.gates, self.defaults, self.now))
                if any(a is b for a, b in zip(self.gates, gates)):
                    raise TypeError("a gate returned itself from deepcopy")
            except Exception as exc:  # noqa: BLE001 - arbitrary policy deepcopy must fail closed
                raise TypeError("rebuild requires independently copyable write gates and policy") from exc
            ks = KnowledgeStore(None, gates=gates, defaults=defaults, now=now, revival_window=self.revival_window, _replay=False)
            # WriterGate is already present in the copied gate sequence. Preserve
            # the separate, explicit capability; do not add a second registry.
            ks.reassert_writers = self.reassert_writers
        return self._replay_entries(ks, entries, skip, _verified_memory=self._mem is not None and not skip)

    @staticmethod
    def _replay_entries(ks, entries, skip=(), *, _verified_memory=False):
        """Replay recorded history only, with no write-gate calls or new verdicts."""
        if _verified_memory and not skip:
            # Only rebuild's already validated, independently copied memory
            # prefix qualifies. Persistent records use a different chain schema;
            # removing records also requires the general rechaining path below.
            ks._mem.entries = entries
            ks._mem.prev = entries[-1]["hash"] if entries else ""
            ks.journal = list(entries)
            with ks.batch():
                for r in entries:
                    ks._apply(r)
            return ks
        with ks.batch():
            for r in entries:
                if r.get("item") in skip:
                    continue
                # Re-chain the retained records, keeping their logical positions:
                # assertion/reassert proofs refer to n, including after skip.
                rec = ks._mem.append({k: v for k, v in r.items() if k not in ("hash", "prev")})
                ks.journal.append(rec)
                ks._apply(rec)
        return ks

    def branch(self, storage=None, *, upto=None):
        """Independent writable what-if over a journal prefix; optional empty destination.

        No merge or decision-proof export. Persistent branches refuse inherited verified/of
        assertions. Reopening uses the normal constructor with the same explicit policy.
        A lineage note binds the original parent head and selected prefix hash.
        """
        from ..store import open_storage
        from pathlib import Path
        source_path = getattr(self.storage, "path", None)
        target_path = storage if isinstance(storage, (str, Path)) else getattr(storage, "path", None)
        if source_path is not None and target_path is not None:
            source_path, target_path = Path(source_path), Path(target_path)
            if (source_path.resolve() == target_path.resolve()
                    or source_path.exists() and target_path.exists() and source_path.samefile(target_path)):
                raise ValueError("branch destination must be empty and independent of its parent")
        if upto is not None and (isinstance(upto, bool) or not isinstance(upto, int) or not 0 <= upto <= len(self.journal)):
            raise ValueError("upto must be a journal prefix length")
        with self._lock:
            child = self.rebuild(upto=upto)
            lineage = {"parent": copy.deepcopy(self.head()),
                       "prefix_count": len(child.journal),
                       "prefix_hash": (self.journal[len(child.journal) - 1]["hash"] if child.journal else "")}
            if storage is not None and any(r.get("source") == "verified" or r.get("of") is not None
                                           for r in child.journal):
                raise ValueError("persistent branch cannot export verified decision proofs")
        destination = open_storage(storage)
        if destination is not None:
            if destination is self.storage or destination.head()["count"]:
                raise ValueError("branch destination must be empty and independent of its parent")
            # Backend assigns its own physical chain; logical n and assertion
            # proof positions are retained. The lineage note binds the source.
            with destination.batch():
                records = [destination._append({k: v for k, v in r.items()
                                               if k not in ("hash", "prev", "seq", "id", "time")})
                           for r in child.journal]
            child.storage, child._mem, child.journal = destination, None, records
        child.note(branch=lineage)
        return child

    def _apply(self, r):
        """Apply one journal entry to the raw data (rebuild and replay: no new entry is written)."""
        op, t = r["op"], r["n"]
        self._next_n = max(self._next_n, t + 1)
        if op != "assert":
            self._append_epoch += 1
            self._append_pair = None
        if op == "assert" and r.get("reassert_proof"):
            self._validate_reassert_record(r, t)
        self.clock = r.get("clock", self.clock)
        if self.revival_window is not None:
            self._entry_clocks[t] = self.clock
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

    def _validate_reassert_record(self, r, t):
        iid = r["item"]
        it = self.items.get(iid)
        if (it is None or r.get("body") != it.body or r.get("scope") != it.scope
                or r.get("item_kind") != it.kind or item_id(r["item_kind"], r["body"], r["scope"]) != iid):
            raise ValueError("invalid reassert: no exact earlier item")
        if r.get("reassert_proof"):
            proof = r["reassert_proof"]
            prop = {"id": iid, "kind": r["item_kind"], "by": r.get("by"), "source": r["source"],
                    "valid": r.get("valid"), "derived_from": r.get("just", ()),
                    "consulted": r.get("consulted", ()), "inferred": r.get("inferred", False)}
            if proof != self._reassert_proof(prop, check_capability=False):
                raise ValueError("invalid reassert proof: incomplete withdrawal history")
            expected = {"writer": r.get("by"), "source": r["source"], "valid": r.get("valid"),
                        "cuts": proof.get("cuts")}
            if proof != expected or r.get("just") or not proof["cuts"]:
                raise ValueError("invalid reassert proof")
            for c in proof["cuts"]:
                old = next((e for e in self.journal if e["n"] == c["withdrawal"]), None)
                if (old is None or old["n"] >= t or old["op"] != "retract_assertions"
                        or old.get("by") != proof["writer"] or old.get("assertions") != c["assertions"]
                        or old.get("item") != iid or c.get("actor") != proof["writer"]):
                    raise ValueError("invalid reassert withdrawal proof")
                for n in c["assertions"]:
                    prior = next((a for a in it.asserts if a["t"] == n), None)
                    if (prior is None or prior["by"] != proof["writer"] or prior["source"] != proof["source"]
                            or prior.get("valid") != proof["valid"] or prior["just"]):
                        raise ValueError("invalid exact reassert assertion proof")

    def _raw_assert(self, r, t):
        if r.get("reassert_proof"):
            self._validate_reassert_record(r, t)
        iid = r["item"]
        it = self.items.get(iid)
        fresh = it is None
        if it is None:
            key = key_of(r["body"], r["scope"])
            value = r["body"].get("o") if isinstance(r["body"], dict) else None
            it = _Item(iid, r["item_kind"], r["body"], r["scope"], key, value, r.get("reconfirm_after"))
            self.items[iid] = it
            # Bodies are immutable; retain every status so as-of facts and hints
            # use the same candidate set after writes and journal replay.
            body = it.body if isinstance(it.body, dict) else {}
            for field, index in (("s", self._by_subject), ("r", self._by_relation)):
                value = body.get(field)
                try:
                    hash(value)
                except TypeError:
                    continue  # unhashable filters retain the full-scan fallback
                index[value].add(iid)
            if key is not None:
                self.by_key[key].add(iid)
        # A fresh direct assertion after the optional window may vouch anew;
        # never waive derived/consulted evidence, retirement or pending flags.
        cut = _cut(it) if self.revival_window is not None else -1
        revival_cut = (cut if cut >= 0 and r.get("valid") is not None
                       and not r.get("just") and not r.get("consulted") and not r.get("inferred")
                       and self.clock - self._entry_clocks[cut] > self.revival_window else None)
        it.asserts.append({"t": t, "clock": r.get("clock", 0), "source": r["source"], "by": r.get("by"),
                           "just": tuple(r.get("just") or ()), "supersedes": r.get("supersedes"),
                           "evidence": r.get("evidence"), "valid": r.get("valid"), "level": r.get("level"),
                           "of": r.get("of"), "inferred": bool(r.get("inferred")),
                           "consulted": tuple(r.get("consulted") or ()),
                           "reassert_proof": r.get("reassert_proof")})
        if revival_cut is not None:
            it.asserts[-1]["_revival_cut"] = revival_cut
        for p in it.asserts[-1]["just"] + it.asserts[-1]["consulted"]:     # a consulted item's change re-resolves it
            self.dependents[p].add(iid)
        if it.reconfirm_after:                          # this assertion's clock is a possible last support: past
            insort(self._expiry, (it.asserts[-1]["clock"] + it.reconfirm_after, iid))   # it, the item expires
        # Only a witnessed, dependency-free succession may resolve its last two
        # versions. Any other write keeps the general component resolver.
        self._append_pair = None
        for parent in it.asserts[-1]["just"] + it.asserts[-1]["consulted"]:
            premise = self.items.get(parent)  # rebuild(skip=) may deliberately omit this assertion's support
            if premise is not None:
                self._append_chains.pop(premise.key, None)
        chain = self._append_chains.pop(it.key, None)
        if chain is not None and chain[-1] != self._append_epoch:
            chain = None
        since, until = _interval(r.get("valid"))
        plain = (fresh and it.kind == "fact" and it.key is not None and not self._touched and not self.flags
                 and not self.dependents.get(iid)
                 and self.now is None and all(type(g) in (SourceGate, ConsistencyGate) for g in self.gates)
                 and not self.resolves.get(it.key) and since is not None and until is None
                 and not it.reconfirm_after and r.get("by") is not None
                 and not any(r.get(k) for k in ("just", "consulted", "supersedes", "inferred", "reassert_proof")))
        if plain and chain is not None:
            previous, old_since, writer, source, _ = chain
            try:
                plain = (self.state.get(previous, {}).get("status") == "active"
                         and writer == r.get("by") and source == r["source"]
                         and _time_kind(since) == _time_kind(old_since) and since > old_since)
            except TypeError:
                plain = False
            if plain:
                self._append_pair = (previous, iid)
        if plain and (chain is not None or len(self.by_key[it.key]) == 1):
            self._append_chains[it.key] = (iid, since, r.get("by"), r["source"], self._append_epoch)
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
            reconfirm_after=None, valid=None, level=None, of=None, shadow=None, inferred=False, consulted=(), reassert=False):
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
        if reassert and (derived_from or inferred or consulted):
            raise ValueError("reassert requires a direct assertion")
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
        cyc = _premise_cycle(self.items, prop["id"], prop["derived_from"])
        if cyc is not None:
            warnings.warn(f"derived_from runs in a cycle through item {cyc}: support through a premise cycle is no "
                          "support — retracting an outside premise of the loop taints every item in it",
                          stacklevel=2)
        with self._lock:
            proof = self._reassert_proof(prop) if reassert else None
            verdicts = []
            for g in self.gates:
                v = g.admit(self, prop, shadow)
                v.gate = v.gate or getattr(g, "name", type(g).__name__)
                verdicts.append(v)
                if not v.admit and (v.gate in ("source", "writers", "consistency", "trust") or kind not in NEEDS_PROMOTION):
                    if reassert:
                        raise ValueError(f"reassert {v.gate}: {v.reason}")
                    self._write("refused", item=prop["id"], item_kind=kind, source=source, by=prop["by"],
                                why=f"{v.gate}: {v.reason}")
                    return None
            rec = self._write("assert", item=prop["id"], item_kind=kind, body=prop["body"], scope=prop["scope"],
                              source=source, by=prop["by"], evidence=prop["evidence"], just=prop["derived_from"],
                              supersedes=supersedes, reconfirm_after=prop["reconfirm_after"], valid=prop["valid"],
                              level=level, of=prop["of"], **({"inferred": True} if inferred else {}),
                              **({"reassert_proof": proof} if proof else {}),
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

    def _reassert_proof(self, prop, *, check_capability=True):
        """An exact own assertion withdrawal waiver; never an item-wide re-test."""
        if ((check_capability and prop["by"] not in self.reassert_writers) or prop["kind"] != "fact"
                or prop["valid"] is None or prop["derived_from"] or prop["consulted"] or prop["inferred"]):
            raise ValueError("reassert requires a registered capable writer and a direct exact-validity fact")
        it = self.items.get(prop["id"])
        if it is None:
            raise ValueError("reassert requires an earlier exact assertion")
        exact = {a["t"] for a in it.asserts if a["by"] == prop["by"] and a["source"] == prop["source"]
                 and a.get("valid") == prop["valid"] and not a["just"]
                 and not a.get("inferred") and not a.get("consulted")}
        cuts = []
        for r in self.journal:
            if r.get("item") != prop["id"]:
                continue
            if r["op"] in ("retract", "retract_assertions"):
                ns = r.get("assertions", [])
                if (r["op"] != "retract_assertions" or r.get("by") != prop["by"]
                        or not ns or not set(ns) <= exact):
                    raise ValueError("reassert cannot waive external, global or non-exact withdrawals")
                cuts.append({"withdrawal": r["n"], "assertions": ns, "actor": r["by"]})
            if r["op"] in ("flag", "retire"):
                raise ValueError("reassert cannot waive a flag or retirement")
        if any(self._flag_covers(t, prop["id"], ignore_age=True) for t in self._pending_flags()) or not cuts:
            raise ValueError("reassert needs an exact own withdrawal and no flags")
        return {"writer": prop["by"], "source": prop["source"], "valid": prop["valid"], "cuts": cuts}

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

    def supporters(self, iid, *, at=None):
        """The claims standing behind the item as of journal position `at` (default: now — entries up to `at`
        applied): one row {"by", "at", "source"} per claim written by then and not yet taken back there — by its
        own assertion cut, its writer's cut, or the item's retraction. The answer to "who supported this when the
        decision at that position was made": a retraction later than `at` does not rewrite it (S233). A derived
        claim's row carries its premises as "just"; a claim with `valid` carries it."""
        with self._lock:
            it = self.items.get(iid)
            if it is None:
                raise KeyError(f"no item {iid!r}")
            end = self._next_n - 1 if at is None else int(at)
            cut0 = max((t for t in it.retracted if t <= end), default=-1)
            out = []
            for a in it.asserts:
                if not (cut0 < a["t"] <= end):
                    continue
                if any(tc <= end and tc > a["t"] and w == a["by"] for tc, w in it.cuts):
                    continue
                if any(tc <= end and n == a["t"] for tc, n in it.acuts):
                    continue
                row = {"by": a["by"], "at": a["t"], "source": a["source"]}
                if a.get("valid") is not None:
                    row["valid"] = a["valid"]
                if a.get("just"):
                    row["just"] = list(a["just"])
                out.append(row)
            return out

    def _needs_retest(self, iid):
        it = self.items.get(iid)
        if it is None:
            return False
        t = _taint(self.items, iid, self._taints)
        return (t is not None and not any(p > t for p in it.promoted)
                and not (self.revival_window is not None and _window_vouched(it, t)))

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
        reserved = {"kind", "n", "op", "clock", "v", "seq", "id", "time", "prev", "hash", "run"}
        if reserved & data.keys():
            raise ValueError(f"note cannot override reserved journal fields: {sorted(reserved & data.keys())}")
        with self._lock:
            return self._write("note", **plain_json(data))

    # ------------------------------------------------------------------------------------------------ resolution
    def upstream(self, ids):
        """Items whose state can change the state of `ids`: premises (transitively) and same-key competitors."""
        seen, seen_keys, todo = set(), set(), list(ids)
        while todo:
            i = todo.pop()
            if i in seen or i not in self.items:
                continue
            seen.add(i)
            it = self.items[i]
            for a in it.asserts:
                todo.extend(a["just"])
            if it.key is not None and it.key not in seen_keys:
                seen_keys.add(it.key)
                todo.extend(self.by_key[it.key])
        return seen

    def component(self, ids):
        """The connected component(s) of `ids`: premise edges both ways and shared keys."""
        seen, seen_keys, todo = set(), set(), list(ids)
        while todo:
            i = todo.pop()
            if i in seen or i not in self.items:
                continue
            seen.add(i)
            it = self.items[i]
            for a in it.asserts:
                todo.extend(a["just"])
            todo.extend(self.dependents.get(i, ()))
            if it.key is not None and it.key not in seen_keys:
                seen_keys.add(it.key)
                todo.extend(self.by_key[it.key])
        return seen

    def downstream(self, ids):
        """Items whose state a change to `ids` can change: `ids`, what is derived from them (transitively) and same-key
        competitors — what a write re-resolves (the rest keeps its state)."""
        seen, seen_keys, todo = set(), set(), list(ids)
        while todo:
            i = todo.pop()
            if i in seen or i not in self.items:
                continue
            seen.add(i)
            todo.extend(self.dependents.get(i, ()))
            k = self.items[i].key
            if k is not None and k not in seen_keys:
                seen_keys.add(k)
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
        pair = self._append_pair
        self._append_pair = None
        fast = pair is not None and self._touched == {pair[1]}
        comp = set(pair) if fast else self.downstream(self._touched)  # upstream items keep their state: nothing here changes it
        for i in self._touched:                         # the guard cache mirrors the taint memo: a touched item's
            it = self.items.get(i)                      # raw entries may have moved, so its key re-reads it
            if it is not None and it.key is not None:
                g = self._guards.get(it.key)
                if g is None:
                    g = self._guards[it.key] = _GuardKey()
                g.stale.add(i)
        self._touched = set()
        for i in comp:
            self._taints.pop(i, None)
        pending = self._pending_flags()
        flagged = {i for i in comp if any(self._flag_covers(t, i) for t in pending)}
        by_key = self.by_key
        if fast:
            key = self.items[pair[1]].key
            by_key = {key: comp}
            self._guards.pop(key, None)  # a later general pass must rebuild the whole-key guard
        new = _resolve(self.items, comp, by_key, self.resolves, self.clock, flagged, taints=self._taints,
                       state=self.state, guards=self._guards, revival_window=self.revival_window,
                       entry_clocks=self._entry_clocks)
        for i in comp:
            if i in new:
                self.state[i] = new[i]
            else:
                self.state.pop(i, None)
        self._period_epoch += 1                       # O(1) invalidation; no unrelated-key traversal
        for i in comp:
            row = self._fingerprint_row(i) if i in self.state else None
            if row == self._fingerprint_rows.get(i):
                continue
            if row is None:
                self._fingerprint_rows.pop(i, None)
            else:
                self._fingerprint_rows[i] = row
            self._fingerprint_root = _merkle.put(self._fingerprint_root, i,
                None if row is None else _merkle.digest("solvi.km2.leaf", row))
        self._fp = None
        if not fast:  # a plain succession has no disputed versions to open
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

    def usable(self, iid, *, valid_at=None, _seen=frozenset()):
        """Is the item a fact System 1 may answer on alone: active and not stale? valid_at: a world time — and does the
        item hold then: inside its effective intervals (`periods(id)`; an item without `valid` always is), and, for a
        derived item, where its premises hold. A disputed item is usable at a time outside the contested overlap; so
        is an item derived from it, where for one of its derivations every premise is usable then (a premise met again
        on its own derivation chain is not). No valid_at and no world clock (`KnowledgeStore(now=)`): status and
        staleness only — a version closed by a later since still reads usable."""
        if iid in _seen:
            return False
        if valid_at is None and self.now is not None:
            valid_at = self.now() if callable(self.now) else self.now
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
        seen = _seen | {iid}
        if scoped and not any(a["just"] and all(self.usable(p, valid_at=valid_at, _seen=seen) for p in a["just"])
                              for a in _live(self.items[iid])):
            return False
        return self._holds_at(iid, valid_at)

    def periods(self, iid):
        """The world times the item holds as its key's value → [[since, until], ...] (half-open, None: unbounded), or
        None for an item that holds at any time (no `valid` on it or under it). On a key with `valid` a version holds
        from its since until a later version's since (or its own until; a bounded later version gives it back after
        the bounded one ends), outside any contested overlap. A version pending a re-test takes no
        side, but its writer's standing later different value leaves a gap in that writer's older
        version until review/withdrawal; other writers' pending data do not create this gap. A derived item
        holds where its own validity and, for one of its derivations, every premise holds."""
        self._settle()
        return self._periods(iid, set(), _isolated=True)

    def _periods(self, iid, seen, memo=None, *, _isolated=False):
        # Only periods()/holds_at() explicitly mark isolated roots. An empty
        # shared snapshot/find memo still needs its existing population effects;
        # claims and recursive computations keep their original memo semantics.
        root = _isolated and not seen and memo is None and iid in self.items
        epoch = self._period_epoch
        if root:
            cached = self._period_cache.get(iid)
            if cached is not None and cached[0] == epoch:
                return None if cached[1] is None else [list(iv) for iv in cached[1]]
        out = self._compute_periods(iid, seen, {} if _isolated else memo)
        if root:
            frozen = None if out is None else tuple(tuple(iv) for iv in out)
            self._period_cache[iid] = (epoch, frozen)
            return None if frozen is None else [list(iv) for iv in frozen]
        return out

    def _compute_periods(self, iid, seen, memo=None):
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
            p = self._periods(iid, set(), _isolated=True)
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
        query (so redecide can rebuild it). Refuted and retracted items are not given, nor disputed ones except below.
        valid_at: a world time — an item whose `valid` does not cover it is a hint ("not valid at ..."), not a fact. A
        row of an item with a `valid` carries it ("valid": [since, until]). On keys with `valid`, two kinds of rows are
        facts though their item-level status is not "active": a disputed item outside its contested overlap, and a
        hypothesis derived from such an item where every premise of one of its derivations holds. Without valid_at
        each is a fact row carrying "periods" (where it holds) when that is not empty, and is left out (disputed) or
        a hint (derived) when it holds at no time."""
        self._settle()
        valid_at = _time(valid_at)
        q = {"about": about, "relation": relation, "scope": scope, "kinds": None if kinds is None else list(kinds)}
        if valid_at is not None:
            q["valid_at"] = valid_at
        subjects = None if about is None else ({about} if isinstance(about, str) else set(about))
        candidates = None
        if subjects is not None:
            candidates = set().union(*(self._by_subject.get(subject, ()) for subject in subjects))
        if relation is not None:
            try:
                hash(relation)
            except TypeError:
                pass
            else:
                matching = self._by_relation.get(relation, set())
                candidates = matching if candidates is None else candidates & matching
        ids = self.state if candidates is None else (iid for iid in candidates if iid in self.state)
        facts, hints, memo = [], [], {}
        for iid in sorted(ids):
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
            row = {"id": iid, "kind": it.kind, "body": it.body, "scope": it.scope, "source": it.asserts[-1]["source"]}
            v = self.valid(iid)
            if v is not None:                            # the validity a reader filters on (Knowledge.value(valid_at=))
                row["valid"] = v
            p = self._periods(iid, set(), memo) if v is not None or "periods" in self.state[iid] or \
                any(a["just"] for a in it.asserts) else None
            if p is not None and p != [list(_interval(v))]:
                row["periods"] = p                       # where it holds, when that is not its own `valid`
            if st == "hypothesis" and self.state[iid].get("premise_scoped") and p is not None:
                if self.stale(iid):                                            # derived from such a premise: a fact
                    hints.append({**row, "why": "stale"})                      # where its premises hold
                elif valid_at is None and p:
                    facts.append(row if "periods" in row else {**row, "periods": p})
                elif valid_at is not None and self.usable(iid, valid_at=valid_at):
                    facts.append(row if "periods" in row else {**row, "periods": p})
                else:
                    hints.append({**row, "why": "premises hold at no time" if valid_at is None
                                  else f"premises do not hold at {valid_at}"})
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

    def _fingerprint_row(self, i):
        st, it = self.state[i], self.items[i]
        return [i, it.kind, it.body, it.scope, st["status"], st["rank"], st["n_confirm"], st["refuted_by"],
                st["learned_at"], st["invalidated_at"]]

    def fingerprint(self):
        """Versioned SHA-256 Merkle identity of the existing belief-row payload.

        Version 1.5 uses ``km2:sha256:<64 hex>``; old persisted identities must be
        compared with ``legacy_fingerprint()``. Journal hashes and item IDs are
        unchanged. Updates cost expected O(changed log N), reads O(1). Direct
        mutation of internal state/items is unsupported and bypasses this index.
        """
        self._settle()
        return "km2:sha256:" + (self._fingerprint_root.hash if self._fingerprint_root else _merkle.EMPTY)

    def legacy_fingerprint(self):
        """Exact pre-1.5 16-hex identity; rebuilds the full store after writes."""
        self._settle()
        if self._fp is None:
            self._fp = vhash({"items": [self._fingerprint_row(i) for i in sorted(self.state)]})[:16]
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
    def redecide(self, storage, retracted, decide, *, fact="knowledge", upto="decision",
                 skip_retracted=True, valid_at_key=None):
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
        correction = valid_at_key is not None
        if type(skip_retracted) is not bool or (not skip_retracted and not correction):
            raise ValueError("skip_retracted=False requires explicit valid_at_key")
        if correction:
            if upto != "now" or skip_retracted or not isinstance(valid_at_key, str) or not valid_at_key:
                raise ValueError("correction redecision requires upto='now', skip_retracted=False and a date key")
            if not isinstance(fact, str) or not fact or isinstance(retracted, (str, bytes)):
                raise ValueError("fact and affected IDs must be explicit")
            if not callable(decide) and not callable(getattr(decide, "ask", None)):
                raise TypeError("decide must be callable or a System")
        retracted = set(retracted)
        if correction and (not retracted or any(not isinstance(i, str) or i not in self.items for i in retracted)):
            raise ValueError("affected must contain existing item IDs")
        storage = _storage(storage)
        if correction and not callable(getattr(storage, "iter", None)):
            raise TypeError("storage must provide iter()")
        run = _correction_decider(decide) if correction else _decider(decide)
        rebuilt, changes, same = {}, [], []
        prepared = []
        candidate_ids = set()
        for s in storage.iter():
            init = s.input or {}
            snap = init.get(fact)
            if correction and isinstance(snap, dict) and retracted & set(snap.get("rests_on") or ()):
                if type(snap.get("at")) is not int or snap["at"] < 0 or type(snap.get("query")) is not dict:
                    raise ValueError("affected snapshot requires a journal position and query mapping")
                if not isinstance(s.id, str) or not s.id or s.id in candidate_ids:
                    raise ValueError("affected stored decision IDs must be unique nonempty strings")
                candidate_ids.add(s.id)
            if not isinstance(snap, dict) or "at" not in snap or not retracted & set(snap.get("rests_on") or ()):
                continue
            at = None if upto == "now" else int(snap["at"])
            if at not in rebuilt:
                rebuilt[at] = self.rebuild(upto=at, skip=retracted if skip_retracted else ())
            query = dict(snap.get("query") or {})
            if correction:
                raw = init.get(valid_at_key)
                if type(raw) is _dt.date:
                    date = raw.isoformat()
                elif type(raw) is str:
                    try:
                        date = _dt.date.fromisoformat(raw).isoformat()
                    except ValueError:
                        raise ValueError("valid_at_key requires an ISO YYYY-MM-DD date") from None
                    if date != raw:
                        raise ValueError("valid_at_key requires an ISO YYYY-MM-DD date")
                else:
                    raise ValueError("valid_at_key requires an ISO YYYY-MM-DD date")
                if "valid_at" in query and _time(query["valid_at"]) != date:
                    raise ValueError("recorded snapshot valid_at differs from the explicit input date")
                query["valid_at"] = date
            new_snap = rebuilt[at].snapshot(**query)
            if _content(new_snap) == _content(snap):
                continue
            if correction:
                from copy import deepcopy
                prepared.append((s.id, deepcopy(init), deepcopy({**init, fact: new_snap})))
            else:
                old, new = run(init), run({**init, fact: new_snap})
                (changes if old != new else same).append({"id": s.id, "old": old, "new": new})
        for sid, old_input, new_input in prepared:
            (old, old_key, old_unknown), (new, new_key, new_unknown) = run(old_input), run(new_input)
            change = {"id": sid, "old": old, "new": new}
            if old_unknown or new_unknown:
                change.update(old_unknown_paths=old_unknown, new_unknown_paths=new_unknown)
            (changes if old_key != new_key else same).append(change)
        return changes, same


def _content(snap):
    return {"facts": snap.get("facts"), "hints": snap.get("hints")}


def _storage(where):
    """The decisions' storage: a TraceStorage (anything with iter()) as it is, a path opened (open_storage)."""
    if where is None or hasattr(where, "iter"):
        return where
    from ..store import open_storage
    return open_storage(where)


def _correction_json(value, *, unknown=False, active=None):
    """Return finite JSON and a type-aware comparison tree, without repr coercion."""
    import math
    from ..catalog import Unknown
    if unknown and value is Unknown:
        return {"$solvi": "unknown"}, ("unknown",)
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return value, (kind.__name__, value)
    if kind is float and math.isfinite(value):
        return value, ("float", value)
    if kind not in (dict, list):
        raise ValueError("correction callbacks must return finite JSON values")
    active = set() if active is None else active
    if id(value) in active or len(active) > 128:
        raise ValueError("correction callbacks must return acyclic finite JSON values")
    if kind is dict and any(type(k) is not str for k in value):
        raise ValueError("correction callbacks require string JSON keys")
    active.add(id(value))
    try:
        if kind is list:
            pairs = [_correction_json(v, unknown=unknown, active=active) for v in value]
            return [v for v, _ in pairs], ("list", tuple(k for _, k in pairs))
        pairs = [(k, _correction_json(v, unknown=unknown, active=active)) for k, v in sorted(value.items())]
        return {k: v for k, (v, _) in pairs}, ("dict", tuple((k, tag) for k, (_, tag) in pairs))
    finally:
        active.remove(id(value))


def _correction_decider(decide):
    """Baseline is the CURRENT callback rerun on the historical snapshot, not a saved prediction."""
    if callable(getattr(decide, "ask", None)):
        def run(init):
            response = decide.ask(init, store=False)
            raw = {q: [r.answer, r.status] for q, r in sorted(response.results.items())}
            value, key = _correction_json(raw, unknown=True)
            from ..catalog import Unknown
            paths = []

            def walk(node, path):
                if node is Unknown:
                    paths.append(path)
                elif type(node) is list:
                    for i, child in enumerate(node):
                        walk(child, path + "/" + str(i))
                elif type(node) is dict:
                    for name, child in sorted(node.items()):
                        walk(child, path + "/" + name.replace("~", "~0").replace("/", "~1"))

            walk(raw, "")
            return value, key, paths
        return run
    return lambda init: (*_correction_json(decide(init)), [])


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
_IN_PROGRESS = object()                              # a _taint call still on the stack: a premise chain back


def _premise_cycle(items, iid, premises, cap=64):
    """The premise chain from `premises` back to `iid` — an item citing itself, directly or through the items its
    premises derive from: the id the chain re-enters through, or None. Advisory only (the walk treats support through
    a cycle as no support whatever the length): the search stops after `cap` hops, so a loop longer than that warns
    nothing — the cost of a check on every write must not grow with the chain."""
    seen, stack, hops = set(), list(set(premises)), 0
    while stack and hops < cap:
        p = stack.pop()
        hops += 1
        if p == iid:
            return p
        if p in seen:
            continue
        seen.add(p)
        pit = items.get(p)
        if pit is not None:
            stack.extend(q for a in _live(pit) for q in a.get("just", ()))
    return None


def _window_vouched(it, cut):
    # Local eligibility only: descendants still see the historical premise cut.
    return any(a.get("_revival_cut") == cut for a in _live(it))


def _taint(items, iid, memo, deferred=None):
    """The journal position the item's support was last cut at — its own retraction, or, when every assertion since
    then rests on a premise cut after it was written, that premise's cut — or None. An item retracted and then only
    re-inferred (`inferred` assertions: a rule recomputing it from premises) is not cut by its own retraction. Pure in the journal (`memo` keeps it
    between resolutions); an item with one uncut justification left is not cut (so a retraction stays exact: the store
    after it is the store rebuilt without the retracted item). Support through a premise cycle back into the loop is
    no support (S261): it neither keeps the item alive nor taints it by itself — an item whose only support is such a
    loop stays undecided here (returned as the in-progress mark, collected in `deferred`) until the loop's root
    resolves, or, with no outside support at all, is settled by the caller."""
    if iid in memo:
        return memo[iid]
    memo[iid] = _IN_PROGRESS                         # a chain back through this item reads as a cycle, not uncut
    it = items.get(iid)
    if it is None:
        memo[iid] = None
        return None
    cut = _cut(it)
    cut = None if cut < 0 else cut
    live = _live(it)
    via, whole, seen, cyc = [], bool(live), [], False
    for a in live:                                      # an inferred assertion follows its premises: a premise's cut
        tas, alive = [], not a["just"]                  # that its own re-test cleared does not hold it back
        for p in a["just"]:
            tp = _taint(items, p, memo, deferred)
            if tp is _IN_PROGRESS:                      # the premise chain runs back through an open item
                cyc = True
                continue
            if tp is None or tp <= a["t"] or (a.get("inferred") and any(q > tp for q in items[p].promoted)):
                alive = True                            # this premise keeps the assertion's support
            else:
                tas.append(tp)
        if alive:
            whole = False                               # an uncut premise outside any cycle keeps the item alive
        elif tas:
            via.append(max(tas))
        seen += [tp for p in a.get("consulted", ()) if (tp := _taint(items, p, memo, deferred))
                 not in (None, _IN_PROGRESS) and tp > a["t"]]
    if cut is not None and any(a.get("reassert_proof") and
                               cut in {c["withdrawal"] for c in a["reassert_proof"]["cuts"]} for a in live):
        cut = None  # this direct claim vouches only after its exact own withdrawal
    if cut is not None and live and all(a.get("inferred") for a in live):
        cut = None                                      # recomputed from premises: their own cuts decide (above)
    # a consulted item (an alternative's evidence) cut after the assertion: any one marks it for review
    val = max(([cut] if cut is not None else []) + (via if whole else []) + seen, default=None)
    if val is None and cyc and whole and deferred is not None:
        memo.pop(iid, None)                             # undecided: rests on a loop whose root is still open
        deferred.add(iid)
        return _IN_PROGRESS
    memo[iid] = val
    return val


class _GuardItem:
    """One item's cached contribution to its key's revival guard: its withdrawals, the re-tests that vouch for its
    content and its live assertions' matches against the withdrawals — kept between resolutions (S266: every write
    used to re-read every assertion of every item on the key)."""
    __slots__ = ("vj", "counts", "pending", "fixed", "adj", "clears", "live_w")

    def __init__(self, vj):
        self.vj = vj            # the item's value as JSON text (what competes on the key)
        self.counts = None      # the raw-entry lengths the contribution was computed at: equal → nothing moved
        self.pending = []       # [assertion]: standing (no withdrawal yet); an event reaching it is its first
        self.fixed = []         # [(tc, since, (since, until), by)]: withdrawals with a journal-frozen position
        self.adj = []           # [[a, own, xw, tcr]]: an own assertion cut, in only while the writer no longer
        #                           stands behind the content (the _kept read): tc = min(tcr, xw or own)
        self.clears = []        # [(since, tp)]: re-tests vouching for (the value, since) at tp, over the live
        #                           claims — they shrink when those die, growing only by new re-tests
        self.live_w = {}        # {t: (w, deps)}: a live assertion's matches among the fixed withdrawals, frozen at
        #                           its first classification, and the re-tests suppressing the rest (a re-test that
        #                           stops vouching re-freezes it)


class _GuardKey:
    """A key's guard cache: what each of its items contributes and the last pass's result, reusable while no
    item's raw entries have moved."""
    __slots__ = ("items", "stale", "result")

    def __init__(self):
        self.items = {}         # {id: _GuardItem}
        self.stale = set()      # the ids whose raw entries changed since the last pass
        self.result = None      # the last (pending, held)


def _guard_since(a):
    return json.dumps(_interval(a["valid"])[0], sort_keys=True)


def _guard_refresh(items, i, gi, shrunk):
    """(Re)compute one item's cached contribution to its key's guard. The raw entries are replayed as journal
    events — a retraction, a writer's cut, an assertion's cut each take back what is still standing before it, so
    an assertion's withdrawal entry is its first takedown and never moves again (journal positions only grow, and a
    withdrawal matches only assertions written after it). An assertion cut of the assertion's own is the exception:
    whether it is a withdrawal reads the key's live versions (the `_kept` re-admission), so it is resolved per pass
    instead of frozen. `shrunk` collects the re-tests that stopped vouching (the claims they vouched for died):
    frozen matches they suppressed are taken again in the pass that collects them."""
    it = items[i]
    counts = (len(it.asserts), len(it.retracted), len(it.cuts), len(it.acuts), len(it.rechecked))
    if gi is not None and gi.counts == counts:
        return gi
    if gi is None:
        gi = _GuardItem(json.dumps(it.value, sort_keys=True))
    na, nr, nc, nx, _ = (0, 0, 0, 0, 0) if gi.counts is None else gi.counts
    events = ([("retract", t, None) for t in it.retracted[nr:]]
              + [("cut", t, w) for t, w in it.cuts[nc:]]
              + [("acut", t, n) for t, n in it.acuts[nx:]])
    events.sort(key=lambda e: e[1])
    gi.pending.extend(a for a in it.asserts[na:] if a.get("valid") is not None and not a["just"])
    for kind, t, x in events:
        keep = []
        for a in gi.pending:
            if kind == "acut":
                if a["t"] == x:
                    gi.adj.append([a, t, None, None])
                    continue
            elif a["t"] < t and (kind == "retract" or a["by"] == x):
                gi.fixed.append((t, _guard_since(a), _interval(a["valid"]), str(a["by"])))
                continue
            keep.append(a)
        gi.pending = keep
        if kind == "retract":                          # the whole-support positions an _kept re-admission falls
            for e in gi.adj:                           # back to: min(own cut, the writer's next cut, these)
                if e[0]["t"] < t and (e[3] is None or t < e[3]):
                    e[3] = t
        elif kind == "cut":
            for e in gi.adj:
                if e[0]["t"] < t and e[0]["by"] == x and (e[2] is None or t < e[2]):
                    e[2] = t
    old = set(gi.clears)
    gi.clears = sorted({(_guard_since(a), tp) for tp in it.rechecked for a in _live(it)
                        if a["t"] < tp and a.get("valid") is not None and not a["just"]})
    shrunk |= {(gi.vj, since, tp) for since, tp in old - set(gi.clears)}
    if gi.live_w:                                      # assertions that died are never classified again
        keep = {a["t"] for a in _live(it)}
        gi.live_w = {t: w for t, w in gi.live_w.items() if t in keep}
    gi.counts = counts
    return gi


def _guard(items, ids, cache=None, *, revival_window=None, entry_clocks=None):
    """The revival guard on one key: a later assertion with `valid` of content (the key's value) that was withdrawn
    from a writer on an overlapping validity — by any retraction (plain, only_from, an assertion, retract_source, a
    failed re-test, a rollback), on any item — is a copy. A copy is pending a re-test unless writers the content was
    not withdrawn from still hold it, in their uncopied assertions, wherever the copy could hold (outside the spans
    where a later version holds anyway); once they no longer do, it is pending then. A passed re-test of an item
    vouches for its assertions written before it and clears the withdrawals of its content before it, on that
    assertion's validity: a re-test passed on one version vouches for that version, not for the content on every
    validity (S265). Derived and inferred assertions are not copies (they follow their premises). → ({(id, assertion
    position): why} pending, {(id, assertion position)} copies admitted while held).

    `cache` (the store's, one per key) makes the pass incremental (S266): each item's withdrawals, re-test clears
    and live assertions' matches against the withdrawals are kept between resolutions — a withdrawal matches only
    assertions written after it, so an assertion's matches among the frozen withdrawals are computed once, at its
    first classification — and only the items whose raw entries moved are re-read. The `_kept` re-admissions and
    the matches a re-test stopped vouching for are resolved on every pass of a touched key; an untouched key
    reuses its last result whole. Without a cache the pass is computed from scratch, as a rebuilt store does."""
    def matches(tc, ivb, assertion):
        if revival_window is None:
            return _overlap(ivb, _interval(assertion["valid"]))
        return assertion["clock"] - entry_clocks[tc] <= revival_window

    if cache is not None and not cache.stale and cache.result is not None:
        return cache.result
    shrunk = set()                                    # the re-tests that stopped vouching, this pass
    gis = []
    for i in ids:
        gi = None if cache is None else cache.items.get(i)
        if cache is None or gi is None or i in cache.stale:
            gi = _guard_refresh(items, i, gi, shrunk)
            if cache is not None:
                cache.items[i] = gi
        gis.append(gi)
    if cache is not None:
        cache.stale = set()
    vj = {i: gi.vj for i, gi in zip(ids, gis)}
    fixed = [(gi.vj, tc, since, iv, by) for gi in gis for (tc, since, iv, by) in gi.fixed]
    dyn = []                                          # the _kept-dependent withdrawals, resolved this pass
    for i, gi in zip(ids, gis):
        for a, own, xw, tcr in gi.adj:
            x = own
            if _kept(items, ids, vj, i, a, x):        # its writer still stands behind the content there: the cut
                x = xw                                # took a duplicate back, not the content
            tc = tcr
            if x is not None and (tc is None or x < tc):
                tc = x
            if tc is not None:
                dyn.append((gi.vj, tc, _guard_since(a), _interval(a["valid"]), str(a["by"])))
    if not fixed and not dyn:
        if cache is not None:
            cache.result = ({}, {})
        return {}, {}
    clears = defaultdict(list)                        # (content, since) → the re-tests that vouched for it there
    for gi in gis:
        for since, tp in gi.clears:                   # a passed re-test vouches for the versions live at it, on
            clears[(gi.vj, since)].append(tp)         # their own since — not for the content on every validity (S265)
    clean, copies = [], []
    for i, gi in zip(ids, gis):
        for a in _live(items[i]):
            if a.get("valid") is None or a["just"]:
                continue
            iv = _interval(a["valid"])
            sk = json.dumps(iv[0], sort_keys=True)
            fw = gi.live_w.get(a["t"])
            if fw is None or not shrunk.isdisjoint(fw[1]):
                w, deps = [], set()
                for vj_j, tc, since, ivb, by in fixed:
                    if vj_j != gi.vj or not tc < a["t"] or not matches(tc, ivb, a):
                        continue
                    hit = [tp for tp in clears.get((vj_j, since), ()) if tc < tp <= a["t"]]
                    if hit:                           # a re-test between the withdrawal and this assertion
                        deps.update((vj_j, since, tp) for tp in hit)
                    else:
                        w.append((tc, by))
                fw = (w, deps)
                gi.live_w[a["t"]] = fw
            w = list(fw[0])
            for vj_j, tc, since, ivb, by in dyn:
                if vj_j == gi.vj and tc < a["t"] and matches(tc, ivb, a) \
                        and not any(tc < tp <= a["t"] for tp in clears.get((vj_j, since), ())):
                    w.append((tc, by))
            if a.get("reassert_proof"):
                waived = {c["withdrawal"] for c in a["reassert_proof"]["cuts"]}
                w = [(tc, by) for tc, by in w if tc not in waived]
            if not w or any(tp > a["t"] for tp in clears.get((gi.vj, sk), ())):
                clean.append((i, a))
            else:
                copies.append((i, a, w))
    pending, held = {}, {}
    cinfo = [(i, a, vj[i], _interval(a["valid"]), _guard_since(a), str(a["by"])) for i, a in clean]
    for i, a, w in copies:
        ws = sorted({by for _, by in w})
        base = [(vj_j, ivb, (j, sk_j)) for j, _, vj_j, ivb, sk_j, by in cinfo if not (vj_j == vj[i] and by in ws)]
        why = (f"re-published content withdrawn from {', '.join(ws)} (journal entry {max(tc for tc, _ in w)}) and "
               f"held by no other writer")
        if _held(_interval(a["valid"]), vj[i], base):
            held[(i, a["t"])] = (_interval(a["valid"]), vj[i], base, why)
        else:
            pending[(i, a["t"])] = why
    if cache is not None:
        cache.result = (pending, held)
    return pending, held


def _kept(items, ids, vj, i, a, tc):
    """Does a's writer still stand behind a's content on a's validity after a's own cut at `tc`: a live assertion of
    the same content from the same writer, written before the cut, covering a's interval — as far as the key's live
    versions leave it (a later live version closes the keeper at its since, a bounded one gives it back after its
    until, and the writer's own later value at that since replaces it), not by raw containment: a version the walk
    already closed (periods) does not keep a withdrawn claim alive (S262)."""
    iv = _interval(a["valid"])
    live = [(j, b) for j in ids for b in _live(items[j])]
    keep = []
    for j, b in live:
        if vj[j] != vj[i] or b["by"] != a["by"] or not b["t"] < tc or b.get("valid") is None or b["just"]:
            continue
        biv = _interval(b["valid"])
        if any(vj[k] != vj[i] and c["by"] == a["by"] and c.get("valid") is not None and not c["just"]
               and _same_since(_interval(c["valid"])[0], biv[0]) for k, c in live):
            continue                                # the writer's later value at that since: it replaced this version
        keep += _ivs_minus([biv], [_interval(c["valid"]) for _, c in live
                                   if c.get("valid") is not None
                                   and _lo_lt(biv[0], _interval(c["valid"])[0])])
    return _inside(iv, keep)


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


def _continuity_units(items, units, live, retest, pclaims, key_ids):
    """A writer's standing later pending value cannot certify its older value.

    Only that writer's direct assertion is restricted; another writer's pending
    data takes no side, and actual withdrawal removes the continuity gap.
    """
    if not retest and not pclaims:
        return units
    ids = {u[1] for u in units}
    key = items[next(iter(ids))].key
    pending = [(i, a) for i in key_ids if i in live and items[i].key == key
               for a in live[i] if (i in retest or (i, a["t"]) in pclaims)
               and a.get("valid") is not None and not a["just"] and not a.get("consulted")
               and not a.get("inferred") and a.get("by") is not None]
    if not pending:
        return units
    out = []
    for unit, iid, value, iv, lost in units:
        assertions = [a for a in live[iid] if a.get("valid") is not None
                      and _same_since(_interval(a["valid"])[0], iv[0])
                      and (iid, a["t"]) not in pclaims]
        intervals = []
        for a in assertions:
            gaps = []
            for j, b in pending:
                biv = _interval(b["valid"])
                try:
                    later = _lo_lt(iv[0], biv[0])
                except TypeError:
                    later = False  # no declared later chronology; preserve mixed-time resolution
                if (items[j].kind == items[iid].kind and items[j].value != items[iid].value
                        and b["by"] == a["by"] and later):
                    gaps.append(list(biv))
            try:
                intervals += _ivs_minus([iv], gaps) if gaps else [list(iv)]
            except TypeError:
                intervals = []  # incomparable declared chronology cannot certify the older version
                break
        out.extend((unit, iid, value, tuple(period), lost) for period in _ivs_or(intervals))
    return out


def _resolve(items, comp, by_key, resolves, clock, flagged, max_iter=200, taints=None, state=None, guards=None, revival_window=None, entry_clocks=None):
    """The state of the items in `comp` (closed under dependents and shared keys) from their raw entries → {id:
    record} for present items. Fixed point of: liveness of justifications ↔ per-key walk. Deterministic; premises
    outside `comp` are read from `state` (their resolved records), which nothing in `comp` can change. `guards`: the
    store's per-key guard caches (S266) — an untouched key reuses its last guard result untouched."""
    order = sorted(comp, key=lambda i: (min((a["t"] for a in items[i].asserts), default=0), i))
    fixed = {p: state[p] for i in order for a in items[i].asserts for p in a["just"]
             if p not in comp and p in (state or {})}
    taints = {} if taints is None else taints
    retest, why = set(), {}
    deferred = set()                                     # items whose only support is a premise cycle (S261)
    for i in order:
        _taint(items, i, taints, deferred)               # _flush has popped the component: these recompute
    deferred = {i for i in deferred if i not in taints}   # one resolved later in the same pass is not deferred
    for i in sorted(deferred):                           # no outside support anywhere in the loop: nothing vouches
        taints[i] = max((a["t"] for a in _live(items[i])), default=None)   # for the content — it waits for a re-test
    for i in order:
        t = taints.get(i)
        if t is not None and not any(p > t for p in items[i].promoted) and not (revival_window is not None and _window_vouched(items[i], t)):
            retest.add(i)
            why[i] = ("its only support runs through a premise cycle: nothing outside it vouches for the content"
                      if i in deferred else _retest_why(items, i, t, taints))
    if deferred and taints is not None:
        taints.clear()                                   # a cycle's settlement shifts other items' taints too:
        # recompute from scratch next time (rare: only with cycles)
    pclaims, hinfo = {}, {}                             # the revival guard: copies of withdrawn content
    for k in sorted({items[i].key for i in order if items[i].key is not None}):
        p, h = _guard(items, sorted(by_key[k]), guards.get(k) if guards is not None else None,
                      revival_window=revival_window, entry_clocks=entry_clocks)
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
            effective_units = _continuity_units(items, units[k], live, retest, pclaims, by_key[k])
            effective = _effective(effective_units) or {}
            for _, iid, _, _, _ in units[k]:
                effective.setdefault(iid, {"periods": [], "contested": []})
            eff.update(effective)
    scoped = set()                                      # derived items held up only by a dispute scoped to an
    grew = True                                         # overlap: each holds where its premises hold (usable /
    while grew:                                         # snapshot at a time read the premises' periods there)
        grew = False
        for i in order:
            if i in on_disputed and i not in scoped and any(
                    all(status.get(p) == "active" or p in scoped or (fixed[p].get("premise_scoped") if p in fixed else False)
                        or (status.get(p) == "disputed"
                            and bool(eff[p]["periods"] if p in eff else (fixed.get(p) or {}).get("periods")))
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
        on_disputed = set()                             # would be active but no derivation has all premises active
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
