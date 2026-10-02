"""TraceStorage: stored responses with their traces, a hash chain across them, queries, a replay of everything stored, and
provenance questions over the store (which stored decisions rest on a fact found to be wrong; what a forgotten given fact
touches).

Every stored decision is one record (a dict of plain JSON):

  kind "ask"      the 0.5 journal line's keys — init_hash, answers {question: [answer, confidence, status]}, flow, records
                  [[step, name, hash]] (and producers) — plus index fields (guards, safeguards, models) and the whole
                  response (Response.to_dict(): answers, flow, trace), which get() loads back and replay_all() re-checks;
  kind "teach"    a correction (System.teach): {"teach": question, "init": ..., "answer": ...} and, when given, its
                  "source" ("outcome", "rule"; none: a human), "by" and "of" (the stored id of the decision it corrects);
  kind "update"   a learning update (System.learning): what changed, the gates' results, the state to roll back to;
  every record    seq (0, 1, 2, ...), time (seconds since the epoch), meta (optional, yours), prev and hash.

A record is written with every dict's keys in their own order (a decider reads a dict's keys in that order, so a
stored trace gives its dicts back as they were). `hash` is the SHA-256 of the record's canonical JSON (keys sorted,
without `id` and `hash`) — it does not depend on the order the record was written in — which includes `prev`, the
hash of the record before it: editing, deleting, inserting or reordering a stored record breaks the chain at that point
(verify()). Cutting records off the end leaves a shorter chain that is still valid, so the store keeps its head — the
count and the last hash (head()) — next to the log and verify() checks it; a head you published elsewhere
(verify(anchor=head)) also catches a rewrite of the whole chain together with the stored head.

Backends: JSONLStorage (append-only file, one record per line; what System(storage="file.jsonl") writes), SQLiteStorage (stdlib
sqlite3; indexed by question, answer, status, safeguard, model fingerprint and time; several processes may write),
PostgresStorage (psycopg 3; the same tables, several services writing) and DuckDBStorage (duckdb; the same tables in a
DuckDB file, for analytics)."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time as _time
from dataclasses import dataclass

from . import _deprecate

try:
    import fcntl
except ImportError:                           # Windows, the browser (Pyodide): no advisory file locks
    fcntl = None

GENESIS = ""                                  # prev of the first record
FORMAT = 2                                    # the record format ("v"): 2 — dicts are written with their keys in their
                                              # own order (1, solvi ≤ 0.7.1: sorted, the order a decider read is lost)
LEGACY_ORDER = ("stored by solvi 0.7.1 or earlier, which wrote dict keys sorted: a model that read a dict may have read "
                "its keys in another order than this replay gives it — replay such records with trust_models=True")


TRUSTED_SOURCES = ("human", "outcome", "rule")   # where a label may come from: never the system's own answers


class UntrustedLabel(ValueError):
    """A label from a source outside TRUSTED_SOURCES (the model, the system itself, an unknown process)."""


def check_source(source):
    """A label's source → itself, when it is trusted: "human", "outcome" or "rule"; else UntrustedLabel."""
    if source not in TRUSTED_SOURCES:
        raise UntrustedLabel(f"label source {source!r} is not trusted: labels come only from outside the model "
                             f"({', '.join(TRUSTED_SOURCES)}); the system's own answers are never labels")
    return source


class _Any:
    def __repr__(self):
        return "ANY"


ANY = _Any()                                  # a query / provenance argument that was not given (None is a real answer)


def _cj(obj):
    """Canonical JSON: sorted keys, no spaces — what the record hash is taken over. New records hold no inf / nan (they are
    tagged, see entry); records written before 0.7 may, and still hash as they were written (Infinity / NaN)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@contextlib.contextmanager
def _flock(fh, shared=False):
    """An advisory lock on an open file for the block (POSIX flock: exclusive, or shared for readers); where there is
    none (Windows, the browser) the block runs unlocked."""
    if fcntl is None:
        yield
        return
    fcntl.flock(fh.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _body(rec):
    """A record as it is written: JSON with every dict's keys in their own order. The hash does not depend on that order
    (it is taken over the canonical JSON, _cj); a decider does — the text it reads lists a dict's keys in their order
    (solvi.decide.state_text) — so a stored trace must give back the dicts as they were for its model steps to replay."""
    return json.dumps(rec, ensure_ascii=False, allow_nan=False)


KEPT = ("v", "kind", "seq", "time", "prev", "init_hash", "catalog", "records", "flow", "producers", "models", "guards",
        "safeguards", "teach", "source", "of", "of_hash", "by", "note")
"""The fields of a record that are never erased (redact leaves them; a redaction record is made of them)."""


def _digest(x):
    return hashlib.sha256(_cj(x).encode()).hexdigest()


def record_body(rec):
    """What a record's hash is taken over. Format 1 (solvi ≤ 0.7.1): the record without `id` and `hash`. Format 2: the
    fields that stay for ever (KEPT), the digest of its answers and the digest of everything else (its content: the
    response, the input, the meta) — three parts, so that redact can remove the content, or the answers too, leave their
    digests in the record's mark, and the hash still recomputes: what is left of a redacted record is verified like any
    other record, and a record cannot be passed off as redacted with other answers."""
    if int(rec.get("v") or 1) < 2:
        return {k: v for k, v in rec.items() if k not in ("id", "hash")}
    mark = rec.get("redacted") if isinstance(rec.get("redacted"), dict) else {}
    body = {k: rec[k] for k in KEPT if k in rec}
    body["#answers"] = mark["answers"] if "answers" in mark else _digest(rec.get("answers"))
    body["#content"] = mark["content"] if "content" in mark else _digest(
        {k: v for k, v in rec.items() if k not in KEPT and k not in ("answers", "id", "hash", "redacted")})
    return body


def record_hash(rec):
    """The hash of a stored record: SHA-256 of the canonical JSON of its body (record_body; it covers `prev`)."""
    return _digest(record_body(rec))


def plain(v):
    """A value as stored: JSON data (sets sorted as vhash sorts them, dates as ISO strings, "not stated" as "<not stated>")."""
    from .core import NOT_STATED_KEY, Unknown
    from .schema import jsonable
    if v is Unknown:
        return NOT_STATED_KEY
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return jsonable(v)


def akey(v):
    """An answer as an index key (canonical JSON of its stored form): answer=("a", "b") finds a stored ["a", "b"]."""
    from .schema import tag_floats
    return _cj(tag_floats(plain(v)))


def _when(t):
    """since / until → seconds since the epoch: a number, a datetime (naive: local time), a date (its midnight) or an ISO
    string."""
    import datetime
    if t is None or isinstance(t, (int, float)):
        return t
    if isinstance(t, str):
        t = datetime.datetime.fromisoformat(t)
    if isinstance(t, datetime.datetime):
        return t.timestamp()
    if isinstance(t, datetime.date):
        return datetime.datetime(t.year, t.month, t.day).timestamp()
    raise TypeError(f"not a time: {t!r}")


# --- the record of a response
def entry(resp, meta=None):
    """The stored content of a response (without seq, time, prev, hash)."""
    d = resp.to_dict()
    tr = resp.trace
    e = {"v": FORMAT, "kind": "ask", "init_hash": tr.init_hash,
         "answers": {q: [plain(r.answer), round(r.confidence, 4), r.status] for q, r in resp.results.items()},
         "flow": [s.part.name for s in resp.flow.steps],
         "records": [[r.step, r.name, r.hash] for r in tr.records]}
    if any(r.tried is not None for r in tr.records):
        e["producers"] = {r.name: r.producer for r in tr.records if r.tried is not None}
    e["guards"] = {q: r.guard for q, r in resp.results.items() if r.guard}
    e["safeguards"] = [[s["kind"], s["fact"], sorted(s.get("questions") or [])] for s in resp.safeguards or ()]
    models = {}
    for r in tr.records:                              # the models that ran: the one used and, for a fact with
        for rm in [r.model, *(r.tried_models or {}).values()]:    # alternative producers, the rejected ones too
            if rm is not None:
                m = {k: str(rm.get(k)) for k in ("type", "id", "fp")}
                models[_cj(m)] = m
    e["models"] = [models[k] for k in sorted(models)]
    fp = getattr(tr, "fingerprint", None) or {}
    if fp.get("catalog"):
        e["catalog"] = fp["catalog"]                  # the catalog that decided (System.fingerprint)
    if meta is not None:
        e["meta"] = plain(meta)
    from .schema import tag_floats
    e = tag_floats(e)                                 # strict JSON: an inf threshold is {"$float": "inf"}
    e["response"] = d                                 # to_dict() has tagged it already
    return e


@dataclass
class Stored:
    """One stored record: its id, position, time (seconds since the epoch), kind ("ask": a decision; "teach": a
    correction; "redaction": the mark of an erasure, see TraceStorage.redact) and the record itself."""
    id: str
    seq: int
    time: float
    kind: str
    data: dict
    catalog: object = None

    @property
    def answers(self):
        """question → the stored answer (JSON form: tuples as lists, "not stated" as "<not stated>")."""
        from .schema import untag_floats
        return {q: untag_floats(a[0]) for q, a in (self.data.get("answers") or {}).items()}

    @property
    def meta(self):
        from .schema import untag_floats
        return untag_floats(self.data.get("meta"))

    def response(self, catalog=None):
        """The stored response, loaded back (typed values restored from `catalog` — a Catalog or a System — or from the
        store's own)."""
        from .system import Response
        if self.kind != "ask":
            raise ValueError(f"record {self.id} is a {self.kind} record, not a response")
        if self.data.get("redacted"):
            raise ValueError(f"record {self.id} was redacted: its response is gone")
        r = Response.model_validate(self.data["response"], catalog=catalog if catalog is not None else self.catalog)
        r.stored_id = self.id
        return r


def chained(d):
    """Is this a record of the chain (a dict with its hash)? A JSON object without one — a line another tool appended to
    the file, a hand edit — is not: it is passed over by iter / query / replay_all, and verify reports it."""
    return isinstance(d, dict) and isinstance(d.get("hash"), str)


def _stored(d, catalog=None):
    return Stored(d["id"], d["seq"], d["time"], d.get("kind", "ask"), d, catalog)


# --- filters shared by the backends (SQLite runs the same filters as SQL)
def _matches(d, question, answer, status, safeguard, model, since, until, catalog=None):
    if catalog is not None and d.get("catalog") != catalog:
        return False
    if since is not None and d["time"] < since:
        return False
    if until is not None and d["time"] >= until:
        return False
    if question is not None or answer is not ANY or status is not None:
        ak = None if answer is ANY else akey(answer)
        if not any((question is None or q == question) and (ak is None or _cj(a[0]) == ak) and
                   (status is None or a[2] == status) for q, a in (d.get("answers") or {}).items()):
            return False
    if safeguard is not None:
        if not any(s[0] == safeguard and (question is None or question in s[2]) for s in d.get("safeguards") or ()):
            return False
    if model is not None:
        if not any(model in (m.get("fp"), m.get("id"), m.get("type")) for m in d.get("models") or ()):
            return False
    return True


def _summary_problems(d):
    """A stored response agrees with its own summary: the answers, the input hash and the trace records' hashes, and the
    trace's own chain links (no catalog needed; replay_all re-computes the steps)."""
    out = []
    resp = d.get("response")
    if not isinstance(resp, dict):
        return ["no stored response"]
    tr = resp.get("trace") or {}
    recs = tr.get("records") or []
    if tr.get("init_hash") != d.get("init_hash"):
        out.append("stored response has another init_hash than its record")
    if [[r.get("step"), r.get("name"), r.get("hash")] for r in recs] != d.get("records"):
        out.append("stored trace records differ from the record's hashes")
    prev = tr.get("init_hash")
    for r in recs:
        if r.get("prev") != prev:
            out.append(f"stored trace chain broken at step {r.get('step')} ({r.get('name')})")
            break
        prev = r.get("hash")
    res = resp.get("results") or {}
    got = {q: [r.get("status"), _cj(None if r.get("not_stated") else r.get("answer"))] for q, r in res.items()}
    want = {q: [a[2], _cj(None if a[0] == "<not stated>" else a[0])] for q, a in (d.get("answers") or {}).items()}
    if got != want:
        out.append("stored answers differ from the record's answers")
    return out


class TraceStorage:
    """A store of responses and their traces with a hash chain across the stored records (see the module docstring).

    save(response, meta=None) → id; get(id) → Response; record(id) → the stored dict; iter() / query(...) → [Stored];
    corrections() → the teach records; head() → {"count", "hash"}; signature(); verify(anchor=None, signature=None);
    replay_all(system);
    quarantine(fact, value=...); forget(fact, value=...).

    `catalog`: a Catalog or System used to restore typed values (dates, enums, models) when loading responses;
    System(storage=...) sets it to that system when it is not set. `clock`: a function → seconds since the epoch."""

    def __init__(self, catalog=None, clock=None):
        self.catalog = catalog
        self.clock = clock or _time.time

    # --- a backend implements these
    def _append(self, body):
        """Add a record: assign seq, time, prev, hash and id atomically; → the record."""
        raise NotImplementedError

    def _raw(self, snap=None):
        """Every chained record in stored order (for verify) → iterator of (position, dict or None when unreadable).
        `snap`: a _snapshot() — the records as they were then."""
        raise NotImplementedError

    def _snapshot(self):
        """What verify() reads before the records so that records appended while it runs are not taken for damage: the
        stored head (and what else the backend needs) at one moment. None: the backend has nothing to pin."""
        return None

    def _find(self, id):
        raise NotImplementedError

    def head(self):
        """{"count", "hash"}: the number of chained records and the last record's hash (GENESIS when empty). Publish it
        somewhere else (a ticket, a log, a signed message) to later catch a rewrite of the whole store: verify(anchor=...)."""
        raise NotImplementedError

    def _backend_problems(self, rows, snap=None):
        """Checks specific to a backend (its stored head, index tables) → [(seq, id, reason)]. `snap`: verify's
        _snapshot(), taken before `rows` were read."""
        return []

    def _legacy(self):
        return 0

    # --- writing
    def save(self, response, meta=None):
        """Store a response (its answers, flow and whole trace) → its id. `meta`: your own JSON data kept with it (a
        ticket id, a user) — hashed into the chain like the rest."""
        rec = self._append(entry(response, meta))
        response.stored_id = rec["id"]
        return rec["id"]

    @_deprecate.kwargs(source="label_source")
    def save_correction(self, question, init_state, answer, meta=None, *, label_source="human", by=None, of=None):
        """Store a correction (what System.teach records) → its id. label_source (`source=` in 0.7; stored as the
        record's "source"): where the label comes from — "human" (a person
        corrected or confirmed the answer), "outcome" (what really happened: the parcel was lost, the loan defaulted) or
        "rule" (code rejected a model's proposal and decided instead); anything else is refused (UntrustedLabel): the
        system's own answers are never labels. by: who (a user, a reviewer, a process); of: the stored id of the decision
        it corrects. Records of 0.6 have no source: they are human corrections."""
        source = label_source
        check_source(source)
        body = {"v": FORMAT, "kind": "teach", "teach": question, "init": plain(dict(init_state)), "answer": plain(answer)}
        if source != "human":
            body["source"] = source
        if by is not None:
            body["by"] = str(by)
        if of is not None:
            body["of"] = str(of)
        if meta is not None:
            body["meta"] = plain(meta)
        return self._append(body)["id"]

    # --- reading
    def record(self, id):
        """The stored record (a dict) with this id; KeyError if there is none."""
        d = self._find(id)
        if d is None:
            raise KeyError(f"no stored record {id!r}")
        return d

    def get(self, id, catalog=None):
        """The stored response with this id, loaded back (see Stored.response)."""
        return _stored(self.record(id), self.catalog).response(catalog)

    def iter(self, kind="ask", redacted=False):
        """Stored records in order (kind "ask": responses; "teach": corrections; None: all) → iterator of Stored. A
        redacted record (see redact) has no content left and is passed over; redacted=True: it is yielded too."""
        for _, d in self._raw():
            if chained(d) and (kind is None or d.get("kind", "ask") == kind) and (redacted or not d.get("redacted")):
                yield _stored(d, self.catalog)

    def __iter__(self):
        return self.iter()

    def __len__(self):
        """The number of chained records — decisions, corrections and redaction marks (head()["count"]); the decisions
        alone: len(list(store.iter()))."""
        return self.head()["count"]

    def close(self):
        """Release what the store holds open (a database connection; nothing for a JSON-lines file). A store is also a
        context manager: `with SQLiteStorage("decisions.db") as store: ...` closes it at the end."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _answer_key(self, question, answer):
        """query's `answer=` in the form answers are stored: through the question's answer type when the store knows the
        System (True → "yes" for a yes/no question, an Enum → its value), else a bool as "yes" / "no" (a yes/no answer
        is stored as text) — so answer=True finds what answer="yes" finds."""
        if answer is ANY or answer is None:
            return answer
        qs = getattr(self.catalog, "questions", None) or {}
        names = [question] if question is not None else list(qs)
        for n in names:
            at = getattr(qs.get(n), "answer", None)
            if at is not None:
                try:
                    return at.normalize(answer)
                except ValueError:
                    continue
        if isinstance(answer, bool):
            return "yes" if answer else "no"
        return answer

    def corrections(self):
        """The stored corrections → [{"id", "time", "question", "init", "answer", "source", "by", "of"}] (feed them to fit /
        learn_rule, a CorrectionMemory or System.learning). source: "human" (also every record without one), "outcome",
        "rule" — or, for a record written around save_correction, whatever it says (solvi.memory and System.learning refuse
        anything outside TRUSTED_SOURCES)."""
        from .schema import untag_floats                # stored tagged ({"$float": "inf"}), read back as the float
        return [{"id": s.id, "time": s.time, "question": s.data["teach"], "init": untag_floats(s.data["init"]),
                 "answer": untag_floats(s.data["answer"]), "source": s.data.get("source", "human"), "by": s.data.get("by"),
                 "of": s.data.get("of")}
                for s in self.iter("teach")]

    def query(self, question=None, answer=ANY, status=None, safeguard=None, model=None, since=None, until=None,
              catalog=None):
        """Stored responses that match every filter given → [Stored] in stored order.
        question: asked this question; answer: answered this (with question: that question's answer; None matches an
        abstention); status: "ok" / "forced" / "abstain" (with question: of that question); safeguard: a safeguard of this
        kind fired ("grounding", "hard_check", "low_confidence", ...; with question: one that concerns it); model: a model
        with this fingerprint, id or type produced a step; since / until: stored in [since, until) — seconds since the
        epoch, a datetime, a date or an ISO string; catalog: decided by the catalog with this fingerprint
        (System.fingerprint()["catalog"]). An answer is matched in its stored form: answer=True finds a yes/no "yes"
        (see _answer_key)."""
        since, until = _when(since), _when(until)
        answer = self._answer_key(question, answer)
        return [s for s in self.iter() if _matches(s.data, question, answer, status, safeguard, model, since, until, catalog)]

    def report(self, since=None, until=None, question=None, format="md", examples=3, system=None, **filters):
        """A human-readable report of the stored decisions in [since, until) (optionally of one question): counts by
        answer, status and safeguard, the escalation rate, the guarantee coverage of the answers a model took part in,
        changes of the catalog's and the models' fingerprints, and `examples` stored ids per answer, escalation and
        safeguard. format: "md", "html" (one self-contained page) or "data" (a dict); other `filters` as query. See
        solvi.report."""
        from .report import period, render
        return render(period(self, since, until, question, examples, system, **filters), format)

    # --- erasure
    def _rewrite(self, rec):
        """Replace the stored record with this seq by `rec` (same seq, prev, hash and id)."""
        raise NotImplementedError

    def redact(self, id, by=None, note=None, keep_answers=True):
        """Erase a stored record's content — a person's data that must go — and keep the chain whole.

        The record keeps its place, its time, its hash and its id, so every link after it still verifies; its content
        (the response with its trace and input, the meta; a correction's input and answer) is removed and it is marked
        `redacted` with who and why and the digest of what was removed. A record of kind "redaction" is appended that
        names the erased record and its hash: the erasure is itself in the chain, with its time. keep_answers=False
        removes the answers too (their digest stays in the mark). The record no longer replays and is passed over by
        iter / query / replay_all / reports; record(id) returns what is left. → the redaction record's id.

        What is left stays verified: a record's hash is taken over its lasting fields, the digest of its answers and the
        digest of its content (record_body), so verify() recomputes the hash of a redacted record like any other — an
        answer edited in it afterwards, or a record passed off as redacted with other answers, does not verify — and a
        signature taken before the erasure still holds. A record written by solvi ≤ 0.7.1 (format 1) has one flat hash:
        redacting it leaves its kept fields unverifiable, which verify() lists under "unverified".
        What it cannot do: copies made before (a backup, an exported report, a published anchor's holder) are not
        touched, and derived state (a correction memory, a fitted head) keeps what it learned — rebuild those."""
        rec = self.record(id)
        if rec.get("redacted"):
            raise ValueError(f"record {id} is already redacted")
        if rec.get("kind") == "redaction":
            raise ValueError("a redaction record cannot be redacted")
        new = {k: rec[k] for k in KEPT + ("hash", "id") if k in rec}
        mark = {"by": None if by is None else str(by), "note": None if note is None else str(note)}
        if int(rec.get("v") or 1) < 2:                # one flat hash: only the digest of the whole record can be kept
            from .signature import record_digest
            mark["digest"] = record_digest(rec).hex()
            if "answers" in rec and keep_answers:
                new["answers"] = rec["answers"]
        else:
            body = record_body(rec)
            mark["content"] = body["#content"]
            if "answers" in rec and keep_answers:
                new["answers"] = rec["answers"]
            else:
                mark["answers"] = body["#answers"]
        new["redacted"] = mark
        out = self._append({"v": FORMAT, "kind": "redaction", "of": rec["id"], "of_hash": rec["hash"],
                            "by": mark["by"], "note": mark["note"]})
        self._rewrite(new)
        return out["id"]

    # --- integrity
    def signature(self, alg="syndrome"):
        """The signature of the chained records (solvi.signature.sign): {"alg", "count", "root"} — with the default
        "syndrome" code two numbers (64 bytes). Keep it where you keep the head: verify(signature=...) then names the one
        record that changed — even when every hash after it and the stored head were recomputed — and restores its
        content hash."""
        from .signature import sign
        return sign(self, alg)

    def verify(self, anchor=None, signature=None, candidates=None):
        """Check the chain across stored records: each record's hash, its link to the record before it, the sequence
        numbers, the stored response against its own summary, and the stored head (a cut-off tail). `anchor`: a head()
        taken earlier and kept elsewhere — the chain must still contain it (catches a rewrite of the whole store).
        `signature`: a signature() taken earlier and kept elsewhere — the records it covers must be the ones signed; when one
        changed, its seq is named and `signature` in the result holds solvi.signature.check's answer (the original content
        hash; `candidates`: records, e.g. from a backup, one of which may be the original → its "match").
        → {"ok", "count", "head", "legacy", "problems": [(seq, id, reason)]} (+ "signature" when given; + "unverified":
        [(seq, id, reason)] — redacted records of format 1, whose kept fields no hash covers). Records written before the
        chain (0.5 journal lines) are counted in `legacy` and not checked.
        A store that is being written to verifies as it stands at one moment: the stored head is read first and the
        records are checked against it, so a record appended while verify runs is not reported as damage."""
        problems, rows = [], []
        prev, n = GENESIS, 0
        snap = self._snapshot()
        erased, named = {}, {}                        # redacted records, and the redaction records that name them
        unverified, newest = [], 1                    # format-1 redactions; the highest format seen so far
        for pos, d in self._raw(snap):
            if d is None:
                problems.append((n, None, f"record at position {pos} is not readable JSON (a record cut short by a crash "
                                          "while it was written, or an edit)"))
                continue
            if not chained(d):                        # passed over like an unreadable line: the chain goes on after it
                problems.append((n, None, f"record at position {pos} is not a record of this store: a JSON object "
                                          "without a hash (a line appended by another tool, or an edit)"))
                continue
            rid = d.get("id")
            v = int(d.get("v") or 1)
            if v < newest:                            # formats only go up: a record cannot claim an older one to escape
                problems.append((d.get("seq"), rid, f"record of format {v} after a record of format {newest}"))   # its checks
            newest = max(newest, v)
            if d.get("kind") == "redaction":
                named[(d.get("of"), d.get("of_hash"))] = (d.get("by"), d.get("note"))
            mark = d.get("redacted")
            if mark:                                  # its content is gone
                mark = mark if isinstance(mark, dict) else {}
                erased[rid] = (d.get("seq"), d.get("hash"), mark.get("by"), mark.get("note"))
                extra = sorted(k for k in d if k not in KEPT and k not in ("answers", "id", "hash", "redacted"))
                if extra:
                    problems.append((d.get("seq"), rid, "record marked redacted still holds content: " + ", ".join(extra)))
                if v < 2:                             # one flat hash over content that is gone: nothing to recompute
                    unverified.append((d.get("seq"), rid, "redacted record of format 1: no hash covers what is left of it"))
                elif "content" not in mark or ("answers" in mark and "answers" in d):
                    problems.append((d.get("seq"), rid, "the redaction mark does not hold the digests of what was removed"))
                elif d.get("hash") != record_hash(d):
                    problems.append((d.get("seq"), rid, "redacted record edited after it was stored (its hash does not match)"))
            elif d.get("hash") != record_hash(d):
                problems.append((d.get("seq"), rid, "record edited after it was stored (its hash does not match)"))
            if rid != d["hash"][:16]:
                problems.append((d.get("seq"), rid, "record id does not match its hash"))
            if d.get("prev") != prev:
                problems.append((d.get("seq"), rid, "chain broken: a record before this one was deleted, inserted, "
                                                    "reordered or edited"))
            if d.get("seq") != n:
                problems.append((d.get("seq"), rid, f"sequence number {d.get('seq')} where {n} was expected (records "
                                                    "deleted, inserted or reordered)"))
            if d.get("kind", "ask") == "ask" and not d.get("redacted"):
                problems += [(d.get("seq"), rid, p) for p in _summary_problems(d)]
            rows.append(d)
            prev, n = d.get("hash"), n + 1
        for rid, (seq, h, by, note) in erased.items():
            if (rid, h) not in named:
                problems.append((seq, rid, "record marked redacted without a redaction record that names it and its hash "
                                           "(content removed outside redact)"))
            elif named[(rid, h)] != (by, note):
                problems.append((seq, rid, "the redaction mark differs from its redaction record (who, why)"))
        problems += self._backend_problems(rows, snap)
        if anchor is not None:
            k, h = int(anchor["count"]), anchor["hash"]
            if k > len(rows):
                problems.append((k - 1, None, f"the anchor has {k} records, the store only {len(rows)}: records were "
                                              "removed from the end"))
            elif k > 0 and rows[k - 1].get("hash") != h:
                problems.append((k - 1, rows[k - 1].get("id"), "the record at the anchor differs from the anchored one: "
                                                               "the store was rewritten"))
        out = {"ok": not problems, "count": len(rows), "head": {"count": len(rows), "hash": prev}, "legacy": self._legacy()}
        if unverified:
            out["unverified"] = unverified
        if signature is not None:
            from .signature import check
            sc = check(self, signature, candidates)
            out["signature"] = sc
            if not sc["ok"]:
                i = sc["index"]
                if i is None:
                    problems.append((None, None, f"the signature does not match: {sc['reason']}"))
                else:
                    rid = rows[i].get("id") if i < len(rows) else None
                    problems.append((i, rid, "the record differs from the signed one (its original content hash "
                                             f"{sc['digest'][:16]}...)" + (" — a candidate matches it" if sc["match"]
                                                                          is not None else "")))
        out["ok"] = not problems
        out["problems"] = problems
        return out

    def replay_all(self, system, trust_models=False, **filters):
        """Replay every stored trace (or those matching query `filters`) against `system` (a System or a Catalog): each
        step is re-computed from its recorded inputs (see Trace.replay). → the ones that fail: [{"id", "seq", "time",
        "mismatches": [(step, name, reason)], "models": [(step, name, verdict)], "catalog", "kinds", "summary"}] — empty
        when all replay. Each mismatch has a `.kind`, and "summary" tells damaged data from a catalog or a model that
        changed since (see solvi.runtime.Mismatch). A stored record that cannot be loaded is one mismatch (0, "load", ...),
        a replay that raises is (0, "replay", ...): both of kind "error", no verdict on the data. "note" (a record of
        format 1 whose model step does not recompute): it was stored with sorted dict keys, see LEGACY_ORDER."""
        from .runtime import Mismatch, mismatch_summary
        bad = []

        def failed(s, stage, e):
            ms = [Mismatch(0, stage, f"{type(e).__name__}: {e}", "error")]
            bad.append({"id": s.id, "seq": s.seq, "time": s.time, "mismatches": ms, "models": [], "catalog": None,
                        **mismatch_summary(ms)})

        for s in (self.query(**filters) if filters else self.iter()):
            try:
                r = s.response(system)
            except Exception as e:  # noqa: BLE001
                failed(s, "load", e)
                continue
            try:
                rep = r.trace.replay(system, r.flow, trust_models=trust_models)
            except Exception as e:  # noqa: BLE001
                failed(s, "replay", e)
                continue
            if not rep["ok"]:
                b = {"id": s.id, "seq": s.seq, "time": s.time, "mismatches": rep["mismatches"], "models": rep["models"],
                     "catalog": rep["catalog"], "kinds": rep["kinds"], "summary": rep["summary"]}
                rerun = {step for step, _, verdict in rep["models"] if verdict == "recomputed"}
                if int(s.data.get("v") or 1) < 2 and any(m.kind == "recompute" and m[0] in rerun for m in rep["mismatches"]):
                    b["note"] = LEGACY_ORDER          # a model step of an old record did not recompute: maybe only this
                bad.append(b)
        return bad

    # --- provenance over the store
    def quarantine(self, fact, value=ANY):
        """A fact (or a given input) found to be wrong: the stored decisions whose answers rest on it — through the trace's
        provenance graph (each step's recorded inputs, from the answer back to the fact; a hard check that decided an
        answer counts). `value`: only where the fact had this value (compared by its hash as the consumers recorded it).
        → [{"id", "seq", "time", "questions": {question: {"answer", "status", "path": [fact, ..., "answer:question"]}}}].
        Nothing is changed: re-decide or review these."""
        from .runtime import vhash
        vh = None if value is ANY else vhash(value)
        out = []
        for s in self.iter():
            hit = _dependents(s.data["response"], fact, vh)
            if hit:
                out.append({"id": s.id, "seq": s.seq, "time": s.time, "questions": hit})
        return out

    def forget(self, fact, value=ANY):
        """What removing a given fact (e.g. a person's data) would touch — a report only: nothing is deleted (deleting a
        stored record breaks the chain by design; keep the report as the record of the request, and erase the records it
        lists with redact).
        → {"fact", "value", "dependent": the stored decisions whose answers rest on it (as quarantine), "stored": ids of the
        stored records that hold it without an answer resting on it (responses and corrections), "deleted": 0}."""
        from .runtime import vhash
        from .schema import untag_floats
        vh = None if value is ANY else {vhash(value), vhash(plain(value))}
        dependent, stored = [], []
        for s in self.iter(None):
            if s.kind == "ask":
                init = (s.data["response"].get("trace") or {}).get("init") or {}
                hit = _dependents(s.data["response"], fact, None if vh is None else vhash(value))
                if hit:
                    dependent.append({"id": s.id, "seq": s.seq, "time": s.time, "questions": hit})
                    continue
            else:
                init = s.data.get("init") or {}
            if fact in init and (vh is None or vhash(init[fact]) in vh or vhash(untag_floats(init[fact])) in vh):
                stored.append(s.id)
        return {"fact": fact, "value": None if value is ANY else plain(value), "dependent": dependent, "stored": stored,
                "deleted": 0}


def _dependents(resp, fact, value_hash=None):
    """question → {"answer", "status", "path"} for the answers of a stored response (dict) that rest on `fact`."""
    tr = resp.get("trace") or {}
    by = {}
    for r in tr.get("records") or ():
        by.setdefault(r["name"], []).append(r)
    out = {}
    for q, res in (resp.get("results") or {}).items():
        roots = ["answer:" + q]
        if res.get("guard") == "hard_check" and res.get("source") in by:
            roots.append(res["source"])
        path = _path(by, roots, fact, value_hash)
        if path is not None:
            out[q] = {"answer": "<not stated>" if res.get("not_stated") else res.get("answer"), "status": res.get("status"),
                      "path": path}
    return out


def _path(by, roots, fact, value_hash):
    """Breadth-first from the answer back through the recorded inputs → [fact, ..., root] or None."""
    parent = {r: None for r in roots}
    todo = list(roots)
    while todo:
        n = todo.pop(0)
        for rec in by.get(n, ()):
            for x, h in (rec.get("inputs") or {}).items():
                if x == fact and (value_hash is None or h == value_hash):
                    path, cur = [fact], n
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    return path
                if x not in parent:
                    parent[x] = n
                    todo.append(x)
    return None


# --- JSONL
class JSONLStorage(TraceStorage):
    """Append-only JSON lines, one record per line (the file System(storage="file.jsonl") writes). The head (count and last hash)
    is kept in `<path>.head`. Threads of a process may write; several processes may too where the system has advisory
    file locks (POSIX: an append takes an exclusive flock on the file, reads what other processes appended since it
    last looked, then writes its record and the head) — on Windows keep to one writing process, or use SQLiteStorage.
    Lines of a 0.5 journal at the start of the file are kept and skipped (verify reports them as `legacy`).
    fsync=True: flush every record to disk before save returns (slower; without it a power failure can lose the last
    records, which the OS had not yet written). index=False: open from the stored head (checked
    against the file's last line) without reading every record — a long file opens at once, for a process that only
    appends; get(id) then scans the file. A head that does not match the last line is not trusted: the file is read."""

    def __init__(self, path, catalog=None, clock=None, fsync=False, index=True):
        super().__init__(catalog, clock)
        self.path = os.fspath(path)
        self.head_path = self.path + ".head"
        self.fsync = fsync
        self._lock = threading.Lock()
        self._reset()
        with self._lock:
            if index or not self._open_from_head():
                self._refresh()

    def _open_from_head(self):
        """Take the count and the last hash from the stored head when the file's last line is the record it names."""
        try:
            with open(self.path, "rb") as fh, _flock(fh, shared=True):
                head, last = self._read_head(), json.loads(_last_line(fh))
                if last.get("hash") != head["hash"] or last.get("seq") != head["count"] - 1:
                    return False
                self._count, self._last, self._size = head["count"], head["hash"], os.fstat(fh.fileno()).st_size
                self._n_legacy = None                 # not counted yet (see _legacy)
                return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    def _reset(self):
        self._offsets = {}
        self._count, self._last, self._n_legacy, self._size = 0, GENESIS, 0, 0

    def _refresh(self):
        """Catch up with the file (another process, or another store object on the same path, may have appended): under
        a shared lock, so no line is read half-written."""
        if os.path.exists(self.path):
            with open(self.path, "rb") as fh, _flock(fh, shared=True):
                self._sync(fh)
        elif self._size:
            self._reset()

    def _sync(self, fh):
        """Read the lines appended since this store last looked (all of them the first time; again from the start when
        the file got shorter: it was replaced). Called under the thread lock and the file's lock."""
        size = os.fstat(fh.fileno()).st_size
        if size == self._size:
            return
        if size < self._size:
            self._reset()
        fh.seek(self._size)
        off = self._size
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                d = None
            if chained(d):
                self._offsets[d.get("id")] = off
                self._count, self._last = self._count + 1, d["hash"]
            elif isinstance(d, dict) and self._count == 0 and self._n_legacy is not None:
                self._n_legacy += 1
            off += len(line)
        self._size = off

    def _append(self, body):
        with self._lock, open(self.path, "a+b") as fh, _flock(fh):
            self._sync(fh)                            # what other writers appended: the chain goes on from their last
            rec = dict(body, seq=self._count, time=float(self.clock()), prev=self._last)
            rec["hash"] = record_hash(rec)
            rec["id"] = rec["hash"][:16]
            line = (_body(rec) + "\n").encode()
            off = fh.seek(0, os.SEEK_END)
            if off:                                   # a crash cut the last line short: end it, never glue onto it
                fh.seek(off - 1)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")                    # the fragment stays a line of its own: verify reports it
                    off += 1
            fh.write(line)
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
            self._offsets[rec["id"]] = off
            self._count, self._last, self._size = self._count + 1, rec["hash"], off + len(line)
            tmp = self.head_path + ".tmp"             # still under the file's lock: one writer of the head at a time
            with open(tmp, "w") as hf:
                json.dump({"count": self._count, "hash": self._last}, hf)
            os.replace(tmp, self.head_path)
            return rec

    def head(self):
        with self._lock:
            self._refresh()
            return {"count": self._count, "hash": self._last}

    def _rewrite(self, rec):
        """In place, under the append lock: other writers hold this very file open, so it is not swapped for a new one.
        The new content is first written whole to `<path>.rewrite` (a crash while copying it back leaves that file)."""
        with self._lock, open(self.path, "r+b") as fh, _flock(fh):
            tmp = self.path + ".rewrite"
            with open(tmp, "wb") as out:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        d = None
                    if isinstance(d, dict) and d.get("id") == rec["id"] and d.get("hash") == rec["hash"]:
                        line = (_body(rec) + "\n").encode()
                    out.write(line)
                out.flush()
                os.fsync(out.fileno())
            with open(tmp, "rb") as src:
                fh.seek(0)
                fh.truncate()
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
            os.remove(tmp)
            self._reset()
            self._sync(fh)

    def _snapshot(self):
        """The file's size and the stored head at one moment (under the locks an append holds while it writes both)."""
        with self._lock:
            if not os.path.exists(self.path):
                return {"size": 0, "head": self._read_head()}
            with open(self.path, "rb") as fh, _flock(fh, shared=True):
                return {"size": os.fstat(fh.fileno()).st_size, "head": self._read_head()}

    def _read_head(self):
        """The stored head → {"count", "hash"}, or "missing" / "unreadable"."""
        if not os.path.exists(self.head_path):
            return "missing"
        try:
            with open(self.head_path) as fh:
                h = json.load(fh)
        except (OSError, ValueError):
            return "unreadable"
        return h if isinstance(h, dict) else "unreadable"

    def _lines(self, limit=None):
        """(line number, record or None when unreadable) for every line — with `limit`, for the lines that start before
        that many bytes (the file as it was at a snapshot)."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as fh:
            off = 0
            for i, line in enumerate(fh):
                if limit is not None and off >= limit:
                    return
                off += len(line)
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    d = None
                yield i, d if isinstance(d, dict) else None

    def _raw(self, snap=None):
        started = False
        for i, d in self._lines(None if snap is None else snap["size"]):
            if d is not None and not chained(d) and not started:
                continue                                  # a 0.5 journal line before the chain
            started = True                                # an unchained line inside the chain: verify reports it
            yield i, d

    def _legacy(self):
        if self._n_legacy is None:                    # opened from the head: the 0.5 lines before the chain, counted now
            n = 0
            for _, d in self._lines():
                if chained(d):
                    break
                n += d is not None
            self._n_legacy = n
        return self._n_legacy

    def _find(self, id):
        off = self._offsets.get(id)
        if off is not None and os.path.exists(self.path):
            with open(self.path, "rb") as fh:
                fh.seek(off)
                try:
                    d = json.loads(fh.readline())
                    if isinstance(d, dict) and d.get("id") == id:
                        return d
                except ValueError:
                    pass
        for _, d in self._raw():                          # the file changed under us: scan
            if d is not None and d.get("id") == id:
                return d
        return None

    def _backend_problems(self, rows, snap=None):
        h = self._read_head() if snap is None else snap["head"]
        if h == "missing":
            return [(len(rows) - 1, None, "the head file is missing")] if rows else []
        if h == "unreadable":
            return [(None, None, "the head file is not readable")]
        return _head_problems(h, rows)


def _last_line(fh):
    """The last line of an open binary file (without its line end), reading from the end."""
    end = fh.seek(0, os.SEEK_END)
    if not end:
        return b""
    size = min(end, 1 << 16)
    while True:
        fh.seek(end - size)
        body = fh.read(size).rstrip(b"\n")
        i = body.rfind(b"\n")
        if i >= 0 or size == end:
            return body[i + 1:]
        size = min(end, size * 4)


def _head_problems(h, rows, later=None):
    """The stored head against the records. `later`: the head read again after the records (a backend whose head was read
    before them, with writers going on): records beyond the first head are appends made meanwhile when the first head's
    record is in its place and the later head counts them."""
    n = len(rows)
    last = rows[-1].get("hash") if rows else GENESIS
    if h.get("count") == n and h.get("hash") == last:
        return []
    if isinstance(h.get("count"), int) and h["count"] > n:
        return [(n, None, f"the stored head says {h['count']} records, the log has {n}: records were removed from the end")]
    if isinstance(h.get("count"), int) and h["count"] < n:
        k = h["count"]
        if (later is not None and isinstance(later.get("count"), int) and later["count"] >= n
                and (rows[k - 1].get("hash") if k else GENESIS) == h.get("hash")):
            return []                                 # appended while verify was reading: the head has them by now
        return [(h["count"], None, f"the log has {n} records, the stored head says {h['count']}: records were appended "
                                   "outside the store (or it stopped between writing a record and its head)")]
    return [(n - 1, rows[-1].get("id") if rows else None, "the last record is not the one the stored head names: the end "
                                                         "of the log was replaced")]


# --- SQL backends: SQLite, PostgreSQL, DuckDB
_SCHEMA = """
CREATE TABLE IF NOT EXISTS {p}records (seq {INT} PRIMARY KEY, "id" {TEXT} NOT NULL UNIQUE, kind {TEXT} NOT NULL,
                                       "time" {REAL} NOT NULL, init_hash {TEXT}, "catalog" {TEXT}, prev {TEXT} NOT NULL,
                                       hash {TEXT} NOT NULL, body {TEXT} NOT NULL);
CREATE INDEX IF NOT EXISTS {p}records_time ON {p}records("time");
CREATE INDEX IF NOT EXISTS {p}records_catalog ON {p}records("catalog");
CREATE INDEX IF NOT EXISTS {p}records_init ON {p}records(init_hash);
CREATE TABLE IF NOT EXISTS {p}answers (seq {INT} NOT NULL, question {TEXT} NOT NULL, answer {TEXT} NOT NULL,
                                       status {TEXT} NOT NULL, guard {TEXT});
CREATE INDEX IF NOT EXISTS {p}answers_question ON {p}answers(question, answer);
CREATE INDEX IF NOT EXISTS {p}answers_answer ON {p}answers(answer);
CREATE INDEX IF NOT EXISTS {p}answers_seq ON {p}answers(seq);
CREATE TABLE IF NOT EXISTS {p}safeguards (seq {INT} NOT NULL, kind {TEXT} NOT NULL, fact {TEXT} NOT NULL, question {TEXT});
CREATE INDEX IF NOT EXISTS {p}safeguards_kind ON {p}safeguards(kind, question);
CREATE INDEX IF NOT EXISTS {p}safeguards_seq ON {p}safeguards(seq);
CREATE TABLE IF NOT EXISTS {p}models (seq {INT} NOT NULL, fp {TEXT}, "id" {TEXT}, "type" {TEXT});
CREATE INDEX IF NOT EXISTS {p}models_fp ON {p}models(fp);
CREATE INDEX IF NOT EXISTS {p}models_id ON {p}models("id");
CREATE INDEX IF NOT EXISTS {p}models_seq ON {p}models(seq);
CREATE TABLE IF NOT EXISTS {p}meta ("key" {TEXT} PRIMARY KEY, "value" {TEXT} NOT NULL)
"""


def _index_rows(d):
    """The index rows of a record: (answers, safeguards, models) as tuples without seq."""
    ans = [(q, _cj(a[0]), a[2], (d.get("guards") or {}).get(q)) for q, a in sorted((d.get("answers") or {}).items())]
    sg = []
    for kind, fact, qs in d.get("safeguards") or ():
        for q in qs or [None]:
            sg.append((kind, fact, q))
    ms = [(m.get("fp"), m.get("id"), m.get("type")) for m in d.get("models") or ()]
    return ans, sg, ms


class _SQLStorage(TraceStorage):
    """What the SQL backends share: one row per record with the record's JSON (as written: keys in their own order),
    plus index tables — answers (question, answer, status), safeguards (kind, question), models (fingerprint, id, type) —
    and the time; the head in the `meta` table. Every append reads the head and writes the record, its index rows and the
    new head in one transaction (a backend's `lock` statement keeps two writers from taking the same head). A backend
    sets the connection (`db`: a DB-API connection in autocommit mode, whose execute returns something with fetchall),
    its placeholder, its column types and the statement that begins a write transaction. Table names start with
    `prefix`."""

    placeholder = "?"
    begin = "BEGIN"
    lock = None                                   # a statement run first in a write transaction (PostgreSQL: LOCK TABLE)
    types = {"INT": "BIGINT", "REAL": "DOUBLE PRECISION", "TEXT": "TEXT"}

    def _open(self, db, prefix=""):
        if not isinstance(prefix, str) or not all(c.isalnum() or c == "_" for c in prefix):
            raise ValueError(f"prefix must be letters, digits and _, not {prefix!r}")
        self.db, self.prefix = db, prefix
        self._lock = threading.Lock()
        with self._lock:
            for stmt in _SCHEMA.format(p=prefix, **self.types).split(";"):
                if stmt.strip():
                    self._x(stmt)

    def close(self):
        self.db.close()

    def _sql(self, sql):
        sql = sql.replace("{p}", self.prefix)
        return sql if self.placeholder == "?" else sql.replace("?", self.placeholder)

    def _x(self, sql, args=()):
        """Run one statement (SQL written with ? placeholders and {p} for the table prefix)."""
        return self.db.execute(self._sql(sql), tuple(args))

    def _head(self):
        row = self._x("SELECT \"value\" FROM {p}meta WHERE \"key\" = 'head'").fetchone()
        return json.loads(row[0]) if row else {"count": 0, "hash": GENESIS}

    def head(self):
        with self._lock:
            return self._head()

    def _append(self, body):
        with self._lock:
            self._x(self.begin)
            try:
                if self.lock:
                    self._x(self.lock)
                h = self._head()
                rec = dict(body, seq=h["count"], time=float(self.clock()), prev=h["hash"])
                rec["hash"] = record_hash(rec)
                rec["id"] = rec["hash"][:16]
                self._insert(rec)
                self._x("INSERT INTO {p}meta (\"key\", \"value\") VALUES ('head', ?) "
                        "ON CONFLICT (\"key\") DO UPDATE SET \"value\" = excluded.\"value\"",
                        (json.dumps({"count": rec["seq"] + 1, "hash": rec["hash"]}),))
                self._x("COMMIT")
            except BaseException:
                self._x("ROLLBACK")
                raise
            return rec

    def _insert(self, rec):
        s = rec["seq"]
        self._x("INSERT INTO {p}records (seq, \"id\", kind, \"time\", init_hash, \"catalog\", prev, hash, body) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (s, rec["id"], rec.get("kind", "ask"), rec["time"], rec.get("init_hash"), rec.get("catalog"),
                 rec["prev"], rec["hash"], _body(rec)))
        ans, sg, ms = _index_rows(rec)
        for a in ans:
            self._x("INSERT INTO {p}answers (seq, question, answer, status, guard) VALUES (?, ?, ?, ?, ?)", (s, *a))
        for x in sg:
            self._x("INSERT INTO {p}safeguards (seq, kind, fact, question) VALUES (?, ?, ?, ?)", (s, *x))
        for m in ms:
            self._x("INSERT INTO {p}models (seq, fp, \"id\", \"type\") VALUES (?, ?, ?, ?)", (s, *m))

    def _snapshot(self):
        return {"head": self.head()}                  # before the records: what is appended meanwhile is beyond it

    def _rewrite(self, rec):
        with self._lock:
            self._x(self.begin)
            try:
                if self.lock:
                    self._x(self.lock)
                s = rec["seq"]
                for t in ("answers", "safeguards", "models"):
                    self._x(f"DELETE FROM {{p}}{t} WHERE seq = ?", (s,))        # noqa: S608 — a fixed table name
                self._x("DELETE FROM {p}records WHERE seq = ? AND \"id\" = ?", (s, rec["id"]))
                self._insert(rec)
                self._x("COMMIT")
            except BaseException:
                self._x("ROLLBACK")
                raise

    def _rows(self, sql="SELECT seq, body FROM {p}records ORDER BY seq", args=()):
        with self._lock:
            rows = self._x(sql, args).fetchall()
        for seq, body in rows:
            try:
                d = json.loads(body)
            except ValueError:
                d = None
            yield seq, d if isinstance(d, dict) else None

    def _raw(self, snap=None):
        return self._rows()

    def _find(self, id):
        for _, d in self._rows("SELECT seq, body FROM {p}records WHERE \"id\" = ?", (id,)):
            return d
        return None

    def iter(self, kind="ask", redacted=False):
        if kind is None:
            rows = self._rows()
        else:
            rows = self._rows("SELECT seq, body FROM {p}records WHERE kind = ? ORDER BY seq", (kind,))
        for _, d in rows:
            if chained(d) and (redacted or not d.get("redacted")):
                yield _stored(d, self.catalog)

    def query(self, question=None, answer=ANY, status=None, safeguard=None, model=None, since=None, until=None,
              catalog=None):
        since, until = _when(since), _when(until)
        answer = self._answer_key(question, answer)
        where, args = ["r.kind = 'ask'"], []
        if catalog is not None:
            where.append("r.\"catalog\" = ?")
            args.append(catalog)
        if question is not None or answer is not ANY or status is not None:
            sub = ["a.seq = r.seq"]
            if question is not None:
                sub.append("a.question = ?")
                args.append(question)
            if answer is not ANY:
                sub.append("a.answer = ?")
                args.append(akey(answer))
            if status is not None:
                sub.append("a.status = ?")
                args.append(status)
            where.append(f"EXISTS (SELECT 1 FROM {{p}}answers a WHERE {' AND '.join(sub)})")  # noqa: S608 — fixed clauses, values bound
        if safeguard is not None:
            sub = ["s.seq = r.seq", "s.kind = ?"]
            args.append(safeguard)
            if question is not None:
                sub.append("s.question = ?")
                args.append(question)
            where.append(f"EXISTS (SELECT 1 FROM {{p}}safeguards s WHERE {' AND '.join(sub)})")  # noqa: S608
        if model is not None:
            where.append("EXISTS (SELECT 1 FROM {p}models m WHERE m.seq = r.seq AND (m.fp = ? OR m.\"id\" = ? OR "
                         "m.\"type\" = ?))")
            args += [model, model, model]
        if since is not None:
            where.append("r.\"time\" >= ?")
            args.append(since)
        if until is not None:
            where.append("r.\"time\" < ?")
            args.append(until)
        sql = f"SELECT r.seq, r.body FROM {{p}}records r WHERE {' AND '.join(where)} ORDER BY r.seq"  # noqa: S608
        return [_stored(d, self.catalog) for _, d in self._rows(sql, args) if chained(d) and not d.get("redacted")]

    def _backend_problems(self, rows, snap=None):
        out = []
        with self._lock:
            cols = {s: (i, k, t, ih, c, p, h) for s, i, k, t, ih, c, p, h in self._x(
                "SELECT seq, \"id\", kind, \"time\", init_hash, \"catalog\", prev, hash FROM {p}records").fetchall()}
            idx = {}
            for table, q in (("answers", "SELECT seq, question, answer, status, guard FROM {p}answers"),
                             ("safeguards", "SELECT seq, kind, fact, question FROM {p}safeguards"),
                             ("models", "SELECT seq, fp, \"id\", \"type\" FROM {p}models")):
                for row in self._x(q).fetchall():
                    idx.setdefault((table, row[0]), []).append(tuple(row[1:]))
        for d in rows:
            s = d.get("seq")
            c = cols.get(s)
            if c is None:
                continue
            if c != (d.get("id"), d.get("kind", "ask"), d.get("time"), d.get("init_hash"), d.get("catalog"), d.get("prev"),
                     d.get("hash")):
                out.append((s, d.get("id"), "the record's columns differ from its stored JSON (edited in the table)"))
            for table, want in zip(("answers", "safeguards", "models"), _index_rows(d)):
                if sorted(idx.get((table, s), []), key=repr) != sorted(want, key=repr):
                    out.append((s, d.get("id"), f"the {table} index differs from the stored record (queries would lie)"))
        with self._lock:
            now = self._head()                        # after everything was read
        seqs = {d.get("seq") for d in rows}
        n = len(rows)                                 # index rows of records appended after `rows` were read (the head
        meanwhile = range(n, now["count"]) if isinstance(now.get("count"), int) else ()   # has them by now) are not orphans
        orphans = sorted({s for (_, s) in idx if s not in seqs and s not in meanwhile}, key=repr)
        if orphans:
            out.append((orphans[0], None, f"index rows without a record (seq {', '.join(map(str, orphans[:5]))})"))
        if snap is None:
            return out + _head_problems(now, rows)
        return out + _head_problems(snap["head"], rows, later=now)


class SQLiteStorage(_SQLStorage):
    """SQLite (stdlib sqlite3): one row per record with the record's JSON, plus index tables — answers (question, answer,
    status), safeguards (kind, question), models (fingerprint, id, type) — and the time. The head is kept in the `meta`
    table. Appends run in a write transaction, so several processes may write to one file. Each record is one committed
    transaction: durable once save returns (unlike JSONLStorage without fsync=True), at the cost of a disk sync."""

    begin = "BEGIN IMMEDIATE"
    types = {"INT": "INTEGER", "REAL": "REAL", "TEXT": "TEXT"}

    def __init__(self, path, catalog=None, clock=None, timeout=30.0):
        import sqlite3
        super().__init__(catalog, clock)
        self.path = os.fspath(path)
        self._open(sqlite3.connect(self.path, timeout=timeout, isolation_level=None, check_same_thread=False))


class PostgresStorage(_SQLStorage):
    """PostgreSQL (psycopg 3, `pip install solvi[postgres]`): the tables of SQLiteStorage, named with `prefix`
    ("solvi_records", ...), created when missing. Several processes and services may write: an append locks the head
    table (LOCK TABLE ... IN SHARE ROW EXCLUSIVE MODE — readers are not blocked) for its transaction, so the chain has
    no forks. `conninfo`: a connection string ("postgresql://user@host/db") or an open psycopg connection in autocommit
    mode (the store runs its own BEGIN / COMMIT)."""

    placeholder = "%s"
    lock = "LOCK TABLE {p}meta IN SHARE ROW EXCLUSIVE MODE"

    def __init__(self, conninfo, catalog=None, clock=None, prefix="solvi_"):
        super().__init__(catalog, clock)
        if isinstance(conninfo, str):
            try:
                import psycopg
            except ImportError as e:
                raise ImportError("PostgresStorage needs psycopg 3: pip install 'solvi[postgres]'") from e
            db = psycopg.connect(conninfo, autocommit=True)
        else:
            db = conninfo
        self.conninfo = conninfo if isinstance(conninfo, str) else None
        self._open(db, prefix)


class DuckDBStorage(_SQLStorage):
    """DuckDB (`pip install solvi[duckdb]`): the tables of SQLiteStorage in a DuckDB file (":memory:" for none) — for
    analytics over the stored decisions next to JSONL or Parquet files (`store.db.sql(...)`). One writing process at a
    time (DuckDB's own rule); threads of that process are fine."""

    begin = "BEGIN TRANSACTION"
    types = {"INT": "BIGINT", "REAL": "DOUBLE", "TEXT": "VARCHAR"}

    def __init__(self, path, catalog=None, clock=None, prefix=""):
        try:
            import duckdb
        except ImportError as e:
            raise ImportError("DuckDBStorage needs duckdb: pip install 'solvi[duckdb]'") from e
        super().__init__(catalog, clock)
        self.path = os.fspath(path)
        self._open(duckdb.connect(self.path), prefix)


def open_storage(where, catalog=None):
    """A TraceStorage from a path: .db / .sqlite / .sqlite3 → SQLiteStorage, .duckdb → DuckDBStorage (in any case), a
    postgresql:// (or postgres://) URL → PostgresStorage, anything else → JSONLStorage; a TraceStorage is returned as
    it is."""
    if where is None or isinstance(where, TraceStorage):
        return where
    p = os.fspath(where)
    if p.startswith(("postgresql://", "postgres://")):
        return PostgresStorage(p, catalog)
    ext = p.lower()                                   # "decisions.DB" is a database too, not a JSON-lines file
    if ext.endswith(".duckdb"):
        return DuckDBStorage(p, catalog)
    if ext.endswith((".db", ".sqlite", ".sqlite3")):
        return SQLiteStorage(p, catalog)
    return JSONLStorage(p, catalog)
