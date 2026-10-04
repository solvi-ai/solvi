"""TraceStorage: stored responses with their traces, a hash chain across them, queries, a replay of everything stored, and
provenance questions over the store (which stored decisions rest on a fact found to be wrong; what a forgotten given fact
touches).

Every stored decision is one record (a dict of plain JSON):

  kind "ask"      the 0.5 journal line's keys — init_hash, answers {question: [answer, confidence, status]}, flow, records
                  [[step, name, hash]] (and producers) — plus index fields (guards, safeguards, models) and the whole
                  response (Response.to_dict(): answers, flow, trace), which get() loads back and replay_all() re-checks;
  kind "teach"    a correction (System.teach): {"teach": question, "init": ..., "answer": ...} and, when given, its
                  "source" ("outcome", "rule", "verified"; none: a human), "by" and "of" (the stored id of the decision it
                  corrects; for "verified", the System 2 decision the label is);
  kind "update"   a learning update (solvi.experimental.learning.Learning): what changed, the gates' results, the state to roll back to;
  every record    seq (0, 1, 2, ...), time (seconds since the epoch), meta (optional, yours; a decision made with
                  solvi.experimental pieces has meta["experimental"]: their names), prev and hash.

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

import abc
import contextlib
import hashlib
import json
import os
import threading
import time as _time
from dataclasses import dataclass

from ... import _deprecate
from ..sources import TRUSTED_SOURCES, VERIFIED, VERIFIED_REFUSED, UntrustedLabel, check_source   # noqa: F401 — re-exported

try:
    import fcntl
except ImportError:                           # Windows, the browser (Pyodide): no advisory file locks
    fcntl = None

GENESIS = ""                                  # prev of the first record
_SAMPLE_LOCK = threading.Lock()
FORMAT = 2                                    # the record format ("v"): 2 — dicts are written with their keys in their
                                              # own order (1, solvi ≤ 0.7.1: sorted, the order a decider read is lost)
LEGACY_ORDER = ("stored by solvi 0.7.1 or earlier, which wrote dict keys sorted: a model that read a dict may have read "
                "its keys in another order than this replay gives it — replay such records with trust_models=True")


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
    (solvi.core.deciders.state_text) — so a stored trace must give back the dicts as they were for its model steps to replay."""
    return json.dumps(rec, ensure_ascii=False, allow_nan=False)


KEPT = ("v", "kind", "seq", "time", "prev", "init_hash", "catalog", "records", "flow", "producers", "models", "guards",
        "safeguards", "teach", "source", "of", "of_hash", "by", "note", "segment")
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
    from ..catalog import NOT_STATED_KEY, Unknown
    from ..schema import jsonable
    if v is Unknown:
        return NOT_STATED_KEY
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return jsonable(v)


def akey(v):
    """An answer as an index key (canonical JSON of its stored form): answer=("a", "b") finds a stored ["a", "b"]."""
    from ..schema import tag_floats
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
RECORD_MODES = ("full", "compact", "sample:N")


def record_mode(record):
    """A store's `record=` → ("full", None), ("compact", None) or ("sample", N); ValueError for anything else."""
    if record in ("full", "compact"):
        return record, None
    if isinstance(record, str) and record.startswith("sample:"):
        try:
            n = int(record[7:])
        except ValueError:
            n = 0
        if n >= 1:
            return "sample", n
    raise ValueError(f'record must be "full", "compact" or "sample:N" (N ≥ 1: one decision in N kept in full, the '
                     f"others compact), not {record!r}")


KEPT_STEPS = ("head", "guard", "plan", "textin")
"""The trace records a compact record keeps whole besides every model-backed step: answer heads, guarantees, the
strategist's plan and the text an ask_text read (they are small, and a re-run of the flow does not produce them)."""


def _kept_step(r):
    return r.get("kind") in KEPT_STEPS or r.get("model") is not None or bool(r.get("tried_models"))


_RESULT_DEFAULTS = {"status": "ok", "probs": {}, "provenance": None, "source": None, "guard": None, "repaired": None,
                    "kind": None, "not_stated": False, "evidence": [], "extra": None}


def compact_content(d):
    """A response's dict (Response.to_dict()) → what a compact record keeps of it (see TraceStorage, record="compact"):
    the given facts (the input as read, with the types that restore it), each answer with its why / guard / source /
    evidence / guarantee (not its probabilities), each check's result and reasons, the model-backed steps and the
    answer heads, guarantees, plan and text records whole (with the models' ids, fingerprints and token usage), the
    steps skipped, the catalog's and questions' fingerprints, and the time. Left out: the values of the computed facts
    (they are re-computed), the other steps' records (their hashes stay in the record's `records`), the planned flow,
    the timings and the per-part fingerprints."""
    tr = d.get("trace") or {}
    recs = tr.get("records") or []
    trace = {k: v for k, v in tr.items() if k not in ("records", "timings", "schedule", "fingerprint")}
    fp = tr.get("fingerprint") or {}
    trace["fingerprint"] = {k: v for k, v in fp.items() if k != "parts"}
    trace["records"] = [r for r in recs if _kept_step(r)]
    results = {}
    for q, r in (d.get("results") or {}).items():
        results[q] = {k: v for k, v in r.items() if k != "probs" and not (k in _RESULT_DEFAULTS and v == _RESULT_DEFAULTS[k])}
    checks = {}
    for r in recs:
        if r.get("kind") == "check":
            checks[r["name"]] = [None if r.get("missing") else r.get("value"), (r.get("extra") or {}).get("reasons")]
    c = {"trace": trace, "results": results, "checks": checks, "ms": d.get("ms"), "safeguards": d.get("safeguards") or []}
    if d.get("feasible") is False:
        c["feasible"], c["violations"] = False, d.get("violations")
    if d.get("model_outputs"):
        c["model_outputs"] = d["model_outputs"]
    return c


def entry(resp, meta=None, record="full"):
    """The stored content of a response (without seq, time, prev, hash). record: "full" (the whole response) or
    "compact" (compact_content: the input, answers, checks and model steps — see TraceStorage)."""
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
    exp = list(getattr(resp, "experimental", None) or ())
    if exp and (meta is None or isinstance(meta, dict)):       # made with solvi.experimental pieces: said in the record
        meta = {**(meta or {}), "experimental": exp}
    if meta is not None:
        e["meta"] = plain(meta)
    from ..schema import tag_floats
    e = tag_floats(e)                                 # strict JSON: an inf threshold is {"$float": "inf"}
    if record == "compact":
        e["record"] = "compact"
        e["compact"] = compact_content(d)             # to_dict() has tagged it already
    else:
        e["response"] = d
    return e


class CompactRecord(ValueError):
    """A stored decision kept compact (record="compact"): its trace is not in the store, so it cannot be loaded as a
    Response. TraceStorage.rederive(id, system) re-runs it from the recorded input and checks every step against the
    recorded hashes; replay_all does so too."""


def view(d):
    """A stored decision's response as a dict, for readers that do not need the whole trace (reports): the stored
    response, or for a compact record what it kept, shaped the same way ("results", "trace" with "init" and the kept
    "records", "ms", "safeguards"). None for a record without either (a redacted one)."""
    if isinstance(d.get("response"), dict):
        return d["response"]
    c = d.get("compact")
    if not isinstance(c, dict):
        return None
    res = {q: {**_RESULT_DEFAULTS, **r} for q, r in (c.get("results") or {}).items()}
    return {"results": res, "trace": c.get("trace") or {}, "ms": c.get("ms"), "safeguards": c.get("safeguards") or [],
            "feasible": c.get("feasible", True), "violations": c.get("violations") or [], "values": {},
            "model_outputs": c.get("model_outputs", 0)}


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
        from ..schema import untag_floats
        return {q: untag_floats(a[0]) for q, a in (self.data.get("answers") or {}).items()}

    @property
    def meta(self):
        from ..schema import untag_floats
        return untag_floats(self.data.get("meta"))

    @property
    def compact(self):
        """Was this decision stored compact (record="compact", or not drawn for a full record under "sample:N")?"""
        return self.data.get("record") == "compact"

    @property
    def input(self):
        """The given facts of a stored decision as JSON data (full or compact; None for a redacted one)."""
        from ..schema import untag_floats
        v = view(self.data)
        return None if v is None else untag_floats((v.get("trace") or {}).get("init"))

    def response(self, catalog=None):
        """The stored response, loaded back (typed values restored from `catalog` — a Catalog or a System — or from the
        store's own). A compact record raises CompactRecord: re-derive it with TraceStorage.rederive(id, system)."""
        from ..response import Response
        if self.kind != "ask":
            raise ValueError(f"record {self.id} is a {self.kind} record, not a response")
        if self.data.get("redacted"):
            raise ValueError(f"record {self.id} was redacted: its response is gone")
        if self.compact:
            raise CompactRecord(f"record {self.id} was stored compact: its input, answers, checks and model steps are "
                                "kept, its other steps only as hashes — store.rederive(id, system) re-runs it from the "
                                "recorded input and checks every step against them")
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
    if d.get("record") == "compact":
        return _compact_problems(d)
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


def _compact_problems(d):
    """A compact record agrees with its own summary: the input hash, each kept step's place, hash and link in the
    recorded chain of step hashes, and the answers (no catalog needed; replay_all re-runs the steps)."""
    c = d.get("compact")
    if not isinstance(c, dict):
        return ["a compact record without its compact content"]
    out = []
    tr = c.get("trace") or {}
    if tr.get("init_hash") != d.get("init_hash"):
        out.append("compact record holds another init_hash than its record")
    steps = d.get("records") or []
    where = {(s[0], s[1], s[2]): i for i, s in enumerate(steps) if isinstance(s, list) and len(s) == 3}
    for r in tr.get("records") or []:
        i = where.get((r.get("step"), r.get("name"), r.get("hash")))
        if i is None:
            out.append(f"kept step {r.get('step')} ({r.get('name')}) is not in the record's step hashes")
            continue
        if r.get("prev") != (steps[i - 1][2] if i else d.get("init_hash")):
            out.append(f"kept step {r.get('step')} ({r.get('name')}) is not linked to the step before it")
    res = c.get("results") or {}
    got = {q: [r.get("status", "ok"), _cj(None if r.get("not_stated") else r.get("answer"))] for q, r in res.items()}
    want = {q: [a[2], _cj(None if a[0] == "<not stated>" else a[0])] for q, a in (d.get("answers") or {}).items()}
    if got != want:
        out.append("compact answers differ from the record's answers")
    return out


def _not_rerun(flow, recorded, trust_models):
    """The model-backed steps of a flow that ran in the recorded decision and whose model is not re-run on replay
    (trust_models=True, a model marked deterministic=False such as a generator, or one not available) → {name: why}."""
    out = {}
    for st in flow.steps:
        p = st.part
        if p.name not in recorded:
            continue
        for a in (p.alternatives if p.alternatives is not None else [p]):
            m = a.model
            if m is None:
                continue
            if getattr(m, "available", True) is False:
                out[p.name] = f"its model {a.name} is not available"
            elif trust_models:
                out[p.name] = "trust_models=True: its model is not re-run"
            elif getattr(m, "deterministic", True) is False:
                out[p.name] = f"its model ({type(m).__name__}) is not re-run on replay"
            else:
                continue
            break
    return out


def _recorded_output(k):
    """A kept model step's record → an output that gives the recorded value again (its quote, confidence,
    probabilities, evidence and details), for a re-run that must not call the model."""
    from ..catalog import Claim, Decision, Quote
    from ..runtime import MISSING
    if k.value is MISSING:
        raise RuntimeError(k.error or "no value recorded")
    ex = dict(k.extra or {})
    ev = [Quote(t, a, b, src) for a, b, src, t in ex.pop("evidence", None) or ()]
    val = Quote(k.value, k.quote[0], k.quote[1], k.quote[2], k.confidence) if k.quote else k.value
    if k.probs is not None:
        return Decision(val, dict(k.probs), confidence=k.confidence, extra=ex, evidence=ev)
    return Claim(val, evidence=ev, confidence=k.confidence, extra=ex)


def _same_output(r, k):
    """Did a re-run step give what the kept record says (value, quote, error, inputs, producer)? Its details (a model's
    latency, usage) may differ."""
    from ..runtime import vhash
    return (r.name == k.name and r.step == k.step and vhash(r.value) == vhash(k.value) and r.error == k.error
            and (None if r.quote is None else list(r.quote)) == (None if k.quote is None else list(k.quote))
            and r.inputs == k.inputs and r.producer == k.producer)


def rederive(d, system, trust_models=False):
    """A compact record (dict) → (Response or None, [Mismatch]): its decision rebuilt with `system` from the recorded
    input. The flow is planned again for the recorded questions and run; a model-backed step whose model is not re-run
    on replay (trust_models=True, a generator, a model not available) gives its kept output instead of calling the
    model (a compact record keeps every model step whole), and a model step that is re-run and gives the kept output
    takes the kept record (its latency and usage); then every step's hash is chained again and compared with the
    recorded one — a hash covers the step's inputs, value, error and the link to the step before, so equal hashes mean
    the step gave what it gave then — and the kept records after the flow (answer heads, guarantees, plan, text) are
    appended. The Response is the decision as it was: replay it like a stored one. None, with the reasons, when it
    cannot be rebuilt: a step whose hash differs (kind "recompute": the catalog changed since — the old value is not
    kept, only its hash), a kept model output that does not give its record again or a model step of several producers
    that is not re-run (kind "not_kept": no verdict past it), the input not restored."""
    import dataclasses
    from ..response import Response
    from ..runtime import MISSING, Mismatch, Trace, execute, vhash
    from ..schema import load
    if not (hasattr(system, "catalog") and hasattr(system, "questions") and hasattr(system, "_prepare")):
        return None, [Mismatch(0, "load", "a compact record is re-run from its input: replay it with the System, not a "
                                          "Catalog", "error")]
    c = d["compact"]
    kept = load(Trace, c["trace"], system)
    steps = [tuple(s) for s in d.get("records") or []]
    lost = sorted(q for q in d.get("answers") or {} if q not in system.questions)
    if lost:
        return None, [Mismatch(0, f"answer:{q}", "the question is not in the system (renamed or removed)", "missing_part")
                      for q in lost]
    p = system._prepare(dict(kept.init), list(d.get("answers") or {}), None)
    at = {s[1]: s for s in steps}
    have = {(r.step, r.name, r.hash): r for r in kept.records}
    by_name = {r.name: r for r in kept.records}
    skip = _not_rerun(p.flow, set(at), trust_models)
    flow = p.flow
    if skip:
        new_steps = []
        for st in flow.steps:
            k = by_name.get(st.part.name)
            if st.part.name in skip:
                if k is None or st.part.alternatives is not None:
                    why = ("its record is not kept" if k is None else "several producers: the recorded one cannot be "
                           "given without running the others")
                    return None, [Mismatch(at[st.part.name][0], st.part.name, f"not re-run ({skip[st.part.name]}) and "
                                           f"{why} — the decision is not checked past this step", "not_kept")]
                st = dataclasses.replace(st, part=dataclasses.replace(st.part, func=lambda _k=k, **_: _recorded_output(_k)))
            new_steps.append(st)
        flow = dataclasses.replace(flow, steps=new_steps,
                                   batches=[b for b in flow.batches if not any(n in skip for n in b)])
    trace, vals = execute(system.catalog, flow, p.state, workers=1, order=p.order, costs=system.cost_book,
                          policy=p.policy, known=p.known, early_exit=kept.early_exit)
    lossy = getattr(kept, "unrestored", None) or {}
    if trace.init_hash != d.get("init_hash"):
        if lossy:
            return None, [Mismatch(0, "init", "the recorded input cannot be checked: " + "; ".join(
                f"{k} came back from storage as JSON gave it — {v}" for k, v in lossy.items()), "not_restored")]
        return None, [Mismatch(0, "init", "init_hash does not match the recorded input", "integrity")]
    prev = trace.init_hash
    for i, r in enumerate(trace.records):
        want = steps[i] if i < len(steps) else None
        k = have.get(want) if want is not None else None
        if k is not None and vhash(k.body()) != k.hash:
            return None, [Mismatch(k.step, k.name, "kept record modified after execution", "integrity")]
        if k is not None and _same_output(r, k) and k.prev == prev:
            r = trace.records[i] = k                  # the kept record: the model's details as they were
        else:
            r.prev = prev
            r.hash = vhash(r.body())
        if want != (r.step, r.name, r.hash):
            kind = "not_kept" if r.name in skip else "recompute"
            why = (f"re-run gives step {r.step} {r.name} #{r.hash[:16]}, the record has "
                   + (f"step {want[0]} {want[1]} #{want[2][:16]}" if want else "no such step")
                   + (" — the kept model output does not give its record again" if kind == "not_kept" else
                      " — a compact record keeps the hash of the old step, not its value"))
            return None, [Mismatch(r.step, r.name, why, kind)]
        prev = r.hash
    for s in steps[len(trace.records):]:
        r = have.get(s)
        if r is None:
            return None, [Mismatch(s[0], s[1], "a step recorded after the flow is not kept in the compact record",
                                   "not_kept")]
        trace.records.append(r)
    trace.fingerprint = dict(kept.fingerprint or {})
    trace.rejected = list(kept.rejected or [])
    results = {q: load(_result_cls(), {**_RESULT_DEFAULTS, **r}) for q, r in (c.get("results") or {}).items()}
    values = {k: v for k, v in vals.items() if v is not MISSING}
    for r in trace.records:                           # the kept records' values (a model step given back, a head)
        if r.value is not MISSING and r.kind not in KEPT_STEPS and not r.name.startswith("answer:"):
            values.setdefault(r.name, r.value)
    resp = Response(results, p.flow, trace, values, float(c.get("ms") or 0.0), c.get("feasible", True),
                    c.get("violations"), system.catalog, list(c.get("safeguards") or []), int(c.get("model_outputs") or 0))
    resp._system, resp._heads = system, system.heads
    resp.stored_id = d.get("id")
    return resp, []


def _result_cls():
    from ..runtime import Result
    return Result


class TraceStorage(abc.ABC):
    """A store of responses and their traces with a hash chain across the stored records (see the module docstring).

    The base class of every store (an extension point: `from solvi.core import TraceStorage`). A backend of your own
    implements five methods — `_append(body)` (assign seq, time, prev, hash and id atomically; → the record),
    `_raw(snap=None)` (every chained record in stored order: (position, dict or None when unreadable)), `_find(id)`
    (the record or None), `head()` ({"count", "hash"}) and `_rewrite(rec)` (replace the record with that seq: redact
    uses it) — and may override `close()`, `_snapshot()`, `_backend_problems(rows, snap)`, `iter` / `query` (for an
    index). It gets for free: the hash chain and `verify` (with an anchor and a signature), `save` / `get` /
    `rederive` (full and compact records), corrections and verified labels with their source checks, `query`,
    `report`, `replay_all`, `quarantine` / `where_is`, `redact`, `signature`, and every reader of a store (the audit,
    the system report, diff). `solvi.testing.conformance.check_storage(MyStorage)` checks a backend. Stability:
    stable; the record format is stable (stores of 0.7 to 0.9 read and verify).

    save(response, meta=None) → id; get(id) → Response; record(id) → the stored dict; iter() / query(...) → [Stored];
    corrections() → the teach records; head() → {"count", "hash"}; signature(); verify(anchor=None, signature=None);
    replay_all(system);
    quarantine(fact, value=...); where_is(fact, value=...).

    `system`: the System (or a Catalog) used to restore typed values (dates, enums, models) when loading responses;
    System(storage=...) sets it to that system when it is not set. `clock`: a function → seconds since the epoch.

    record: what a decision's record holds — "full" (default: the whole response, every step's value and inputs),
    "compact" (for frequent decisions: the input, the answers, every check's result and reasons, the model-backed steps
    whole with the models' ids, fingerprints and tokens, and every step's hash; not the computed values, the planned
    flow or the timings — see compact_content) or "sample:N" (one decision in N, counted by this store object, in full,
    the others compact). A compact record is hashed into the chain like any other: verify() checks it, its kept steps'
    places and links in the chain of step hashes, and its answers. What it does not keep is re-computed, not read:
    replay_all re-runs the decision from the recorded input and compares every step's hash (rederive); a model step
    that is not re-run on replay (trust_models=True, a generator) leaves the decision unchecked past the chain, and the
    replay says so (mismatch kind "not_kept") instead of passing it. get(id) raises CompactRecord; the reports read the
    answers, checks, models and costs from what is kept; quarantine re-derives compact decisions when the store knows
    the System; diff compares their answers only (no steps are kept to say what changed them)."""

    @_deprecate.removed_kwargs(catalog="system")
    def __init__(self, system=None, clock=None, record="full"):
        self.catalog = system                         # the System (or Catalog) typed values are restored with
        self.clock = clock or _time.time
        self.record_mode = record_mode(record)   # ("full" | "compact" | "sample", N): what save() keeps of a decision
        self._saved = 0                               # decisions saved by this object ("sample:N" draws every N-th)

    # --- a backend implements these
    @abc.abstractmethod
    def _append(self, body):
        """Add a record: assign seq, time, prev, hash and id atomically; → the record."""
        raise NotImplementedError

    @abc.abstractmethod
    def _raw(self, snap=None):
        """Every chained record in stored order (for verify) → iterator of (position, dict or None when unreadable).
        `snap`: a _snapshot() — the records as they were then."""
        raise NotImplementedError

    def _snapshot(self):
        """What verify() reads before the records so that records appended while it runs are not taken for damage: the
        stored head (and what else the backend needs) at one moment. None: the backend has nothing to pin."""
        return None

    @abc.abstractmethod
    def _find(self, id):
        """The stored record (a dict) with this id, or None."""
        raise NotImplementedError

    @abc.abstractmethod
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
        ticket id, a user) — hashed into the chain like the rest. What the record holds follows the store's `record=`."""
        mode, n = self.record_mode
        if mode == "sample":
            with _SAMPLE_LOCK:
                k, self._saved = self._saved, self._saved + 1
            mode = "full" if k % n == 0 else "compact"
        rec = self._append(entry(response, meta, mode))
        response.stored_id = rec["id"]
        return rec["id"]

    @_deprecate.removed_kwargs(source="label_source")
    def save_correction(self, question, init_state, answer, meta=None, *, label_source="human", by=None, of=None,
                        note=None):
        """Store a correction (what System.teach records) → its id. label_source (`source=` in 0.7; stored as the
        record's "source"): where the label comes from — "human" (a person
        corrected or confirmed the answer), "outcome" (what really happened: the parcel was lost, the loan defaulted) or
        "rule" (code rejected a model's proposal and decided instead), or "verified" — a System 2 answer that passed
        the checks and a guarantee: `of` must then name that decision, stored in this store, which answered `question`
        alone (status "ok", not escalated or abstained) under a guarantee (System.guarantee) with this very answer;
        anything else raises UntrustedLabel, and the label is not stored. Anything else is refused (UntrustedLabel):
        the system's own unverified answers are never labels. by: who (a user, a reviewer, a process); of: the stored id
        of the decision it corrects (for "verified", the one it is); note: a line of text kept with it (what happened —
        System.outcome writes it). Records of 0.6 have no source: they are human corrections. Storing a verified label
        teaches nothing by itself: which channel may read it, see check_source."""
        source = label_source
        check_source(source, accept=(VERIFIED,))
        if source == VERIFIED:
            self._check_vouched(question, answer, of)
        body = {"v": FORMAT, "kind": "teach", "teach": question, "init": plain(dict(init_state)), "answer": plain(answer)}
        if source != "human":
            body["source"] = source
        if by is not None:
            body["by"] = str(by)
        if of is not None:
            body["of"] = str(of)
        if note is not None:                          # content (redact erases it), not a lasting field like a mark's note
            body["happened"] = str(note)
        if meta is not None:
            body["meta"] = plain(meta)
        return self._append(body)["id"]

    def _check_vouched(self, question, answer, of):
        """A "verified" label: `of` is a decision in this store that answered `question` alone, under a guarantee, with
        `answer` — else UntrustedLabel saying which of these fails."""
        if of is None:
            raise UntrustedLabel("a verified label names the System 2 decision it comes from: of=<its stored id>")
        d = self._find(str(of))
        if d is None or d.get("kind", "ask") != "ask":
            raise UntrustedLabel(f"a verified label's decision {of!r} is not a stored decision of this store")
        got = (d.get("answers") or {}).get(question)
        if got is None:
            raise UntrustedLabel(f"the stored decision {of!r} did not answer {question!r}")
        if got[2] != "ok":
            raise UntrustedLabel(f"the stored decision {of!r} did not answer {question!r} alone (status {got[2]!r}): "
                                 "only an answer that passed the checks and its guarantee is a verified label")
        guarded = (d.get("guards") or {}).get(question) or any(r[1] == f"guard:{question}" for r in d.get("records") or ())
        if not guarded:                           # a decision part's act_guard, or a question's System.guarantee
            raise UntrustedLabel(f"the stored decision {of!r} answered {question!r} without a guarantee "
                                 "(System.guarantee): it is not verified")
        from ..schema import tag_floats
        if tag_floats(plain(self._answer_key(question, answer))) != got[0]:
            raise UntrustedLabel(f"the stored decision {of!r} answered {question!r} with {got[0]!r}, not {answer!r}: a "
                                 "verified label is that decision's own answer")

    # --- reading
    def record(self, id):
        """The stored record (a dict) with this id; KeyError if there is none."""
        d = self._find(id)
        if d is None:
            raise KeyError(f"no stored record {id!r}")
        return d

    @_deprecate.removed_kwargs(catalog="system")
    def get(self, id, system=None):
        """The stored response with this id, loaded back with `system` (default: the store's; see Stored.response)."""
        return _stored(self.record(id), self.catalog).response(system)

    def rederive(self, id, system=None, trust_models=False):
        """A stored decision as a Response — a full record loaded as get() loads it; a compact one re-run with `system`
        (default: the store's) from its recorded input, every step checked against the recorded hashes (see the module
        function rederive). Raises CompactRecord with the reasons when a compact decision cannot be rebuilt."""
        d = self.record(id)
        if d.get("record") != "compact":
            return self.get(id, system)
        resp, bad = rederive(d, system if system is not None else self.catalog, trust_models)
        if resp is None:
            raise CompactRecord(f"record {id} cannot be re-derived: " + "; ".join(f"{m[1]}: {m[2]}" for m in bad))
        return resp

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
        return None

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
        learn_rule, a CorrectionMemory or solvi.experimental.learning.Learning). source: "human" (also every record without one), "outcome",
        "rule", "verified" — or, for a record written around save_correction, whatever it says (solvi.core.knowledge.memory and
        solvi.experimental.learning.Learning refuse anything outside TRUSTED_SOURCES, "verified" included; System.guarantee takes "verified"
        when told to)."""
        from ..schema import untag_floats                # stored tagged ({"$float": "inf"}), read back as the float
        return [{"id": s.id, "time": s.time, "question": s.data["teach"], "init": untag_floats(s.data["init"]),
                 "answer": untag_floats(s.data["answer"]), "source": s.data.get("source", "human"), "by": s.data.get("by"),
                 "of": s.data.get("of"), **({"note": s.data["happened"]} if "happened" in s.data else {})}
                for s in self.iter("teach")]

    @_deprecate.removed_kwargs(catalog="catalog_fp")
    def query(self, question=None, answer=ANY, status=None, safeguard=None, model=None, since=None, until=None,
              catalog_fp=None):
        """Stored responses that match every filter given → [Stored] in stored order.
        question: asked this question; answer: answered this (with question: that question's answer; None matches an
        abstention); status: "ok" / "forced" / "abstain" (with question: of that question); safeguard: a safeguard of this
        kind fired ("grounding", "hard_check", "low_confidence", ...; with question: one that concerns it); model: a model
        with this fingerprint, id or type produced a step; since / until: stored in [since, until) — seconds since the
        epoch, a datetime, a date or an ISO string; catalog_fp: decided by the catalog with this fingerprint
        (System.fingerprint()["catalog"]). An answer is matched in its stored form: answer=True finds a yes/no "yes"
        (see _answer_key)."""
        since, until = _when(since), _when(until)
        answer = self._answer_key(question, answer)
        return [s for s in self.iter() if _matches(s.data, question, answer, status, safeguard, model, since, until, catalog_fp)]

    def report(self, since=None, until=None, question=None, format="md", examples=3, system=None, **filters):
        """A human-readable report of the stored decisions in [since, until) (optionally of one question): counts by
        answer, status and safeguard, the escalation rate, the guarantee coverage of the answers a model took part in,
        changes of the catalog's and the models' fingerprints, and `examples` stored ids per answer, escalation and
        safeguard. format: "md", "html" (one self-contained page) or "data" (a dict); other `filters` as query. See
        solvi.core.store.report."""
        from .report import period, render
        return render(period(self, since, until, question, examples, system, **filters), format)

    # --- erasure
    @abc.abstractmethod
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
        """The signature of the chained records (solvi.core.store.signature.sign): {"alg", "count", "root"} — with the default
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
        changed, its seq is named and `signature` in the result holds solvi.core.store.signature.check's answer (the original content
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
        changed since (see solvi.core.runtime.Mismatch). A stored record that cannot be loaded is one mismatch (0, "load", ...),
        a replay that raises is (0, "replay", ...): both of kind "error", no verdict on the data. "note" (a record of
        format 1 whose model step does not recompute): it was stored with sorted dict keys, see LEGACY_ORDER.
        A compact record (record="compact") is re-run from its recorded input and every step compared with its recorded
        hash (rederive), then replayed like a full one; one that cannot be — a model step not re-run (kind "not_kept":
        no verdict on the data past the chain), a step whose hash differs (kind "recompute": the old value is not kept)
        — is listed with "record": "compact"."""
        from ..runtime import Mismatch, mismatch_summary
        bad = []

        def failed(s, stage, e):
            ms = [Mismatch(0, stage, f"{type(e).__name__}: {e}", "error")]
            bad.append({"id": s.id, "seq": s.seq, "time": s.time, "mismatches": ms, "models": [], "catalog": None,
                        **mismatch_summary(ms)})

        for s in (self.query(**filters) if filters else self.iter()):
            if s.compact:
                try:
                    r, ms = rederive(s.data, system, trust_models)
                except Exception as e:  # noqa: BLE001
                    failed(s, "rederive", e)
                    bad[-1]["record"] = "compact"
                    continue
                if r is None:
                    bad.append({"id": s.id, "seq": s.seq, "time": s.time, "mismatches": ms, "models": [],
                                "catalog": None, "record": "compact", **mismatch_summary(ms)})
                    continue
            else:
                try:
                    r = s.response(system)
                except Exception as e:  # noqa: BLE001
                    failed(s, "load", e)
                    continue
            try:                                      # a compact decision's steps (and models) were just re-run by
                rep = r.trace.replay(system, r.flow, trust_models=trust_models or s.compact)   # rederive: not again
            except Exception as e:  # noqa: BLE001
                failed(s, "replay", e)
                continue
            if not rep["ok"]:
                b = {"id": s.id, "seq": s.seq, "time": s.time, "mismatches": rep["mismatches"], "models": rep["models"],
                     "catalog": rep["catalog"], "kinds": rep["kinds"], "summary": rep["summary"]}
                rerun = {step for step, _, verdict in rep["models"] if verdict == "recomputed"}
                if int(s.data.get("v") or 1) < 2 and any(m.kind == "recompute" and m[0] in rerun for m in rep["mismatches"]):
                    b["note"] = LEGACY_ORDER          # a model step of an old record did not recompute: maybe only this
                if s.compact:
                    b["record"] = "compact"
                bad.append(b)
        return bad

    # --- provenance over the store
    def quarantine(self, fact, value=ANY):
        """A fact (or a given input) found to be wrong: the stored decisions whose answers rest on it — through the trace's
        provenance graph (each step's recorded inputs, from the answer back to the fact; a hard check that decided an
        answer counts). `value`: only where the fact had this value (compared by its hash as the consumers recorded it).
        → [{"id", "seq", "time", "questions": {question: {"answer", "status", "path": [fact, ..., "answer:question"]}}}].
        Nothing is changed: re-decide or review these.
        A compact record keeps no step inputs: it is re-derived with the store's System (rederive) and its rebuilt
        trace is searched; one that cannot be (no System, a model step not re-run, a step that no longer recomputes) is
        not searched, and a UserWarning names how many and their ids."""
        from ..runtime import vhash
        vh = None if value is ANY else vhash(value)
        out, unchecked = [], []
        for s in self.iter():
            d = s.data.get("response")
            if s.compact:
                r, _ = rederive(s.data, self.catalog) if self.catalog is not None else (None, None)
                if r is None:
                    unchecked.append(s.id)
                    continue
                d = r.to_dict()
            hit = _dependents(d, fact, vh)
            if hit:
                out.append({"id": s.id, "seq": s.seq, "time": s.time, "questions": hit})
        if unchecked:
            import warnings
            warnings.warn(f"quarantine({fact!r}): {len(unchecked)} compact record(s) could not be re-derived and were not "
                          f"searched ({', '.join(unchecked[:5])}{', ...' if len(unchecked) > 5 else ''}): a compact record "
                          "keeps no step inputs — give the store its System (TraceStorage(system=...)), or review them",
                          UserWarning, stacklevel=2)
        return out

    forget = _deprecate.removed_attr("forget()", "where_is(fact, value) (forget never deleted anything)", "TraceStorage")

    def where_is(self, fact, value=ANY):
        """Where a given fact (e.g. a person's data) is held, and what removing it would touch (`forget` in 0.7) — a
        report only: nothing is deleted (deleting a
        stored record breaks the chain by design; keep the report as the record of the request, and erase the records it
        lists with redact).
        → {"fact", "value", "dependent": the stored decisions whose answers rest on it (as quarantine), "stored": ids of the
        stored records that hold it without an answer resting on it (responses and corrections), "deleted": 0}. A compact
        record that holds it in its input is listed under "stored": it keeps no step inputs to say what rests on it
        (quarantine re-derives it for that)."""
        from ..runtime import vhash
        from ..schema import untag_floats
        vh = None if value is ANY else {vhash(value), vhash(plain(value))}
        dependent, stored = [], []
        for s in self.iter(None):
            if s.kind == "ask":
                v = view(s.data) or {}
                init = (v.get("trace") or {}).get("init") or {}
                hit = _dependents(v, fact, None if vh is None else vhash(value)) if not s.compact else None
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
    appends; get(id) then scans the file. A head that does not match the last line is not trusted: the file is read.
    record: "full", "compact" or "sample:N" (see TraceStorage).

    Rotation. rotate_bytes / rotate_records: when the file has reached that many bytes (or chained records), the next
    append first moves it to a closed segment `<name>.000001.jsonl` (with its head; the number counts up) and starts
    `path` again with a record of kind "segment" that names the closed segment, its record count and its last hash —
    so each file verifies on its own, the first record of a file is linked to the head of the one before it, and
    verify_segments() checks every segment and every link (a segment replaced, cut short or removed breaks a link).
    Reads (iter, query, get, replay_all) and redact see the current file; segments() lists the closed ones, each
    readable as a JSONLStorage of its own. A closed segment is not written to: redacting one of its records there would
    append to it and break the link of the file after it (not supported yet). With rotation, writers of the same path
    lock `<path>.lock` (a file that is never moved): every process writing a rotated store must open it with rotation
    on."""

    @_deprecate.removed_kwargs(catalog="system")
    def __init__(self, path, system=None, clock=None, fsync=False, index=True, *, record="full", rotate_bytes=None,
                 rotate_records=None):
        super().__init__(system, clock, record)
        self.path = os.fspath(path)
        self.head_path = self.path + ".head"
        self.fsync = fsync
        for k, v in (("rotate_bytes", rotate_bytes), ("rotate_records", rotate_records)):
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise ValueError(f"{k} is a positive int or None, not {v!r}")
        self.rotate_bytes, self.rotate_records = rotate_bytes, rotate_records
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
        self._ino = None

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
        st = os.fstat(fh.fileno())
        size = st.st_size
        if self._ino is not None and st.st_ino != self._ino:
            self._reset()                             # the file was rotated (or replaced): read the new one from the start
        self._ino = st.st_ino
        if size == self._size:
            return
        if size < self._size:
            self._reset()
            self._ino = st.st_ino
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

    @property
    def rotates(self):
        return self.rotate_bytes is not None or self.rotate_records is not None

    @contextlib.contextmanager
    def _writing(self):
        """The data file open for appending, under the thread lock and the file locks (with rotation also `<path>.lock`,
        which is never moved, so a writer never appends to a file another writer has just rotated away)."""
        with self._lock, contextlib.ExitStack() as stack:
            if self.rotates:
                lk = stack.enter_context(open(self.path + ".lock", "a+b"))
                stack.enter_context(_flock(lk))
            fh = stack.enter_context(open(self.path, "a+b"))
            stack.enter_context(_flock(fh))
            yield fh

    def _due(self):
        return self._count > 0 and ((self.rotate_bytes is not None and self._size >= self.rotate_bytes) or
                                    (self.rotate_records is not None and self._count >= self.rotate_records))

    def segments(self):
        """The closed segments of a rotated store, oldest first → [path] (the current file, `path`, is not listed)."""
        import glob
        import re
        stem, ext = os.path.splitext(self.path)
        pat = re.compile(re.escape(os.path.basename(stem)) + r"\.(\d{6})" + re.escape(ext) + "$")
        found = [(int(m.group(1)), f) for f in glob.glob(glob.escape(stem) + ".*" + ext)
                 if (m := pat.search(os.path.basename(f)))]
        return [f for _, f in sorted(found)]

    def verify_segments(self, anchor=None):
        """Verify a rotated store whole: every closed segment and the current file (each as verify() checks it) and
        every link — the first record of each file after the first is a "segment" record naming the file before it, its
        record count and its last hash, which must be that file's. `anchor`: a head() of the current file kept
        elsewhere. → {"ok", "files": [{"path", "ok", "count", "head"}], "count": records in all files, "problems":
        [(path, seq, id, reason)]}."""
        files = self.segments() + [self.path]
        out, problems, before = [], [], None
        for i, f in enumerate(files):
            st = self if f == self.path else JSONLStorage(f, self.catalog)
            v = st.verify(anchor=anchor if f == self.path else None)
            problems += [(f, *p) for p in v["problems"]]
            first = next((d for _, d in st._raw() if chained(d)), None)
            link = (first or {}).get("segment") if (first or {}).get("kind") == "segment" else None
            if before is not None:
                want = {"previous": os.path.basename(before[0]), "count": before[1]["count"], "hash": before[1]["hash"]}
                if link is None:
                    problems.append((f, 0, None, f"the file does not start with the link to {os.path.basename(before[0])}"))
                elif link != want:
                    problems.append((f, 0, first.get("id"), f"the link names {link}, the segment before it is {want}"))
            elif link is not None:
                problems.append((f, 0, first.get("id"), f"the first file links to {link.get('previous')!r}, which is "
                                                         "not among the segments (removed?)"))
            out.append({"path": f, "ok": v["ok"], "count": v["count"], "head": v["head"]})
            before = (f, v["head"])
        return {"ok": not problems, "files": out, "count": sum(x["count"] for x in out), "problems": problems}

    def _rotate(self):
        """Close the current file as the next segment and start `path` with the segment record (under _writing's
        locks) → the link the new file's first record carries: the closed segment's name, record count and last hash."""
        segs = self.segments()
        stem, ext = os.path.splitext(self.path)
        n = int(os.path.basename(segs[-1])[len(os.path.basename(stem)) + 1:][:6]) + 1 if segs else 1
        seg = f"{stem}.{n:06d}{ext}"
        link = {"previous": os.path.basename(seg), "count": self._count, "hash": self._last}
        os.replace(self.path, seg)
        if os.path.exists(self.head_path):
            os.replace(self.head_path, seg + ".head")
        self._reset()
        return link

    def _append(self, body):
        with self._writing() as fh:
            self._sync(fh)                            # what other writers appended: the chain goes on from their last
            if self.rotates and self._due() and body.get("kind") != "redaction":   # an erasure stays in the file of
                                                                                   # the record it erases
                link = self._rotate()
                with open(self.path, "a+b") as nf, _flock(nf):
                    self._sync(nf)
                    self._write(nf, {"v": FORMAT, "kind": "segment", "segment": link})
                    return self._write(nf, body)
            return self._write(fh, body)

    def _write(self, fh, body):
        """Append one record to the open, locked file → the record (seq, time, prev, hash, id assigned)."""
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
        with self._lock, contextlib.ExitStack() as stack:
            if self.rotates:                          # the lock every writer of a rotated store takes
                stack.enter_context(_flock(stack.enter_context(open(self.path + ".lock", "a+b"))))
            fh = stack.enter_context(open(self.path, "r+b"))
            stack.enter_context(_flock(fh))
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

    @_deprecate.removed_kwargs(catalog="catalog_fp")
    def query(self, question=None, answer=ANY, status=None, safeguard=None, model=None, since=None, until=None,
              catalog_fp=None):
        since, until = _when(since), _when(until)
        answer = self._answer_key(question, answer)
        where, args = ["r.kind = 'ask'"], []
        if catalog_fp is not None:
            where.append("r.\"catalog\" = ?")
            args.append(catalog_fp)
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

    @_deprecate.removed_kwargs(catalog="system")
    def __init__(self, path, system=None, clock=None, timeout=30.0, *, record="full"):
        import sqlite3
        super().__init__(system, clock, record)
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

    @_deprecate.removed_kwargs(catalog="system")
    def __init__(self, conninfo, system=None, clock=None, prefix="solvi_", *, record="full"):
        super().__init__(system, clock, record)
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

    @_deprecate.removed_kwargs(catalog="system")
    def __init__(self, path, system=None, clock=None, prefix="", *, record="full"):
        try:
            import duckdb
        except ImportError as e:
            raise ImportError("DuckDBStorage needs duckdb: pip install 'solvi[duckdb]'") from e
        super().__init__(system, clock, record)
        self.path = os.fspath(path)
        self._open(duckdb.connect(self.path), prefix)


def open_storage(where, system=None):
    """A TraceStorage from a path: .db / .sqlite / .sqlite3 → SQLiteStorage, .duckdb → DuckDBStorage (in any case), a
    postgresql:// (or postgres://) URL → PostgresStorage, anything else → JSONLStorage; a TraceStorage is returned as
    it is."""
    if where is None or isinstance(where, TraceStorage):
        return where
    p = os.fspath(where)
    if p.startswith(("postgresql://", "postgres://")):
        return PostgresStorage(p, system)
    ext = p.lower()                                   # "decisions.DB" is a database too, not a JSON-lines file
    if ext.endswith(".duckdb"):
        return DuckDBStorage(p, system)
    if ext.endswith((".db", ".sqlite", ".sqlite3")):
        return SQLiteStorage(p, system)
    return JSONLStorage(p, system)


__all__ = ["chained", "check_source", "compact_content", "CompactRecord", "DuckDBStorage", "entry", "FORMAT",
           "JSONLStorage", "open_storage", "plain", "PostgresStorage", "record_body", "record_hash", "record_mode",
           "RECORD_MODES", "rederive", "SQLiteStorage", "Stored", "TraceStorage", "TRUSTED_SOURCES", "UntrustedLabel",
           "VERIFIED", "VERIFIED_REFUSED", "view"]
