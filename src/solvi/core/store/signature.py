"""A signature of a trace or a store: a few numbers that, besides saying "changed", say WHICH record changed and what its
content hash was.

The hash chain (Trace.replay, TraceStorage.verify) is strict integrity. When a record is edited and every hash after it
is recomputed together with the stored head, a head kept elsewhere (`verify(anchor=...)`) still says "rewritten" — but
not where, and not what was there. A signature kept next to that head answers both, for one changed record:

    sig = sign(store)                  # {"alg": "syndrome", "count", "root": 2 numbers}; keep it where you keep the head
    ...
    locate(store, sig)                 # None: unchanged; an index: the one record that changed
    repair(store, sig)                 # {"index", "digest": its original content hash, "match": the candidate it equals}

Every signed item is reduced to its content hash (SHA-256; the chain fields `prev`, `hash`, `id` left out, so recomputing
the chain does not move the other items). The code is named in the signature's "alg":

- "syndrome": S0 = Σ h_i and S1 = Σ (i+1)·h_i mod a 256-bit prime — 64 bytes. One change at k by d moves S0
  by d and S1 by (k+1)·d: k = ΔS1 / ΔS0, and the original hash is h_k − d, all 32 bytes. Several changes give a k outside
  the store (except with probability ~n / 2^256): detected, not located. A classical single-error-locating code.

The positional octonion code that was the second "alg" up to 0.7 is an experiment in benchmarks/octonion_signature.py
since 0.8: on a flat store it located exactly as the syndrome code does, 4x larger and ~60x slower
(benchmarks/trace_signature.py compares the two).

Limits (tests/test_signature.py pins each):
- one changed record is located and its content hash restored; the record itself (its value) only from `candidates`
  (a backup, another replica, the values that fact had elsewhere) — the signature holds a hash, not the record;
- several changed records (a reorder is two) are detected but not located: `locate` raises NotLocatable;
- a deleted or inserted record shifts every position after it: detected as a count or content change, not located;
- records appended after signing are not covered (sign again or extend(), as with the head); the signed prefix is
  checked;
- it is an error-locating code, not a MAC: someone who can rewrite the signature too can forge it. Keep it where you
  keep the head."""
from __future__ import annotations

import hashlib
import json

class NotLocatable(ValueError):
    """The signature does not match, and not because of one changed record (several changed, reordered, deleted or
    inserted records, or another count)."""


# --- what is signed: content hashes of the items in order
def _cj(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)


def record_digest(rec):
    """Content hash (32 bytes) of one signed item: a stored record (dict; without `id`, `hash`, `prev`), a trace Record
    (its hashed body without `prev`), bytes (as they are) or anything else (its canonical JSON)."""
    from ..runtime import Record, _canon
    if isinstance(rec, (bytes, bytearray)):
        return hashlib.sha256(rec).digest()
    if isinstance(rec, Record):
        b = {k: v for k, v in rec.body().items() if k != "prev"}
        return hashlib.sha256(_cj(_canon(b)).encode()).digest()
    if isinstance(rec, dict) and int(rec.get("v") or 1) >= 2 and "seq" in rec:
        from . import record_body               # a stored record of format 2: computed the same before and after
        return hashlib.sha256(_cj({k: v for k, v in record_body(rec).items() if k != "prev"}).encode()).digest()   # redact
    if isinstance(rec, dict) and isinstance(rec.get("redacted"), dict) and rec["redacted"].get("digest"):
        return bytes.fromhex(rec["redacted"]["digest"])       # format 1, redacted: the digest it reports of what it held
    if isinstance(rec, dict):
        return hashlib.sha256(_cj({k: v for k, v in rec.items() if k not in ("id", "hash", "prev")}).encode()).digest()
    if isinstance(rec, str):
        return hashlib.sha256(rec.encode()).digest()
    return hashlib.sha256(_cj(rec).encode()).digest()


def items(obj):
    """What sign() signs, in order: a TraceStorage → its chained records (dicts; unreadable ones as None); a Response or
    Trace → its input (init_hash) then its records; a list → its items."""
    from ..runtime import Trace
    from . import TraceStorage
    if isinstance(obj, TraceStorage):
        from . import chained
        return [d if chained(d) else None for _, d in obj._raw()]
    tr = getattr(obj, "trace", None)
    if isinstance(tr, Trace):
        obj = tr
    if isinstance(obj, Trace):
        return ["init|" + obj.init_hash, *obj.records]
    return list(obj)


def digests(obj):
    return [None if x is None else record_digest(x) for x in items(obj)]


# --- the syndrome code (the default): two sums mod a prime
P = 2**256 - 189                 # the largest prime below 2^256


def _int(d):
    return int.from_bytes(d, "big") % P


def _syn(ds, start=0, s0=0, s1=0):
    for i, d in enumerate(ds, start):
        h = _int(d)
        s0, s1 = (s0 + h) % P, (s1 + (i + 1) * h) % P
    return s0, s1


def _syn_root(sig):
    return int(sig["root"][0], 16), int(sig["root"][1], 16)


def _syn_hits(ds, bad, sig):
    """→ (unchanged, [(index, original digest bytes)])."""
    c0, c1 = _syn(ds)
    s0, s1 = _syn_root(sig)
    d0, d1 = (c0 - s0) % P, (c1 - s1) % P
    if d0 == 0 and d1 == 0 and not bad:
        return True, []
    if d0 == 0:                               # the sum kept, the weighted sum moved: a reorder or several changes
        return False, []
    k = d1 * pow(d0, -1, P) % P - 1            # one change at k by d0: the weighted sum moves by (k + 1) d0
    if not 0 <= k < len(ds):
        return False, []
    return False, [(k, ((_int(ds[k]) - d0) % P).to_bytes(32, "big"))]


ALGS = ("syndrome",)
_MOVED = ("the octonion signature code left the package in solvi 0.8: it is an experiment in "
          "benchmarks/octonion_signature.py (the syndrome code locates the same changes, 4x smaller, ~60x faster: "
          "benchmarks/trace_signature.py)")


def sign(obj, alg="syndrome"):
    """The signature of a trace, a Response, a TraceStorage (its chained records) or a list of items → {"alg", "count",
    "root"}: plain JSON — keep it next to the head, in a ticket, a log you do not control.

    alg "syndrome" (the only one): root = two numbers mod a 256-bit prime (hex; 64 bytes) — S0 = Σ h_i, S1 = Σ (i+1) h_i
    over the content hashes; one change at k by d moves them by d and (k+1) d, which gives k and the whole original hash."""
    if alg == "octonion":
        raise ValueError(_MOVED)
    if alg != "syndrome":
        raise ValueError(f"unknown signature alg {alg!r}: one of {ALGS}")
    ds = digests(obj)
    if any(d is None for d in ds):
        raise ValueError("the store has unreadable records: verify() it first")
    return {"alg": alg, "count": len(ds), "root": [format(x, "x") for x in _syn(ds)]}


def extend(signature, new_items, start=None):
    """The signature after appending `new_items` to what `signature` signed, in O(len(new_items)). `start`: the position
    of the first new item (default: signature["count"])."""
    _check(signature)
    n = signature["count"] if start is None else start
    new = [record_digest(x) for x in new_items]
    root = [format(x, "x") for x in _syn(new, n, *_syn_root(signature))]
    return {"alg": signature["alg"], "count": n + len(new), "root": root}


def _check(signature):
    size = {"syndrome": 2}
    if isinstance(signature, dict) and signature.get("alg") == "octonion":
        raise ValueError(_MOVED)
    if not isinstance(signature, dict) or signature.get("alg") not in size \
            or len(signature.get("root") or ()) != size[signature["alg"]] or not isinstance(signature.get("count"), int):
        raise ValueError(f"not a solvi signature (alg {' / '.join(ALGS)}): {signature!r:.120}")


def load(s):
    """A signature from JSON text, a file path, or a dict (checked: its "alg" says how to read it)."""
    import os
    if not isinstance(s, dict):
        s = os.fspath(s)
        if not s.lstrip().startswith("{"):
            with open(s) as fh:
                s = fh.read()
        s = json.loads(s)
    _check(s)
    return s


def check(obj, signature, candidates=None):
    """Compare `obj` with a signature taken earlier (its "alg" says which code) → {"ok", "alg", "count", "signed",
    "index", "digest", "match", "reason"}.

    ok: the first `signed` items are the ones signed. Otherwise, when one item changed: index (its position; for a trace
    0 is the input, i the record i-1), digest (the hex of its original content hash, 32 bytes) and match (the first of `candidates` — records, or for a trace record also plain values — whose
    content hash is that digest; None). When it cannot be located: index None and the reason. Items appended after
    signing are not covered and do not count."""
    _check(signature)
    its = items(obj)
    n = int(signature["count"])
    out = {"ok": False, "alg": signature["alg"], "count": len(its), "signed": n, "index": None, "digest": None,
           "match": None, "reason": None}
    if len(its) < n:
        out["reason"] = f"{n - len(its)} signed item(s) missing (deleted, or cut off the end)"
        return out
    its = its[:n]
    ds = [None if x is None else record_digest(x) for x in its]
    bad = [i for i, d in enumerate(ds) if d is None]
    ds = [d if d is not None else b"\0" * 32 for d in ds]        # an unreadable record: an item that is certainly wrong
    if n == 0:
        out["ok"] = True
        return out
    same, hits = _syn_hits(ds, bad, signature)
    if same:
        out["ok"] = True
        return out
    if len(hits) != 1:
        out["reason"] = ("several items changed, reordered, deleted or inserted: detected, not located" if not hits else
                         f"ambiguous: {len(hits)} positions fit")
        return out
    i, dig = hits[0]
    out["index"], out["digest"] = i, dig.hex()
    out["reason"] = "one item changed" + (" (unreadable)" if i in bad else "")
    for c in candidates or ():
        if any(record_digest(form)[:len(dig)] == dig for form in _forms(c, its[i])):
            out["match"] = c
            break
    return out


def _forms(candidate, current):
    """A candidate as it would be signed: itself, and for a trace record also the current record with its value."""
    from ..runtime import Record
    yield candidate
    if isinstance(current, Record) and not isinstance(candidate, Record):
        import dataclasses
        yield dataclasses.replace(current, value=candidate)


def locate(obj, signature):
    """None when the signed items are unchanged; else the position of the one that changed (for a Trace / Response 0 is
    the input, i the record i-1; for a store, its seq). NotLocatable when the change is not one item."""
    r = check(obj, signature)
    if r["ok"]:
        return None
    if r["index"] is None:
        raise NotLocatable(r["reason"])
    return r["index"]


def repair(obj, signature, candidates=()):
    """The one changed item → {"index", "digest": hex of its original content hash, "match": the candidate that is the
    original, or None}; None when nothing changed; NotLocatable as locate. A candidate is a record (a stored dict, a trace
    Record, bytes) — or, for a trace record, its original value."""
    r = check(obj, signature, candidates)
    if r["ok"]:
        return None
    if r["index"] is None:
        raise NotLocatable(r["reason"])
    return {"index": r["index"], "digest": r["digest"], "match": r["match"]}


__all__ = ["ALGS", "NotLocatable", "check", "extend", "load", "locate", "repair", "sign"]
