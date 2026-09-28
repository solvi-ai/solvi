"""An algebraic signature of a trace or a store: 32 numbers that, besides saying "changed", say WHICH record changed and
what its content hash was.

The hash chain (Trace.replay, TraceStorage.verify) is strict integrity. When a record is edited and every hash after it
is recomputed together with the stored head, a head kept elsewhere (`verify(anchor=...)`) still says "rewritten" — but
not where, and not what was there. A signature kept next to that head answers both, for one changed record:

    sig = sign(store)                  # {"alg", "count", "root": 32 numbers}; keep it where you keep the head
    ...
    locate(store, sig)                 # None: unchanged; an index: the one record that changed
    repair(store, sig)                 # {"index", "digest": its original content hash, "match": the candidate it equals}

How (the positional octonion code): every signed item is a leaf — its content hash (SHA-256, the chain fields `prev`,
`hash`, `id` left out, so recomputing the chain does not move other leaves) written into 4 octonions and multiplied by
an element of its position; the signature is the ordered product of the leaves, ((L0 L1) L2)... Octonions are a
division algebra (no zero divisors), so changing one leaf always changes the product; they are alternative, so a leaf
and a position are removed by division exactly, and one leaf can be solved back from the product and its neighbours.
The solved leaf decodes to the original content hash (the first 28 of its 32 bytes) at the changed position only;
anywhere else it decodes to noise, which the decoder rejects (a false hit needs 28 numbers to fall within 1e-6 of a
byte grid by chance, about 1e-90 per position).

Limits (tests/test_signature.py pins each):
- one changed record is located and its content hash restored; the record itself (its value) only from `candidates`
  (a backup, another replica, the values that fact had elsewhere) — the signature holds 28 bytes, not the record;
- several changed records (a reorder is two) are detected but not located: `locate` raises NotLocatable;
- a deleted or inserted record shifts every position after it: detected as a count or content change, not located;
- records appended after signing are not covered (sign again, as with the head); the signed prefix is checked;
- it is an error-locating code, not a MAC: someone who can rewrite the signature too can forge it. Keep it where you
  keep the head.

The method comes from our trace-signature experiment (solver traces, 34 356 substitutions): the positional code found
the changed step and restored the original record in 100% of single substitutions with 32 numbers. There leaves were
looked up in a knowledge base; here the content hash is written into the leaf itself, so no base is needed to name the
record and restore its hash. On a flat list of records the octonions' non-associativity (what lets the signature see a
change of brackets in a derivation tree) is not used: a classical syndrome code does the same job here (see the docs).
numpy only."""
from __future__ import annotations

import hashlib
import json

import numpy as np

ALG = "solvi-octonion-pos/1"
BLOCKS = 4                       # octonions per element: 4 x 8 = 32 numbers
BYTES = 7 * BLOCKS               # content-hash bytes written into a leaf (7 per octonion, the 8th component is the anchor)
GRID_TOL = 1e-6                  # a decoded component must lie this close to its byte grid point
SAME_TOL = 1e-8                  # two signatures within this (max abs) are the same


class NotLocatable(ValueError):
    """The signature does not match, and not because of one changed record (several changed, reordered, deleted or
    inserted records, or another count)."""


# --- the algebra
def _cayley_dickson(a, b):
    """Octonion (or smaller) product by Cayley–Dickson doubling: (p, q)(r, s) = (pr − s*q, sp + qr*)."""
    n = len(a)
    if n == 1:
        return [a[0] * b[0]]
    h = n // 2
    p, q, r, s = a[:h], a[h:], b[:h], b[h:]

    def conj(x):
        return [x[0]] + [-v for v in x[1:]]

    def add(x, y, sgn=1):
        return [u + sgn * v for u, v in zip(x, y)]
    return add(_cayley_dickson(p, r), _cayley_dickson(conj(s), q), -1) + add(_cayley_dickson(s, p), _cayley_dickson(q, conj(r)))


def _table():
    c = np.zeros((8, 8, 8))
    eye = np.eye(8)
    for i in range(8):
        for j in range(8):
            c[i, j] = _cayley_dickson(list(eye[i]), list(eye[j]))
    return c.reshape(64, 8)


_C = _table()
_CONJ = np.array([1.0] + [-1.0] * 7)


def mul(a, b):
    """Blockwise octonion product of arrays [..., BLOCKS, 8]."""
    a, b = np.broadcast_arrays(a, b)
    return (a[..., :, None] * b[..., None, :]).reshape(*a.shape[:-1], 64) @ _C


def inv(a):
    """Blockwise octonion inverse: conjugate / squared norm."""
    return a * _CONJ / (a * a).sum(-1, keepdims=True)


# --- leaves
_POS = {}


def _position(i):
    """The unit element of position i (seeded by SHA-256; the same everywhere)."""
    e = _POS.get(i)
    if e is None:
        seed = int.from_bytes(hashlib.sha256(f"{ALG}|pos|{i}".encode()).digest()[:8], "little")
        x = np.random.default_rng(seed).normal(size=(BLOCKS, 8))
        e = x / np.linalg.norm(x, axis=-1, keepdims=True)
        if len(_POS) < 1 << 16:
            _POS[i] = e
    return e


def encode(ds):
    """Content hashes (a list of bytes; the first 28 of each are used) → their elements [n, BLOCKS, 8]: per octonion an
    anchor 1 and 7 bytes mapped to (−1, 1), scaled to unit norm (decoding divides by the anchor: the scale does not
    matter)."""
    b = np.frombuffer(b"".join(bytes(d[:BYTES]) for d in ds), dtype=np.uint8).astype(np.float64).reshape(-1, BLOCKS, 7)
    x = np.concatenate([np.ones((len(b), BLOCKS, 1)), (b - 127.5) / 128.0], axis=-1)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def decode(x):
    """Elements [..., BLOCKS, 8] → (valid [...], bytes [..., 28]): valid where every component lies on the byte grid."""
    a = x[..., :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = x[..., 1:] / a * 128.0 + 127.5
    r = np.rint(v)
    ok = np.isfinite(v).all((-1, -2)) & (np.abs(v - r) < GRID_TOL * 128).all((-1, -2)) & (r >= 0).all((-1, -2)) \
        & (r <= 255).all((-1, -2))
    return ok, np.clip(np.nan_to_num(r), 0, 255).astype(np.uint8).reshape(*x.shape[:-2], BYTES)


def leaves(ds, start=0):
    """The leaves of content hashes at positions start, start+1, ... → [n, BLOCKS, 8]."""
    if not ds:
        return np.zeros((0, BLOCKS, 8))
    return mul(encode(ds), np.stack([_position(start + i) for i in range(len(ds))]))


# --- what is signed: content hashes of the items in order
def _cj(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)


def record_digest(rec):
    """Content hash (32 bytes) of one signed item: a stored record (dict; without `id`, `hash`, `prev`), a trace Record
    (its hashed body without `prev`), bytes (as they are) or anything else (its canonical JSON)."""
    from .runtime import Record, _canon
    if isinstance(rec, (bytes, bytearray)):
        return hashlib.sha256(rec).digest()
    if isinstance(rec, Record):
        b = {k: v for k, v in rec.body().items() if k != "prev"}
        return hashlib.sha256(_cj(_canon(b)).encode()).digest()
    if isinstance(rec, dict):
        return hashlib.sha256(_cj({k: v for k, v in rec.items() if k not in ("id", "hash", "prev")}).encode()).digest()
    if isinstance(rec, str):
        return hashlib.sha256(rec.encode()).digest()
    return hashlib.sha256(_cj(rec).encode()).digest()


def items(obj):
    """What sign() signs, in order: a TraceStorage → its chained records (dicts; unreadable ones as None); a Response or
    Trace → its input (init_hash) then its records; a list → its items."""
    from .runtime import Trace
    from .storage import TraceStorage
    if isinstance(obj, TraceStorage):
        return [d if isinstance(d, dict) and "hash" in d else None for _, d in obj._raw()]
    tr = getattr(obj, "trace", None)
    if isinstance(tr, Trace):
        obj = tr
    if isinstance(obj, Trace):
        return ["init|" + obj.init_hash, *obj.records]
    return list(obj)


def digests(obj):
    return [None if x is None else record_digest(x) for x in items(obj)]


def _fold(lv, acc=None):
    """The running products ((L0 L1) L2)... of leaves [n, BLOCKS, 8] (after `acc` when given) → [n, BLOCKS, 8]."""
    out = np.empty_like(lv)
    for i in range(len(lv)):
        acc = lv[i] if acc is None else mul(acc, lv[i])
        out[i] = acc
    return out


def sign(obj):
    """The signature of a trace, a Response, a TraceStorage (its chained records) or a list of items →
    {"alg", "count", "root": 32 numbers}. Plain JSON: keep it next to the head, in a ticket, a log you do not control."""
    ds = digests(obj)
    if any(d is None for d in ds):
        raise ValueError("the store has unreadable records: verify() it first")
    root = _fold(leaves(ds))[-1] if ds else np.zeros((BLOCKS, 8))
    return {"alg": ALG, "count": len(ds), "root": [float(x) for x in root.ravel()]}


def extend(signature, new_items, start=None):
    """The signature after appending `new_items` to what `signature` signed (O(len(new_items)): the product is a left
    fold). `start`: the position of the first new item (default: signature["count"])."""
    _check(signature)
    n = signature["count"] if start is None else start
    acc = np.asarray(signature["root"], dtype=np.float64).reshape(BLOCKS, 8) if signature["count"] else None
    new_items = list(new_items)
    if new_items:
        acc = _fold(leaves([record_digest(x) for x in new_items], n), acc)[-1]
    return {"alg": ALG, "count": n + len(new_items),
            "root": [float(x) for x in (acc if acc is not None else np.zeros((BLOCKS, 8))).ravel()]}


def _check(signature):
    if not isinstance(signature, dict) or signature.get("alg") != ALG or len(signature.get("root") or ()) != BLOCKS * 8:
        raise ValueError(f"not a {ALG} signature: {signature!r:.120}")


def load(s):
    """A signature from JSON text, a file path, or a dict."""
    import os
    if isinstance(s, dict):
        return s
    s = os.fspath(s)
    if not s.lstrip().startswith("{"):
        with open(s) as fh:
            s = fh.read()
    return json.loads(s)


def check(obj, signature, candidates=None):
    """Compare `obj` with a signature taken earlier → {"ok", "count", "signed", "index", "digest", "match", "reason"}.

    ok: the first `signed` items are the ones signed. Otherwise, when one item changed: index (its position; for a trace
    0 is the input, i the record i-1), digest (the hex of its original content hash, 28 bytes) and match (the first of
    `candidates` — records, or for a trace record also plain values — whose content hash is that digest; None). When it
    cannot be located: index None and the reason. Items appended after signing are not covered and do not count."""
    _check(signature)
    root = np.asarray(signature["root"], dtype=np.float64).reshape(BLOCKS, 8)
    its = items(obj)
    n = int(signature["count"])
    out = {"ok": False, "count": len(its), "signed": n, "index": None, "digest": None, "match": None, "reason": None}
    if len(its) < n:
        out["reason"] = f"{n - len(its)} signed item(s) missing (deleted, or cut off the end)"
        return out
    its = its[:n]
    ds = [None if x is None else record_digest(x) for x in its]
    bad = [i for i, d in enumerate(ds) if d is None]
    ds = [d if d is not None else b"\0" * 32 for d in ds]        # an unreadable record: a leaf that is certainly wrong
    if n == 0:
        out["ok"] = True
        return out
    lv = leaves(ds)
    prefix = _fold(lv)
    if not bad and np.max(np.abs(prefix[-1] - root)) < SAME_TOL:
        out["ok"] = True
        return out
    # peel the suffix off the signed root: back[k] is what the product up to k was, if every item after k is unchanged;
    # the item k solved from it and the current prefix decodes to a content hash only where k is the one that changed
    back = np.empty_like(lv)
    iv = inv(lv)
    b = root
    for k in range(n - 1, -1, -1):
        back[k] = b
        b = mul(b, iv[k])
    before = np.concatenate([np.tile(np.array([1.0] + [0.0] * 7), (1, BLOCKS, 1)), prefix[:-1]])
    solved = mul(inv(before), back)
    pos = np.stack([_position(i) for i in range(n)])
    ok, raw = decode(mul(solved, inv(pos)))
    hits = [int(i) for i in np.nonzero(ok)[0]]
    hits = [i for i in hits if bytes(raw[i]) != ds[i][:BYTES] or i in bad]
    if len(hits) != 1:
        out["reason"] = ("several items changed, reordered, deleted or inserted: detected, not located" if not hits else
                         f"ambiguous: {len(hits)} positions fit")
        return out
    i = hits[0]
    out["index"], out["digest"] = i, bytes(raw[i]).hex()
    out["reason"] = "one item changed" + (" (unreadable)" if i in bad else "")
    for c in candidates or ():
        for form in _forms(c, its[i]):
            if record_digest(form)[:BYTES] == bytes(raw[i]):
                out["match"] = c
                break
        if out["match"] is not None:
            break
    return out


def _forms(candidate, current):
    """A candidate as it would be signed: itself, and for a trace record also the current record with its value."""
    from .runtime import Record
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
    """The one changed item → {"index", "digest": hex of its original content hash (28 bytes), "match": the candidate
    that is the original, or None}; None when nothing changed; NotLocatable as locate. A candidate is a record (a stored
    dict, a trace Record, bytes) — or, for a trace record, its original value."""
    r = check(obj, signature, candidates)
    if r["ok"]:
        return None
    if r["index"] is None:
        raise NotLocatable(r["reason"])
    return {"index": r["index"], "digest": r["digest"], "match": r["match"]}


__all__ = ["ALG", "NotLocatable", "check", "extend", "load", "locate", "repair", "sign"]
