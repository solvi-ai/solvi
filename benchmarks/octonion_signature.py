"""The positional octonion signature: an experiment, moved out of solvi.signature in 0.8.

    uv run python benchmarks/octonion_signature.py           # its self-check: algebra, one change located, limits

Each content hash (solvi.signature.record_digest) is written into 4 octonions (7 bytes and an anchor each) times an
element of its position; the signature is the ordered product ((L0 L1) L2)... — 32 floats. Octonions have no zero
divisors (one change always moves the product) and are alternative (a leaf is divided back out exactly); the solved leaf
decodes to a hash (28 bytes) at the changed position only. It comes from a trace-signature experiment where derivation
TREES were signed and the non-associativity saw a change of brackets. On a flat store it locates exactly as the syndrome
code of solvi.signature does, 4x larger and ~60x slower (benchmarks/trace_signature.py compares the two), so the package
keeps the syndrome code only. The position elements come from numpy's seeded normal generator, which NumPy does not
promise to keep across versions: a signature taken with one NumPy may not check with another.

sign(obj) → {"alg": "octonion", "count", "root": 32 floats}; check(obj, sig) → {"ok", "index", "digest", "reason"}."""
from __future__ import annotations

import hashlib

import numpy as np
from solvi.signature import digests

ALG = "solvi-octonion-pos/1"            # seeds the position elements of the octonion code
BLOCKS = 4                       # octonions per element: 4 x 8 = 32 numbers
BYTES = 7 * BLOCKS               # content-hash bytes written into a leaf (7 per octonion, the 8th component is the anchor)
GRID_TOL = 1e-6                  # a decoded component must lie this close to its byte grid point
SAME_TOL = 1e-8                  # two signatures within this (max abs) are the same



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


def _fold(lv, acc=None):
    """The running products ((L0 L1) L2)... of leaves [n, BLOCKS, 8] (after `acc` when given) → [n, BLOCKS, 8]."""
    out = np.empty_like(lv)
    for i in range(len(lv)):
        acc = lv[i] if acc is None else mul(acc, lv[i])
        out[i] = acc
    return out


# --- the octonion code: the positional product
def _oct_root(ds):
    return _fold(leaves(ds))[-1] if ds else np.zeros((BLOCKS, 8))


def _oct_hits(ds, bad, sig):
    n = len(ds)
    root = np.asarray(sig["root"], dtype=np.float64).reshape(BLOCKS, 8)
    lv = leaves(ds)
    prefix = _fold(lv)
    if not bad and np.max(np.abs(prefix[-1] - root)) < SAME_TOL:
        return True, []
    # peel the suffix off the signed root: back[k] is what the product up to k was, if every item after k is unchanged;
    # the item k solved from it and the current prefix decodes to a content hash only where k is the one that changed
    back = np.empty_like(lv)
    iv = inv(lv)
    b = root
    for k in range(n - 1, -1, -1):
        back[k] = b
        b = mul(b, iv[k])
    before = np.concatenate([np.tile(np.array([1.0] + [0.0] * 7), (1, BLOCKS, 1)), prefix[:-1]])
    ok, raw = decode(mul(mul(inv(before), back), inv(np.stack([_position(i) for i in range(n)]))))
    hits = [int(i) for i in np.nonzero(ok)[0]]
    return False, [(i, bytes(raw[i])) for i in hits if bytes(raw[i]) != ds[i][:BYTES] or i in bad]



def sign(obj):
    """The octonion signature of what solvi.signature.sign would sign (a trace, a Response, a store, a list)."""
    ds = digests(obj)
    if any(d is None for d in ds):
        raise ValueError("the store has unreadable records: verify() it first")
    return {"alg": "octonion", "count": len(ds), "root": [float(x) for x in _oct_root(ds).ravel()]}


def extend(signature, new_items, start=None):
    """The signature after appending `new_items`, in O(len(new_items))."""
    from solvi.signature import record_digest
    n = signature["count"] if start is None else start
    new = [record_digest(x) for x in new_items]
    acc = np.asarray(signature["root"], dtype=np.float64).reshape(BLOCKS, 8) if signature["count"] else None
    if new:
        acc = _fold(leaves(new, n), acc)[-1]
    return {"alg": "octonion", "count": n + len(new),
            "root": [float(x) for x in (acc if acc is not None else np.zeros((BLOCKS, 8))).ravel()]}


def check(obj, signature):
    """→ {"ok", "index", "digest" (hex of the first 28 bytes of the original content hash), "reason"}."""
    from solvi.signature import items, record_digest
    its = items(obj)
    n = int(signature["count"])
    out = {"ok": False, "index": None, "digest": None, "reason": None}
    if len(its) < n:
        out["reason"] = f"{n - len(its)} signed item(s) missing"
        return out
    ds = [record_digest(x) if x is not None else b"\0" * 32 for x in its[:n]]
    bad = [i for i, x in enumerate(its[:n]) if x is None]
    if n == 0:
        out["ok"] = True
        return out
    same, hits = _oct_hits(ds, bad, signature)
    if same:
        out["ok"] = True
    elif len(hits) != 1:
        out["reason"] = "several items changed, reordered, deleted or inserted: detected, not located" if not hits \
            else f"ambiguous: {len(hits)} positions fit"
    else:
        out["index"], out["digest"], out["reason"] = hits[0][0], hits[0][1].hex(), "one item changed"
    return out


def _self_check():
    import random
    rng = np.random.default_rng(0)
    a, b, c = (rng.normal(size=(500, 4, 8)) for _ in range(3))
    n = lambda x: np.linalg.norm(x, axis=-1)  # noqa: E731
    assert np.allclose(n(mul(a, b)), n(a) * n(b))                        # a composition algebra: no zero divisors
    assert np.allclose(mul(mul(a, b), b), mul(a, mul(b, b)))              # alternative
    assert not np.allclose(mul(mul(a, b), c), mul(a, mul(b, c)))          # not associative
    assert np.allclose(mul(mul(a, b), inv(b)), a) and np.allclose(mul(inv(a), mul(a, b)), b)
    r = random.Random(1)
    located = 0
    for _ in range(200):
        its = [r.randbytes(r.randint(1, 60)) for _ in range(r.randint(1, 200))]
        sig = sign(its)
        assert check(its, sig)["ok"]
        k = r.randrange(len(its))
        orig, its[k] = its[k], r.randbytes(20)
        res = check(its, sig)
        located += res["index"] == k and res["digest"] == hashlib.sha256(orig).digest()[:BYTES].hex()
    more = [r.randbytes(9) for _ in range(30)]
    assert np.allclose(extend(sign(more[:20]), more[20:])["root"], sign(more)["root"])
    print(f"algebra ok; one change located and its hash restored: {located}/200")
    assert located == 200


if __name__ == "__main__":
    _self_check()
