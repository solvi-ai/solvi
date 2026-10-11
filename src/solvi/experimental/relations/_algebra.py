"""Relations over a state as packed-bit matrices, the relation algebra on boolean matrices, programs as tuples (show,
size, evaluate) and the readout of a relation as a state bit (see the package docstring).

Operators: UNARY are all the unary operators the algebra knows; OPS is the default search. CNT1 (the nodes with at
least one successor, as a diagonal) is opt-in: it evaluates, enumerates (ops=...) and compiles like CNT2 / CNT3, but it
is not in OPS, so the default search space is the one the battery was measured on."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

UNARY = ("T", "NOT", "PLUS", "CNT1", "CNT2", "CNT3")
BINARY = ("OR", "AND", "MINUS", "XOR", "COMP", "C2", "C3")
COMMUTATIVE = frozenset({"OR", "AND", "XOR"})
OPS = tuple(o for o in UNARY if o != "CNT1") + BINARY       # the default search; CNT1 (has a successor) is opt-in
SYM = {"OR": "∪", "AND": "∩", "MINUS": "∖", "XOR": "⊕", "COMP": "∘", "C2": "#2∘", "C3": "#3∘"}


# ───────────────────────────────────────────── relations over one state
class Relations:
    """Named binary relations over the nodes 0..n-1 of one state (node 0: the subject — the player, the path acted on),
    stored as packed bits: `bits` is uint8 (K, n, ⌈n/8⌉), row x of relation k packs the y with (x, y) in it. A unary
    predicate is a diagonal relation (`mark`). `nodes`: optional labels of the nodes (e.g. paths)."""

    def __init__(self, names: Iterable[str], n: int, nodes: Sequence | None = None):
        self.names = tuple(str(x) for x in names)
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"two relations share a name: {self.names}")
        self.n = int(n)
        self.index = {nm: k for k, nm in enumerate(self.names)}
        self.bits = np.zeros((len(self.names), self.n, (self.n + 7) // 8), dtype=np.uint8)
        self.nodes = list(nodes) if nodes is not None else list(range(self.n))

    @classmethod
    def from_dense(cls, names: Iterable[str], dense, nodes: Sequence | None = None) -> "Relations":
        dense = np.asarray(dense, dtype=bool)
        r = cls(names, dense.shape[-1], nodes)
        if dense.shape != (len(r.names), r.n, r.n):
            raise ValueError(f"dense relations of shape {dense.shape}, expected {(len(r.names), r.n, r.n)}")
        r.bits = np.packbits(dense, axis=-1)
        return r

    @classmethod
    def from_sets(cls, names: Iterable[str], sets: dict, n: int, nodes: Sequence | None = None) -> "Relations":
        r = cls(names, n, nodes)
        for nm, pairs in sets.items():
            r.update(nm, pairs)
        return r

    def _k(self, name):
        try:
            return self.index[name]
        except KeyError:
            raise KeyError(f"no relation {name!r} (have {list(self.names)})") from None

    def add(self, name: str, x: int, y: int):
        self.bits[self._k(name), x, y >> 3] |= np.uint8(0x80 >> (y & 7))

    def update(self, name: str, pairs: Iterable[tuple[int, int]]):
        k = self._k(name)
        for x, y in pairs:
            self.bits[k, x, y >> 3] |= np.uint8(0x80 >> (y & 7))

    def mark(self, name: str, nodes: Iterable[int]):
        """A unary predicate: the diagonal pairs (v, v)."""
        self.update(name, ((v, v) for v in nodes))

    def set(self, name: str, matrix):
        m = np.asarray(matrix, dtype=bool)
        if m.shape != (self.n, self.n):
            raise ValueError(f"a {m.shape} matrix for {self.n} nodes")
        self.bits[self._k(name)] = np.packbits(m, axis=-1)

    def matrix(self, name: str) -> np.ndarray:
        return np.unpackbits(self.bits[self._k(name)], axis=-1, count=self.n).astype(bool)

    __getitem__ = matrix

    def dense(self) -> np.ndarray:
        """bool (K, n, n)."""
        return np.unpackbits(self.bits, axis=-1, count=self.n).astype(bool)

    def pairs(self, name: str) -> set[tuple[int, int]]:
        xs, ys = np.nonzero(self.matrix(name))
        return set(zip(xs.tolist(), ys.tolist()))

    def sets(self) -> dict[str, set[tuple[int, int]]]:
        """{name: set of pairs}: the input of compiled set code."""
        return {nm: self.pairs(nm) for nm in self.names}

    def __repr__(self):
        return f"Relations({list(self.names)}, n={self.n})"


# ───────────────────────────────────────────── the algebra on bool matrices (any leading batch dimensions)
def _count(a, b):
    return np.matmul(a.astype(np.float32), b.astype(np.float32))


def compose(a, b, k: int = 1, mask=None):
    """(x, z) with at least k paths x→y→z (k=1: the composition a∘b)."""
    out = _count(a, b) >= k
    return out & mask if mask is not None else out


def converse(a):
    return np.swapaxes(a, -1, -2)


def union(a, b):
    return a | b


def intersect(a, b):
    return a & b


def minus(a, b):
    return a & ~b


def xor(a, b):
    return a ^ b


def complement(a, mask=None):
    """The complement bounded by `mask` (the real pairs; default: every pair of the matrix)."""
    return ~a & mask if mask is not None else ~a


def closure(a, mask=None):
    """The transitive closure a⁺ (paths of length ≥ 1): Warshall on row bitmasks up to 64 nodes, else squaring to a
    fixed point."""
    r = np.asarray(a, dtype=bool)
    n = r.shape[-1]
    if 0 < n <= 64:
        width = next(w for w in (1, 2, 4, 8) if 8 * w >= n)
        dt = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
        packed = np.packbits(r, axis=-1, bitorder="little")
        if packed.shape[-1] < width:
            packed = np.concatenate([packed, np.zeros(packed.shape[:-1] + (width - packed.shape[-1],), np.uint8)], -1)
        rows = np.ascontiguousarray(packed).view(dt)[..., 0]           # (..., n): row x as a bitmask of its y
        one = dt(1)
        for k in range(n):
            rows |= ((rows >> dt(k)) & one) * rows[..., k:k + 1]      # x→k: x reaches what k reaches
        out = np.unpackbits(rows[..., None].view(np.uint8), axis=-1, count=n, bitorder="little").astype(bool)
        return out & mask if mask is not None else out
    while True:
        nxt = r | compose(r, r, mask=mask)
        if np.array_equal(nxt, r):
            return r
        r = nxt


def count_at_least(a, k: int, mask=None):
    """A unary from a relation: the diagonal relation of the nodes with at least k successors in `a`."""
    n = a.shape[-1]
    rows = a.sum(-1) >= k
    out = rows[..., :, None] & np.eye(n, dtype=bool)
    return out & mask if mask is not None else out


# ───────────────────────────────────────────── programs
def show(ast) -> str:
    """A program as text: E0, (E0 ∘ E1), (E0)⁺, E2ᵀ, ¬E1, #≥2(E0), (E0 #2∘ E1)."""
    op = ast[0]
    if op in ("R", "L"):
        return str(ast[1])
    if op in ("I", "U"):
        return op
    if op == "T":
        return show(ast[1]) + "ᵀ"
    if op == "NOT":
        return "¬" + show(ast[1])
    if op == "PLUS":
        return f"({show(ast[1])})⁺"
    if op in ("CNT1", "CNT2", "CNT3"):
        return f"#≥{op[-1]}({show(ast[1])})"
    return f"({show(ast[1])} {SYM[op]} {show(ast[2])})"


def size(ast) -> int:
    """The number of nodes of a program (a terminal or a library entry counts 1)."""
    if ast[0] in ("R", "L", "I", "U"):
        return 1
    return 1 + sum(size(x) for x in ast[1:])


class _Batch:
    """Examples padded to one node count: E bool (K, B, n, n), M (B, n, n) the real pairs, I the identity on them."""

    def __init__(self, examples: Sequence[Relations], names: Sequence[str] | None = None):
        examples = list(examples)
        if not examples:
            raise ValueError("no examples")
        self.names = tuple(names) if names is not None else examples[0].names
        for r in examples:
            if r.names != self.names and not set(self.names) <= set(r.names):
                raise ValueError(f"an example has relations {r.names}, expected {self.names}")
        self.sizes = [r.n for r in examples]
        self.B, self.n = len(examples), max(max(self.sizes), 1)
        self.E = np.zeros((len(self.names), self.B, self.n, self.n), dtype=bool)
        self.M = np.zeros((self.B, self.n, self.n), dtype=bool)
        for b, r in enumerate(examples):
            d = r.dense()
            for k, nm in enumerate(self.names):
                self.E[k, b, :r.n, :r.n] = d[r.index[nm]]
            self.M[b, :r.n, :r.n] = True
        self.I = np.eye(self.n, dtype=bool)[None] & self.M
        self.index = {nm: k for k, nm in enumerate(self.names)}

    def pad(self, values) -> np.ndarray:
        """Per-example (n_b, n_b) matrices (or one padded array) → bool (B, n, n), masked."""
        if isinstance(values, np.ndarray) and values.ndim == 3:
            if values.shape != (self.B, self.n, self.n):
                raise ValueError(f"target of shape {values.shape}, expected {(self.B, self.n, self.n)}")
            return values.astype(bool) & self.M
        out = np.zeros((self.B, self.n, self.n), dtype=bool)
        for b, v in enumerate(values):
            v = np.asarray(v, dtype=bool)
            out[b, :v.shape[0], :v.shape[1]] = v
        return out & self.M

    def unpad(self, value) -> list[np.ndarray]:
        return [value[b, :m, :m] for b, m in enumerate(self.sizes)]


def apply_unary(op, v, M):
    """One unary operator on bool (..., n, n) values with the real-pair mask M."""
    if op == "T":
        return converse(v)
    if op == "NOT":
        return ~v & M
    if op == "PLUS":
        return closure(v, mask=M)
    if op in ("CNT1", "CNT2", "CNT3"):
        return count_at_least(v, int(op[-1]), mask=M)
    raise ValueError(f"unknown unary operator {op!r}")


def apply_binary(op, a, b, M, counts=None):
    """One binary operator; `counts`: the path counts a@b when already computed."""
    if op == "OR":
        return a | b
    if op == "AND":
        return a & b
    if op == "MINUS":
        return a & ~b
    if op == "XOR":
        return a ^ b
    if op in ("COMP", "C2", "C3"):
        c = counts if counts is not None else _count(a, b)
        return (c >= {"COMP": 1, "C2": 2, "C3": 3}[op]) & M
    raise ValueError(f"unknown binary operator {op!r}")


def _eval(ast, batch: _Batch, library=None, _stack=()):
    op = ast[0]
    if op == "R":
        try:
            return batch.E[batch.index[ast[1]]]
        except KeyError:
            raise KeyError(f"no relation {ast[1]!r} in the examples (have {list(batch.names)})") from None
    if op == "I":
        return batch.I
    if op == "U":
        return batch.M
    if op == "L":
        if library is None or ast[1] not in library:
            raise KeyError(f"the library has no entry {ast[1]!r}")
        if ast[1] in _stack:
            raise ValueError(f"the library entry {ast[1]!r} refers to itself")
        return _eval(library[ast[1]].program, batch, library, _stack + (ast[1],))
    if op in UNARY:
        return apply_unary(op, _eval(ast[1], batch, library, _stack), batch.M)
    if op in BINARY:
        return apply_binary(op, _eval(ast[1], batch, library, _stack), _eval(ast[2], batch, library, _stack), batch.M)
    raise ValueError(f"unknown operator {op!r} in {ast!r}")


def evaluate(ast, rel, library=None):
    """A program on one `Relations` → bool (n, n); on a list of them → a list of per-example matrices."""
    if isinstance(rel, Relations):
        b = _Batch([rel])
        return _eval(ast, b, library)[0, :rel.n, :rel.n].copy()
    b = _Batch(rel)
    return b.unpad(_eval(ast, b, library))


def readout(value) -> bool | np.ndarray:
    """A relation read as a state bit: does node 0 relate to anything? bool for one (n, n) matrix, a bool array for a
    batch (B, n, n)."""
    v = np.asarray(value, dtype=bool)
    if v.ndim == 2:
        return bool(v[0].any()) if v.size else False
    return np.asarray(v[..., 0, :].any(-1), dtype=bool)


__all__ = ["BINARY", "COMMUTATIVE", "OPS", "Relations", "UNARY", "closure", "complement", "compose", "converse",
           "count_at_least", "evaluate", "intersect", "minus", "readout", "show", "size", "union", "xor"]
