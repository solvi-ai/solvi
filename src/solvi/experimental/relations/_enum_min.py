"""THE MINIMAL ENUMERATOR — a stand-in for the full relation-algebra enumerator of `solvi.experimental.relations`, kept
only so that `library.find_rules` runs on its own. When the two are merged, `find_rules` swaps this module for the full
one (same AST, same observational equivalence on the full relation tensor over the examples); nothing else imports it.

Relations over a batch of example worlds: each example is a mapping name → an (n, n) boolean matrix (a unary predicate
is a diagonal matrix); the batch pads them to (B, npad, npad), padding always false. The AST is nested tuples:

    ("R", name)      a base relation          ("I",)            the identity (on the real nodes)
    ("L", name)      a library entry           ("T", x)          the converse
    ("PLUS", x)      the transitive closure    ("CNT", k, x)     the nodes with at least k x-successors, as a diagonal
    ("COMP", a, b)   composition a ∘ b         ("AND"|"OR"|"MINUS", a, b)   ∩, ∪, a ∖ b

No complement and no universal relation: every answer stays as sparse as its inputs. Bottom-up by size (the node count;
a library entry is one node). Two programs are the same behaviour when their packed full relation tensors over all
examples are equal (a 16-byte hash of the packed bits): the first, smallest one stands for the class. Sound for the
search because every operator is a function of its arguments' full relations."""
from __future__ import annotations

import hashlib
from typing import Iterator, Mapping, Sequence

import numpy as np

OPS1 = ("T", "PLUS", "CNT")
OPS2 = ("COMP", "AND", "OR", "MINUS")
_COMMUTATIVE = {"AND", "OR"}
_CHUNK = 256
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


class Batch:
    """Example worlds padded into one tensor per relation: rel[name] (B, npad, npad) bool, I the identity on real
    nodes, sizes the real node counts."""

    def __init__(self, examples: Sequence[Mapping[str, np.ndarray]], names: Sequence[str]):
        if not examples:
            raise ValueError("no examples")
        self.sizes = [_n(ex, names) for ex in examples]
        self.B, self.npad = len(examples), max(self.sizes)
        self.shape = (self.B, self.npad, self.npad)
        self.rel = {}
        for name in names:
            t = np.zeros(self.shape, bool)
            for b, ex in enumerate(examples):
                m = np.asarray(ex[name], bool)
                t[b, :m.shape[0], :m.shape[1]] = m
            self.rel[name] = t
        self.I = np.zeros(self.shape, bool)
        for b, n in enumerate(self.sizes):
            self.I[b, np.arange(n), np.arange(n)] = True

    def pad(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Per-example (n, n) matrices → one padded tensor; ValueError on a wrong shape."""
        t = np.zeros(self.shape, bool)
        for b, (a, n) in enumerate(zip(arrays, self.sizes)):
            a = np.asarray(a, bool)
            if a.shape != (n, n):
                raise ValueError(f"example {b}: a target of shape {a.shape}, the world has {n} nodes")
            t[b, :n, :n] = a
        return t


def _n(ex, names):
    ns = {np.asarray(ex[x]).shape for x in names if x in ex}
    missing = [x for x in names if x not in ex]
    if missing:
        raise ValueError(f"an example has no relation {missing}")
    if len(ns) != 1 or len(next(iter(ns))) != 2 or next(iter(ns))[0] != next(iter(ns))[1]:
        raise ValueError(f"an example's relations must be square and of one size, got {ns}")
    return next(iter(ns))[0]


# ───────────────────────────────────────────── the operators
def comp(a, b):
    return np.matmul(a.astype(np.float32), b.astype(np.float32)) > 0.5


def closure(a):
    r = a
    while True:
        nxt = r | comp(r, r)
        if np.array_equal(nxt, r):
            return r
        r = nxt


def count_at_least(a, k):
    rows = a.sum(-1) >= k
    out = np.zeros(a.shape, bool)
    i = np.arange(a.shape[-1])
    out[..., i, i] = rows
    return out


def apply1(op, x, k=1):
    if op == "T":
        return np.swapaxes(x, -1, -2)
    if op == "PLUS":
        return closure(x)
    if op == "CNT":
        return count_at_least(x, k)
    raise ValueError(f"unknown operator {op!r}")


def apply2(op, a, b):
    if op == "COMP":
        return comp(a, b)
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "MINUS":
        return a & ~b
    raise ValueError(f"unknown operator {op!r}")


def evaluate(ast, batch: Batch, library: Mapping[str, tuple] | None = None, _cache: dict | None = None,
             _depth: int = 0) -> np.ndarray:
    """The program's relation on every example of the batch: (B, npad, npad) bool. Library entries resolve by name
    (KeyError for an unknown one)."""
    if _depth > 64:
        raise RecursionError("a library entry refers to itself")
    op = ast[0]
    if op == "R":
        return batch.rel[ast[1]]
    if op == "I":
        return batch.I
    if op == "L":
        name = ast[1]
        if _cache is not None and name in _cache:
            return _cache[name]
        if library is None or name not in library:
            raise KeyError(f"no library entry {name!r}")
        v = evaluate(library[name], batch, library, _cache, _depth + 1)
        if _cache is not None:
            _cache[name] = v
        return v
    if op == "CNT":
        return count_at_least(evaluate(ast[2], batch, library, _cache, _depth + 1), int(ast[1]))
    if op in OPS1:
        return apply1(op, evaluate(ast[1], batch, library, _cache, _depth + 1))
    if op in OPS2:
        return apply2(op, evaluate(ast[1], batch, library, _cache, _depth + 1),
                      evaluate(ast[2], batch, library, _cache, _depth + 1))
    raise ValueError(f"unknown node {ast!r}")


def size(ast) -> int:
    """The node count (a library entry is one node; CNT's threshold is not a node)."""
    if ast[0] in ("R", "I", "L"):
        return 1
    if ast[0] == "CNT":
        return 1 + size(ast[2])
    return 1 + sum(size(x) for x in ast[1:])


def show(ast) -> str:
    """((Eᵀ ∘ E) ∖ I), (PARENT)⁺, #≥1(x)."""
    op = ast[0]
    if op in ("R", "L"):
        return str(ast[1])
    if op == "I":
        return "I"
    if op == "T":
        inner = show(ast[1])
        return f"{inner}ᵀ" if ast[1][0] in ("R", "L", "I") else f"({inner})ᵀ"
    if op == "PLUS":
        return f"({show(ast[1])})⁺"
    if op == "CNT":
        return f"#≥{ast[1]}({show(ast[2])})"
    sym = {"COMP": "∘", "AND": "∩", "OR": "∪", "MINUS": "∖"}[op]
    return f"({show(ast[1])} {sym} {show(ast[2])})"


def _binary_block(op, pa, xa, block, shape):
    """op(a, b) for one left operand against a block of right operands (packed rows) → packed rows."""
    if op == "AND":
        return pa[None, :] & block
    if op == "OR":
        return pa[None, :] | block
    if op == "MINUS":
        return pa[None, :] & ~block
    n = int(np.prod(shape))
    xs = np.unpackbits(block, axis=1, count=n).reshape((len(block),) + tuple(shape)).astype(np.float32)
    return np.packbits((np.matmul(xa[None], xs) > 0.5).reshape(len(block), n), axis=1)


# ───────────────────────────────────────────── observational equivalence
def pack(value: np.ndarray) -> np.ndarray:
    return np.packbits(value.reshape(-1))


def unpack(packed: np.ndarray, shape) -> np.ndarray:
    return np.unpackbits(packed, count=int(np.prod(shape))).view(bool).reshape(shape)


def key(packed: np.ndarray) -> bytes:
    return hashlib.blake2b(packed.tobytes(), digest_size=16).digest()


def mismatches(packed_a: np.ndarray, packed_b: np.ndarray) -> int:
    return int(_POP[np.bitwise_xor(packed_a, packed_b)].sum())


class Enumerator:
    """Bottom-up enumeration over a batch: terminals (the base relations, I, the library entries) at size 1, then every
    unary operator over the classes of size s−1 and every binary one over pairs of sizes summing to s−1 (∩ / ∪ once per
    unordered pair). `programs` counts every program built and evaluated (the budget's unit); `classes` the distinct
    behaviours kept. `run(size, budget)` yields (ast, packed, key) for each new behaviour, in a fixed order."""

    def __init__(self, batch: Batch, base: Sequence[str], library: Sequence = (), ops1=OPS1, ops2=OPS2,
                 counts: Sequence[int] = (1,)):
        self.batch, self.base = batch, list(base)
        self.library = list(library)                   # (name, program) pairs, in admission order
        self.ops1, self.ops2, self.counts = tuple(ops1), tuple(ops2), tuple(int(k) for k in counts)
        self.programs = 0
        self.classes = 0

    def run(self, max_size: int, budget: int) -> Iterator[tuple]:
        b = self.batch
        lib = dict(self.library)
        seen: set[bytes] = set()
        levels: dict[int, list[tuple]] = {}
        cache: dict = {}

        def add(ast, value, s):
            p = pack(value)
            k = key(p)
            if k in seen:
                return None
            seen.add(k)
            levels.setdefault(s, []).append((ast, p))
            self.classes += 1
            return ast, p, k

        terms = [("R", x) for x in self.base] + [("I",)] + [("L", name) for name, _ in self.library]
        for t in terms:
            if self.programs >= budget:
                return
            self.programs += 1
            got = add(t, evaluate(t, b, lib, cache), 1)
            if got:
                yield got
        for s in range(2, max_size + 1):
            for op in self.ops1:
                for ast, p in list(levels.get(s - 1, ())):
                    for k in (self.counts if op == "CNT" else (None,)):
                        if self.programs >= budget:
                            return
                        self.programs += 1
                        x = unpack(p, b.shape)
                        if op == "CNT":
                            new, val = ("CNT", k, ast), count_at_least(x, k)
                        else:
                            new, val = (op, ast), apply1(op, x)
                        got = add(new, val, s)
                        if got:
                            yield got
            for op in self.ops2:
                for s1 in range(1, s - 1):
                    s2 = s - 1 - s1
                    if op in _COMMUTATIVE and s1 > s2:
                        continue
                    left, right = list(levels.get(s1, ())), list(levels.get(s2, ()))
                    if not left or not right:
                        continue
                    rp = np.stack([p for _, p in right])           # (R, bytes): one block per chunk below
                    for i, (a, pa) in enumerate(left):
                        j0 = i if op in _COMMUTATIVE and s1 == s2 else 0
                        xa = unpack(pa, b.shape).astype(np.float32) if op == "COMP" else None
                        for c0 in range(j0, len(right), _CHUNK):
                            room = budget - self.programs
                            if room <= 0:
                                return
                            block = rp[c0:min(c0 + _CHUNK, len(right), c0 + room)]
                            vals = _binary_block(op, pa, xa, block, b.shape)
                            for jj in range(len(block)):
                                self.programs += 1
                                got = self._add_packed((op, a, right[c0 + jj][0]), vals[jj], s, seen, levels)
                                if got:
                                    yield got

    def _add_packed(self, ast, p, s, seen, levels):
        k = key(p)
        if k in seen:
            return None
        seen.add(k)
        levels.setdefault(s, []).append((ast, p))
        self.classes += 1
        return ast, p, k


__all__ = ["Batch", "Enumerator", "OPS1", "OPS2", "evaluate", "key", "mismatches", "pack", "show", "size", "unpack"]
