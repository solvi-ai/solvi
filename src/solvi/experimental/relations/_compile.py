"""A program compiled into plain Python set code, refusing dense answers, and the differential check against the
interpreter (see the package docstring)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ._algebra import BINARY, UNARY, Relations, evaluate, show

# The helpers every compiled program calls. Relations are sets of (x, y) pairs over the nodes 0..n-1; an index maps x to
# the set of its y; composition is a join on the index; closure iterates joins of the new pairs to a fixed point.
RUNTIME = '''def _rel_index(P):
    idx = {}
    for x, y in P:
        s = idx.get(x)
        if s is None:
            idx[x] = {y}
        else:
            s.add(y)
    return idx


def _rel_comp(A, B):
    idx = _rel_index(B)
    return {(x, z) for x, y in A if y in idx for z in idx[y]}


def _rel_count(A, B, k):
    idx = _rel_index(B)
    c = {}
    for x, y in A:
        for z in idx.get(y, ()):
            c[(x, z)] = c.get((x, z), 0) + 1
    return {p for p, m in c.items() if m >= k}


def _rel_conv(A):
    return {(y, x) for x, y in A}


def _rel_plus(A):
    idx = _rel_index(A)
    total = set(A)
    delta = total
    while delta:
        delta = {(x, z) for x, y in delta if y in idx for z in idx[y]} - total
        total |= delta
    return total


def _rel_cnt(A, k):
    c = {}
    for x, _ in A:
        c[x] = c.get(x, 0) + 1
    return {(x, x) for x, m in c.items() if m >= k}


def _rel_id(n):
    return {(x, x) for x in range(n)}


def _rel_all(n):
    return {(x, y) for x in range(n) for y in range(n)}


def _rel_offdiag(A):
    return {p for p in A if p[0] != p[1]}


def _rel_diag(A):
    return {p for p in A if p[0] == p[1]}


def _rel_row0(P):
    return any(x == 0 for x, _ in P)
'''

# The same over index arrays, for large graphs: a relation is a sorted int64 array of keys x·n + y (so its pairs are
# grouped by x — an adjacency index); composition is a join by binary search with np.repeat; closure joins the new
# pairs with the base to a fixed point, deduplicating against the pairs found so far.
RUNTIME_ARRAYS = '''import numpy as _np

_E = _np.zeros(0, _np.int64)


def _ra_keys(pairs, n):
    """A set of (x, y) pairs → the sorted key array."""
    a = _np.fromiter((x * n + y for x, y in pairs), _np.int64, len(pairs))
    a.sort()
    return a


def _ra_pairs(K, n):
    """A key array → the set of (x, y) pairs."""
    xs, ys = _np.divmod(K, n)
    return set(zip(xs.tolist(), ys.tolist()))


def _ra_csr(B, n):
    """The adjacency index of B: per node, where its successors start in B and how many there are, and each pair's y."""
    cnt = _np.bincount(B // n, minlength=n)
    return _np.cumsum(cnt) - cnt, cnt, B % n


def _ra_expand(A, csr, n):
    start, count, by = csr
    ay = A % n
    cnt = count[ay]
    tot = int(cnt.sum())
    if tot == 0:
        return _E
    ends = _np.cumsum(cnt)
    idx = _np.repeat(start[ay] - (ends - cnt), cnt) + _np.arange(tot)
    return _np.repeat(A // n, cnt) * n + by[idx]


def _ra_join(A, B, n):
    if A.size == 0 or B.size == 0:
        return _E
    return _ra_expand(A, _ra_csr(B, n), n)


def _ra_comp(A, B, n):
    return _np.unique(_ra_join(A, B, n))


def _ra_count(A, B, n, k):
    u, c = _np.unique(_ra_join(A, B, n), return_counts=True)
    return u[c >= k]


def _ra_conv(A, n):
    out = (A % n) * n + A // n
    out.sort()
    return out


def _ra_plus(A, n):
    if A.size == 0:
        return A
    csr = _ra_csr(A, n)
    seen = set(A.tolist())
    parts, delta = [A], A
    while delta.size:
        fresh = set(_ra_expand(delta, csr, n).tolist())
        fresh -= seen
        seen |= fresh
        delta = _np.fromiter(fresh, _np.int64, len(fresh))
        parts.append(delta)
    out = _np.concatenate(parts)
    out.sort()
    return out


def _ra_cnt(A, n, k):
    u, c = _np.unique(A // n, return_counts=True)
    return u[c >= k] * (n + 1)


def _ra_id(n):
    return _np.arange(n, dtype=_np.int64) * (n + 1)


def _ra_all(n):
    return _np.arange(n * n, dtype=_np.int64)


def _ra_offdiag(A, n):
    return A[A // n != A % n]


def _ra_diag(A, n):
    return A[A // n == A % n]
'''


class DenseAnswer(ValueError):
    """A program whose compiled answer would be a complement over every pair of nodes (¬x or U outside ∩ / ∖ with a
    sparse side): refused unless the domain is declared bounded (`dense=True`)."""


_SETS = {"OR": "{a} | {b}", "AND": "{a} & {b}", "MINUS": "{a} - {b}", "XOR": "{a} ^ {b}", "COMP": "_rel_comp({a}, {b})",
         "C2": "_rel_count({a}, {b}, 2)", "C3": "_rel_count({a}, {b}, 3)", "T": "_rel_conv({a})",
         "PLUS": "_rel_plus({a})", "CNT1": "_rel_cnt({a}, 1)", "CNT2": "_rel_cnt({a}, 2)", "CNT3": "_rel_cnt({a}, 3)", "I": "_rel_id(n)",
         "U": "_rel_all(n)", "NOT": "_rel_all(n) - {a}", "EMPTY": "set()", "OFFDIAG": "_rel_offdiag({a})",
         "DIAG": "_rel_diag({a})", "R": "R[{a}]"}
_ARRAYS = {"OR": "_np.union1d({a}, {b})", "AND": "_np.intersect1d({a}, {b}, assume_unique=True)",
           "MINUS": "_np.setdiff1d({a}, {b}, assume_unique=True)", "XOR": "_np.setxor1d({a}, {b}, assume_unique=True)",
           "COMP": "_ra_comp({a}, {b}, n)", "C2": "_ra_count({a}, {b}, n, 2)", "C3": "_ra_count({a}, {b}, n, 3)",
           "T": "_ra_conv({a}, n)", "PLUS": "_ra_plus({a}, n)", "CNT1": "_ra_cnt({a}, n, 1)",
           "CNT2": "_ra_cnt({a}, n, 2)", "CNT3": "_ra_cnt({a}, n, 3)",
           "I": "_ra_id(n)", "U": "_ra_all(n)", "NOT": "_np.setdiff1d(_ra_all(n), {a}, assume_unique=True)",
           "EMPTY": "_E", "OFFDIAG": "_ra_offdiag({a}, n)", "DIAG": "_ra_diag({a}, n)", "R": "R[{a}]"}
BACKENDS = {"sets": (_SETS, RUNTIME), "arrays": (_ARRAYS, RUNTIME_ARRAYS)}


def _dense(node):
    return node[0] in ("NOT", "U")


class _Gen:
    def __init__(self, dense: bool, backend: str = "sets"):
        if backend not in BACKENDS:
            raise ValueError(f"backend {backend!r}: one of {list(BACKENDS)}")
        self.dense, self.lines, self.k, self.memo = dense, [], 0, {}
        self.t = BACKENDS[backend][0]

    def var(self, expr):
        if expr in self.memo:
            return self.memo[expr]
        self.k += 1
        v = f"v{self.k}"
        self.lines.append(f"    {v} = {expr}")
        self.memo[expr] = v
        return v

    def refuse(self, node, where):
        raise DenseAnswer(f"{show(node)} {where} is a complement over every pair of nodes (a dense answer): compile it "
                          f"only under ∩ or ∖ with a sparse side, or pass dense=True when the domain is bounded")

    def e(self, op, a=None, b=None):
        return self.var(self.t[op].format(a=a, b=b))

    def gen(self, node, where="at the top"):
        op = node[0]
        if op == "R":
            return self.e("R", repr(node[1]))
        if op == "I":
            return self.e("I")
        if op == "U":
            if not self.dense:
                self.refuse(node, where)
            return self.e("U")
        if op == "NOT":
            if not self.dense:
                self.refuse(node, where)
            return self.e("NOT", self.gen(node[1], "under ¬"))
        if op in UNARY:
            return self.e(op, self.gen(node[1], f"under {op}"))
        if op not in BINARY:
            raise ValueError(f"unknown operator {op!r} (expand library entries first)")
        a, b = node[1], node[2]
        if op == "AND" and (_dense(a) or _dense(b)) and not (_dense(a) and _dense(b)):
            sparse, dense = (a, b) if _dense(b) else (b, a)                     # a filter: x ∩ ¬y = x ∖ y, x ∩ U = x
            s = self.gen(sparse, "under ∩")
            return s if dense[0] == "U" else self.e("MINUS", s, self.gen(dense[1], "under ¬"))
        if op == "MINUS" and not _dense(a) and _dense(b):
            s = self.gen(a, "under ∖")
            return self.e("EMPTY") if b[0] == "U" else self.e("AND", s, self.gen(b[1], "under ¬"))
        if op in ("MINUS", "AND") and b == ("I",) and not _dense(a):            # x ∖ I, x ∩ I: a filter, no identity
            return self.e("OFFDIAG" if op == "MINUS" else "DIAG", self.gen(a, "under ∖"))
        sym = {"OR": "∪", "AND": "∩", "MINUS": "∖", "XOR": "⊕", "COMP": "∘", "C2": "#2∘", "C3": "#3∘"}[op]
        return self.e(op, self.gen(a, f"under {sym}"), self.gen(b, f"under {sym}"))


def compile_ast(ast, name: str = "rel", library=None, *, dense: bool = False, backend: str = "sets") -> str:
    """The source of `def <name>(R, n)` → the program's relation, where R maps each base relation's name to its pairs
    and n is the node count. backend "sets": pairs are sets of (x, y) tuples and the code calls the helpers in RUNTIME
    (plain Python, no import); "arrays": pairs are sorted int64 key arrays x·n + y and the helpers are in
    RUNTIME_ARRAYS (numpy) — for large graphs. Library entries are inlined. DenseAnswer: a complement or U outside ∩ / ∖
    with a sparse side (dense=True allows them, computed against all n·n pairs: only for a bounded domain such as one
    state's window)."""
    if library is not None:
        ast = library.expand(ast)
    g = _Gen(dense, backend)
    out = g.gen(ast)
    body = "\n".join(g.lines) if g.lines else "    pass"
    return f"def {name}(R, n):\n    # {show(ast)}\n{body}\n    return {out}\n"


def compile_program(ast, library=None, *, dense: bool = False, backend: str = "sets") -> Callable:
    """The compiled program as a function f(R, n) (the backend's runtime and the source executed in a fresh
    namespace): sets → a set of pairs from sets of pairs; arrays → a key array from key arrays (`to_keys`,
    `from_keys` convert)."""
    ns: dict = {}
    exec(BACKENDS[backend][1], ns)                               # noqa: S102 — our runtime source
    exec(compile_ast(ast, "_compiled", library, dense=dense, backend=backend), ns)   # noqa: S102 — generated here
    return ns["_compiled"]


_ans: dict = {}
exec(RUNTIME_ARRAYS, _ans)                                       # noqa: S102 — the source above


def to_keys(pairs, n: int) -> np.ndarray:
    """A set of (x, y) pairs → the arrays backend's sorted int64 keys x·n + y."""
    return _ans["_ra_keys"](pairs, n)


def from_keys(keys, n: int) -> set:
    """The arrays backend's keys → a set of (x, y) pairs."""
    return _ans["_ra_pairs"](np.asarray(keys, dtype=np.int64), n)


@dataclass
class ProgramCheck:
    """Compiled code against the interpreter: worlds compared, differing pairs over all of them (the bar is 0),
    pairs in the interpreter's answers."""
    worlds: int
    differences: int
    pairs: int

    @property
    def ok(self) -> bool:
        return self.differences == 0


def check_program(ast, worlds: Sequence[Relations], library=None, *, dense: bool = False,
                  backend: str = "sets") -> ProgramCheck:
    """Run the compiled program (either backend) and the interpreter on every world: the sets of pairs must be
    equal."""
    fn = compile_program(ast, library, dense=dense, backend=backend)
    diffs = pairs = 0
    for w in worlds:
        m = evaluate(ast, w, library)
        xs, ys = np.nonzero(m)
        want = set(zip(xs.tolist(), ys.tolist()))
        if backend == "arrays":
            got = from_keys(fn({k: to_keys(v, w.n) for k, v in w.sets().items()}, w.n), w.n)
        else:
            got = fn(w.sets(), w.n)
        diffs += len(got ^ want)
        pairs += len(want)
    return ProgramCheck(len(worlds), diffs, pairs)


__all__ = ["BACKENDS", "DenseAnswer", "ProgramCheck", "RUNTIME", "RUNTIME_ARRAYS", "check_program", "compile_ast",
           "compile_program", "from_keys", "to_keys"]
