"""The bottom-up enumerator with observational equivalence, exact synthesis, the readout score and the top programs,
and the library of accepted programs (see the package docstring)."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from ._algebra import (BINARY, COMMUTATIVE, OPS, UNARY, _Batch, _eval, apply_unary, readout, show)


# ───────────────────────────────────────────── the library
@dataclass
class Entry:
    """An accepted program kept as a new terminal: the program (a tuple), its name, the examples it was accepted on
    (ids or labels) and its test record (whatever the test reported: support, precision, held-out groups, ...)."""
    program: tuple
    name: str
    examples: tuple = ()
    test: dict | None = None


class Library:
    """Accepted programs, in order of admission; each is a terminal ("L", name) of later enumerations, so a composite
    rule becomes a small program over earlier ones. An entry may use earlier entries, never itself."""

    def __init__(self, entries: Iterable[Entry] = ()):
        self._entries: dict[str, Entry] = {}
        for e in entries:
            self.add(e.program, e.name, examples=e.examples, test=e.test)

    def add(self, program: tuple, name: str | None = None, *, examples: Iterable = (), test: dict | None = None) -> Entry:
        name = str(name) if name is not None else show(program)
        if name in self._entries:
            raise ValueError(f"the library already has {name!r}")
        for ref in _refs(program):
            if ref not in self._entries:
                raise ValueError(f"{name!r} uses {ref!r}, which is not in the library (yet)")
        e = Entry(tuple(program), name, tuple(examples), test)
        self._entries[name] = e
        return e

    def __contains__(self, name):
        return name in self._entries

    def __getitem__(self, name) -> Entry:
        return self._entries[name]

    def __iter__(self):
        return iter(list(self._entries.values()))

    def __len__(self):
        return len(self._entries)

    def names(self) -> list[str]:
        return list(self._entries)

    def expand(self, ast):
        """The program with every library entry replaced by its own program (recursively)."""
        if ast[0] == "L":
            return self.expand(self._entries[ast[1]].program)
        if ast[0] in ("R", "I", "U"):
            return ast
        return (ast[0],) + tuple(self.expand(x) for x in ast[1:])

    def __repr__(self):
        return f"Library({self.names()})"


def _refs(ast):
    if ast[0] == "L":
        yield ast[1]
    elif ast[0] not in ("R", "I", "U"):
        for x in ast[1:]:
            yield from _refs(x)


def _library(library) -> Library:
    if isinstance(library, Library):
        return library
    return Library(library or ())


# ───────────────────────────────────────────── the enumerator
class Program(NamedTuple):
    """One behaviour class: its first (smallest, earliest) program, its size and its value on every example, bool
    (B, n, n) padded."""
    ast: tuple
    size: int
    value: np.ndarray


def _digest(row) -> bytes:
    return hashlib.blake2b(row, digest_size=16).digest()


class Enumerator:
    """Programs bottom-up by size (size 1: the base relations, I, U, then the library entries; size s: every unary
    operator over size s−1, then every binary operator over sizes s1 + s2 = s−1, commutative ones once), keeping one
    program per behaviour — two programs are the same when their full relations on every example are the same
    (packed bits, a 16-byte hash), never when only a readout agrees. Deterministic: the order is fixed. Iterate it for
    the new behaviours in order; `evals`, `behaviours`, `stopped` ("exhausted" | "cap" | "found" | "seconds") after.

    base: the examples (a sequence of Relations with the same relation names). ops: the operators to use (default
    all but CNT1: T, NOT, PLUS, CNT2, CNT3, OR, AND, MINUS, XOR, COMP, C2, C3; CNT1 — at least one successor — is
    opt-in). max_behaviours / max_evals / seconds: caps;
    max_bytes (default 1.5 GB): the store of behaviours (B·n² bits each, packed, plus a 16-byte key) is capped to fit —
    with many or large examples search on a subset and re-check on all (synthesize(check=...)). include_universal=False
    leaves U out of the terminals (opt-in; the default keeps it).

    Equivalence is on the examples only: two programs that agree on every example are one class, represented by the
    first (smallest) found, even if they differ elsewhere — so a find must be checked on held-out examples of other
    sizes and families (LawSet.test does that for the proposer). The order of terminals and the one-sided commutative
    operators only decide which equivalent program represents a class. A complement is taken within the real pairs of
    each example (padding stays empty)."""

    def __init__(self, base, size: int = 5, library=(), *, ops: Sequence[str] = OPS, names: Sequence[str] | None = None,
                 max_behaviours: int | None = None, max_evals: int | None = None, seconds: float | None = None,
                 max_bytes: int | None = 1_500_000_000, chunk: int = 2048, include_universal: bool = True):
        self.batch = base if isinstance(base, _Batch) else _Batch(base, names)
        # each kept behaviour stores B·n² bits (packed) and a 16-byte key: cap the count so the store fits max_bytes
        self.bytes_per_behaviour = (self.batch.B * self.batch.n * self.batch.n + 7) // 8 + 16 + 120
        if max_bytes is not None:
            fit = max(1, int(max_bytes) // self.bytes_per_behaviour)
            max_behaviours = fit if max_behaviours is None else min(max_behaviours, fit)
        self.size, self.library = int(size), _library(library)
        bad = [o for o in ops if o not in UNARY + BINARY]
        if bad:
            raise ValueError(f"unknown operators {bad} (have {list(UNARY + BINARY)})")
        self.unary = [o for o in UNARY if o in ops]
        self.binary = [o for o in BINARY if o in ops]
        self.max_behaviours, self.max_evals, self.seconds = max_behaviours, max_evals, seconds
        self.chunk = max(1, int(chunk))
        self.include_universal = bool(include_universal)
        self.evals = self.behaviours = 0
        self.by_size: dict[int, int] = {}
        self.stopped: str | None = None
        self.found: tuple | None = None
        self._target: tuple[bytes, np.ndarray] | None = None
        self._seen: set[bytes] = set()
        self._asts: dict[int, list] = {}
        self._packed: dict[int, list] = {}
        self._t0 = 0.0

    # storage
    def _unpack(self, s, lo, hi):
        b = self.batch
        arr = self._packed[s]
        return np.unpackbits(arr[lo:hi], axis=1, count=b.B * b.n * b.n).reshape(-1, b.B, b.n, b.n).view(bool)

    def _finish(self, s):
        parts = self._packed.get(s, [])
        nb = (self.batch.B * self.batch.n * self.batch.n + 7) // 8
        self._packed[s] = np.concatenate(parts) if parts else np.zeros((0, nb), np.uint8)
        self.by_size[s] = len(self._asts.get(s, ()))

    def _offer(self, asts, vals, s):
        C = len(asts)
        packed = np.packbits(vals.reshape(C, -1), axis=1)
        keep = []
        try:
            for j in range(C):
                self.evals += 1
                key = _digest(packed[j])
                if key not in self._seen:
                    self._seen.add(key)
                    keep.append(j)
                    self.behaviours += 1
                    if self._target is not None and key == self._target[0] and np.array_equal(vals[j], self._target[1]):
                        self.found, self.stopped = asts[j], "found"
                    yield Program(asts[j], s, vals[j])
                    if self.stopped:
                        return
                if self._capped():
                    return
        finally:
            if keep:
                self._asts.setdefault(s, []).extend(asts[j] for j in keep)
                self._packed.setdefault(s, []).append(packed[keep])

    def _capped(self):
        if self.max_behaviours is not None and self.behaviours >= self.max_behaviours:
            self.stopped = "cap"
        elif self.max_evals is not None and self.evals >= self.max_evals:
            self.stopped = "cap"
        elif self.seconds is not None and time.monotonic() - self._t0 > self.seconds:
            self.stopped = "seconds"
        return self.stopped is not None

    def __iter__(self):
        return self._run()

    def _run(self):
        self._t0 = time.monotonic()
        b, M = self.batch, self.batch.M
        universal = [("U",)] if self.include_universal else []
        terms = [("R", nm) for nm in b.names] + [("I",)] + universal + [("L", e.name) for e in self.library]
        vals = np.stack([_eval(t, b, self.library) for t in terms])
        yield from self._offer(terms, vals, 1)
        self._finish(1)
        for s in range(2, self.size + 1):
            if self.stopped:
                return
            # unary operators over size s−1, in order of the operand, then of the operator
            prev = self._asts.get(s - 1, [])
            if self.unary:
                for lo in range(0, len(prev), self.chunk):
                    hi = min(len(prev), lo + self.chunk)
                    V = self._unpack(s - 1, lo, hi)
                    outs = np.stack([apply_unary(op, V, M) for op in self.unary], axis=1).reshape(-1, b.B, b.n, b.n)
                    asts = [(op, prev[i]) for i in range(lo, hi) for op in self.unary]
                    yield from self._offer(asts, outs, s)
                    if self.stopped:
                        return
            # binary operators over sizes s1 + s2 = s − 1
            for s1 in range(1, s - 1):
                s2 = s - 1 - s1
                ops = [op for op in self.binary if not (op in COMMUTATIVE and s1 > s2)]
                left, right = self._asts.get(s1, []), self._asts.get(s2, [])
                if not ops or not left or not right:
                    continue
                counts_needed = any(op in ("COMP", "C2", "C3") for op in ops)
                n2 = len(right)
                if n2 <= self.chunk:
                    step, spans = max(1, self.chunk // n2), [(0, n2)]
                    cached = self._unpack(s2, 0, n2)
                else:
                    step, spans, cached = 1, [(lo, min(n2, lo + self.chunk)) for lo in range(0, n2, self.chunk)], None
                for lo1 in range(0, len(left), step):
                    hi1 = min(len(left), lo1 + step)
                    A = self._unpack(s1, lo1, hi1)[:, None]
                    for lo2, hi2 in spans:
                        Bv = (cached if cached is not None else self._unpack(s2, lo2, hi2))[None]
                        cnt = np.matmul(A.astype(np.float32), Bv.astype(np.float32)) if counts_needed else None
                        outs = _binary_block(ops, A, Bv, M, cnt).reshape(-1, b.B, b.n, b.n)
                        asts = [(op, left[i], right[j]) for i in range(lo1, hi1) for j in range(lo2, hi2) for op in ops]
                        yield from self._offer(asts, outs, s)
                        if self.stopped:
                            return
            self._finish(s)
        self.stopped = self.stopped or "exhausted"


_UFUNC = {"OR": np.bitwise_or, "AND": np.bitwise_and, "MINUS": np.greater, "XOR": np.not_equal}
_AT_LEAST = {"COMP": 1, "C2": 2, "C3": 3}


def _binary_block(ops, A, Bv, M, cnt):
    """Every operator of `ops` on every pair of a block, written in place: bool (a, c, len(ops), B, n, n)."""
    shape = np.broadcast_shapes(A.shape, Bv.shape)
    out = np.empty(shape[:2] + (len(ops),) + shape[2:], dtype=bool)
    for i, op in enumerate(ops):
        o = out[:, :, i]
        if op in _UFUNC:
            _UFUNC[op](A, Bv, out=o)                          # a & ~b is a > b on booleans
        else:
            np.greater_equal(cnt, _AT_LEAST[op], out=o)
            np.bitwise_and(o, M, out=o)
    return out


def enumerate_programs(base, size: int = 5, library=(), **kw) -> Enumerator:
    """An Enumerator over the examples `base` (see Enumerator): iterate it for Program(ast, size, value) in order."""
    return Enumerator(base, size, library, **kw)


class Found(NamedTuple):
    """The first program whose relation equals the target on every example (None if none within the size or caps),
    with the evaluations and the distinct behaviours the search took; by: the search that found it (or ran last) —
    "enumerate" or "residual"; steps: residual steps spent (`method="residual-first"` / "residual"; another unit than
    evaluations)."""
    ast: tuple | None
    evals: int
    behaviours: int
    stopped: str
    by: str = "enumerate"
    steps: int = 0


METHODS = ("enumerate", "residual-first", "residual")


def synthesize(base, target, size: int = 7, library=(), *, check=None, method: str = "enumerate",
               residual_budget: int = 100_000, **kw) -> Found:
    """The first (smallest, in enumeration order) program equal to `target` — per-example (n_b, n_b) matrices or one
    padded (B, n, n) array — on every example: exact equality, byte for byte. check=(examples, targets): search on
    `base` (e.g. a subset, to bound memory) and accept a match only if it is also exact on these; a failed re-check
    continues the search.

    method: "enumerate" (default: the bottom-up enumerator), "residual-first" (`synthesize_residual` first — chains of
    at most `size` atoms, `residual_budget` steps, the same `check`; the enumerator only when it finds nothing) or
    "residual" (residuals only). Residuals use converses and ∖ I only when "T" and "MINUS" are among `ops`, and run
    only when "COMP" is; `seconds` caps each search."""
    if method not in METHODS:
        raise ValueError(f"method is one of {METHODS}, not {method!r}")
    steps = 0
    if method != "enumerate":
        from ._residual import synthesize_residual
        ops = kw.get("ops", OPS)
        if "COMP" in ops:
            r = synthesize_residual(base, target, depth=size, budget=residual_budget, seconds=kw.get("seconds"),
                                    library=library, names=kw.get("names"), check=check, converses="T" in ops,
                                    masks="MINUS" in ops)
            if r.program is not None or method == "residual":
                return Found(r.program, 0, 0, r.stopped, "residual", r.steps)
            steps = r.steps
        elif method == "residual":
            return Found(None, 0, 0, "exhausted", "residual", 0)
    en = Enumerator(base, size, library, **kw)
    t = en.batch.pad(target)
    en._target = (_digest(np.packbits(t.reshape(-1))), t)
    full = None
    if check is not None:
        fb = check[0] if isinstance(check[0], _Batch) else _Batch(check[0], en.batch.names)
        full = (fb, fb.pad(check[1]))
    for _ in en:
        if en.found is not None:
            if full is None or np.array_equal(_eval(en.found, full[0], en.library), full[1]):
                break
            en.found, en.stopped, en.rechecked = None, None, getattr(en, "rechecked", 0) + 1
    return Found(en.found, en.evals, en.behaviours, en.stopped or "exhausted", "enumerate", steps)


# ───────────────────────────────────────────── scoring a readout
class Score(NamedTuple):
    """Of a program's readout against the target bits: on how many examples it is true (support), on how many of
    those the target is true too (hits), hits / support (0.0 without support)."""
    support: int
    hits: int
    precision: float


def _score(r, y) -> Score:
    sup = int(r.sum())
    hits = int((r & y).sum())
    return Score(sup, hits, hits / sup if sup else 0.0)


def score(ast, examples, y, library=None) -> Score:
    """Support and precision of the program's readout (node 0 relates to something) against the bits `y`, one per
    example."""
    b = examples if isinstance(examples, _Batch) else _Batch(examples)
    return _score(readout(_eval(ast, b, _library(library))), np.asarray(y, dtype=bool))


def lower_bound(hits: int, support: int, z: float = 1.645) -> float:
    """The Wilson lower bound of the precision: what ranks programs (a high precision on a large support first)."""
    if support == 0:
        return 0.0
    p = hits / support
    d = 1 + z * z / support
    return (p + z * z / (2 * support) - z * math.sqrt(p * (1 - p) / support + z * z / (4 * support * support))) / d


@dataclass
class Scored:
    """A ranked program: its tuple, its score, its readout (one bit per example) and its place in the enumeration."""
    ast: tuple
    score: Score
    readout: np.ndarray = field(repr=False)
    order: int = 0


def rank(en: Enumerator, ys: Sequence, *, k: int = 20, per_readout: int = 1, must: int | None = None,
         min_support: int = 1) -> list[list[Scored]]:
    """One pass over an enumeration, scored against each of several target vectors: per target the top `k`
    programs by the lower bound of the precision (then support, then enumeration order), at most `per_readout` per
    readout pattern (the earliest, i.e. smallest, ones), each true on example `must` if given. Every behaviour is
    scored: the search does not stop at the first exact match."""
    Y = [np.asarray(y, dtype=bool) for y in ys]
    kept: list[dict[bytes, list[Scored]]] = [{} for _ in Y]
    for i, p in enumerate(en):
        r = readout(p.value)
        if must is not None and not r[must]:
            continue
        sup = int(r.sum())
        if sup < min_support:
            continue
        key = np.packbits(r).tobytes()
        for t, y in enumerate(Y):
            lst = kept[t].setdefault(key, [])
            if len(lst) < per_readout:
                lst.append(Scored(p.ast, _score(r, y), r, i))
    out = []
    for t in range(len(Y)):
        allx = [s for lst in kept[t].values() for s in lst]
        allx.sort(key=lambda s: (-lower_bound(s.score.hits, s.score.support), -s.score.support, s.order))
        out.append(allx[:k])
    return out


def top_programs(examples, y, *, size: int = 4, k: int = 20, library=(), ops: Sequence[str] = OPS,
                 per_readout: int = 1, must: int | None = None, min_support: int = 1, **kw) -> list[Scored]:
    """The top `k` programs of size ≤ `size` for the target bits `y` (see `rank`): noise-tolerant — ranked by the
    lower bound of the precision of the readout, never stopped at the first exact match."""
    en = Enumerator(examples, size, library, ops=ops, **kw)
    return rank(en, [y], k=k, per_readout=per_readout, must=must, min_support=min_support)[0]


__all__ = ["METHODS", "Entry", "Enumerator", "Found", "Library", "Program", "Score", "Scored", "enumerate_programs",
           "lower_bound", "rank", "score", "synthesize", "top_programs"]
