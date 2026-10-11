"""Top-down synthesis of a relation program through residuals of the target (see `synthesize_residual`).

The enumerator builds programs bottom-up and their number grows exponentially with the size. This search starts from
the target instead. A subgoal is an interval Lo ⊆ X ⊆ Up on a mask C of the pairs that matter. The step X = A ∘ Y (A an
atom: a base relation, a library entry, or the converse of one) turns it into a subgoal for Y: the largest Y that keeps
A ∘ Y inside Up is the residual A \\ Up = ¬(Aᵀ ∘ ¬Up); a pair of Lo with exactly one witness k forces (k, y) into Y; a
pair of Lo with no witness left prunes the step at once. X = Y ∘ A is the mirror, and X = Y ∖ I drops the diagonal.
Failed subgoals are remembered by a hash of (Lo, Up, C, atoms left). Iterative deepening on the number of atoms, so the
shortest chain comes first. Every top-level candidate is checked exactly by the package's evaluator on the examples,
then on `check` worlds and by a `test` (a HeldOut) — a candidate that fails is counted (`rejected`) and the search goes
on, so nothing reaches the result that the checks did not pass.

The grammar is chains: compositions of atoms and their converses, peeled one atom at a time from either end, with ∖ I
on any segment of two atoms or more (cousin = Eᵀ ∘ ((Eᵀ ∘ E) ∖ I) ∘ E). Two masked segments side by side,
((…) ∖ I) ∘ ((…) ∖ I), have no bare atom at either end and are outside — unless one of them is a library entry, which
is an atom. Union, intersection, closure and counting are not covered; for those the enumerator stays the search
(`synthesize(method="residual-first")` runs it on what the residuals do not find). "exhausted" means: no program of
this grammar with at most `depth` atoms equals the target on the examples. Ported from the author's earlier
research line (a numpy module next to this package), with the package's own evaluator and checks."""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Mapping, NamedTuple, Sequence

import numpy as np

from ._algebra import Relations, _Batch, _eval


class ResidualFound(NamedTuple):
    """What synthesize_residual returns. program: the package's program tuple, or None; steps: residual steps taken
    (one per atom tried at a subgoal — not the enumerator's programs: another unit); seconds: wall time; atoms: the
    program's number of atoms (None when nothing was found); stopped: "found" | "budget" | "seconds" | "exhausted"
    (every chain up to `depth` atoms was ruled out); rejected: candidates equal to the target on the examples that
    failed the `check` worlds or the `test`."""
    program: tuple | None
    steps: int
    seconds: float
    atoms: int | None
    stopped: str
    rejected: int


def _is_world(x) -> bool:
    return isinstance(x, Mapping)


def _square_names(world) -> list[str]:
    return sorted(k for k, v in world.items() if np.ndim(v) == 2 and np.shape(v)[0] == np.shape(v)[1])


def _as_batch(base, names):
    """Relations, a _Batch or worlds (name → (n, n) bool) → (the package's batch, the worlds or None)."""
    if isinstance(base, _Batch):
        return base, None
    base = list(base)
    if not base:
        raise ValueError("no examples")
    if _is_world(base[0]):
        names = list(names) if names is not None else _square_names(base[0])
        rels = []
        for b, w in enumerate(base):
            missing = [k for k in names if k not in w]
            if missing:
                raise ValueError(f"example {b} has no relation {missing}")
            rels.append(Relations.from_dense(names, np.stack([np.asarray(w[k], bool) for k in names])))
        return _Batch(rels, names), base
    if not all(isinstance(r, Relations) for r in base):
        raise TypeError("examples are Relations or worlds (a mapping name → (n, n) boolean matrix)")
    return _Batch(base, names), None


def _targets(target, batch, worlds):
    if callable(target):
        if worlds is None:
            raise TypeError("a target given as a function needs worlds (mappings) as the examples")
        target = [np.asarray(target(w), bool) for w in worlds]
    if isinstance(target, np.ndarray) and target.ndim == 3:
        return batch.pad(target)
    target = list(target)
    if len(target) != batch.B:
        raise ValueError(f"{len(target)} target matrices for {batch.B} examples")
    for b, (t, n) in enumerate(zip(target, batch.sizes)):
        if np.shape(t) != (n, n):
            raise ValueError(f"example {b}: a target of shape {np.shape(t)}, the example has {n} nodes")
    return batch.pad(target)


def _library(library):
    """A Library or a sequence of Entry → (the entries in order, a name → entry mapping the evaluator reads)."""
    entries = list(library or ())
    by_name = {}
    for e in entries:
        if e.name in by_name:
            raise ValueError(f"two library entries are named {e.name!r}")
        by_name[e.name] = e
    return entries, by_name


def _atoms_of(ast) -> int:
    if ast[0] in ("R", "L"):
        return 1
    if ast[0] in ("I", "U"):
        return 0
    return sum(_atoms_of(x) for x in ast[1:])


class _Stop(Exception):
    pass


class _Search:
    def __init__(self, batch, entries, lib, names, *, converses, masks, budget, seconds, t0):
        self.b, self.M, self.I = batch, batch.M, batch.I
        self.budget, self.seconds, self.t0 = budget, seconds, t0
        self.masks = masks
        self.steps = 0
        self.stopped: str | None = None
        self.fail: set = set()
        terms = [("R", nm) for nm in names] + [("L", e.name) for e in entries]
        self.atoms = []                          # (ast, value, converse value as float32)
        for t in terms:
            v = _eval(t, batch, lib)
            for conv in ((False, True) if converses else (False,)):
                val = np.swapaxes(v, -1, -2) if conv else v
                ast = ("T", t) if conv else t
                self.atoms.append((ast, np.ascontiguousarray(val), np.ascontiguousarray(np.swapaxes(val, -1, -2))))

    def _comp(self, a, b):
        return (np.matmul(a.astype(np.float32), b.astype(np.float32)) > 0.5) & self.M

    def _tick(self):
        self.steps += 1
        if self.steps > self.budget:
            self.stopped = "budget"
            raise _Stop
        if self.seconds is not None and time.monotonic() - self.t0 > self.seconds:
            self.stopped = "seconds"
            raise _Stop

    @staticmethod
    def _key(Lo, Up, C, d, can_mask):
        h = hashlib.blake2b(digest_size=16)
        for x in (Lo, Up, C):
            h.update(np.packbits(x).tobytes())
        return h.digest(), d, can_mask

    def gen(self, Lo, Up, C, d, can_mask=True):
        """Candidate programs of exactly d atoms with Lo ⊆ X ⊆ Up on C."""
        key = self._key(Lo, Up, C, d, can_mask)
        if key in self.fail:
            return
        produced = False
        LoC = Lo & C
        if d == 1:
            for ast, A, _ in self.atoms:
                self._tick()
                if not ((LoC & ~A).any() or (A & C & ~Up).any()):
                    produced = True
                    yield ast
        else:
            notUp = ~Up & C
            for side in ("L", "R"):
                for ast, A, At in self.atoms:
                    self._tick()
                    if side == "L":                                   # X = A ∘ Y
                        Up2 = ~self._comp(At, notUp) & self.M
                        cnt = np.matmul(A.astype(np.float32), Up2.astype(np.float32))
                        if (LoC & (cnt < 0.5)).any():
                            continue
                        Lo2 = self._comp(At, LoC & (np.abs(cnt - 1) < 0.5)) & Up2
                        C2 = self._comp(At, C)
                    else:                                             # X = Y ∘ A
                        Up2 = ~self._comp(notUp, At) & self.M
                        cnt = np.matmul(Up2.astype(np.float32), A.astype(np.float32))
                        if (LoC & (cnt < 0.5)).any():
                            continue
                        Lo2 = self._comp(LoC & (np.abs(cnt - 1) < 0.5), At) & Up2
                        C2 = self._comp(C, At)
                    for sub in self.gen(Lo2, Up2, C2, d - 1, True):
                        produced = True
                        yield ("COMP", ast, sub) if side == "L" else ("COMP", sub, ast)
        if self.masks and can_mask and d >= 2 and not (LoC & self.I).any():     # X = Y ∖ I: X is 0 on the diagonal
            for sub in self.gen(Lo & ~self.I, Up | self.I, C & ~self.I, d, False):
                produced = True
                yield ("MINUS", sub, ("I",))
        if not produced:
            self.fail.add(key)


def synthesize_residual(base, target, *, depth: int = 12, budget: int = 500_000, seconds: float | None = None,
                        library=(), names: Sequence[str] | None = None, check=None,
                        test: Callable[..., dict] | None = None, name: str | None = None, converses: bool = True,
                        masks: bool = True, log: Callable[[str], None] | None = None) -> ResidualFound:
    """The shortest chain program (fewest atoms) equal to `target` on every example, found top-down through residuals
    (see the module docstring), or None.

    base: the examples — a sequence of Relations (as `synthesize` takes) or of worlds, mappings name → (n, n) boolean
    matrix (as `find_rules` takes). target: per-example (n, n) matrices, one padded (B, n, n) array, or (with worlds)
    a function world → matrix. names: the relations used as atoms (default every relation of the examples; with
    worlds every square matrix); library: a Library or a sequence of Entry — each entry is one more atom ("L", name).
    depth: the most atoms; budget: the most residual steps; seconds: a wall-clock stop. check=(examples, targets): a
    candidate must also be exact there (examples in either form, targets as above); test: a HeldOut or any
    test(name, program, library) → a record with "passed" — `name` is then required. converses / masks: allow Aᵀ and
    ∖ I (both on by default). log: called with one line per depth. Deterministic: atoms in the order of `names`, then
    the library; no randomness."""
    t0 = time.monotonic()
    if test is not None and name is None:
        raise ValueError("a test judges a named target: pass name=")
    if int(depth) < 1:
        raise ValueError("depth is at least 1")
    batch, worlds = _as_batch(base, names)
    T = _targets(target, batch, worlds)
    entries, lib = _library(library)
    full = None
    if check is not None:
        cb, cworlds = _as_batch(check[0], batch.names)
        full = (cb, _targets(check[1], cb, cworlds))
    S = _Search(batch, entries, lib, batch.names, converses=converses, masks=masks, budget=int(budget), seconds=seconds, t0=t0)
    rejected = 0
    tried: set = set()
    try:
        for d in range(1, int(depth) + 1):
            before = S.steps
            for cand in S.gen(T, T, batch.M, d):
                if cand in tried:                       # one chain is reached from both ends (A ∘ Y and Y ∘ A)
                    continue
                tried.add(cand)
                if not np.array_equal(_eval(cand, batch, lib), T):       # the package's evaluator: exact, every pair
                    continue
                if full is not None and not np.array_equal(_eval(cand, full[0], lib), full[1]):
                    rejected += 1
                    continue
                if test is not None and not test(name, cand, entries).get("passed"):
                    rejected += 1
                    continue
                if log is not None:
                    log(f"depth {d}: found after {S.steps} steps, {rejected} rejected")
                return ResidualFound(cand, S.steps, time.monotonic() - t0, _atoms_of(cand), "found", rejected)
            if log is not None:
                log(f"depth {d}: nothing in {S.steps - before} steps ({S.steps} in all, {rejected} rejected, "
                    f"{time.monotonic() - t0:.2f} s)")
    except _Stop:
        if log is not None:
            log(f"stopped ({S.stopped}) after {S.steps} steps")
        return ResidualFound(None, S.steps, time.monotonic() - t0, None, S.stopped or "budget", rejected)
    return ResidualFound(None, S.steps, time.monotonic() - t0, None, "exhausted", rejected)


__all__ = ["ResidualFound", "synthesize_residual"]
