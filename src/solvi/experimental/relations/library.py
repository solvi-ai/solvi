"""Composites by enumeration with a library: find rules over relations, admit only what passes a held-out test, and
build the next rules on the admitted ones.

    from solvi.experimental.relations.library import HeldOut, find_rules, run

    test = HeldOut(held_out_worlds, targets)                  # the judge: other worlds than the examples
    r = find_rules(worlds, targets, base=["PARENT", "IS_DIR", "NAME=secret"], size=6, budget=50_000,
                   library=(), test=test)
    r.found["denied"].program                                  # an AST over the base relations and earlier finds
    run(r.found["denied"].program, new_world, library=r.library)   # runs on any world, of any size

A world is a mapping name → (n, n) boolean matrix (a unary predicate such as IS_DIR is a diagonal matrix). A target is,
per name, a function world → (n, n) boolean matrix, or the matrices of the examples in order.

The construction rule:
1. Enumerate small programs bottom-up with observational equivalence, from the base relations plus the library.
2. Accept only what passes the held-out test (exact, or support / precision / recall / net).
3. Put accepted rules into the library as new terminals, so composites become small programs; enumerate again.
4. Use a learned proposer (a gradient-trained rule memory, for example) only when the enumeration budget runs out or the
   labels are noisy, and only as a proposer: its programs are checked like any other candidate (`proposer=`).
5. Rank candidates with a learned or feature ranker only when a check is expensive (`solvi.experimental.laws.
   RankedProposer`).

What it does not do: invent the intermediate concepts. The targets are given (as an unordered set: find_rules sorts
them by name, so the caller's order is not a curriculum); what it finds by itself is which target is within reach of the
current library, and the programs. No gradient learner ships with solvi (its core has no torch): the `proposer=` slot is
documented and tested with a stand-in. The search is the package's relation-algebra enumerator (`Enumerator`, with
the operators `ops1` / `ops2` / `counts` name; the identity I and the universal relation U are always terminals) and the
programs are the package's tuples (`show`, `size`, `evaluate`); `library=` takes `Library` entries of this module or
the package's `Library` of accepted programs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from . import _algebra as _A
from ._enumerate import Enumerator, _digest
from ._enumerate import Library as _Programs

OPS1 = ("T", "PLUS", "CNT")         # "CNT": at least k successors, one operator CNT<k> per k in `counts` (1 to 3)
OPS2 = ("COMP", "AND", "OR", "MINUS")
show, size = _A.show, _A.size
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


@dataclass(frozen=True, eq=False)
class Library:
    """One accepted program, usable as a terminal ("L", name) by later programs: the program (an AST over base
    relations and earlier entries), its name (the target it was accepted for), the examples it was found on, and the
    held-out test's record (passed, support, hits, precision, recall, net, errors, by)."""
    program: tuple
    name: str
    examples: tuple = ()
    test_record: dict = field(default_factory=dict)


def _lib_map(library) -> dict[str, tuple]:
    out = {}
    for e in library or ():
        if e.name in out:
            raise ValueError(f"two library entries are named {e.name!r}")
        out[e.name] = e.program
    return out


def _programs(library) -> _Programs:
    """The entries as the package's Library (what the enumerator and the evaluator take), in an order where every
    entry comes after the entries it uses."""
    todo, out = list(library or ()), _Programs()
    while todo:
        rest = [e for e in todo if not all(r in out for r in _names(e.program, "L"))]
        if len(rest) == len(todo):
            raise ValueError(f"library entries {[e.name for e in rest]} use entries that are not in the library")
        for e in todo:
            if e not in rest:
                out.add(e.program, e.name)
        todo = rest
    return out


def _world_n(ex, names) -> int:
    missing = [x for x in names if x not in ex]
    if missing:
        raise ValueError(f"an example has no relation {missing}")
    ns = {np.asarray(ex[x]).shape for x in names}
    shape = next(iter(ns))
    if len(ns) != 1 or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"an example's relations must be square and of one size, got {ns}")
    return shape[0]


def _batch(examples, names) -> "_A._Batch":
    """Example worlds (name → (n, n) bool) as the package's padded batch over `names`."""
    if not examples:
        raise ValueError("no examples")
    names = list(names)
    rels = []
    for ex in examples:
        _world_n(ex, names)
        rels.append(_A.Relations.from_dense(names, np.stack([np.asarray(ex[x], bool) for x in names])))
    return _A._Batch(rels, names)


def _pad(batch, arrays) -> np.ndarray:
    """Per-example (n, n) matrices → one padded (B, n, n) tensor; ValueError on a wrong shape."""
    arrays = list(arrays)
    for b, (a, n) in enumerate(zip(arrays, batch.sizes)):
        if np.shape(a) != (n, n):
            raise ValueError(f"example {b}: a target of shape {np.shape(a)}, the world has {n} nodes")
    return batch.pad(arrays)


def _pack(value) -> np.ndarray:
    return np.packbits(np.asarray(value, bool).reshape(-1))


def _mismatches(a, b) -> int:
    return int(_POP[np.bitwise_xor(a, b)].sum())


def _any_relation(world) -> list[str]:
    return [next(k for k, v in world.items() if np.ndim(v) == 2 and np.shape(v)[0] == np.shape(v)[1])]


def _ops(ops1, ops2, counts) -> list[str]:
    out = []
    for o in ops1:
        if o == "CNT":
            bad = [k for k in counts if int(k) not in (1, 2, 3)]
            if bad:
                raise ValueError(f"counts {bad}: at least 1, 2 or 3 successors")
            out += [f"CNT{int(k)}" for k in counts]
        else:
            out.append(o)
    out += list(ops2)
    bad = [o for o in out if o not in _A.UNARY + _A.BINARY]
    if bad:
        raise ValueError(f"unknown operators {bad} (have {list(_A.UNARY + _A.BINARY)} and CNT)")
    return out


def _names(ast, kind) -> set[str]:
    if ast[0] == kind:
        return {ast[1]}
    if ast[0] in ("R", "L", "I", "U"):
        return set()
    return set().union(*(_names(x, kind) for x in ast[1:] if isinstance(x, tuple)))


def expand(program, library=()) -> tuple:
    """The program with every library entry replaced by its own program, down to base relations."""
    lib = _lib_map(library)

    def go(ast, depth=0):
        if depth > 64:
            raise RecursionError("a library entry refers to itself")
        if ast[0] == "L":
            return go(lib[ast[1]], depth + 1)
        if ast[0] in ("R", "I", "U"):
            return ast
        return (ast[0],) + tuple(go(x, depth + 1) for x in ast[1:])
    return go(program)


def _base_of(program, lib) -> list[str]:
    return sorted(_names(expand(program, [Library(p, n) for n, p in lib.items()]), "R"))


def run(program, world: Mapping[str, np.ndarray], library: Sequence[Library] = ()) -> np.ndarray:
    """The program's relation on one world: an (n, n) boolean matrix."""
    lib = _lib_map(library)
    for name in _names(program, "L"):
        if name not in lib:
            raise KeyError(f"no library entry {name!r}")
    batch = _batch([world], _base_of(program, lib) or _any_relation(world))
    n = batch.sizes[0]
    return _A._eval(program, batch, _programs(library))[0, :n, :n].copy()


def _target_arrays(target, examples, name) -> list[np.ndarray]:
    if callable(target):
        return [np.asarray(target(ex), bool) for ex in examples]
    arrs = [np.asarray(a, bool) for a in target]
    if len(arrs) != len(examples):
        raise ValueError(f"target {name!r}: {len(arrs)} matrices for {len(examples)} examples")
    return arrs


# ───────────────────────────────────────────── the held-out test
class HeldOut:
    """The judge: the program against the targets on held-out worlds (other worlds than the examples; any sizes).

    exact (default): every pair of every world right. Otherwise: support (predicted pairs) ≥ min_support, precision ≥
    min_precision, recall ≥ min_recall and net = hits − false pairs > 0 — for noisy labels. A target with no true pair
    in the held-out worlds cannot be tested and never passes. → a record: passed, support, hits, positives, precision,
    recall, net, errors, worlds, why."""

    def __init__(self, examples: Sequence[Mapping[str, np.ndarray]], targets: Mapping[str, Any], *, exact: bool = True,
                 min_support: int = 5, min_precision: float = 0.9, min_recall: float = 0.9):
        self.examples = list(examples)
        self.targets = dict(targets)
        self.exact = bool(exact)
        self.min_support, self.min_precision, self.min_recall = int(min_support), float(min_precision), float(min_recall)
        self._truth: dict[str, list[np.ndarray]] = {}
        self._batches: dict[tuple, Any] = {}

    def _batch(self, names):
        k = tuple(names)
        if k not in self._batches:
            self._batches[k] = _batch(self.examples, names)
        return self._batches[k]

    def __call__(self, name: str, program, library: Sequence[Library] = ()) -> dict:
        if name not in self.targets:
            raise KeyError(f"no held-out target {name!r}")
        if name not in self._truth:
            self._truth[name] = _target_arrays(self.targets[name], self.examples, name)
        lib = _lib_map(library)
        batch = self._batch(_base_of(program, lib) or _any_relation(self.examples[0]))
        pred = _A._eval(program, batch, _programs(library))
        support = hits = positives = errors = 0
        for b, (y, n) in enumerate(zip(self._truth[name], batch.sizes)):
            if y.shape != (n, n):
                raise ValueError(f"held-out target {name!r}, world {b}: shape {y.shape}, the world has {n} nodes")
            p = pred[b, :n, :n]
            support += int(p.sum())
            hits += int((p & y).sum())
            positives += int(y.sum())
            errors += int((p ^ y).sum())
        precision = hits / support if support else None
        recall = hits / positives if positives else None
        net = hits - (support - hits)
        why = None
        if positives == 0:
            why = "no true pair in the held-out worlds"
        elif self.exact and errors:
            why = f"{errors} wrong pairs"
        elif not self.exact and (support < self.min_support or (precision or 0) < self.min_precision
                                 or (recall or 0) < self.min_recall or net <= 0):
            why = f"support {support}, precision {precision}, recall {recall}, net {net}"
        return {"passed": why is None, "support": support, "hits": hits, "positives": positives,
                "precision": precision, "recall": recall, "net": net, "errors": errors, "worlds": len(self.examples),
                "why": why}


# ───────────────────────────────────────────── the proposer slot
class RuleProposer(Protocol):
    """A fallback proposer for find_rules — the slot for a gradient-trained rule memory (none ships: solvi's core has
    no torch). Called once per open target when enumeration ran out: `propose(name, examples, target, library)` →
    programs (ASTs); each is checked on the examples and by the held-out test like any enumerated program."""
    name: str

    def propose(self, name: str, examples: Sequence, target: list, library: Sequence[Library]) -> Iterable[tuple]: ...


# ───────────────────────────────────────────── find_rules
@dataclass
class TargetRecord:
    """Per target: programs = the enumerated programs spent (cumulative, all rounds) when it was admitted, or None;
    round_programs = those of its own round only (from the round's start to its admission: the search it needed with
    the library of that moment); round = the round it was admitted in; size = its program's size; rejected = programs
    that fit the examples but failed the held-out test; recheck_failed = matches on the search subset that failed on all
    the examples (`search_on`); by = "enumerate" or the proposer's name."""
    name: str
    programs: int | None = None
    round_programs: int | None = None
    round: int | None = None
    size: int | None = None
    rejected: int = 0
    recheck_failed: int = 0
    by: str | None = None


@dataclass
class Rules:
    """The result: library (the given entries, then the admitted ones in order), found (name → its entry), order (the
    names admitted, in order), records (name → TargetRecord), programs (enumerated in total), rounds."""
    library: list
    found: dict
    order: list
    records: dict
    programs: int = 0
    rounds: int = 0


def find_rules(examples: Sequence[Mapping[str, np.ndarray]], targets: Mapping[str, Any], *, base: Sequence[str],
               size: int = 6, budget: int = 50_000, library: Sequence[Library] = (),
               test: Callable[..., dict], proposer: RuleProposer | None = None, grow: bool = True,
               tolerance: float = 0.0, ops1: Sequence[str] = OPS1, ops2: Sequence[str] = OPS2,
               counts: Sequence[int] = (1,), search_on: int | None = None) -> Rules:
    """Find a program for each target, building composites on earlier finds.

    Rounds: enumerate programs of size ≤ `size` from `base` (relation names in every example), the identity and the
    library (bottom-up, observational equivalence on the examples); a program whose relation equals a target's on every
    example (or misses at most `tolerance` × its true pairs) goes to `test(name, program, library)` — a HeldOut, or any
    function returning a record with "passed"; the first that passes is admitted (a `Library` entry) and, with `grow`,
    the next round starts with it as a terminal. A round that admits nothing ends the search; then `proposer` (if any)
    proposes for the open targets, its programs checked the same way. `budget` caps the enumerated programs over all
    rounds. `search_on=k`: enumerate on the first k examples only (cheaper keys), and re-check a match on all the
    examples before the test (observational equivalence on a subset merges programs that differ elsewhere: a failed
    re-check is counted, `recheck_failed`, and the search goes on). `grow=False`: one flat pass, finds are recorded but
    never used as terminals (the control). Deterministic;
    targets are taken in name order, so the caller's order is no curriculum."""
    examples = list(examples)
    base = list(base)
    names_ok = set.intersection(*(set(ex) for ex in examples)) if examples else set()
    unknown = [x for x in base if x not in names_ok]
    if unknown:
        raise ValueError(f"base relations {unknown} are not in every example")
    ops = _ops(ops1, ops2, counts)
    batch = _batch(examples, base)
    sub = examples[:search_on] if search_on and 0 < search_on < len(examples) else examples
    sbatch = _batch(sub, base) if sub is not examples else batch
    order_t = sorted(targets)
    truth, struth = {}, {}
    for name in order_t:
        arrs = _target_arrays(targets[name], examples, name)
        truth[name] = _pack(_pad(batch, arrs))
        struth[name] = _pack(_pad(sbatch, arrs[:len(sub)]))
    tkey = {}
    for name in order_t:
        tkey.setdefault(_digest(struth[name]), []).append(name)
    tpos = {name: int(_POP[truth[name]].sum()) for name in order_t}
    spos = {name: int(_POP[struth[name]].sum()) for name in order_t}

    lib = list(library)
    _lib_map(lib)
    clash = [x for x in order_t if x in {e.name for e in lib}]
    if clash:
        raise ValueError(f"targets {clash} are already library entries")
    records = {name: TargetRecord(name) for name in order_t}
    found: dict[str, Library] = {}
    order: list[str] = []
    open_ = set(order_t)
    spent = 0
    rounds = 0
    ex_t = tuple(examples)

    def full_match(name, prog):
        if sbatch is batch:
            return True
        val = _pack(_A._eval(prog, batch, _programs(lib)))
        if _mismatches(val, truth[name]) <= tolerance * tpos[name]:
            return True
        records[name].recheck_failed += 1
        return False

    def check(name, prog, by, cost, own):
        rec = dict(test(name, prog, list(lib)), by=by)
        if not rec.get("passed"):
            records[name].rejected += 1
            return False
        e = Library(program=prog, name=name, examples=ex_t, test_record=rec)
        lib.append(e)
        found[name] = e
        order.append(name)
        r = records[name]
        r.programs, r.round_programs, r.round, r.size, r.by = cost, own, rounds, _A.size(prog), by
        open_.discard(name)
        return True

    while open_:
        if spent < budget:
            rounds += 1
            en = Enumerator(sbatch, size, _programs(lib if grow else library), ops=ops, max_evals=budget - spent)
            admitted = False
            for p in en:
                packed = _pack(p.value)
                if tolerance > 0:
                    hit = [n for n in sorted(open_) if _mismatches(packed, struth[n]) <= tolerance * spos[n]]
                else:
                    hit = [n for n in tkey.get(_digest(packed), ()) if n in open_]
                for name in hit:
                    if full_match(name, p.ast):
                        admitted |= check(name, p.ast, "enumerate", spent + en.evals, en.evals)
                if admitted and grow:
                    break
            spent += en.evals
            if admitted and grow:
                continue
        if proposer is None:
            break
        more = False                                    # the fallback: the proposer's programs, checked the same way
        for name in sorted(open_):
            for prog in proposer.propose(name, examples, _target_arrays(targets[name], examples, name), list(lib)):
                spent += 1
                val = _pack(_A._eval(prog, batch, _programs(lib)))
                if _mismatches(val, truth[name]) <= tolerance * tpos[name] and check(
                        name, prog, getattr(proposer, "name", "proposer"), spent, 1):
                    more = True
                    break
        if not (more and grow):
            break
    return Rules(library=lib, found=found, order=order, records=records, programs=spent, rounds=rounds)


__all__ = ["HeldOut", "Library", "OPS1", "OPS2", "RuleProposer", "Rules", "TargetRecord", "expand", "find_rules", "run",
           "show", "size"]
