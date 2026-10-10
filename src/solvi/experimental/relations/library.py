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
documented and tested with a stand-in. The enumerator inside is a minimal one (`_enum_min`) until the full relation
enumerator of this package replaces it; the AST and the `Library` fields stay."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from . import _enum_min as _E

OPS1, OPS2 = _E.OPS1, _E.OPS2
show, size = _E.show, _E.size


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


def _names(ast, kind) -> set[str]:
    if ast[0] == kind:
        return {ast[1]}
    if ast[0] in ("R", "L", "I"):
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
        if ast[0] in ("R", "I"):
            return ast
        if ast[0] == "CNT":
            return ("CNT", ast[1], go(ast[2], depth + 1))
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
    batch = _E.Batch([world], _base_of(program, lib) or [next(iter(world))])
    n = batch.sizes[0]
    return _E.evaluate(program, batch, lib, {})[0, :n, :n].copy()


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
            self._batches[k] = _E.Batch(self.examples, names)
        return self._batches[k]

    def __call__(self, name: str, program, library: Sequence[Library] = ()) -> dict:
        if name not in self.targets:
            raise KeyError(f"no held-out target {name!r}")
        if name not in self._truth:
            self._truth[name] = _target_arrays(self.targets[name], self.examples, name)
        lib = _lib_map(library)
        batch = self._batch(_base_of(program, lib) or [next(iter(self.examples[0]))])
        pred = _E.evaluate(program, batch, lib, {})
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
    batch = _E.Batch(examples, base)
    sub = examples[:search_on] if search_on and 0 < search_on < len(examples) else examples
    sbatch = _E.Batch(sub, base) if sub is not examples else batch
    order_t = sorted(targets)
    truth, struth = {}, {}
    for name in order_t:
        arrs = _target_arrays(targets[name], examples, name)
        truth[name] = _E.pack(batch.pad(arrs))
        struth[name] = _E.pack(sbatch.pad(arrs[:len(sub)]))
    tkey = {}
    for name in order_t:
        tkey.setdefault(_E.key(struth[name]), []).append(name)
    tpos = {name: int(_E._POP[truth[name]].sum()) for name in order_t}
    spos = {name: int(_E._POP[struth[name]].sum()) for name in order_t}

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
        val = _E.pack(_E.evaluate(prog, batch, _lib_map(lib), {}))
        if _E.mismatches(val, truth[name]) <= tolerance * tpos[name]:
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
        r.programs, r.round_programs, r.round, r.size, r.by = cost, own, rounds, _E.size(prog), by
        open_.discard(name)
        return True

    while open_:
        if spent < budget:
            rounds += 1
            terminals = [(e.name, e.program) for e in (lib if grow else library)]
            en = _E.Enumerator(sbatch, base, terminals, ops1, ops2, counts)
            admitted = False
            for ast, packed, k in en.run(size, budget - spent):
                if tolerance > 0:
                    hit = [n for n in sorted(open_) if _E.mismatches(packed, struth[n]) <= tolerance * spos[n]]
                else:
                    hit = [n for n in tkey.get(k, ()) if n in open_]
                for name in hit:
                    if full_match(name, ast):
                            admitted |= check(name, ast, "enumerate", spent + en.programs, en.programs)
                if admitted and grow:
                    break
            spent += en.programs
            if admitted and grow:
                continue
        if proposer is None:
            break
        more = False                                    # the fallback: the proposer's programs, checked the same way
        for name in sorted(open_):
            for prog in proposer.propose(name, examples, _target_arrays(targets[name], examples, name), list(lib)):
                spent += 1
                val = _E.pack(_E.evaluate(prog, batch, _lib_map(lib), {}))
                if _E.mismatches(val, truth[name]) <= tolerance * tpos[name] and check(
                        name, prog, getattr(proposer, "name", "proposer"), spent, 1):
                    more = True
                    break
        if not (more and grow):
            break
    return Rules(library=lib, found=found, order=order, records=records, programs=spent, rounds=rounds)


__all__ = ["HeldOut", "Library", "OPS1", "OPS2", "RuleProposer", "Rules", "TargetRecord", "expand", "find_rules", "run",
           "show", "size"]
