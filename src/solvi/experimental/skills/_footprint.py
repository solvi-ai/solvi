"""Footprints with variables and their generalization judged by the code of the world (see the package docstring).

Pure Python with no package imports, so another program can load this file by path next to its own solvi install.

A footprint is a conjunction of atoms; an atom is a tuple `(relation, *args)` whose arguments are constants or
variables (`Var`). What a relation means is the world's: a `Vocabulary` says whether a ground atom holds in a state,
which values a variable may take there, how a term is grounded under a binding and which lifts (constants to variables,
or one relation to a wider one) a verified plan admits. `generalize` builds the bounded lattice of candidate footprints
from a verified plan's own footprint — lifts applied, atoms dropped — and accepts a candidate only when the plan,
instantiated for every verification state that matches the candidate but not the original footprint, reaches its goal
on the exact copy of that state. The most general candidate that passes is returned; none passing returns the original.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence


@dataclass(frozen=True, order=True)
class Var:
    """A variable of a footprint, a plan template or a goal template. kind: what it ranges over (the vocabulary's
    `domain` reads it)."""
    name: str
    kind: str = ""

    def __repr__(self):
        return f"?{self.name}"


def variables(term) -> tuple:
    """The variables of a term (nested tuples / lists), in order of first appearance."""
    out: list = []

    def walk(t):
        if isinstance(t, Var):
            if t not in out:
                out.append(t)
        elif isinstance(t, (tuple, list)):
            for x in t:
                walk(x)
    walk(term)
    return tuple(out)


def substitute(term, binding: dict):
    """The term with every bound variable replaced by its value (structural; unbound variables stay)."""
    if isinstance(term, Var):
        return binding.get(term, term)
    if isinstance(term, tuple):
        return tuple(substitute(x, binding) for x in term)
    if isinstance(term, list):
        return [substitute(x, binding) for x in term]
    return term


def unify(template, value, binding: dict | None = None) -> dict | None:
    """Extend `binding` so that `template` grounds to `value` (structural: a variable takes the value at its place, a
    bound one must equal it; tuples element by element) → the binding, or None."""
    b = dict(binding or {})

    def go(t, v):
        if isinstance(t, Var):
            if t in b:
                return b[t] == v
            b[t] = v
            return True
        if isinstance(t, (tuple, list)):
            return (isinstance(v, (tuple, list)) and len(t) == len(v)
                    and all(go(x, y) for x, y in zip(t, v)))
        return t == v
    return b if go(template, value) else None


def in_domain(vocab: "Vocabulary", binding: dict, state) -> bool:
    """Every bound variable takes a value of its domain in `state` (a goal unified with a template can bind a variable
    to a value the vocabulary never offers — a goal outside the skill's reach)."""
    return all(val in list(vocab.domain(v, state, binding)) for v, val in binding.items() if isinstance(v, Var))


def render(term) -> str:
    """A term in words: `near(?D, stone, 2)`."""
    if isinstance(term, Var):
        return repr(term)
    if isinstance(term, tuple) and term and isinstance(term[0], str):
        return f"{term[0]}({', '.join(render(x) for x in term[1:])})"
    if isinstance(term, (tuple, list)):
        return "(" + ", ".join(render(x) for x in term) + ")"
    return str(term)


def _plain(term):
    if isinstance(term, Var):
        return f"?{term.name}"
    if isinstance(term, (tuple, list)):
        return [_plain(x) for x in term]
    try:
        json.dumps(term)
        return term
    except (TypeError, ValueError):
        return str(term)


def safe_key(state) -> str:
    """A state's key that works for any state: its canonical JSON, or the hash of its repr when it is not JSON."""
    try:
        return json.dumps(state, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return "repr:" + hashlib.sha256(repr(state).encode()).hexdigest()[:16]


class Vocabulary:
    """The world's relations — subclass it per world.

    holds(atom, state): does a ground atom hold in `state` (required). domain(var, state, binding): the values a
    variable may take in `state` given the binding so far (finite, in a fixed order: the first binding found is the one
    used, at verification and at use alike). ground(term, binding): a term with its variables replaced (default
    structural; override for terms whose grounding is computed, e.g. an offset turned with a direction). goal(name) /
    goal_name(term): a goal's name as a term and back (default: the name itself); action(a) / act(term): an action as a
    term and back. lifts(atoms, actions, goal): the generalizing moves a verified plan admits, in the order they are
    applied (default: none — only drops). footprint(state, actions, predicted): the ground atoms a plan read from its
    start (what `SkillLearner(lift="lattice")` starts the lattice from)."""

    def holds(self, atom, state) -> bool:
        raise NotImplementedError

    def domain(self, var: Var, state, binding: dict) -> Iterable:
        return ()

    def ground(self, term, binding: dict):
        return substitute(term, binding)

    def goal(self, name):
        return name

    def goal_name(self, term):
        return term

    def action(self, a):
        return a

    def act(self, term):
        return term

    def lifts(self, atoms: tuple, actions: tuple, goal) -> list["Lift"]:
        return []

    def footprint(self, state, actions, predicted) -> tuple:
        raise NotImplementedError("a vocabulary used by SkillLearner(lift='lattice') says what a plan read: "
                                  "footprint(state, actions, predicted) → ground atoms")


@dataclass(frozen=True)
class Lift:
    """One generalizing move: apply(atoms, actions, goal) → (atoms, actions, goal), e.g. a constant replaced by a
    variable everywhere it appears, or a relation replaced by a wider one."""
    name: str
    apply: Callable[[tuple, tuple, Any], tuple]


def _order(a):
    return repr(_plain(a))


@dataclass(frozen=True)
class Footprint:
    """A conjunction of atoms (constants or variables in their arguments)."""
    atoms: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "atoms", tuple(sorted(set(self.atoms), key=_order)))

    @property
    def variables(self) -> tuple:
        return variables(self.atoms)

    @property
    def is_ground(self) -> bool:
        return not self.variables

    def __len__(self):
        return len(self.atoms)

    def bindings(self, vocab: Vocabulary, state, binding: dict | None = None) -> Iterator[dict]:
        """Every binding of the footprint's variables under which all its atoms hold in `state` (depth first: the atom
        with the fewest unbound variables next; values in the vocabulary's domain order)."""
        b0 = dict(binding or {})

        def go(rest, b):
            if not rest:
                yield b
                return
            i = min(range(len(rest)), key=lambda j: sum(1 for v in variables(rest[j]) if v not in b))
            atom = rest[i]
            free = [v for v in variables(atom) if v not in b]
            if not free:
                if vocab.holds(vocab.ground(atom, b), state):
                    yield from go(rest[:i] + rest[i + 1:], b)
                return
            v = free[0]
            for val in vocab.domain(v, state, b):
                yield from go(rest, {**b, v: val})
        yield from go(self.atoms, b0)

    def match(self, vocab: Vocabulary, state, binding: dict | None = None) -> dict | None:
        """The first binding under which the footprint holds in `state`, or None."""
        return next(self.bindings(vocab, state, binding), None)

    def ground(self, vocab: Vocabulary, binding: dict) -> "Footprint":
        return Footprint(tuple(vocab.ground(a, binding) for a in self.atoms))

    def to_list(self) -> list:
        return [_plain(a) for a in self.atoms]

    def __str__(self):
        return " & ".join(render(a) for a in self.atoms) or "always"


@dataclass(frozen=True)
class Candidate:
    """One point of the lattice: a footprint, the plan template and goal template under the same variables, the lifts
    applied, the atoms dropped, whether the context (the caller's other conditions) is kept, and its generality
    (lifts + drops; the context counts one)."""
    footprint: Footprint
    actions: tuple
    goal: Any
    lifts: tuple = ()
    dropped: tuple = ()
    context: bool = True
    generality: int = 0

    def describe(self) -> dict:
        return {"footprint": str(self.footprint), "atoms": self.footprint.to_list(),
                "do": [_plain(a) for a in self.actions], "goal": _plain(self.goal), "lifts": list(self.lifts),
                "dropped": [_plain(a) for a in self.dropped], "context": self.context, "generality": self.generality}


def lattice(atoms, actions, goal, vocab: Vocabulary, *, context: bool = False, drop_atoms: bool = True,
            max_candidates: int = 12) -> list[Candidate]:
    """The bounded lattice above a verified plan's footprint, most general first (the original itself is not in it).

    atoms: the ground atoms the plan read; actions / goal: the plan and its goal as they ran. context: the caller has
    other conditions (the old condition of a skill) a candidate may keep or drop as a block. Points: every subset of the
    vocabulary's lifts (applied in its order) × the context kept or dropped, and — when drop_atoms — each single atom
    dropped from the candidate with every lift applied and the context dropped. Ordered by generality (lifts + atoms
    dropped + 1 for a dropped context), then more lifts first, then the vocabulary's order; duplicates removed; the
    first `max_candidates` kept."""
    acts0 = tuple(vocab.action(a) for a in actions)
    goal0 = vocab.goal(goal)
    base = tuple(atoms)
    lifts = list(vocab.lifts(base, acts0, goal0))
    pts = []
    seen = set()

    def add(fp_atoms, acts, g, names, dropped, keep):
        fp = Footprint(tuple(fp_atoms))
        k = (fp.atoms, acts, repr(g), keep)
        if k in seen or (not names and not dropped and keep):
            return
        seen.add(k)
        gen = len(names) + len(dropped) + (0 if keep or not context else 1)
        pts.append((Candidate(fp, acts, g, tuple(names), tuple(dropped), keep, gen), len(pts)))

    keeps = (False, True) if context else (True,)
    full = None
    for r in range(len(lifts), -1, -1):
        for sub in itertools.combinations(range(len(lifts)), r):
            a, ac, g = base, acts0, goal0
            for i in sub:
                a, ac, g = lifts[i].apply(tuple(a), tuple(ac), g)
            names = [lifts[i].name for i in sub]
            if r == len(lifts):
                full = (tuple(a), tuple(ac), g, names)
            for keep in keeps:
                add(a, tuple(ac), g, names, (), keep)
    if drop_atoms and full is not None:
        a, ac, g, names = full
        for atom in a:
            rest = tuple(x for x in a if x != atom)
            if not rest:
                continue
            add(rest, ac, g, names, (atom,), not context)
    pts.sort(key=lambda p: (-p[0].generality, -len(p[0].lifts), p[1]))
    return [c for c, _ in pts[:max(0, int(max_candidates))]]


@dataclass
class Generalization:
    """What `generalize` found. accepted: a candidate more general than the original passed; footprint / actions /
    goal / lifts / dropped / context: the chosen candidate's (the original's when none passed); original: the verified
    plan's own footprint; verified: the verification states the chosen candidate passed (0 when none); candidates: one
    record per candidate tried (its footprint, the states it was checked on, where it failed and why); states: the
    accepted candidate's verification states (key, binding, the ground plan, the simulation's hash, pool or perturbed);
    checks: the plan instantiations simulated in all. `evidence` (not exported) keeps (state, handle, actions, goal,
    simulation) of the accepted candidate's states for `replay`."""
    accepted: bool
    footprint: Footprint
    actions: tuple
    goal: Any
    original: Footprint
    lifts: tuple = ()
    dropped: tuple = ()
    context: bool = True
    verified: int = 0
    candidates: list = field(default_factory=list)
    states: list = field(default_factory=list)
    checks: int = 0
    short: bool = False
    evidence: list = field(default_factory=list, repr=False)

    @property
    def variables(self) -> tuple:
        return self.footprint.variables

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "footprint": str(self.footprint), "atoms": self.footprint.to_list(),
                "do": [_plain(a) for a in self.actions], "goal": _plain(self.goal), "original": str(self.original),
                "original_atoms": self.original.to_list(), "variables": [repr(v) for v in self.variables],
                "lifts": list(self.lifts), "dropped": [_plain(a) for a in self.dropped], "context": self.context,
                "verified": self.verified, "checks": self.checks, "short": self.short,
                "candidates": list(self.candidates),
                "states": list(self.states)}


def _states(pool, perturb, cand, rng, base, n_target, tries):
    """Verification sources, alternating: the pool (shuffled) and perturbations of the pool / the plan's start."""
    order = list(range(len(pool)))
    rng.shuffle(order)
    src = [pool[i] for i in order]
    bases = [base] + src if base is not None else src
    pi, made = 0, 0
    turn = 0
    while True:
        gave = False
        if turn % 2 == 0 and pi < len(src):
            st, h = src[pi]
            pi += 1
            yield st, h, "pool"
            gave = True
        elif turn % 2 == 1 and perturb is not None and made < tries and bases:
            st, h = bases[made % len(bases)]
            made += 1
            for st2, h2 in perturb(h, st, rng, cand) or ():
                yield st2, h2, "perturbed"
            gave = True
        turn += 1
        if not gave and pi >= len(src) and (perturb is None or made >= tries or not bases):
            return


def _cases(cand, vocab, state, goals):
    """The bindings a use of the candidate would take in `state`: one per value of its goal's variables (in the
    vocabulary's domain order; the goal is what a use is asked for), each the first binding of the footprint."""
    gvs = variables(cand.goal)
    doms = [list(vocab.domain(v, state, {})) for v in gvs]
    for combo in itertools.product(*doms):
        b0 = dict(zip(gvs, combo))
        if goals is not None and not goals(vocab.goal_name(vocab.ground(cand.goal, b0))):
            continue
        b = cand.footprint.match(vocab, state, b0)
        if b is None or any(v not in b for v in variables(cand.actions)):
            continue
        yield b


def generalize(atoms, actions, goal, vocab: Vocabulary, *, pool: Sequence = (), check: Callable,
               original: Callable[[Any, Any], bool], context_holds: Callable | None = None,
               perturb: Callable | None = None, start: tuple | None = None, goals: Callable[[Any], bool] | None = None,
               verify_n: int = 16, verify_min: int | None = None, max_candidates: int = 12, drop_atoms: bool = True,
               drop_witness: int = 2, seed: int | str = 0, key: Callable[[Any], str] | None = None) -> Generalization:
    """Generalize a verified plan's footprint on the exact copy (the code of the world is the judge; no model).

    atoms / actions / goal: the plan's ground footprint (what it read at its start), the plan and its goal name.
    pool: the recorded (state, handle) pairs of the stream — a handle is whatever `check` needs to rebuild the world at
    that state (a copy of the environment, a snapshot). check(handle, state, actions, goal) → (ok, simulation hash):
    run the ground plan on a fresh copy of the world at that state; ok when every step is accepted and the goal is
    reached. original(state, goal): the original skill applies there (its goal, its own footprint and old condition) —
    such a case proves nothing new and is never a verification state. context_holds(state, actions): the caller's other
    condition (kept by candidates with context=True; None: no context). perturb(handle, state, rng, candidate) →
    [(state, handle)]: states near a pool state or the plan's start (`start` = (state, handle)), edited by the world's
    code (None: the pool only). goals(name): the ground goals that can be checked (None: any).

    A verification state is a (state, goal) case: for each value of the candidate's goal variables (a use is asked for a
    goal), the footprint's first binding; the plan and goal instantiated under it. A candidate is checked on up to
    `verify_n` cases that match it and not the original, alternating pool and perturbed states; it fails at its first
    failed check, and passes when every check passed, at least `verify_min` (default verify_n // 2) cases were found, and
    every drop was exercised: each dropped atom was false (grounded under the case's binding) in at least
    `drop_witness` passing cases, and a dropped context did not hold in at least as many — a case that differs from the
    original elsewhere says nothing about an atom it never falsified. `short` records that a candidate was not refuted
    but lacked cases or witnesses: worth another try on a larger pool.
    The most general passing candidate (the lattice's order) is returned; none → the original (accepted False).
    Deterministic given the seed, the pool order and the world."""
    key = key or safe_key
    verify_n = int(verify_n)
    verify_min = verify_n // 2 if verify_min is None else int(verify_min)
    verify_min = max(1, verify_min)
    orig = Footprint(tuple(atoms))
    acts0 = tuple(vocab.action(a) for a in actions)
    goal0 = vocab.goal(goal)
    out = Generalization(False, orig, acts0, goal0, orig, context=True)
    cands = lattice(atoms, actions, goal, vocab, context=context_holds is not None, drop_atoms=drop_atoms,
                    max_candidates=max_candidates)
    for ci, cand in enumerate(cands):
        rng = random.Random(f"{seed}|{ci}")
        rec = cand.describe()
        rec.update(states=0, pool=0, perturbed=0, failed=None, why=None)
        passed: list = []
        seen = set()
        for st, h, where in _states(list(pool), perturb, cand, rng, start, verify_n, 4 * verify_n):
            if len(passed) >= verify_n or rec["failed"] is not None:
                break
            for b in _cases(cand, vocab, st, goals):
                if len(passed) >= verify_n:
                    break
                g = vocab.goal_name(vocab.ground(cand.goal, b))
                acts = tuple(vocab.act(vocab.ground(a, b)) for a in cand.actions)
                if cand.context and context_holds is not None and not context_holds(st, acts):
                    continue
                if original(st, g):
                    continue
                k = key(st)
                if (k, repr(g)) in seen:
                    continue
                seen.add((k, repr(g)))
                ok, sim = check(h, st, acts, g)
                out.checks += 1
                rec["states"] += 1
                rec[where] += 1
                if not ok:
                    rec["failed"], rec["why"] = k[:200], f"the plan {list(acts)} did not reach {g!r} on the copy"
                    break
                passed.append((st, h, acts, g, sim, where, b))
        out.candidates.append(rec)
        if rec["failed"] is not None:
            continue
        if len(passed) < verify_min:
            rec["why"] = f"{len(passed)} verification states found (at least {verify_min} needed)"
            out.short = True
            continue
        wit = {}
        for atom in cand.dropped:
            n = 0
            for st, _h, _a, _g, _s, _w, b in passed:
                gr = vocab.ground(atom, b)
                if not variables(gr) and not vocab.holds(gr, st):
                    n += 1
            wit[render(atom)] = n
        if not cand.context and context_holds is not None:
            wit["context"] = sum(1 for st, _h, a, _g, _s, _w, _b in passed if not context_holds(st, a))
        rec["witnesses"] = wit
        low = {k: n for k, n in wit.items() if n < drop_witness}
        if low:
            rec["why"] = (f"a drop was not exercised: {', '.join(f'{k} false in {n}' for k, n in low.items())} of "
                          f"{len(passed)} cases (at least {drop_witness} needed)")
            out.short = True
            continue
        out.accepted = True
        out.footprint, out.actions, out.goal = cand.footprint, cand.actions, cand.goal
        out.lifts, out.dropped, out.context = cand.lifts, cand.dropped, cand.context
        out.verified = len(passed)
        out.states = [{"key": key(st)[:200], "binding": {repr(v): _plain(x) for v, x in b.items()},
                       "do": [_plain(a) for a in acts], "goal": _plain(g), "simulation": sim, "source": where}
                      for st, h, acts, g, sim, where, b in passed]
        out.evidence = [(st, h, acts, g, sim) for st, h, acts, g, sim, _, _ in passed]
        return out
    return out


def replay(gen: Generalization, check: Callable) -> dict:
    """Re-run the accepted candidate's verification states on fresh copies → {"states", "same", "differ"}: a state
    differs when its check fails now or its simulation hash is not the recorded one."""
    same, differ = 0, []
    for st, h, acts, g, sim in gen.evidence:
        ok, sim2 = check(h, st, acts, g)
        if ok and sim2 == sim:
            same += 1
        else:
            differ.append({"do": [_plain(a) for a in acts], "goal": _plain(g), "ok": ok, "simulation": sim2,
                           "recorded": sim})
    return {"states": len(gen.evidence), "same": same, "differ": differ}


__all__ = ["Candidate", "Footprint", "Generalization", "Lift", "Var", "Vocabulary", "generalize", "in_domain", "lattice",
           "render", "replay", "safe_key", "substitute", "unify", "variables"]
