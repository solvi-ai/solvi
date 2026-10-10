"""The grammar, the transitions and their held-out buffer, the laws, the law set and its proposers (see the package
docstring)."""
from __future__ import annotations

import hashlib
import inspect
import json
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterable, Protocol


# ───────────────────────────────────────────── the grammar
class Atom:
    """A named boolean feature of a state: `fn(state)` or `fn(state, action)` → bool. `feature` groups atoms that
    describe the same thing (thresholds of one counter, the material of one cell): a condition holds at most one atom
    per feature. Default: the atom's own name.

    code: optional, a Python expression over `s` (the state) and `a` (the action) that equals `fn` — what a compiled
    law set (`LawSet.to_module`) runs in its place; defs: the module-level source the expression needs (a string or a
    tuple of strings; identical ones are written once). An atom without code cannot be compiled."""
    __slots__ = ("name", "fn", "feature", "code", "defs", "_with_action")

    def __init__(self, name: str, fn: Callable[..., Any], feature: str | None = None, *, code: str | None = None,
                 defs: str | Iterable[str] = ()):
        self.name, self.fn, self.feature = str(name), fn, str(feature) if feature is not None else str(name)
        self.code = str(code) if code is not None else None
        self.defs = (defs,) if isinstance(defs, str) else tuple(defs)
        self._with_action = _required_args(fn) >= 2

    def __call__(self, state, action):
        return bool(self.fn(state, action) if self._with_action else self.fn(state))

    def __repr__(self):
        return f"Atom({self.name!r}, feature={self.feature!r})"


def _required_args(fn):
    try:
        ps = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 1
    return sum(1 for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)


class Grammar:
    """The hypothesis space, written by you: the atoms and the largest conjunction `k`. A condition is a sorted tuple of
    atom names, at most `k`, at most one per feature; the empty condition () holds everywhere."""

    def __init__(self, atoms: Iterable[Atom], k: int = 3):
        self.atoms: list[Atom] = []
        self.k = int(k)
        self.by_name: dict[str, Atom] = {}
        self.feature: dict[str, str] = {}
        self.position: dict[str, int] = {}
        for a in atoms:
            self.add(a)

    def add(self, atom: Atom) -> Atom:
        """Append an atom (a proposer may invent one, e.g. a relational atom). A buffer built before evaluates it on
        its older rows when a condition first names it. ValueError if the name is taken."""
        if atom.name in self.by_name:
            raise ValueError(f"two atoms are named {atom.name!r}")
        self.position[atom.name] = len(self.atoms)
        self.atoms.append(atom)
        self.by_name[atom.name] = atom
        self.feature[atom.name] = atom.feature
        return atom

    def true(self, state, action) -> tuple[str, ...]:
        """The names of the atoms true for `action` in `state`, sorted."""
        return tuple(sorted(a.name for a in self.atoms if a(state, action)))

    def check(self, condition) -> tuple[str, ...]:
        """A condition in its one spelling (sorted) → tuple; ValueError for an unknown atom, two atoms of one feature or
        more than k atoms."""
        cond = tuple(sorted({str(x) for x in condition}))
        unknown = [x for x in cond if x not in self.by_name]
        if unknown:
            raise ValueError(f"unknown atoms {unknown}")
        feats = [self.feature[x] for x in cond]
        if len(set(feats)) != len(feats):
            raise ValueError(f"{cond}: at most one atom per feature")
        if len(cond) > self.k:
            raise ValueError(f"{cond}: more than k={self.k} atoms")
        return cond

    def conditions(self, true: Iterable[str]) -> list[tuple[str, ...]]:
        """Every condition built from the atoms `true` (in size order, then by name): what holds in one state."""
        names = sorted(set(true))
        out = []
        for size in range(min(self.k, len(names)) + 1):
            for c in combinations(names, size):
                if len({self.feature[x] for x in c}) == size:
                    out.append(c)
        return out

    def sizes(self, true: Iterable[str]) -> list[int]:
        """How many conditions of each size 0..k `conditions(true)` gives."""
        n = [0] * (self.k + 1)
        for c in self.conditions(true):
            n[len(c)] += 1
        return n

    def random_condition(self, rng: random.Random, size: int) -> tuple[str, ...]:
        """A condition of `size` atoms drawn from the whole grammar (one per feature)."""
        by_feat: dict[str, list[str]] = {}
        for a in self.atoms:
            by_feat.setdefault(a.feature, []).append(a.name)
        feats = sorted(by_feat)
        size = min(size, len(feats))
        return tuple(sorted(rng.choice(by_feat[f]) for f in rng.sample(feats, size)))


# ───────────────────────────────────────────── transitions and the buffer
def _key(x):
    try:
        hash(x)
        return x
    except TypeError:
        return repr(x)


@dataclass(frozen=True)
class Transition:
    """One observed step: the state before, the action, what was observed after, and its group (the world, the session,
    the path — what must not judge a law born from it). id: given, or a content hash."""
    state: Any
    action: Any
    observed: Any
    group: Any = None
    id: str = ""

    def __post_init__(self):
        if not self.id:
            raw = repr((self.group, self.action, self.state, self.observed)).encode()
            object.__setattr__(self, "id", hashlib.sha256(raw).hexdigest()[:16])


class Buffer:
    """Held-out transitions, grouped. Per action: one bitset per atom over its rows and one per group, so a law's
    support and precision on the rows of every group but one are a few AND operations. The proposing transition's group
    is left out of every test (`exclude=`)."""

    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self._rows: dict[Any, list[Transition]] = {}            # action → its rows, in order
        self._atoms: dict[Any, dict[str, int]] = {}             # action → atom → bitset
        self._groups: dict[Any, dict[Any, int]] = {}            # action → group → bitset
        self._order: list[Any] = []                              # groups in order of first appearance
        self._seen: set = set()
        self._cache: dict = {}                                   # (action, what, key) → (bitset, rows done)
        self._width: dict[Any, list[int]] = {}                  # action → per row, the grammar's size when added
        self._late: set = set()                                  # (action, atom) indexed on the rows before it
        self.n = 0

    def add(self, t: Transition):
        a = _key(t.action)
        rows = self._rows.setdefault(a, [])
        bit = 1 << len(rows)
        rows.append(t)
        self._width.setdefault(a, []).append(len(self.grammar.atoms))
        atoms = self._atoms.setdefault(a, {})
        for name in self.grammar.true(t.state, t.action):
            atoms[name] = atoms.get(name, 0) | bit
        g = _key(t.group)
        groups = self._groups.setdefault(a, {})
        groups[g] = groups.get(g, 0) | bit
        if g not in self._seen:
            self._seen.add(g)
            self._order.append(t.group)
        self.n += 1

    def extend(self, ts: Iterable[Transition]):
        for t in ts:
            self.add(t)

    def __len__(self):
        return self.n

    def groups(self) -> list:
        """The groups, in order of first appearance."""
        return list(self._order)

    def rows(self, action=None, exclude=None) -> list[Transition]:
        """The rows of one action (or every row), without the group `exclude`."""
        acts = [_key(action)] if action is not None else list(self._rows)
        out = []
        for a in acts:
            out += [t for t in self._rows.get(a, ()) if exclude is None or _key(t.group) != _key(exclude)]
        return out

    # bitsets over the rows of one action
    def allowed(self, action, exclude=None) -> int:
        a = _key(action)
        m = (1 << len(self._rows.get(a, ()))) - 1
        if exclude is not None:
            m &= ~self._groups.get(a, {}).get(_key(exclude), 0)
        return m

    def group_mask(self, action, group) -> int:
        return self._groups.get(_key(action), {}).get(_key(group), 0)

    def match(self, action, condition) -> int:
        a = _key(action)
        m = (1 << len(self._rows.get(a, ()))) - 1
        atoms = self._atoms.get(a, {})
        for name in condition:
            self._index_late(a, name)
            m &= atoms.get(name, 0)
        return m

    def _index_late(self, a, name):
        """An atom the grammar gained after some rows of `a` were added: evaluate it on those rows, once."""
        i = self.grammar.position.get(name)
        if i is None or (a, name) in self._late:
            return
        widths = self._width.get(a, ())
        if not widths or widths[0] > i:
            return                                               # every row saw the atom when it was added
        self._late.add((a, name))
        atom = self.grammar.by_name[name]
        rows = self._rows[a]
        m = 0
        for j, w in enumerate(widths):
            if w > i:
                break
            if atom(rows[j].state, rows[j].action):
                m |= 1 << j
        if m:
            atoms = self._atoms.setdefault(a, {})
            atoms[name] = atoms.get(name, 0) | m

    def where(self, action, what, key, test: Callable[[Transition], bool]) -> int:
        """The rows of `action` where test(row) holds, cached under (what, key) and extended as rows arrive."""
        a = _key(action)
        rows = self._rows.get(a, ())
        ck = (a, what, key)
        m, done = self._cache.get(ck, (0, 0))
        for j in range(done, len(rows)):
            if test(rows[j]):
                m |= 1 << j
        self._cache[ck] = (m, len(rows))
        return m

    def ids(self, action, mask) -> list[str]:
        rows = self._rows.get(_key(action), ())
        return [rows[j].id for j in range(len(rows)) if mask >> j & 1]


# ───────────────────────────────────────────── candidates and laws
@dataclass(frozen=True)
class Candidate:
    """A proposed law: for `action` under `condition` (atom names), the effect is `effect`."""
    action: Any
    condition: tuple
    effect: Any
    proposer: str = "enumerate"


@dataclass(eq=False)
class Law:
    """An accepted law (see the package docstring). support / hits / precision / net: on the held-out buffer at
    acceptance (rows of every group but `born`), counted; sources: the ids of the rows it was accepted on; n / hits_after:
    observations from other groups since, while it was the law applied; held: (hits, n) on the buffer rows where it is
    the applied law, from the last test; status: "active" | "retired"; why: the retirement's reason."""
    id: str
    action: Any
    condition: tuple
    effect: Any
    support: int = 0
    hits: int = 0
    precision: float | None = None
    net: int = 0
    born: Any = None
    status: str = "active"
    proposer: str = "enumerate"
    sources: tuple = ()
    seq: int = 0
    n: int = 0
    hits_after: int = 0
    held: tuple = (0, 0)
    why: str | None = None

    def holds(self, true: Iterable[str]) -> bool:
        t = set(true)
        return all(x in t for x in self.condition)

    def running(self) -> tuple[int, int]:
        """(hits, n): held-out rows where it applies plus observations from other groups since its acceptance."""
        return self.held[0] + self.hits_after, self.held[1] + self.n

    def text(self) -> str:
        return f"{self.action}{' & ' + ' & '.join(self.condition) if self.condition else ''} -> {self.effect!r}"

    def to_code(self, grammar: "Grammar", name: str | None = None) -> str:
        """The source of a pure function `def law_<n>(s, a)` → (True, effect) where this law applies, else (False,
        None), from the atoms' code. ValueError naming the atoms without code, or for an action / effect that is not a
        literal."""
        from ._compile import law_to_code
        return law_to_code(self, grammar, name)


@dataclass
class Step:
    """One transition of the loop: what was predicted (and by which law), whether it was a mismatch, how many
    candidates were proposed, which laws were accepted and which retired."""
    transition: Transition
    predicted: Any
    law: Law | None
    mismatch: bool
    candidates: int = 0
    accepted: list = field(default_factory=list)
    retired: list = field(default_factory=list)


# ───────────────────────────────────────────── proposers
class Proposer(Protocol):
    """Proposes candidate laws for a mismatch: `propose(lawset, state, action, observed)` → candidates. It proposes;
    the held-out test decides."""
    name: str

    def propose(self, lawset: "LawSet", state, action, observed) -> Iterable[Candidate]: ...


class Enumerate:
    """The default: every condition the grammar builds from the atoms true in the mismatched state, with every effect
    variant of the observation (lawset.effects)."""
    name = "enumerate"

    def propose(self, lawset, state, action, observed):
        conds = lawset.grammar.conditions(lawset.grammar.true(state, action))
        return [Candidate(action, c, e, self.name) for e in lawset.effects(state, observed) for c in conds]


class RandomProposer:
    """The control: `n` conditions drawn from the whole grammar (default: as many as Enumerate would propose for this
    mismatch, with the same mix of sizes), each with an effect variant of the observation. Seeded."""
    name = "random"

    def __init__(self, seed: int = 0, n: int | None = None):
        self.rng = random.Random(seed)
        self.n = n

    def propose(self, lawset, state, action, observed):
        g = lawset.grammar
        sizes = g.sizes(g.true(state, action))
        n = self.n if self.n is not None else sum(sizes)
        effs = list(lawset.effects(state, observed))
        out = []
        for _ in range(n):
            size = self.rng.choices(range(len(sizes)), weights=sizes)[0] if sum(sizes) else 0
            out.append(Candidate(action, g.random_condition(self.rng, size), self.rng.choice(effs), self.name))
        return out


class CallableProposer:
    """Any proposer as a function — an LLM, a person's list: `fn(state, action, observed, true_atoms, grammar)` →
    conditions (iterables of atom names). A condition the grammar refuses is dropped and counted (`dropped`). On a
    game simulator with written mechanics, an LLM proposer under the same test ended 0.95 points below enumeration:
    use one where the grammar is too large to enumerate, never as the judge."""

    def __init__(self, fn: Callable, name: str = "callable"):
        self.fn, self.name, self.dropped = fn, name, 0

    def propose(self, lawset, state, action, observed):
        g = lawset.grammar
        out = []
        for cond in self.fn(state, action, observed, g.true(state, action), g) or ():
            try:
                c = g.check(cond)
            except (ValueError, TypeError):
                self.dropped += 1
                continue
            out += [Candidate(action, c, e, self.name) for e in lawset.effects(state, observed)]
        return out


# ───────────────────────────────────────────── the law set
def _identity(effect, state):
    return effect


def _observed_only(state, observed):
    return [observed]


class LawSet:
    """Laws over one grammar, most specific first (see the package docstring).

    base: the predictor when no law applies, `base(state, action)` (default: None). apply: `apply(effect, state)` → the
    predicted observation (default: the effect is the observation). effects: `effects(state, observed)` → the effect
    variants an observation supports (default: [observed]). proposer: default Enumerate(). The acceptance: support ≥
    min_support, precision ≥ min_precision and a net error reduction > 0 on the held-out rows; retirement: running
    held-out precision < retire_below over at least retire_min cases; at most `cap` active laws."""

    def __init__(self, grammar: Grammar, *, base: Callable | None = None, apply: Callable | None = None,
                 effects: Callable | None = None, proposer: Proposer | None = None, min_support: int = 5,
                 min_precision: float = 0.9, retire_below: float = 0.8, retire_min: int = 5, cap: int = 300):
        self.grammar = grammar
        self.base = base
        self.apply = apply or _identity
        self.effects = effects or _observed_only
        self.proposer = proposer if proposer is not None else Enumerate()
        self.min_support, self.min_precision = int(min_support), float(min_precision)
        self.retire_below, self.retire_min, self.cap = float(retire_below), int(retire_min), int(cap)
        self.laws: list[Law] = []          # active, priority order
        self.retired: list[Law] = []
        self.capped = 0                    # tests refused because the cap was reached
        self.transition: Transition | None = None   # the transition `step` is proposing for (a proposer may read it)
        self._seq = 0
        self._base_key = ("base", id(base))

    # ----------------------------------------------------------------------------------- prediction
    def all(self) -> list[Law]:
        """Every law ever accepted, in acceptance order."""
        return sorted(self.laws + self.retired, key=lambda law: law.seq)

    def applied(self, state, action) -> Law | None:
        """The law that speaks for `action` in `state`: the first matching active law (most specific first, ties to the
        earlier one), or None."""
        true = None
        k = _key(action)
        for law in self.laws:
            if _key(law.action) != k:
                continue
            if true is None:
                true = set(self.grammar.true(state, action))
            if law.holds(true):
                return law
        return None

    def predict(self, state, action, base: Callable | None = None):
        """The first matching law's effect applied to the state, or the base predictor's answer."""
        law = self.applied(state, action)
        if law is not None:
            return self.apply(law.effect, state)
        b = base if base is not None else self.base
        return b(state, action) if b is not None else None

    def hypothesis(self, candidate: Candidate) -> Callable:
        """The predictor this set would be with `candidate` accepted (nothing is changed): for the probe planner."""
        extra = Law(id="?", action=candidate.action, condition=tuple(candidate.condition), effect=candidate.effect,
                    seq=self._seq + 1)
        laws = sorted(self.laws + [extra], key=lambda law: (-len(law.condition), law.seq))

        def predict(state, action):
            true = set(self.grammar.true(state, action))
            for law in laws:
                if _key(law.action) == _key(action) and law.holds(true):
                    return self.apply(law.effect, state)
            return self.base(state, action) if self.base is not None else None
        return predict

    # ----------------------------------------------------------------------------------- mismatch → candidates
    def mismatch(self, state, action, observed) -> list[Candidate]:
        """No mismatch (the prediction is the observation) → []; else the proposer's candidates, without the laws
        already active."""
        if self.predict(state, action) == observed:
            return []
        active = {(_key(law.action), law.condition, _key(law.effect)) for law in self.laws}
        return [c for c in self.proposer.propose(self, state, action, observed)
                if (_key(c.action), tuple(c.condition), _key(c.effect)) not in active]

    # ----------------------------------------------------------------------------------- the held-out test
    def _effect_mask(self, buffer, action, effect):
        return buffer.where(action, ("effect", id(self.apply)), _key(effect),
                            lambda t: self.apply(effect, t.state) == t.observed)

    def _view(self, buffer, action, exclude=None):
        """Over the rows of `action` (without the group `exclude`): cov[s] = rows some active law of ≥ s atoms matches,
        ok = rows the set predicts right, take = law id → rows where that law is the applied one."""
        allowed = buffer.allowed(action, exclude)
        rem, ok = allowed, 0
        cov = [0] * (self.grammar.k + 2)
        take = {}
        for law in self.laws:
            if _key(law.action) != _key(action):
                continue
            m = buffer.match(action, law.condition) & allowed
            t = m & rem
            take[law.id] = t
            ok |= t & self._effect_mask(buffer, action, law.effect)
            rem &= ~m
            for s in range(min(len(law.condition), self.grammar.k) + 1):
                cov[s] |= m
        if self.base is not None:
            base = self.base
            ok |= rem & buffer.where(action, self._base_key, None, lambda t: base(t.state, t.action) == t.observed)
        return allowed, cov, ok, take

    def score(self, candidate: Candidate, buffer: Buffer, exclude=None, _view=None) -> tuple[int, int, int]:
        """(support, hits, net) of a candidate on the held-out rows (every group but `exclude`): the rows its condition
        matches, those its effect predicts right, and the errors it removes minus the errors it adds where it would be
        the applied law."""
        allowed, cov, ok, _ = _view or self._view(buffer, candidate.action, exclude)
        cond = tuple(candidate.condition)
        m = buffer.match(candidate.action, cond) & allowed
        E = self._effect_mask(buffer, candidate.action, candidate.effect)
        take = m & ~cov[min(len(cond), self.grammar.k + 1)]
        return m.bit_count(), (m & E).bit_count(), (take & E).bit_count() - (take & ok).bit_count()

    def test(self, candidates: Iterable[Candidate], buffer: Buffer, exclude=None, accept: int = 1) -> list[Law]:
        """Accept up to `accept` candidates (default one: the best) that pass on the held-out rows of every group but
        `exclude` (the proposing transition's): support ≥ min_support, precision ≥ min_precision, net > 0. Best: the
        largest net, then fewer atoms, then the higher precision, then the earlier candidate. → the accepted laws."""
        candidates = list(candidates)
        if not candidates:
            return []
        if len(self.laws) >= self.cap:
            self.capped += 1
            return []
        out = []
        for _ in range(accept):
            views = {}
            best = None
            active = {(_key(law.action), law.condition, _key(law.effect)) for law in self.laws}
            for i, c in enumerate(candidates):
                cond = tuple(c.condition)
                if (_key(c.action), cond, _key(c.effect)) in active:
                    continue
                a = _key(c.action)
                if a not in views:
                    views[a] = self._view(buffer, c.action, exclude)
                sup, hits, net = self.score(c, buffer, exclude, _view=views[a])
                if sup < self.min_support or hits < self.min_precision * sup or net <= 0:
                    continue
                key = (net, -len(cond), hits / sup, -i)
                if best is None or key > best[0]:
                    best = (key, c, sup, hits, net)
            if best is None or len(self.laws) >= self.cap:
                break
            _, c, sup, hits, net = best
            m = buffer.match(c.action, tuple(c.condition)) & views[_key(c.action)][0]
            law = self._accept(c, born=exclude, support=sup, hits=hits, net=net,
                               sources=tuple(buffer.ids(c.action, m)))
            out.append(law)
            self.check(buffer, c.action)
        return out

    def add(self, candidate: Candidate, *, born=None) -> Law:
        """Put a law in without the test (a person's law, a written rule, a test fixture) → the law."""
        return self._accept(candidate, born=born)

    def _accept(self, c, *, born, support=0, hits=0, net=0, sources=()):
        self._seq += 1
        law = Law(id=f"law-{self._seq}", action=c.action, condition=tuple(c.condition), effect=c.effect,
                  support=support, hits=hits, precision=(hits / support) if support else None, net=net, born=born,
                  proposer=c.proposer, sources=sources, seq=self._seq)
        self.laws.append(law)
        self.laws.sort(key=lambda x: (-len(x.condition), x.seq))
        return law

    # ----------------------------------------------------------------------------------- retirement
    def check(self, buffer: Buffer, action=None) -> list[Law]:
        """Recount each law's held-out rows on `buffer` (rows of other groups than its birth where it is the applied
        law) and retire the laws whose running precision fell below retire_below → the retired laws."""
        acts = [action] if action is not None else list({_key(law.action): law.action for law in self.laws}.values())
        retired = []
        for a in acts:
            while True:
                _, _, _, take = self._view(buffer, a)
                gone = None
                for law in self.laws:
                    if law.id not in take:
                        continue
                    t = take[law.id] & ~buffer.group_mask(a, law.born) if law.born is not None else take[law.id]
                    law.held = ((t & self._effect_mask(buffer, a, law.effect)).bit_count(), t.bit_count())
                    if gone is None and self._falls(law):
                        gone = law
                if gone is None:
                    break
                h, n = gone.running()
                self.retire(gone, why=f"held-out precision {h}/{n} below {self.retire_below}")
                retired.append(gone)
        return retired

    def _falls(self, law):
        h, n = law.running()
        return n >= self.retire_min and h < self.retire_below * n

    def observe(self, t: Transition) -> list[Law]:
        """Count a transition against the law applied to it (unless it comes from the law's birth group) and retire
        the law if its running precision fell → the retired laws."""
        law = self.applied(t.state, t.action)
        if law is None or (law.born is not None and _key(t.group) == _key(law.born)):
            return []
        law.n += 1
        law.hits_after += int(self.apply(law.effect, t.state) == t.observed)
        if self._falls(law):
            h, n = law.running()
            self.retire(law, why=f"held-out precision {h}/{n} below {self.retire_below}")
            return [law]
        return []

    def retire(self, law: Law, *, why: str | None = None):
        """Take a law out of use (kept in `retired` with the reason)."""
        if law in self.laws:
            self.laws.remove(law)
            law.status, law.why = "retired", why
            self.retired.append(law)

    # ----------------------------------------------------------------------------------- the loop
    def step(self, t: Transition, buffer: Buffer, *, remember: bool = False) -> Step:
        """One transition of the loop: predict, count it against the applied law (retire), on a mismatch propose and
        test on the buffer without this transition's group; `remember`: then add it to the buffer."""
        law = self.applied(t.state, t.action)
        predicted = self.predict(t.state, t.action)
        retired = self.observe(t)
        miss = predicted != t.observed
        st = Step(t, predicted, law, miss, retired=retired)
        if miss:
            self.transition = t
            try:
                cands = self.mismatch(t.state, t.action, t.observed)
            finally:
                self.transition = None
            st.candidates = len(cands)
            st.accepted = self.test(cands, buffer, exclude=t.group) if cands else []
        if remember:
            buffer.add(t)
        return st

    def learn(self, stream: Iterable[Transition], buffer: Buffer, *, remember: bool = False) -> list[Step]:
        """The loop over a stream, in order (not a schedule: every transition is predicted before it is learned from)."""
        return [self.step(t, buffer, remember=remember) for t in stream]

    # ----------------------------------------------------------------------------------- compiled
    def to_module(self) -> str:
        """The active laws as one Python module that imports nothing from solvi: one function per law, `LAWS` in this
        set's dispatch order (most specific first), `applied(s, a)` and `predict(s, a, base=None, apply=None)` with
        the base predictor as the fallback. Check it with `check_compiled(module, buffer, lawset)`. ValueError naming
        every atom without code."""
        from ._compile import lawset_to_module
        return lawset_to_module(self)

    # ----------------------------------------------------------------------------------- knowledge
    def to_knowledge(self, store, scope: dict | None = None) -> dict[str, str]:
        """Write the laws into a KnowledgeStore: each a "rule" item with body {"kind": "law", "action", "if", "then"},
        source "outcome" (the environment answered), by "laws:<proposer>", evidence = support, hits, precision, net,
        the birth group and the ids of the held-out transitions it was accepted on. A retired law is retire()d in the
        store with its reason. Idempotent: a law already there as it is writes nothing. → law id → item id."""
        from ...core.knowledge.store import item_id, plain_json
        out = {}
        for law in self.all():
            body = {"kind": "law", "action": _plain(law.action), "if": list(law.condition), "then": _plain(law.effect)}
            sc = plain_json(scope or {})
            iid = item_id("rule", plain_json(body), sc)
            st = store.status(iid)
            if st is None or (law.status == "active" and st != "active"):
                ev = {"support": law.support, "hits": law.hits, "precision": law.precision, "net": law.net,
                      "born": _plain(law.born), "transitions": list(law.sources), "law": law.id}
                got = store.add("rule", body, sc, source="outcome", by=f"laws:{law.proposer}", evidence=ev)
                if got is None:
                    continue
                st = store.status(iid)
            if law.status == "retired" and st != "expired":
                store.retire(iid, why=law.why or "retired")
            out[law.id] = iid
        return out


def _plain(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)


__all__ = ["Atom", "Buffer", "CallableProposer", "Candidate", "Enumerate", "Grammar", "Law", "LawSet", "Proposer",
           "RandomProposer", "Step", "Transition"]
