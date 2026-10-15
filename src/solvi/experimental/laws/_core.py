"""The grammar, the transitions and their held-out buffer, the laws, the law set and its proposers (see the package
docstring)."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
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
    is left out of every test (`exclude=`). tested: how many distinct candidates (action, condition, effect) have been
    scored on this buffer, by every law set that tested on it — the N the sized acceptance divides its error by; scoring
    the same candidate again does not raise it."""

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
        self.tested = 0                                          # distinct candidates scored on this buffer, in all
        self._tested: set = set()                                # their (action, condition, effect) keys

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

    def count_tested(self, keys: Iterable) -> int:
        """Add candidate keys (action, condition, effect) to the tested ones → how many were new (N grows by these)."""
        new = 0
        for k in keys:
            if k not in self._tested:
                self._tested.add(k)
                new += 1
        self.tested += new
        return new

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
    acceptance (rows of every group but `born`), counted on the rows where it would be the applied law (the set's
    `support_on`); sources: the ids of the rows it was accepted on (the same rows); n / hits_after:
    observations from other groups since, while it was the law applied; held: (hits, n) on the buffer rows where it is
    the applied law, from the last test; status: "active" | "retired"; why: the retirement's reason. tested: the
    candidates scored on the buffer when it was accepted (with the sized acceptance, the N of its bound; admitted by
    probes: the candidates it was told apart from); null: the null precision it was tested against (the effect's
    base rate, or the set's fixed null); need: the clean support the sized rule asked for at that N; level:
    the per-candidate error level δ_N it passed (None: fixed thresholds, `sized=False`, or a law added by hand);
    admitted: "test" (the held-out buffer), "probes" (`admit_by_probes`) or "hand" (`add`)."""
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
    tested: int = 0
    need: int = 0
    level: float | None = None
    null: float | None = None
    admitted: str = "test"

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
    candidates were proposed and how many of them were scored on the buffer (tested), which laws were accepted and
    which retired — by this transition's observation, or by the check the test runs after an acceptance (a law accepted
    and retired by that check is in both lists)."""
    transition: Transition
    predicted: Any
    law: Law | None
    mismatch: bool
    candidates: int = 0
    tested: int = 0
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


# ───────────────────────────────────────────── the sized acceptance
def sized_level(tested: int, delta: float, space: int | None = None) -> float:
    """The error level one candidate is tested at when `tested` candidates have been scored on the buffer: δ / space
    when you declared the number of candidates you will test (`space`, Bonferroni; valid while tested ≤ space), else
    δ / (N·(N + 1)) with N = tested — the n-th candidate gets at most δ / (n·(n + 1)) and these sum to less than δ over
    any number of candidates, so the bound holds however long the stream runs."""
    n = max(1, int(tested))
    if space is not None and n <= int(space):
        return float(delta) / int(space)
    return float(delta) / (n * (n + 1))


def sized_need(tested: int, delta: float, null: float, space: int | None = None) -> int:
    """The smallest support with which a law that is right on every row passes the sized rule at this N: the least s
    with null**s ≤ level, i.e. ⌈ln(1/level) / ln(1/null)⌉ — about (2·ln N + ln 1/δ) / ln(1/null) without a declared
    space. At N = 52,000, δ = 0.1: 9 at null 0.05 (a rare effect's base rate, floored), 68 at 0.7, 228 at 0.9."""
    lvl = sized_level(tested, delta, space)
    return max(1, math.ceil(math.log(1.0 / lvl) / math.log(1.0 / float(null)) - 1e-9))


@lru_cache(maxsize=65536)
def _log_tail(s: int, h: int, p: float) -> float:
    """log P[Binomial(s, p) ≥ h]."""
    if h <= 0:
        return 0.0
    if h > s:
        return -math.inf
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma(s + 1)
    terms = [lg - math.lgamma(j + 1) - math.lgamma(s - j + 1) + j * lp + (s - j) * lq for j in range(h, s + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(x - m) for x in terms))


@dataclass(frozen=True, eq=False)
class Undetermined:
    """The answer of `predict(..., disagree="abstain")` where laws that are not exceptions of one another disagree:
    the laws (the most specific matching ones, none of whose conditions contains another's) and what each predicts.
    Never equal to an observation."""
    laws: tuple
    predictions: tuple

    def __repr__(self):
        return f"Undetermined({[law.id for law in self.laws]}, predictions={list(self.predictions)!r})"


@dataclass
class Admission:
    """What `admit_by_probes` decided. status: "accepted" (one equivalence class of candidates survived and it is
    not the set without a law) or "undetermined"; law: the accepted law (the shortest candidate of the surviving
    class) or None; classes: the surviving candidates grouped by what they predict on every probe offered (shortest
    first in each, classes by their shortest); asked: the probes asked, as Transitions (group "probe"); refuted: the
    candidates the logged rows and the probes refuted (a count); null: the set without any new law still survives;
    why: the reason; witnesses: the candidates refuted only as too narrow that are right wherever they were applied and
    predict otherwise than the surviving class where they apply — while any is left nothing is accepted (several laws
    may be missing)."""
    status: str
    law: Law | None
    classes: list
    asked: list
    refuted: int
    null: bool
    why: str
    witnesses: list = field(default_factory=list)


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
    min_support, precision ≥ min_precision and a net error reduction > 0 on the held-out rows, and — sized=True, the
    default — the sized rule below; retirement: running held-out precision < retire_below over at least retire_min
    cases; at most `cap` active laws. min_precision below retire_below is warned at construction: a law that passes
    with a precision between the two is retired by the check its own test runs, and the next mismatch accepts it again.

    support_on: the rows a candidate's support, hits and precision are counted on — "applied" (the default): where it
    would be the applied law, its condition's rows less those a more specific (or, at equal length, one `ties` puts
    first) active law already claims, the rows its net is counted on too; "matched": every row its condition matches
    (before 1.2). A default whose exceptions are already laws is judged on the rows they leave it: a default right on
    45 of its 45 rows with 10 rows of a known exception under it is 45 / 45, not 45 / 55 (refused at min_precision 0.9
    under "matched"), and its precision at acceptance is the precision its first `check` recounts.
    readmit: "any" (the default) — a retired law may be accepted again by the next test; "grown" — not on a buffer
    that has not grown since its retirement (the same evidence that retired it; refusals counted in
    `refused_readmit`). A retirement from `observe` outside `step` does not see the buffer: it is stamped with the
    buffer's size at the next `test`, which errs toward waiting.

    The sized acceptance (sized=True). Fixed thresholds do not see how many candidates were tried on one buffer: on a
    buffer with no law at all (2,000 rows, an effect at 70% independent of 40 sparse atoms) the 821 conditions of at
    most two atoms passed support ≥ 5 and precision ≥ 0.9 in 100 of 100 buffers. The sized rule tests each candidate
    against a null at the level δ_N = sized_level(N, delta, space), N the candidates scored on this buffer so far
    (`buffer.tested`): it passes when P[Binomial(support, null) ≥ hits] ≤ δ_N. The null (`null=`): "base" (the
    default) — "the condition is no better than chance": the effect's base rate, the share of the buffer's rows of
    that action (the proposing group left out) whose observation the candidate's effect predicts, floored at 0.05 and
    capped at min_precision (0.95 when that is 1) — or a fixed float (the precision the law must beat). min_precision
    stays a separate filter on the observed precision. A law right on every row needs support ≥ sized_need(N, δ,
    null) = ⌈ln(1/δ_N) / ln(1/null)⌉: at 52,000 candidates and δ = 0.1, 9 rows at a base rate of 3% (floored to 5%),
    68 at 70%, and 228 against a fixed null of 0.9; every row it gets wrong asks for more. Then a candidate no better
    than its null passes with probability ≤ δ_N, and over every candidate tested on the buffer the chance of any such
    acceptance is ≤ delta — on the same null buffers: 0 of 100. With the base null that statement is "better than
    chance", not "precision above min_precision" (the filter adds that on the observed rows). The bound assumes the
    held-out rows are independent draws and the candidates were not chosen by looking at them. sized=False: the
    fixed thresholds (the measurements ran so). Each accepted Law records `tested` (N), `null`, `need` and
    `level`. N counts distinct candidates: the same (action, condition, effect) scored again on the buffer — the next
    mismatch proposing it again — does not raise it (it did before 1.2: the same 56 candidates scored three times on an
    unchanged buffer made N 168).

    acceptance: "sized" (the default with sized=True), "fixed" (sized=False) or "exact" — for few-shot buffers, where
    the sized rule cannot pass anything (3 groups × 6 rows, a deterministic one-atom law: 0 of 20 learned): a candidate
    passes when it is right on every held-out row where it applies, with support ≥ min_support, and net > 0; no
    binomial bound, so nothing bounds the false accepts over many candidates (with min_support=2 the same few-shot task
    learned the law in 20 of 20) — give it an independent check through `confirm`. Overrides `sized`.
    confirm: `confirm(candidate)` → bool, an external check (a probe, a person, a second data set) asked about each
    candidate that passed, best first, before it is accepted; a refusal sets the candidate aside for this test.

    ties: the order between laws of equal length — "earlier" (the default: the earlier accepted law speaks) or "newer"
    (the later one does: a law learned after the world changed is not hidden behind an older one of the same length).
    The order is used everywhere a law is chosen: `applied`/`predict`, the rows a candidate would take in the test,
    `hypothesis` and `admit_by_probes`; more specific laws always come first.

    disagree: "first" (default: the most specific matching law, ties by `ties`) or "abstain" (`predict` returns
    `Undetermined` where laws none of which is an exception of another disagree; see `predict`).

    tol / rtol / same: how a prediction is compared with an observation — and two predictions with each other — in
    `step`, `mismatch`, `test` (support, hits and net), `observe`, `check`, `conflict` and, unless the call gives its
    own, `admit_by_probes`: exact equality by default; numbers within an absolute `tol` and a relative `rtol`
    (|a − b| ≤ tol + rtol·max(|a|, |b|); tuples and lists element by element), or `same(a, b)` → bool. For a numeric
    effect measured with noise: y = 2.5·x over x from 1 to 1e5 with a relative noise of 1e-7 is a mismatch on every
    row under exact equality (no law learned in 60 steps); with rtol=1e-5 the law is learned and no later row is a
    mismatch (tests/test_laws_rtol.py).

    cost: the search-cost record — candidates (proposed for mismatches), tests (candidates scored on a buffer),
    accepted, probes (asked by `admit_by_probes`), simulated (0 here; a model-based learner adds its steps) — report it
    next to accuracy."""

    def __init__(self, grammar: Grammar, *, base: Callable | None = None, apply: Callable | None = None,
                 effects: Callable | None = None, proposer: Proposer | None = None, min_support: int = 5,
                 min_precision: float = 0.9, retire_below: float = 0.8, retire_min: int = 5, cap: int = 300,
                 sized: bool = True, delta: float = 0.1, space: int | None = None,
                 null: str | float = "base", disagree: str = "first", support_on: str = "applied",
                 readmit: str = "any", acceptance: str | None = None, confirm: Callable | None = None,
                 ties: str = "earlier", tol: float | None = None, rtol: float | None = None,
                 same: Callable[[Any, Any], bool] | None = None):
        if disagree not in ("first", "abstain"):
            raise ValueError(f"disagree is 'first' or 'abstain', not {disagree!r}")
        if not 0.0 < float(delta) < 1.0:
            raise ValueError("delta is a probability strictly between 0 and 1")
        self.grammar = grammar
        self.base = base
        self.apply = apply or _identity
        self.effects = effects or _observed_only
        self.proposer = proposer if proposer is not None else Enumerate()
        self.min_support, self.min_precision = int(min_support), float(min_precision)
        self.retire_below, self.retire_min, self.cap = float(retire_below), int(retire_min), int(cap)
        if self.min_precision < self.retire_below:
            warnings.warn(f"LawSet: min_precision {self.min_precision} < retire_below {self.retire_below} — a law "
                          f"accepted at a precision between the two is retired by its own check and accepted again "
                          f"on the next mismatch", UserWarning, stacklevel=2)
        if acceptance is None:
            acceptance = "sized" if sized else "fixed"
        if acceptance not in ("sized", "fixed", "exact"):
            raise ValueError(f"acceptance is 'sized', 'fixed' or 'exact', not {acceptance!r}")
        self.acceptance, self.confirm = acceptance, confirm
        self.sized, self.delta = acceptance == "sized", float(delta)
        self.space = int(space) if space is not None else None
        if isinstance(null, str):
            if null != "base":
                raise ValueError(f"null is 'base' or a precision strictly between 0 and 1, not {null!r}")
        elif not 0.0 < float(null) < 1.0:
            raise ValueError("a fixed null is a precision strictly between 0 and 1")
        else:
            null = float(null)
        self.null = null
        self.disagree = disagree
        if support_on not in ("applied", "matched"):
            raise ValueError(f"support_on is 'applied' or 'matched', not {support_on!r}")
        if readmit not in ("any", "grown"):
            raise ValueError(f"readmit is 'any' or 'grown', not {readmit!r}")
        self.support_on, self.readmit = support_on, readmit
        if ties not in ("earlier", "newer"):
            raise ValueError(f"ties is 'earlier' or 'newer', not {ties!r}")
        self.ties = ties
        from ._probes import sameness
        self.same = sameness(tol, same, rtol=rtol)          # None: exact equality
        self._cmp = None if self.same is None else (("same", id(same)) if same is not None else ("tol", tol, rtol))
        self.refused_readmit = 0           # candidates refused because they were retired on a buffer of this size
        self._retired_at: dict = {}        # (action, condition, effect) → buffer size at retirement (None: not seen)
        self.abstained = 0                 # predictions that abstained (disagree="abstain")
        self.cost = {"candidates": 0, "tests": 0, "accepted": 0, "probes": 0, "simulated": 0}
        self.laws: list[Law] = []          # active, priority order
        self.retired: list[Law] = []
        self.capped = 0                    # tests refused because the cap was reached
        self.last_retired: list[Law] = []  # the laws the check after the last test's acceptances retired
        self.transition: Transition | None = None   # the transition `step` is proposing for (a proposer may read it)
        self._seq = 0
        self._last_tested = 0
        self._base_key = ("base", id(base)) if self._cmp is None else ("base", id(base), self._cmp)

    # ----------------------------------------------------------------------------------- prediction
    def agrees(self, predicted, observed) -> bool:
        """A prediction is the observation: == by default, else within the set's tol / rtol or by its `same`."""
        return predicted == observed if self.same is None else bool(self.same(predicted, observed))

    def all(self) -> list[Law]:
        """Every law ever accepted, in acceptance order."""
        return sorted(self.laws + self.retired, key=lambda law: law.seq)

    def applied(self, state, action) -> Law | None:
        """The law that speaks for `action` in `state`: the first matching active law (most specific first, ties by the
        set's `ties`), or None."""
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

    def matching(self, state, action) -> list[Law]:
        """Every active law of `action` whose condition holds in `state`, in the set's order (most specific first)."""
        true = set(self.grammar.true(state, action))
        k = _key(action)
        return [law for law in self.laws if _key(law.action) == k and law.holds(true)]

    def conflict(self, state, action) -> Undetermined | None:
        """Where the matching laws that are not exceptions of one another disagree → Undetermined, else None. Of the
        matching laws only the maximal ones count — those whose condition no other matching law's condition strictly
        contains (a law with a larger condition is an exception that was tested where the wider one applied); when
        these predict different observations here, nothing in the evidence says which one speaks."""
        ms = self.matching(state, action)
        if len(ms) < 2:
            return None
        sets = [set(law.condition) for law in ms]
        top = [law for law, c in zip(ms, sets) if not any(o > c for o in sets)]
        preds = [self.apply(law.effect, state) for law in top]
        if self.same is None:
            if len({_key(p) for p in preds}) < 2:
                return None
        else:
            from ._probes import groups_of
            if len(groups_of(preds, self.same)) < 2:
                return None
        return Undetermined(tuple(top), tuple(preds))

    def predict(self, state, action, base: Callable | None = None, *, disagree: str | None = None):
        """The first matching law's effect applied to the state, or the base predictor's answer. disagree="abstain"
        (or the set's own `disagree`): where the maximal matching laws disagree (`conflict`), an `Undetermined` instead
        — the abstention is counted in `abstained`. The default "first" keeps most specific first, ties by the set's
        `ties`.

        A version space in miniature: two laws that both passed their test and neither of which is an exception of
        the other cover this state and say different things — the order between them is a tie-break, not evidence."""
        mode = disagree or self.disagree
        if mode not in ("first", "abstain"):
            raise ValueError(f"disagree is 'first' or 'abstain', not {mode!r}")
        if mode == "abstain":
            u = self.conflict(state, action)
            if u is not None:
                self.abstained += 1
                return u
        law = self.applied(state, action)
        if law is not None:
            return self.apply(law.effect, state)
        b = base if base is not None else self.base
        return b(state, action) if b is not None else None

    def hypothesis(self, candidate: Candidate) -> Callable:
        """The predictor this set would be with `candidate` accepted (nothing is changed): for the probe planner."""
        extra = Law(id="?", action=candidate.action, condition=tuple(candidate.condition), effect=candidate.effect,
                    seq=self._seq + 1)
        laws = sorted(self.laws + [extra], key=self._order)

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
        if self.agrees(self.predict(state, action, disagree="first"), observed):
            return []
        active = {(_key(law.action), law.condition, _key(law.effect)) for law in self.laws}
        proposed = list(self.proposer.propose(self, state, action, observed))
        self.cost["candidates"] += len(proposed)
        return [c for c in proposed if (_key(c.action), tuple(c.condition), _key(c.effect)) not in active]

    # ----------------------------------------------------------------------------------- the held-out test
    def _effect_mask(self, buffer, action, effect):
        what = ("effect", id(self.apply)) if self._cmp is None else ("effect", id(self.apply), self._cmp)
        return buffer.where(action, what, _key(effect),
                            lambda t: self.agrees(self.apply(effect, t.state), t.observed))

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
            ok |= rem & buffer.where(action, self._base_key, None, lambda t: self.agrees(base(t.state, t.action), t.observed))
        return allowed, cov, ok, take

    def _rows_of(self, candidate, buffer, exclude=None, _view=None):
        """(counted, take, E, ok): the rows support is counted on (`support_on`), the rows where the candidate would be
        the applied law, the rows its effect predicts right, the rows the set predicts right now."""
        allowed, cov, ok, _ = _view or self._view(buffer, candidate.action, exclude)
        cond = tuple(candidate.condition)
        m = buffer.match(candidate.action, cond) & allowed
        E = self._effect_mask(buffer, candidate.action, candidate.effect)
        take = m & ~cov[min(len(cond) + (self.ties == "newer"), self.grammar.k + 1)]
        return (take if self.support_on == "applied" else m), take, E, ok

    def score(self, candidate: Candidate, buffer: Buffer, exclude=None, _view=None) -> tuple[int, int, int]:
        """(support, hits, net) of a candidate on the held-out rows (every group but `exclude`): the rows it is counted
        on — where it would be the applied law (support_on="applied", the default), or every row its condition matches
        ("matched") — those of them its effect predicts right, and the errors it removes minus the errors it adds where
        it would be the applied law."""
        m, take, E, ok = self._rows_of(candidate, buffer, exclude, _view)
        return m.bit_count(), (m & E).bit_count(), (take & E).bit_count() - (take & ok).bit_count()

    NULL_FLOOR = 0.05

    def base_rate(self, buffer: Buffer, action, effect, exclude=None) -> float:
        """The base null for (action, effect): the share of the buffer's rows of `action` (without the group
        `exclude`) whose observation `effect` predicts, floored at 0.05 and capped at min_precision (0.95 when that is
        1); the floor when the buffer has no row of the action."""
        allowed = buffer.allowed(action, exclude)
        n = allowed.bit_count()
        cap = self.min_precision if self.min_precision < 1.0 else 0.95
        if not n:
            return min(self.NULL_FLOOR, cap)
        share = (allowed & self._effect_mask(buffer, action, effect)).bit_count() / n
        return min(max(share, self.NULL_FLOOR), cap)

    def passes(self, support: int, hits: int, tested: int, null: float | None = None) -> bool:
        """The acceptance on counts alone (net aside): support ≥ min_support, hits ≥ min_precision · support, and with
        sized=True P[Binomial(support, null) ≥ hits] ≤ sized_level(tested, delta, space). null: the set's fixed null
        when it has one; with null="base" pass the rate (`base_rate`). acceptance="exact": support ≥ min_support and
        hits == support."""
        if self.acceptance == "exact":
            return support >= max(1, self.min_support) and hits == support
        if support < self.min_support or hits < self.min_precision * support:
            return False
        if not self.sized:
            return True
        if null is None:
            if isinstance(self.null, str):
                raise ValueError("this set's null is the base rate: pass null=lawset.base_rate(buffer, action, effect)")
            null = self.null
        if support < sized_need(tested, self.delta, null, self.space):
            return False
        return _log_tail(int(support), int(hits), float(null)) <= math.log(
            sized_level(tested, self.delta, self.space)) + 1e-12

    def _null_of(self, buffer, c, exclude, cache):
        if not isinstance(self.null, str):
            return self.null
        k = (_key(c.action), _key(c.effect))
        if k not in cache:
            cache[k] = self.base_rate(buffer, c.action, c.effect, exclude)
        return cache[k]

    def test(self, candidates: Iterable[Candidate], buffer: Buffer, exclude=None, accept: int = 1) -> list[Law]:
        """Accept up to `accept` candidates (default one: the best) that pass on the held-out rows of every group but
        `exclude` (the proposing transition's): `passes(support, hits, N)` — support ≥ min_support, precision ≥
        min_precision and, sized, the binomial bound at N = the candidates scored on this buffer so far, this call's
        included — and net > 0. Best: the largest net, then fewer atoms, then the higher precision, then the earlier
        candidate. → the accepted laws. The check after each acceptance may retire laws (the accepted one too):
        they are in `last_retired` (and in `Step.retired` when `step` ran the test)."""
        candidates = list(candidates)
        self.last_retired = []
        if self.readmit == "grown" and self._retired_at:
            candidates = self._not_readmitted(candidates, len(buffer))
        if not candidates:
            return []
        if len(self.laws) >= self.cap:
            self.capped += 1
            return []
        active0 = {(_key(law.action), law.condition, _key(law.effect)) for law in self.laws}
        keys = [k for k in ((_key(c.action), tuple(c.condition), _key(c.effect)) for c in candidates)
                if k not in active0]
        scored = len(keys)
        buffer.count_tested(keys)
        self.cost["tests"] += scored
        self._last_tested = scored
        n_tested = buffer.tested
        nulls: dict = {}
        out = []
        refused: set = set()                       # passed, but `confirm` said no
        for _ in range(accept):
            views = {}
            passing = []
            active = {(_key(law.action), law.condition, _key(law.effect)) for law in self.laws}
            for i, c in enumerate(candidates):
                cond = tuple(c.condition)
                if (_key(c.action), cond, _key(c.effect)) in active or i in refused:
                    continue
                a = _key(c.action)
                if a not in views:
                    views[a] = self._view(buffer, c.action, exclude)
                sup, hits, net = self.score(c, buffer, exclude, _view=views[a])
                if net <= 0:
                    continue
                nl = self._null_of(buffer, c, exclude, nulls) if self.sized else None
                if not self.passes(sup, hits, n_tested, nl):
                    continue
                passing.append(((net, -len(cond), hits / sup, -i), i, c, sup, hits, net, nl))
            passing.sort(key=lambda r: r[0], reverse=True)
            best = None
            for row in passing:
                if self.confirm is None or self.confirm(row[2]):
                    best = row
                    break
                refused.add(row[1])
            if best is None or len(self.laws) >= self.cap:
                break
            _, _, c, sup, hits, net, nl = best
            m = self._rows_of(c, buffer, exclude, views[_key(c.action)])[0]
            law = self._accept(c, born=exclude, support=sup, hits=hits, net=net,
                               sources=tuple(buffer.ids(c.action, m)))
            law.tested = n_tested
            self.cost["accepted"] += 1
            if self.sized:
                law.level = sized_level(n_tested, self.delta, self.space)
                law.null = nl
                law.need = sized_need(n_tested, self.delta, nl, self.space)
            out.append(law)
            self.last_retired += self.check(buffer, c.action)
        return out

    def _not_readmitted(self, candidates, size):
        """The candidates without the laws retired on a buffer of this size or larger (readmit="grown"); retirements
        not yet stamped are stamped with `size`."""
        for k, at in self._retired_at.items():
            if at is None:
                self._retired_at[k] = size
        out = []
        for c in candidates:
            at = self._retired_at.get((_key(c.action), tuple(c.condition), _key(c.effect)))
            if at is not None and size <= at:
                self.refused_readmit += 1
                continue
            out.append(c)
        return out

    def _stamp(self, laws, size):
        if self.readmit == "grown":
            for law in laws:
                self._retired_at[(_key(law.action), law.condition, _key(law.effect))] = size

    def add(self, candidate: Candidate, *, born=None) -> Law:
        """Put a law in without the test (a person's law, a written rule, a test fixture) → the law."""
        law = self._accept(candidate, born=born)
        law.admitted = "hand"
        return law

    # ----------------------------------------------------------------------------------- admission by probes
    def admit_by_probes(self, candidates: Iterable[Candidate], ask: Callable[[Any, Any], Any], *,
                        probes: Iterable[tuple[Any, Any]], safe: Callable[[Any, Any], bool] | None = None,
                        budget: int = 8, buffer: Buffer | None = None, exclude=None,
                        read_only: Callable[..., bool] | None = None, cost: Callable[[Any, Any], Any] | None = None,
                        order: str = "split", unique: bool = True, seed: int = 0, tol: float | None = None,
                        same: Callable[[Any, Any], bool] | None = None, assume_one_law: bool = False,
                        rtol: float | None = None) -> Admission:
        """Admit one of `candidates` by asking the world where they disagree (CEGIS-style), not by a random held-out
        sample. ask(state, action) → the observation (the environment, or an exact model of it); probes: the (state,
        action) pairs you may ask about (a finite pool; also what the equivalence classes are computed on); safe:
        which of them may be asked (default: all); budget: at most this many asks.

        A candidate is the set with it added (`hypothesis`): where it is the applied law it predicts its effect,
        elsewhere what the set predicts now. It is refuted by a row (a logged one from `buffer` — every group but
        `exclude`, its own actions only — or an asked probe) where it is the applied law and its effect is wrong, or
        where its condition does not hold, the set is wrong and its effect is what was observed (it is too narrow).
        The set without a new law ("null") is a hypothesis too, refuted where the set is wrong. Loop: group the
        surviving hypotheses by their predictions on every probe of the pool; while more than one group is left and
        the budget lasts, ask the safe probe that splits the groups most (`split`: read-only first, the larger split,
        the lower cost), drop the refuted. Accept the shortest candidate (fewest atoms, then the order given) only when
        one group of candidates is left and the null is not in it or alive (identifiability); otherwise
        "undetermined" with the surviving groups — never a guess. order="random", unique=False: the control — `budget`
        random safe probes, the shortest survivor accepted (a held-out sample of that size). tol / same: predictions
        and observations compared within an absolute tolerance (numbers; tuples element by element) or by `same(a, b)`,
        for refutation, for the classes and for the probe choice (`split`) — with exact equality a measurement noise of
        1e-6 refuted every candidate of y = c·x; with tol=1e-4 the right c is admitted. rtol: a relative tolerance
        (|a − b| ≤ tol + rtol·max(|a|, |b|)) — over x from 1 to 1e5 with a relative noise of 1e-6, tol=1e-4 refutes
        every candidate (undetermined) and rtol=1e-4 admits the right c. None of the three: the set's own comparison
        (its tol / rtol / same; exact by default).

        The "too narrow" refutation is sound only when exactly one law is missing. Where two are (the world x0 ∨ x1,
        a base that says False, 10 binary atoms, k = 2, a pool of 200 random states, 8 checks), each right law is too
        narrow where the other fires, both were refuted, and a wrong survivor — the empty condition, or an unrelated
        atom that held by chance where it was asked — was accepted as the last class: 29 of 100 trials accepted a law
        false on some state. So a candidate refuted only as too narrow and never wrong where it applied is a witness,
        and the set with every witness added is the several-laws alternative: while it is right on every row seen and
        predicts otherwise than the surviving class somewhere on the pool where the class would apply, the probes go
        on to tell the two apart (within the budget) and nothing is accepted before — on that world 6 of 100 wrong
        (77 accepted, 23 undetermined; the same at 20 checks: in the 6 the witnesses together cover the wrong survivor
        wherever it applies on the pool, so no probe tells them apart). assume_one_law=True: the narrow refutation is
        final (before 1.2).

        Measured on a synthetic world (10 binary features, 20 atoms, k = 3; the hidden law a random conjunction of two
        atoms; the 176 candidates Enumerate proposes from one state where it fires; a pool of 200 random states; 300
        trials; a wrong law = one that differs from the hidden law on any of the 1,024 states): at 8 checks a random
        held-out sample with the shortest survivor accepted took a wrong law in 0.593 of the trials (0.407 right); the
        active admission asked 7.56 probes on average and accepted 0.897 right, 0.000 wrong, 0.103 undetermined (at
        10 checks: 1.000 right in 7.66 asks; with assume_one_law=True at 8: 0.987 right, 0.013 undetermined, 7.45
        asks — the check against the several-laws alternative costs about one probe). The
        uniqueness rule alone on 8 random probes: 0.000 wrong but 0.893 undetermined; 20 random probes with the
        shortest survivor: 0.037 wrong (tests/test_laws_w2.py reruns the 8-check comparison on 150 trials). The
        accepted law records admitted="probes", tested = the number of candidates, support = the rows where it
        applied, born = "probe"."""
        if order not in ("split", "random"):
            raise ValueError(f"order is 'split' or 'random', not {order!r}")
        from ._probes import groups_of, sameness, split
        eq = sameness(tol, same, rtol=rtol) if (tol, rtol, same) != (None, None, None) else self.same
        E = eq or (lambda u, v: u == v)
        veq = None if eq is None else (lambda u, v: len(u) == len(v) and all(eq(x, y) for x, y in zip(u, v)))
        cands: list[Candidate] = []
        seen = set()
        for c in candidates:
            k = (_key(c.action), tuple(c.condition), _key(c.effect))
            if k not in seen:
                seen.add(k)
                cands.append(c)
        pool = list(probes)
        rng = random.Random(seed)

        def info(x, a):
            ms = self.matching(x, a)
            return set(self.grammar.true(x, a)), self.predict(x, a, disagree="first"), \
                max((len(law.condition) for law in ms), default=-1)

        cache: dict[int, tuple] = {}

        def pinfo(i):
            if i not in cache:
                cache[i] = info(*pool[i])
            return cache[i]

        def pred(c, x, a, inf):
            T, cur, shadow = inf
            if _key(c.action) != _key(a):
                return cur, False, False
            hold = set(c.condition) <= T
            applied = hold and (len(c.condition) >= shadow if self.ties == "newer" else len(c.condition) > shadow)
            return (self.apply(c.effect, x) if applied else cur), applied, hold

        def refutes(c, x, a, obs, inf):
            """None, "applied" (it is the applied law and its effect is wrong) or "narrow" (it does not hold, the set
            is wrong and its effect is what was observed)."""
            p, applied, hold = pred(c, x, a, inf)
            if _key(c.action) != _key(a):
                return None
            if applied:
                return "applied" if not E(p, obs) else None
            return "narrow" if (not hold) and not E(inf[1], obs) and E(self.apply(c.effect, x), obs) else None

        acts = {_key(c.action): c.action for c in cands}
        alive = list(range(len(cands)))
        null = True
        refuted = 0
        narrow: set[int] = set()        # refuted only as too narrow, never wrong where applied: the witnesses' pool

        seen: list = []                 # (state, action, observation, info) of every logged row and asked probe

        def see(x, a, obs, inf):
            nonlocal alive, refuted, null
            if not assume_one_law:
                seen.append((x, a, obs, inf))
            keep = []
            for j in alive:
                k = refutes(cands[j], x, a, obs, inf)
                if k is None:
                    keep.append(j)
                elif k == "narrow" and not assume_one_law:
                    narrow.add(j)
            refuted += len(alive) - len(keep)
            alive = keep
            for j in [j for j in narrow if refutes(cands[j], x, a, obs, inf) == "applied"]:
                narrow.discard(j)
            if null and not E(inf[1], obs):
                null = False

        if buffer is not None:
            for a in acts.values():
                for row in buffer.rows(a, exclude):
                    see(row.state, row.action, row.observed, info(row.state, row.action))
        vec: dict[int, tuple] = {}
        acting: dict[int, tuple] = {}
        pe = (lambda u, v: u == v) if eq is None else eq

        def applies(j):
            if j not in acting:
                acting[j] = tuple(pred(cands[j], pool[i][0], pool[i][1], pinfo(i))[1] for i in range(len(pool)))
            return acting[j]

        def spoken(ds, applied, value, default):
            """The several-laws alternative at one point: the most specific witness applied there, else `default`."""
            best = None
            for d in ds:
                if applied(d) and (best is None or len(cands[d].condition) > len(cands[best].condition)):
                    best = d
            return default if best is None else value(best)

        def witnesses(j0):
            """The several-laws alternative to candidate j0: the set with every witness (`narrow`) added. → (the
            witnesses, the probe target) while it is right on every row seen and predicts otherwise than j0 somewhere
            on the pool where j0 would apply (only there is j0's own claim at stake; the target is the alternative
            there, j0's own prediction elsewhere), else ([], None)."""
            ds = sorted(narrow)
            if not ds:
                return [], None
            for x, a, obs, inf in seen:
                p = spoken(ds, lambda d: pred(cands[d], x, a, inf)[1], lambda d: self.apply(cands[d].effect, x), inf[1])
                if not E(p, obs):
                    return [], None
            v = tuple(spoken(ds, lambda d: applies(d)[i], lambda d: vector(d)[i], vector(-1)[i])
                      for i in range(len(pool)))
            v0, ap0 = vector(j0), applies(j0)
            inside = tuple(x if ap else y for x, y, ap in zip(v, v0, ap0))
            if not all(pe(x, y) for x, y in zip(inside, v0)):
                return ds, inside
            return [], None
        kf = _key if eq is None else (lambda x: x)               # with a tolerance the raw predictions are compared

        def vector(j):
            if j not in vec:
                if j < 0:
                    vec[j] = tuple(kf(pinfo(i)[1]) for i in range(len(pool)))
                else:
                    vec[j] = tuple(kf(pred(cands[j], pool[i][0], pool[i][1], pinfo(i))[0]) for i in range(len(pool)))
            return vec[j]

        def classes_of(ids):
            return [[ids[i] for i in g] for g in groups_of([vector(j) for j in ids], veq)]

        def groups():
            return classes_of(alive + ([-1] if null else []))

        asked: list[Transition] = []
        done: set[int] = set()
        ok_safe = (lambda i: True) if safe is None else (lambda i: bool(safe(*pool[i])))
        why = ""
        while True:
            gs = groups()
            if order == "split" and len(gs) <= 1:
                ws, alt = witnesses(gs[0][0]) if gs and gs[0][0] >= 0 and len(gs[0]) == len(alive) else ([], None)
                if not ws:
                    why = "one class left"
                    break
                vec[-2] = alt                                    # tell the class apart from the several-laws one
                gs = [gs[0], [-2]]
                why = "one class left, a several-laws alternative too"
            if len(asked) >= int(budget):
                why = f"the budget of {int(budget)} probes is spent"
                break
            if not alive:
                why = "every candidate refuted"
                break
            if order == "random":
                left = [i for i in range(len(pool)) if i not in done and ok_safe(i)]
                if not left:
                    why = "no safe probe left"
                    break
                i = rng.choice(left)
            else:
                reps = [(lambda st, ac, j=g[0]: vector(j)[st]) for g in gs]
                ro = None if read_only is None else (lambda st, ac: read_only(*pool[st]) if _arity2(read_only)
                                                     else read_only(pool[st][1]))
                cs = None if cost is None else (lambda st, ac: cost(*pool[st]))
                pr = split(reps, lambda st, ac: ok_safe(st), [(i, pool[i][1]) for i in range(len(pool))
                                                             if i not in done], read_only=ro, cost=cs, same=eq)
                if pr is None:
                    why = ("no safe probe tells the classes apart (they differ only where it is unsafe to ask)"
                           if len(gs) > 2 or gs[-1] != [-2] else
                           "no safe probe tells the class apart from the several-laws alternative")
                    break
                i = pr.state
            done.add(i)
            x, a = pool[i]
            obs = ask(x, a)
            asked.append(Transition(x, a, obs, group="probe", id=f"probe-{len(asked) + 1}-{i}"))
            see(x, a, obs, pinfo(i))
        self.cost["probes"] += len(asked)
        order_key = (lambda j: (len(cands[j].condition), j))
        cls = sorted((sorted(g, key=order_key) for g in classes_of(alive)), key=lambda g: order_key(g[0]))
        classes = [[cands[j] for j in g] for g in cls]
        same_vec = veq or (lambda u, v: u == v)
        null_with = null and bool(cls) and any(same_vec(vector(g[0]), vector(-1)) for g in cls)

        def undetermined(reason, ws=()):
            return Admission("undetermined", None, classes, asked, refuted, null, reason, [cands[d] for d in ws])

        if not alive:
            return undetermined("no candidate survived" + (f" ({why})" if why else ""))
        if unique:
            if len(cls) > 1:
                return undetermined(f"{len(cls)} classes of candidates survive: {why}")
            if null_with:
                return undetermined(f"the surviving candidates predict what the set without them does on every probe: "
                                    f"{why}")
            if null:
                return undetermined(f"the set without a new law survives too: {why}")
            j = cls[0][0]
            ws, _ = witnesses(j)
            if ws:
                return undetermined(f"{len(ws)} candidate(s) refuted only as too narrow are right wherever they were "
                                    f"applied and predict otherwise than the surviving class — more than one law may "
                                    f"be missing: {why}", ws)
        else:
            j = min(alive, key=order_key)
        if len(self.laws) >= self.cap:
            self.capped += 1
            return undetermined("the cap of active laws is reached")
        c = cands[j]
        rows = list(asked)
        if buffer is not None:
            rows = [r for a in acts.values() for r in buffer.rows(a, exclude)] + rows
        sup = [r for r in rows if pred(c, r.state, r.action, info(r.state, r.action))[1]]
        hits = sum(1 for r in sup if E(self.apply(c.effect, r.state), r.observed))
        net = sum(1 for r in sup if E(self.apply(c.effect, r.state), r.observed)
                  and not E(self.predict(r.state, r.action, disagree="first"), r.observed))
        law = self._accept(c, born="probe", support=len(sup), hits=hits, net=net,
                           sources=tuple(r.id for r in sup))
        law.admitted, law.tested = "probes", len(cands)
        self.cost["accepted"] += 1
        return Admission("accepted", law, classes, asked, refuted, null, why or "one class left")

    def _accept(self, c, *, born, support=0, hits=0, net=0, sources=()):
        self._seq += 1
        law = Law(id=f"law-{self._seq}", action=c.action, condition=tuple(c.condition), effect=c.effect,
                  support=support, hits=hits, precision=(hits / support) if support else None, net=net, born=born,
                  proposer=c.proposer, sources=sources, seq=self._seq)
        self.laws.append(law)
        self.laws.sort(key=self._order)
        return law

    def _order(self, law):
        """The set's order: most specific first, then by `ties` (seq: the order of acceptance)."""
        return (-len(law.condition), law.seq if self.ties == "earlier" else -law.seq)

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
        self._stamp(retired, len(buffer))
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
        law.hits_after += int(self.agrees(self.apply(law.effect, t.state), t.observed))
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
            self._stamp([law], None)

    # ----------------------------------------------------------------------------------- the loop
    def step(self, t: Transition, buffer: Buffer, *, remember: bool = False) -> Step:
        """One transition of the loop: predict, count it against the applied law (retire), on a mismatch propose and
        test on the buffer without this transition's group; `remember`: then add it to the buffer."""
        law = self.applied(t.state, t.action)
        predicted = self.predict(t.state, t.action, disagree="first")
        retired = self.observe(t)
        self._stamp(retired, len(buffer))
        miss = predicted != t.observed if self.same is None else not self.agrees(predicted, t.observed)
        st = Step(t, predicted, law, miss, retired=retired)
        if miss:
            self.transition = t
            try:
                cands = self.mismatch(t.state, t.action, t.observed)
            finally:
                self.transition = None
            st.candidates = len(cands)
            self._last_tested = 0
            st.accepted = self.test(cands, buffer, exclude=t.group) if cands else []
            st.tested = self._last_tested
            if cands:
                st.retired = st.retired + self.last_retired
        if remember:
            buffer.add(t)
        return st

    def learn(self, stream: Iterable[Transition], buffer: Buffer, *, remember: bool = False) -> list[Step]:
        """The loop over a stream, in order (not a schedule: every transition is predicted before it is learned from)."""
        return [self.step(t, buffer, remember=remember) for t in stream]

    def need_at(self, tested: int, null: float, precision: float = 1.0, cap: int = 1_000_000) -> int | None:
        """The least support with which a law right on `precision` of its rows passes the acceptance at N = `tested`
        candidates against `null` (min_support, min_precision and — sized — the binomial bound); None when no support
        does: a precision at or below the null, or below min_precision. At precision 1 and sized: sized_need."""
        p = float(precision)
        if p < self.min_precision or (self.sized and p <= float(null)):
            return None
        if p >= 1.0 and self.sized:
            return max(self.min_support, sized_need(tested, self.delta, null, self.space))

        def ok(sup):
            return self.passes(sup, math.ceil(p * sup - 1e-9), tested, null)
        hi = max(1, self.min_support)
        while not ok(hi):                          # doubling, then the least passing support below the first that passes
            if hi >= cap:
                return None
            hi *= 2
        lo = hi // 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        return hi

    def report(self, buffer: Buffer, steps: Iterable[Step] | None = None, *, precision: float | None = None) -> dict:
        """What a learning run did, and per action whether the buffer could admit a law at all — so «0 laws accepted»
        reads as «too few rows» where that is the reason. → {"accepted", "active", "retired", "tested" (N, the
        candidates scored on the buffer), "cost", the run's "steps" / "mismatches" (with `steps`), "actions": {action:
        {"rows", "outcomes", "null", "need", "need_at_precision", "precision", "enough"}}, "too_few_rows" (the actions
        whose rows are fewer than "need"), "summary" (in words)}.

        Per action: "rows" in the buffer; "null" the least base null among its outcomes (the rarest outcome's share,
        floored at 0.05 — the set's fixed null when it has one); "need" the least support a law right on every row
        needs to pass at this N against that null (sized_need; min_support with sized=False) — a law's support is at
        most the action's rows, so rows < need means no law of that action can be accepted on this buffer, whatever
        its condition; "need_at_precision" the same for a law right on `precision` of its rows (default
        min_precision; None when no support suffices). More outcomes are harder: their nulls are higher."""
        from collections import Counter
        steps = list(steps) if steps is not None else None
        N = max(1, int(buffer.tested))
        prec = self.min_precision if precision is None else float(precision)
        cap = self.min_precision if self.min_precision < 1.0 else 0.95
        actions, short = {}, []
        for a in list(buffer._rows):
            rows = buffer._rows[a]
            n = len(rows)
            counts = Counter(_key(t.observed) for t in rows)
            if isinstance(self.null, str):
                null = min(max(min(counts.values()) / n, self.NULL_FLOOR), cap) if n else min(self.NULL_FLOOR, cap)
            else:
                null = float(self.null)
            need = (max(self.min_support, sized_need(N, self.delta, null, self.space)) if self.sized
                    else self.min_support)
            info = {"rows": n, "outcomes": len(counts), "null": round(null, 4), "need": need,
                    "need_at_precision": self.need_at(N, null, prec), "precision": prec, "enough": n >= need}
            actions[rows[0].action if rows else a] = info
            if n < need:
                short.append(rows[0].action if rows else a)
        out = {"accepted": self.cost["accepted"], "active": len(self.laws), "retired": len(self.retired), "tested": N,
               "cost": dict(self.cost), "actions": actions, "too_few_rows": short}
        if steps is not None:
            out["steps"] = len(steps)
            out["mismatches"] = sum(1 for st in steps if st.mismatch)
        few = sorted(((actions[x]["rows"], actions[x]["need"], x) for x in short), key=lambda r: (r[0] - r[1], repr(r[2])))
        text = f"{self.cost['accepted']} law(s) accepted ({len(self.laws)} active) after {N} candidate(s) tested"
        if short:
            eg = "; ".join(f"{x!r}: {r} rows, a law needs ≥ {nd}" for r, nd, x in few[:3])
            text += (f"; {len(short)} of {len(actions)} action(s) have fewer rows than any law of theirs needs at "
                     f"this N (e.g. {eg}) — too few rows, not a missing law")
        out["summary"] = text
        return out

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


def _arity2(fn):
    return _required_args(fn) >= 2


def _plain(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)


__all__ = ["Admission", "Atom", "Buffer", "CallableProposer", "Candidate", "Enumerate", "Grammar", "Law", "LawSet",
           "Proposer", "RandomProposer", "Step", "Transition", "Undetermined", "sized_level", "sized_need"]
