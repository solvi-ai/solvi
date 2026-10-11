"""Datalog-lite rules over pair facts with unary guards, learned by exact counting on labelled query pairs, applied by
forward chaining to a fixed point, with a negative stratum.

    from solvi.experimental.relations import GuardedRules, Story, chain_rules
    st = Story.from_facts([(1, 2, "parent_of"), (2, 3, "parent_of")], [(1, "is_male"), (9, "is_place")])
    rules = GuardedRules(majority=0.9, support=2).fit(examples)       # examples: (story, (x, y), set of relations)
    rules.predict(st, (1, 3))                                          # frozenset({"grandparent_of", ...}) or ∅
    c = rules.closure(st); c.answer(1, 3); c.status(1, 3, "grandparent_of")
    rules.to_knowledge(store, {"domain": "kin"}); chain_rules(store, scope={"story": 1}, rule_scope={"domain": "kin"})

Facts. A story is a set of binary facts r(x, y) — a pair may carry several relations — and unary facts u(x) (a self-
loop edge is_* / no_* of a story graph). An atom is a relation read forwards, s(x, y), or backwards, s⁻(x, y) = s(y, x).

Rule bodies (head r(x, y); x ≠ y always, z a third entity):
    B1   r ← s(x,y)                      B1u  r ← s(x,y) ∧ u(x)  or  ∧ u(y)
    B2   r ← s1(x,z) ∧ s2(z,y)           B2z  r ← s1(x,z) ∧ u(z) ∧ s2(z,y)
    BNE  r ← s(x,z) ∧ z ≠ y ∧ u(y)       (composition with the complement of identity: «lives somewhere else»)
    NEG  r ← u1(x) ∧ u2(y) ∧ ¬s(x,y)     (the negative stratum)
The positive rules are applied to a fixed point (a derived fact feeds further bodies; depth is reached by rounds, a body
has at most two binary atoms). The negative stratum comes after: a NEG rule fires on (x, y) only when the positive
fixed point holds no s(x, y) and no fact of the head's own key on (x, y) (the head or its declared complement) — a
fallback where nothing positive derives, never fed back into the positive rules.

Keys and disputes. Each relation is its own key on a pair, except relations declared complements of each other
(`complements={"Not_living_in": "living_in"}`): they share a key, so two of them on one pair are a dispute — the pair
is answered on neither (the store's «two values of one key from equal-rank sources → a question for a person»). Two
given values of one key: the later one in the story stands (an observation refutes the earlier one, as in the store).
Status of a fact at the fixed point: "disputed" (its key holds two values), "active" (given, or derived through a rule
from active premises), "hypothesis" (derived only through disputed premises). `answer(x, y)` = the active relations;
the empty set is a refusal.

The closed world by declaration (`functional={"living_in": "person"}`: relation → the subject's entity type, or
(subject type, object type)). A declaration, never inferred: a subject of that type has at most one object for the
relation, and the relation has a declared complement. A third stratum after NEG: for each subject x of the type with
exactly one known positive r(x, z) — given or derived by a positive rule, active before this stratum — the complement
r̄(x, y) on every other candidate object y (the entities of the object type; without one, those carrying every is_*
unary of z), stratum 3 ("closed"), resting on r(x, z) (`Closure.closed`). No known positive: open world, nothing.
Two or more: the declaration is contradicted there, nothing (`closed_conflicts`). A closed fact on a pair that holds
a non-known r(x, y) is a dispute (`closed_disputes`). Closed facts feed no rule; statuses are recomputed after the
stratum. Learning counts on fixed points without it unless `closed_world_in_rounds=True`.

Learning (`fit`). Each labelled query pair (x, y) with its set of relations Y: every body true on (x, y) at the current
fixed point gets one vote, and each relation of Y one hit for that body. A rule body → r is admitted when its hits for r
are ≥ `support` and a share ≥ `majority` of its votes; never r ← r (∧ u); a guarded body (B1u, B2z) is not admitted
when its unguarded generalization (B1, B2) is admitted for the same head. NEG bodies are counted on the pairs whose
positive fixed point holds nothing of the head's key, and only for a relation s that occurs on at least `support`
labelled pairs of the same unary types (so that ¬s excludes something). `check=`: labelled pairs of an independent
slice; a rule whose share there is below `majority` (on at least one pair) is refused, and one with fewer than
`check_min` hits there too.

Rounds. Round k counts the bodies on the fixed point of the rules admitted so far (round 1: the given facts; votes
are recounted every round, never accumulated) and admits the rules that pass; the rule set only grows (a guarded body
leaves it when its unguarded generalization joins: its consequences are a subset), so the rounds stop at a fixed point
of the rule set or after `rounds`. Each rule is admitted on the fixed point of the rules before it; rules admitted
together can still be wrong together (a rule right at the labelled pairs and wrong elsewhere derives facts that make
others fire wrongly), so after each round (`verify=True`) the round's new rules are recounted on the fixed point of
the whole new set — on the check slice when one is given — and a new rule whose share there is below `majority` is
retracted and never admitted again — the worse half of the failing rules per recount, lowest share first, since a
rule can fail only through the wrong facts of another; the recount repeats until no new rule falls (the rules of
earlier rounds passed the same recount with the rules before them and are kept). The NEG rules are learned once, on the final positive fixed
point.

Residual votes (`residual=True`), the composition table's residual fill generalised: within a round, the rules
admitted on that round's direct votes are applied to the length-3 walks x –a1– z1 –a2– z2 –a3– y of each labelled pair;
where a B2/B2z rule gives s(z1, y) from a2 ∘ a3 (or s(x, z2) from a1 ∘ a2) and that fact is not yet in the fixed point,
the body a1 ∘ s (or s ∘ a3), with its z-guards, gets the pair's vote — the evidence a chain body needs while its
intermediate fact is unlabelled and not yet derivable. A pair votes once per body (direct or residual). A rule whose
hits are all residual has origin "residual".

`chain_rules(store, ...)` applies the rules kept in a KnowledgeStore (`to_knowledge`) to the pair and unary facts of a
scope: the same fixed point, every derived fact written through `store.add` with each of its justifications (the
premise facts — unary guards included — and the rule as `derived_from`, `inferred=True`, the weakest premise's source);
the NEG stratum written after, its absent relation in the evidence; a NEG fact whose condition no longer holds on a
later call is retracted. The store's key semantics give the disputes; `explain` reads the derivation back."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

TEMPLATES = ("B1", "B1u", "B2", "B2z", "BNE", "NEG")
_SHAPE = {"B1": (1, ()), "B1u": (1, (("x", "y"),)), "B2": (2, ()), "B2z": (2, (("z",),)), "BNE": (1, (("y",),)),
          "NEG": (1, (("x",), ("y",)))}
UNARY = "unary"
STRATA = ("given", "positive", "negative", "closed")     # Closure.pairs stratum 0..3


def unary_fact(x, u) -> dict:
    """The body of the unary fact u(x): {"s": x, "r": "unary/<u>", "o": True} (one key per entity and predicate)."""
    return {"s": x, "r": f"{UNARY}/{u}", "o": True}


def _atom_text(a, p, q):
    return f"{a[0]}{'⁻' if a[1] else ''}({p},{q})"


@dataclass(frozen=True)
class GuardedRule:
    """One rule: head ← body. template: one of TEMPLATES; atoms: ((relation, inverse 0|1), ...) — one atom (B1, B1u,
    BNE; NEG: the negated relation) or two (B2, B2z); unary: ((predicate, "x" | "y" | "z"), ...) — B1u one on x or
    y, B2z one on z, BNE one on y, NEG two (x, y). hits / support: the counts it was admitted with (support = the
    pairs where the body held); origin "direct" | "residual" | "given"; round: the round it was admitted in.
    Equality and hashing use the rule only, not its counts."""
    head: str
    template: str
    atoms: tuple
    unary: tuple = ()
    hits: int = field(default=0, compare=False)
    support: int = field(default=0, compare=False)
    origin: str = field(default="given", compare=False)
    round: int = field(default=0, compare=False)

    def __post_init__(self):
        if self.template not in _SHAPE:
            raise ValueError(f"template is one of {TEMPLATES}, not {self.template!r}")
        n, sides = _SHAPE[self.template]
        atoms = tuple((str(a[0]), int(a[1])) for a in self.atoms)
        unary = tuple((str(u[0]), str(u[1])) for u in self.unary)
        if len(atoms) != n or any(i not in (0, 1) for _, i in atoms):
            raise ValueError(f"{self.template} has {n} atom(s) (relation, 0 | 1)")
        if len(unary) != len(sides) or any(u[1] not in s for u, s in zip(unary, sides)):
            raise ValueError(f"{self.template}: unary guards on {sides}")
        if self.template == "NEG" and atoms[0][1]:
            raise ValueError("NEG negates a relation read forwards")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "unary", unary)

    @property
    def body(self) -> tuple:
        return (self.template, self.atoms, self.unary)

    @property
    def share(self) -> float:
        return self.hits / self.support if self.support else 0.0

    def text(self) -> str:
        t, a, u = self.template, self.atoms, self.unary
        if t == "B1":
            return _atom_text(a[0], "x", "y")
        if t == "B1u":
            return f"{_atom_text(a[0], 'x', 'y')} ∧ {u[0][0]}({u[0][1]})"
        if t == "B2":
            return f"{_atom_text(a[0], 'x', 'z')} ∧ {_atom_text(a[1], 'z', 'y')}"
        if t == "B2z":
            return f"{_atom_text(a[0], 'x', 'z')} ∧ {u[0][0]}(z) ∧ {_atom_text(a[1], 'z', 'y')}"
        if t == "BNE":
            return f"{_atom_text(a[0], 'x', 'z')} ∧ z≠y ∧ {u[0][0]}(y)"
        return f"{u[0][0]}(x) ∧ {u[1][0]}(y) ∧ ¬{a[0][0]}(x,y)"

    def __str__(self):
        return f"{self.head}(x,y) ← {self.text()}"


class Story:
    """The facts of one story: binary (x, y, relation) in order, unary (x, predicate). `from_edges(edges, types)` reads
    a story graph: a self-loop whose type starts with one of `unary_prefixes` is a unary fact; other self-loops are
    dropped (counted in `dropped`)."""

    __slots__ = ("facts", "unary", "dropped")

    def __init__(self, facts: Iterable = (), unary: Iterable = (), dropped: int = 0):
        self.facts = tuple((x, y, str(r)) for x, y, r in facts)
        self.unary = tuple((x, str(u)) for x, u in unary)
        self.dropped = dropped

    @classmethod
    def from_facts(cls, facts: Iterable = (), unary: Iterable = ()) -> "Story":
        return cls(facts, unary)

    @classmethod
    def from_edges(cls, edges: Iterable, types: Iterable, unary_prefixes=("is_", "no_")) -> "Story":
        fs, us, dropped = [], [], 0
        for (a, b), t in zip(edges, types):
            if a == b:
                if str(t).startswith(tuple(unary_prefixes)):
                    us.append((a, t))
                else:
                    dropped += 1
            else:
                fs.append((a, b, t))
        return cls(fs, us, dropped)

    def entities(self) -> set:
        return {e for x, y, _ in self.facts for e in (x, y)} | {x for x, _ in self.unary}


def _fact_of(p, q, a):
    return (p, q, a[0]) if a[1] == 0 else (q, p, a[0])


class _Index:
    """The positive rules indexed by atom for the join, the NEG rules apart."""

    def __init__(self, rules):
        self.b1, self.b1u, self.bne = defaultdict(list), defaultdict(list), defaultdict(list)
        self.b2f, self.b2s = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        self.neg = []
        for i, r in enumerate(rules):
            t = r.template
            if t == "B1":
                self.b1[r.atoms[0]].append((r.head, i))
            elif t == "B1u":
                self.b1u[r.atoms[0]].append((r.unary[0][1], r.unary[0][0], r.head, i))
            elif t in ("B2", "B2z"):
                u = r.unary[0][0] if t == "B2z" else None
                self.b2f[r.atoms[0]][r.atoms[1]].append((r.head, u, i))
                self.b2s[r.atoms[1]][r.atoms[0]].append((r.head, u, i))
            elif t == "BNE":
                self.bne[r.atoms[0]].append((r.unary[0][0], r.head, i))
            else:
                self.neg.append(i)


class Closure:
    """The fixed point of a rule set on one story. pairs: (x, y) → {relation: (stratum, depth)} (stratum 0 given, 1
    positive rule, 2 negative stratum, 3 closed world — `STRATA` names them; depth: how deep in rule applications it appeared — the worklist engine counts
    the first derivation found, the bitset engine the sweep it appeared in: a report figure); unary: x → set; just
    (with justifications): fact → [(premises, rule index)] — premises are binary facts (x, y, r) and unary facts
    (x, u); absent: NEG fact → [the negated relation per justification]; statuses: fact → status (see the module);
    closed: closed-world fact → [the known positives r(x, z) it rests on] (also for a fact present through another
    stratum); closed_disputes: the keys (x, y, family) disputed only once the closed-world stratum ran;
    closed_conflicts: (x, relation, objects) for subjects with several known positives (left open)."""

    def __init__(self, rules, complements):
        self.rules = rules
        self.complements = dict(complements or {})
        self.pairs: dict = {}
        self.unary = defaultdict(set)
        self.out = defaultdict(dict)                     # x → {z: set of atoms from x to z}
        self.just: dict = {}
        self.absent: dict = {}
        self.statuses: dict = {}
        self.order: list = []                            # derived facts in the order of their first derivation
        self.refuted: list = []
        self.closed: dict = {}                           # closed-world fact → [the known positives r(x, z) it rests on]
        self.closed_disputes: list = []                  # keys (x, y, family) the closed-world stratum made disputed
        self.closed_conflicts: list = []                 # (x, relation, objects): several known positives, no closure

    def family(self, rel):
        return self.complements.get(rel, rel)

    def relations(self, x, y) -> set:
        """Every relation present on (x, y) at the fixed point (any status)."""
        return set(self.pairs.get((x, y), ()))

    def stratum(self, x, y, rel):
        v = self.pairs.get((x, y), {}).get(rel)
        return None if v is None else v[0]

    def depth(self, x, y, rel):
        v = self.pairs.get((x, y), {}).get(rel)
        return None if v is None else v[1]

    def status(self, x, y, rel):
        return self.statuses.get((x, y, rel))

    def info(self, x, y, rel) -> dict | None:
        """{"status", "stratum": "given" | "positive" | "negative" | "closed", "depth"} of one fact, None when absent."""
        v = self.pairs.get((x, y), {}).get(rel)
        if v is None:
            return None
        return {"status": self.statuses.get((x, y, rel)), "stratum": STRATA[v[0]], "depth": v[1]}

    def answer(self, x, y) -> frozenset:
        """The active relations on (x, y): the answer; the empty set is a refusal."""
        return frozenset(r for r in self.pairs.get((x, y), ()) if self.statuses.get((x, y, r)) == "active")

    @property
    def rounds(self) -> int:
        return max((d for v in self.pairs.values() for _, d in v.values()), default=0)

    def facts(self):
        """Every binary fact (x, y, relation) at the fixed point."""
        return [(x, y, r) for (x, y), v in self.pairs.items() for r in v]


def _closure(story: Story, rules, complements=None, *, justify=False, neg=True, statuses=None,
             observe=True, functional=None) -> Closure:
    """The positive fixed point of `rules` on `story` (a worklist join: every pair of facts is joined when the later
    of them is taken), then — with neg — the negative stratum, then — with functional (relation → (subject unary,
    object unary | None, complement)) — the closed-world stratum, and — with statuses (default: justify; always with
    functional) — the status of every fact. observe: two given values of one key → the later one stands (False: both
    stand, a dispute)."""
    rules = list(rules)
    c = Closure(rules, complements)
    idx = _Index(rules)
    U, out, P = c.unary, c.out, c.pairs
    for x, u in story.unary:
        U[x].add(u)
    nodes_u = defaultdict(list)
    ents = story.entities()
    for x in sorted(ents, key=repr):
        for u in U.get(x, ()):
            nodes_u[u].append(x)
    fam = c.family
    last = {}
    for i, (x, y, r) in enumerate(story.facts if observe else ()):   # an observation refutes an earlier value
        if x == y:
            continue
        k = (x, y, fam(r))
        prev = last.get(k)
        if prev is not None and prev != r:
            c.refuted.append((x, y, prev))
        last[k] = r
    base = sorted({(x, y, r) for (x, y, f), r in last.items()}, key=repr) if c.refuted else None
    queue = []
    J = c.just

    def put(x, y, r, stratum, depth):
        P.setdefault((x, y), {})[r] = (stratum, depth)
        out[x].setdefault(y, set()).add((r, 0))
        out[y].setdefault(x, set()).add((r, 1))

    seen_base = set()
    for x, y, r in (base if base is not None else story.facts):
        if x == y or (x, y, r) in seen_base:
            continue
        seen_base.add((x, y, r))
        put(x, y, r, 0, 0)
        queue.append((x, y, r))

    def add(x, y, r, prem, ri):
        if x == y:
            return
        cur = P.get((x, y))
        if cur is not None and r in cur:
            if justify and cur[r][0] == 1:
                J[(x, y, r)].add((prem, ri))
            return
        d = 1 + max((P[(p[0], p[1])][p[2]][1] for p in prem if len(p) == 3), default=0)
        put(x, y, r, 1, d)
        f = (x, y, r)
        c.order.append(f)
        if justify:
            J[f] = {(prem, ri)}
        queue.append(f)

    b1, b1u, b2f, b2s, bne = idx.b1, idx.b1u, idx.b2f, idx.b2s, idx.bne
    qi = 0
    while qi < len(queue):
        f = queue[qi]
        qi += 1
        x0, y0, r0 = f
        for p, q, a in ((x0, y0, (r0, 0)), (y0, x0, (r0, 1))):
            for head, ri in b1.get(a, ()):
                add(p, q, head, (f,), ri)
            for side, u, head, ri in b1u.get(a, ()):
                node = p if side == "x" else q
                if u in U.get(node, ()):
                    add(p, q, head, (f, (node, u)), ri)
            m = b2f.get(a)
            if m:                                        # a = s1(p, q): join with s2(q, y)
                for y2, A2 in list(out[q].items()):
                    if y2 == p or y2 == q:
                        continue
                    for a2 in tuple(A2):
                        for head, u, ri in m.get(a2, ()):
                            if u is None:
                                add(p, y2, head, (f, _fact_of(q, y2, a2)), ri)
                            elif u in U.get(q, ()):
                                add(p, y2, head, (f, _fact_of(q, y2, a2), (q, u)), ri)
            m = b2s.get(a)
            if m:                                        # a = s2(p, q): join with s1(x, p)
                for x2, B in list(out[p].items()):
                    if x2 == p or x2 == q:
                        continue
                    for b in tuple(B):
                        a1 = (b[0], 1 - b[1])            # the atom from x2 to p
                        for head, u, ri in m.get(a1, ()):
                            if u is None:
                                add(x2, q, head, (_fact_of(x2, p, a1), f), ri)
                            elif u in U.get(p, ()):
                                add(x2, q, head, (_fact_of(x2, p, a1), f, (p, u)), ri)
            for u, head, ri in bne.get(a, ()):          # a = s(p, z=q): every y ≠ z with u(y)
                for y2 in nodes_u.get(u, ()):
                    if y2 != q and y2 != p:
                        add(p, y2, head, (f, (y2, u)), ri)
    if justify:
        for k in J:
            J[k] = sorted(J[k], key=repr)
    if neg and idx.neg:
        negf = {}
        for ri in idx.neg:
            r = rules[ri]
            s, (u1, _), (u2, _) = r.atoms[0][0], r.unary[0], r.unary[1]
            hf = fam(r.head)
            for x in nodes_u.get(u1, ()):
                for y in nodes_u.get(u2, ()):
                    if x == y:
                        continue
                    cur = P.get((x, y), ())
                    if s in cur or any(fam(v) == hf for v in cur):
                        continue
                    negf.setdefault((x, y, r.head), []).append((((x, u1), (y, u2)), ri, s))
        for (x, y, h), js in sorted(negf.items(), key=repr):
            put(x, y, h, 2, 0)
            c.order.append((x, y, h))
            if justify:
                J[(x, y, h)] = [(p, ri) for p, ri, _ in js]
                c.absent[(x, y, h)] = [s for _, _, s in js]
    if functional:
        _statuses(c)                                     # which positives are known: the active ones
        before = _disputed_keys(c)
        _closed_world(c, functional, ents)
        _statuses(c)
        c.closed_disputes = sorted(_disputed_keys(c) - before, key=repr)
    elif statuses if statuses is not None else justify:
        _statuses(c)
    return c


def _disputed_keys(c: Closure) -> set:
    keys = defaultdict(set)
    for (x, y), v in c.pairs.items():
        for r in v:
            keys[(x, y, c.family(r))].add(r)
    return {k for k, vs in keys.items() if len(vs) > 1}


def _closed_world(c: Closure, spec, ents):
    """The closed-world stratum: for a declared functional relation r (spec: r → (subject unary, object unary | None,
    complement r̄)), every subject x of the type with a known positive r(x, z) — given or derived by a positive rule,
    active at the statuses before this stratum — gets r̄(x, y) on every candidate object y ≠ z, x (the entities with
    the object unary; None: those carrying every is_* unary of z). Recorded in `closed` (the positives it rests on);
    a new fact enters the pairs at stratum 3; nothing here feeds a rule. A subject with two or more known positives
    contradicts the declaration: nothing is derived for it, the conflict is recorded in `closed_conflicts`."""
    P, U, st = c.pairs, c.unary, c.statuses
    by_u = defaultdict(list)
    for e in sorted(ents, key=repr):
        for u in U.get(e, ()):
            by_u[u].append(e)
    known = defaultdict(list)
    for (x, z), v in P.items():
        for r, (stratum, _) in v.items():
            if r in spec and stratum in (0, 1) and st.get((x, z, r)) == "active":
                known[r].append((x, z))
    for r in sorted(spec):
        su, ou, rbar = spec[r]
        by_x = defaultdict(list)
        for x, z in sorted(known[r], key=repr):
            by_x[x].append(z)
        for x, z in sorted(known[r], key=repr):
            if su not in U.get(x, ()):
                continue
            if len(by_x[x]) > 1:                         # the declaration is contradicted here: no closed world
                if not c.closed_conflicts or c.closed_conflicts[-1][:2] != (x, r):
                    c.closed_conflicts.append((x, r, tuple(by_x[x])))
                continue
            if ou is not None:
                cands = by_u.get(ou, ())
            else:
                kinds = sorted(u for u in U.get(z, ()) if u.startswith("is_"))
                if not kinds:
                    continue
                cs = set(by_u.get(kinds[0], ()))
                for u in kinds[1:]:
                    cs &= set(by_u.get(u, ()))
                cands = sorted(cs, key=repr)
            for y in cands:
                if y == x or y == z:
                    continue
                f = (x, y, rbar)
                c.closed.setdefault(f, []).append((x, z, r))
                cur = P.setdefault((x, y), {})
                if rbar not in cur:
                    cur[rbar] = (3, 0)
                    c.order.append(f)


def _closed_status(c: Closure, st: dict):
    """A fact with a closed-world justification and no active one otherwise: active when a positive it rests on is
    active (and its key is not disputed). After the positive statuses: it never makes a positive fact active."""
    for f, prem in c.closed.items():
        if st.get(f) == "hypothesis" and any(st.get(p) == "active" for p in prem):
            st[f] = "active"


def _statuses(c: Closure):
    """Disputed: a key holding two values. Active: a given fact not disputed, or a derived fact not disputed with a
    justification whose binary premises are all active (least fixed point). Hypothesis: the rest."""
    keys = defaultdict(set)
    for (x, y), v in c.pairs.items():
        for r in v:
            keys[(x, y, c.family(r))].add(r)
    disputed = {(x, y, r) for (x, y, _), vs in keys.items() if len(vs) > 1 for r in vs}
    st = {}
    for (x, y), v in c.pairs.items():
        for r, (s, _) in v.items():
            f = (x, y, r)
            st[f] = "disputed" if f in disputed else ("active" if s in (0, 2) else "hypothesis")
    changed = True
    while changed:
        changed = False
        for f in c.order:
            if st.get(f) != "hypothesis":
                continue
            for prem, _ in c.just.get(f, ()):
                if all(st.get(p) == "active" for p in prem if len(p) == 3):
                    st[f] = "active"
                    changed = True
                    break
    _closed_status(c, st)
    c.statuses = st


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def _fast(story: Story, rules, complements=None, *, neg=True, statuses=True, observe=True, functional=None) -> Closure:
    """The same fixed point as _closure without justifications: per relation a row mask and a column mask per entity
    (bit y of row x: r(x, y)); a rule is one OR over masks per entity; Jacobi iterations re-evaluate only the rules
    whose body relations changed. Statuses: disputed = a key holding two values; active = the fixed point re-run from
    the given facts with the disputed facts blocked (the least fixed point over active premises); hypothesis = the
    rest. Depth = the sweep a fact first appears in (sweeps are Gauss-Seidel: a rule sees the updates of the rules
    before it). functional: the closed-world stratum after NEG, as in _closure (statuses are computed with it)."""
    c = Closure(list(rules), complements)
    fam = c.family
    ents = sorted(story.entities(), key=repr)
    ix = {e: i for i, e in enumerate(ents)}
    n = len(ents)
    for x, u in story.unary:
        c.unary[x].add(u)
    um = defaultdict(int)
    for x, u in story.unary:
        um[u] |= 1 << ix[x]
    last, base = {}, []
    for x, y, r in story.facts:
        if x == y:
            continue
        if observe:
            k = (x, y, fam(r))
            prev = last.get(k)
            if prev is not None and prev != r:
                c.refuted.append((x, y, prev))
            last[k] = r
        else:
            base.append((x, y, r))
    if observe:
        base = [(x, y, r) for (x, y, _), r in last.items()]
    rel = {}
    for _, _, r in base:
        rel.setdefault(r, len(rel))
    pos, negs = [], []
    for r in rules:
        for a in r.atoms:
            rel.setdefault(a[0], len(rel))
        rel.setdefault(r.head, len(rel))
        (negs if r.template == "NEG" else pos).append(r)
    names = list(rel)
    R = len(names)

    def run(block):
        F = [[0] * n for _ in range(R)]
        B = [[0] * n for _ in range(R)]
        D = {}                                           # (rel id, x) → {y: depth} for derived bits
        for x, y, r in base:
            i, j, k = ix[x], ix[y], rel[r]
            if block is not None and (block[k][i] >> j) & 1:
                continue
            F[k][i] |= 1 << j
            B[k][j] |= 1 << i
        groups = {}                                      # one body, every head it gives: evaluated once
        for r in pos:
            a = tuple((rel[t], v) for t, v in r.atoms)
            u = um.get(r.unary[0][0], 0) if r.unary else 0
            side = r.unary[0][1] if r.unary else None
            groups.setdefault((r.template, a, u, side), []).append(rel[r.head])
        comp = [(t, a, u, side, heads) for (t, a, u, side), heads in groups.items()]
        full = (1 << n) - 1
        drow, dcol = [full] * R, [full] * R              # rows / columns changed in the last sweep (all, at first)
        it, more = 0, True
        while more:                                      # Gauss-Seidel sweeps, semi-naive by row
            it += 1
            nrow, ncol = [0] * R, [0] * R
            more = False
            for t, a, u, side, heads in comp:
                k1, i1 = a[0]
                A1 = F[k1] if i1 == 0 else B[k1]
                d1 = drow[k1] if i1 == 0 else dcol[k1]
                if t == "B2" or t == "B2z":
                    k2, i2 = a[1]
                    A2 = F[k2] if i2 == 0 else B[k2]
                    d2 = drow[k2] if i2 == 0 else dcol[k2]
                    xs = d1
                    if d2:
                        for x in range(n):
                            if A1[x] & d2:
                                xs |= 1 << x
                else:
                    xs = d1
                while xs:
                    lowx = xs & -xs
                    x = lowx.bit_length() - 1
                    xs ^= lowx
                    zs = A1[x]
                    if not zs:
                        continue
                    bx = lowx
                    if t == "B1" or t == "B1u":
                        m = zs
                        if t == "B1u":
                            if side == "x":
                                if not (u >> x) & 1:
                                    continue
                            else:
                                m &= u
                    elif t == "BNE":
                        zs &= ~bx
                        if not zs:
                            continue
                        m = u if zs & (zs - 1) else u & ~zs  # one z: every y but z
                    else:                                # B2 / B2z
                        zs &= ~bx
                        if t == "B2z":
                            zs &= u
                        m = 0
                        while zs:
                            low = zs & -zs
                            m |= A2[low.bit_length() - 1] & ~low
                            zs ^= low
                    m &= ~bx
                    if not m:
                        continue
                    for h in heads:
                        mh = m & ~F[h][x]
                        if block is not None:
                            mh &= ~block[h][x]
                        if not mh:
                            continue
                        F[h][x] |= mh
                        nrow[h] |= bx
                        ncol[h] |= mh
                        more = True
                        dd = D.setdefault((h, x), {})
                        Bh = B[h]
                        while mh:
                            low = mh & -mh
                            y = low.bit_length() - 1
                            Bh[y] |= bx
                            dd[y] = it
                            mh ^= low
            drow, dcol = nrow, ncol
        return F, D

    F, D = run(None)
    P = c.pairs
    for k in range(R):
        r = names[k]
        for i in range(n):
            m = F[k][i]
            if not m:
                continue
            dd = D.get((k, i), {})
            x = ents[i]
            for j in _bits(m):
                d = dd.get(j)
                P.setdefault((x, ents[j]), {})[r] = (0, 0) if d is None else (1, d)
    if neg and negs:
        famof = {}
        for k, r in enumerate(names):
            famof.setdefault(fam(r), []).append(k)
        negf = {}
        for r in negs:
            s_, h = rel[r.atoms[0][0]], rel[r.head]
            m1, m2 = um.get(r.unary[0][0], 0), um.get(r.unary[1][0], 0)
            fk = famof.get(fam(r.head), [h])
            for i in _bits(m1):
                m = m2 & ~(1 << i) & ~F[s_][i]
                for k in fk:
                    m &= ~F[k][i]
                if m:
                    negf[(h, i)] = negf.get((h, i), 0) | m
        for (h, i), m in negf.items():
            for j in _bits(m):
                P.setdefault((ents[i], ents[j]), {})[names[h]] = (2, 0)
    def fstat():
        disputed = {(x, y, r) for (x, y, f) in _disputed_keys(c) for r in P[(x, y)] if fam(r) == f}
        active = None
        if any(P[(x, y)][r][0] in (0, 1) for (x, y, r) in disputed):
            block = [[0] * n for _ in range(R)]
            for x, y, r in disputed:
                if P[(x, y)][r][0] in (0, 1):
                    block[rel[r]][ix[x]] |= 1 << ix[y]
            Fa, _ = run(block)
            active = Fa
        st = {}
        for (x, y), v in P.items():
            for r, (stratum, _) in v.items():
                f = (x, y, r)
                if f in disputed:
                    st[f] = "disputed"
                elif stratum == 2 or (stratum in (0, 1) and (active is None or (active[rel[r]][ix[x]] >> ix[y]) & 1)):
                    st[f] = "active"
                else:
                    st[f] = "hypothesis"
        _closed_status(c, st)
        return st

    if functional:
        c.statuses = fstat()
        before = _disputed_keys(c)
        _closed_world(c, functional, ents)
        c.statuses = fstat()
        c.closed_disputes = sorted(_disputed_keys(c) - before, key=repr)
    elif statuses:
        c.statuses = fstat()
    return c


def _with_out(c: Closure) -> Closure:
    """Fill the atom map (x → {z: atoms}) the body counts read."""
    out = c.out
    for (x, y), v in c.pairs.items():
        for r in v:
            out[x].setdefault(y, set()).add((r, 0))
            out[y].setdefault(x, set()).add((r, 1))
    return c


# ───────────────────────────────────────────────────────────────── learning: bodies true on a pair
def _bodies(c: Closure, x, y, templates) -> set:
    out, U = c.out, c.unary
    B = set()
    ox, oy = out.get(x, {}), out.get(y, {})
    Ux, Uy = U.get(x, ()), U.get(y, ())
    for a in ox.get(y, ()):
        if "B1" in templates:
            B.add(("B1", (a,), ()))
        if "B1u" in templates:
            for u in Ux:
                B.add(("B1u", (a,), ((u, "x"),)))
            for u in Uy:
                B.add(("B1u", (a,), ((u, "y"),)))
    b2, b2z, bne = "B2" in templates, "B2z" in templates, "BNE" in templates
    for z, A1 in ox.items():
        if z == y or z == x:
            continue
        if bne:
            for a in A1:
                for u in Uy:
                    B.add(("BNE", (a,), ((u, "y"),)))
        if not (b2 or b2z):
            continue
        A2r = oy.get(z)
        if not A2r:
            continue
        Uz = U.get(z, ())
        for a1 in A1:
            for r, i in A2r:
                a2 = (r, 1 - i)
                if b2:
                    B.add(("B2", (a1, a2), ()))
                if b2z:
                    for u in Uz:
                        B.add(("B2z", (a1, a2), ((u, "z"),)))
    return B


def _residual(c: Closure, x, y, inner, templates) -> set:
    """Bodies made true on (x, y) by a rule of `inner` ((a, b) → [(s, u | None)], B2/B2z) applied to the middle or the
    left two atoms of a length-3 walk, where the fact it would give is not yet in the fixed point."""
    out, U, P = c.out, c.unary, c.pairs
    R = set()
    b2, b2z = "B2" in templates, "B2z" in templates
    ox, oy = out.get(x, {}), out.get(y, {})
    for z1, A1 in ox.items():
        if z1 == x or z1 == y:
            continue
        U1 = U.get(z1, ())
        for z2, A2 in out.get(z1, {}).items():
            if z2 == x or z2 == y or z2 == z1:
                continue
            A3r = oy.get(z2)
            if not A3r:
                continue
            U2 = U.get(z2, ())
            A3 = [(r, 1 - i) for r, i in A3r]
            for a2 in A2:
                for a3 in A3:                            # right: a2 ∘ a3 → s(z1, y); body a1 ∘ s
                    for s, u in inner.get((a2, a3), ()):
                        if (u is not None and u not in U2) or s in P.get((z1, y), ()):
                            continue
                        for a1 in A1:
                            if b2:
                                R.add(("B2", (a1, (s, 0)), ()))
                            if b2z:
                                for w in U1:
                                    R.add(("B2z", (a1, (s, 0)), ((w, "z"),)))
                for a1 in A1:                            # left: a1 ∘ a2 → s(x, z2); body s ∘ a3
                    for s, u in inner.get((a1, a2), ()):
                        if (u is not None and u not in U1) or s in P.get((x, z2), ()):
                            continue
                        for a3 in A3:
                            if b2:
                                R.add(("B2", ((s, 0), a3), ()))
                            if b2z:
                                for w in U2:
                                    R.add(("B2z", ((s, 0), a3), ((w, "z"),)))
    return R


class _Votes:
    def __init__(self):
        self.tot, self.hits, self.dhits = Counter(), Counter(), Counter()
        self.pairs = 0

    def add(self, direct, resid, labels):
        self.pairs += 1
        for b in direct:
            self.tot[b] += 1
            for h in labels:
                self.hits[(b, h)] += 1
                self.dhits[(b, h)] += 1
        for b in resid - direct:
            self.tot[b] += 1
            for h in labels:
                self.hits[(b, h)] += 1


class _NegVotes:
    """Counts for NEG bodies: per unary type pair (u1, u2) of a labelled pair, which relations and keys it holds."""

    def __init__(self, family):
        self.family = family
        self.A, self.Bp, self.Cf, self.Df = Counter(), Counter(), Counter(), Counter()
        self.Hh, self.Hs = Counter(), Counter()

    def add(self, c: Closure, x, y, labels):
        rels = set(c.pairs.get((x, y), ()))
        fams = {self.family(r) for r in rels}
        elig = [h for h in labels if self.family(h) not in fams]
        for u1 in c.unary.get(x, ()):
            for u2 in c.unary.get(y, ()):
                t = (u1, u2)
                self.A[t] += 1
                for s in rels:
                    self.Bp[t + (s,)] += 1
                    for f in fams:
                        self.Df[t + (s, f)] += 1
                for f in fams:
                    self.Cf[t + (f,)] += 1
                for h in elig:
                    self.Hh[t + (h,)] += 1
                    for s in rels:
                        self.Hs[t + (h, s)] += 1

    def count(self, u1, u2, s, h):
        """(hits, support): pairs of types (u1, u2) with s absent and nothing of h's key, and among them h labelled."""
        t, f = (u1, u2), self.family(h)
        tot = self.A[t] - self.Bp[t + (s,)] - self.Cf[t + (f,)] + self.Df[t + (s, f)]
        return self.Hh[t + (h,)] - self.Hs[t + (h, s)], tot


class GuardedRules:
    """See the module docstring. majority: the least share (> 0.5); support: the least hits; residual: residual votes
    from length-3 walks; rounds: the cap on rounds; neg: learn the NEG stratum; templates: the bodies counted;
    complements: relation → the relation whose key it shares (its complement); check_min: the least hits a rule needs on
    the check slice (0: a rule the check slice never meets is kept; its share there must be ≥ majority whenever the
    slice meets it); verify: after each round, recount the round's new rules on the fixed point of the whole new rule
    set (on the check slice, else on the examples) and retract — for good — each whose share there falls below
    majority, until none does (`retracted`); functional: relation → the entity type of its subject, or (subject type,
    object type) — a declaration that a subject of that type has at most one object for the relation (a type "t" is
    the unary is_t; a name starting with "is_" is the unary itself; without an object type the candidates are the
    entities carrying every is_* unary of the known object); each needs a declared complement; it adds the closed-world
    stratum at prediction time (`closed_world_in_rounds=True`: also in the fixed points learning counts on)."""

    def __init__(self, majority: float = 0.9, support: int = 2, *, residual: bool = True, rounds: int = 10,
                 neg: bool = True, templates: Iterable[str] = TEMPLATES, complements: Mapping[str, str] | None = None,
                 check_min: int = 0, verify: bool = True, functional: Mapping | None = None,
                 closed_world_in_rounds: bool = False, log=None):
        if not 0.5 < float(majority) <= 1.0:
            raise ValueError("majority is a share in (0.5, 1]")
        if int(support) < 1:
            raise ValueError("support is at least 1")
        if int(rounds) < 1:
            raise ValueError("rounds is at least 1")
        bad = set(templates) - set(TEMPLATES)
        if bad:
            raise ValueError(f"unknown templates {sorted(bad)}")
        self.majority, self.support, self.residual, self.max_rounds = float(majority), int(support), bool(residual), \
            int(rounds)
        self.neg, self.templates = bool(neg), frozenset(templates)
        self.complements = dict(complements or {})
        self._closed_spec = functional_spec(functional, self.complements)
        self.functional = {r: (su, ou) for r, (su, ou, _) in self._closed_spec.items()}
        self.closed_world_in_rounds = bool(closed_world_in_rounds)
        if self.closed_world_in_rounds and not self._closed_spec:
            raise ValueError("closed_world_in_rounds needs a functional= declaration")
        self.check_min = int(check_min)
        self.verify = bool(verify)
        self.log = log
        self._banned: set = set()
        self.retracted: list[GuardedRule] = []
        self._rules: list[GuardedRule] = []
        self.rounds = 0
        self.refused_by_check = 0
        self.history: list[dict] = []
        self._votes, self._check = None, None

    def family(self, rel):
        return self.complements.get(rel, rel)

    # ───────────────────────────── the rule set
    def with_rules(self, rules: Iterable[GuardedRule]) -> "GuardedRules":
        """Use these rules (no learning)."""
        self._rules = sorted(set(rules), key=_rule_key)
        return self

    @property
    def rules(self) -> list[GuardedRule]:
        return list(self._rules)

    def __len__(self):
        return len(self._rules)

    def __iter__(self):
        return iter(self._rules)

    def by_template(self) -> dict:
        c = Counter(r.template for r in self._rules)
        return {t: c.get(t, 0) for t in TEMPLATES}

    # ───────────────────────────── applying
    def closure(self, story: Story, *, justify: bool = False) -> Closure:
        """The fixed point on one story with every fact's status. justify=True: also every justification of every
        derived fact (`just`, `absent`; slower — what chain_rules writes); the facts and statuses are the same."""
        if justify:
            return _closure(story, self._rules, self.complements, justify=True, neg=True, statuses=True,
                            functional=self._closed_spec)
        return _fast(story, self._rules, self.complements, neg=True, statuses=True, functional=self._closed_spec)

    def predict(self, story: Story, query) -> frozenset:
        """The active relations on the query pair (x, y) at the fixed point; the empty set is a refusal."""
        return self.closure(story).answer(*query)

    def statuses(self, story: Story, query) -> dict:
        """Every relation present on the query pair at the fixed point → {"status", "stratum" ("given" | "positive" |
        "negative" | "closed"), "depth"}."""
        c = self.closure(story)
        return {r: c.info(*query, r) for r in sorted(c.relations(*query))}

    # ───────────────────────────── learning
    def _say(self, *a):
        if self.log is not None:
            self.log(" ".join(str(x) for x in a))

    def _pass(self, examples, rules, inner, neg=False):
        votes, nv = _Votes(), (_NegVotes(self.family) if neg else None)
        pos = [r for r in rules if r.template != "NEG"]
        cw = self._closed_spec if self.closed_world_in_rounds else None
        c, cur = None, None
        for i, (story, q, labels) in enumerate(examples):
            if story is not cur:                         # several queries of one story: one fixed point
                c, cur = _with_out(_fast(story, pos, self.complements, neg=False, statuses=False, functional=cw)), \
                    story
            x, y = q
            labels = frozenset(labels)
            if nv is not None:
                nv.add(c, x, y, labels)
                continue
            d = _bodies(c, x, y, self.templates)
            r = _residual(c, x, y, inner, self.templates) if inner else set()
            votes.add(d, r, labels)
        return nv if neg else votes

    def _admit(self, votes: _Votes, check: _Votes | None, rnd: int) -> list[GuardedRule]:
        m, s = self.majority - 1e-12, self.support
        cand = {}
        refused = 0
        by_body = defaultdict(list)
        for (b, h), n in votes.hits.items():
            if n >= s:
                by_body[b].append((h, n))
        for b in sorted(by_body, key=repr):
            tot = votes.tot[b]
            for h, n in sorted(by_body[b]):
                if n / tot < m:
                    continue
                t, atoms, unary = b
                if t in ("B1", "B1u") and atoms[0] == (h, 0):
                    continue
                if check is not None:
                    ct, chh = check.tot.get(b, 0), check.hits.get((b, h), 0)
                    if chh < self.check_min or (ct and chh / ct < m):
                        refused += 1
                        continue
                if GuardedRule(h, t, atoms, unary) in self._banned:
                    continue
                origin = "direct" if votes.dhits.get((b, h), 0) else "residual"
                cand[(h, b)] = GuardedRule(h, t, atoms, unary, n, tot, origin, rnd)
        out = []
        for (h, b), r in cand.items():
            if r.template == "B1u" and (h, ("B1", r.atoms, ())) in cand:
                continue
            if r.template == "B2z" and (h, ("B2", r.atoms, ())) in cand:
                continue
            out.append(r)
        self.refused_by_check = refused
        return sorted(out, key=_rule_key)

    def _inner(self, rules):
        inner = defaultdict(list)
        for r in rules:
            if r.template in ("B2", "B2z"):
                inner[r.atoms].append((r.head, r.unary[0][0] if r.unary else None))
        return dict(inner)

    def fit(self, examples: Iterable, check: Iterable | None = None) -> "GuardedRules":
        """Learn from labelled query pairs (story, (x, y), relations); `check`: labelled pairs of an independent slice.
        Deterministic and independent of the order of the examples."""
        examples = _sorted_examples(examples)
        check = None if check is None else _sorted_examples(check)
        rules: list[GuardedRule] = []
        self._banned, self.retracted = set(), []
        self.history, self.rounds = [], 0
        refused = 0
        for rnd in range(1, self.max_rounds + 1):
            t0 = time.perf_counter()
            v = self._pass(examples, rules, None)
            cv = self._pass(check, rules, None) if check is not None else None
            new = self._admit(v, cv, rnd)
            n_direct = len(set(new) - set(rules))
            if self.residual:                            # the rules of earlier rounds are in the fixed point already
                inner = self._inner([r for r in _merge(rules, new) if r not in set(rules)])
                if inner:
                    v = self._pass(examples, rules, inner)
                    cv = self._pass(check, rules, inner) if check is not None else None
                    new = self._admit(v, cv, rnd)
            refused = max(refused, self.refused_by_check)
            merged = _merge(rules, [r for r in new if r not in self._banned])
            n_retracted = 0
            if self.verify:                              # the joint fixed point: retract what it contradicts
                ver = check if check is not None else examples
                while True:
                    vv = self._pass(ver, merged, None)
                    old = set(rules)                     # verified together with the rules before them
                    sh = {r: vv.hits.get((r.body, r.head), 0) / vv.tot[r.body] for r in merged
                          if r not in old and vv.tot.get(r.body, 0)}
                    bad = sorted((r for r, x in sh.items() if x < self.majority - 1e-12),
                                 key=lambda r: (sh[r], -vv.tot[r.body], _rule_key(r)))
                    if not bad:
                        break
                    bad = bad[:(len(bad) + 1) // 2]      # worst first: a rule wrong only through others may recover
                    self._banned.update(bad)
                    self.retracted += bad
                    n_retracted += len(bad)
                    merged = [r for r in merged if r not in set(bad)]
            added = set(merged) - set(rules)
            self.rounds = rnd
            row = {"round": rnd, "rules": len(merged), "added": len(added), "direct_added": n_direct,
                   "residual_origin_added": sum(r.origin == "residual" for r in merged if r in added),
                   "by_template_added": dict(Counter(r.template for r in merged if r in added)),
                   "refused_by_check": self.refused_by_check, "retracted": n_retracted, "bodies": len(v.tot),
                   "seconds": round(time.perf_counter() - t0, 3)}
            self.history.append(row)
            self._say(f"round {rnd}: {row}")
            done = set(merged) == set(rules)
            rules = merged
            self._votes, self._check = v, cv
            if done:
                break
        self.refused_by_check = refused
        if self.neg and "NEG" in self.templates:
            t0 = time.perf_counter()
            nv = self._pass(examples, rules, None, neg=True)
            cnv = self._pass(check, rules, None, neg=True) if check is not None else None
            negs = self._admit_neg(nv, cnv)
            rules = sorted(rules + negs, key=_rule_key)
            self.history.append({"round": "NEG", "rules": len(negs), "seconds": round(time.perf_counter() - t0, 3)})
            self._say(f"NEG stratum: {len(negs)} rules")
        self._rules = rules
        return self

    def _admit_neg(self, nv: _NegVotes, cnv: _NegVotes | None) -> list[GuardedRule]:
        m, s = self.majority - 1e-12, self.support
        rels = sorted({k[2] for k in nv.Bp})
        out = []
        for (u1, u2, h), n in sorted(nv.Hh.items()):
            if n < s:
                continue
            for r in rels:
                if r == h or nv.Bp[(u1, u2, r)] < s:
                    continue
                hits, tot = nv.count(u1, u2, r, h)
                if hits < s or not tot or hits / tot < m:
                    continue
                if cnv is not None:
                    ch, ct = cnv.count(u1, u2, r, h)
                    if ch < self.check_min or (ct and ch / ct < m):
                        self.refused_by_check += 1
                        continue
                out.append(GuardedRule(h, "NEG", ((r, 0),), ((u1, "x"), (u2, "y")), hits, tot, "direct",
                                       self.rounds + 1))
        return out

    # ───────────────────────────── diagnostics
    def votes_for(self, head: str, body: tuple) -> dict:
        """The counts of the last round for one body (template, atoms, unary) and head: hits / support on the fit
        examples and on the check slice (None when no check slice was given)."""
        body = (body[0], tuple(tuple(a) for a in body[1]), tuple(tuple(u) for u in body[2]))
        v, c = self._votes, self._check
        out = {"hits": v.hits.get((body, head), 0) if v else 0, "support": v.tot.get(body, 0) if v else 0,
               "check_hits": None, "check_support": None}
        if c is not None:
            out["check_hits"], out["check_support"] = c.hits.get((body, head), 0), c.tot.get(body, 0)
        return out

    def near_misses(self, heads: Iterable[str], min_support: int = 20, k: int = 15) -> list[dict]:
        """The bodies of the last round most often right about each head without being admitted (share ≥ 0.5,
        support ≥ min_support), best share first: why a head has no rule."""
        if self._votes is None:
            return []
        have = {(r.head, r.body) for r in self._rules}
        v, c, out = self._votes, self._check, []
        heads = set(heads)
        for (b, h), n in v.hits.items():
            tot = v.tot[b]
            if h not in heads or tot < min_support or n / tot < 0.5 or (h, b) in have:
                continue
            row = {"rule": str(GuardedRule(h, *b)), "hits": n, "support": tot, "share": round(n / tot, 4)}
            if c is not None:
                row["check"] = [c.hits.get((b, h), 0), c.tot.get(b, 0)]
            out.append(row)
        out.sort(key=lambda r: (-r["share"], -r["support"], r["rule"]))
        by = defaultdict(list)
        for r in out:
            hd = r["rule"].split("(", 1)[0]
            if len(by[hd]) < k:
                by[hd].append(r)
        return [r for h in sorted(by) for r in by[h]]

    def templates_on(self, c: Closure, query, relations: Iterable[str]) -> dict:
        """Which rules give each relation on the query pair at the fixed point c: "given" (a story fact), "NEG" (the
        negative stratum), "closed" (the closed-world stratum) or the sorted templates of the positive rules whose body
        holds there (+ "closed" when the closed world gives it too)."""
        x, y = query
        pos = Closure(self._rules, self.complements)
        pos.pairs = {k: {r: v for r, v in d.items() if v[0] != 2} for k, d in c.pairs.items()}
        pos.unary = c.unary
        _with_out(pos)
        bodies = _bodies(pos, x, y, self.templates)
        out = {}
        for r in relations:
            st = c.stratum(x, y, r)
            if st == 0:
                out[r] = ["given"]
            elif st == 2:
                out[r] = ["NEG"]
            elif st == 3:
                out[r] = ["closed"]
            else:
                out[r] = sorted({g.template for g in self._rules if g.head == r and g.body in bodies})
                if (x, y, r) in c.closed:
                    out[r].append("closed")
        return out

    # ───────────────────────────── knowledge
    def to_knowledge(self, store, scope: dict | None = None, *, by: str = "relations:GuardedRules") -> dict:
        """Write each rule into a KnowledgeStore as a "rule" item: body {"kind": "guarded", "template", "atoms",
        "unary", "s": [the body's text, head], "r": "guarded", "o": head} (one key per body and head), source
        "outcome" (labelled pairs), evidence = hits, support, share, origin, round; and each functional declaration
        as a "rule" item of source "spec" (a declaration is a specification, not an observation; `functional_body`).
        Idempotent. → rule → item id (the guarded rules)."""
        from ...core.knowledge.store import item_id, plain_json
        sc = plain_json(scope or {})
        write_declarations(store, self._closed_spec, sc, by=by)
        out = {}
        for r in self._rules:
            body = plain_json(rule_body(r))
            iid = item_id("rule", body, sc)
            if store.status(iid) is None:
                ev = {"hits": r.hits, "support": r.support, "share": round(r.share, 6), "origin": r.origin,
                      "round": r.round}
                if store.add("rule", body, sc, source="outcome", by=by, evidence=ev) is None:
                    continue
            out[r] = iid
        return out


def functional_spec(functional, complements) -> dict:
    """The functional declarations normalized: relation → (subject unary, object unary | None, complement)."""
    out = {}
    anti = {}
    for k, v in sorted((complements or {}).items()):
        anti.setdefault(v, k)
    for r, t in sorted((functional or {}).items()):
        if isinstance(t, str):
            su, ou = t, None
        elif isinstance(t, (tuple, list)) and len(t) == 2:
            su, ou = t
        else:
            raise ValueError(f"functional[{r!r}] is a subject type or (subject type, object type)")
        if not su or not isinstance(su, str) or (ou is not None and (not ou or not isinstance(ou, str))):
            raise ValueError(f"functional[{r!r}]: types are non-empty names")
        if r not in anti:
            raise ValueError(f"functional {r!r} needs a declared complement (complements={{'Not_{r}': {r!r}}})")
        out[str(r)] = (_unary_of(su), None if ou is None else _unary_of(ou), anti[r])
    return out


def _unary_of(t: str) -> str:
    return t if t.startswith("is_") else f"is_{t}"


def functional_body(relation, spec) -> dict:
    """The body of a functional declaration item in a KnowledgeStore (one key per relation and subject type)."""
    su, ou, rbar = spec
    return {"kind": "functional", "s": [relation, su], "r": "functional", "o": ou, "complement": rbar}


def write_declarations(store, spec, scope, *, by) -> dict:
    """Each declaration as a "rule" item of source "spec" (idempotent). → relation → item id."""
    from ...core.knowledge.store import item_id, plain_json
    out = {}
    for r, sp in sorted(spec.items()):
        body = plain_json(functional_body(r, sp))
        iid = item_id("rule", body, scope)
        if store.status(iid) is None:
            if store.add("rule", body, scope, source="spec", by=by, evidence={"declared": "functional"}) is None:
                continue
        out[r] = iid
    return out


def _merge(old, new):
    """Rounds are monotone: the rules of earlier rounds stay (with the counts they were admitted with) and the new ones
    join; a guarded body is dropped once its unguarded generalization is in the set (its consequences are a subset)."""
    have = {r: r for r in old}
    for r in new:
        have.setdefault(r, r)
    keys = {(r.head, r.template, r.atoms) for r in have}
    out = [r for r in have.values()
           if not (r.template == "B1u" and (r.head, "B1", r.atoms) in keys)
           and not (r.template == "B2z" and (r.head, "B2", r.atoms) in keys)]
    return sorted(out, key=_rule_key)


def _sorted_examples(examples):
    """A canonical order (fit is independent of the order it is given): stories by content, queries within a story
    by pair and labels; queries of one story object stay together."""
    groups, keys = {}, {}
    for st, q, y in examples:
        k = id(st)
        if k not in groups:
            groups[k] = []
            keys[k] = (repr(st.facts), repr(st.unary))
        groups[k].append((st, q, frozenset(y)))
    out = []
    for k in sorted(groups, key=lambda k: keys[k]):
        out += sorted(groups[k], key=lambda e: (repr(e[1]), sorted(e[2])))
    return out


def _rule_key(r: GuardedRule):
    return (TEMPLATES.index(r.template), r.head, r.atoms, r.unary)


def rule_body(r: GuardedRule) -> dict:
    """The body of a rule item in a KnowledgeStore."""
    return {"kind": "guarded", "template": r.template, "atoms": [list(a) for a in r.atoms],
            "unary": [list(u) for u in r.unary], "s": [r.text(), r.head], "r": "guarded", "o": r.head}


def rule_from_body(body) -> GuardedRule | None:
    if not (isinstance(body, dict) and body.get("kind") == "guarded" and body.get("r") == "guarded"):
        return None
    try:
        return GuardedRule(body["o"], body["template"], tuple(tuple(a) for a in body["atoms"]),
                           tuple(tuple(u) for u in body.get("unary") or ()))
    except (KeyError, TypeError, ValueError):
        return None


# ───────────────────────────────────────────────────────────────── the store
def chain_rules(store, *, scope=None, rule_scope=None, rules="active", complements: Mapping[str, str] | None = None,
                neg: bool = True, functional: Mapping | None = None, max_facts=None, by="relations:chain_rules"):
    """Apply the guarded rules kept in `store` to the pair facts (`pair_fact`, single or multi) and unary facts
    (`unary_fact`) of each exact scope matching `scope`, to the positive fixed point, then the NEG stratum.

    rule_scope: where the rules are (default: scope); rules: "active" | "any" (hypothesis rules too: what they give is
    a hypothesis); complements: relation → the relation whose key it shares; derived pair facts are written as multi
    pair facts (`pair_fact(x, y, r, multi=True, family=complements.get(r, r))`). Every justification of a derived fact
    is written (premises: the binary and unary facts, and the rule; a justification that would rest on the fact it
    justifies is skipped), with `inferred=True` and the weakest premise's source; evidence {"rule", "premises" (in
    body order), "via", "depth", "stratum"} (+ "absent" for NEG). A NEG fact written by an earlier call whose condition
    no longer holds is retracted.

    The closed-world stratum last: the functional declarations kept in the rule scope (`to_knowledge` writes them;
    `functional=` writes them there first, as "spec" rule items) — for each active pair fact r(x, z) of a declared
    relation (given or derived, not of the NEG stratum) whose subject has the declared type, r̄(x, y) on every
    candidate object y ≠ z, justified by r(x, z) and the declaration (`derived_from`; the source of r(x, z); evidence
    stratum 3). Retracting the declaration or the positive takes the facts along; a closed fact that meets an explicit
    r(x, y) is a dispute. Idempotent. → ChainReport."""
    from ._chain import FACT_STATUSES, RULE_STATUSES, ChainLimit, ChainReport, _live, _pair, _rests_on, _scope_key, \
        pair_fact
    from ...core.knowledge.gates import RANK
    from ...core.knowledge.store import item_id, plain_json
    if rules not in RULE_STATUSES:
        raise ValueError('rules is "active" or "any"')
    comp = dict(complements or {})
    by_rank = {v: k for k, v in RANK.items()}
    t0 = time.perf_counter()
    rep = ChainReport()
    q0 = len(store.questions)
    if functional:
        write_declarations(store, functional_spec(functional, comp),
                           plain_json(rule_scope if rule_scope is not None else (scope or {})), by=by)
    rlist, rids, decls = [], [], []
    for rec in store.find(kind="rule", scope=scope if rule_scope is None else rule_scope, status=RULE_STATUSES[rules]):
        b = rec["body"]
        if isinstance(b, dict) and b.get("kind") == "functional" and b.get("r") == "functional" \
                and isinstance(b.get("s"), list) and len(b["s"]) == 2 and isinstance(b.get("complement"), str):
            decls.append((rec["id"], b["s"][0], b["s"][1], b.get("o"), b["complement"]))
            comp.setdefault(b["complement"], b["s"][0])
            continue
        r = rule_from_body(b)
        if r is not None:
            rlist.append(r)
            rids.append(rec["id"])
    decls.sort(key=repr)
    if not neg:
        keep = [i for i, r in enumerate(rlist) if r.template != "NEG"]
        rlist, rids = [rlist[i] for i in keep], [rids[i] for i in keep]
    rep.rules_read = len(rlist)
    stale, mine = defaultdict(list), set()               # NEG facts written earlier by this chain, per scope
    for iid, it in store.items.items():
        if it.kind != "fact" or store.status(iid) in (None, "retracted"):
            continue
        live = _live(it)
        if live and all(a.get("by") == by and (a.get("evidence") or {}).get("stratum") in (2, 3) for a in live):
            mine.add(iid)                                 # an output of the negative or closed stratum, never an input
            if all((a.get("evidence") or {}).get("stratum") == 2 for a in live) \
                    and (scope is None or all(it.scope.get(kk) == vv for kk, vv in scope.items())):
                stale[_scope_key(it.scope)].append(iid)
    groups, scopes = defaultdict(dict), {}
    for rec in store.find(kind="fact", scope=scope, status=FACT_STATUSES):
        if rec["id"] in mine:
            continue
        b = rec["body"]
        k = _scope_key(rec["scope"])
        p = _pair(b)
        if p is not None:
            groups[k].setdefault(("b",) + tuple(p), rec["id"])
        elif isinstance(b, dict) and isinstance(b.get("r"), str) and b["r"].startswith(UNARY + "/"):
            groups[k].setdefault(("u", b["s"], b["r"][len(UNARY) + 1:]), rec["id"])
        else:
            continue
        scopes[k] = rec["scope"]
    rep.facts_read = sum(len(g) for g in groups.values())
    derived_ids, depth_max = set(), 0
    for k in sorted(set(groups) | set(stale)):
        g, sc = groups.get(k, {}), scopes.get(k)
        facts = [(x, y, r) for t, x, y, r in (key for key in g if key[0] == "b")]
        unary = [(x, u) for t, x, u in (key for key in g if key[0] == "u")]
        st = Story(sorted(facts, key=repr), sorted(unary, key=repr))
        c = _closure(st, rlist, comp, justify=True, neg=neg, statuses=False, observe=False)
        ids = {("b",) + f: i for f, i in ((key[1:], v) for key, v in g.items() if key[0] == "b")}
        ids.update({key: v for key, v in g.items() if key[0] == "u"})

        def pid(p):
            return ids.get(("b",) + p if len(p) == 3 else ("u",) + p)
        ents = st.entities()
        cap = len(ents) ** 2 * max(1, len({r for _, _, r in c.facts()})) if max_facts is None else int(max_facts)
        new_items = 0
        counted = set()                                    # new items counted once (a batch resolves them at its end)
        neg_now = set()
        levels = defaultdict(list)
        for f in c.order:
            x, y, r = f
            s_, d_ = c.pairs[(x, y)][r]
            levels[(s_, d_)].append(f)
        deferred = []

        def write(f, lvl, ji, prem, ri, row, final):
            nonlocal new_items
            x, y, r = f
            pids = [pid(p) for p in prem]
            if any(p is None for p in pids):
                if not final:
                    deferred.append((f, lvl, ji, prem, ri))
                return
            body = plain_json(pair_fact(x, y, r, multi=True, family=comp.get(r, r)))
            iid = item_id("fact", body, sc)
            just = sorted(set(pids) | {rids[ri]})
            it = store.items.get(iid)
            if it is not None:
                if any(list(a["just"]) == just for a in _live(it)):
                    ids[("b",) + f] = iid
                    return
                if _rests_on(store, iid, pids):
                    row["circular"] += 1
                    return
            sts = [store.state.get(p) for p in pids]
            if any(s_ is None for s_ in sts):              # a premise written in this batch: not resolved yet
                if not final:
                    deferred.append((f, lvl, ji, prem, ri))
                return
            rank = min(s_["rank"] for s_ in sts)
            if rank not in by_rank:
                return
            new = it is None or store.status(iid) in (None, "retracted")
            rule = rlist[ri]
            via = prem[0][1] if rule.template in ("B2", "B2z", "BNE") else None
            ev = {"rule": rids[ri], "premises": pids, "via": via, "depth": lvl[1], "stratum": lvl[0]}
            if lvl[0] == 2:
                ev["absent"] = c.absent[f][ji]
            got = store.add("fact", body, sc, source=by_rank[rank], by=by, derived_from=just, evidence=ev,
                            inferred=True)
            if got is None:
                row["refused"] += 1
                return
            ids[("b",) + f] = iid
            row["proposed"] += 1
            derived_ids.add(iid)
            if new and iid not in counted:
                counted.add(iid)
                row["added"] += 1
                new_items += 1
                if new_items > cap:
                    raise ChainLimit(f"more than max_facts={cap} new facts in scope {k}")

        def account(row, lvl):
            nonlocal depth_max
            if row["proposed"] or row["refused"] or row["circular"]:
                rep.per_round.append(row)
            rep.facts_proposed += row["proposed"]
            rep.facts_added += row["added"]
            rep.refused += row["refused"]
            rep.circular += row["circular"]
            if row["proposed"]:
                depth_max = max(depth_max, lvl[1] + (1 if lvl[0] >= 2 else 0))

        def mkrow(lvl):
            return {"round": lvl[1] if lvl[0] == 1 else depth_max + 1, "proposed": 0, "added": 0, "refused": 0,
                    "circular": 0}

        for lvl in sorted(levels):                         # by stratum, then depth: premises are written first
            if lvl[0] == 2 and deferred:                   # the justifications whose premises came later
                row = mkrow((1, depth_max))
                with store.batch():
                    for f, l2, ji, prem, ri in sorted(deferred, key=lambda d: (d[1], repr(d[0]), d[2])):
                        write(f, l2, ji, prem, ri, row, True)
                deferred = []
                account(row, (1, 0))
            row = mkrow(lvl)
            with store.batch():
                for f in levels[lvl]:
                    if lvl[0] == 2:
                        neg_now.add(f)
                    for ji, (prem, ri) in enumerate(c.just.get(f, ())):
                        write(f, lvl, ji, prem, ri, row, False)
            account(row, lvl)
        if deferred:
            row = mkrow((1, depth_max))
            with store.batch():
                for f, l2, ji, prem, ri in sorted(deferred, key=lambda d: (d[1], repr(d[0]), d[2])):
                    write(f, l2, ji, prem, ri, row, True)
            account(row, (1, 0))
        if decls:                                          # the closed-world stratum, on the store's statuses
            U = defaultdict(set)
            for x, u in st.unary:
                U[x].add(u)
            by_u = defaultdict(list)
            for e in sorted(ents, key=repr):
                for u in U.get(e, ()):
                    by_u[u].append(e)
            todo = []
            for did, r, su, ou, rbar in decls:
                known = [(key[1], key[2], iid) for key, iid in sorted(ids.items(), key=lambda kv: repr(kv[0]))
                         if key[0] == "b" and key[3] == r and (key[1], key[2], r) not in neg_now
                         and su in U.get(key[1], ()) and store.status(iid) == "active"]
                nx = Counter(x for x, _, _ in known)
                rep.closed_conflicts += sum(1 for v in nx.values() if v > 1)
                for x, z, iid in known:
                    if nx[x] > 1:                          # several known positives: the declaration is contradicted
                        continue
                    if ou is not None:
                        cands = by_u.get(ou, [])
                    else:
                        kinds = sorted(u for u in U.get(z, ()) if u.startswith("is_"))
                        cands = sorted(set.intersection(*(set(by_u.get(u, ())) for u in kinds)), key=repr) \
                            if kinds else []
                    todo += [((x, y, rbar), iid, did, z) for y in cands if y != x and y != z]
            row = mkrow((3, 0))
            with store.batch():
                for (x, y, rbar), pid_, did, z in todo:
                    body = plain_json(pair_fact(x, y, rbar, multi=True, family=comp.get(rbar, rbar)))
                    iid = item_id("fact", body, sc)
                    just = sorted({pid_, did})
                    it = store.items.get(iid)
                    if it is not None and any(list(a["just"]) == just for a in _live(it)):
                        continue
                    s0 = store.state.get(pid_)
                    if s0 is None or s0["rank"] not in by_rank:
                        continue
                    new = it is None or store.status(iid) in (None, "retracted")
                    got = store.add("fact", body, sc, source=by_rank[s0["rank"]], by=by, derived_from=just,
                                    evidence={"rule": did, "premises": [pid_, did], "via": z, "depth": 0,
                                              "stratum": 3}, inferred=True)
                    if got is None:
                        row["refused"] += 1
                        continue
                    row["proposed"] += 1
                    derived_ids.add(iid)
                    if new and iid not in counted:
                        counted.add(iid)
                        row["added"] += 1
                        new_items += 1
                        if new_items > cap:
                            raise ChainLimit(f"more than max_facts={cap} new facts in scope {k}")
            account(row, (3, 0))
        for iid in stale.get(k, ()):
            it = store.items[iid]
            p = _pair(it.body)
            if p is not None and tuple(p) not in neg_now:
                store.retract(iid, by=by, why="the negative stratum's condition no longer holds")
                rep.retracted += 1
    rep.rounds = depth_max
    rep.disputes_opened = len(store.questions) - q0
    rep.hypotheses = sum(store.status(i) == "hypothesis" for i in derived_ids)
    rep.seconds = time.perf_counter() - t0
    return rep


__all__ = ["Closure", "GuardedRule", "GuardedRules", "Story", "TEMPLATES", "chain_rules", "rule_body",
           "rule_from_body", "unary_fact"]
