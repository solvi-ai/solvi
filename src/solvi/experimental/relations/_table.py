"""A composition table of named relations learned by exact counting (kinship chains: father ∘ sister → aunt).

    t = CompositionTable(majority=0.9, support=2).fit(examples)   # examples: (chain, relation), chains of length 2 and 3
    d = t.derive(("father", "sister", "son"))                     # CYK over every bracketing
    d.relation, d.reason, d.cells                                 # "cousin", "derived", the cells used — or None + why
    t.to_knowledge(store, {"domain": "kinship"})                  # one "rule" item per cell, with its support

Learning. A chain of length 2 (r1, r2) labelled r is one vote for the cell r1 ∘ r2 → r. A cell is admitted when its
most frequent relation has a share of the votes ≥ `majority` and at least `support` votes (and `admit`, when given,
says yes). Residual fill: a chain (a, b, c) labelled r whose a ∘ b = x is in the table is a vote for x ∘ c → r, and
one whose b ∘ c = y is in the table is a vote for a ∘ y → r. The residual votes are recomputed from the length-2 votes
and the current table each round (a chain counts once per round, never once per round accumulated) until the table
stops changing or `rounds` runs out. With `identity=` a name, a chain (a, b, c) labelled c (or (a, b, c) labelled a)
is a vote for a ∘ b (b ∘ c) → identity where the length-2 votes alone admit no cell there; the identity then
composes as neutral and is never an answer.

Derivation. CYK over the chain: every span holds the relations it composes to and how many bracketings give each; a
bracketing that meets an empty cell gives nothing. agree="all": answered when every bracketing that gives a relation
gives the same one; "majority": the relation of strictly the most bracketings. No relation → "underivable"; more than
one under "all" (or a tie) → "disagree"; only the identity → "identity". The answer's reason is the set of cells used by
the bracketings that give it; an "underivable" refusal names the absent cells that block it (`Derivation.missing`), and
`why_not` reads them over every path of a story — a refusal becomes a learning target: the cell to learn."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Iterator, Mapping, Sequence

Admit = Callable[[tuple, str, int, int, int], bool]


@dataclass(frozen=True)
class Cell:
    """One admitted cell: left ∘ right → relation, with hits votes for it among support votes (the votes of chains
    of length 2 and the residual votes of chains of length 3 together); origin "pair" (the cell has votes from chains
    of length 2), "residual" (only from chains of length 3) or "identity"."""
    left: str
    right: str
    relation: str
    hits: int
    support: int
    origin: str

    @property
    def share(self) -> float:
        return self.hits / self.support if self.support else 0.0


@dataclass(frozen=True)
class Derivation:
    """What derive() returns. relation: the answer or None; reason: "derived" | "underivable" | "disagree" |
    "identity"; candidates: relation → number of bracketings giving it; cells: the cells used by the answer's
    bracketings (empty for a refusal; explain(chain, relation) shows a candidate's); missing: for an "underivable"
    refusal, the absent cells (left, right) that block it — on the shortest span no bracketing composes, the pairs of
    its two composed parts that have no cell, most blocked chains first (derive_paths) — the rule to learn, by name
    (`why_not` reads it over a story's paths); () otherwise."""
    chain: tuple
    relation: str | None
    reason: str
    candidates: dict = field(default_factory=dict)
    cells: tuple = ()
    missing: tuple = ()

    @property
    def answered(self) -> bool:
        return self.relation is not None

    @property
    def bracketings(self) -> int:
        return sum(self.candidates.values())


def base_rates(examples: Iterable) -> dict[str, float]:
    """The share of each relation among the labels of the chains of length 2 in `examples` ((chain, relation) pairs) —
    the base-rate null of `sized_admit`."""
    c = Counter(r for chain, r in examples if len(chain) == 2)
    n = sum(c.values())
    return {r: k / n for r, k in sorted(c.items())} if n else {}


def sized_admit(null: Mapping[str, float] | float, delta: float = 0.1) -> Admit:
    """An `admit=` hook with the sized acceptance of `solvi.experimental.laws`: a cell whose majority relation r has
    `hits` of `support` votes passes when P[Binomial(support, null_r) ≥ hits] ≤ δ / N, N the number of cells tested
    (cells with votes; a declared space, Bonferroni). null: per relation (base_rates) or one number."""
    from ..laws._core import _log_tail, sized_level
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta is a probability strictly between 0 and 1")

    def admit(cell, relation, hits, support, tested):
        p = null if isinstance(null, (int, float)) else null.get(relation, 0.0)
        p = min(max(float(p), 1e-12), 1.0 - 1e-12)
        n = max(1, int(tested))
        return _log_tail(int(support), int(hits), p) <= math.log(sized_level(n, delta, n)) + 1e-12
    return admit


class CompositionTable:
    """See the module docstring. majority: the least share of the majority relation (> 0.5); support: the least votes
    for it; residual: fill from chains of length 3; rounds: the cap on residual rounds; agree: "all" | "majority";
    identity: the name of a learned identity relation (None: off); admit: an extra gate (cell, relation, hits,
    support, tested) → bool, e.g. sized_admit(base_rates(examples))."""

    def __init__(self, majority: float = 0.9, support: int = 2, *, residual: bool = True, rounds: int = 10,
                 agree: str = "all", identity: str | None = None, admit: Admit | None = None):
        if not 0.5 < float(majority) <= 1.0:
            raise ValueError("majority is a share in (0.5, 1]")
        if int(support) < 1:
            raise ValueError("support is at least 1")
        if agree not in ("all", "majority"):
            raise ValueError('agree is "all" or "majority"')
        self.majority, self.support, self.residual, self.max_rounds = float(majority), int(support), bool(residual), \
            int(rounds)
        self.agree, self.identity, self.admit = agree, identity, admit
        self._cells: dict[tuple, Cell] = {}
        self.votes: dict[tuple, Counter] = {}
        self.rounds = 0
        self.tested = 0
        self.refused_by_admit = 0
        self.ignored = 0

    # ───────────────────────────── learning
    def fit(self, examples: Iterable) -> "CompositionTable":
        """Learn from (chain, relation) pairs: chains of length 2 vote, chains of length 3 fill residuals; other
        lengths are counted in `ignored`. Deterministic and independent of the order of the examples."""
        pairs, triples = Counter(), Counter()
        self.ignored = 0
        for chain, rel in examples:
            chain = tuple(chain)
            if len(chain) == 2:
                pairs[(chain, rel)] += 1
            elif len(chain) == 3:
                triples[(chain, rel)] += 1
            else:
                self.ignored += 1
        base = defaultdict(Counter)
        for (chain, rel), k in pairs.items():
            base[chain][rel] += k
        paired = set(base)
        if self.identity is not None:
            plain = self._accept(base, paired)            # the identity is voted only where the pairs admit nothing
            for ((a, b, c), rel), k in sorted(triples.items()):
                if rel == c and (a, b) not in plain:
                    base[(a, b)][self.identity] += k
                if rel == a and (b, c) not in plain:
                    base[(b, c)][self.identity] += k
        votes = base
        cells = self._accept(votes, paired)
        self.rounds = 0
        if self.residual and triples:
            for _ in range(self.max_rounds):
                votes = defaultdict(Counter, {k: Counter(v) for k, v in base.items()})
                for ((a, b, c), rel), k in sorted(triples.items()):
                    left, right = cells.get((a, b)), cells.get((b, c))
                    if left is not None and left.relation != self.identity:
                        votes[(left.relation, c)][rel] += k
                    if right is not None and right.relation != self.identity:
                        votes[(a, right.relation)][rel] += k
                self.rounds += 1
                new = self._accept(votes, paired)
                if new == cells:
                    break
                cells = new
        self._cells, self.votes = cells, {k: Counter(v) for k, v in sorted(votes.items())}
        return self

    def _accept(self, votes, paired) -> dict:
        out, self.tested, self.refused_by_admit = {}, 0, 0
        for key in sorted(votes):
            c = votes[key]
            tot = sum(c.values())
            if not tot:
                continue
            self.tested += 1
        for key in sorted(votes):
            c = votes[key]
            tot = sum(c.values())
            if not tot:
                continue
            rel, n = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if n < self.support or n / tot < self.majority - 1e-12:
                continue
            if self.admit is not None and not self.admit(key, rel, n, tot, self.tested):
                self.refused_by_admit += 1
                continue
            origin = "identity" if rel == self.identity else "pair" if key in paired else "residual"
            out[key] = Cell(key[0], key[1], rel, n, tot, origin)
        return out

    # ───────────────────────────── the table
    @property
    def cells(self) -> dict[tuple, str]:
        return {k: c.relation for k, c in self._cells.items()}

    def cell(self, left: str, right: str) -> Cell | None:
        return self._cells.get((left, right))

    @property
    def conflicts(self) -> int:
        """Cells whose votes name more than one relation."""
        return sum(1 for c in self.votes.values() if len(c) > 1)

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[Cell]:
        return iter(self._cells[k] for k in sorted(self._cells))

    # ───────────────────────────── derivation
    def _cyk(self, chain):
        span, back = self._spans(chain)
        return span[(0, len(chain))], back

    def _missing(self, chain, span) -> list:
        """The absent cells on the shortest span that composes to nothing while both parts of a split compose."""
        n = len(chain)
        for length in range(2, n + 1):
            out = set()
            for i in range(0, n - length + 1):
                j = i + length
                if span[(i, j)]:
                    continue
                for m in range(i + 1, j):
                    for r1 in span[(i, m)]:
                        for r2 in span[(m, j)]:
                            if (r1, r2) not in self._cells and self.identity not in (r1, r2):
                                out.add((r1, r2))
            if out:
                return sorted(out)
        return []

    def _spans(self, chain):
        n = len(chain)
        span = {(i, i + 1): Counter({r: 1}) for i, r in enumerate(chain)}
        back = defaultdict(list)                      # (i, j, r) → [(m, r1, r2, cell key or None)]
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                acc = Counter()
                for m in range(i + 1, j):
                    for r1, c1 in sorted(span[(i, m)].items()):
                        for r2, c2 in sorted(span[(m, j)].items()):
                            if self.identity is not None and r1 == self.identity:
                                r, key = r2, None
                            elif self.identity is not None and r2 == self.identity:
                                r, key = r1, None
                            else:
                                cell = self._cells.get((r1, r2))
                                if cell is None:
                                    continue
                                r, key = cell.relation, (r1, r2)
                            acc[r] += c1 * c2
                            back[(i, j, r)].append((m, r1, r2, key))
                span[(i, j)] = acc
        return span, back

    def _used(self, back, n, relation) -> tuple:
        seen, out, stack = set(), set(), [(0, n, relation)]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            i, j, _ = node
            for m, r1, r2, key in back.get(node, ()):
                if key is not None:
                    out.add(key)
                stack += [(i, m, r1), (m, j, r2)]
        return tuple(self._cells[k] for k in sorted(out))

    def _decide(self, chain, total: Counter, used: Callable) -> Derivation:
        cand = {r: k for r, k in sorted(total.items()) if r != self.identity}
        if not cand:
            reason = "identity" if self.identity is not None and total.get(self.identity) else "underivable"
            return Derivation(chain, None, reason, dict(sorted(total.items())))
        if len(cand) == 1:
            r = next(iter(cand))
        elif self.agree == "majority":
            ranked = sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))
            if ranked[0][1] == ranked[1][1]:
                return Derivation(chain, None, "disagree", cand)
            r = ranked[0][0]
        else:
            return Derivation(chain, None, "disagree", cand)
        return Derivation(chain, r, "derived", cand, used(r))

    def derive(self, chain: Sequence[str]) -> Derivation:
        """The relation the chain composes to under the table, or a refusal with its reason (see the module)."""
        chain = tuple(chain)
        if not chain:
            raise ValueError("an empty chain")
        span, back = self._spans(chain)
        d = self._decide(chain, span[(0, len(chain))], lambda r: self._used(back, len(chain), r))
        if d.reason == "underivable":
            d = replace(d, missing=tuple(self._missing(chain, span)))
        return d

    def derive_paths(self, chains: Iterable[Sequence[str]]) -> Derivation:
        """One query reached by several chains (e.g. every shortest path between two people): the bracketings of every
        chain are pooled under the same agreement rule; the cells are the union of the cells each chain used."""
        chains = [tuple(c) for c in chains]
        if not chains:
            return Derivation((), None, "underivable")
        total, backs, blocked = Counter(), [], Counter()
        for c in chains:
            span, b = self._spans(c)
            t = span[(0, len(c))]
            total.update(t)
            backs.append((c, t, b))
            if not t:
                blocked.update(self._missing(c, span))

        def used(r):
            keys = {}
            for c, t, b in backs:
                if t.get(r):
                    for cell in self._used(b, len(c), r):
                        keys[(cell.left, cell.right)] = cell
            return tuple(keys[k] for k in sorted(keys))
        d = self._decide(tuple(chains), total, used)
        if d.reason == "underivable":
            d = replace(d, missing=tuple(k for k, _ in sorted(blocked.items(), key=lambda kv: (-kv[1], kv[0]))))
        return d

    def explain(self, chain: Sequence[str], relation: str | None = None) -> tuple:
        """The cells the bracketings that give `relation` use (default: the derived answer; () when refused)."""
        chain = tuple(chain)
        if relation is None:
            return self.derive(chain).cells
        total, back = self._cyk(chain)
        return self._used(back, len(chain), relation) if total.get(relation) else ()

    # ───────────────────────────── knowledge
    def to_knowledge(self, store, scope: dict | None = None, *, by: str = "relations:CompositionTable") -> dict:
        """Write each cell into a KnowledgeStore as a "rule" item: body {"kind": "composition", "s": [left, right],
        "r": "compose", "o": relation} (one value per cell and scope), source "outcome" (labelled chains), evidence =
        hits, support, share, origin. Idempotent. → (left, right) → item id."""
        from ...core.knowledge.store import item_id, plain_json
        sc = plain_json(scope or {})
        out = {}
        for c in self:
            body = {"kind": "composition", "s": [c.left, c.right], "r": "compose", "o": c.relation}
            iid = item_id("rule", plain_json(body), sc)
            if store.status(iid) is None:
                ev = {"hits": c.hits, "support": c.support, "share": round(c.share, 6), "origin": c.origin}
                got = store.add("rule", body, sc, source="outcome", by=by, evidence=ev)
                if got is None:
                    continue
            out[(c.left, c.right)] = iid
        return out


__all__ = ["Cell", "CompositionTable", "Derivation", "base_rates", "sized_admit"]
