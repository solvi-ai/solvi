"""Kinship sentences read into a relation graph by code (a text frontend), with a global consistency check.

    fe = TextFrontend()                                         # KINSHIP: 16 relations, their genders and inverses
    g = fe.read("[Ann] took her son [Bob] out. [Bob]'s sister [Kim] sang.", genders={"Ann": "female"})
    g.edges        # TextEdge(x, relation, y, pattern, sentence, inverse): «y is relation of x», with inverses
    g.genders      # name → GenderFact(gender, source): "declared", "inferred:<relation word>", "inferred:pronoun"
    g.dropped      # what the check removed and why ("gender", "pair", "functional")
    t.derive_paths(g.paths("Ann", "Kim"))                       # → "daughter" under a fitted CompositionTable
    fe.story(text) # a Story (facts (x, y, relation), unary is_male / is_female) for GuardedRules / forward_chain

Patterns (one sentence at a time; names bracketed `[Name]`, or given as a list and bracketed by `read`):
  x_rel_y         «[X]'s REL [Y]», «[X]'s REL, [Y]»
  y_x_rel         «[Y], [X]'s REL»
  y_is_rel_of_x   «[Y] is / was (the | a | an) REL of [X]»
  y_the_rel_of_x  «[Y], (the | a | an) REL of [X]»
  his_her_rel_y   «his / her REL [Y]»: the owner X is the nearest name before the phrase, in the same sentence, whose
                  gender is the pronoun's and that is not itself the object of a relation phrase of the sentence (an
                  enumeration «[R] took his son [P] and his daughter [J]» keeps R as the owner of both); no such name:
                  no edge.
Each pattern reads «Y is REL of X» as the edge (X, REL, Y). Every edge carries its inverse (Y, REL⁻, X), REL⁻ taken
from the relation table by X's gender; with X's gender unknown and the two inverses different, no inverse is added
(`TextGraph.no_inverse` lists those edges). Genders: declared (`genders=`) first, else inferred from the relation
words that name the person (mother ⇒ female); two words that disagree leave the gender unknown (source "conflict").
An owner of unknown gender is taken only when it is the one eligible name before the phrase (source
"inferred:pronoun").

The check (`GraphCheck`, on by default; `check=False` keeps the raw edges — the control): an edge (X, REL, Y) whose
relation's gender contradicts Y's gender (declared, or inferred without conflict; or the inferred words disagree);
two different relations on one ordered pair; two different people as one person's father / mother / husband / wife.
A contradiction drops every stated edge involved, with its inverse — never a choice between them — and names it in
`TextGraph.dropped`. The check is code; no model.

Limits: English kinship phrasings of the patterns above; a sentence no pattern matches gives no edge (a refusal that
lowers coverage, never a guess); the frontend is not a general relation extractor."""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ._rules import Story

MALE, FEMALE = "male", "female"
# relation → (gender of the person it names, the inverse when the other person is male, … female)
KINSHIP: dict[str, tuple[str | None, str, str]] = {
    "father": (MALE, "son", "daughter"), "mother": (FEMALE, "son", "daughter"),
    "son": (MALE, "father", "mother"), "daughter": (FEMALE, "father", "mother"),
    "brother": (MALE, "brother", "sister"), "sister": (FEMALE, "brother", "sister"),
    "husband": (MALE, "husband", "wife"), "wife": (FEMALE, "husband", "wife"),
    "grandfather": (MALE, "grandson", "granddaughter"), "grandmother": (FEMALE, "grandson", "granddaughter"),
    "grandson": (MALE, "grandfather", "grandmother"), "granddaughter": (FEMALE, "grandfather", "grandmother"),
    "uncle": (MALE, "nephew", "niece"), "aunt": (FEMALE, "nephew", "niece"),
    "nephew": (MALE, "uncle", "aunt"), "niece": (FEMALE, "uncle", "aunt"),
}
FUNCTIONAL = ("father", "mother", "husband", "wife")
PATTERNS = ("x_rel_y", "y_x_rel", "y_is_rel_of_x", "y_the_rel_of_x", "his_her_rel_y")
_NAME = r"\[([^\]]+)\]"
_NAME_RE = re.compile(_NAME)
_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextEdge:
    """«y is `relation` of x» — the edge (x, relation, y) as a Story fact (x, y, relation); `pattern` the one that
    read it (an inverse carries its source's), `sentence` its index, `inverse` True for the edge added by the table."""
    x: str
    relation: str
    y: str
    pattern: str
    sentence: int
    inverse: bool = False

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.x, self.relation, self.y)


@dataclass(frozen=True)
class GenderFact:
    """A person's gender (None: unknown) and where it came from: "declared", "inferred:<relation word>",
    "inferred:pronoun", "conflict" (relation words disagree) or "unknown"."""
    name: str
    gender: str | None
    source: str


@dataclass(frozen=True)
class Dropped:
    """An edge the check removed (with its inverse): reason "gender" | "pair" | "functional" and the contradiction."""
    edge: TextEdge
    reason: str
    detail: str


@dataclass(frozen=True)
class TextGraph:
    """What `TextFrontend.read` returns: the kept edges (stated and inverse, in reading order), the genders, the
    dropped edges with their reasons, the stated edges with no inverse (the owner's gender unknown), the names."""
    edges: tuple[TextEdge, ...]
    genders: Mapping[str, GenderFact]
    dropped: tuple[Dropped, ...] = ()
    no_inverse: tuple[TextEdge, ...] = ()
    names: tuple[str, ...] = ()
    raw: tuple[TextEdge, ...] = field(default=(), repr=False)

    def facts(self) -> list[tuple[str, str, str]]:
        """(x, y, relation) per kept edge — the Story / pair_fact order."""
        return [(e.x, e.y, e.relation) for e in self.edges]

    def unary(self) -> list[tuple[str, str]]:
        return [(g.name, f"is_{g.gender}") for g in self.genders.values() if g.gender is not None]

    def story(self) -> Story:
        return Story.from_facts(self.facts(), self.unary())

    def paths(self, x: str, y: str, cap: int = 32) -> list[tuple[str, ...]]:
        """Every shortest directed path x → y over the kept edges, as relation chains (at most `cap`) — the input of
        `CompositionTable.derive_paths`; [] when y is not reachable."""
        adj: dict[str, list] = defaultdict(list)
        for e in self.edges:
            adj[e.x].append((e.y, e.relation))
        dist, preds, dq = {x: 0}, defaultdict(list), deque([x])
        while dq:
            u = dq.popleft()
            for v, r in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
                    preds[v].append((u, r))
                elif dist[v] == dist[u] + 1:
                    preds[v].append((u, r))
        if y not in dist or x == y:
            return []
        out: list[tuple[str, ...]] = []

        def back(v, acc):
            if len(out) >= cap:
                return
            if v == x:
                out.append(tuple(acc[::-1]))
                return
            for u, r in preds[v]:
                back(u, acc + [r])
        back(y, [])
        return out


def _validate(relations: Mapping[str, tuple]) -> dict[str, tuple[str | None, str, str]]:
    rel = {}
    for r, v in relations.items():
        if not isinstance(r, str) or not r or not isinstance(v, tuple) or len(v) != 3:
            raise ValueError(f"relation table: {r!r} → (gender, inverse if male, inverse if female)")
        g, im, iff = v
        if g not in (MALE, FEMALE, None):
            raise ValueError(f"relation table: the gender of {r!r} is 'male', 'female' or None, not {g!r}")
        for inv in (im, iff):
            if inv not in relations:
                raise ValueError(f"relation table: the inverse {inv!r} of {r!r} is not a relation of the table")
        rel[r] = (g, im, iff)
    return rel


class GraphCheck:
    """The global check (see the module): gender consistency, one relation per ordered pair, at most one person per
    functional relation (`functional`, default father / mother / husband / wife). `__call__(claims, genders)` →
    (kept claims, dropped); a claim is a stated edge with its inverse (or None)."""

    def __init__(self, functional: Sequence[str] = FUNCTIONAL, relations: Mapping[str, tuple] = KINSHIP):
        self.relations = _validate(relations)
        for r in functional:
            if r not in self.relations:
                raise ValueError(f"functional relation {r!r} is not in the relation table")
        self.functional = tuple(functional)

    def __call__(self, claims: Sequence[tuple[TextEdge, TextEdge | None]], genders: Mapping[str, GenderFact]):
        drop: dict[int, tuple[str, str]] = {}

        def edges(i):
            s, inv = claims[i]
            return (s,) if inv is None else (s, inv)
        live = lambda: [i for i in range(len(claims)) if i not in drop]  # noqa: E731
        # gender: the relation names a person of one gender
        for i in live():
            for e in edges(i):
                want = self.relations[e.relation][0]
                g = genders.get(e.y)
                if want is None or g is None:
                    continue
                if g.source == "conflict":
                    drop[i] = ("gender", f"{e.y}: the relation words disagree on the gender ({e.relation} is one)")
                elif g.gender is not None and g.gender != want:
                    drop[i] = ("gender", f"{e.y} is {g.gender} ({g.source}); {e.relation} names a {want} person")
                if i in drop:
                    break
        # one relation per ordered pair
        on_pair: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i in live():
            for e in edges(i):
                on_pair[(e.x, e.y)][e.relation].append(i)
        for (x, y), rs in on_pair.items():
            if len(rs) > 1:
                for r, ids in rs.items():
                    for i in ids:
                        drop.setdefault(i, ("pair", f"({x}, {y}) read as {' and '.join(sorted(rs))}"))
        # at most one father / mother / husband / wife
        holder: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i in live():
            for e in edges(i):
                if e.relation in self.functional:
                    holder[(e.x, e.relation)][e.y].append(i)
        for (x, r), ys in holder.items():
            if len(ys) > 1:
                for y, ids in ys.items():
                    for i in ids:
                        drop.setdefault(i, ("functional", f"{x} has {len(ys)} as {r}: {', '.join(sorted(ys))}"))
        kept = [claims[i] for i in range(len(claims)) if i not in drop]
        dropped = [Dropped(claims[i][0], *drop[i]) for i in sorted(drop)]
        return kept, dropped


class TextFrontend:
    """See the module docstring. relations: the relation table (relation → (gender it names, inverse if the other
    person is male, inverse if female)); genders: default declared genders (name → "male" / "female"), overridden
    per call; check: True (the default `GraphCheck`), a `GraphCheck`, or False (raw edges — the control)."""

    def __init__(self, relations: Mapping[str, tuple] = KINSHIP, genders: Mapping[str, str] | None = None,
                 check: bool | GraphCheck = True):
        self.relations = _validate(relations)
        self.genders = dict(genders or {})
        if check is True:
            check = GraphCheck(relations=self.relations,
                               functional=tuple(r for r in FUNCTIONAL if r in self.relations))
        self.check: GraphCheck | None = check or None
        rel = "(" + "|".join(re.escape(r) for r in sorted(self.relations, key=lambda r: (-len(r), r))) + ")"
        n = _NAME
        self._patterns = (
            (re.compile(n + r"'s " + rel + r",? " + n), "x_rel_y"),
            (re.compile(n + r", " + n + r"'s " + rel + r"(?![\w-])"), "y_x_rel"),
            (re.compile(n + r",? (?:is|was) (?:the |a |an )?" + rel + r" of " + n), "y_is_rel_of_x"),
            (re.compile(n + r", (?:the |a |an )?" + rel + r" of " + n), "y_the_rel_of_x"),
        )
        self._his = re.compile(r"\b(his|her) " + rel + r",? " + n)

    @staticmethod
    def mark(text: str, names: Iterable[str]) -> str:
        """Bracket every whole-word occurrence of each name not already bracketed (longest names first)."""
        for nm in sorted(set(names), key=lambda s: (-len(s), s)):
            text = re.sub(r"(?<![\[\w])" + re.escape(nm) + r"(?![\]\w])", f"[{nm}]", text)
        return text

    def _sentences(self, text: str) -> list[str]:
        return [s for s in _SPLIT.split(text.strip()) if s]

    def _stated(self, sents):
        """Pass 1: the edges of the four name-only patterns and the his/her matches, per sentence."""
        out = []
        for k, sent in enumerate(sents):
            found = []
            for pat, kind in self._patterns:
                for m in pat.finditer(sent):
                    if kind == "x_rel_y":
                        x, r, y = m.group(1), m.group(2), m.group(3)
                    elif kind == "y_x_rel":
                        y, x, r = m.group(1), m.group(2), m.group(3)
                    else:
                        y, r, x = m.group(1), m.group(2), m.group(3)
                    if x != y:
                        found.append((x, r, y, kind))
            out.append((k, sent, found, list(self._his.finditer(sent))))
        return out

    def read(self, text: str, names: Iterable[str] | None = None,
             genders: Mapping[str, str] | None = None) -> TextGraph:
        """Read a story into a TextGraph (see the module)."""
        if names is not None:
            names = list(names)
            text = self.mark(text, names)
        declared = {**self.genders, **(genders or {})}
        for nm, g in declared.items():
            if g not in (MALE, FEMALE):
                raise ValueError(f"gender of {nm!r} is 'male' or 'female', not {g!r}")
        sents = self._sentences(text)
        seen_names = list(dict.fromkeys(_NAME_RE.findall(text)))
        if names is not None:
            seen_names = list(dict.fromkeys(list(names) + seen_names))
        parsed = self._stated(sents)
        # genders inferred from the relation words that name a person
        words: dict[str, dict[str, str]] = defaultdict(dict)
        for _, _, found, his in parsed:
            for r, y in [(r, y) for _x, r, y, _ in found] + [(m.group(2), m.group(3)) for m in his]:
                g = self.relations[r][0]
                if g is not None:
                    words[y].setdefault(g, r)
        gf: dict[str, GenderFact] = {}
        for nm in seen_names:
            if nm in declared:
                gf[nm] = GenderFact(nm, declared[nm], "declared")
            elif len(words.get(nm, {})) == 1:
                (g, r), = words[nm].items()
                gf[nm] = GenderFact(nm, g, f"inferred:{r}")
            elif len(words.get(nm, {})) > 1:
                gf[nm] = GenderFact(nm, None, "conflict")
            else:
                gf[nm] = GenderFact(nm, None, "unknown")
        # pass 2: owners of his / her, then the stated edges in reading order
        stated: dict[tuple, TextEdge] = {}
        for k, sent, found, his in parsed:
            for x, r, y, kind in found:
                stated.setdefault((x, r, y), TextEdge(x, r, y, kind, k))
            objs = {y for _, _, y, _ in found}
            his_obj = {(m.start(3) - 1, m.group(3)) for m in his}
            for m in his:
                want = MALE if m.group(1) == "his" else FEMALE
                r, y = m.group(2), m.group(3)
                prev = [(mm.start(), mm.group(1)) for mm in _NAME_RE.finditer(sent) if mm.start() < m.start()]
                cand = [nm for p, nm in sorted(prev, reverse=True)
                        if (p, nm) not in his_obj and nm not in objs and nm != y]
                owner = next((nm for nm in cand if gf[nm].gender == want), None)
                if owner is None:
                    unknown = [nm for nm in cand if gf[nm].gender is None and gf[nm].source == "unknown"]
                    if len(set(cand)) == 1 and unknown:
                        owner = unknown[0]
                        gf[owner] = GenderFact(owner, want, "inferred:pronoun")
                if owner is not None:
                    stated.setdefault((owner, r, y), TextEdge(owner, r, y, "his_her_rel_y", k))
        # inverses by the owner's gender
        claims: list[tuple[TextEdge, TextEdge | None]] = []
        no_inv = []
        for (x, r, y), e in stated.items():
            _, im, iff = self.relations[r]
            g = gf.get(x, GenderFact(x, None, "unknown")).gender
            inv_r = im if im == iff else (im if g == MALE else iff if g == FEMALE else None)
            if inv_r is None:
                no_inv.append(e)
                claims.append((e, None))
            elif (y, inv_r, x) in stated:
                claims.append((e, None))                               # both directions stated
            else:
                claims.append((e, TextEdge(y, inv_r, x, e.pattern, e.sentence, True)))
        raw = tuple(z for c in claims for z in c if z is not None)
        dropped: list[Dropped] = []
        if self.check is not None:
            claims, dropped = self.check(claims, gf)
        edges, seen = [], set()
        for c in claims:
            for z in c:
                if z is not None and z.triple not in seen:
                    seen.add(z.triple)
                    edges.append(z)
        return TextGraph(tuple(edges), gf, tuple(dropped), tuple(no_inv), tuple(seen_names), raw)

    def story(self, text: str, names: Iterable[str] | None = None,
              genders: Mapping[str, str] | None = None) -> Story:
        """`read(...).story()`: the facts (x, y, relation) and the gender facts as a Story."""
        return self.read(text, names, genders).story()


__all__ = ["FUNCTIONAL", "KINSHIP", "PATTERNS", "Dropped", "GenderFact", "GraphCheck", "TextEdge", "TextFrontend",
           "TextGraph"]
