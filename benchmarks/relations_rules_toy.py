"""Guarded rules learned by counting on a planted world, against an independent oracle:

    python benchmarks/relations_rules_toy.py [--fit 8] [--check 4] [--test 10] [--store 3]

The world (seeded, generated here): three places; families of three generations (a couple, two or three children, one
to three grandchildren of the first child), genders, each person's home drawn on its own; colleagues and
«same place» edges between people of one home; half of the people with their residence given. The oracle defines
every relation directly (not by rules): child_of, father_of / mother_of (a parent's gender), grandparent_of /
grandchild_of, grandfather_of / grandmother_of, paternal_grandparent_of (the middle parent male), sibling_of,
living_in_same_place (colleagues and same-place edges, both ways, transitively), living_in (given, or through
living_in_same_place), Not_living_in (every other place, and every place when the residence is unknown — the
closed-world default). Every ordered pair of every story is a labelled query (its truth may be empty).

Rules: GuardedRules(majority=0.9, support=2) on the fit stories, the check stories as `check=`; scored on fresh
stories: exact set match over every ordered pair (refusal = wrong), precision among answered, coverage; the planted
rules found (guards, a chain through a derived relation, the BNE «elsewhere» rule, the negative stratum); the same
learner without residual votes and with one round; the store path (`to_knowledge` + `chain_rules` in a fresh
KnowledgeStore per story) against the in-memory fixed point on --store stories. One deterministic run.

Measured (2026-10, this machine, CPU, one deterministic run, defaults: 8 fit stories = 4,106 labelled pairs, 4 check
stories = 2,118, 10 test stories = 4,992 pairs of which 43.1 % hold a relation): `rules` — 242 rules (B1 19, B1u 10,
B2 201, B2z 8, BNE 1, NEG 3) in 4 rounds, 3.0 s; exact set match 1.000 on the 4,992 test pairs (precision among the
answered 1.000, coverage 0.431 = every pair that holds a relation; the rest correctly empty); all 8 planted rules
found. Without residual votes: the same 242 rules (the rounds reach them; 1.4 s). One round without residual votes
(bodies over the story's own facts only): 35 rules, exact 0.975, precision 0.941, 7 of 8 planted rules (the guarded
rule over a derived relation needs a second round). The store path on 3 more stories: 1,568 pairs, 2,960 facts
written, 0 differences from the in-memory fixed point, 1.1 s."""
from __future__ import annotations

import argparse
import random
import sys
import time
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import GuardedRules, Story, chain_rules, pair_fact, unary_fact

from solvi.core.knowledge import KnowledgeStore

COMPLEMENTS = {"Not_living_in": "living_in"}
PLANTED = {
    "child_of ← parent_of⁻": ("child_of", "B1", (("parent_of", 1),), ()),
    "father_of ← parent_of ∧ is_male(x)": ("father_of", "B1u", (("parent_of", 0),), (("is_male", "x"),)),
    "grandparent_of ← parent_of ∘ parent_of": ("grandparent_of", "B2", (("parent_of", 0), ("parent_of", 0)), ()),
    "grandfather_of ← grandparent_of ∧ is_male(x)": ("grandfather_of", "B1u", (("grandparent_of", 0),),
                                                      (("is_male", "x"),)),
    "paternal_grandparent_of ← parent_of ∘ is_male(z) ∘ parent_of": (
        "paternal_grandparent_of", "B2z", (("parent_of", 0), ("parent_of", 0)), (("is_male", "z"),)),
    "living_in ← living_in_same_place⁻ ∘ living_in": ("living_in", "B2", (("living_in_same_place", 1),
                                                                         ("living_in", 0)), ()),
    "Not_living_in ← living_in(x,z) ∧ z≠y ∧ is_place(y)": ("Not_living_in", "BNE", (("living_in", 0),),
                                                            (("is_place", "y"),)),
    "Not_living_in ← is_person(x) ∧ is_place(y) ∧ ¬living_in(x,y)": (
        "Not_living_in", "NEG", (("living_in", 0),), (("is_person", "x"), ("is_place", "y"))),
}


def world(seed, families=3, places=3, known=0.5):
    rng = random.Random(seed)
    nid = iter(range(10_000))
    unary, edges, male, persons, place, parents = [], [], {}, [], [], {}
    for _ in range(places):
        p = next(nid)
        place.append(p)
        unary.append((p, "is_place"))
    for _ in range(families):
        g = [next(nid) for _ in range(2)]
        kids = [next(nid) for _ in range(rng.randint(2, 3))]
        grand = [next(nid) for _ in range(rng.randint(1, 3))]
        male[g[0]], male[g[1]] = True, False
        for k in kids + grand:
            male[k] = rng.random() < 0.5
        for k in kids:
            parents[k] = list(g)
        for c in grand:
            parents[c] = [kids[0]]
        persons += g + kids + grand
    home = {x: rng.choice(place) for x in persons}
    for x in persons:
        unary += [(x, "is_person"), (x, "is_male" if male[x] else "is_female")]
    for c, ps in sorted(parents.items()):
        for p in ps:
            edges.append((p, c, "parent_of"))
    for x in persons:
        for y in persons:
            if x < y and home[x] == home[y]:
                d = rng.random()
                if d < 0.12:
                    edges.append((x, y, "colleague_of"))
                elif d < 0.24:
                    edges.append((x, y, "living_in_same_place"))
    for x in persons:
        if rng.random() < known:
            edges.append((x, home[x], "living_in"))
    rng.shuffle(edges)
    return Story.from_facts(edges, unary), oracle(edges, unary)


def oracle(edges, unary):
    U = {}
    for x, u in unary:
        U.setdefault(x, set()).add(u)
    parent = {(a, b) for a, b, r in edges if r == "parent_of"}
    coll = {(a, b) for a, b, r in edges if r == "colleague_of"}
    lsp = {(a, b) for a, b, r in edges if r == "living_in_same_place"} | coll
    lsp |= {(b, a) for a, b in lsp}
    while True:
        more = {(a, c) for a, b in lsp for b2, c in lsp if b == b2 and a != c} - lsp
        if not more:
            break
        lsp |= more
    lives = {(a, b) for a, b, r in edges if r == "living_in"}
    while True:
        more = {(a, p) for a, b in lsp for b2, p in lives if b == b2} - lives
        if not more:
            break
        lives |= more
    T = {}

    def put(a, b, r):
        if a != b:
            T.setdefault((a, b), set()).add(r)
    for a, b, r in edges:
        put(a, b, r)
    for a, b in parent:
        put(b, a, "child_of")
        put(a, b, "father_of" if "is_male" in U[a] else "mother_of")
        for b2, c in parent:
            if b == b2:
                put(a, c, "grandparent_of")
                put(c, a, "grandchild_of")
                put(a, c, "grandfather_of" if "is_male" in U[a] else "grandmother_of")
                if "is_male" in U[b]:
                    put(a, c, "paternal_grandparent_of")
        for a2, c in parent:
            if a == a2 and b != c:
                put(b, c, "sibling_of")
    for a, b in coll:
        put(b, a, "colleague_of")
    for a, b in lsp:
        put(a, b, "living_in_same_place")
    for a, p in lives:
        put(a, p, "living_in")
    people = sorted(x for x in U if "is_person" in U[x])
    places = sorted(x for x in U if "is_place" in U[x])
    for x in people:
        mine = {p for a, p in lives if a == x}
        for p in places:
            if p not in mine:
                put(x, p, "Not_living_in")
    return T


def examples(seeds):
    out = []
    for s in seeds:
        st, T = world(s)
        ents = sorted(st.entities())
        out += [(st, (x, y), frozenset(T.get((x, y), ()))) for x in ents for y in ents if x != y]
    return out


def score(rules, test):
    n = len(test)
    right = answered = right_answered = 0
    c, cur = None, None
    for st, q, y in test:
        if st is not cur:
            c, cur = rules.closure(st), st
        p = c.answer(*q)
        right += p == y
        if p:
            answered += 1
            right_answered += p == y
    return {"exact": right / n, "precision": right_answered / answered if answered else None,
            "coverage": answered / n, "n": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=8)
    ap.add_argument("--check", type=int, default=4)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--store", type=int, default=3)
    a = ap.parse_args(argv)
    fit = examples(range(a.fit))
    check = examples(range(100, 100 + a.check))
    test = examples(range(500, 500 + a.test))
    print(f"labelled pairs: fit {len(fit)} ({a.fit} stories), check {len(check)}, test {len(test)} ({a.test} stories)")
    arms = {"rules": dict(), "no residual votes": dict(residual=False),
            "one round, no residual votes": dict(residual=False, rounds=1)}
    print("| arm | rules | B1 | B1u | B2 | B2z | BNE | NEG | rounds | fit s | exact | precision | coverage | planted |")
    print("|" + "---|" * 14)
    main_rules = None
    for name, kw in arms.items():
        t0 = time.perf_counter()
        g = GuardedRules(0.9, 2, complements=COMPLEMENTS, **kw).fit(fit, check=check)
        fs = time.perf_counter() - t0
        sc = score(g, test)
        have = {(r.head, r.template, r.atoms, r.unary) for r in g}
        found = sum(v in have for v in PLANTED.values())
        t = g.by_template()
        print(f"| {name} | {len(g)} | " + " | ".join(str(t[k]) for k in ("B1", "B1u", "B2", "B2z", "BNE", "NEG")) +
              f" | {g.rounds} | {fs:.1f} | {sc['exact']:.3f} | {sc['precision']:.3f} | {sc['coverage']:.3f} | "
              f"{found}/{len(PLANTED)} |")
        if main_rules is None:
            main_rules = (g, have)
    g, have = main_rules
    print("\nplanted rules (rules arm):")
    for k, v in PLANTED.items():
        print(f"  {'found  ' if v in have else 'MISSING'} {k}")
    diffs, pairs, t0, facts = 0, 0, time.perf_counter(), 0
    for s in range(900, 900 + a.store):
        st, T = world(s)
        ks = KnowledgeStore()
        g.to_knowledge(ks, {"rules": 1})
        for x, y, r in st.facts:
            ks.add("fact", pair_fact(x, y, r, multi=True, family=COMPLEMENTS.get(r, r)), {"s": s}, source="outcome")
        for x, u in st.unary:
            ks.add("fact", unary_fact(x, u), {"s": s}, source="outcome")
        rep = chain_rules(ks, scope={"s": s}, rule_scope={"rules": 1}, complements=COMPLEMENTS)
        facts += rep.facts_added
        c = g.closure(st)
        ents = sorted(st.entities())
        for x in ents:
            for y in ents:
                if x != y:
                    pairs += 1
                    got = frozenset(r["body"]["o"] for r in ks.find(kind="fact", s=[x, y], scope={"s": s},
                                                                    status="active"))
                    diffs += got != c.answer(x, y)
    print(f"\nstore path: {a.store} stories, {pairs} pairs, {facts} facts written, {diffs} differences from the "
          f"in-memory fixed point, {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

