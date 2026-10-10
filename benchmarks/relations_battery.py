"""The relation-algebra enumerator (solvi.experimental.relations) on a pre-registered battery of random rules:

    python benchmarks/relations_battery.py --rules rules.json      # or SOLVI_RULE_BATTERY=rules.json
    pytest -m slow tests/test_relations_gate.py                     # the same with the bars

The battery: 28 random rules over three relations E0, E1, E2 (seven types, four each — a two-step path, a path and a
relation, a path and not a relation, a relation or a path, at least k paths, a transitive closure, a three-step path;
each optionally without the diagonal), drawn by a seeded generator that rejected degenerate densities, written to a
JSON file before any search ran (`{"rules": [{"id", "type", "p": {"atoms": [[relation, transposed], ...], "nodiag",
"k"}, "text"}, ...]}`; rules 2 and 3 are the same rule, so 27 are distinct). The file is not shipped; pass its path.

Examples (fixed seeds): 32 random directed graphs of 6–12 nodes (each relation an Erdős–Rényi graph with p = 1.5/n, no
self-loops, nodes permuted) from numpy's default_rng(15); from the same generator, 24 held-out graphs: 4 each of three
families (Erdős–Rényi, a preferential-attachment hub graph, a random forest) at 32 and at 64 nodes.

The search: synthesize(examples, target, size=7) with the operators T, NOT, PLUS (closure), OR, AND, MINUS, XOR, COMP,
C2, C3 and the terminals E0, E1, E2, I, U — the first program, in a fixed bottom-up order, whose relation equals the
target on all 32 examples, byte for byte; "generalised": the same program exact on all 24 held-out graphs.

Bars (fixed before this port ran, from the reference run of the same method): at least 22 of 28 found within 7 nodes,
every find generalised, peak memory of the process at most 3 GB. Second table: 8 hand-written rules that need union,
xor, the closure of a composite or a complement (all should be found). Third, the control (controls()): on dense
symmetric worlds, "x and y lie in a common 4-clique" needs four variables at once and cannot be said by this grammar —
the right outcome is not found, or found on the examples and failing the held-out check; a find that passes it would
be a false accept. Next to it two patterns of treewidth 2 (a diamond, a 4-cycle) that counting can express in
principle. Equivalence is on the examples only, so the held-out check on other sizes and families is what makes a find
a rule; with noisy labels use score / top_programs (agreement), never the exact target hash.

Measured (2026-10, this machine, one deterministic run; see tests/test_relations_gate.py): 22 of 28 found within 7
nodes (21 of 27 distinct), every find generalised; the six not found are the four "a path and not a relation without
the diagonal", one "a relation or a path without the diagonal" and one three-step path without the diagonal — each
needs 8 or 9 nodes. The 8 hand-written rules: 8 found, 8 generalised. Controls: the 4-clique not found (0 false
accepts), the diamond not found within 7 nodes, the 4-cycle found as ((E0 #2∘ E0) ∖ I) and generalised. Peak memory
0.4 GB; the time is printed by the run."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import Relations, evaluate, show, size, synthesize

K = 3
NAMES = ("E0", "E1", "E2")
SEARCH_OPS = ("T", "NOT", "PLUS", "OR", "AND", "MINUS", "XOR", "COMP", "C2", "C3")


def gen(rng, n, family="ER"):
    """K random relations over n nodes: Erdős–Rényi (p = 1.5/n), FOREST (a parent among the earlier nodes with
    probability 0.8) or HUB (preferential attachment, two links per node, random direction); no self-loops; nodes
    permuted. → bool (K, n, n)."""
    G = np.zeros((K, n, n), dtype=bool)
    for k in range(K):
        if family == "ER":
            A = rng.random((n, n)) < 1.5 / n
        elif family == "FOREST":
            A = np.zeros((n, n), bool)
            for i in range(1, n):
                if rng.random() < 0.8:
                    A[rng.integers(0, i), i] = True
        else:
            A = np.zeros((n, n), bool)
            deg = np.ones(n)
            for i in range(1, n):
                for t in rng.choice(i, size=min(i, 2), replace=False, p=deg[:i] / deg[:i].sum()):
                    if rng.random() < 0.5:
                        A[i, t] = True
                    else:
                        A[t, i] = True
                    deg[t] += 1
                    deg[i] += 1
        np.fill_diagonal(A, False)
        perm = rng.permutation(n)
        G[k] = A[perm][:, perm]
    return G


def _closure(A):
    R = A.copy()
    while True:
        nxt = R | (R.astype(np.int64) @ R.astype(np.int64) > 0)
        if np.array_equal(nxt, R):
            return R
        R = nxt


def apply_rule(rule, G):
    """The rule's relation on one graph G (K, n, n) → bool (n, n)."""
    t, p = rule["type"], rule["p"]

    def A(i):
        rel, tr = p["atoms"][i]
        X = G[rel]
        return (X.T if tr else X).astype(np.int64)

    if t == "PATH2":
        Y = A(0) @ A(1) > 0
    elif t == "AND":
        Y = (A(0) @ A(1) > 0) & (A(2) > 0)
    elif t == "ANDNOT":
        Y = (A(0) @ A(1) > 0) & (A(2) == 0)
    elif t == "OR":
        Y = (A(0) + A(1) @ A(2)) > 0
    elif t == "COUNT":
        Y = A(0) @ A(1) >= p["k"]
    elif t == "CLOSURE":
        base = A(0) > 0 if len(p["atoms"]) == 1 else (A(0) + A(1)) > 0
        Y = _closure(base)
    elif t == "HOP3":
        Y = A(0) @ A(1) @ A(2) > 0
    else:
        raise ValueError(f"unknown rule type {t!r}")
    if p["nodiag"]:
        Y = Y & ~np.eye(Y.shape[0], dtype=bool)
    return Y


def hand_rules():
    """Rules that need union, xor, the closure of a composite or a complement."""
    E0, E1, E2 = ("R", "E0"), ("R", "E1"), ("R", "E2")
    return {
        "closure(E0 ∪ E1ᵀ)": ("PLUS", ("OR", E0, ("T", E1))),
        "E0 ⊕ E1": ("XOR", E0, E1),
        "(E0 ∘ E1) ∖ E2⁺": ("MINUS", ("COMP", E0, E1), ("PLUS", E2)),
        "not adjacent: ¬(E0 ∪ E0ᵀ) ∖ I": ("MINUS", ("NOT", ("OR", E0, ("T", E0))), ("I",)),
        "closure(E0 ∘ E1)": ("PLUS", ("COMP", E0, E1)),
        "(E0 ∪ E1) ∘ E2": ("COMP", ("OR", E0, E1), E2),
        "E0⁺ ∩ E1⁺": ("AND", ("PLUS", E0), ("PLUS", E1)),
        "#2(E0 ∘ (E1 ∪ E2))": ("C2", E0, ("OR", E1, E2)),
    }


def gen_dense(rng, n):
    """A dense symmetric E0 (p = 0.45 up to 12 nodes, 0.2 at 32, 0.12 at 64) over random E1, E2."""
    G = gen(rng, n)
    p = 0.45 if n <= 12 else (0.2 if n <= 32 else 0.12)
    A = np.triu(rng.random((n, n)) < p, 1)
    G[0] = A | A.T
    return G


PATTERNS = ("4-clique", "diamond", "4-cycle")


def pattern_truth(name, G):
    """Over the undirected E0: x, y in a common 4-clique (needs four variables: not expressible here); a diamond (two
    common neighbours joined to each other) and a 4-cycle (at least two common neighbours): treewidth 2, expressible
    with counting in principle."""
    A = G[0].astype(bool)
    n = A.shape[-1]
    Y = np.zeros((n, n), dtype=bool)
    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            S = np.where(A[x] & A[y])[0]
            if name == "4-clique":
                Y[x, y] = A[x, y] and len(S) >= 2 and A[np.ix_(S, S)].any()
            elif name == "diamond":
                Y[x, y] = len(S) >= 2 and A[np.ix_(S, S)].any()
            else:
                Y[x, y] = len(S) >= 2
    return Y


def controls(max_size=7):
    """The inexpressible-rule control and two patterns of treewidth 2, on dense worlds from default_rng(1515): 32
    examples of 6–12 nodes; held out 6 graphs of 32 and 4 of 64 nodes. A 4-clique program that passes the held-out
    check would be a false accept (the right outcome: not found, or found and failing the held-out check)."""
    rng = np.random.default_rng(1515)
    ex_g = [gen_dense(rng, int(rng.integers(6, 13))) for _ in range(32)]
    held_g = [gen_dense(rng, 32) for _ in range(6)] + [gen_dense(rng, 64) for _ in range(4)]
    ex = [Relations.from_dense(NAMES, g) for g in ex_g]
    held = [Relations.from_dense(NAMES, g) for g in held_g]
    out = []
    for name in PATTERNS:
        target = [pattern_truth(name, g) for g in ex_g]
        t0 = time.time()
        f = synthesize(ex, target, max_size, ops=SEARCH_OPS)
        gen_ok = None if f.ast is None else all(np.array_equal(evaluate(f.ast, r), pattern_truth(name, g))
                                                for r, g in zip(held, held_g))
        out.append({"task": name, "found": f.ast is not None, "program": show(f.ast) if f.ast else None,
                    "generalized": gen_ok, "evals": f.evals, "behaviours": f.behaviours,
                    "seconds": round(time.time() - t0, 2)})
    false_accepts = sum(1 for r in out if r["task"] == "4-clique" and r["generalized"])
    return {"rules": out, "false_accepts": false_accepts}


def worlds():
    """The examples and the held-out graphs, from the fixed seed."""
    rng = np.random.default_rng(15)
    ex = [gen(rng, int(rng.integers(6, 13))) for _ in range(32)]
    held = [gen(rng, n, fam) for n in (32, 64) for fam in ("ER", "HUB", "FOREST") for _ in range(4)]
    as_rel = lambda gs: [Relations.from_dense(NAMES, g) for g in gs]   # noqa: E731
    return ex, held, as_rel(ex), as_rel(held)


def _peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def _run(tasks, max_size, **kw):
    ex_g, held_g, ex, held = worlds()
    out = []
    for name, truth in tasks:
        target = [truth(g) for g in ex_g]
        t0 = time.time()
        f = synthesize(ex, target, max_size, ops=SEARCH_OPS, **kw)
        dt = time.time() - t0
        gen_ok = None
        if f.ast is not None:
            gen_ok = all(np.array_equal(evaluate(f.ast, r), truth(g)) for r, g in zip(held, held_g))
        out.append({"task": name, "found": f.ast is not None, "program": show(f.ast) if f.ast else None,
                    "size": size(f.ast) if f.ast else None, "generalized": gen_ok, "evals": f.evals,
                    "behaviours": f.behaviours, "stopped": f.stopped, "seconds": round(dt, 2)})
        print(f"[{dt:6.1f}s] {name:48s} {'found ' + out[-1]['program'] if f.ast else 'not found'}"
              f"{'' if gen_ok is None else (' generalised' if gen_ok else ' NOT generalised')} "
              f"(evals {f.evals}, behaviours {f.behaviours})", file=sys.stderr, flush=True)
    return out


def battery(rules, max_size=7, **kw):
    """Search every rule of the battery → {"rules": [{id, type, text, found, program, size, generalized, evals,
    behaviours, seconds}], "found", "generalized", "distinct_found", "peak_rss_gb", "seconds"}."""
    t0 = time.time()
    recs = _run([(f"#{r['id']} {r['type']} {r['text']}", lambda g, r=r: apply_rule(r, g)) for r in rules], max_size, **kw)
    for r, rec in zip(rules, recs):
        rec.update({"id": r["id"], "type": r["type"], "text": r["text"]})
    distinct, seen = 0, set()
    for r, rec in zip(rules, recs):
        key = json.dumps(r["p"], sort_keys=True) + r["type"]
        if key not in seen:
            seen.add(key)
            distinct += rec["found"]
    return {"rules": recs, "found": sum(x["found"] for x in recs), "generalized": sum(bool(x["generalized"]) for x in recs),
            "total": len(recs), "distinct_found": distinct, "distinct_total": len(seen), "max_size": max_size,
            "peak_rss_gb": round(_peak_gb(), 3), "seconds": round(time.time() - t0, 1)}


def hand(max_size=7):
    """The hand-written rules → {"rules": [...], "found", "generalized"}."""
    from solvi.experimental.relations import Relations as _R  # noqa: F401  (same import as above)

    def truth_of(ast):
        def t(g):
            return evaluate(ast, Relations.from_dense(NAMES, g))
        return t
    recs = _run([(nm, truth_of(ast)) for nm, ast in hand_rules().items()], max_size)
    return {"rules": recs, "found": sum(x["found"] for x in recs),
            "generalized": sum(bool(x["generalized"]) for x in recs), "total": len(recs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rules", default=os.environ.get("SOLVI_RULE_BATTERY"), help="the battery's JSON file")
    ap.add_argument("--max-size", type=int, default=7)
    a = ap.parse_args(argv)
    if not a.rules:
        ap.error("pass --rules (or set SOLVI_RULE_BATTERY): the battery file is not shipped")
    with open(a.rules) as f:
        rules = json.load(f)["rules"]
    out = {"battery": battery(rules, a.max_size), "hand": hand(a.max_size), "controls": controls(a.max_size)}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
