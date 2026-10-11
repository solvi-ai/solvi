"""Found rules executed as plain code (solvi.experimental.relations.compile_program, solvi.experimental.laws
LawSet.to_module), each accepted only after a differential check against the interpreter:

    python benchmarks/relations_compile.py           # all three, as JSON
    pytest -m slow tests/test_relations_gate.py      # the same with the bars

1. forest(n=100_000) — a random forest (each node's parent drawn among the earlier nodes with probability 0.8, nodes
   permuted; E = parent → child). Three programs compiled to set code — closure (E)⁺, grandparent (E ∘ E), sibling
   ((Eᵀ ∘ E) ∖ I) — against hand-written set code over the parent array (walk up the ancestors; parent of parent; pairs
   of children of one parent): the pairs must be equal, pair for pair; the bar is the compiled code within 2× the
   hand-written code's time (best of three runs each; building the edge set the compiled code reads is timed apart).
2. differential(worlds=100, n=1000) — ten programs over every operator that compiles sparsely (composition, converse,
   closure, at least 2 / 3 paths, a count unary, union, intersection with a complement, difference, xor) on random
   worlds of three relations at 1,000 nodes (half Erdős–Rényi with p = 0.8/n, below the giant component so that
   closures stay sparse; half forests): both compiled backends against the dense interpreter; the bar is 0 differing
   pairs.
3. laws(rows=100_000) — a law set learned by the law loop on a grid with written mechanics (atoms that carry their
   code), compiled with LawSet.to_module and checked by check_compiled on 100,000 fresh rows (the bar: 0 differences);
   the time of the compiled module and of LawSet.predict on the same rows is reported. A second setup: a path policy
   with a relational atom found by RelAlgProposer ("a directory named secret on the path"), compiled through its
   set code, on 20,000 fresh rows.

Measured (2026-10, this machine, one run; see tests/test_relations_gate.py and the JSON the script prints): every
comparison equal (0 differing pairs, 0 differing rows); the times are in the output."""
from __future__ import annotations

import json
import random
import time
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Grammar, LawSet, Transition, check_compiled, load_module
    from solvi.experimental.relations import (PathTree, RelAlgProposer, Relations, compile_program, evaluate,
                                              from_keys, show, to_keys)


# ───────────────────────────────────────────── 1. a forest of 100,000 nodes
def parents_of(n, rng):
    """A parent array (−1 for a root): node i's parent is a random earlier node with probability 0.8; permuted."""
    par = -np.ones(n, dtype=np.int64)
    for i in range(1, n):
        if rng.random() < 0.8:
            par[i] = rng.integers(0, i)
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    out = -np.ones(n, dtype=np.int64)
    has = par >= 0
    out[inv[np.nonzero(has)[0]]] = inv[par[has]]
    return out


def hand_written(rule, par):
    """Set code written by hand over the parent array."""
    n = len(par)
    out = set()
    if rule == "closure":
        for c in range(n):
            p = par[c]
            while p >= 0:
                out.add((p, c))
                p = par[p]
    elif rule == "grand":
        for c in range(n):
            p = par[c]
            if p >= 0 and par[p] >= 0:
                out.add((par[p], c))
    elif rule == "sibling":
        children: dict[int, list[int]] = {}
        for c in range(n):
            if par[c] >= 0:
                children.setdefault(par[c], []).append(c)
        for kids in children.values():
            for a in kids:
                for b in kids:
                    if a != b:
                        out.add((a, b))
    return out


E = ("R", "E")
FOREST_PROGRAMS = {"closure": ("PLUS", E), "grand": ("COMP", E, E), "sibling": ("MINUS", ("COMP", ("T", E), E), ("I",))}


def _best(fn, repeat=3):
    best, out = None, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return out, best


def forest(n=100_000, seed=0, repeat=3):
    """The bar is on the arrays backend (index arrays; its output is a key array, converted to pairs for the
    comparison, the conversion timed apart); the sets backend (plain Python, for small per-state graphs) is reported."""
    rng = np.random.default_rng(seed)
    par_np = parents_of(n, rng)
    par = par_np.tolist()
    t0 = time.perf_counter()
    R = {"E": {(p, c) for c, p in enumerate(par) if p >= 0}}
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    has = par_np >= 0
    keys = par_np[has] * n + np.nonzero(has)[0]
    keys.sort()
    RA = {"E": keys}
    build_keys = time.perf_counter() - t0
    out = {"n": n, "edges": len(R["E"]), "edge_set_s": round(build, 4), "edge_keys_s": round(build_keys, 4)}
    for rule, ast in FOREST_PROGRAMS.items():
        want, t_hand = _best(lambda: hand_written(rule, par), repeat)
        fa = compile_program(ast, backend="arrays")
        got_keys, t_arr = _best(lambda: fa(RA, n), repeat)
        t0 = time.perf_counter()
        got = from_keys(got_keys, n)
        t_conv = time.perf_counter() - t0
        fs = compile_program(ast)
        got_sets, t_sets = _best(lambda: fs(R, n), repeat)
        out[rule] = {"program": show(ast), "pairs": len(want), "equal": got == want and got_sets == want,
                     "missing": len(want - got), "extra": len(got - want), "hand_s": round(t_hand, 4),
                     "compiled_s": round(t_arr, 4), "ratio": round(t_arr / t_hand, 3),
                     "to_pairs_s": round(t_conv, 4), "sets_backend_s": round(t_sets, 4),
                     "sets_ratio": round(t_sets / t_hand, 3)}
    return out


# ───────────────────────────────────────────── 2. compiled vs the interpreter on large worlds
E0, E1, E2 = ("R", "E0"), ("R", "E1"), ("R", "E2")
PROGRAMS = [("COMP", E0, ("T", E1)), ("PLUS", E0), ("PLUS", ("COMP", E0, ("T", E2))), ("C2", E0, ("T", E0)),
            ("C3", ("OR", E0, E1), ("T", E2)), ("COMP", ("CNT2", E1), E0), ("AND", ("PLUS", E1), ("NOT", E2)),
            ("MINUS", ("COMP", ("T", E2), E0), ("I",)), ("XOR", E0, ("T", E1)), ("MINUS", ("PLUS", E1), ("PLUS", E2))]


def _world(rng, n, family):
    r = Relations(("E0", "E1", "E2"), n)
    for nm in r.names:
        if family == "ER":
            m = rng.random((n, n)) < 0.8 / n            # below the giant component: closures stay sparse
            np.fill_diagonal(m, False)
        else:
            m = np.zeros((n, n), bool)
            par = parents_of(n, rng)
            has = par >= 0
            m[par[has], np.nonzero(has)[0]] = True
        r.set(nm, m)
    return r


def differential(worlds=100, n=1000, seed=0):
    """Every program on every world: the interpreter once, both compiled backends against it."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    fns = {(show(ast), b): compile_program(ast, backend=b) for ast in PROGRAMS for b in ("sets", "arrays")}
    per = {k: {"differing_pairs": 0, "pairs": 0} for k in fns}
    diffs = pairs = checks = 0
    for i in range(worlds):
        w = _world(rng, n, "ER" if i % 2 == 0 else "FOREST")
        sets = w.sets()
        keys = {k: to_keys(v, n) for k, v in sets.items()}
        for ast in PROGRAMS:
            xs, ys = np.nonzero(evaluate(ast, w))
            want = set(zip(xs.tolist(), ys.tolist()))
            pairs += len(want)
            for b in ("sets", "arrays"):
                got = fns[(show(ast), b)](sets, n) if b == "sets" else from_keys(fns[(show(ast), b)](keys, n), n)
                d = len(got ^ want)
                diffs += d
                checks += 1
                per[(show(ast), b)]["differing_pairs"] += d
                per[(show(ast), b)]["pairs"] += len(want)
    return {"worlds": worlds, "n": n, "programs": len(PROGRAMS), "checks": checks, "differing_pairs": diffs,
            "pairs": pairs, "per_program": {f"{a} [{b}]": v for (a, b), v in per.items()},
            "seconds": round(time.time() - t0, 1)}


# ───────────────────────────────────────────── 3. compiled law sets on 100,000 rows
MATS = ("grass", "sand", "tree", "stone", "water", "table")


def grid_grammar():
    atoms = [Atom(f"front={m}", lambda s, m=m: s["front"] == m, feature="front", code=f"s['front'] == {m!r}")
             for m in MATS]
    atoms += [Atom(f"wood>={k}", lambda s, k=k: s["wood"] >= k, feature="wood", code=f"s['wood'] >= {k}")
              for k in (1, 2, 3)]
    atoms += [Atom(f"stone>={k}", lambda s, k=k: s["stone"] >= k, feature="stone", code=f"s['stone'] >= {k}")
              for k in (1, 2)]
    atoms += [Atom("night", lambda s: s["night"], feature="night", code="s['night']"),
              Atom("day", lambda s: not s["night"], feature="night", code="not s['night']")]
    return Grammar(atoms, k=3)


def mechanics(s, a):
    """The written mechanics: what an action does."""
    f = s["front"]
    if a == "move":
        return "moved" if f in ("grass", "sand") else "blocked"
    if a == "do":
        if f == "tree":
            return "+wood"
        if f == "stone":
            return "+stone" if s["wood"] >= 1 else "nothing"
        return "nothing"
    if a == "place_table":
        return "table" if s["wood"] >= 2 and f in ("grass", "sand") else "nothing"
    if a == "make_pickaxe":
        return "pickaxe" if f == "table" and s["wood"] >= 1 and s["stone"] >= 1 else "nothing"
    return "nothing"


ACTIONS = ("move", "do", "place_table", "make_pickaxe", "noop")


def grid_rows(rng, count, group):
    out = []
    for i in range(count):
        s = {"front": rng.choice(MATS), "wood": rng.randrange(4), "stone": rng.randrange(3), "night": rng.random() < .3}
        a = rng.choice(ACTIONS)
        out.append(Transition(s, a, mechanics(s, a), group=group, id=f"{group}-{i}"))
    return out


def _time_predict(ls, mod, rows, apply=None):
    t0 = time.perf_counter()
    for t in rows:
        ls.predict(t.state, t.action)
    t_int = time.perf_counter() - t0
    t0 = time.perf_counter()
    for t in rows:
        mod.predict(t.state, t.action, base=ls.base, apply=apply)
    return t_int, time.perf_counter() - t0


def _learned_grid(seed):
    rng = random.Random(seed)
    g = grid_grammar()
    held = Buffer(g)
    for w in range(6):
        held.extend(grid_rows(rng, 300, f"h{w}"))
    ls = LawSet(g, base=lambda s, a: "nothing", sized=False)       # measured under the fixed thresholds
    for w in range(6):
        ls.learn(grid_rows(rng, 300, f"s{w}"), held)
    return ls, rng


def laws(rows=100_000, seed=0, path_rows=20_000):
    ls, rng = _learned_grid(seed)
    src = ls.to_module()
    mod = load_module(src)
    fresh = Buffer(ls.grammar)
    fresh.extend(grid_rows(rng, rows, "fresh"))
    chk = check_compiled(mod, fresh, ls)
    t_int, t_comp = _time_predict(ls, mod, fresh.rows())
    acc = sum(ls.predict(t.state, t.action) == t.observed for t in fresh.rows()) / len(fresh)
    out = {"rows": chk.rows, "differences": chk.differences, "laws": len(ls.laws), "accuracy": round(acc, 4),
           "interpreter_s": round(t_int, 3), "compiled_s": round(t_comp, 3), "module_lines": src.count("\n")}
    out["paths"] = path_policy(path_rows, seed)
    return out


# the path policy: a read is denied under any directory named "secret"; files may be named "secret" too
DIRS = ("a", "b", "c", "data", "src", "secret")


def tree_path(rng):
    parts = [rng.choice(DIRS) for _ in range(rng.randint(1, 5))]
    return "/" + "/".join(parts) + "/" + rng.choice(("f", "g", "secret")) + rng.choice((".txt", ".md", ""))


def policy(path):
    return "denied" if "secret" in path.split("/")[:-1] else "ok"


def path_grammar():
    atoms = [Atom(f"ext={e}", lambda s, e=e: s["path"].endswith("." + e), feature="ext", code=f"s['path'].endswith('.{e}')")
             for e in ("txt", "md")]
    atoms += [Atom(f"depth>={d}", lambda s, d=d: s["path"].count("/") >= d, feature="depth",
                   code=f"s['path'].count('/') >= {d}") for d in (2, 3, 4)]
    atoms += [Atom("parent=secret", lambda s: s["path"].split("/")[-2] == "secret", feature="parent",
                   code="s['path'].split('/')[-2] == 'secret'")]
    return Grammar(atoms, k=3)


def path_policy(rows=20_000, seed=0):
    rng = random.Random(seed)
    g = path_grammar()
    held = Buffer(g)
    for i in range(6):
        held.extend(Transition({"path": p}, "read", policy(p), group=f"h{i}") for p in (tree_path(rng) for _ in range(12)))
    ls = LawSet(g, base=lambda s, a: "ok", min_support=3, sized=False)   # measured under the fixed thresholds
    ls.proposer = RelAlgProposer(g, k=20, encoder=PathTree("path", names=("secret",)), rows=held, size=5)
    for i in range(6):
        for p in (tree_path(rng) for _ in range(12)):
            ls.step(Transition({"path": p}, "read", policy(p), group=f"s{i}"), held, remember=True)
    mod = load_module(ls.to_module())
    fresh = Buffer(g)
    fresh.extend(Transition({"path": p}, "read", policy(p), group="fresh") for p in (tree_path(rng) for _ in range(rows)))
    chk = check_compiled(mod, fresh, ls)
    t_int, t_comp = _time_predict(ls, mod, fresh.rows())
    acc = sum(mod.predict(t.state, t.action, base=ls.base) == t.observed for t in fresh.rows()) / len(fresh)
    return {"rows": chk.rows, "differences": chk.differences, "laws": [law.text() for law in ls.laws],
            "accuracy": round(acc, 4), "interpreter_s": round(t_int, 3), "compiled_s": round(t_comp, 3)}


def main():
    print(json.dumps({"forest": forest(), "differential": differential(), "laws": laws()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
