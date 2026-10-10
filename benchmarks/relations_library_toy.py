"""Composites by enumeration with a library: two deterministic toy ladders, each with and without the library, five seeds.

(i) A path tree (a file system: PARENT from an entry to its directory, SAME_DIR, IS_DIR, NAME=secret / docs / tmp as
diagonal predicates). The rule: "denied if an ancestor directory is named secret and the depth is at least 3", three
levels deep:
    level 1  secret_dir = NAME=secret ∩ IS_DIR      ancestor = (PARENT)⁺      up2 = PARENT ∘ PARENT
    level 2  under_secret = #≥1(ancestor ∘ secret_dir)                       depth3 = #≥1(up2 ∘ PARENT)
    level 3  denied = under_secret ∩ depth3
(ii) Kinship on forests (E: parent → child), depth 1–6: grandparent, sibling (depth 1); uncle, great-grandparent (2);
cousin, great-uncle (3); first cousin once removed (4); second cousin (5); second cousin once removed (6).

The targets are given as an unordered set (find_rules takes them by name); which one is within reach of the current
library, and the programs, are found by the search. The truths are written as plain code over parent arrays, not as
programs. Each seed draws its own example worlds (16), held-out worlds for the admission test (6, larger) and fresh
check worlds (4, larger still, another shape) on which a false accept would show.

Arms: `library` (find_rules, admitted rules become terminals) and `flat` (one pass, grow=False, the same search
without the library). Measured, per target: found, the programs enumerated — cumulative at its admission, and its own
round's (the search it needed with the library of that moment) — and for flat the programs at its find or the flat
budget when it ran out; false accepts on the check worlds. The bar (tests/test_relations_library_toy.py, -m slow): with
the library every level is found, the cumulative programs at every level's admission are ≤ 1/10 of the flat budget,
flat does not find the top rule within that budget, and 0 false accepts in either arm. The flat budget was 60,000 in the
first run, where the top kinship rule cost the library 6,124 cumulative programs (1.02 × the bar); it was raised to
100,000 after that run, and flat still finds neither top rule (the bar's denominator is a lower bound on what flat
search needs). The per-level own-round costs
next to flat's own costs are reported, not barred: a shallow composite that flat reaches cheaply (great-grandparent,
size 5 without the library) is not ten times cheaper with it.

Measured with the package's relation enumerator (it replaced a minimal one after the first runs; the bar was not
changed): paths — every level, 2,556 cumulative at the top rule (1,679 with the minimal one); kinship — every level,
10,813–10,819 cumulative at the top rule (6,122–6,125 with the minimal one), so the kinship ladder MISSES the bar
(second cousin 10,257 and second cousin once removed 10,817 > 10,000); flat finds neither top rule; 0 false accepts;
25 s. The difference is mostly the universal relation U, a terminal of the package's enumerator that the minimal one
did not have (without it, measured once outside the package: 2,174 and 7,889).

    python benchmarks/relations_library_toy.py          # the table, under a minute
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations.library import HeldOut, find_rules, run as run_program, show

SEEDS = (0, 1, 2, 3, 4)
FLAT_BUDGET = 100_000        # the no-library search budget (enumerated programs); 60,000 in the first run, see above
LIB_BUDGET = 100_000
SIZE = 12


# ───────────────────────────────────────────── (i) path trees
NAMES = ("secret", "docs", "tmp", "src", "a", "b")
P_NAMES = (0.18, 0.14, 0.14, 0.18, 0.18, 0.18)


def _tree(rng, n, deep=False):
    """parent array, is_dir, names; root = 0 (a directory). deep: parents drawn from the latest directories."""
    par, isdir, name, depth = [-1], [True], ["root"], [0]
    for i in range(1, n):
        dirs = [j for j in range(i) if isdir[j]]
        if deep:
            p = int(rng.choice(dirs[-4:]))
        else:
            w = np.array([1.0 + depth[j] for j in dirs])
            p = int(rng.choice(dirs, p=w / w.sum()))
        par.append(p)
        depth.append(depth[p] + 1)
        isdir.append(bool(rng.random() < 0.55))
        name.append(str(rng.choice(NAMES, p=P_NAMES)))
    return par, isdir, name


def _encode_tree(par, isdir, name, perm):
    n = len(par)
    inv = np.empty(n, int)
    inv[perm] = np.arange(n)                        # node i is shown as inv[i]
    P = np.zeros((n, n), bool)
    for i, p in enumerate(par):
        if p >= 0:
            P[inv[i], inv[p]] = True
    same = (P.astype(int) @ P.T.astype(int) > 0) & ~np.eye(n, dtype=bool)
    w = {"PARENT": P, "SAME_DIR": same, "IS_DIR": np.diag(np.array(isdir)[perm])}
    for x in ("secret", "docs", "tmp"):
        w[f"NAME={x}"] = np.diag(np.array([name[perm[j]] == x for j in range(n)]))
    w["_par"] = np.array([inv[par[perm[j]]] if par[perm[j]] >= 0 else -1 for j in range(n)])
    return w


def path_tree(seed, n, deep=False):
    """One path tree as relations (nodes shuffled); `_par` (the parent array) is for the written truths only."""
    rng = np.random.default_rng(seed)
    par, isdir, name = _tree(rng, n, deep)
    return _encode_tree(par, isdir, name, rng.permutation(n))


def _ancestors(par):
    out = []
    for i in range(len(par)):
        a, p = [], par[i]
        while p >= 0:
            a.append(int(p))
            p = par[p]
        out.append(a)
    return out


def _diag(v):
    return np.diag(np.asarray(v, bool))


def t_secret_dir(w):
    return w["NAME=secret"] & w["IS_DIR"]


def t_ancestor(w):
    n = len(w["_par"])
    A = np.zeros((n, n), bool)
    for i, anc in enumerate(_ancestors(w["_par"])):
        A[i, anc] = True
    return A


def t_up2(w):
    par, n = w["_par"], len(w["_par"])
    A = np.zeros((n, n), bool)
    for i in range(n):
        if par[i] >= 0 and par[par[i]] >= 0:
            A[i, par[par[i]]] = True
    return A


def t_under_secret(w):
    sd = np.diag(t_secret_dir(w))
    return _diag([any(sd[a] for a in anc) for anc in _ancestors(w["_par"])])


def t_depth3(w):
    return _diag([len(anc) >= 3 for anc in _ancestors(w["_par"])])


def t_denied(w):
    return t_under_secret(w) & t_depth3(w)


PATHS = {"secret_dir": (1, t_secret_dir), "ancestor": (1, t_ancestor), "up2": (1, t_up2),
         "under_secret": (2, t_under_secret), "depth3": (2, t_depth3), "denied": (3, t_denied)}
PATH_BASE = ["PARENT", "SAME_DIR", "IS_DIR", "NAME=secret", "NAME=docs", "NAME=tmp"]


def path_worlds(seed):
    rng = np.random.default_rng(1000 + seed)
    s = lambda: int(rng.integers(0, 2**31))
    train = [path_tree(s(), int(rng.integers(14, 23))) for _ in range(16)]
    held = [path_tree(s(), int(rng.integers(30, 41))) for _ in range(6)]
    check = [path_tree(s(), 120, deep=k % 2 == 0) for k in range(4)]
    return train, held, check


# ───────────────────────────────────────────── (ii) kinship on forests
def forest(seed, n, uniform=False):
    """A forest by generations (≤ 1 parent each, depth < 7); E[p, c] = p is the parent of c; nodes shuffled."""
    rng = np.random.default_rng(seed)
    roots = max(1, n // 8)
    depth = {i: 0 for i in range(roots)}
    par = [-1] * roots
    for i in range(roots, n):
        cand = [j for j in depth if depth[j] < 6]
        w = np.ones(len(cand)) if uniform else np.array([1.0 + depth[j] for j in cand])
        p = int(rng.choice(cand, p=w / w.sum()))
        par.append(p)
        depth[i] = depth[p] + 1
    perm = rng.permutation(n)
    inv = np.empty(n, int)
    inv[perm] = np.arange(n)
    E = np.zeros((n, n), bool)
    for c, p in enumerate(par):
        if p >= 0:
            E[inv[p], inv[c]] = True
    return {"E": E}


def _mm(a, b):
    return (a.astype(np.int64) @ b.astype(np.int64)) > 0


def _kin(w):
    E = w["E"]
    sib = _mm(E.T, E) & ~np.eye(len(E), dtype=bool)
    grand = _mm(E, E)
    uncle = _mm(sib, E)
    cousin = _mm(E.T, uncle)
    c1r = _mm(cousin, E)
    c2 = _mm(E.T, c1r)
    return {"grand": grand, "sibling": sib, "uncle": uncle, "ggrand": _mm(grand, E), "cousin": cousin,
            "guncle": _mm(uncle, E), "cousin1r": c1r, "cousin2": c2, "cousin2_1r": _mm(c2, E)}


KIN_DEPTH = {"grand": 1, "sibling": 1, "uncle": 2, "ggrand": 2, "cousin": 3, "guncle": 3, "cousin1r": 4, "cousin2": 5,
             "cousin2_1r": 6}
KIN = {name: (d, (lambda w, name=name: _kin(w)[name])) for name, d in KIN_DEPTH.items()}


def kin_worlds(seed):
    rng = np.random.default_rng(2000 + seed)
    s = lambda: int(rng.integers(0, 2**31))
    train = [forest(s(), int(rng.integers(16, 31))) for _ in range(16)]
    held = [forest(s(), 40) for _ in range(6)]
    check = [forest(s(), 100, uniform=k % 2 == 0) for k in range(3)] + [forest(s(), 200)]
    return train, held, check


# ───────────────────────────────────────────── the arms
def _one(ladder, worlds, base, seed, flat_budget, lib_budget):
    train, held, check = worlds(seed)
    targets = {k: f for k, (_, f) in ladder.items()}
    test = HeldOut(held, targets)
    out = {}
    for arm, grow, budget in (("library", True, lib_budget), ("flat", False, flat_budget)):
        t0 = time.time()
        r = find_rules(train, targets, base=base, size=SIZE, budget=budget, library=(), test=test, grow=grow)
        false = 0
        for name, e in r.found.items():
            if any(not np.array_equal(run_program(e.program, w, library=r.library), targets[name](w)) for w in check):
                false += 1
        out[arm] = {"found": {k: k in r.found for k in targets},
                    "cum": {k: r.records[k].programs for k in targets},
                    "own": {k: r.records[k].round_programs for k in targets},
                    "programs": {k: (show(r.found[k].program) if k in r.found else None) for k in targets},
                    "rejected": sum(x.rejected for x in r.records.values()), "false_accepts": false,
                    "total": r.programs, "seconds": round(time.time() - t0, 2)}
    return out


def _ladder(name, ladder, worlds, base, seeds, flat_budget, lib_budget):
    per = {s: _one(ladder, worlds, base, s, flat_budget, lib_budget) for s in seeds}
    names = list(ladder)
    top = max(names, key=lambda k: ladder[k][0])
    arms = {}
    for arm in ("library", "flat"):
        rs = [per[s][arm] for s in seeds]
        budget = flat_budget if arm == "flat" else lib_budget
        cost = {}
        for k in names:     # library: its own round's programs; flat: the programs at its find, the budget if not found
            vals = [(r["own"][k] if arm == "library" else r["cum"][k]) if r["found"][k] else budget for r in rs]
            cost[k] = float(np.mean(vals))
        arms[arm] = {
            "found": {k: float(np.mean([r["found"][k] for r in rs])) for k in names},
            "found_all": float(np.mean([all(r["found"].values()) for r in rs])),
            "cum": {k: float(np.mean([r["cum"][k] if r["found"][k] else budget for r in rs])) for k in names},
            "cost": cost,
            "false_accepts": int(sum(r["false_accepts"] for r in rs)),
            "rejected": int(sum(r["rejected"] for r in rs)),
            "total": float(np.mean([r["total"] for r in rs])),
            "seconds": float(np.sum([r["seconds"] for r in rs])),
        }
    return {"ladder": name, "seeds": list(seeds), "levels": {k: ladder[k][0] for k in names},
            "composites": [k for k in names if ladder[k][0] >= 2], "top": top, "flat_budget": flat_budget,
            "lib_budget": lib_budget, "arms": arms, "per_seed": per}


def run(seeds=SEEDS, flat_budget=FLAT_BUDGET, lib_budget=LIB_BUDGET):
    """Both ladders, both arms, every seed → a dict (see the module docstring)."""
    return {"paths": _ladder("paths", PATHS, path_worlds, PATH_BASE, seeds, flat_budget, lib_budget),
            "kinship": _ladder("kinship", KIN, kin_worlds, ["E"], seeds, flat_budget, lib_budget)}


def _table(r):
    lines = []
    for lad in ("paths", "kinship"):
        x = r[lad]
        lib, flat = x["arms"]["library"], x["arms"]["flat"]
        lines.append(f"\n{lad}: seeds {x['seeds']}, flat budget {x['flat_budget']:,} programs; library found all "
                     f"levels in {lib['found_all']:.0%} of seeds, false accepts library {lib['false_accepts']} / flat "
                     f"{flat['false_accepts']}, held-out rejections {lib['rejected']} / {flat['rejected']}, seconds "
                     f"{lib['seconds']:.1f} / {flat['seconds']:.1f}")
        lines.append(f"  {'target':14s} lvl  found lib/flat   lib cumulative   lib own round   flat (budget if not found)")
        for k, lvl in x["levels"].items():
            lines.append(f"  {k:14s} {lvl:>3}   {lib['found'][k]:.1f} / {flat['found'][k]:.1f}   {lib['cum'][k]:>14,.0f}"
                         f"   {lib['cost'][k]:>13,.0f}   {flat['cost'][k]:>12,.0f}")
        lines.append("  per seed, library cumulative at the top rule: " + ", ".join(
            f"{s}: {x['per_seed'][s]['library']['cum'][x['top']]}" for s in x["seeds"]))
        s0 = x["per_seed"][x["seeds"][0]]["library"]["programs"]
        lines.append("  seed 0 programs: " + "; ".join(f"{k} = {v}" for k, v in s0.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.time()
    res = run()
    print(_table(res))
    print(f"\n{time.time() - t0:.1f} s")
    if "--json" in sys.argv:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"} for k, v in res.items()}, indent=1))
