"""Residual synthesis against flat enumeration, without a library: the kinship ladder and a composite path-tree policy.

(i) The kinship ladder of benchmarks/relations_library_toy.py — the same worlds (`kin_worlds(seed)`: 16 example forests,
6 held-out, 4 check worlds of another shape), the same nine targets written as plain code over the parent matrix,
seeds 0–4. Two arms, neither with a library:
    residual  synthesize_residual(examples, target, depth=9, budget=500,000 steps), the held-out worlds as its `test`
              (a HeldOut: a candidate that fails is rejected and the search goes on);
    flat      synthesize(examples, target, size=12, include_universal=False, max_behaviours=100,000, seconds=60) — the
              enumerator's best setting on this ladder; the first exact match on the examples is its answer (as the
              earlier measurement ran it: no held-out re-check inside the search).
Both arms' answers are judged on the check worlds (larger, another shape) by the package's evaluator: a find that is
not exact there is a false accept. Units differ — residual steps are atoms tried at a subgoal, flat counts distinct
behaviours — so the comparison is found / seconds / false accepts.

(ii) A composite file-server policy on path trees: «denied if an ancestor directory is named secret and the depth is at
least 3», five levels — anc = (PARENT)⁺, grand = PARENT ∘ PARENT, under_secret = #≥1(anc ∘ NAME=secret),
deep = #≥1(grand ∘ PARENT), deny = under_secret ∩ deep. Random nested folder trees (folders up to depth 6, `secret`
at random places and decoy names, 1–2 files per folder); 3 example trees, 3 held-out (the test), 5 larger check trees.
Terminals: PARENT (child → parent), SAME_DIR, IS_DIR and NAME=<name> for the recurring names. Residuals per level
(budget 200,000 steps, 120 s) and the flat enumerator on `deny` (size 12, 100,000 behaviours, the operators T, ⁺, #≥1,
∘, ∩, ∪, ∖). The residual grammar has no closure, count or intersection, so only `grand` is a chain; the question is
whether the residuals find anything false on the others (they must not) and how fast they give up.

    python benchmarks/relations_residual.py              # both parts, a few minutes
    python benchmarks/relations_residual.py --json out.json

Measured (CPU, numpy, OMP_NUM_THREADS=4 on a shared machine; 321 s for both parts):
(i) residual 45 / 45 found, 45 exact on the check worlds, 0 false accepts, 3.0 s in all — per rung the mean steps
4 (grand), 9 (sibling), 26 (uncle), 21 (ggrand), 118 (cousin), 69 (guncle), 431 (cousin1r), 1,957 (cousin2), 7,981
(cousin2_1r; 7,628–9,306 by seed, 0.40–0.55 s); flat 20 / 45 — grand, sibling, uncle, ggrand on every seed (18, 647,
9,431, 145 behaviours), none of the five rungs above (≈ 6.2 s each at the cap) — 0 false accepts, 160 s in all.
The earlier measurement in the author's research line reported the same 45 / 45 against 20 / 45 (and 7,644 steps for
the top rung on seed 0, which this code repeats).
(ii) residual: grand 5 / 5 as (PARENT ∘ PARENT); anc stopped at its budget (≈ 10 s); under_secret, deep and deny
exhausted in 506 steps (0.04 s): no chain of up to 12 atoms equals them on the examples; 0 false accepts. Flat on
deny: the cap on four seeds (21–29 s); on seed 1 it returned #≥1(((PARENT)⁺ ∘ NAME=secret)) after 377 behaviours —
exact on the three example trees, wrong on the check trees (a false accept: the flat arm has no held-out test inside
its search)."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import Relations, evaluate, show, synthesize, synthesize_residual
    from solvi.experimental.relations.library import HeldOut

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2, 3, 4)
FLAT_CAP = 100_000
FLAT_SECONDS = 60.0
RES_DEPTH = 9
RES_BUDGET = 500_000


def _toy():
    spec = importlib.util.spec_from_file_location("relations_library_toy", HERE / "relations_library_toy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


K = _toy()


def _rels(worlds, names):
    return [Relations.from_dense(list(names), np.stack([np.asarray(w[k], bool) for k in names])) for w in worlds]


def _exact(program, worlds, names, truth):
    got = evaluate(program, _rels(worlds, names))
    return all(np.array_equal(np.asarray(g, bool), np.asarray(t, bool)) for g, t in zip(got, truth))


# ───────────────────────────────────────────── (i) the kinship ladder
def ladder(seeds=SEEDS, flat=True, flat_cap=FLAT_CAP, flat_seconds=FLAT_SECONDS, depth=RES_DEPTH, budget=RES_BUDGET,
           say=print):
    rows = []
    for seed in seeds:
        train, held, check = K.kin_worlds(seed)
        test = HeldOut(held, {k: f for k, (_, f) in K.KIN.items()})
        for name, (lvl, f) in K.KIN.items():
            truth = [f(w) for w in check]
            r = synthesize_residual(train, f, depth=depth, budget=budget, name=name, test=test)
            row = {"seed": seed, "target": name, "level": lvl,
                   "residual": {"found": r.program is not None, "sec": round(r.seconds, 3), "steps": r.steps,
                                "stopped": r.stopped, "rejected": r.rejected,
                                "check_ok": r.program is not None and _exact(r.program, check, ["E"], truth),
                                "program": show(r.program) if r.program else None}}
            if flat:
                t0 = time.time()
                fl = synthesize(_rels(train, ["E"]), [f(w) for w in train], size=12, include_universal=False,
                                max_behaviours=flat_cap, seconds=flat_seconds, max_bytes=1_000_000_000)
                row["flat"] = {"found": fl.ast is not None, "sec": round(time.time() - t0, 3),
                               "behaviours": fl.behaviours, "stopped": fl.stopped,
                               "check_ok": fl.ast is not None and _exact(fl.ast, check, ["E"], truth),
                               "program": show(fl.ast) if fl.ast else None}
            rows.append(row)
            say(f"seed {seed} {name:11s} residual {'found' if row['residual']['found'] else '-':5s} "
                f"{row['residual']['sec']:7.2f} s {r.steps:7d} steps"
                + (f" | flat {'found' if row['flat']['found'] else '-':5s} {row['flat']['sec']:6.2f} s "
                   f"{row['flat']['behaviours']:7d} behaviours ({row['flat']['stopped']})" if flat else ""))
    return _summary(rows, ("residual", "flat") if flat else ("residual",), list(K.KIN))


def _summary(rows, arms, targets):
    out = {"rows": rows, "rungs": {}}
    for arm in arms:
        out[arm] = {"runs": len(rows), "found": sum(r[arm]["found"] for r in rows),
                    "exact_on_check": sum(r[arm]["check_ok"] for r in rows),
                    "false_accepts": sum(r[arm]["found"] and not r[arm]["check_ok"] for r in rows),
                    "seconds": round(sum(r[arm]["sec"] for r in rows), 2)}
    for t in targets:
        rs = [r for r in rows if r["target"] == t]
        out["rungs"][t] = {arm: {"found": sum(r[arm]["found"] for r in rs), "of": len(rs),
                                 "false_accepts": sum(r[arm]["found"] and not r[arm]["check_ok"] for r in rs),
                                 "mean_sec": round(float(np.mean([r[arm]["sec"] for r in rs])), 3),
                                 "max_sec": round(float(np.max([r[arm]["sec"] for r in rs])), 3)}
                           for arm in arms}
        out["rungs"][t]["level"] = rs[0]["level"]
        out["rungs"][t]["residual"]["mean_steps"] = round(float(np.mean([r["residual"]["steps"] for r in rs])))
        if "flat" in arms:
            out["rungs"][t]["flat"]["mean_behaviours"] = round(float(np.mean([r["flat"]["behaviours"] for r in rs])))
    return out


# ───────────────────────────────────────────── (ii) the composite path-tree policy
_W = ["alpha", "bravo", "coral", "dingo", "eagle", "fable", "gecko", "hydra", "ibis", "jolt", "koala", "lotus",
      "mango", "nova", "otter", "panda", "quartz", "rhino", "sable", "tango", "uvula", "viper", "walrus", "xenon"]
_GEN = ["cache", "tmp", "old", "src", "docs", "data", "logs", "build"]
LEVELS = ("anc", "grand", "under_secret", "deep", "deny")


def random_tree(rng, n_dirs=30):
    dirs = []
    while len(dirs) < n_dirs:
        parent = rng.choice([""] + dirs) if dirs else ""
        if parent.count("/") >= 5:
            continue
        u = rng.random()
        name = "secret" if u < 0.12 else rng.choice(["secrets", "secret_old", "top_secret"]) if u < 0.2 else \
            rng.choice(_GEN) if u < 0.5 else rng.choice(_W)
        d = f"{parent}/{name}".lstrip("/")
        if d not in dirs:
            dirs.append(d)
    if sum(d.split("/")[-1] == "secret" for d in dirs) < 2:
        return random_tree(rng, n_dirs)
    files = [f"{d}/{rng.choice(_W)}_{k}.txt" for d in dirs for k in range(1 + rng.randrange(2))]
    return {"dirs": dirs, "files": files}


def tree_graph(tree):
    """Nodes: the root (0), every folder, every file (parents before children) → parent array, is_dir, names."""
    nodes = {"": 0}
    par, isd, names = [-1], [True], [""]
    for p in tree["dirs"] + tree["files"]:
        parts = p.split("/")
        for k in range(1, len(parts) + 1):
            q = "/".join(parts[:k])
            if q not in nodes:
                nodes[q] = len(par)
                par.append(nodes["/".join(parts[:k - 1])])
                isd.append(q in tree["dirs"] or k < len(parts))
                names.append(parts[k - 1])
    return par, isd, names


def tree_world(tree, name_set):
    par, isd, names = tree_graph(tree)
    n = len(par)
    w = {k: np.zeros((n, n), bool) for k in ["PARENT", "SAME_DIR", "IS_DIR"] + [f"NAME={x}" for x in name_set]}
    for c, p in enumerate(par):
        if p >= 0:
            w["PARENT"][c, p] = True
    P = w["PARENT"].astype(np.int64)
    w["SAME_DIR"] = ((P @ P.T) > 0) & ~np.eye(n, dtype=bool)
    for i in range(n):
        w["IS_DIR"][i, i] = isd[i]
        if f"NAME={names[i]}" in w:
            w[f"NAME={names[i]}"][i, i] = True
    return w


def tree_truth(tree):
    """The policy's levels by plain code over the parent array (not programs)."""
    par, _, names = tree_graph(tree)
    n = len(par)
    anc = np.zeros((n, n), bool)
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[par[i]] + 1
        p = par[i]
        while p >= 0:
            anc[i, p] = True
            p = par[p]
    grand = np.zeros((n, n), bool)
    for i in range(n):
        if par[i] >= 0 and par[par[i]] >= 0:
            grand[i, par[par[i]]] = True
    under = np.diag([any(names[a] == "secret" for a in np.flatnonzero(anc[i])) for i in range(n)])
    deep = np.diag([d >= 3 for d in depth])
    return {"anc": anc, "grand": grand, "under_secret": under, "deep": deep, "deny": under & deep}


def recurring_names(trees, top=8):
    from collections import Counter
    c = Counter(d.split("/")[-1] for t in trees for d in t["dirs"])
    rec = sorted([(k, x) for x, k in c.items() if k >= 2], key=lambda z: (-z[0], z[1]))[:top]
    return sorted(set(x for _, x in rec) | {"secret"})


def composite_seed(seed):
    rng = random.Random(f"composite-policy-{seed}")
    tr = [random_tree(rng) for _ in range(3)]
    va = [random_tree(rng) for _ in range(3)]
    te = [random_tree(rng, n_dirs=45) for _ in range(5)]
    names = recurring_names(tr)
    return tr, va, te, names


def composite(seeds=SEEDS, budget=200_000, seconds=120.0, flat=True, flat_cap=FLAT_CAP, flat_seconds=120.0,
              say=print):
    rows = []
    for seed in seeds:
        tr, va, te, names = composite_seed(seed)
        base = ["PARENT", "SAME_DIR", "IS_DIR"] + [f"NAME={x}" for x in names]
        W = {k: [tree_world(t, names) for t in g] for k, g in (("train", tr), ("held", va), ("check", te))}
        Y = {k: [tree_truth(t) for t in g] for k, g in (("train", tr), ("held", va), ("check", te))}
        test = HeldOut(W["held"], {lv: [y[lv] for y in Y["held"]] for lv in LEVELS})
        for lv in LEVELS:
            r = synthesize_residual(W["train"], [y[lv] for y in Y["train"]], names=base, depth=12, budget=budget,
                                    seconds=seconds, name=lv, test=test)
            ok = r.program is not None and _exact(r.program, W["check"], base, [y[lv] for y in Y["check"]])
            row = {"seed": seed, "level": lv, "residual": {"found": r.program is not None, "check_ok": ok,
                                                           "sec": round(r.seconds, 2), "steps": r.steps,
                                                           "stopped": r.stopped, "rejected": r.rejected,
                                                           "program": show(r.program) if r.program else None}}
            if flat and lv == "deny":
                t0 = time.time()
                fl = synthesize(_rels(W["train"], base), [y[lv] for y in Y["train"]], size=12,
                                ops=("T", "PLUS", "CNT1", "COMP", "AND", "OR", "MINUS"), max_behaviours=flat_cap,
                                seconds=flat_seconds, max_bytes=1_000_000_000)
                fok = fl.ast is not None and _exact(fl.ast, W["check"], base, [y[lv] for y in Y["check"]])
                row["flat"] = {"found": fl.ast is not None, "check_ok": fok, "sec": round(time.time() - t0, 2),
                               "behaviours": fl.behaviours, "stopped": fl.stopped,
                               "program": show(fl.ast) if fl.ast else None}
            rows.append(row)
            say(f"composite seed {seed} {lv:12s} residual {r.stopped:9s} {r.seconds:7.2f} s {r.steps:7d} steps "
                f"rejected {r.rejected}" + (f" | flat {row['flat']['stopped']} {row['flat']['sec']} s "
                                            f"{row['flat']['behaviours']} behaviours" if "flat" in row else ""))
    out = {"rows": rows, "levels": {}}
    for lv in LEVELS:
        rs = [r for r in rows if r["level"] == lv]
        out["levels"][lv] = {"residual": {"found": sum(r["residual"]["found"] for r in rs), "of": len(rs),
                                          "false_accepts": sum(r["residual"]["found"] and not r["residual"]["check_ok"]
                                                               for r in rs),
                                          "stopped": sorted({r["residual"]["stopped"] for r in rs}),
                                          "mean_sec": round(float(np.mean([r["residual"]["sec"] for r in rs])), 2),
                                          "programs": sorted({r["residual"]["program"] for r in rs
                                                              if r["residual"]["program"]})}}
        fr = [r["flat"] for r in rs if "flat" in r]
        if fr:
            out["levels"][lv]["flat"] = {"found": sum(x["found"] for x in fr), "of": len(fr),
                                         "false_accepts": sum(x["found"] and not x["check_ok"] for x in fr),
                                         "stopped": sorted({x["stopped"] for x in fr}),
                                         "mean_sec": round(float(np.mean([x["sec"] for x in fr])), 2)}
    return out


def _table(lad, comp):
    lines = ["(i) kinship ladder, no library (found of seeds; mean seconds; false accepts on check worlds)",
             f"  {'rung':11s} lvl  residual found  sec   steps     | flat found  sec   behaviours"]
    for t, x in lad["rungs"].items():
        r, f = x["residual"], x.get("flat")
        lines.append(f"  {t:11s} {x['level']:>3}  {r['found']}/{r['of']} (fa {r['false_accepts']})  {r['mean_sec']:6.2f} "
                     f"{r['mean_steps']:>8,}" + (f"  | {f['found']}/{f['of']} (fa {f['false_accepts']})  "
                                                 f"{f['mean_sec']:6.2f} {f['mean_behaviours']:>10,}" if f else ""))
    for arm in ("residual", "flat"):
        if arm in lad:
            a = lad[arm]
            lines.append(f"  {arm}: found {a['found']}/{a['runs']}, exact on check {a['exact_on_check']}, false accepts "
                         f"{a['false_accepts']}, {a['seconds']} s in all")
    if comp:
        lines.append("(ii) composite path-tree policy, no library")
        for lv, x in comp["levels"].items():
            r = x["residual"]
            s = (f"  {lv:12s} residual {r['found']}/{r['of']} (fa {r['false_accepts']}) stopped {r['stopped']} "
                 f"{r['mean_sec']} s {r['programs']}")
            if "flat" in x:
                f = x["flat"]
                s += f" | flat {f['found']}/{f['of']} (fa {f['false_accepts']}) stopped {f['stopped']} {f['mean_sec']} s"
            lines.append(s)
    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.time()
    lad = ladder()
    comp = composite() if "--no-composite" not in sys.argv else None
    print(_table(lad, comp))
    print(f"\n{time.time() - t0:.1f} s")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps({"ladder": lad, "composite": comp},
                                                                            indent=1, ensure_ascii=False))
