"""The law loop (solvi.experimental.laws) on two deterministic toys, each well under a minute on a laptop CPU:

    python benchmarks/laws_toy.py              # both, as JSON
    python benchmarks/laws_toy.py --probes     # and 1b, the gridworld with probes (20 seeds × 5 arms × 2 pools, ~4 min)
    pytest -m slow tests/test_laws_toy.py      # the same with the bars

1. gridworld() — a grid of walkable (grass, sand), blocked (stone, tree, water) and pickup (apple) cells with written
   rules: a move into a walkable cell moves, into a blocked cell or off the map does not, onto an apple moves and picks
   it up (+1 apple, the cell becomes grass); "do" at a tree gives +1 wood; "do" at stone with wood gives +1 stone and
   leaves sand; counters stop at 9. The base predictor is a naive map ("a move moves, do does nothing"). A random
   walker gives a training stream of 500 transitions (5 worlds), a held-out buffer (10 other worlds) and a test (20
   fresh worlds). Arms: the base; the loop with the enumerating proposer; the loop with a random proposer under the same
   held-out test; the written rules (the ceiling). Reported: exact-effect accuracy on the test, the share of the gap
   between the base and the written rules the laws close, the number of laws.

1b. gridworld_probes() — the same gridworld learned by `admit_by_probes` (the written rules answer the probes) from
   each mismatch of a 300-transition stream: arms with and without `Grammar(implies=)`, `coverage="varied"`, and with
   both encodings of a +1 offered, with and without the declared equivalences (`LawSet(equivalent=)`). Reported:
   admissions refused for an implied atom or for congruent twins only, laws admitted, probes per law, false laws on 20
   fresh worlds, the gap closed.

2. policy() — a stateless path policy, unknown to the agent: read, list and write on paths of a sandbox are allowed,
   except relative paths, paths outside the root, a prefix sibling of the root and paths through a link to a folder or
   a file outside — those are denied; read / list of a missing path answers "missing", a call of the wrong kind on a
   file or folder "other_error". Two sessions of 200 calls (70 to denied paths), each on a fresh layout with other
   names. The agent predicts each call (default "ok"), skips a call it predicts "denied", and on a mismatch spends up to
   3 probes on other paths before the held-out test (rows of every other path; a law is never tested on the path it was
   born from). Arms: probes planned by the mismatch (split the candidate laws, read-only first, closest to the
   mismatched call), and random probes at the same budget, spent at the same steps. Reported: refusals and wrongly
   skipped allowed calls per session, probes, laws."""
from __future__ import annotations

import json
import random
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, Buffer, Enumerate, Grammar, LawSet, RandomProposer,
                                         Transition, discriminate, split)

# ───────────────────────────────────────────── 1. the gridworld
SIZE = 9
CELLS = (("grass", 0.50), ("sand", 0.10), ("stone", 0.13), ("tree", 0.13), ("water", 0.07), ("apple", 0.07))
WALKABLE = {"grass", "sand", "apple"}
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ACTIONS = ("up", "down", "left", "right", "do")
ITEMS = ("wood", "stone", "apple")


def new_world(seed):
    rng = random.Random(f"grid-{seed}")
    names, weights = zip(*CELLS)
    grid = [[rng.choices(names, weights)[0] for _ in range(SIZE)] for _ in range(SIZE)]
    c = SIZE // 2
    grid[c][c] = "grass"
    return {"grid": grid, "pos": (c, c), "facing": (0, 1), "inv": {k: 0 for k in ITEMS}}


def target(state, action):
    """The destination cell of a move, the faced cell of "do"; "edge" off the map."""
    dx, dy = MOVES[action] if action in MOVES else state["facing"]
    x, y = state["pos"][0] + dx, state["pos"][1] + dy
    if not (0 <= x < SIZE and 0 <= y < SIZE):
        return "edge", None
    return state["grid"][y][x], (x, y)


def written(state, action):
    """The written rules → the effect (moved, +wood, +stone, +apple, the target cell after or "same")."""
    cell, _ = target(state, action)
    inv = state["inv"]
    if action in MOVES:
        if cell == "apple":
            return (True, 0, 0, int(inv["apple"] < 9), "grass")
        return (cell in WALKABLE, 0, 0, 0, "same")
    if cell == "tree":
        return (False, int(inv["wood"] < 9), 0, 0, "same")
    if cell == "stone" and inv["wood"] >= 1:
        return (False, 0, int(inv["stone"] < 9), 0, "sand")
    return (False, 0, 0, 0, "same")


def step_world(state, action):
    """Apply the written rules → the next state (a fresh dict; the grid is copied only when a cell changes)."""
    moved, dw, ds, da, after = written(state, action)
    _, xy = target(state, action)
    grid = state["grid"]
    if after != "same":
        grid = [row[:] for row in grid]
        grid[xy[1]][xy[0]] = after
    inv = dict(state["inv"])
    inv["wood"] += dw
    inv["stone"] += ds
    inv["apple"] += da
    facing = MOVES[action] if action in MOVES else state["facing"]
    pos = xy if moved else state["pos"]
    return {"grid": grid, "pos": pos, "facing": facing, "inv": inv}


def episode(seed, steps):
    """A random walker (60% moves, 40% "do") → transitions, each observed effect from the written rules."""
    rng = random.Random(f"walker-{seed}")
    s = new_world(seed)
    out = []
    for i in range(steps):
        a = rng.choice(ACTIONS[:4]) if rng.random() < 0.6 else "do"
        out.append(Transition(s, a, written(s, a), group=seed, id=f"{seed}-{i}"))
        s = step_world(s, a)
    return out


def naive_map(state, action):
    return (action in MOVES, 0, 0, 0, "same")


def grid_grammar(implies=False):
    atoms = [Atom(f"target={m}", (lambda m: lambda s, a: target(s, a)[0] == m)(m), feature="target")
             for m in ("grass", "sand", "stone", "tree", "water", "apple", "edge")]
    atoms += [Atom("target walkable", lambda s, a: target(s, a)[0] in WALKABLE, feature="walk"),
              Atom("target blocked", lambda s, a: target(s, a)[0] not in WALKABLE, feature="walk")]
    for it in ITEMS:
        atoms += [Atom(f"{it}=0", (lambda it: lambda s: s["inv"][it] == 0)(it), feature=it),
                  Atom(f"{it}>=1", (lambda it: lambda s: s["inv"][it] >= 1)(it), feature=it),
                  Atom(f"{it}<9", (lambda it: lambda s: s["inv"][it] < 9)(it), feature=it),
                  Atom(f"{it}=9", (lambda it: lambda s: s["inv"][it] == 9)(it), feature=it)]
    return Grammar(atoms, k=3, implies=GRID_IMPLIES if implies else None)


# implications by construction: a target material fixes walkable / blocked; the counter atoms (one feature each, so
# never in one condition together) are declared as written
GRID_IMPLIES = {**{f"target={m}": ["target walkable" if m in WALKABLE else "target blocked"]
                   for m in ("grass", "sand", "stone", "tree", "water", "apple", "edge")},
                **{f"{it}=9": [f"{it}>=1"] for it in ITEMS}, **{f"{it}=0": [f"{it}<9"] for it in ITEMS}}


def accuracy(predict, rows):
    return sum(predict(t.state, t.action) == t.observed for t in rows) / len(rows)


def gridworld_one(seed=0, train_worlds=5, steps=100, buffer_worlds=10, test_worlds=20):
    g = grid_grammar()
    train = [t for w in range(train_worlds) for t in episode(1000 * seed + w, steps)]
    held = Buffer(g)
    for w in range(buffer_worlds):
        held.extend(episode(1000 * seed + 100 + w, steps))
    test = [t for w in range(test_worlds) for t in episode(1000 * seed + 200 + w, steps)]
    arms = {"base": {"accuracy": accuracy(naive_map, test)}}
    for name, proposer in (("laws", Enumerate()), ("random", RandomProposer(seed=seed))):
        ls = LawSet(g, base=naive_map, proposer=proposer, min_support=3, sized=False)   # measured so
        steps_ = ls.learn(train, held)
        arms[name] = {"accuracy": accuracy(ls.predict, test), "laws": len(ls.laws), "retired": len(ls.retired),
                      "mismatches": sum(s.mismatch for s in steps_),
                      "candidates": sum(s.candidates for s in steps_),
                      "examples": [law.text() for law in ls.laws[:8]]}
    arms["written"] = {"accuracy": accuracy(written, test)}
    base, laws, top = arms["base"]["accuracy"], arms["laws"]["accuracy"], arms["written"]["accuracy"]
    return {"seed": seed, "train_transitions": len(train), "buffer_transitions": len(held),
            "test_transitions": len(test), "arms": arms,
            "gap_closed": (laws - base) / (top - base) if top > base else None,
            "random_minus_laws": arms["random"]["accuracy"] - laws}


def _mean(xs):
    return sum(xs) / len(xs)


def gridworld(seeds=(0, 1, 2, 3, 4), **kw):
    """The gridworld on several seeds (each: its own worlds for the stream, the buffer and the test) → per seed and the
    means."""
    runs = [gridworld_one(s, **kw) for s in seeds]
    arms = {a: {"accuracy": _mean([r["arms"][a]["accuracy"] for r in runs])} for a in ("base", "laws", "random", "written")}
    for a in ("laws", "random"):
        arms[a]["laws"] = _mean([r["arms"][a]["laws"] for r in runs])
    return {"toy": "gridworld", "seeds": list(seeds), "train_transitions": max(r["train_transitions"] for r in runs),
            "arms": arms, "gap_closed": _mean([r["gap_closed"] for r in runs]),
            "random_minus_laws": arms["random"]["accuracy"] - arms["laws"]["accuracy"],
            "worst_random_minus_laws": min(r["random_minus_laws"] for r in runs), "runs": runs}


# ───────────────────────────────────────────── 1b. the gridworld with probes (Grammar(implies=), equivalent=)
# The written rules answer probes (`ask`); a mismatch on the training stream offers its candidates to
# `admit_by_probes` over a pool of (state, action) pairs from other worlds, with the logged rows as `buffer`.
# Effects: the observation (moved, +wood, +stone, +apple, the cell after) — relative counts; with variants=True also
# the absolute encoding ("set", moved, item, value, cell after) of a +1, so the caller does not know which encoding the
# world uses. Declared equivalences: +1 ≡ set 1 under item=0.
def set_effect(state, observed):
    moved, dw, ds, da, after = observed
    for it, d in zip(ITEMS, (dw, ds, da)):
        if d == 1:
            return ("set", moved, it, state["inv"][it] + 1, after)
    return None


def grid_apply(effect, state):
    if effect and effect[0] == "set":
        _, moved, it, v, after = effect
        d = [v - state["inv"][it] if x == it else 0 for x in ITEMS]
        return (moved, d[0], d[1], d[2], after)
    return effect


def grid_variants(state, observed):
    e = set_effect(state, observed)
    return [observed] if e is None else [observed, e]


GRID_EQUIVALENT = [((False, 1, 0, 0, "same"), ("set", False, "wood", 1, "same"), "wood=0"),
                   ((False, 0, 1, 0, "sand"), ("set", False, "stone", 1, "sand"), "stone=0"),
                   ((True, 0, 0, 1, "grass"), ("set", True, "apple", 1, "grass"), "apple=0")]

PROBE_ARMS = {
    # gap 4: the same loop with and without the declared implications
    "base": dict(implies=False, coverage="undetermined"),
    "implies": dict(implies=True, coverage="undetermined"),
    "varied": dict(implies=True, coverage="varied"),
    # gap 10: both encodings offered; siblings="undetermined" (else the order picks between twins)
    "variants": dict(implies=True, coverage="varied", variants=True, siblings="undetermined"),
    "equivalent": dict(implies=True, coverage="varied", variants=True, siblings="undetermined", equivalent=True),
}


def _ref():
    """A reference set with every declaration, to classify refusals (it admits nothing)."""
    return LawSet(grid_grammar(implies=True), base=naive_map, apply=grid_apply, equivalent=GRID_EQUIVALENT)


def _congruent_only(ref, adm):
    """An admission refused only because its surviving class holds congruent twins (same condition, effects
    declared one under an atom it holds)."""
    if adm.status != "undetermined" or len(adm.classes) != 1 or "cannot tell apart" not in adm.why:
        return False
    j, *rest = adm.classes[0]
    twins = [c for c in rest if len(c.condition) == len(j.condition)]
    return bool(twins) and all(ref.congruent(j, c) for c in twins)


def probe_pool(seed, pool_worlds, steps, inventory="walk"):
    """(state, action) pairs from random walks on other worlds; inventory="random": each state's counters redrawn
    uniformly from 0..9 (a pool where every counter atom varies), "walk": as the walker had them."""
    pool = [(t.state, t.action) for w in range(pool_worlds) for t in episode(1000 * seed + 300 + w, steps)]
    if inventory == "random":
        rng = random.Random(f"pool-{seed}")
        pool = [({**s, "inv": {it: rng.randrange(10) for it in ITEMS}}, a) for s, a in pool]
    return pool


def gridworld_probes_one(seed=0, arm="implies", train_worlds=3, steps=100, pool_worlds=3, cover_worlds=10,
                         test_worlds=20, budget=8, inventory="walk"):
    cfg = PROBE_ARMS[arm]
    g = grid_grammar(implies=cfg.get("implies", False))
    train = [t for w in range(train_worlds) for t in episode(1000 * seed + w, steps)]
    pool = probe_pool(seed, pool_worlds, steps, inventory)
    cover = Buffer(g)
    for w in range(cover_worlds):
        cover.extend(episode(1000 * seed + 100 + w, steps))
    test = [t for w in range(test_worlds) for t in episode(1000 * seed + 200 + w, steps)]
    ls = LawSet(g, base=naive_map, apply=grid_apply, effects=grid_variants if cfg.get("variants") else None,
                equivalent=GRID_EQUIVALENT if cfg.get("equivalent") else None)
    ref, rg = _ref(), grid_grammar(implies=True)
    buf = Buffer(g)
    rec = {"admissions": 0, "accepted": 0, "undetermined": 0, "implied_refusals": 0, "congruent_only": 0,
           "probes": 0, "with_aliases": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in train:
            ls.observe(t)
            buf.add(t)
            if ls.agrees(ls.predict(t.state, t.action), t.observed):
                continue
            cands = ls.mismatch(t.state, t.action, t.observed)
            if not cands:
                continue
            adm = ls.admit_by_probes(cands, written, probes=pool, buffer=buf, budget=budget,
                                     coverage=cfg["coverage"], siblings=cfg.get("siblings", "shortest"),
                                     coverage_on=cover if cfg["coverage"] == "varied" else None)
            rec["admissions"] += 1
            rec["probes"] += len(adm.asked)
            if adm.status == "accepted":
                rec["accepted"] += 1
                rec["with_aliases"] += bool(adm.law.aliases)
            else:
                rec["undetermined"] += 1
                if adm.untested and adm.classes and set(adm.untested) & rg.implied(adm.classes[0][0].condition):
                    rec["implied_refusals"] += 1
                rec["congruent_only"] += _congruent_only(ref, adm)
    false = set()
    for t in test:
        law = ls.applied(t.state, t.action)
        if law is not None and grid_apply(law.effect, t.state) != t.observed:
            false.add(law.id)
    acc = accuracy(ls.predict, test)
    base, top = accuracy(naive_map, test), accuracy(written, test)
    rec.update({"seed": seed, "arm": arm, "inventory": inventory, "laws": len(ls.laws), "retired": len(ls.retired),
                "false_laws": len(false), "false_examples": [x.text() for x in ls.laws if x.id in false][:4],
                "accuracy": acc, "base": base, "gap_closed": (acc - base) / (top - base) if top > base else None,
                "probes_per_law": rec["probes"] / rec["accepted"] if rec["accepted"] else None})
    return rec


def gridworld_probes(seeds=tuple(range(20)), arms=tuple(PROBE_ARMS), **kw):
    """Every arm on every seed → per run and per arm (sums; means of accuracy and gap closed). Measured (20 seeds,
    budget 8; pool "random": 600 pairs with random inventories / "walk": 300 pairs as walked): implied-atom refusals
    base 485 / 536 → 0 with implies; laws admitted 81 → 99 (varied 139) / 0 → 0 (varied 111); gap closed 0.62 → 0.81
    (varied 0.95) / 0 → 0 (varied 0.73); both encodings offered: probes per admitted law 12.0 → 8.9 / 14.3 → 10.4
    with the equivalences, laws 118 → 126 / 97 → 105, refusals with congruent-only twins 0 in both arms; false laws
    on the fresh worlds 0 in every arm and seed."""
    runs = [gridworld_probes_one(s, a, **kw) for s in seeds for a in arms]
    return {"toy": "gridworld_probes", "seeds": list(seeds), "arms": _probe_summary(runs), "runs": runs}


def _probe_summary(runs):
    out = {}
    for a in dict.fromkeys(r["arm"] for r in runs):
        rs = [r for r in runs if r["arm"] == a]
        sums = {k: sum(r[k] for r in rs) for k in ("admissions", "accepted", "undetermined", "implied_refusals",
                                                   "congruent_only", "probes", "with_aliases", "false_laws",
                                                   "laws")}
        sums["probes_per_law"] = sums["probes"] / sums["accepted"] if sums["accepted"] else None
        sums["accuracy"] = _mean([r["accuracy"] for r in rs])
        sums["gap_closed"] = _mean([r["gap_closed"] for r in rs])
        sums["seeds_with_false_laws"] = sum(r["false_laws"] > 0 for r in rs)
        out[a] = sums
    return out


# ───────────────────────────────────────────── 2. the path policy
TOOLS = ("read", "list", "write")
DENIED_CLASSES = (("inside", "relative", "none"), ("outside", "absolute", "none"), ("sibling", "absolute", "none"),
                  ("inside", "absolute", "dirlink"), ("inside", "absolute", "filelink"))
DISTRACTORS = {"exists": ("yes", "no"), "kind": ("file", "dir"), "depth": ("0", "1", "2"), "hidden": ("no", "yes"),
               "ext": ("txt", "csv", "md")}


def layout(seed, per_denied_class=6):
    """A sandbox: every combination of the distractors on an allowed path (inside, absolute, no link) and
    `per_denied_class` paths of each denied class with random distractors; names differ per layout."""
    rng = random.Random(f"layout-{seed}")
    paths = []
    keys = sorted(DISTRACTORS)
    combos = [{}]
    for k in keys:
        combos = [{**c, k: v} for c in combos for v in DISTRACTORS[k]]
    for c in combos:
        paths.append({"where": "inside", "form": "absolute", "link": "none", **c})
    for where, form, link in DENIED_CLASSES:
        for _ in range(per_denied_class):
            paths.append({"where": where, "form": form, "link": link, **{k: rng.choice(DISTRACTORS[k]) for k in keys}})
    for i, p in enumerate(paths):
        p["path"] = f"s{seed}-p{i}-{rng.randrange(10**6)}"
    return paths


def policy_outcome(path, tool):
    """The written policy and the server's answers (the agent is never told)."""
    if path["form"] == "relative" or path["where"] != "inside" or path["link"] != "none":
        return "denied"
    return "ok"


def workload(paths, seed, n=200, denied=70):
    rng = random.Random(f"workload-{seed}")
    bad = [p for p in paths if policy_outcome(p, "read") == "denied"]
    good = [p for p in paths if p not in bad]
    calls = [(rng.choice(bad), rng.choice(TOOLS)) for _ in range(denied)]
    calls += [(rng.choice(good), rng.choice(TOOLS)) for _ in range(n - denied)]
    rng.shuffle(calls)
    return calls


def policy_grammar():
    atoms = []
    for attr, values in (("tool", TOOLS), ("where", ("inside", "outside", "sibling")), ("form", ("absolute", "relative")),
                         ("link", ("none", "dirlink", "filelink")), *DISTRACTORS.items()):
        atoms += [Atom(f"{attr}={v}", (lambda attr, v: lambda s: s[attr] == v)(attr, v), feature=attr)
                  for v in values]
    return Grammar(atoms, k=3)


def ok_base(state, action):
    return "ok"


def call(path, tool):
    """One call as the law set sees it: the state is the path's classes and the tool; the action is "call" (the policy
    is stateless, so one law may cover every tool)."""
    return {**path, "tool": tool}


def _closeness(ref):
    keys = [k for k in ref if k != "path"]
    return lambda s, a: sum(s[k] != ref[k] for k in keys)


def _read_only(state, action):
    return state["tool"] != "write"


def _safe(state, action):
    return True                                        # every target is inside the toy's own sandbox


def policy_one(seed=0, max_probes=3):
    g = policy_grammar()
    sessions = [layout(1000 * seed + 1), layout(1000 * seed + 2)]
    loads = [workload(sessions[0], 1000 * seed + 1), workload(sessions[1], 1000 * seed + 2)]
    budget = {}                                        # (session, step) → probes the mismatch arm spent there
    arms = {}
    for arm in ("mismatch", "random"):
        rng = random.Random(f"probes-{seed}")
        ls = LawSet(g, base=ok_base, min_support=1, min_precision=1.0, retire_below=1.0, retire_min=1,
                    sized=False)                      # one answer of a deterministic server: measured so
        buf = Buffer(g)
        rec = {"probes": 0}
        for si, (paths, calls) in enumerate(zip(sessions, loads), 1):
            refusals = skipped_wrong = skipped = 0
            space = [(call(p, t), "call") for p in paths for t in TOOLS]
            for step, (path, tool) in enumerate(calls):
                state = call(path, tool)
                truth = policy_outcome(path, tool)
                spent = 0
                if arm == "random":                    # the same budget at the same step, spent at random first
                    spent = _random_probes(ls, buf, rng, space, budget.get((si, step), 0), si, step)
                pred = ls.predict(state, "call")
                if pred == "denied":                   # the agent skips it: nothing is executed, nothing observed
                    skipped += 1
                    skipped_wrong += truth != "denied"
                else:
                    refusals += truth == "denied"
                    t = Transition(state, "call", truth, group=path["path"], id=f"{si}-{step}")
                    ls.observe(t)
                    cands = ls.mismatch(state, "call", truth) if pred != truth else []
                    if arm == "mismatch" and cands:
                        spent = _planned_probes(ls, buf, cands, state, space, max_probes, si, step)
                    if cands:
                        ls.test(cands, buf, exclude=path["path"])
                    buf.add(t)
                if arm == "mismatch":
                    budget[(si, step)] = spent
                rec["probes"] += spent
            rec[f"refusals_{si}"] = refusals
            rec[f"wrongly_skipped_{si}"] = skipped_wrong
            rec[f"skipped_{si}"] = skipped
        rec["laws"] = len(ls.laws)
        rec["retired"] = len(ls.retired)
        rec["examples"] = [law.text() for law in ls.laws if law.effect == "denied"][:8]
        arms[arm] = rec
    for rec in arms.values():
        rec["refusals"] = rec["refusals_1"] + rec["refusals_2"]
        rec["wrongly_skipped"] = rec["wrongly_skipped_1"] + rec["wrongly_skipped_2"]
    return {"seed": seed, "calls_per_session": len(loads[0]),
            "denied_per_session": sum(policy_outcome(p, t) == "denied" for p, t in loads[0]), "arms": arms}


def policy(seeds=(0, 1, 2, 3, 4), **kw):
    """The policy toy on several seeds (layouts, workloads and random probes) → per seed and the sums."""
    runs = [policy_one(s, **kw) for s in seeds]
    keys = ("probes", "refusals_1", "refusals_2", "refusals", "wrongly_skipped_1", "wrongly_skipped_2",
            "wrongly_skipped")
    arms = {a: {k: sum(r["arms"][a][k] for r in runs) for k in keys} for a in ("mismatch", "random")}
    return {"toy": "policy", "seeds": list(seeds), "sessions": 2, "calls_per_session": runs[0]["calls_per_session"],
            "denied_per_session": runs[0]["denied_per_session"], "arms": arms, "runs": runs}


def _probe(ls, state, si, step, k):
    out = policy_outcome(state, state["tool"])
    t = Transition(state, "call", out, group=state["path"], id=f"{si}-{step}-p{k}")
    ls.observe(t)
    return t


def _planned_probes(ls, buf, cands, ref, space, max_probes, si, step):
    """Split the mismatch's candidate laws (read-only tools first, closest to the mismatched call); when one is left
    (or nothing splits) and it has no support from another path yet, a probe where it and the current set disagree.
    Never the mismatched path itself, never a call already observed. → probes spent."""
    spent = 0
    seen = {(t.state["path"], t.state["tool"]) for t in buf.rows()}
    cost = _closeness(ref)
    for _ in range(max_probes):
        alive = [c for c in cands if _consistent(ls, buf, c)]
        if not alive:
            break
        options = [(s, a) for s, a in space if (s["path"], s["tool"]) not in seen and s["path"] != ref["path"]]
        pr = None
        if len(alive) > 1:
            pr = split([ls.hypothesis(c) for c in alive], _safe, options, read_only=_read_only, cost=cost)
        if pr is None:
            best = min(alive, key=lambda c: len(c.condition))
            if ls.score(best, buf, exclude=ref["path"])[0] >= 1:
                break
            pr = discriminate(ls, ls.hypothesis(best), _safe, options, read_only=_read_only, cost=cost)
            if pr is None:
                break
        buf.add(_probe(ls, pr.state, si, step, spent))
        seen.add((pr.state["path"], pr.state["tool"]))
        spent += 1
    return spent


def _consistent(ls, buf, c):
    """No held row (any group) contradicts the candidate where its condition holds."""
    sup, hits, _ = ls.score(c, buf)
    return hits == sup


def _random_probes(ls, buf, rng, space, n, si, step):
    """n uniformly random calls; a probe whose answer differs from the prediction proposes candidates, tested like a
    workload mismatch. → n."""
    for k in range(n):
        state, _ = rng.choice(space)
        pred = ls.predict(state, "call")
        t = _probe(ls, state, si, step, k)
        if pred != t.observed:
            cands = ls.mismatch(state, "call", t.observed)
            if cands:
                ls.test(cands, buf, exclude=state["path"])
        buf.add(t)
    return n


def main(argv=None):
    out = {"gridworld": gridworld(), "policy": policy()}
    if "--probes" in (argv or []):
        out["gridworld_probes"] = {inv: gridworld_probes(inventory=inv, pool_worlds=pw)
                                   for inv, pw in (("random", 6), ("walk", 3))}
    for r in out.values():
        for run in r.get("runs", []):
            for a in run["arms"].values():
                a.pop("examples", None)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
