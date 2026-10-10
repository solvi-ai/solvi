"""Learning in a deterministic gridworld described by code: plan on the exact model, compile what the code verified,
act alone under the certificate (solvi.experimental.skills).

The world (9 × 5 cells): a wall down the middle with a closed door; a key in the left room, a gem and an exit in the
right one. Goals in order: take the key ("pick" on its cell), open the door ("open" next to it, with the key), take
the gem, stand on the exit with the gem. Every episode is a fresh layout drawn from its seed: the key, the door and
the gem each at one of two places (the start and the exit fixed), a floor colour and two flowers (walkable,
irrelevant) anywhere, and in a quarter of the episodes a boulder (blocking) on a random free cell — no layout repeats
exactly. The conditions of skills are atoms of a grammar over all of it (positions, holdings, the layout, the
decoration, the boulder); a skill widened over two starts that differed in the boulder can meet a boulder on its path:
that is the failure the certificate is there for.

Four arms, 20 episodes × 5 seeds, knowledge carried within a seed:
- skills       the loop: System 1 runs a certified skill alone, else System 2 re-verifies a matching skill or plans on
               the exact model, verifies and compiles; the certificate (20%, δ 0.1) is pooled over the set;
- plan-always  System 2 plans every goal (the ceiling of goals; System 1's share is 0 by construction);
- cases        the action sequence that worked from the same start state is replayed (no condition, no certificate);
- scrambled    compiled WITHOUT verification: the planner's output replaced by a random action sequence of the same
               length, executed and compiled as it is.

    uv run python benchmarks/skills_toy.py            # the table; run() returns the numbers (tests/test_skills_toy.py)

Why a 20% promise pooled over the set: a skill of this world is used at most about once per episode, and certifying
at δ = 0.1 takes 12 clean outcomes of the certified unit at 20%, 25 at 10%, 50 at 5% (a failure under supervision
costs as many again). Measured on this script (System 1's share in the last 5 episodes, mean of the five seeds; the
choice was made after these runs): per skill nothing certified within 20 episodes at any of the three promises, per
goal only at 20% and late (0.09); pooled over the set, 0 / 0.49 / 0.81 at 5% / 10% / 20% with this layout, 0 / 0.29 / 0.68 when the start and
the exit vary too, 0.12 / 0.64 / 0.88 when only the key and the gem vary. Failures among System 1's decisions stayed
at 0–4.4% everywhere. More episodes, fewer distinct situations or a looser promise certify sooner; a 5% promise needs
streams long enough to give each certified unit fifty clean uses."""
from __future__ import annotations

import json
import random
import sys
import time
import warnings

from solvi.core import Outcome

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Grammar
    from solvi.experimental.skills import Certificate, Plan, SkillLearner, plan

W, H, WALL = 9, 5, 4
STARTS, KEYS, DOORS, GEMS, EXITS = ((0, 2),), ((2, 0), (2, 4)), (1, 3), ((6, 0), (6, 4)), ((8, 2),)
COLOURS = ("grey", "sand", "moss", "slate")
P_BOULDER = 0.25
ACTIONS = ("up", "down", "left", "right", "pick", "open")
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
CELLS = [(x, y) for x in range(W) for y in range(H) if x != WALL]
PROMISE, DELTA = 0.2, 0.1
CERTIFICATE = dict(promise=PROMISE, delta=DELTA, scope="set")
EPISODES, SEEDS, STEPS, HORIZON, BUDGET = 20, (0, 1, 2, 3, 4), 80, 16, 20_000


class GridWorld:
    """The world as solvi.core.Environment. The layout is fixed per episode and shared by copies; the moving parts are
    a few fields, so a fork (deepcopy) is cheap."""

    def __init__(self):
        self.reset(0)

    def reset(self, seed=None):
        rng = random.Random(seed)
        start, key, door, gem, ext = (rng.choice(STARTS), rng.choice(KEYS), rng.choice(DOORS), rng.choice(GEMS),
                                      rng.choice(EXITS))
        colour = rng.choice(COLOURS)
        special = {start, key, gem, ext, (WALL - 1, door), (WALL + 1, door)}
        free = [c for c in CELLS if c not in special]
        flowers = tuple(sorted(rng.sample(free, 2)))
        boulder = rng.choice([c for c in free if c not in flowers]) if rng.random() < P_BOULDER else None
        self.layout = {"key": key, "door": door, "gem": gem, "exit": ext, "colour": colour, "flowers": flowers,
                       "boulder": boulder}
        self.pos, self.has_key, self.door_open, self.has_gem = start, False, False, False
        return self.state()

    def __deepcopy__(self, memo):
        c = GridWorld.__new__(GridWorld)
        c.layout, c.pos, c.has_key, c.door_open, c.has_gem = self.layout, self.pos, self.has_key, self.door_open, \
            self.has_gem
        return c

    def state(self):
        L = self.layout
        return {"pos": list(self.pos), "has_key": self.has_key, "door_open": self.door_open, "has_gem": self.has_gem,
                "key": None if self.has_key else list(L["key"]), "gem": None if self.has_gem else list(L["gem"]),
                "door": L["door"], "exit": list(L["exit"]), "colour": L["colour"],
                "flowers": [list(f) for f in L["flowers"]],
                "boulder": None if L["boulder"] is None else list(L["boulder"])}

    def actions(self, state):
        return list(ACTIONS)

    def _free(self, c):
        x, y = c
        if not (0 <= x < W and 0 <= y < H) or c == self.layout["boulder"]:
            return False
        return x != WALL or (y == self.layout["door"] and self.door_open)

    def step(self, action):
        L, ok = self.layout, False
        if action in MOVES:
            to = (self.pos[0] + MOVES[action][0], self.pos[1] + MOVES[action][1])
            if self._free(to):
                self.pos, ok = to, True
        elif action == "pick":
            if self.pos == L["key"] and not self.has_key:
                self.has_key = ok = True
            elif self.pos == L["gem"] and not self.has_gem:
                self.has_gem = ok = True
        elif action == "open":
            if self.has_key and not self.door_open and self.pos == (WALL - 1, L["door"]):
                self.door_open = ok = True
        done = self.has_gem and self.pos == L["exit"]
        return Outcome(self.state(), accepted=ok, done=done)


GOALS = [("key", lambda s: s["has_key"]), ("door", lambda s: s["door_open"]), ("gem", lambda s: s["has_gem"]),
         ("exit", lambda s: s["has_gem"] and s["pos"] == s["exit"])]


def _cell(c):
    return "none" if c is None else f"{c[0]},{c[1]}"


def grammar():
    """Atoms over everything the state shows: the agent's cell, holdings, the layout, the decoration, the boulder."""
    atoms = [Atom(f"x={i}", (lambda s, i=i: s["pos"][0] == i), feature="x") for i in range(W)]
    atoms += [Atom(f"y={j}", (lambda s, j=j: s["pos"][1] == j), feature="y") for j in range(H)]
    for f in ("has_key", "door_open", "has_gem"):
        atoms += [Atom(f, (lambda s, f=f: s[f]), feature=f), Atom(f"not_{f}", (lambda s, f=f: not s[f]), feature=f)]
    for f, places in (("key", KEYS), ("gem", GEMS), ("exit", EXITS)):
        atoms += [Atom(f"{f}@{_cell(c)}", (lambda s, f=f, c=c: _cell(s[f]) == _cell(c)), feature=f)
                  for c in tuple(places) + ((None,) if f != "exit" else ())]
    atoms += [Atom(f"door@{d}", (lambda s, d=d: s["door"] == d), feature="door") for d in DOORS]
    atoms += [Atom(f"colour={c}", (lambda s, c=c: s["colour"] == c), feature="colour") for c in COLOURS]
    for i in range(2):
        atoms += [Atom(f"flower{i}@{_cell(c)}", (lambda s, i=i, c=c: _cell(s["flowers"][i]) == _cell(c)),
                       feature=f"flower{i}") for c in CELLS]
    atoms += [Atom(f"boulder@{_cell(c)}", (lambda s, c=c: _cell(s["boulder"]) == _cell(c)), feature="boulder")
              for c in CELLS + [None]]
    return Grammar(atoms, k=3)


def scrambled_planner(seed):
    """The planner's output replaced by a random action sequence of the same length (never verified)."""
    rng = random.Random(seed)

    def planner(model, state, goal, **kw):
        p = plan(model, state, goal, **kw)
        n = len(p.actions) if p is not None else kw.get("horizon", 8)
        return Plan(tuple(rng.choice(ACTIONS) for _ in range(n)), (), float(n), 0 if p is None else p.searched)

    return planner


def learner(arm, seed):
    env = GridWorld()
    common = dict(horizon=HORIZON, budget=BUDGET, seed=seed)
    if arm == "skills":
        return SkillLearner(env, grammar(), GOALS, certificate=Certificate(**CERTIFICATE), **common)
    if arm == "plan-always":
        return SkillLearner(env, grammar(), GOALS, learn=False, **common)
    if arm == "cases":
        return SkillLearner(env, None, GOALS, certificate=False, **common)
    if arm == "scrambled":
        return SkillLearner(env, grammar(), GOALS, planner=scrambled_planner(seed), verify_plans=False,
                            certificate=False, **common)
    raise ValueError(arm)


def _share(eps):
    steps = sum(e["steps"] for e in eps)
    return sum(e["s1"] for e in eps) / steps if steps else 0.0


def one(arm, seed, episodes=EPISODES):
    L = learner(arm, seed)
    eps = [L.run(seed=1000 * seed + n, steps=STEPS) for n in range(episodes)]
    s1 = [d for d in L.decisions if d["by"] == "s1"]
    return {"seed": seed, "episodes": eps, "goals": sum(e["goals_reached"] for e in eps) / len(eps),
            "all_goals": sum(e["all_goals"] for e in eps) / len(eps), "s1_share": _share(eps),
            "s1_share_last5": _share(eps[-5:]), "s1_share_first5": _share(eps[:5]),
            "s1_decisions": len(s1), "s1_failed": sum(1 for d in s1 if d["run_ok"] is False),
            "alone_runs": sum(e["alone_runs"] for e in eps), "alone_failures": sum(e["alone_failures"] for e in eps),
            "alone_without_certificate": sum(1 for d in s1 if not d["certified"]),
            "planner_calls_first5": sum(e["planner_calls"] for e in eps[:5]) / 5,
            "planner_calls_last5": sum(e["planner_calls"] for e in eps[-5:]) / 5,
            "simulated_first5": sum(e["simulated"] for e in eps[:5]) / 5,
            "simulated_last5": sum(e["simulated"] for e in eps[-5:]) / 5,
            "steps": sum(e["steps"] for e in eps) / len(eps), "mismatch": sum(e["mismatch"] for e in eps),
            "guard_violations": sum(e["guard_violations"] for e in eps),
            "skills": len(L.skills.skills), "retired": len(L.skills.retired()),
            "supervised": sum(e["supervised"] for e in eps),
            "supervised_failed": sum(e["supervised_failed"] for e in eps)}


MEAN = ("goals", "all_goals", "s1_share", "s1_share_last5", "s1_share_first5", "planner_calls_first5",
        "planner_calls_last5", "simulated_first5", "simulated_last5", "steps", "skills", "retired")
SUM = ("s1_decisions", "s1_failed", "alone_runs", "alone_failures", "alone_without_certificate", "mismatch",
       "guard_violations", "supervised", "supervised_failed")


def run(seeds=SEEDS, episodes=EPISODES, arms=("skills", "plan-always", "cases", "scrambled")):
    """Every arm on every seed → {"seeds", "episodes", "promise", "certificate", "arms": {arm: means over seeds and
    totals}, "runs": {arm: [per seed]}, "seconds"}. s1_failure_rate: System 1 decisions in runs that did not reach
    their goal / System 1 decisions (all seeds); alone_failure_rate: the same per run."""
    t0 = time.time()
    out = {"seeds": list(seeds), "episodes": episodes, "promise": PROMISE, "certificate": dict(CERTIFICATE),
           "arms": {}, "runs": {}}
    for arm in arms:
        rs = [one(arm, s, episodes) for s in seeds]
        a = {k: sum(r[k] for r in rs) / len(rs) for k in MEAN}
        a.update({k: sum(r[k] for r in rs) for k in SUM})
        a["s1_failure_rate"] = a["s1_failed"] / a["s1_decisions"] if a["s1_decisions"] else 0.0
        a["alone_failure_rate"] = a["alone_failures"] / a["alone_runs"] if a["alone_runs"] else 0.0
        out["arms"][arm] = a
        out["runs"][arm] = [{k: v for k, v in r.items() if k != "episodes"} | {
            "curve": [round(e["s1_share"] or 0.0, 3) for e in r["episodes"]]} for r in rs]
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    r = run()
    cols = ("goals", "all_goals", "s1_share", "s1_share_first5", "s1_share_last5", "s1_failure_rate",
            "alone_failure_rate", "planner_calls_first5", "planner_calls_last5", "simulated_first5", "simulated_last5",
            "skills", "retired")
    print("arm".ljust(12) + "".join(c[:14].rjust(15) for c in cols))
    for arm, a in r["arms"].items():
        print(arm.ljust(12) + "".join(f"{a[c]:15.3f}" for c in cols))
    print(f"{r['seconds']} s; promise {r['promise']}; certificate {r['certificate']}")
    if "--json" in sys.argv:
        print(json.dumps(r, indent=1))
