"""The knowledge pieces in an environment met again: a toy crafting world with the protocol of the research program's
Crafter runs — not Crafter, and not its numbers.

    uv run python benchmarks/knowledge/toy_crafting.py                 # 10 streams × 20 worlds, ~1 min
    uv run python benchmarks/knowledge/toy_crafting.py --streams 2 --worlds 5      # a quick run
    → benchmarks/knowledge/results/toy_crafting.json

Why a toy: Crafter installs (pip install crafter), but the research agent that played it is ~1,000 lines of
hand-written reflexes (survival at night, fleeing, a planner) that are not part of solvi; porting them would measure that
agent, not the library. This script keeps the protocol and uses only solvi.core.knowledge: ConservativeActionModel over
a Vocabulary, Agenda (goals with done checks, gates from the written rules), FailureMemory, WorldMap over a
KnowledgeStore, the store's journal verified at the end of every stream.

The world (seeded): 8 places joined by exits (a random connected graph); each place holds a resource — tree, stone,
iron, water or nothing. Six operators with the environment's own checks (the "spec" the agenda arms are given):
collect_wood (a tree here), place_table (wood ≥ 2), make_pickaxe (a table here, wood ≥ 1), collect_stone (stone here,
a pickaxe), make_stone_pickaxe (a table here, wood ≥ 1, stone ≥ 1), collect_iron (iron here, a stone pickaxe). Six
goals = the six operators done once (an achievement), 150 steps per episode, a budget of 60 System 2 steps.

The agent: System 1 takes an operator of an open goal when the action model predicts "accept" (and the agenda and the
failure memory allow it), or walks the map to a place with the resource it needs; otherwise System 2 explores — an
operator the model says "unknown" about, or an exit not taken yet — within its budget. Arms:
  none          the action model and the map start empty every episode (nothing carried)
  km            the action model is carried across worlds (rules carried); the map is carried within a world only
  none_agenda   none + the agenda (goals in tech-tree order, gates = the written rules)
  km_agenda     km + the agenda
  km_place      km with an incidental condition in the vocabulary — the place's name, which differs in every world —
                to show what does not transfer (sufficient conditions learned in one world)
  proposer      a proposer that does not learn (like a model asked for the next action with no memory): a random
                operator of an open goal or an exit, no action model
  proposer_fm   the same proposer behind FailureMemory's hard check: a failed operator is not proposed again in the
                same situation for 10 steps (re-drawn); the failure memory in every other arm too
Reported per arm, over the last third of each stream's worlds: achievements per episode, steps to collect_stone (150 =
not reached), System 1 share of steps, failed attempts and wasted steps per 100 steps, actions past an unmet gate,
repeated failures (the same operator failing again in the same situation within 10 steps). And a changed world: 5
episodes on world A, then 5 on world B with the map carried — the first contradicted arrival drops the map (a drift
flag), carried claims refuted in B, System 1 moves on a wrong carried claim before the drop (the move that finds the
change) and after it (none expected: dropped claims are hypotheses, System 1 walks only confirmed ones)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solvi.core.knowledge import (Agenda, ConservativeActionModel, FailureMemory, KnowledgeStore,  # noqa: E402
                                  Vocabulary, WorldMap)

OUT = Path(__file__).resolve().parent / "results"
RESOURCES = ["tree", "tree", "stone", "iron", "water", None, None, "stone"]
OPS = ["collect_wood", "place_table", "make_pickaxe", "collect_stone", "make_stone_pickaxe", "collect_iron"]
NEEDS_HERE = {"collect_wood": "tree", "collect_stone": "stone", "collect_iron": "iron"}
STEPS, S2_BUDGET = 150, 60


# ------------------------------------------------------------------------------------------------ the environment
class World:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.name = f"w{seed}"
        self.places = [f"{self.name}:p{i}" for i in range(8)]
        res = RESOURCES[:]
        rng.shuffle(res)
        self.resource = dict(zip(self.places, res))
        self.exits = {p: {} for p in self.places}
        order = self.places[:]
        rng.shuffle(order)
        for a, b in zip(order, order[1:]):                       # a path through all: connected
            self._link(a, b, rng)
        for _ in range(4):
            a, b = rng.sample(self.places, 2)
            self._link(a, b, rng)
        self.start = order[0]

    def _link(self, a, b, rng):
        if b in self.exits[a].values():
            return
        self.exits[a][f"exit{len(self.exits[a])}"] = b
        self.exits[b][f"exit{len(self.exits[b])}"] = a

    def reset(self):
        return {"place": self.start, "wood": 0, "stone": 0, "iron": 0, "pickaxe": False, "stone_pickaxe": False,
                "tables": [], "here": self.resource[self.start]}

    def step(self, s, action):
        """→ (accepted, new state)."""
        s = dict(s, tables=list(s["tables"]))
        here, table = self.resource[s["place"]], s["place"] in s["tables"]
        if action.startswith("exit"):
            if action not in self.exits[s["place"]]:
                return False, s
            s["place"] = self.exits[s["place"]][action]
            s["here"] = self.resource[s["place"]]
            return True, s
        ok = {"collect_wood": here == "tree", "place_table": s["wood"] >= 2,
              "make_pickaxe": table and s["wood"] >= 1, "collect_stone": here == "stone" and s["pickaxe"],
              "make_stone_pickaxe": table and s["wood"] >= 1 and s["stone"] >= 1,
              "collect_iron": here == "iron" and s["stone_pickaxe"]}[action]
        if not ok:
            return False, s
        if action == "collect_wood":
            s["wood"] += 1
        elif action == "place_table":
            s["wood"] -= 2
            s["tables"].append(s["place"])
        elif action == "make_pickaxe":
            s["wood"] -= 1
            s["pickaxe"] = True
        elif action == "collect_stone":
            s["stone"] += 1
        elif action == "make_stone_pickaxe":
            s["wood"] -= 1
            s["stone"] -= 1
            s["stone_pickaxe"] = True
        else:
            s["iron"] += 1
        return True, s


def vocabulary(incidental=False):
    preds = {"here": lambda s, a: s["here"], "wood": lambda s, a: min(s["wood"], 2), "stone": lambda s, a: min(s["stone"], 1),
             "pickaxe": lambda s, a: s["pickaxe"], "stone_pickaxe": lambda s, a: s["stone_pickaxe"],
             "table_here": lambda s, a: s["place"] in s["tables"]}
    if incidental:
        preds["place"] = lambda s, a: s["place"]
    return Vocabulary(preds)


def spec_agenda(store):
    """The written rules as agenda goals and gates."""
    ag = Agenda(store)
    done = {op: (lambda op: lambda s: op in s.get("_achieved", ()))(op) for op in OPS}
    ag.goal("collect_wood", done=done["collect_wood"])
    ag.goal("place_table", done=done["place_table"], requires=["collect_wood"])
    ag.goal("make_pickaxe", done=done["make_pickaxe"], requires=["place_table"])
    ag.goal("collect_stone", done=done["collect_stone"], requires=["make_pickaxe"])
    ag.goal("make_stone_pickaxe", done=done["make_stone_pickaxe"], requires=["collect_stone"])
    ag.goal("collect_iron", done=done["collect_iron"], requires=["make_stone_pickaxe"])
    ag.gate("tree_here", lambda s: s["here"] == "tree", blocks=["collect_wood"])
    ag.gate("two_wood", lambda s: s["wood"] >= 2, blocks=["place_table"])
    ag.gate("table_here", lambda s: s["place"] in s["tables"], blocks=["make_pickaxe", "make_stone_pickaxe"])
    ag.gate("wood_for_tools", lambda s: s["wood"] >= 1, blocks=["make_pickaxe", "make_stone_pickaxe"])
    ag.gate("stone_here_and_pickaxe", lambda s: s["here"] == "stone" and s["pickaxe"], blocks=["collect_stone"])
    ag.gate("stone_for_tool", lambda s: s["stone"] >= 1, blocks=["make_stone_pickaxe"])
    ag.gate("iron_here_and_tool", lambda s: s["here"] == "iron" and s["stone_pickaxe"], blocks=["collect_iron"])
    return ag


def sig(s):
    return json.dumps({k: s[k] for k in ("here", "wood", "stone", "pickaxe", "stone_pickaxe")} |
                      {"table_here": s["place"] in s["tables"]}, sort_keys=True)


# ------------------------------------------------------------------------------------------------ the agent
class Agent:
    def __init__(self, arm, store, rng):
        self.arm, self.store, self.rng = arm, store, rng
        self.model = ConservativeActionModel(vocabulary(arm == "km_place"), store=store)
        self.fm = None if arm == "proposer" else FailureMemory(window=10, max_blocked=6, min_open=1, store=store)

    def episode(self, world, wmap):
        """One episode → metrics."""
        if self.arm in ("none", "none_agenda"):                 # nothing carried
            self.model = ConservativeActionModel(vocabulary(), store=self.store)
        agenda = spec_agenda(None) if self.arm.endswith("agenda") else None
        s, achieved = world.reset(), set()
        m = {"achievements": 0, "steps_to_stone": STEPS, "s1": 0, "s2": 0, "failed": 0, "wasted": 0, "past_gate": 0,
             "repeat_failures": 0, "refuted_moves": 0, "refuted_moves_after_drop": 0}
        last_fail = {}
        s2_left = S2_BUDGET
        for t in range(STEPS):
            s["_achieved"] = sorted(achieved)
            if agenda is not None:
                agenda.update(s)
            wmap.visit(s["place"], step=t, resource=s["here"])
            for ex in world.exits[s["place"]]:
                wmap.see(s["place"], ex, step=t)
            goals = agenda.open(s) if agenda is not None else [g for g in OPS if g not in achieved]
            if agenda is not None:                               # ordered goals: also the ones waiting on a gate
                goals = goals + [g for g in agenda.blocked(s) if g not in achieved and g not in goals]
            if self.arm.startswith("proposer"):
                action, system = self.propose(s, goals, world), "s2"
            else:
                action, system = self.choose(s, goals, agenda, wmap, s2_left)
            if action is None:
                action, system = self.rng.choice(sorted(world.exits[s["place"]])), "s2"
            if system == "s2":
                s2_left -= 1
            m[system] += 1
            if agenda is not None and not action.startswith("exit") and not agenda.allows(s, action)[0]:
                m["past_gate"] += 1
            before = s
            ok, s = world.step(s, action)
            if action.startswith("exit"):
                dropped = any(r["op"] == "drop" for r in wmap.journal)
                if wmap.arrive(before["place"], action, s["place"], step=t) and system == "s1":
                    m["refuted_moves_after_drop" if dropped else "refuted_moves"] += 1
                continue
            self.model.observe(before, action, {}, ok, None)
            if not ok:
                m["failed"] += 1
                m["wasted"] += 1
                key = (sig(before), action)
                if key in last_fail and t - last_fail[key] <= 10:
                    m["repeat_failures"] += 1
                last_fail[key] = t
                if self.fm is not None:
                    self.fm.failed([sig(before), action], why="refused")
            elif action in achieved:
                m["wasted"] += 1 if action != "collect_wood" or s["wood"] > 3 else 0
            else:
                achieved.add(action)
                if action == "collect_stone":
                    m["steps_to_stone"] = t + 1
            if self.fm is not None:
                self.fm.step()
        m["achievements"] = len(achieved)
        return m

    def propose(self, s, goals, world):
        """A proposer without memory: a random operator of an open goal or a random exit — re-drawn (up to 5 times)
        when the failure memory's hard check refuses it."""
        options = goals + sorted(world.exits[s["place"]])
        for _ in range(5):
            a = self.rng.choice(options)
            if self.fm is None or a.startswith("exit") or self.fm.check([sig(s), a]).verdict == "accept":
                return a
        return self.rng.choice(sorted(world.exits[s["place"]]))

    def ok_to_try(self, s, op, agenda):
        if agenda is not None and not agenda.allows(s, op)[0]:
            return False
        return self.fm is None or self.fm.check([sig(s), op]).verdict == "accept"

    def choose(self, s, goals, agenda, wmap, s2_left):
        """→ (action, "s1" | "s2")."""
        for g in goals:                                          # System 1: an operator of an open goal, predicted to work
            p = self.model.predict(s, g, {})
            if p.verdict == "accept" and self.ok_to_try(s, g, agenda):
                return g, "s1"
        for op, low in (("collect_wood", s["wood"] < 3), ("collect_stone", s["stone"] < 1)):   # the stock reflex
            v = self.model.predict(s, op, {}).verdict
            if low and v != "refuse" and self.ok_to_try(s, op, agenda) and (v == "accept" or s2_left > 0):
                return op, "s1" if v == "accept" else "s2"
        want = "tree" if s["wood"] < 3 else ("stone" if s["pickaxe"] and s["stone"] < 1 else
                                              ("iron" if s["stone_pickaxe"] else None))
        if want is not None and s["here"] != want:               # System 1: walk the map to what is needed
            targets = {st for st, info in wmap.states.items() if info["facts"].get("resource") == want}
            if targets:
                a = wmap.next(s["place"], targets, confirmed_only=True)
                if a is not None:
                    return a, "s1"
        if s2_left <= 0:
            return None, "s2"
        for g in goals:                                          # System 2: an experiment the model cannot predict
            if self.model.predict(s, g, {}).verdict == "unknown" and self.ok_to_try(s, g, agenda):
                return g, "s2"
        a = wmap.explore(s["place"])
        return (a, "s2") if a is not None else (None, "s2")


# ------------------------------------------------------------------------------------------------ runs
def run_stream(arm, stream, worlds):
    rng = random.Random(1000 * stream + 7)
    store = KnowledgeStore()
    agent = Agent(arm, store, rng)
    rows = []
    for w in range(worlds):
        world = World(10_000 * stream + w)
        wmap = WorldMap(knowledge=store, scope={"map": world.name})          # a new world: a new map (rules carried)
        rows.append(agent.episode(world, wmap))
    return rows, store.verify()


def changed_world(stream):
    rng = random.Random(stream)
    store = KnowledgeStore()
    agent = Agent("km", store, rng)
    a, b = World(50_000 + stream), World(60_000 + stream)
    b.places = a.places[:]                                   # the same place names, rewired and restocked
    b2 = World(60_000 + stream)
    mapping = dict(zip(b2.places, a.places))
    b.resource = {mapping[p]: r for p, r in b2.resource.items()}
    b.exits = {mapping[p]: {e: mapping[d] for e, d in ex.items()} for p, ex in b2.exits.items()}
    b.start = a.start
    wmap = WorldMap(knowledge=store, scope={"map": "carried"})
    out = {"A": [], "B": [], "dropped_in_B": False, "refuted_in_B": 0, "s1_refuted_moves_B": 0, "after_drop": 0}
    for ep in range(5):
        out["A"].append(agent.episode(a, wmap)["achievements"])
    refuted_before = sum(1 for r in wmap.journal if r["op"] == "refute")
    for ep in range(5):
        if ep == 0:
            orig = wmap.arrive

            def arrive(st, act, to, step=None, _orig=orig):
                refuted = _orig(st, act, to, step)
                if refuted and not out["dropped_in_B"]:
                    out["dropped_in_B"] = True                   # the first contradicted arrival: a drift flag
                    out["dropped_claims"] = wmap.drop(why="an arrival contradicted the carried map")
                    for info in wmap.states.values():
                        info["facts"].pop("resource", None)      # what was seen where: re-learned in the new world
                return refuted
            wmap.arrive = arrive
        m = agent.episode(b, wmap)
        out["B"].append(m["achievements"])
        out["s1_refuted_moves_B"] += m["refuted_moves"]
        out["after_drop"] += m["refuted_moves_after_drop"]
    out["refuted_in_B"] = sum(1 for r in wmap.journal if r["op"] == "refute") - refuted_before
    out["verify"] = store.verify()
    return out


def mean(v):
    return sum(v) / len(v) if v else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=10)
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--tag", default="toy_crafting")
    a = ap.parse_args(argv)
    t0, last = time.time(), time.time()
    arms = ["none", "km", "none_agenda", "km_agenda", "km_place", "proposer", "proposer_fm"]
    res, verified = {}, True
    for arm in arms:
        per = []
        for st in range(a.streams):
            rows, ok = run_stream(arm, st, a.worlds)
            verified &= ok
            per.append(rows)
            if time.time() - last > 30:
                print(f"{time.strftime('%H:%M:%S')} {arm} stream {st + 1}/{a.streams}", flush=True)
                last = time.time()
        third = a.worlds - a.worlds // 3
        last_third = [r for rows in per for r in rows[third:]]
        first = [rows[0] for rows in per]
        steps = sum(r["s1"] + r["s2"] for r in last_third)
        res[arm] = {"achievements_first_world": mean([r["achievements"] for r in first]),
                    "achievements_last_third": mean([r["achievements"] for r in last_third]),
                    "steps_to_stone_last_third": mean([r["steps_to_stone"] for r in last_third]),
                    "s1_share_last_third": sum(r["s1"] for r in last_third) / steps,
                    "failed_per_100_steps": 100 * sum(r["failed"] for r in last_third) / steps,
                    "wasted_per_100_steps": 100 * sum(r["wasted"] for r in last_third) / steps,
                    "actions_past_a_gate": sum(r["past_gate"] for rows in per for r in rows),
                    "repeat_failures": sum(r["repeat_failures"] for rows in per for r in rows),
                    "repeat_failures_per_100_steps": 100 * sum(r["repeat_failures"] for r in last_third) / steps}
        print(f"{time.strftime('%H:%M:%S')} {arm}: {json.dumps({k: round(v, 3) for k, v in res[arm].items()})}",
              flush=True)
    changed = [changed_world(st) for st in range(a.streams)]
    res["changed_world"] = {"achievements_A_first": mean([c["A"][0] for c in changed]),
                            "achievements_A_last": mean([c["A"][-1] for c in changed]),
                            "achievements_B_first": mean([c["B"][0] for c in changed]),
                            "achievements_B_last": mean([c["B"][-1] for c in changed]),
                            "map_dropped_in_B": sum(c["dropped_in_B"] for c in changed),
                            "claims_dropped": mean([c.get("dropped_claims", 0) for c in changed]),
                            "claims_refuted_in_B": mean([c["refuted_in_B"] for c in changed]),
                            "s1_moves_on_wrong_claims_before_drop_B": sum(c["s1_refuted_moves_B"] for c in changed),
                            "s1_moves_on_wrong_claims_after_drop_B": sum(c["after_drop"] for c in changed),
                            "streams": a.streams, "verify": all(c["verify"] for c in changed)}
    res["journals_verify"] = verified and res["changed_world"]["verify"]
    res["config"] = {"streams": a.streams, "worlds": a.worlds, "steps": STEPS, "s2_budget": S2_BUDGET,
                     "seconds": round(time.time() - t0, 1)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["changed_world"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
