"""An environment agent (`solvi.Agent`) on a toy crafting world: System 1 acts on what the knowledge predicts, System 2
searches when it cannot, and the same world met again needs fewer slow decisions; then protection vs justified risk.

The world is the toy of benchmarks/knowledge/toy_crafting.py, made smaller: 8 places joined by exits, each with a
resource (tree, stone, iron, water or nothing), and six operators with the environment's own checks — collect_wood (a
tree here), place_table (wood), make_pickaxe (a table here, wood), collect_stone (stone here, a pickaxe),
make_stone_pickaxe (a table here, stone), collect_iron (iron here, a stone pickaxe). Tools do not use materials up.
Six goals: each operator done once. No extra dependencies, no model, a few seconds.

Part 1 — memory. `solvi.Knowledge(vocabulary=...)`: an action model learns from outcomes what each operator needs
(over the conditions in the vocabulary), skills record which action finished each goal, map facts record where each
exit leads. Run 1 on world 7 explores (System 2), run 2 on world 7 walks the confirmed routes (System 1); a new world
(world 8) keeps the rules and re-learns the map.

Part 2 — protection vs justified risk. Another world where the iron lies only across a rope bridge that breaks under
the hero one time in three (the episode ends). An action model of your own (`OutcomeRates`, 20 lines: the outcome
rate of each action in each situation) predicts the bridge "refuse" after a fall, with its rate and the number of
crossings behind it. With protection (the default) the agent never crosses again; with
`RiskBudget(max_risk_per_episode=1.0, min_gain_ratio=1.0, min_support=1)` it crosses when the expected gain outweighs
the estimated risk, within the episode's budget. A written rule stays hard in both: the agenda gate "no bridge without
the stone pickaxe" is never traded.

    uv run python examples/25_environment_agent.py
"""
import random
from collections import defaultdict

import solvi
from solvi.core import Outcome
from solvi.core.knowledge import Prediction, RiskBudget

RESOURCES = ["tree", "tree", "stone", "iron", "water", None, None, "stone"]
OPS = ["collect_wood", "place_table", "make_pickaxe", "collect_stone", "make_stone_pickaxe", "collect_iron"]


class Crafting:
    """The world (an Environment: reset, actions, step). reset(seed): the seed is the world (its map and resources).
    bridge=True: the iron place is reached only over a bridge from the start, which breaks with probability `fall`
    (drawn from its own random stream, `luck`, across episodes)."""

    def __init__(self, bridge=False, fall=1 / 3, luck=0):
        self.bridge, self.fall = bridge, fall
        self.rng = random.Random(luck)

    def reset(self, seed=None):
        rng = random.Random(seed)
        self.places = [f"p{i}" for i in range(8)]
        res = RESOURCES[:]
        rng.shuffle(res)
        self.resource = dict(zip(self.places, res))
        iron = next(p for p in self.places if self.resource[p] == "iron")
        linked = [p for p in self.places if not (self.bridge and p == iron)]
        self.exits = {p: {} for p in self.places}
        order = linked[:]
        rng.shuffle(order)
        for a, b in zip(order, order[1:]):                       # a path through the places: connected
            self._link(a, b)
        for _ in range(3):
            self._link(*rng.sample(linked, 2))
        self.start = order[0]
        if self.bridge:                                          # the iron only across the bridge from the start
            self.exits[iron] = {"exit0": self.start}
        self.s = {"place": self.start, "here": self.resource[self.start], "wood": False, "stone": False,
                  "pickaxe": False, "stone_pickaxe": False, "tables": [], "achieved": []}
        return self._view()

    def _link(self, a, b):
        if a != b and b not in self.exits[a].values():
            self.exits[a][f"exit{len(self.exits[a])}"] = b
            self.exits[b][f"exit{len(self.exits[b])}"] = a

    def _view(self):
        return dict(self.s, tables=list(self.s["tables"]), achieved=list(self.s["achieved"]))

    def actions(self, state):
        acts = sorted(self.exits[state["place"]]) + OPS
        if self.bridge and state["place"] == self.start:
            acts.append("bridge")
        return acts

    def step(self, action):
        s = self.s
        here, table = self.resource[s["place"]], s["place"] in s["tables"]
        if action == "bridge":
            if self.rng.random() < self.fall:
                return Outcome(self._view(), accepted=False, effect="the bridge broke", done=True)
            s["place"] = next(p for p in self.places if self.resource[p] == "iron")
        elif action.startswith("exit"):
            s["place"] = self.exits[s["place"]][action]
        else:
            ok = {"collect_wood": here == "tree", "place_table": s["wood"], "make_pickaxe": table and s["wood"],
                  "collect_stone": here == "stone" and s["pickaxe"], "make_stone_pickaxe": table and s["stone"],
                  "collect_iron": here == "iron" and s["stone_pickaxe"]}[action]
            if not ok:
                return Outcome(self._view(), accepted=False)
            gets = {"collect_wood": "wood", "make_pickaxe": "pickaxe", "collect_stone": "stone",
                    "make_stone_pickaxe": "stone_pickaxe"}
            if action in gets:
                s[gets[action]] = True
            if action == "place_table":
                s["tables"].append(s["place"])
            if action not in s["achieved"]:
                s["achieved"].append(action)
        s["here"] = self.resource[s["place"]]
        return Outcome(self._view(), accepted=True, done=len(s["achieved"]) == len(OPS))


VOCABULARY = {"here": lambda s, a: s["here"], "wood": lambda s, a: s["wood"], "stone": lambda s, a: s["stone"],
              "pickaxe": lambda s, a: s["pickaxe"], "stone_pickaxe": lambda s, a: s["stone_pickaxe"],
              "table_here": lambda s, a: s["place"] in s["tables"]}


def place(state):                                                # the map's keys: the places
    return state["place"]


def goals(km):
    """The six goals: done when the operator has worked once (code over the state, never a model's claim)."""
    for op, needs in zip(OPS, [None] + OPS[:-1]):
        km.goal(op, done=lambda s, op=op: op in s["achieved"], requires=[needs] if needs else [])
    return km


class OutcomeRates:
    """An action model of your own: the outcome rate of each action in each situation (the vocabulary's values).
    Never tried → "unknown"; never refused → "accept"; refused at least once → "refuse" with the refusal rate
    (Laplace) as its risk and the number of tries as its support."""

    def __init__(self, vocabulary, only=None):
        self.vocabulary, self.only = vocabulary, only or {}
        self.n, self.bad = defaultdict(int), defaultdict(int)

    def _k(self, state, action):
        names = self.only.get(action, sorted(self.vocabulary))
        return action, tuple(self.vocabulary[n](state, None) for n in names)

    def observe(self, state, action, args, accepted, effect=None):
        k = self._k(state, action)
        before = self.predict(state, action, args)
        self.n[k] += 1
        self.bad[k] += not accepted
        return before

    def predict(self, state, action, args):
        k = self._k(state, action)
        n, bad = self.n[k], self.bad[k]
        if n == 0:
            return Prediction("unknown", 0.5, 0, "never tried in this situation", action=action)
        if bad == 0:
            return Prediction("accept", 1 / (n + 2), n, f"worked {n} of {n} times", action=action)
        return Prediction("refuse", (bad + 1) / (n + 2), n, f"failed {bad} of {n} times", action=action)

    def fingerprint(self):
        return f"OutcomeRates({sum(self.n.values())} observations)"


def bridge_gain(state, action):
    """What the user knows: the bridge leads to the iron — the one risk worth taking (a refused operator only wastes a
    step: no gain in taking it against a prediction)."""
    return 1.0 if action == "bridge" else 0.0


def part1():
    km = goals(solvi.Knowledge(vocabulary=VOCABULARY))
    agent = solvi.Agent(Crafting(), knowledge=km, key=place)
    runs = [("world 7, run 1", agent.run(seed=7, steps=150)), ("world 7, run 2", agent.run(seed=7, steps=150)),
            ("world 8 (new)", agent.run(seed=8, steps=150))]
    print("Part 1 — the same world met again needs fewer slow decisions")
    for name, ep in runs:
        print(f"  {name}: {ep['steps']} steps, {len(ep['goals_done'])} of 6 goals, System 1 {ep['s1']}, "
              f"System 2 {ep['s2']}, fallback {ep['fallback']}, refused {ep['refused']}")
    rep = agent.replay()
    print(f"  every decision replays: {rep['ok']} of {rep['decisions']}; the knowledge journal verifies: "
          f"{km.store.verify()}")
    print(f"  learned: collect_stone needs {km.report()['actions']['collect_stone']['allowed']['here']} here and "
          f"a pickaxe {km.report()['actions']['collect_stone']['allowed']['pickaxe']}")
    return agent, runs


def part2(episodes=30):
    print(f"Part 2 — protection vs justified risk ({episodes} episodes on a world with a breaking bridge to the iron)")
    out = {}
    for name, risk in (("protect", None), ("risk", RiskBudget(max_risk_per_episode=1.0, min_gain_ratio=1.0,
                                                              min_support=1))):
        km = goals(solvi.Knowledge(actions=OutcomeRates(VOCABULARY, only={"bridge": []})))
        km.agenda.gate("bridge_needs_the_tool", lambda s: s["stone_pickaxe"], blocks=["bridge"])
        agent = solvi.Agent(Crafting(bridge=True), knowledge=km, key=place, risk=risk, gain=bridge_gain)
        eps = [agent.run(seed=3, steps=150) for _ in range(episodes)]
        iron = sum("collect_iron" in e["goals_done"] for e in eps)
        falls = sum(e["done"] and "collect_iron" not in e["goals_done"] for e in eps)
        takes = sum(e["risky_takes"] for e in eps)
        out[name] = {"iron": iron, "falls": falls, "takes": takes, "agent": agent,
                     "goals": sum(len(e["goals_done"]) for e in eps) / episodes}
        print(f"  {name:8s}: iron in {iron} of {episodes} episodes, fell {falls} times, {takes} risky crossings, "
              f"{out[name]['goals']:.2f} goals per episode")
    print("  the gate 'no bridge without the stone pickaxe' held in both: "
          f"{all(not past for past in gate_violations(out))}")
    return out


def gate_violations(out):
    """Did any agent cross the bridge without the stone pickaxe? (read from the stored decisions' facts)"""
    for arm in out.values():
        for d in arm["agent"].decisions:
            o = d.s1.trace.init["options"][int(d.answer)] if d.answer is not None else None
            yield o is not None and o["action"] == "bridge" and o["blocks"]


if __name__ == "__main__":
    part1()
    print()
    part2()
