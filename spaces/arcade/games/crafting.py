"""The toy crafting world of solvi's examples/25_environment_agent.py (vendored), for the "Agent and knowledge (1.0)"
tab: an environment agent (`solvi.Agent`) acts on what its knowledge (`solvi.Knowledge`) predicts — System 1 — and
searches where it cannot — System 2; the same world met again needs fewer slow decisions; then protection vs justified
risk (`RiskBudget`) on a world whose iron lies across a breaking bridge, behind a written gate.

The world: 8 places joined by exits, each with a resource (tree, stone, iron, water or nothing), and six operators
with the environment's own checks — collect_wood (a tree here), place_table (wood), make_pickaxe (a table here, wood),
collect_stone (stone here, a pickaxe), make_stone_pickaxe (a table here, stone), collect_iron (iron here, a stone
pickaxe). Six goals: each operator done once. No model; a run takes a fraction of a second."""
from __future__ import annotations

import random
import warnings
from collections import defaultdict

import solvi
from solvi.core import Outcome
from solvi.core.knowledge import Prediction, RiskBudget

RESOURCES = ["tree", "tree", "stone", "iron", "water", None, None, "stone"]
OPS = ["collect_wood", "place_table", "make_pickaxe", "collect_stone", "make_stone_pickaxe", "collect_iron"]
ICON = {"tree": "🌳", "stone": "🪨", "iron": "⛓️", "water": "💧", None: "·"}


class Crafting:
    """The world (an Environment: reset, actions, step). reset(seed): the seed is the world (its map and resources).
    bridge=True: the iron place is reached only over a bridge from the start, which breaks with probability `fall`
    (its own random stream, `luck`, across episodes). `log` keeps (place, action, accepted, effect) of every step of
    the current episode, for the page."""

    def __init__(self, bridge=False, fall=1 / 3, luck=0):
        self.bridge, self.fall = bridge, fall
        self.rng = random.Random(luck)
        self.log = []

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
        self.log = []
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
        at = self.s["place"]
        out = self._step(action)
        self.log.append((at, action, out.accepted, out.effect))
        return out

    def _step(self, action):
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
    """What the user knows: the bridge leads to the iron — the one risk worth taking."""
    return 1.0 if action == "bridge" else 0.0


# ---------------------------------------------------------------------------------------------------- part 1: memory
def new_agent():
    """A fresh agent with an empty memory: Knowledge over the vocabulary (the built-in ConservativeActionModel)."""
    km = goals(solvi.Knowledge(vocabulary=VOCABULARY))
    return solvi.Agent(Crafting(), knowledge=km, key=place)


def run(agent, seed, steps=150):
    """One episode in world `seed` → (the episode's numbers, the step log [(n, place, action, by, accepted, why)])."""
    before = len(agent.decisions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ep = agent.run(seed=seed, steps=steps)
    rows = []
    for i, (d, (at, action, ok, effect)) in enumerate(zip(agent.decisions[before:], agent.env.log), 1):
        o = d.s1.trace.init["options"][int(d.answer)] if d.answer is not None else {}
        rows.append((i, at, action, d.by, ok, o.get("why") or "", effect))
    return ep, rows


def retraction_demo(agent):
    """A person tells the agent two facts (the second derived from the first), the agent's own guess is offered as a
    fact, then the first is retracted → a dict for the page."""
    km = agent.knowledge
    f1 = km.tell(("world 7", "iron_is_at", "p3"), source="person", by="a guide")
    f2 = km.tell(("world 7", "shortest_way_to_iron", "via p3's exits"), source="person", by="a guide",
                 derived_from=[f1])
    try:
        own = km.tell(("world 7", "iron_is_at", "p6"), source="s1", by="the agent's own guess")
    except Exception as e:  # noqa: BLE001 — the source check may raise instead of refusing
        own = f"refused ({type(e).__name__}: {e})"
    rep_before = km.store.report()
    out = km.retract(f1, why="the guide was wrong", by="a visitor")
    rebuilt = km.store.rebuild(skip=[f1]).fingerprint()
    return {"told": [f1, f2], "own": own, "status": out["status"], "before": rep_before,
            "after": km.store.report(), "rebuilt": rebuilt, "verify": km.store.verify()}


# ---------------------------------------------------------------------------------------------------- part 2: risk
def risk_arms(episodes=10, seed=3):
    """Protection vs justified risk on the bridge world → {arm: numbers}. The written gate 'no bridge without the stone
    pickaxe' is a hard check in both."""
    out = {}
    for name, risk in (("protect", None), ("risk", RiskBudget(max_risk_per_episode=1.0, min_gain_ratio=1.0,
                                                              min_support=1))):
        km = goals(solvi.Knowledge(actions=OutcomeRates(VOCABULARY, only={"bridge": []})))
        km.agenda.gate("bridge_needs_the_tool", lambda s: s["stone_pickaxe"], blocks=["bridge"])
        agent = solvi.Agent(Crafting(bridge=True), knowledge=km, key=place, risk=risk, gain=bridge_gain)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eps = [agent.run(seed=seed, steps=150) for _ in range(episodes)]
        blocked = crossed_blocked = 0
        for d in agent.decisions:
            opts = d.s1.trace.init["options"]
            bridge = [o for o in opts if o["action"] == "bridge"]
            if bridge and bridge[0]["blocks"]:
                blocked += 1
                if d.answer is not None and opts[int(d.answer)]["action"] == "bridge":
                    crossed_blocked += 1
        rep = agent.replay()
        out[name] = {"iron": sum("collect_iron" in e["goals_done"] for e in eps),
                     "falls": sum(e["done"] and "collect_iron" not in e["goals_done"] for e in eps),
                     "takes": sum(e["risky_takes"] for e in eps),
                     "goals": sum(len(e["goals_done"]) for e in eps) / episodes,
                     "gate_blocked": blocked, "gate_broken": crossed_blocked,
                     "replay": rep, "decisions": len(agent.decisions)}
    return out
