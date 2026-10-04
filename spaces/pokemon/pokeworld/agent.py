"""The player: System 1 acts fast from what practice compiled, System 2 deliberates over the world map when System 1 is
unsure or surprised, and consolidation turns what System 2 found into System 1's routes.

One decision = "which exit do I take here?". Its input is a plain view (`view`), so every decision replays:

    {"view": {here, stage, goal, targets, s1: the route System 1 remembers, surprise, plans: System 2's candidates},
     "on_offer": "Door at (5,5)\\nLeave north\\n..."}               # the exits on the screen, one per line

System 1 (`fast`) — a catalog of two rules, no search: take the exit of the remembered route to the objective (a
route compiled by consolidation), or the only exit there is. It stands back (abstains) when it remembers no route,
when the remembered exit is not on offer here, or when the last step surprised it (it led somewhere else than it
expected, or did not lead anywhere).

System 2 (`slow`) — a search (`solvi.core.slow.search` through `solvi.core.dispatch.SearchPath`) over plans the world map gives: the
known way to the objective when the map has one, otherwise every unexplored exit within reach, scored by expected
value: P(the place behind it is of a kind the goal needs) × the goal's direction × places the goal names, discounted
by the steps to get there. A hard check keeps only plans whose first step is on offer here. Optionally an LLM reads the
goal and the unexplored exits and names one; that plan gets a bonus (off by default; `llm=`).

The dispatcher (`solvi.core.dispatch.Dispatcher`) asks System 1 first and System 2 when System 1 abstains; every decision
is one hash-chained record in a store and replays without the game.

Memory: the world map (`solvi.core.knowledge.worldmap.WorldMap`, semantic: every exit seen and every arrival, hash-chained journal),
System 1's routes (procedural: a table compiled from the map's confirmed claims), and the episode notes (events,
surprises). Consolidation runs after each objective: it compiles a route to every place a goal has named and the player
has reached, from confirmed claims only, and records what changed."""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

from solvi import Answer, Catalog, Question, System
from solvi.core.dispatch import Dispatcher, SearchPath
from solvi.core.knowledge.worldmap import WorldMap

from .world import place_map, place_type

RADIUS = 4              # System 2 looks at unexplored exits this many known steps away (all of them when none is)
MAX_MOVES = 400         # moves per objective before a run gives up on it

# P(kind of place behind an exit | kind of place here, door or edge) — a prior from what places in towns, on routes and
# in caves usually connect to, not learned from the recording
BEHIND = {
    ("town", "door"): {"building": 0.45, "center": 0.12, "mart": 0.1, "gym": 0.15, "gate": 0.08, "cave": 0.05,
                       "ship": 0.05},
    ("town", "edge"): {"route": 0.85, "town": 0.15},
    ("route", "edge"): {"town": 0.5, "route": 0.4, "cave": 0.1},
    ("route", "door"): {"gate": 0.35, "cave": 0.25, "building": 0.2, "forest": 0.1, "center": 0.1},
    ("gate", "door"): {"route": 0.6, "town": 0.2, "forest": 0.2},
    ("forest", "door"): {"gate": 0.8, "forest": 0.2},
    ("cave", "door"): {"cave": 0.75, "route": 0.25},
    ("ship", "door"): {"ship": 0.8, "town": 0.2},
}
INDOORS = {"town": 0.6, "route": 0.2, "building": 0.2}
DIRS = ("north", "south", "east", "west")


def norm(text):
    """A place name as a goal writes it → as the map names it: "Mt. Moon" → "MT_MOON", "S.S. Anne" → "SS_ANNE"."""
    return re.sub(r"[^A-Z0-9]+", "_", text.upper().replace(".", "")).strip("_")


def named_in(goal):
    """Place names a goal mentions ("Route 2", "Viridian Forest", "Pewter City") in map spelling."""
    pat = r"((?:[A-Z][a-z]*\.? ?)+(?:City|Town|Forest|Island|Tunnel|Path|Moon|House|Gym|Anne)|Route \d+|Mt\. [A-Z][a-z]+)"
    return sorted({norm(m) for m in re.findall(pat, goal)})


# ---------------------------------------------------------------------------------------------------- System 1
fast = Catalog()


@fast.fn
def exits(on_offer: str) -> list:
    return [x for x in on_offer.split("\n") if x]


@fast.fn
def remembered(view: dict):
    """The route System 1 remembers from here to the objective (compiled by consolidation), or None."""
    return view.get("s1")


@fast.fn
def stands_back(view: dict, remembered, exits: list) -> str:
    """Why System 1 does not answer alone ("" when it does)."""
    if view.get("surprise"):
        return "surprised: " + view["surprise"]
    if remembered and remembered["exit"] not in exits:
        return f"the remembered way ({remembered['exit']}) is not on offer here"
    if remembered and remembered["exit"] in view.get("blocked_here", []):
        return f"the remembered way ({remembered['exit']}) did not lead anywhere at this point of the story"
    if not remembered and len(exits) != 1:
        return "no remembered route to the objective from here"
    return ""


@fast.rule("exit")
def exit_fast(remembered, exits: list, stands_back: str):
    if stands_back:
        return None                               # System 1 abstains: the dispatcher wakes System 2
    return remembered["exit"] if remembered else exits[0]


def system1():
    return System(fast, [Question("exit", "Which exit to take?", Answer.span(source="on_offer"))])


# ---------------------------------------------------------------------------------------------------- System 2
def kind_of_exit(key):
    return "edge" if key.startswith("Leave ") else "door"


def behind(here_type, key):
    """P(kind of place behind this exit)."""
    return BEHIND.get((here_type, kind_of_exit(key)), INDOORS)


def slow_catalog(llm=None):
    slow = Catalog()

    @slow.fn
    def wanted(view: dict) -> list:
        """The kinds of place the goal needs and the player has not seen yet."""
        need = {place_type(t) for t in view["targets"]}
        need |= {place_type(n) for n in named_in(view["goal"]) if n not in view.get("named_seen", [])}
        return sorted(need)

    @slow.fn
    def directions(view: dict) -> list:
        return [d for d in DIRS if re.search(r"\b" + d + r"\b", view["goal"].lower())]

    if llm is not None:                             # an LLM reads the goal and the unexplored exits and names one
        slow.fn(llm.decision("hint_llm", "Which of these unexplored exits (place | exit) is most likely on the way to "
                                         "the goal? Answer with one of the lines after the goal, as written.", "frontier",
                             kind="span"), provides="hint")

    @slow.fn(provides="hint")
    def no_hint(frontier: str) -> str:
        return ""                                   # no LLM, or it did not answer: no hint

    @slow.fn
    def value(plan: dict, wanted: list, directions: list, hint, view: dict) -> float:
        """Expected value of a plan: a known way to the objective beats any exploration; an unexplored exit scores
        P(a wanted kind behind it) × direction × named place, discounted by the steps to reach it."""
        if plan["kind"] == "route":
            return 10.0 / (1 + plan["hops"])
        p = sum(behind(place_type(plan["place"]), plan["exit"]).get(t, 0.0) for t in wanted)
        v = 0.15 + p
        if any(d in plan["exit"].lower() for d in directions):
            v *= 1.3
        if any(norm(plan["place"]).startswith(n) for n in named_in(view["goal"])):
            v *= 1.2
        if plan.get("blocked_before"):
            v *= 0.5
        h = str(getattr(hint, "value", hint) or "")
        if h and h.strip() == f"{plan['place']} | {plan['exit']}":
            v *= 1.5
        return round(v / (1 + 0.6 * plan["hops"]), 6)

    @slow.check(hard=True)
    def first_step_on_offer(plan: dict, on_offer: str) -> bool:
        return plan["first"] in on_offer.split("\n")

    @slow.rule("exit")
    def exit_slow(plan: dict, first_step_on_offer: bool):
        return plan["first"]

    return slow


def system2(llm=None):
    return System(slow_catalog(llm), [Question("exit", "Which exit to take?", Answer.span(source="on_offer"),
                                               requires=["value"])])


def slow_path(sys2):
    return SearchPath(sys2, space=lambda facts: list(facts["view"]["plans"]), into="plan", search={"objective": "value"})


def dispatcher(storage=None, llm=None):
    """System 1 first; System 2 (the search) when System 1 abstains; a person when neither answers."""
    return Dispatcher(system1(), slow_path(system2(llm)), question="exit", storage=storage, store_responses=False)


# ---------------------------------------------------------------------------------------------------- memory
@dataclass
class Memory:
    """What the player keeps between objectives and between runs."""
    map: WorldMap = field(default_factory=WorldMap)             # semantic: the world map
    routes: dict = field(default_factory=dict)                  # procedural: {target place: {place: [exit, to, hops]}}
    goals: set = field(default_factory=set)                     # places goals have named and the player has reached
    notes: list = field(default_factory=list)                   # episodic: events and surprises
    consolidations: list = field(default_factory=list)

    def s1_route(self, here, targets):
        """System 1's lookup: the remembered exit from here to the nearest target → {"exit", "to", "hops", "target"}."""
        best = None
        for t in targets:
            r = self.routes.get(t, {}).get(here)
            if r and (best is None or r[2] < best["hops"]):
                best = {"exit": r[0], "to": r[1], "hops": r[2], "target": t}
        return best

    def consolidate(self, step, after):
        """Compile System 1's routes from the map's confirmed claims: a route to every place a goal has named and the
        player has reached. → what changed {"added", "changed", "removed", "routes"}."""
        new = {}
        for t in sorted(self.goals):
            d = self.map.distances({t}, confirmed_only=True)
            new[t] = {p: [a, self.map.edges[(p, a)]["to"], c] for p, (c, a) in sorted(d.items()) if a is not None}
        old = {(t, p): tuple(v[:2]) for t, rs in self.routes.items() for p, v in rs.items()}
        cur = {(t, p): tuple(v[:2]) for t, rs in new.items() for p, v in rs.items()}
        rec = {"after": after, "step": step, "map_head": self.map.journal[-1]["hash"] if self.map.journal else "",
               "added": sum(k not in old for k in cur), "changed": sum(k in old and old[k] != v for k, v in cur.items()),
               "removed": sum(k not in cur for k in old), "routes": len(cur)}
        self.routes = new
        self.consolidations.append(rec)
        return rec


def blocked_at(mem, place, stage):
    """The exits tried in `place` at this stage of the story (or later) that did not lead anywhere (episodic)."""
    b = mem.map.states.get(place, {}).get("facts", {}).get("blocked") or {}
    return sorted(k for k, st in b.items() if st >= stage)


def plans(mem, here, targets, stage):
    """System 2's candidates from the world map: the known way to a target, else the unexplored exits within reach."""
    m = mem.map
    adj = {}
    for (s, a), e in sorted(m.edges.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if a in blocked_at(mem, s, stage):
            continue
        if e["status"] == "confirmed" and e["to"] is not None and e["to"] != s:
            adj.setdefault(s, []).append((a, e["to"]))
    dist, first = {here: 0}, {here: None}
    q = deque([here])
    while q:
        s = q.popleft()
        for a, t in adj.get(s, ()):
            if t not in dist:
                dist[t], first[t] = dist[s] + 1, first[s] or a
                q.append(t)
    reach = [t for t in targets if t in dist and t != here]
    if reach:
        t = min(reach, key=lambda x: (dist[x], x))
        return [{"kind": "route", "place": t, "exit": first[t], "first": first[t], "hops": dist[t]}]
    out = []
    for (s, a), e in sorted(m.edges.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if e["taken"] or s not in dist:
            continue
        blocked = (m.states.get(s, {}).get("facts", {}).get("blocked") or {}).get(a)
        if blocked is not None and blocked >= stage:
            continue                                  # tried at this stage of the story and it did not lead anywhere
        out.append({"kind": "explore", "place": s, "exit": a, "first": a if s == here else first[s], "hops": dist[s],
                    "blocked_before": blocked is not None})
    near = [p for p in out if p["hops"] <= RADIUS]
    return near or out


# ---------------------------------------------------------------------------------------------------- a run
@dataclass
class Move:
    """One decision as the replay viewer shows it."""
    n: int
    run: str
    objective: str
    stage: int
    here: str
    exit: str | None
    by: str
    why: str
    to: str | None
    event: str | None = None
    surprise: str | None = None
    ms: float = 0.0
    plans: int = 0

    def to_dict(self):
        return dict(vars(self))


def play(world, mem, run, store=None, llm=None, story=None, log=None):
    """Play the story once with memory `mem` (carried in and out). store: a TraceStorage (or path) for the dispatch
    records. → (moves, per-objective results, the dispatcher)."""
    d = dispatcher(storage=store, llm=llm)
    here, fired, moves, results = world.start, set(), [], []
    surprise = None
    n = 0

    def arrive_at(place, step):
        sv = world.services(place)
        mem.map.visit(place, step, **({"services": sv} if sv else {}))

    arrive_at(here, f"{run}#0")
    for ob in story or world.story:
        stage, targets = ob["stage"], world.targets(ob["targets"])
        t0, k0, done = time.perf_counter(), len(moves), False
        while not done:
            if place_map(here) in ob["targets"]:
                done = True
                break
            if len(moves) - k0 >= MAX_MOVES:
                break
            n += 1
            step = f"{run}#{n}"
            offer = world.on_offer(here, stage)
            for k in offer:
                mem.map.see(here, k, step=step)
            s1 = mem.s1_route(here, targets)
            ps = plans(mem, here, targets, stage)
            view = {"here": here, "stage": stage, "goal": ob["goal"], "targets": targets, "s1": s1,
                    "surprise": surprise, "blocked_here": blocked_at(mem, here, stage), "plans": ps,
                    "named_seen": [w for w in named_in(ob["goal"])
                                   if any(norm(str(x)).startswith(w) for x in mem.map.states)]}
            frontier = "\n".join([f"Goal: {ob['goal']}"] + [f"{p['place']} | {p['exit']}" for p in ps
                                                            if p["kind"] == "explore"])
            res = d.ask({"view": view, "on_offer": "\n".join(offer), "frontier": frontier})
            ans = getattr(res.answer, "value", res.answer)
            why = res.reasons[-1] if res.reasons else ""
            if res.by == "s1":
                why = "remembered route" if s1 else "the only way on"
            elif res.s1 is not None:
                why = str(res.s1.values.get("stands_back", "") or why)
            if ans is None:
                moves.append(Move(n, run, ob["id"], stage, here, None, res.by, why, None, ms=res.cost["total"].ms,
                                  plans=len(ps)))
                break                                  # nobody could choose: a person would have to (not in the demo)
            st = world.take(here, ans, stage, fired)
            surprise = None
            expect = s1["to"] if (res.by == "s1" and s1) else None
            if st.event is not None:
                mem.notes.append({"step": step, "event": st.event["say"]})
                if expect and expect != st.to:
                    surprise = f"{here}: {ans} was expected to lead to {expect}; {st.event['say']}"
                if st.event.get("done") == ob["id"]:
                    done = True
            elif st.blocked:
                facts = dict(mem.map.states.get(here, {}).get("facts", {}).get("blocked") or {})
                facts[ans] = stage
                mem.map.visit(here, step, blocked=facts)
                if expect:
                    surprise = f"{here}: {ans} did not lead anywhere"
            else:
                refuted = mem.map.arrive(here, ans, st.to, step=step)
                if refuted or (expect and expect != st.to):
                    surprise = f"{here}: {ans} led to {st.to}, not {expect or 'where the map said'}"
            if surprise:
                mem.notes.append({"step": step, "surprise": surprise})
            moves.append(Move(n, run, ob["id"], stage, here, ans, res.by, why, st.to,
                              event=st.event["say"] if st.event else None, surprise=surprise, ms=res.cost["total"].ms,
                              plans=len(ps)))
            if not st.blocked:
                arrive_at(st.to, step)
            here = st.to
        mem.goals |= {t for t in mem.map.states if any(place_map(t) in o["targets"] for o in world.story
                                                         if o["stage"] <= stage)}
        rec = mem.consolidate(f"{run}#{n}", ob["id"])
        mine = moves[k0:]
        results.append({"objective": ob["id"], "stage": stage, "done": done, "moves": len(mine),
                        "s1": sum(m.by == "s1" for m in mine), "s2": sum(m.by == "s2" for m in mine),
                        "human": sum(m.by == "human" for m in mine), "ms": round((time.perf_counter() - t0) * 1000, 1),
                        "consolidation": rec})
        if log is not None:
            log(results[-1])
    return moves, results, d
