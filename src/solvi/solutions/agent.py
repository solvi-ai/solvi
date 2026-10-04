"""The environment agent: System 1 acts on what the knowledge predicts, System 2 searches when it cannot, hard checks
hold in both, and what the environment shows is written back — so an environment met again needs fewer slow decisions.

    import solvi
    from solvi.core.knowledge import RiskBudget

    km = solvi.Knowledge(vocabulary={"here": lambda s, a: s["here"], "wood": lambda s, a: s["wood"] > 0})
    km.goal("wood", done=lambda s: s["wood"] > 0)
    km.goal("table", done=lambda s: s["table"], requires=["wood"])
    agent = solvi.Agent(env, knowledge=km)                       # protection: a predicted refusal is never taken
    agent.run(seed=1, steps=150)                                 # one episode → its numbers
    agent.run(seed=1, steps=150)                                 # the same world again: fewer slow decisions
    agent.report(); agent.replay()                               # what it did; every decision re-checked
    solvi.Agent(env, knowledge=km, risk=RiskBudget(max_risk_per_episode=1.0))   # justified risk, within a budget

`env` is a solvi.core.Environment (`reset(seed)`, `actions(state)`, `step(action) → Outcome`). Every step is one
decision of a Dispatcher (`agent.dispatcher`) over System 1 (`agent.system`), a solvi System with one question,
"action" — which of the offered actions — answered over the given facts of the step:

  options       one row per action the environment offers: the action model's prediction (verdict accept / refuse /
                unknown, risk, support, reason, hard), the risk policy's choice on it (take / avoid / ask_s2, why), the
                hard blocks (agenda gates, a hard refusal, the failure memory), the open goal it advances and how
                ("skill": the goal was done right after it before; "route": the first step of a confirmed route to
                where a skill worked), the expected gain, System 1's and System 2's order;
  goals_open    the agenda's open goals; at: the state's key; knowledge: the store's journal position and hash;
  s2_left       what is left of System 2's per-episode budget (when `budget=` is a number).

System 1 takes the first option that advances an open goal and that the risk policy takes (with the default Protect:
the action model predicts "accept"); its hard checks — `not_past_a_gate`, `not_avoided` (a hard refusal is never
traded; a refusal the policy avoids), `not_a_recent_failure` — are catalog checks over the chosen option. When it has
no such option it abstains and the dispatcher hands the step to System 2: a SearchPath over the offered actions in an
exploration order (an open goal's action first, then actions never tried at this key, then the first step towards the
nearest key with untried actions, ...) through the same hard checks — an action the model is unsure of ("unknown") may
be tried there, an avoided one never. `s2=` replaces it with a SlowPath of your own (it answers "action" with the index
of an option, from the same facts). When System 2 has nothing (its budget is used up, every candidate is blocked) the
agent takes a reproducible pseudo-random option the hard checks allow ("fallback"), or stops the episode when there is
none.

What is written back (Knowledge.observe): the action model learns from the outcome, the failure memory blocks a
refused action at this key for a while, the agenda's done checks run on the new state, a goal done right after an
action makes a skill, and map facts ("leads_to", "accepts") are written on the episode's map. Knowledge carried across
episodes: rules (the action model, skills) are carried everywhere; map facts are scoped to the map an episode plays on
(`run(seed, steps, map=)`, default: the seed — the same seed, the same world) and the first contradiction of a carried
map fact drops the carried map (a flag: its facts become hints until confirmed again).

Risk. `risk=None` is Protect (the verdict is final). `risk=RiskBudget(...)` takes a refused action when its expected
gain (`gain=`: a function (state, action) → number; default 1.0 for an action that advances an open goal, else 0)
outweighs its estimated risk within a per-episode budget; a hard prediction, a gate and the failure memory are never
traded. Every risky take is decided again for real (the budget is charged) and counted.

Every decision is stored (`storage=`: a path or a TraceStorage, hash-chained "dispatch" records; else kept in memory)
and replayable without the environment: `replay()` re-checks System 1's trace, the dispatch, System 2's search record
and the answer from the recorded facts.

What it does not promise: it learns what the environment shows over the vocabulary you give; rules learned in one
world can be too cautious in another (sufficient conditions do not transfer); a learned memory on top of written gates
added nothing in the research behind it; justified risk reduces the cost of protection, not to parity with an agent
without knowledge. See the guide's "Using solvi: agents and knowledge"."""
from __future__ import annotations

import hashlib
import json
from collections import deque

from ..core.catalog import Answer, Catalog, Question
from .knowledge import action_key, split_action

QUESTION = "action"


# ------------------------------------------------------------------------------------------------ the catalog parts
def _row(pick, options):
    if pick is None:
        return None
    i = int(pick)
    return options[i] if 0 <= i < len(options) else None


def s1_pick(options):
    """System 1's choice: walking the options that advance an open goal in System 1's order, the first one the risk
    policy takes and no hard block stops — or None (System 1 abstains: System 2's turn) when there is none, or when a
    goal's own skill comes first and the policy sends it to System 2 (the action model is unsure of it here)."""
    for o in sorted((o for o in options if o.get("s1") is not None), key=lambda o: o["s1"]):
        if o.get("blocks"):
            continue
        if o.get("choice") == "take":
            return o["i"]
        if o.get("via") == "skill" and o.get("choice") == "ask_s2":
            return None
    return None


def not_past_a_gate(pick, options):
    """The action passes every agenda gate that applies to it (a gate is a hard check, never traded)."""
    o = _row(pick, options)
    if o is None:
        return True
    gates = [b for b in o.get("blocks") or () if b.startswith("gate ")]
    if gates:
        from ..core.slow.refine import Fail
        return Fail(*gates)
    return True


def not_avoided(pick, options):
    """The action is not a hard refusal and not one the risk policy avoids (a hard prediction is never traded)."""
    o = _row(pick, options)
    if o is None:
        return True
    from ..core.slow.refine import Fail
    hard = [b for b in o.get("blocks") or () if b.startswith("hard ")]
    if hard:
        return Fail(*hard)
    if o.get("choice") == "avoid":
        return Fail(f"avoided: {o.get('why')}")
    return True


def not_a_recent_failure(pick, options):
    """The action did not fail at this key recently (the failure memory: a hard check with an expiry and a bound)."""
    o = _row(pick, options)
    if o is None:
        return True
    failed = [b for b in o.get("blocks") or () if b.startswith("failed ")]
    if failed:
        from ..core.slow.refine import Fail
        return Fail(*failed)
    return True


def chosen_action(pick):
    """The answer: the index of the chosen option (None: abstain)."""
    return pick


def search_order(facts):
    """System 2's space: the offered actions in System 2's order (none when its per-episode budget is used up)."""
    if facts.get("s2_left") == 0:
        return []
    return [o["i"] for o in sorted(facts["options"], key=lambda o: (o["s2"], int(o["i"])))]


CHECKS = (not_past_a_gate, not_avoided, not_a_recent_failure)


def _system(max_options, s2=False):
    from ..core.system import System
    cat = Catalog()
    if not s2:
        cat.fn(s1_pick, provides="pick")
    for f in CHECKS:
        cat.check(f, hard=True)
    cat.rule(QUESTION)(chosen_action)
    q = Question(QUESTION, "Which of the offered actions does the agent take?",
                 Answer.choice([str(i) for i in range(max_options)]), requires=[f.__name__ for f in CHECKS])
    return System(cat, [q])


def _plain(v):
    try:
        return json.loads(json.dumps(v, sort_keys=True, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(v)


def state_key(state):
    """The default key of a state: its canonical JSON."""
    return json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)


# ------------------------------------------------------------------------------------------------ the agent
class Agent:
    """See the module docstring.

    env: an Environment. knowledge: a solvi.Knowledge (goals, gates, the action model, the failure memory, the store).
    s2: a SlowPath of your own (default: the search over the offered actions). budget: System 2's budget per episode —
    a number (System 2 decisions) or a Budget (dollars, calls, ms of a model-backed SlowPath; the dispatcher's total,
    reset each episode). storage: where every decision is stored (a path or a TraceStorage). risk: a RiskPolicy
    (None: Protect). key: state → a string key (the map's places; default: the state's canonical JSON). gain:
    (state, action) → expected gain for the risk policy (None → the default). seed: the tie-breaks of the exploration
    order and the fallback. max_options: the most actions one state may offer."""

    def __init__(self, env, *, knowledge, s2=None, budget=None, storage=None, risk=None, key=None, gain=None, seed=0,
                 max_options=64):
        from ..core.costs import Budget
        from ..core.dispatch import Dispatcher, SearchPath, SlowPath
        from ..core.environment import Environment
        from ..core.knowledge import Protect
        from ..core.store import open_storage
        from .knowledge import Knowledge
        if not isinstance(env, Environment):
            raise TypeError("env is an Environment: reset(seed), actions(state), step(action) → Outcome")
        if not isinstance(knowledge, Knowledge):
            raise TypeError("knowledge= takes a solvi.Knowledge (its goals, gates and action model are what the agent "
                            "acts on)")
        if budget is not None and not isinstance(budget, Budget) and not (isinstance(budget, int) and budget >= 0):
            raise TypeError("budget= is System 2's budget per episode: a number of decisions, or a Budget")
        if s2 is not None and not isinstance(s2, SlowPath):
            raise TypeError("s2= takes a SlowPath (it answers \"action\" with the index of an option)")
        if risk is not None and not callable(getattr(risk, "decide", None)):
            raise TypeError("risk= takes a RiskPolicy (Protect, RiskBudget or your own: decide(...), new_episode())")
        self.env, self.knowledge = env, knowledge
        self.risk = risk if risk is not None else Protect()
        self.key = key or state_key
        self.gain = gain
        self.seed, self.max_options = int(seed), int(max_options)
        self.budget = budget
        self.system = _system(self.max_options)
        self.s2 = s2 if s2 is not None else SearchPath(_system(self.max_options, s2=True), space=search_order,
                                                       into="pick", question=QUESTION)
        self.storage = open_storage(storage)
        self.dispatcher = Dispatcher(self.system, self.s2, question=QUESTION, storage=self.storage,
                                     total=budget if isinstance(budget, Budget) else None)
        self.decisions = []              # the Dispatched decisions, when there is no storage
        self.episodes = []               # one report per episode
        self.steps = []                  # the steps of the current episode
        self._offered = {}               # (map, key) → action keys offered there (for the exploration order)
        self._ep = None
        self._pending = None
        self._dead = {}                  # goal → keys where its skill did not work this episode (no route there)
        self._visited = set()            # keys seen this episode (not a frontier of the exploration again)
        self._route_steps = 0            # System 1's route steps since the last goal done

    # --- episodes
    def new_episode(self, seed=None, map=None):
        """Start an episode by hand (run() does it): the map it plays on (default: the seed), the risk policy's and
        System 2's per-episode budgets reset, the agenda's goals open again."""
        from ..core.costs import Cost
        self._close()
        mp = str(map) if map is not None else f"seed {seed}"
        self.knowledge._begin(mp)
        if callable(getattr(self.risk, "new_episode", None)):
            self.risk.new_episode()
        self.dispatcher.spent = Cost()
        self._dead = {}
        self._visited = set()
        self._route_steps = 0
        self._ep = {"n": len(self.episodes) + 1, "seed": seed, "map": mp, "steps": 0, "s1": 0, "s2": 0, "fallback": 0,
                    "refused": 0, "risky_takes": 0, "goals_done": [], "done": False, "stopped": None, "drift": False,
                    "s2_used": 0}
        self.steps = []
        return self._ep

    def _close(self):
        if self._ep is not None and self._ep not in self.episodes:
            ep = self._ep
            ep["goals_done"] = sorted(self.knowledge.agenda.done())
            ep["s1_share"] = ep["s1"] / ep["steps"] if ep["steps"] else None
            self.knowledge._end()
            self.episodes.append(ep)

    def run(self, seed=None, steps=100, map=None):
        """One episode: reset the environment with `seed`, act and observe for at most `steps` steps (or until the
        environment says done). → the episode's numbers: steps, decisions by System 1 / System 2 / fallback, System 1's
        share, refused actions, risky takes, the goals done, drift."""
        state = self.env.reset(seed)
        self.new_episode(seed, map)
        for _ in range(int(steps)):
            action = self.act(state)
            if action is None:
                break
            out = self.env.step(action)
            self.observe(out)
            state = out.state
            if out.done:
                self._ep["done"] = True
                break
        ep = self._ep
        self._close()
        self._ep = None
        return ep

    # --- one step
    def act(self, state):
        """Decide the next action in `state` → the action (one of env.actions(state)), or None when nothing the hard
        checks allow is offered. The decision is stored; call `observe(outcome)` after the environment's step."""
        if self._ep is None:
            self.new_episode()
        km, ep = self.knowledge, self._ep
        acts = list(self.env.actions(state))
        if not acts:
            ep["stopped"] = "the environment offers no action"
            return None
        if len(acts) > self.max_options:
            raise ValueError(f"{len(acts)} actions offered, more than max_options={self.max_options}")
        key = self.key(state)
        km.agenda.update(state)
        options = self._options(state, key, acts)
        given = {"options": options, "goals_open": km.agenda.open(state), "at": key,
                 "knowledge": km.store.head()}
        if isinstance(self.budget, int):
            given["s2_left"] = max(0, self.budget - ep["s2_used"])
        d = self.dispatcher.ask(given)
        if self.storage is None:
            self.decisions.append(d)
        by, idx = d.by, None
        if by in ("s1", "s2") and d.answer is not None:
            idx = int(d.answer)
        if d.s2 is not None:
            ep["s2_used"] += 1
        if idx is None:
            idx = self._fallback(options, ep["steps"])
            by = "fallback"
            if idx is None:
                ep["stopped"] = "no offered action passes the hard checks"
                return None
        o = options[idx]
        if o["verdict"] != "accept":                  # recorded with its rationale; a risky take charges the budget
            real = self.risk.decide(o["action"], self._prediction(o), o["gain"])
            if o["verdict"] == "refuse" and real.choice == "take":
                ep["risky_takes"] += 1
        ep[by] += 1
        ep["steps"] += 1
        self._pending = {"state": state, "key": key, "action": acts[idx], "option": o, "by": by,
                         "decision": d.stored_id if d.stored_id is not None else len(self.decisions) - 1}
        return acts[idx]

    def observe(self, outcome):
        """The environment's Outcome of the last action (or True / False): written back to the knowledge (the action
        model, the failure memory, the agenda, skills, map facts). → what Knowledge.observe returns."""
        from ..core.environment import Outcome
        p = self._pending
        if p is None:
            raise ValueError("observe() follows act(): no action is pending")
        self._pending = None
        out = outcome if isinstance(outcome, Outcome) else Outcome(p["state"], accepted=bool(outcome))
        to = self.key(out.state) if out.accepted else p["key"]
        info = self.knowledge.observe(p["state"], p["action"], out, at=p["key"], to=to, map=self._ep["map"])
        if not out.accepted:
            self._ep["refused"] += 1
        if info["goals_done"]:
            self._route_steps = 0
        elif p["by"] == "s1" and p["option"]["via"] == "route":
            self._route_steps += 1
        if info["drift"] is not None:
            self._ep["drift"] = True
        self.steps.append({"action": p["option"]["action"], "by": p["by"], "accepted": bool(out.accepted),
                           "goals_done": info["goals_done"], "decision": p["decision"]})
        return info

    # --- the facts of a step
    def _options(self, state, key, acts):
        km, mp = self.knowledge, self._ep["map"]
        akeys = [action_key(a) for a in acts]
        self._offered[(mp, key)] = set(akeys)
        self._visited.add(key)
        plans = [[key, a] for a in akeys]
        rows = []
        for i, (a, ak) in enumerate(zip(acts, akeys)):
            name, args = split_action(a)
            p = self._predict(state, key, name, args, ak)
            blocks = [f"gate {g}" for g in km.agenda.allows(state, name)[1]]
            if p.verdict == "refuse" and p.hard:
                blocks.append(f"hard refusal: {p.reason}")
            if km.failures is not None:
                f = km.failures.check([key, ak], options=plans)
                if f.verdict == "refuse":
                    blocks.append(f"failed here: {f.reason}")
            rows.append({"i": str(i), "action": ak, "name": name, "p": p, "blocks": blocks, "goal": None, "via": None,
                         "s1": None})
        self._advance(state, key, rows, mp)
        tried = self._tried(mp, key)
        frontier = self._frontier(mp, key)
        out = []
        for r in rows:
            p = r["p"]
            if r["via"] == "route" and p.verdict == "unknown":   # a confirmed map edge: the step is known to work
                from ..core.knowledge import Prediction
                p = Prediction("accept", 0.0, 1, "a step of a confirmed route (map fact)", action=r["name"])
            g = self._gain(state, r["action"], r["goal"] is not None)
            rd = self._peek(r["action"], p, g)
            row = {"i": r["i"], "action": r["action"], "verdict": p.verdict, "risk": round(float(p.risk), 6),
                   "support": int(p.support), "reason": p.reason, "hard": bool(p.hard), "choice": rd.choice,
                   "why": rd.reason, "blocks": r["blocks"], "goal": r["goal"], "via": r["via"], "gain": g,
                   "s1": r["s1"]}
            row["s2"] = self._s2_rank(row, r["action"] in tried, r["action"] == frontier)
            out.append(row)
        order = sorted(range(len(out)), key=lambda j: (out[j]["s2"], self._tie(key, out[j]["action"])))
        for rank, j in enumerate(order):
            out[j]["s2"] = rank
        return out

    def _predict(self, state, key, name, args, ak):
        from ..core.knowledge import Prediction
        km = self.knowledge
        if km.actions is not None:
            return km.actions.predict(state, name, args)
        acc = km.accepted_at(self._ep["map"], key, ak)      # no action model: what the map says about this key
        if acc is True:
            return Prediction("accept", 0.0, 1, "accepted here before (map fact)", action=name)
        if acc is False:
            return Prediction("refuse", 0.5, 1, "refused here before (map fact)", action=name)
        return Prediction("unknown", 0.5, 0, "no action model, not tried here", action=name)

    def _advance(self, state, key, rows, mp):
        """Mark the options that advance an open goal: a skill of the goal (direct), or the first step of a confirmed
        route to a key where a skill of the goal was accepted. A route uses only map edges whose action the gates and
        the risk policy allow in the current state (the walk is assumed not to change what the gates read); a key where
        the skill turned out not to work this episode is no longer a target; and when System 1 has walked routes for
        longer than twice the map's size without progress, routes are off for the rest of the episode (System 2
        explores instead)."""
        km = self.knowledge
        by_key = {r["action"]: r for r in rows}
        n_keys = len(km._edges.get(mp, {})) + 1
        if self._route_steps > 2 * n_keys + 2:
            routes = False
        else:
            routes = True
        ok_cache = {}

        def edge_ok(akey):
            if akey not in ok_cache:
                r = by_key.get(akey)
                if r is not None:
                    ok_cache[akey] = not r["blocks"] and (r["p"].verdict != "refuse" or self._takes(state, r))
                else:
                    name, args = split_action(json.loads(akey) if akey[:1] in "[{" else akey)
                    p = self._predict(state, key, name, args, akey)
                    ok_cache[akey] = km.agenda.allows(state, name)[0] and not (p.verdict == "refuse" and p.hard) \
                        and self._peek(akey, p, self._gain(state, akey, True)).choice != "avoid"
            return ok_cache[akey]

        for gi, g in enumerate(km.agenda.open(state)):
            skills = km.skills(g)
            for a in skills:
                r = by_key.get(a)
                if r is not None and r["goal"] is None:
                    r.update(goal=g, via="skill", s1=[gi, 0, 0])
            if not routes:
                continue
            targets = {k for a in skills for k in km.where(mp, a)}
            dead = self._dead.setdefault(g, set())
            if key in targets and not any(by_key.get(a) is not None and not by_key[a]["blocks"]
                                          and self._takes(state, by_key[a]) for a in skills):
                dead.add(key)                               # arrived, and the skill does not work here now
            targets -= dead | {key}
            if not targets:
                continue
            path = self._route(mp, key, targets, edge_ok) or []
            r = by_key.get(path[0]) if path else None
            if r is not None and r["goal"] is None:
                r.update(goal=g, via="route", s1=[gi, 1, len(path)])

    def _gain(self, state, action, advances):
        g = None if self.gain is None else self.gain(state, action)
        return float(g) if g is not None else (1.0 if advances else 0.0)

    def _takes(self, state, r):
        return self._peek(r["action"], r["p"], self._gain(state, r["action"], True)).choice == "take"

    def _route(self, mp, start, targets, edge_ok=None):
        """The shortest path of action keys over usable map facts from `start` to any of `targets`, through edges
        `edge_ok(action key)` allows, or None."""
        km = self.knowledge
        prev: dict = {start: None}
        todo = deque([start])
        while todo:
            k = todo.popleft()
            if k in targets:
                path = []
                while prev[k] is not None:
                    k, a = prev[k]
                    path.append(a)
                return path[::-1]
            for a, to in sorted(km.edges(mp, k).items()):
                if to not in prev and (edge_ok is None or edge_ok(a)):
                    prev[to] = (k, a)
                    todo.append(to)
        return None

    def _tried(self, mp, key):
        return set(self.knowledge._accepts.get(mp, {}).get(key, {}))

    def _frontier(self, mp, key):
        """The first action of the shortest path to the nearest other key with an offered action never tried there."""
        km = self.knowledge
        prev: dict = {key: None}
        todo = deque([key])
        while todo:
            k = todo.popleft()
            if k != key and k not in self._visited and self._offered.get((mp, k), set()) - self._tried(mp, k):
                while prev[k][0] != key:
                    k = prev[k][0]
                return prev[k][1]
            for a, to in sorted(km.edges(mp, k).items()):
                if to not in prev:
                    prev[to] = (k, a)
                    todo.append(to)
        return None

    @staticmethod
    def _s2_rank(o, tried, frontier):
        if o["blocks"] or o["choice"] == "avoid":
            return 9
        if o["goal"] is not None or (o["gain"] > 0 and o["choice"] == "take"):
            return 0
        if o["verdict"] == "unknown" and not tried:
            return 1
        if not tried:
            return 2
        if frontier:
            return 3
        if o["verdict"] == "unknown":
            return 4
        if o["verdict"] == "accept":
            return 5
        return 6

    def _tie(self, key, ak):
        h = hashlib.sha256(f"{self.seed}|{len(self.episodes)}|{self._ep['steps']}|{key}|{ak}".encode()).hexdigest()
        return int(h[:12], 16)

    def _peek(self, akey, p, gain):
        """The risk policy's choice on one option without changing its state: its attributes are put back (a list,
        such as a decision log, is cut back to its length — policies append to their lists)."""
        pol = self.risk
        saved = {}
        for k, v in vars(pol).items():
            if isinstance(v, list):
                saved[k] = (v, len(v))
            else:
                saved[k] = (set(v) if isinstance(v, set) else dict(v) if isinstance(v, dict) else v, None)
        try:
            return pol.decide(akey, p, gain)
        finally:
            for k, (v, n) in saved.items():
                if n is not None:
                    del v[n:]
                setattr(pol, k, v)

    @staticmethod
    def _prediction(o):
        from ..core.knowledge import Prediction
        return Prediction(o["verdict"], o["risk"], o["support"], o["reason"], hard=o["hard"], action=o["action"])

    def _fallback(self, options, step):
        ok = [o for o in options if not o["blocks"] and o["choice"] != "avoid"]
        if not ok:
            return None
        ok.sort(key=lambda o: self._tie(str(step), o["action"]))
        return int(ok[0]["i"])

    # --- reading
    def replay(self):
        """Re-check every stored decision without the environment or a model → {"decisions", "ok", "mismatches": the
        first few [(n, mismatches)]}."""
        ds = self.dispatcher.stored() if self.storage is not None else list(self.decisions)
        bad = []
        for d in ds:
            rep = self.dispatcher.replay(d)
            if not rep["ok"]:
                bad.append((d.n, rep["mismatches"][:3]))
        return {"decisions": len(ds), "ok": len(ds) - len(bad), "mismatches": bad[:5]}

    def report(self):
        """What the agent did: per episode (steps, System 1 / System 2 / fallback decisions, System 1's share, refused
        actions, risky takes, goals done, drift, why it stopped), the totals, and the knowledge's report."""
        eps = list(self.episodes) + ([self._ep] if self._ep is not None and self._ep not in self.episodes else [])
        tot: dict = {k: sum(e[k] for e in eps) for k in ("steps", "s1", "s2", "fallback", "refused", "risky_takes")}
        tot["episodes"] = len(eps)
        tot["s1_share"] = tot["s1"] / tot["steps"] if tot["steps"] else None
        return {"episodes": [_plain(e) for e in eps], "totals": tot, "risk": type(self.risk).__name__,
                "knowledge": self.knowledge.report()}

    def __repr__(self):
        return f"Agent({type(self.env).__name__}, {len(self.episodes)} episodes, risk={type(self.risk).__name__})"


__all__ = ["Agent", "CHECKS", "QUESTION", "s1_pick", "search_order", "state_key"]
