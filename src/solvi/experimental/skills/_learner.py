"""The loop (SkillLearner) and its adapter as System 2 of solvi.Agent (see the package docstring)."""
from __future__ import annotations

import hashlib
import json

from ...core.dispatch import SlowPath, Thought
from ._certificate import Certificate
from ._model import ExactModel, Plan, _plain, plan, verify
from ._skills import SkillSet

_EPISODE = ("steps", "s1", "s2", "fallback", "planner_calls", "plans_found", "plans_rejected", "compiled", "supervised",
            "supervised_failed", "alone_runs", "alone_failures", "s1_failed", "mismatch", "guard_violations", "searched",
            "verify_calls")


def _cost(rec):
    return {"candidates": rec.get("searched", 0), "tests": rec.get("verify_calls", 0), "probes": 0,
            "simulated": rec.get("simulated", 0)}


def _goals(goals):
    items = list(goals.items()) if isinstance(goals, dict) else [tuple(g) for g in goals]
    if not items:
        raise ValueError("goals: at least one (name, done(state)) pair")
    for name, fn in items:
        if not callable(fn):
            raise TypeError(f"goal {name!r}: done(state) → bool is a function")
    names = [str(n) for n, _ in items]
    if len(set(names)) != len(names):
        raise ValueError("two goals have one name")
    return [(str(n), fn) for n, fn in items]


class SkillLearner:
    """The loop of learning in a world described by code (see the package docstring).

    env: an Environment whose copy is an exact model (deepcopy, or `clone=`). grammar: a laws Grammar for skill
    conditions (None: cases keyed by the start state). goals: {name: done(state)} or [(name, done)] in order — the
    first goal not yet reached in the episode is worked on (the order is the prerequisites). planner: plan (or a
    function with its signature); horizon, budget, actions, cost: passed to it. promise, delta: the default
    Certificate(promise, delta); certificate: a Certificate of your own, or False (no certificate: skills act alone
    at once — a control). verify_plans: verify each plan on a fresh fork before it is executed or compiled (False: a
    control without the judge). learn: compile skills (False: System 2 plans every goal — "plan always").
    generalize: the condition rule's merge. guard: check after every planner and verify call that the real
    environment's hash did not change (a RuntimeError if it did). seed: the fallback's tie-breaks.

    Per step (`act(state)` → an action; the caller steps the environment and calls `observe(outcome)`; `run(seed,
    steps)` does both): a skill run in progress takes its next action; else, for the first open goal, System 1 runs
    the most specific matching skill whose certificate holds, alone; else the most specific matching skill is
    re-verified by System 2 on the exact model from here — its outcome feeds its certificate, and if it verifies it
    runs as System 2's; else System 2 plans on the exact model, verifies the plan, runs it and compiles it into a
    skill; when no plan is found, a reproducible pseudo-random offered action ("fallback").

    Records. `decisions`: one per step — episode, step, by ("s1" / "s2" / "fallback"), kind ("alone" / "supervised" /
    "planned" / "fallback"), skill id, goal, action, certified (System 1 acted under a holding certificate), accepted,
    run_ok (did the run this step belonged to reach its goal). `episodes`: one per episode — steps, decisions by
    System 1 / System 2 / fallback, s1_share, goals (reached, in order), goals_reached, all_goals, planner_calls,
    plans_found / plans_rejected, simulated (steps on the exact model), compiled, supervised (failed), alone_runs,
    alone_failures, s1_failed (System 1 decisions in failed runs), mismatch (a verified plan the real world did not
    follow), guard_violations, skills / usable / retired at the end; searched (simulated steps the planner's searches
    took), verify_calls, and cost — the search-cost record next to the outcome: {"candidates": searched (the states
    the planner expanded), "tests": verify_calls (re-simulations of a plan or skill), "probes": 0 (the real
    environment is never asked to tell hypotheses apart here), "simulated": every step on the exact model}."""

    def __init__(self, env, grammar, goals, *, planner=plan, horizon: int = 12, budget: int = 20_000,
                 promise: float = 0.05, delta: float = 0.1, certificate=None, actions=None, cost=None, key=None,
                 clone=None, verify_plans: bool = True, learn: bool = True, generalize: bool = True,
                 guard: bool = True, seed: int = 0):
        self.env = env
        self.model = ExactModel(env, clone=clone, key=key)
        self.goals = _goals(goals)
        self._goal_fn = dict(self.goals)
        if certificate is None:
            certificate = Certificate(promise, delta)
        elif certificate is False:
            certificate = None
        elif not isinstance(certificate, Certificate):
            raise TypeError("certificate= takes a Certificate, or False (none: a control)")
        self.skills = SkillSet(grammar, certificate, generalize=generalize, key=self.model.key)
        self.planner, self.horizon, self.budget = planner, int(horizon), int(budget)
        self.actions, self.cost = actions, cost
        self.verify_plans, self.learn, self.guard, self.seed = bool(verify_plans), bool(learn), bool(guard), int(seed)
        self.decisions: list[dict] = []
        self.episodes: list[dict] = []
        self.totals = {k: 0 for k in _EPISODE}
        self._ep = None
        self._run = None
        self._pending = None
        self._reached: list[str] = []
        self._runs = 0
        self._sim0 = 0
        self._cache = None

    # --- episodes
    def new_episode(self, seed=None):
        """Start an episode by hand (run() does it): no goal reached yet."""
        self._close()
        self._ep = {"n": len(self.episodes) + 1, "seed": seed, **{k: 0 for k in _EPISODE}, "goals": [],
                    "done": False, "stopped": None}
        self._run, self._pending, self._reached = None, None, []
        self._sim0 = self.model.steps
        return self._ep

    def _close(self):
        ep = self._ep
        if ep is None or ep in self.episodes:
            return
        self._run = None
        ep["goals"] = list(self._reached)
        ep["goals_reached"] = len(self._reached)
        ep["all_goals"] = len(self._reached) == len(self.goals)
        ep["s1_share"] = ep["s1"] / ep["steps"] if ep["steps"] else None
        ep["simulated"] = self.model.steps - self._sim0
        rep = self.skills.report()
        ep["skills"], ep["usable"], ep["retired"] = rep["count"], rep["usable"], rep["retired"]
        ep["cost"] = _cost(ep)
        self.episodes.append(ep)

    def run(self, seed=None, steps: int = 200) -> dict:
        """One episode: reset the environment with `seed`, act and observe for at most `steps` steps (or until the
        environment says done, or every goal is reached) → the episode's record."""
        state = self.env.reset(seed)
        self.new_episode(seed)
        for _ in range(int(steps)):
            a = self.act(state)
            if a is None:
                break
            out = self.env.step(a)
            self.observe(out)
            state = out.state
            if out.done:
                break
        ep = self._ep if self._ep is not None else {}
        self._close()
        self._ep = None
        return ep

    # --- goals
    def _update(self, state):
        if state is not None:
            for name, fn in self.goals:
                if name not in self._reached and fn(state):
                    self._reached.append(name)
        return next((n for n, _ in self.goals if n not in self._reached), None)

    def _count(self, k, n=1):
        self.totals[k] += n
        if self._ep is not None:
            self._ep[k] += n

    # --- System 2 on the exact model
    def _guarded(self, f, *a, **kw):
        before = self.model.fingerprint() if self.guard else None
        out = f(*a, **kw)
        if self.guard and self.model.fingerprint() != before:
            self._count("guard_violations")
            raise RuntimeError("the real environment changed during a call on the exact model: a fork is not "
                               "independent (give clone=)")
        return out

    def _plan(self, state, fn):
        self._count("planner_calls")
        p = self._guarded(self.planner, self.model, state, fn, actions=self.actions, horizon=self.horizon,
                          budget=self.budget, cost=self.cost)
        self._count("searched", int(self.model.last.get("searched", 0) or 0))
        if p is not None:
            self._count("plans_found")
        return p

    def _verify(self, state, p, fn):
        self._count("verify_calls")
        return self._guarded(verify, self.model, state, p, fn)

    # --- one step
    def act(self, state):
        """Decide the next action in `state` → an action, or None (every goal reached, or nothing offered). The real
        environment is not stepped here: call `observe(outcome)` after stepping it."""
        if self._ep is None:
            self.new_episode()
        if self._pending is not None:
            raise ValueError("act() follows observe(): the last action's outcome is pending")
        g = self._update(state)
        run = self._run
        if run is not None:
            if run["goal"] in self._reached:
                self._end_run(True)
            elif run["i"] < len(run["actions"]):
                return self._take(run["actions"][run["i"]], run)
            else:
                self._end_run(False)
        if g is None:
            self._ep["stopped"] = "every goal reached"
            return None
        fn = self._goal_fn[g]
        run = None
        if self.learn:
            atoms = self.skills.atoms(state)
            sk = self.skills.match(state, g, atoms=atoms)
            if sk is not None:
                run = self._start(sk.actions, g, "s1", sk, "alone", state)
            elif self.skills.certificate is not None:
                cands = self.skills.candidates(state, g, atoms=atoms)
                if cands:
                    top = cands[0]
                    chk = self._verify(state, Plan(top.actions, goal=fn), fn)
                    self.skills.observe(top, chk.ok, alone=False, state=state)
                    self._count("supervised")
                    if chk.ok:
                        run = self._start(top.actions, g, "s2", top, "supervised", state)
                    else:
                        self._count("supervised_failed")
        if run is None:
            p = self._plan(state, fn)
            if p is not None and self.verify_plans and not self._verify(state, p, fn).ok:
                self._count("plans_rejected")
                p = None
            if p is not None and p.actions:
                sk = None
                if self.learn:
                    n0 = len(self.skills.skills)
                    sk = self.skills.compile(g, p, state, verified=self.verify_plans)
                    self._count("compiled", len(self.skills.skills) - n0)
                run = self._start(p.actions, g, "s2", sk, "planned", state)
        if run is None:
            a = self._fallback(state)
            if a is None:
                self._ep["stopped"] = "the environment offers no action"
                return None
            return self._take(a, None, g)
        return self._take(run["actions"][0], run)

    def _start(self, actions, goal, by, skill, kind, state):
        self._runs += 1
        self._run = {"id": self._runs, "actions": tuple(actions), "i": 0, "goal": goal, "by": by, "skill": skill,
                     "kind": kind, "state": state, "decisions": []}
        return self._run

    def _take(self, action, run, goal=None):
        ep = self._ep
        by = run["by"] if run is not None else "fallback"
        sk = run["skill"] if run is not None else None
        ep[by] += 1
        ep["steps"] += 1
        self.totals[by] += 1
        self.totals["steps"] += 1
        alone = run is not None and run["kind"] == "alone"
        rec = {"episode": ep["n"], "step": ep["steps"], "by": by, "kind": run["kind"] if run else "fallback",
               "skill": sk.id if sk is not None else None, "goal": run["goal"] if run else goal,
               "action": _plain(action), "certified": bool(alone and self.skills.certificate is not None
                                                          and self.skills.usable(sk)),
               "accepted": None, "run_ok": None}
        self.decisions.append(rec)
        if run is not None:
            run["decisions"].append(rec)
        self._pending = {"run": run, "rec": rec}
        return action

    def observe(self, outcome):
        """The environment's Outcome of the last action (or True / False): the run goes on, ends (its goal reached, an
        action refused, its actions used up) and its outcome feeds the skill's certificate when it ran alone."""
        from ...core.environment import Outcome
        p = self._pending
        if p is None:
            raise ValueError("observe() follows act(): no action is pending")
        self._pending = None
        out = outcome if isinstance(outcome, Outcome) else Outcome(None, accepted=bool(outcome))
        p["rec"]["accepted"] = bool(out.accepted)
        self._update(out.state)
        run = p["run"]
        if run is not None and run is self._run:
            run["i"] += 1
            if not out.accepted:
                self._end_run(False)
            elif run["goal"] in self._reached:
                self._end_run(True)
            elif run["i"] >= len(run["actions"]) or out.done:
                self._end_run(False)
        if out.done:
            self._ep["done"] = True
        return {"accepted": bool(out.accepted), "goals": list(self._reached)}

    def _end_run(self, ok):
        run, self._run = self._run, None
        for rec in run["decisions"]:
            rec["run_ok"] = ok
        sk, kind = run["skill"], run["kind"]
        if kind == "alone":
            self._count("alone_runs")
            if not ok:
                self._count("alone_failures")
                self._count("s1_failed", len(run["decisions"]))
            if sk is not None:
                self.skills.observe(sk, ok, alone=True, state=run["state"])
        elif kind == "supervised":
            if not ok:
                self._count("mismatch")              # the exact model said it works here; the world did not follow
        elif kind == "planned":
            if sk is not None:
                sk.outcomes["source_ok" if ok else "source_fail"] += 1
            if not ok and self.verify_plans:
                self._count("mismatch")

    def _fallback(self, state):
        acts = list(self.env.actions(state))
        if not acts:
            return None
        n = self._ep["n"] if self._ep else 0
        s = self._ep["steps"] if self._ep else 0

        def tie(a):
            k = a if isinstance(a, str) else json.dumps(a, sort_keys=True, default=str)
            return hashlib.sha256(f"{self.seed}|{n}|{s}|{k}".encode()).hexdigest()

        return sorted(acts, key=tie)[0]

    # --- as System 2 of solvi.Agent
    def propose(self, state, prefer=()):
        """One action for `state` from System 2 on the exact model, without an episode of its own (what `as_s2` asks):
        the goal is the first of `prefer` (the agent's open goals) this learner knows and that does not hold, else its
        first goal that does not hold. A plan is searched, verified and compiled once, then followed step by step while
        the state is the one it predicted. → (goal, action or None, a record)."""
        key = self.model.key
        open_ = [n for n, fn in self.goals if not fn(state)]
        g = next((n for n in prefer or () if n in open_), open_[0] if open_ else None)
        k = key(state)
        if g is None:
            return None, None, {"why": "no open goal of this learner", "at": k}
        c = self._cache
        if c is not None and c["goal"] == g and k in c["at"]:
            i = c["at"][k]
            return g, c["plan"].actions[i], {"goal": g, "plan": [_plain(a) for a in c["plan"].actions], "step": i,
                                             "at": k, "searched": 0, "cached": True, "skill": c["skill"]}
        fn = self._goal_fn[g]
        p = self._plan(state, fn)
        if p is not None and self.verify_plans and not self._verify(state, p, fn).ok:
            self._count("plans_rejected")
            p = None
        if p is None or not p.actions:
            self._cache = None
            return g, None, {"goal": g, "why": f"no verified plan ({self.model.last.get('why')})", "at": k,
                             "searched": self.model.last.get("searched", 0)}
        sk = self.skills.compile(g, p, state, verified=self.verify_plans) if self.learn else None
        at = {k: 0}
        for i, s in enumerate(p.predicted[:-1]):
            at.setdefault(key(s), i + 1)
        self._cache = {"goal": g, "plan": p, "at": at, "skill": sk.id if sk is not None else None}
        return g, p.actions[0], {"goal": g, "plan": [_plain(a) for a in p.actions], "step": 0, "at": k,
                                 "searched": p.searched, "cached": False, "skill": self._cache["skill"],
                                 "simulation": p.simulation}

    # --- reading
    def report(self) -> dict:
        """Per episode and in total: the records above, System 1's share, and the skill set's report."""
        eps = list(self.episodes)
        tot = dict(self.totals)
        tot["episodes"] = len(eps)
        tot["s1_share"] = tot["s1"] / tot["steps"] if tot["steps"] else None
        tot["simulated"] = self.model.steps
        tot["cost"] = _cost(tot)
        return {"episodes": eps, "totals": tot, "skills": self.skills.report()}

    def __repr__(self):
        return f"SkillLearner({type(self.env).__name__}, {len(self.episodes)} episodes, {len(self.skills.skills)} skills)"


class ExactPlanPath(SlowPath):
    """System 2 for solvi.Agent that plans on the exact model of a SkillLearner (`as_s2(learner)`): it answers "action"
    with the index of the option the verified plan's next action is, from the step's facts — `at` (the state key,
    decoded by `decode`, default JSON) and `goals_open` (the agent's open goals, matched by name). The option must
    pass the agent's hard checks (no blocks, not avoided); otherwise it does not answer. The record holds the goal, the
    plan, the step, the state key, whether the plan was cached and the simulated steps it took."""
    mode = "exact-plan"

    def __init__(self, learner: SkillLearner, *, decode=None, max_options: int = 64):
        from ...solutions.agent import QUESTION, _system
        super().__init__(_system(int(max_options), s2=True), question=QUESTION)
        self.learner, self.decode = learner, decode or json.loads

    def fingerprint(self):
        from ...core.provenance import code_fingerprint, digest
        L = self.learner
        return digest("ExactPlanPath", self.mode, self.system.fingerprint(), L.horizon, L.budget,
                      [n for n, _ in L.goals], [code_fingerprint(fn) for _, fn in L.goals])

    def think(self, state, question, *, price=None, budget=None, expected_round=None, store=True, **kw):
        from ...solutions.knowledge import action_key
        state_ = self.decode(state["at"])
        g, action, rec = self.learner.propose(state_, prefer=state.get("goals_open") or ())
        if action is None:
            return Thought(self.mode, None, False, why=rec.get("why"), record=rec)
        ak = action_key(action)
        row = next((o for o in state["options"] if o["action"] == ak), None)
        if row is None:
            return Thought(self.mode, None, False, why=f"the planned action {ak!r} is not offered", record=rec)
        if row.get("blocks") or row.get("choice") == "avoid":
            why = "; ".join(row.get("blocks") or []) or f"avoided: {row.get('why')}"
            return Thought(self.mode, None, False, why=f"the planned action {ak!r} is blocked ({why})", record=rec)
        return Thought(self.mode, row["i"], True, record=rec)


def as_s2(learner: SkillLearner, *, decode=None, max_options: int = 64) -> ExactPlanPath:
    """The learner as System 2 of `solvi.Agent(env, knowledge=km, s2=as_s2(learner))`: planning on the exact model
    when System 1 abstains (the learner's env must be the agent's env; the agent's state keys must decode to states —
    its default key is the state's canonical JSON)."""
    if not isinstance(learner, SkillLearner):
        raise TypeError("as_s2 takes a SkillLearner")
    return ExactPlanPath(learner, decode=decode, max_options=max_options)


__all__ = ["ExactPlanPath", "SkillLearner", "as_s2"]
