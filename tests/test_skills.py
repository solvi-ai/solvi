"""solvi.experimental.skills: learning in a world described by code — System 2 plans on an exact model (a fork of the
environment), only plans the code verified are compiled into System 1 skills, and a skill acts alone only while its
outcomes keep an anytime-valid certificate.

Tiny worlds used throughout: a corridor (walk right to a cell; "left" at 0 and "right" at the end are refused) whose
start carries an irrelevant colour, so plans from two starts differ only in an atom that does not matter."""
from __future__ import annotations

import math
import warnings

import pytest

from solvi.core import Outcome
from solvi.core.knowledge import KnowledgeStore
from solvi.core.store import JSONLStorage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Grammar
    from solvi.experimental.skills import (Certificate, Check, EProcess, ExactModel, Plan, Skill, SkillLearner,
                                           SkillSet, as_s2, plan, verify)


class Corridor:
    """Cells 0..n-1; walk right to `goal`. A colour drawn from the seed is part of the state and changes nothing."""

    def __init__(self, n=6, goal=3):
        self.n, self.goal = n, goal
        self.reset(0)

    def reset(self, seed=None):
        self.pos, self.colour = 0, (seed or 0) % 3
        return self._state()

    def _state(self):
        return {"pos": self.pos, "colour": self.colour}

    def actions(self, state):
        return ["left", "right"]

    def step(self, action):
        to = self.pos + (1 if action == "right" else -1)
        if not 0 <= to < self.n:
            return Outcome(self._state(), accepted=False)
        self.pos = to
        return Outcome(self._state(), accepted=True, effect={"pos": to}, done=to == self.goal)


def at_goal(s):
    return s["pos"] == 3


def grammar(n=6):
    return Grammar([Atom(f"pos={i}", (lambda s, i=i: s["pos"] == i), feature="pos") for i in range(n)]
                   + [Atom(f"colour={c}", (lambda s, c=c: s["colour"] == c), feature="colour") for c in range(3)])


# ------------------------------------------------------------------------------------------------ the exact model
def test_a_fork_is_independent_and_the_real_environment_is_never_stepped_by_the_planner():
    env = Corridor()
    env.reset(1)
    m = ExactModel(env)
    before = m.fingerprint()
    w = m.fork()
    w.step("right")
    assert env.pos == 0 and w.pos == 1
    p = plan(m, env._state(), at_goal, horizon=6, budget=1000)
    assert p is not None and m.fingerprint() == before and env.pos == 0          # the real state's hash: before == after
    outs = m.simulate(p.actions)
    assert [o.state["pos"] for o in outs] == [1, 2, 3] and m.fingerprint() == before


def test_clone_hook_is_used_for_forks():
    env = Corridor()
    calls = []

    def clone(e, state):
        calls.append(state)
        c = Corridor(e.n, e.goal)
        c.pos, c.colour = e.pos, e.colour
        return c

    m = ExactModel(env, clone=clone)
    p = plan(m, env._state(), at_goal, horizon=5, budget=1000)
    assert p.actions == ("right", "right", "right") and calls


# ------------------------------------------------------------------------------------------------ the planner
def test_plan_finds_the_shortest_plan_deterministically_and_records_what_it_predicted():
    env = Corridor()
    m = ExactModel(env)
    s = env.reset(0)
    p = plan(m, s, at_goal, horizon=6, budget=1000)
    assert isinstance(p, Plan) and p.actions == ("right", "right", "right")
    assert [x["pos"] for x in p.predicted] == [1, 2, 3] and p.cost == 3 and p.searched > 0
    assert p.start == m.key(s) and p.end == m.key(p.predicted[-1]) and p.simulation
    q = plan(m, s, at_goal, horizon=6, budget=1000)
    assert (q.actions, q.simulation, q.searched) == (p.actions, p.simulation, p.searched)   # deterministic
    assert "left" not in p.actions                                                    # a refused step is never planned


def test_no_plan_within_the_horizon_or_the_budget_is_none():
    env = Corridor()
    m = ExactModel(env)
    s = env.reset(0)
    assert plan(m, s, at_goal, horizon=2, budget=1000) is None
    assert m.last["why"] == "horizon"
    assert plan(m, s, at_goal, horizon=6, budget=3) is None
    assert m.last["why"] == "budget" and m.last["searched"] <= 3
    assert plan(m, s, lambda st: st["pos"] == 99, horizon=20, budget=10_000) is None    # nothing left to search
    assert m.last["why"] == "exhausted"
    p = plan(m, {"pos": 0, "colour": 0}, lambda st: st["pos"] == 0, horizon=3, budget=10)
    assert p.actions == () and p.searched == 0                                         # the goal holds at the start


def test_plan_with_a_cost_and_a_fixed_action_list():
    env = Corridor(n=8, goal=7)
    m = ExactModel(env)
    s = env.reset(0)
    p = plan(m, s, lambda st: st["pos"] == 2, actions=["right"], horizon=4, budget=100,
             cost=lambda st, a: 2.0)
    assert p.actions == ("right", "right") and p.cost == 4.0
    b = plan(m, s, lambda st: st["pos"] == 2, horizon=4, budget=100, beam=1)
    assert b is not None and b.actions == ("right", "right")


# ------------------------------------------------------------------------------------------------ verify
def test_verify_accepts_the_plan_and_catches_a_wrong_one():
    env = Corridor()
    m = ExactModel(env)
    s = env.reset(0)
    p = plan(m, s, at_goal, horizon=6, budget=1000)
    c = verify(m, s, p)
    assert isinstance(c, Check) and c.ok and bool(c) and c.simulation == p.simulation and c.end["pos"] == 3
    short = Plan(("right", "right"), p.predicted[:2], 2.0, 0, p.start, p.end, "", goal=at_goal)
    assert not verify(m, s, short).ok and "goal" in verify(m, s, short).why
    refused = Plan(("left",), ({"pos": 0, "colour": 0},), 1.0, 0, p.start, p.end, "", goal=at_goal)
    assert not verify(m, s, refused) and "refused" in verify(m, s, refused).why
    lying = Plan(p.actions, ({"pos": 2, "colour": 0},) + p.predicted[1:], 3.0, 0, p.start, p.end, p.simulation,
                 goal=at_goal)
    assert not verify(m, s, lying).ok and "predicted" in verify(m, s, lying).why       # the predicted states are checked
    env.reset(0)
    env.pos = 1                                                                        # another start: the goal only
    s1 = env._state()
    assert not verify(m, s1, p).ok                                                     # from cell 1 the plan overshoots
    env.pos = 0


# ------------------------------------------------------------------------------------------------ the certificate
def test_the_certificate_needs_its_clean_outcomes_and_once_certified_stays_certified():
    cert = Certificate(promise=0.05, delta=0.1)
    g = math.log1p(0.9 * 0.05 / 0.95)
    assert cert.need == math.ceil(math.log(10) / g) == 50
    e = cert.process()
    assert isinstance(e, EProcess) and not e.certified and e.remaining() == 50
    for _ in range(49):
        e.update(True, alone=False)
    assert not e.certified and e.remaining() == 1
    ev = e.update(True, alone=False)
    assert e.certified and ev["certified"] and e.remaining() == 0
    e.update(False, alone=False)                     # a failure under supervision lowers the wealth ...
    assert e.certified                               # ... but what was certified stays certified
    with pytest.raises(ValueError):
        Certificate(promise=0.0)
    with pytest.raises(ValueError):
        Certificate(scope="world")


def test_the_alarm_on_alone_outcomes_restarts_the_process():
    e = Certificate(promise=0.05, delta=0.1).process()
    for _ in range(50):
        e.update(True, alone=True)
    assert e.certified
    ev = e.update(False, alone=True)                 # at 5%, one failure alone is an alarm (e-detector ≥ 1/δ)
    assert ev["alarm"] and not e.certified and e.alarms == 1 and e.remaining() == 50
    e2 = Certificate(promise=0.3, delta=0.1).process()
    for _ in range(e2.cert.need):
        e2.update(True, alone=True)
    assert e2.certified and not e2.update(False, alone=True)["alarm"]   # at 30% one failure is not enough


# ------------------------------------------------------------------------------------------------ skills
def compiled(skills, env, seed, goal=at_goal, name="door"):
    m = ExactModel(env)
    s = env.reset(seed)
    p = plan(m, s, goal, horizon=6, budget=1000)
    assert verify(m, s, p).ok
    return skills.compile(name, p, s), s


def test_condition_generalization_keeps_the_atoms_every_verified_start_shares():
    env = Corridor()
    ks = SkillSet(grammar(), certificate=Certificate(0.5, 0.5))
    a, s0 = compiled(ks, env, 0)
    assert isinstance(a, Skill) and a.condition == ("colour=0", "pos=0") and a.support == 1
    assert a.source == "verified" and a.start and a.simulation
    b, s1 = compiled(ks, env, 1)                                      # the same plan from another colour
    assert b is a and a.condition == ("pos=0",) and a.support == 2 and len(a.starts) == 2
    assert ks.candidates({"pos": 0, "colour": 2}, "door") == [a]
    assert ks.candidates({"pos": 1, "colour": 2}, "door") == []       # not where no verified plan started


def test_a_recorded_failure_blocks_a_generalization_that_would_cover_it():
    env = Corridor()
    ks = SkillSet(grammar(), certificate=Certificate(0.5, 0.5))
    a, _ = compiled(ks, env, 0)
    ks.observe(a, False, alone=False, state={"pos": 0, "colour": 1})  # it failed from a state with colour 1
    b, _ = compiled(ks, env, 1)                                       # dropping the colour would cover that failure
    assert b is not a and a.condition == ("colour=0", "pos=0") and b.condition == ("colour=1", "pos=0")
    flat = SkillSet(grammar(), certificate=Certificate(0.5, 0.5), generalize=False)
    c, _ = compiled(flat, env, 0)
    d, _ = compiled(flat, env, 1)
    assert c is not d and len(flat.skills) == 2


def test_match_is_the_most_specific_certified_skill_and_retirement_and_recertification():
    env = Corridor()
    ks = SkillSet(grammar(), certificate=Certificate(0.5, 0.5))     # need 2 clean outcomes
    general, _ = compiled(ks, env, 0)
    compiled(ks, env, 1)                                              # general: pos=0
    specific = ks.add(Skill(("colour=2", "pos=0"), ("right", "right", "right"), "door", source="verified"))
    s = {"pos": 0, "colour": 2}
    assert ks.candidates(s, "door") == [specific, general] and ks.match(s, "door") is None
    for _ in range(2):
        ks.observe(general, True, alone=False)
    assert ks.match(s, "door") is general                             # the specific one is not certified yet
    for _ in range(2):
        ks.observe(specific, True, alone=False)
    assert ks.match(s, "door") is specific
    ev = None
    for _ in range(3):
        ev = ks.observe(specific, False, alone=True, state=s)
        if specific.status == "retired":
            break
    assert specific.status == "retired" and ev["retired"] and ks.match(s, "door") is general
    assert specific in ks.retired()
    for _ in range(2):
        ev = ks.observe(specific, True, alone=False)
    assert specific.status == "active" and ev["recertified"] and ks.match(s, "door") is specific


def test_a_pooled_certificate_is_shared_by_the_skills_of_its_scope():
    env = Corridor()
    ks = SkillSet(grammar(), certificate=Certificate(0.5, 0.5, scope="set"))
    a, _ = compiled(ks, env, 0)
    b, _ = compiled(ks, env, 0, goal=lambda st: st["pos"] == 2, name="half")
    for _ in range(2):
        ks.observe(a, True, alone=False)
    assert ks.usable(a) and ks.usable(b)                              # b rides on the pool a certified
    assert ks.report()["pools"]["*"]["certified"]


def test_without_a_grammar_a_skill_is_a_case_keyed_by_its_start_state():
    env = Corridor()
    cases = SkillSet(None, certificate=None)
    a, s0 = compiled(cases, env, 0)
    assert cases.match(s0, "door") is a                               # no certificate: acts at once (a control)
    assert cases.match({"pos": 0, "colour": 1}, "door") is None       # another start key: nothing transfers


# ------------------------------------------------------------------------------------------------ knowledge
def test_to_knowledge_writes_verified_skills_with_their_evidence_and_mirrors_retirement(tmp_path):
    env = Corridor()
    ks = SkillSet(grammar(), certificate=Certificate(0.5, 0.5))
    a, _ = compiled(ks, env, 0)
    store = KnowledgeStore(JSONLStorage(tmp_path / "k.jsonl"))
    ids = ks.to_knowledge(store, {"env": "corridor"})
    it = store.item(ids[a.id])
    assert it["kind"] == "skill" and store.status(ids[a.id]) == "active"
    assert it["body"] == {"kind": "plan", "goal": "door", "if": ["colour=0", "pos=0"], "do": ["right", "right", "right"]}
    ev = it["evidence"][-1]
    assert ev["plan"] == ["right", "right", "right"] and ev["start"] == a.start and ev["simulation"] == a.simulation
    assert set(ev["outcomes"]) >= {"ok", "fail"}
    n = len(store)
    assert ks.to_knowledge(store, {"env": "corridor"}) == ids and len(store) == n      # idempotent
    compiled(ks, env, 1)                                                               # widened: a new version
    ids2 = ks.to_knowledge(store, {"env": "corridor"})
    assert ids2[a.id] != ids[a.id] and store.status(ids2[a.id]) == "active"
    assert store.status(ids[a.id]) == "expired"                                        # the narrower version retired
    for _ in range(2):
        ks.observe(a, True, alone=False)
    while a.status != "retired":
        ks.observe(a, False, alone=True, state={"pos": 0, "colour": 2})
    ks.to_knowledge(store, {"env": "corridor"})
    assert store.status(ids2[a.id]) == "expired"
    for _ in range(2):
        ks.observe(a, True, alone=False)
    ks.to_knowledge(store, {"env": "corridor"})
    assert store.status(ids2[a.id]) == "active"                                        # re-certified: back
    assert store.verify()
    again = KnowledgeStore(JSONLStorage(tmp_path / "k.jsonl"))                         # the journal replays
    assert again.fingerprint() == store.fingerprint() and again.verify()


def test_unverified_skills_are_never_written_as_knowledge():
    env = Corridor()
    ks = SkillSet(grammar(), certificate=None)
    m = ExactModel(env)
    s = env.reset(0)
    sk = ks.compile("door", Plan(("left", "right"), (), 2.0, 0), s, verified=False)
    assert sk.source == "unverified"
    store = KnowledgeStore()
    assert ks.to_knowledge(store) == {} and len(store) == 0
    del m


# ------------------------------------------------------------------------------------------------ the loop
def run_episodes(learner, seeds, steps=20):
    return [learner.run(seed=s, steps=steps) for s in seeds]


def test_the_learner_plans_compiles_supervises_then_acts_alone():
    env = Corridor()
    L = SkillLearner(env, grammar(), {"door": at_goal}, horizon=6, budget=1000,
                     certificate=Certificate(0.5, 0.5))                 # need 2 clean outcomes
    eps = run_episodes(L, [0, 1, 2, 0, 1])
    e1, e2, e3, e4, e5 = eps
    assert e1["s2"] == 3 and e1["s1"] == 0 and e1["planner_calls"] == 1 and e1["compiled"] == 1
    assert e1["goals"] == ["door"] and e1["simulated"] > 0
    assert e2["planner_calls"] == 1 and len(L.skills.skills) == 1          # the same plan: generalized, not a new skill
    assert e3["planner_calls"] == 0 and e3["supervised"] == 1 and e3["s2"] == 3   # under supervision: System 2's
    assert e4["supervised"] == 1
    assert e5["s1"] == 3 and e5["s1_share"] == 1.0 and e5["planner_calls"] == 0 and e5["alone_runs"] == 1
    assert e5["alone_failures"] == 0 and e5["simulated"] == 0
    d = [r for r in L.decisions if r["episode"] == 5]
    assert {r["by"] for r in d} == {"s1"} and all(r["skill"] == L.skills.skills[0].id and r["certified"] for r in d)
    assert all(r["run_ok"] for r in d)
    rep = L.report()
    assert rep["totals"]["episodes"] == 5 and rep["skills"]["count"] == 1


def test_the_learner_never_steps_the_real_environment_while_deciding():
    env = Corridor()
    L = SkillLearner(env, grammar(), {"door": at_goal}, horizon=6, budget=1000)
    s = env.reset(0)
    before = L.model.fingerprint()
    a = L.act(s)
    assert a == "right" and L.model.fingerprint() == before
    L.observe(env.step(a))


def test_fallback_when_no_plan_is_found_and_plan_always_learns_nothing():
    env = Corridor()
    L = SkillLearner(env, grammar(), {"door": at_goal}, horizon=2, budget=1000)
    ep = L.run(seed=0, steps=4)
    assert ep["fallback"] >= 1 and ep["compiled"] in (0, 1)
    P = SkillLearner(env, grammar(), {"door": at_goal}, horizon=6, budget=1000, learn=False)
    eps = run_episodes(P, [0, 0, 0])
    assert all(e["s1"] == 0 and e["planner_calls"] == 1 for e in eps) and P.skills.skills == []


def test_unverified_plans_compile_without_the_judge():
    env = Corridor()

    def scrambled(model, state, goal, **kw):
        return Plan(("left", "left", "right"), (), 3.0, 0)

    L = SkillLearner(env, grammar(), {"door": at_goal}, planner=scrambled, verify_plans=False, certificate=False)
    ep = L.run(seed=0, steps=6)
    assert ep["goals"] == [] and L.skills.skills and L.skills.skills[0].source == "unverified"


def test_goals_are_worked_in_order_and_records_carry_the_goal():
    env = Corridor()
    L = SkillLearner(env, grammar(), [("half", lambda s: s["pos"] >= 2), ("door", at_goal)], horizon=6, budget=1000)
    ep = L.run(seed=0, steps=10)
    assert ep["goals"] == ["half", "door"] and ep["planner_calls"] == 2
    assert [r["goal"] for r in L.decisions] == ["half", "half", "door"]


# ------------------------------------------------------------------------------------------------ as System 2 of solvi.Agent
def test_as_s2_plugs_planning_on_the_exact_model_into_the_agent():
    import solvi
    env = Corridor()
    km = solvi.Knowledge()
    km.goal("door", done=at_goal)
    L = SkillLearner(env, grammar(), {"door": at_goal}, horizon=6, budget=1000)
    agent = solvi.Agent(env, knowledge=km, s2=as_s2(L))
    ep = agent.run(seed=0, steps=10)
    assert ep["done"] and ep["steps"] == 3 and ep["s2"] == 3 and ep["fallback"] == 0
    assert agent.replay()["ok"] == 3
    d = agent.decisions[0]
    assert d.by == "s2" and d.s2.mode == "exact-plan" and d.s2.record["plan"] == ["right", "right", "right"]
    assert not d.s2.record["cached"] and agent.decisions[1].s2.record["cached"]         # one search, then its steps
    assert L.skills.skills and L.skills.skills[0].goal == "door"                       # compiled for the learner too
