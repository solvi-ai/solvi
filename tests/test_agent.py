"""The high level of the knowledge memory: solvi.Knowledge (tell, retract, goal, observe, report), solvi.Agent on an
Environment (System 1 on what the knowledge predicts, System 2 a search, hard checks, replay, knowledge carried across
episodes, drift, protection vs justified risk), build(knowledge=) and Guard(knowledge=), and example 25."""
import random

import pytest

import solvi
from solvi import Answer, Catalog, Knowledge, Question
from solvi.core import Outcome
from solvi.core.knowledge import Prediction, Protect, RiskBudget
from examples_loader import load

E = load("25_environment_agent")


def crafting_knowledge(**kw):
    return E.goals(Knowledge(vocabulary=E.VOCABULARY, **kw))


# ------------------------------------------------------------------------------------------------ Knowledge
def test_knowledge_tell_refuses_the_systems_own_answers_and_retract_cascades():
    km = Knowledge()
    a = km.tell({"s": "c1", "r": "tier", "o": "vip"}, source="person", by="crm")
    assert a is not None and km.store.status(a) == "active"
    assert km.tell(("c2", "tier", "vip"), source="model") is None          # never the system's own answer
    b = km.store.add("fact", {"s": "c1", "r": "discount", "o": 0.1}, source="person", by="crm", derived_from=[a])
    out = km.retract(a, why="wrong customer", by="ann")
    assert out["status"][a][1] == "retracted" and out["status"][b][1] == "retracted"
    assert out["answer_changes"] is None                                  # no decisions given to re-run
    with pytest.raises(ValueError, match="a fact is"):
        km.tell({"s": "c1"}, source="person")
    assert km.store.verify()


def test_knowledge_options_and_report():
    km = Knowledge(failures={"window": 3})
    assert km.actions is None and km.failures.window == 3
    with pytest.raises(ValueError, match="not both"):
        Knowledge(vocabulary=E.VOCABULARY, actions=E.OutcomeRates(E.VOCABULARY))
    with pytest.raises(TypeError, match="ActionModel"):
        Knowledge(actions=object())
    km = crafting_knowledge()
    assert list(km.agenda.goals) == E.OPS
    rep = km.report()
    assert set(rep) >= {"store", "agenda", "skills", "maps", "actions"} and rep["agenda"]["goals"] == E.OPS
    assert "6 goals" in repr(km)


def test_knowledge_observe_learns_skills_map_facts_and_drops_a_carried_map_on_contradiction():
    km = crafting_knowledge()
    s0 = {"place": "p0", "here": "tree", "wood": False, "stone": False, "pickaxe": False, "stone_pickaxe": False,
          "tables": [], "achieved": []}
    km._begin("m")
    km.agenda.update(s0)
    s1 = dict(s0, wood=True, achieved=["collect_wood"])
    info = km.observe(s0, "collect_wood", Outcome(s1), at="p0", to="p0", map="m")
    assert info["goals_done"] == ["collect_wood"] and len(info["skills"]) == 1 and info["prediction"]["verdict"] == "unknown"
    assert km.skills("collect_wood") == ["collect_wood"] and km.where("m", "collect_wood") == ["p0"]
    km.observe(s1, "exit0", Outcome(dict(s1, place="p1")), at="p0", to="p1", map="m")
    assert km.edges("m", "p0") == {"collect_wood": "p0", "exit0": "p1"}
    km._end()
    km._begin("m")                                                        # the next episode: the map is carried
    info = km.observe(s1, "exit0", Outcome(dict(s1, place="p2")), at="p0", to="p2", map="m")
    assert info["drift"] is not None                                      # contradicted: the carried map is dropped
    assert km.edges("m", "p0") == {"exit0": "p2"}                         # the new observation is a fact, the rest hints
    assert km.where("m", "collect_wood") == []                            # older map facts are hints until confirmed
    assert km.skills("collect_wood") == ["collect_wood"]                  # rules are carried
    assert km.store.verify()


# ------------------------------------------------------------------------------------------------ Agent
def test_the_same_world_again_needs_fewer_slow_decisions_and_every_decision_replays(tmp_path):
    km = crafting_knowledge()
    agent = solvi.Agent(E.Crafting(), knowledge=km, key=E.place, storage=tmp_path / "agent.jsonl")
    first, second = agent.run(seed=7, steps=150), agent.run(seed=7, steps=150)
    assert len(first["goals_done"]) == 6 and len(second["goals_done"]) == 6
    assert second["s2"] < first["s2"] / 5 and second["s1"] > second["s2"]
    assert second["steps"] < first["steps"] and second["refused"] <= first["refused"]
    rep = agent.replay()
    assert rep["decisions"] == first["steps"] + second["steps"] and rep["ok"] == rep["decisions"]
    tot = agent.report()["totals"]
    assert tot["episodes"] == 2 and tot["s1"] + tot["s2"] + tot["fallback"] == tot["steps"]
    assert km.store.verify()
    assert agent.system is agent.dispatcher.system and agent.knowledge is km


def test_a_stored_decision_that_was_edited_does_not_replay(tmp_path):
    agent = solvi.Agent(E.Crafting(), knowledge=crafting_knowledge(), key=E.place)
    agent.run(seed=7, steps=20)
    d = agent.decisions[3]
    opts = d.s1.trace.init["options"]
    i = int(d.answer)
    opts[i]["blocks"] = ["gate forged: by hand"]                         # the record no longer gives that answer
    rep = agent.replay()
    assert rep["ok"] == rep["decisions"] - 1 and rep["mismatches"][0][0] == d.n


def test_system_one_acts_only_on_predicted_accepts_and_hard_checks_hold():
    km = crafting_knowledge()
    km.agenda.gate("no_stone_without_pickaxe", lambda s: s["pickaxe"], blocks=["collect_stone"])
    agent = solvi.Agent(E.Crafting(), knowledge=km, key=E.place)
    agent.run(seed=7, steps=150)
    agent.run(seed=7, steps=150)
    for d in agent.decisions:
        o = d.s1.trace.init["options"][int(d.answer)] if d.answer is not None else None
        if d.by == "s1":
            assert o["choice"] == "take" and o["goal"] is not None and not o["blocks"]
        if o is not None:
            assert not any(b.startswith("gate ") for b in o["blocks"])   # no action past a gate, in any path


def test_a_new_world_keeps_the_rules_and_relearns_the_map():
    km = crafting_knowledge()
    agent = solvi.Agent(E.Crafting(), knowledge=km, key=E.place)
    fresh = solvi.Agent(E.Crafting(), knowledge=crafting_knowledge(), key=E.place).run(seed=8, steps=150)
    agent.run(seed=7, steps=150)
    new = agent.run(seed=8, steps=150)
    assert new["refused"] < fresh["refused"]                              # the action model was carried
    assert km.report()["maps"].keys() == {"seed 7", "seed 8"}


def test_protection_vs_justified_risk_and_the_gate_is_never_traded():
    out = E.part2(episodes=20)
    assert out["risk"]["iron"] > out["protect"]["iron"] and out["risk"]["takes"] > 0 and out["protect"]["takes"] == 0
    assert out["risk"]["falls"] >= out["protect"]["falls"]               # justified risk is risk
    assert not any(E.gate_violations(out))


def test_risk_budget_takes_a_refusal_only_within_its_budget_and_never_a_hard_one():
    pol = RiskBudget(max_risk_per_episode=0.5, min_gain_ratio=1.0, min_support=1)
    km = Knowledge(actions=_Fixed({"jump": Prediction("refuse", 0.4, 5, "fell before"),
                                   "lava": Prediction("refuse", 0.1, 5, "lava kills", hard=True)}))
    agent = solvi.Agent(_Line(), knowledge=km, risk=pol, gain=lambda s, a: 1.0 if a == "jump" else 0.0)
    ep = agent.run(seed=0, steps=6)
    taken = [s["action"] for s in agent.steps]
    assert "lava" not in taken and taken.count("jump") == 1 and ep["risky_takes"] == 1
    protect = solvi.Agent(_Line(), knowledge=Knowledge(actions=km.actions), gain=lambda s, a: 1.0 if a == "jump" else 0.0)
    protect.run(seed=0, steps=6)
    assert "jump" not in [s["action"] for s in protect.steps] and isinstance(protect.risk, Protect)


def test_system_two_budget_and_the_fallback():
    agent = solvi.Agent(E.Crafting(), knowledge=crafting_knowledge(), key=E.place, budget=5)
    ep = agent.run(seed=7, steps=30)
    assert ep["s2"] == 5 and ep["fallback"] == ep["steps"] - ep["s1"] - 5
    assert agent.replay()["ok"] == ep["steps"]


def test_the_agent_checks_what_it_is_given():
    with pytest.raises(TypeError, match="Environment"):
        solvi.Agent(object(), knowledge=Knowledge())
    with pytest.raises(TypeError, match="solvi.Knowledge"):
        solvi.Agent(E.Crafting(), knowledge={})
    with pytest.raises(TypeError, match="budget"):
        solvi.Agent(E.Crafting(), knowledge=Knowledge(), budget="lots")
    agent = solvi.Agent(E.Crafting(), knowledge=Knowledge(), key=E.place)
    with pytest.raises(ValueError, match="follows act"):
        agent.observe(True)


def test_without_an_action_model_the_map_says_what_worked_here():
    km = E.goals(Knowledge())
    agent = solvi.Agent(E.Crafting(), knowledge=km, key=E.place)
    first, second = agent.run(seed=7, steps=150), agent.run(seed=7, steps=150)
    assert second["s2"] < first["s2"] and second["s1"] > 0


class _Fixed:
    """An action model that says the same thing every time (to test the risk policy)."""

    def __init__(self, preds):
        self.preds = preds

    def observe(self, state, action, args, accepted, effect=None):
        return self.predict(state, action, args)

    def predict(self, state, action, args):
        return self.preds.get(action, Prediction("accept", 0.0, 9, "fine"))

    def fingerprint(self):
        return "fixed"


class _Line:
    """Three actions per step: wait (nothing happens), jump (refused, risky), lava (refused, hard)."""

    def reset(self, seed=None):
        self.t = 0
        return {"t": 0}

    def actions(self, state):
        return ["jump", "lava", "wait"]

    def step(self, action):
        self.t += 1
        return Outcome({"t": self.t}, accepted=action == "wait")


# ------------------------------------------------------------------------------------------------ build and Guard
def test_build_gives_every_decision_the_knowledge_and_a_told_fact_changes_the_answer():
    km = Knowledge()
    km.tell({"s": "c1", "r": "tier", "o": "vip"}, source="person", by="crm")
    cat = Catalog()

    @cat.rule("refund")
    def refund(customer, amount, knowledge) -> str:
        return "yes" if amount < 50 or Knowledge.value(knowledge, customer, "tier") == "vip" else "no"

    rng = random.Random(0)
    ex = []
    for _ in range(200):
        c, a = rng.choice(["c1", "c2", "c3"]), rng.uniform(0, 100)
        ex.append(({"customer": c, "amount": a}, "yes" if a < 50 or c == "c1" else "no"))
    s = solvi.build(Question("refund", "Refund?", Answer.yes_no()), ex, catalog=cat, max_risk=0.05, knowledge=km)
    assert "Knowledge:" in s.explain()
    assert s.ask({"customer": "c2", "amount": 80}).answer == "no"
    km.tell({"s": "c2", "r": "tier", "o": "vip"}, source="person", by="crm")
    r = s.ask({"customer": "c2", "amount": 80})
    assert r.answer == "yes" and r.s1.trace.init["knowledge"]["at"] == len(km.store.journal)
    assert s.replay(r)["ok"]
    own = solvi.build(Question("refund", "Refund?", Answer.yes_no()), ex, catalog=cat, max_risk=0.05,
                      knowledge=lambda st: km.snapshot(about=st.get("customer")))
    assert own.ask({"customer": "c2", "amount": 80}).s1.trace.init["knowledge"]["query"]["about"] == "c2"
    with pytest.raises(TypeError, match="knowledge="):
        solvi.build(Question("refund", "Refund?", Answer.yes_no()), ex, catalog=cat, max_risk=0.05, knowledge=3)


def test_guard_turns_action_model_refusals_and_agenda_gates_into_hard_checks():
    km = Knowledge(vocabulary={"status": lambda st, a: st["orders"][a["order_id"]]})
    guard = solvi.Guard(knowledge=km)

    @guard.tool
    def cancel(order_id: str) -> str:
        return "cancelled"

    d = guard.call({"name": "cancel", "arguments": {"order_id": "A"}}, facts={"orders": {"A": "delivered"}})
    assert d.outcome == "allow"                                           # unknown: the other checks decide
    guard.observe(d, accepted=False)                                      # the shop refused it
    d = guard.call({"name": "cancel", "arguments": {"order_id": "B"}}, facts={"orders": {"B": "pending"}})
    guard.observe(d, accepted=True)
    d = guard.check({"name": "cancel", "arguments": {"order_id": "C"}}, facts={"orders": {"C": "delivered"}})
    assert d.outcome == "deny" and "status='delivered'" in d.reasons[0] and d.replay()["ok"]
    assert guard.check({"name": "cancel", "arguments": {"order_id": "D"}},
                       facts={"orders": {"D": "pending"}}).outcome == "allow"
    km.agenda.gate("authenticated", lambda st: st.get("user") is not None, blocks=["cancel"])
    guard._systems.clear()
    d = guard.check({"name": "cancel", "arguments": {"order_id": "D"}}, facts={"orders": {"D": "pending"}})
    assert d.outcome == "deny" and "gate authenticated" in d.reasons[0]
    ok = guard.check({"name": "cancel", "arguments": {"order_id": "D"}}, facts={"orders": {"D": "pending"}, "user": "u1"})
    assert ok.outcome == "allow"
    with pytest.raises(TypeError, match="solvi.Knowledge"):
        solvi.Guard(knowledge=object())


def test_a_guard_without_knowledge_is_unchanged():
    guard = solvi.Guard()

    @guard.tool
    def cancel(order_id: str) -> str:
        return "cancelled"

    d = guard.check({"name": "cancel", "arguments": {"order_id": "A"}})
    assert d.outcome == "allow" and "action_prediction" not in d.response.trace.init
    assert "action_model_allows" not in guard.system("cancel").catalog.parts


# ------------------------------------------------------------------------------------------------ example 25
def test_example_25_prints_its_numbers(capsys):
    agent, runs = E.part1()
    (_, r1), (_, r2), (_, new) = runs
    assert r2["s2"] < r1["s2"] and new["s2"] < r1["s2"]
    assert agent.replay()["ok"] == r1["steps"] + r2["steps"] + new["steps"]
    out = capsys.readouterr().out
    assert "every decision replays" in out and "journal verifies: True" in out
