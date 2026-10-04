"""solvi.core.knowledge.agenda and .failures: goals with done checks in code, gates that block actions and goals, order,
every state change journaled in the knowledge store, person overrides, the dry run that validates gates on recorded
successes; the failure memory — a hard check with an expiry and a bound, given to decisions as a fact so they replay."""
from typing import Literal

import pytest

from solvi import Catalog, Question, System
from solvi.core.knowledge import Agenda, FailureMemory, KnowledgeStore
from solvi.core.slow.refine import Fail


def shop_agenda(store=None, sticky=True):
    ag = Agenda(store, sticky=sticky)
    ag.goal("authenticated", done=lambda s: s.get("user") is not None)
    ag.goal("order_found", done=lambda s: s.get("order") is not None, requires=["authenticated"])
    ag.goal("exchanged", done=lambda s: s.get("status") == "exchange requested", requires=["order_found"],
            gates=["delivered"])
    ag.gate("auth_first", lambda s: s.get("user") is not None, blocks=["modify_*", "exchange_*"])
    ag.gate("delivered", lambda s: s.get("status") == "delivered" or Fail(f"status is {s.get('status')}"),
            blocks=["exchange_*"])
    return ag


def test_goals_open_blocked_done_and_order():
    ag = shop_agenda()
    s = {}
    ag.update(s)
    assert ag.open(s) == ["authenticated"] and ag.done() == set()
    assert ag.blocked(s)["order_found"] == ["requires authenticated"]
    s = {"user": "u1", "order": "A", "status": "pending"}
    changes = ag.update(s)
    assert changes["authenticated"] == ("open", "done") and ag.done() == {"authenticated", "order_found"}
    assert ag.blocked(s) == {"exchanged": ["delivered: status is pending"]}
    s["status"] = "delivered"
    assert ag.open(s) == ["exchanged"]
    s["status"] = "exchange requested"
    ag.update(s)
    assert ag.done() == {"authenticated", "order_found", "exchanged"} and ag.open(s) == []
    with pytest.raises(ValueError, match="not declared yet"):
        ag.goal("x", done=lambda s: True, requires=["nowhere"])
    with pytest.raises(TypeError):
        ag.goal("y", done="claimed by the model")                         # type: ignore[arg-type]


def test_no_action_past_a_gate_and_a_person_override_is_recorded():
    ks = KnowledgeStore()
    ag = shop_agenda(ks)
    ok, why = ag.allows({}, "exchange_delivered_order_items")
    assert not ok and why == ["auth_first", "delivered: status is None"]
    assert ag.allows({"user": "u", "status": "delivered"}, "exchange_delivered_order_items") == (True, [])
    assert ag.allows({}, "get_order_details") == (True, [])               # no gate applies
    ag.override("auth_first", by="lead", why="phone verified")
    assert ag.allows({"status": "delivered"}, "modify_address")[0]
    assert ks.find(r="override")[0]["source"] == "person"
    ag.clear_override("auth_first")
    assert not ag.allows({"status": "delivered"}, "modify_address")[0]


def test_every_state_change_is_journaled_as_facts_of_the_goal_rules():
    ks = KnowledgeStore()
    ag = shop_agenda(ks)
    ag.update({})
    ag.update({"user": "u"})
    ag.update({"user": "u"})                                               # nothing changed: nothing written
    states = ks.find(r="state", s="goal:authenticated")
    assert {(r["body"]["o"], r["status"]) for r in states} == {("open", "refuted"), ("done", "active")}
    goal_rules = ks.find(kind="rule", r="goal")
    assert len(goal_rules) == 3 and all(r["status"] == "active" and r["source"] == "spec" for r in goal_rules)
    ks.retract(ks.find(kind="rule", s="goal:authenticated")[0]["id"], why="goal dropped")
    assert all(r["status"] == "retracted" for r in ks.find(r="state", s="goal:authenticated"))   # the cascade
    assert ks.verify() and len(ag.history) == 5


def test_a_non_sticky_goal_can_be_undone():
    ag = shop_agenda(sticky=False)
    ag.update({"user": "u"})
    ag.update({})
    assert "authenticated" not in ag.done()


def test_the_dry_run_reports_how_often_a_gate_would_have_blocked_recorded_successes():
    ag = shop_agenda()
    ag.gate("all_items_listed", lambda s: s.get("listed_all", False), blocks=["exchange_*"])
    records = [({"user": "u", "status": "delivered", "listed_all": True}, "exchange_delivered_order_items"),
               ({"user": "u", "status": "delivered"}, "exchange_delivered_order_items"),
               ({"user": "u", "status": "delivered"}, "exchange_delivered_order_items"),
               ({"user": "u"}, "get_order_details")]
    rep = ag.dry_run(records)
    g = rep["gates"]["all_items_listed"]
    assert (rep["records"], rep["blocked_any"]) == (4, 2)
    assert (g["applies"], g["would_block"], g["examples"]) == (3, 2, [1, 2]) and g["share"] == pytest.approx(2 / 3)
    assert rep["gates"]["auth_first"]["would_block"] == 0


# --- the failure memory
def test_a_failed_plan_is_blocked_for_the_window_then_expires():
    fm = FailureMemory(window=3)
    fm.failed("open the door", why="locked")
    assert fm.check("open the door").verdict == "refuse" and fm.check("open the door").hard
    assert fm.check("go north").verdict == "accept"
    fm.step(2)
    assert "open the door" in fm.blocked()
    fm.step(1)
    assert fm.blocked() == {} and fm.check("open the door").verdict == "accept"


def test_the_bound_max_blocked_min_open_and_evidence_reopens():
    fm = FailureMemory(window=100, max_blocked=2, min_open=1)
    for i, p in enumerate(["a", "b", "c"]):
        fm.failed(p)
        fm.step()
    assert sorted(fm.blocked()) == ["b", "c"]                             # at most two: the most recent
    assert sorted(fm.blocked(options=["b", "c"])) == ["c"]                # one of the plans on offer stays open (oldest)
    fm.succeeded("c")
    assert sorted(fm.blocked()) == ["a", "b"]                             # c reopened by its success
    with pytest.raises(ValueError):
        FailureMemory(window=0)
    with pytest.raises(ValueError):
        FailureMemory(unit="hours")


def test_episode_units_and_plans_as_tuples():
    fm = FailureMemory(window=2, unit="episodes")
    fm.failed(("go", "north"))
    fm.step(50)                                                           # steps do not move an episode clock
    assert fm.check(["go", "north"]).verdict == "refuse"
    fm.new_episode()
    fm.new_episode()
    assert fm.check(("go", "north")).verdict == "accept"


def test_the_hard_check_in_a_catalog_replays(tmp_path):
    ks = KnowledgeStore()
    fm = FailureMemory(window=5, store=ks)
    cat = Catalog()
    fm.install(cat, plan="plan", then={"act": "ask_person"})

    @cat.rule("act")
    def act(plan: str) -> Literal["push", "pull", "ask_person"]:
        return plan
    s = System(cat, [Question("act", "What to do?")])
    assert s.ask({"plan": "push", **fm.given()})["act"].answer == "push"
    fm.failed("push", why="it did not move")
    res = s.ask({"plan": "push", **fm.given()})
    assert res["act"].answer == "ask_person" and res["act"].status == "forced"
    (failed,) = [c for c in res.checks if c.status == "failed"]
    assert "push failed at step 0 (it did not move); not repeated before 5" in failed.reason
    assert res.trace.replay(s, res.flow)["ok"]
    assert any(r["op"] == "note" and r.get("failure_memory") == "failed" for r in ks.journal)
    with pytest.raises(ValueError):
        FailureMemory.check_function(plan="not a name")
