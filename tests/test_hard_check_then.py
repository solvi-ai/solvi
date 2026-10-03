"""Hard checks with `then=` (1.0): the check is wired into the flow of every question its `then` names (no requires=),
without changing other questions' flows; a strategist that leaves it out is refused; `then` may be a function of facts
whose value — validated as an answer of the question — is recorded in the trace and re-run by replay."""
import asyncio
from typing import Literal

import pytest

from solvi import Catalog, Question, System
from solvi.core.provenance import part_fingerprint
from solvi.core.slow.refine import Fail
from solvi.core.store import open_storage


def free_side(free: list, facing: str) -> str:
    """The step to take instead: the first free side that is not where we face."""
    return next((d for d in free if d != facing), free[0])


def agent(then=None, extra_question=True):
    cat = Catalog()

    @cat.fn
    def free(walls: list) -> list:
        return [d for d in ("N", "E", "S", "W") if d not in walls]

    @cat.fn
    def danger(enemies: int) -> int:
        return enemies * 2

    @cat.check(hard=True, then={"step": then or free_side})
    def not_looping(history: list) -> bool:
        return Fail("the same room three times") if len(history) > 2 and len(set(history[-3:])) == 1 else True

    @cat.check(hard=True, then={"fight": "no"})
    def has_hearts(hearts: int) -> bool:
        return hearts > 0

    @cat.rule("step")
    def step(plan: str) -> Literal["N", "E", "S", "W"]:
        return plan

    @cat.rule("fight")
    def fight(danger: int) -> bool:
        return danger < 5
    qs = [Question("step", "Which way?")] + ([Question("fight", "Fight?")] if extra_question else [])
    return System(cat, qs)


LOOP = {"plan": "N", "facing": "N", "walls": ["E"], "history": [4, 4, 4], "hearts": 3, "enemies": 1}


# --- wiring
def test_a_then_check_on_given_facts_is_in_its_questions_flow():
    s = agent()
    res = s.ask(dict(LOOP, hearts=0))
    assert "has_hearts" in res.flow.per_question["fight"]          # reads only a given fact: 0.9 never ran it
    assert res["fight"].answer == "no" and res["fight"].status == "forced"
    assert "has_hearts" not in res.flow.per_question["step"]          # other questions keep their flows
    assert res.trace.replay(s, res.flow)["ok"]


def test_wiring_does_not_change_the_flows_of_other_questions():
    full = agent().ask(LOOP).flow.per_question
    alone = agent(extra_question=False).ask(LOOP).flow.per_question
    assert full["step"] == alone["step"]


def test_a_cost_strategist_keeps_the_wired_check():
    from solvi.core.plan.cost import CostStrategist
    s = agent()
    s2 = System(s.catalog, list(s.questions.values()), strategist=CostStrategist())
    res = s2.ask(dict(LOOP, hearts=0))
    assert res["fight"].answer == "no" and res["step"].answer == "S"


def test_a_strategist_that_drops_a_then_check_is_refused():
    class Drop:
        def plan(self, catalog, questions, init_keys, heads=None):
            from solvi.core.plan.strategist import plan
            flow = plan(catalog, questions, init_keys, heads)
            flow.per_question = {q: [f for f in fs if f != "has_hearts"] for q, fs in flow.per_question.items()}
            return flow
    s = agent()
    with pytest.raises(ValueError, match="has_hearts sets `then=` for 'fight'"):
        System(s.catalog, list(s.questions.values()), strategist=Drop())


# --- then as a function of facts
def test_then_function_gives_the_answer_from_facts():
    s = agent()
    res = s.ask(LOOP)
    r = res["step"]
    assert r.answer == "S" and r.status == "forced" and r.guard == "hard_check" and r.source == "not_looping"
    assert "the same room three times" in r.why and "free_side(free = ['N', 'S', 'W'], facing = 'N') = 'S'" in r.why
    rec = res.trace.records[-1]
    assert (rec.kind, rec.name, rec.value, rec.extra) == ("then", "then:step", "S", {"check": "not_looping"})
    assert set(rec.inputs) == {"free", "facing"}
    assert res.trace.replay(s, res.flow)["ok"]


def test_the_facts_it_reads_are_computed_after_an_early_exit():
    s = agent(extra_question=False)
    res = s.ask(LOOP)
    assert "free" in res.flow.per_question["step"]
    assert res.values["free"] == ["N", "S", "W"] and res["step"].answer == "S"


def test_then_function_under_aask_every_order_and_answers_of():
    s = agent(extra_question=False)
    for speculate in (False, True):
        res = asyncio.run(s.aask(LOOP, speculate=speculate))
        assert res["step"].answer == "S" and res.trace.replay(s, res.flow)["ok"], speculate
    res = s.ask(LOOP, early_exit=False)
    assert res["step"].answer == "S" and res.trace.replay(s, res.flow)["ok"]
    assert s.answers_of(res.trace, ["step"], res.flow)["step"].answer == "S"
    s.order = "learned"
    assert s.ask(LOOP)["step"].answer == "S"


def test_a_passing_check_does_not_run_the_function():
    calls = []

    def counted(free: list) -> str:
        calls.append(1)
        return free[0]
    res = agent(counted).ask(dict(LOOP, history=[1, 2, 3]))
    assert res["step"].answer == "N" and not calls
    assert not any(r.kind == "then" for r in res.trace.records)


def test_a_value_that_is_not_an_answer_abstains():
    res = agent(lambda free: "UP").ask(LOOP)
    r = res["step"]
    assert r.status == "abstain" and r.guard == "hard_check"
    assert "gave no answer" in r.why and "'UP' is not one of" in r.why
    assert res.trace.records[-1].error.startswith("its value is not an answer")


def test_a_function_that_raises_abstains():
    def broken(free: list) -> str:
        raise RuntimeError("no map")
    r = agent(broken).ask(LOOP)["step"]
    assert r.status == "abstain" and "RuntimeError: no map" in r.why


def test_typed_inputs_are_checked():
    def typed(hearts: int) -> str:
        return "N" if hearts > 1 else "S"
    s = agent(typed)
    assert s.ask(dict(LOOP, hearts="3"))["step"].answer == "N"       # coerced as for any typed part
    r = s.ask(dict(LOOP, hearts="many"))["step"]
    assert r.status == "abstain" and "type rejected" in r.why


def test_a_bool_answer_is_normalized():
    cat = Catalog()

    @cat.check(hard=True, then={"ok": lambda amount: amount < 10})
    def known(customer: str) -> bool:
        return customer != "x"

    @cat.rule("ok")
    def ok(amount: int) -> bool:
        return True
    s = System(cat, [Question("ok", "OK?")])
    assert s.ask({"customer": "x", "amount": 3})["ok"].answer == "yes"
    assert s.ask({"customer": "x", "amount": 30})["ok"].answer == "no"


def test_then_must_be_a_dict():
    cat = Catalog()
    with pytest.raises(TypeError, match="then= is"):
        @cat.check(hard=True, then=free_side)
        def c(x) -> bool:
            return True


# --- recorded and replayed
def test_stored_then_answers_replay_and_detect_a_changed_function(tmp_path):
    s = agent()
    s.storage = open_storage(str(tmp_path / "d.jsonl"))
    s.storage.catalog = s
    s.ask(LOOP)
    assert s.storage.verify()["ok"] and s.storage.replay_all(s) == []
    stored = next(iter(s.storage.iter()))
    assert s.storage.get(stored.id, s)["step"].answer == "S"
    s.catalog.parts["not_looping"].then["step"] = lambda free: free[-1]
    s.catalog.parts["not_looping"].then_parts = None
    bad = s.storage.replay_all(s)
    assert bad and any("then:step" == m[1] for m in bad[0]["mismatches"])


def test_the_then_function_is_in_the_checks_fingerprint():
    a, b = agent(), agent(lambda free, facing: free[-1])
    assert part_fingerprint(a.catalog.parts["not_looping"]) != part_fingerprint(b.catalog.parts["not_looping"])
    assert part_fingerprint(a.catalog.parts["has_hearts"]) == part_fingerprint(b.catalog.parts["has_hearts"])


def test_solvi_check_reports_a_literal_the_function_returns_that_is_not_an_answer():
    from solvi.check import lint

    def bad_side(free: list) -> str:
        if not free:
            return "UP"
        return free[0]
    rep = lint(agent(bad_side))
    assert rep.codes("error") == ["then_bad_answer"] and "'UP'" in rep.errors[0].message
    assert lint(agent()).findings == []
