"""The budget of solvi.core.dispatch (Budget, Cost) on solvi.core.slow.generate and solvi.core.slow.refine: tokens, dollars, latency recorded per
request and per round; a stop by the total or per decision before a request or a round is sent; the record in the trace,
in the refinement and in the system report."""
import io
import json

import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, JSONLStorage, Question, System
from solvi.core.costs import Budget, BudgetStop, Cost
from solvi.core.slow.generate import generator
from solvi.core.slow.refine import Fail, Refinement, refine
from solvi.core.store.sysreport import system_report


class FakeServer:
    """Replies in turn (the last one repeats), each with 100 input and 20 output tokens."""

    def __init__(self, *replies):
        self.replies, self.n = list(replies), 0

    def __call__(self, req, timeout=None):
        self.n += 1
        body = json.loads(req.data.decode())
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return io.BytesIO(json.dumps({"model": body["model"], "choices": [{"message": {"role": "assistant", "content": r},
                                      "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode())


def gen(*replies, **kw):
    srv = FakeServer(*replies)
    return generator("http://127.0.0.1:9/v1", "m-1", opener=srv, sleep=lambda s: None, **kw), srv


def test_budget_and_cost_are_the_dispatchers_and_take_tokens():
    from solvi.core import dispatch
    assert dispatch.Budget is Budget and dispatch.Cost is Cost and dispatch.BudgetStop is BudgetStop
    assert Budget(usd=1.0).to_dict() == {"usd": 1.0, "calls": None, "ms": None}      # Dispatcher.config unchanged
    b = Budget(tokens=200)
    assert b.to_dict()["tokens"] == 200 and Budget.from_dict(b.to_dict()) == b
    c = Cost(0.0, 1, 5.0, 100, 20)
    assert c.tokens == 120 and b.over(c) is None and "tokens 120 + 120 expected > 200" == b.over(c, c)
    assert Budget(tokens=300).used_up(c + c) is None and Budget(tokens=240).used_up(c + c) == "tokens 240 of 240 used"
    assert Budget(usd=0.1).over(Cost(None, 3)) is None                                 # unpriced: no dollar verdict
    with pytest.raises(ValueError):
        Budget(tokens=-1)


def test_a_generator_records_tokens_dollars_and_latency_and_keeps_what_it_spent():
    g, _ = gen("SELECT 1;", price=(1.0, 2.0))
    out = g.generate("Write a query")
    u = out.meta["usage"]
    assert u["input_tokens"] == 100 and u["usd"] == pytest.approx((100 * 1.0 + 20 * 2.0) / 1e6)
    assert out.meta["ms"] >= 0 and "budget" not in out.extra                             # no limit: no budget line
    assert g.spent.calls == 1 and g.spent.tokens == 120 and g.spent.usd == pytest.approx(u["usd"])
    assert g.expected().calls == 1


def test_a_generator_stops_before_a_request_its_total_would_not_cover():
    g, srv = gen("ok", total=Budget(tokens=250))
    g.generate("a")
    g.generate("b")                                       # 240 spent, one more expected at 120: over
    with pytest.raises(BudgetStop, match=r"no budget left in total \(tokens 240 \+ 120 expected > 250\)"):
        g.generate("c")
    assert srv.n == 2 and g.spent.calls == 2


def test_the_per_decision_budget_limits_one_call_and_records_how_far_over_it_went():
    g, srv = gen("ok", budget=Budget(calls=2))
    out = g.sample("a", k=4)                              # one decision: at most two requests
    assert sum(v is not None for v in out.value) >= 2 and srv.n <= 4
    assert out.extra["budget"]["budget"] == {"usd": None, "calls": 2, "ms": None}
    g2, _ = gen("ok", budget=Budget(tokens=50))
    one = g2.generate("a")                                # the first request has no expected cost: it is sent
    assert one.extra["budget"]["over"] == "tokens 120 > 50"
    with pytest.raises(ValueError, match="needs price="):
        gen("ok", budget=Budget(usd=0.01))


def test_a_budget_stop_inside_a_catalog_fails_the_part_and_its_question_abstains_with_the_cause():
    g, _ = gen("SELECT 1", total=Budget(calls=1))
    cat = Catalog()
    cat.fn(g.part("sql", lambda question: f"SQL for {question}"))

    @cat.rule("ok")
    def ok(sql) -> bool:
        return sql.startswith("SELECT")
    s = System(cat, [Question("ok", "?", Answer.yes_no())])
    assert s.ask({"question": "a"})["ok"].answer == "yes"
    res = s.ask({"question": "b"})
    rec = next(r for r in res.trace.records if r.name == "sql")
    assert res["ok"].status == "abstain" and "BudgetStop: no budget left in total" in rec.error


# --- refine
class Slot(BaseModel):
    day: str


def _judge(storage=None):
    cat = Catalog()

    @cat.check(hard=True, then={"ok": "no"})
    def weekday(proposal) -> bool:
        return True if proposal["day"] in ("Mon", "Tue") else Fail(f"{proposal['day']} is not a weekday")

    @cat.rule("ok")
    def ok(weekday) -> bool:
        return True
    return System(cat, [Question("ok", "?", Answer.yes_no(), requires=["weekday"])], storage=storage)


def test_refine_records_each_rounds_cost_and_stops_when_the_next_round_would_not_fit(tmp_path):
    g, srv = gen('{"day": "Sun"}', '{"day": "Sat"}', '{"day": "Mon"}', price=(1.0, 2.0))
    store = JSONLStorage(tmp_path / "s.jsonl")
    s = _judge(store)
    run = refine(s, {}, "ok", g.proposer("a weekday?", parse=json.loads), rounds=5, budget=Budget(tokens=250))
    assert not run.accepted and len(run.rounds) == 2 and srv.n == 2
    assert run.stopped == "no budget for another round (tokens 240 + 120 expected > 250)"
    assert run.escalation.startswith(run.stopped) and run.cost.tokens == 240 and run.over_budget is None
    assert [r.cost.calls for r in run.rounds] == [1, 1] and run.rounds[0].cost.usd == pytest.approx(1.4e-4)
    back = Refinement.from_dict(json.loads(run.to_json()), catalog=s.catalog)
    assert back.budget == Budget(tokens=250) and back.stopped == run.stopped and back.rounds[1].cost.to_dict() == run.rounds[1].cost.to_dict()
    rep = back.replay(s)
    assert rep["ok"], rep["mismatches"]
    d = json.loads(run.to_json())
    d["rounds"][0]["cost"]["input_tokens"] = 1
    assert any(m[1] == "cost" for m in Refinement.from_dict(d, catalog=s.catalog).replay(s)["mismatches"])
    d = json.loads(run.to_json())
    d["budget"]["tokens"] = 10_000                    # the recorded stop does not follow from this budget
    assert any(m[1] == "budget" for m in Refinement.from_dict(d, catalog=s.catalog).replay(s)["mismatches"])
    loop = store.record(run.stored_id)
    assert loop["kind"] == "refine" and loop["stopped"] == run.stopped and loop["cost"]["calls"] == 2
    assert [r["stored_id"] for r in loop["rounds"]] == [r.stored_id for r in run.rounds] and store.verify()["ok"]
    c = system_report(store).cost["refine"]
    assert c["loops"] == 1 and c["escalated"] == 1 and c["stopped_by_budget"] == 1 and c["rounds"] == 2
    assert c["proposals"]["calls"] == 2 and c["proposals"]["usd"] == pytest.approx(2.8e-4)
    assert "refinement loops: 1 (0 accepted, 1 escalated, 1 stopped by their budget" in str(system_report(store))


def test_a_generators_total_ends_a_refinement_as_a_proposer_failure():
    g, _ = gen('{"day": "Sun"}', total=Budget(calls=1))
    run = refine(_judge(), {}, "ok", g.proposer("a weekday?", schema=Slot), rounds=3)
    assert not run.accepted and len(run.rounds) == 2 and "BudgetStop: no budget left in total" in run.escalation


def test_a_dollar_budget_without_known_dollars_stops_instead_of_guessing():
    g, _ = gen('{"day": "Sun"}')                          # no price anywhere
    run = refine(_judge(), {}, "ok", g.proposer("a weekday?", schema=Slot), rounds=3, budget=Budget(usd=1.0))
    assert len(run.rounds) == 1 and "dollars unknown" in run.stopped
    with pytest.raises(TypeError):
        refine(_judge(), {}, "ok", g.proposer("x", schema=Slot), budget={"usd": 1})
