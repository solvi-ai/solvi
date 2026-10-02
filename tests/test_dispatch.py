"""solvi.dispatch: System 1 first, the slow path or a person by a recorded dispatch, within a budget — against a fake
chat-completions server (no network): which signal wakes the slow path, the budget per decision and in total, supervision
with recorded disagreements, refine and search as the slow path, the cost from the recorded tokens, replay without
calling the model, stored decisions."""
import math

import pytest
from test_llm import TEAMS, FakeLLM

from solvi import Answer, Catalog, Decision, Fail, JSONLStorage, Question, System
from solvi.dispatch import Budget, Cost, Dispatcher, SlowPath, cost_of
from solvi.llm import llm

URL = "http://127.0.0.1:9/v1"


def fast(min_confidence=0.8):
    """System 1: billing when the e-mail says "charged" (sure), shipping otherwise (unsure, 0.6)."""
    cat = Catalog()

    @cat.fn
    def charged(email):
        return "charged" in email.lower()

    @cat.rule("team")
    def team(charged, email):
        if charged:
            return Decision("billing", {"billing": 0.95, "shipping": 0.05})
        return Decision("shipping", {"billing": 0.4, "shipping": 0.6})
    return System(cat, [Question("team", "Which team?", Answer.choice(list(TEAMS)), min_confidence=min_confidence)])


def slow_llm(reply=None):
    """System 2: the same question answered by an LLM decision part (a fake server)."""
    srv = FakeLLM(reply=reply)
    model = llm(URL, "m", api_key="k", opener=srv, sleep=lambda s: None)
    part = model.decision("team", "Which team?", "email", TEAMS)
    cat = Catalog()
    return System(cat, [part.question(cat)]), srv


def test_a_sure_answer_of_system_1_is_given_alone_and_the_slow_path_is_not_called():
    s2, srv = slow_llm()
    d = Dispatcher(fast(), SlowPath(s2), price=(1.0, 2.0))
    res = d.ask({"email": "I was charged twice"})
    assert (res.answer, res.by, res.action) == ("billing", "s1", "accept")
    assert res.s2 is None and srv.bodies == [] and res.cost["s2"].calls == 0
    assert d.replay(res)["ok"]


def test_an_unsure_answer_wakes_the_slow_path_and_its_cost_comes_from_the_recorded_tokens():
    s2, srv = slow_llm()
    d = Dispatcher(fast(), SlowPath(s2), price=(1.0, 2.0))
    res = d.ask({"email": "I was charged for a parcel"})          # "charged": billing, sure
    assert res.by == "s1"
    res = d.ask({"email": "hello, my parcel is late"})
    assert (res.action, res.by, res.answer) == ("think", "s2", "shipping")
    assert "below its threshold" in res.reasons[0]
    assert res.candidates == {"s1": "shipping", "s2": "shipping"}
    c = res.cost["s2"]
    assert (c.calls, c.input_tokens, c.output_tokens) == (1, 120, 30)
    assert math.isclose(c.usd, (120 * 1.0 + 30 * 2.0) / 1e6)
    assert math.isclose(d.spent.usd, c.usd) and d.counts == {"s1": 1, "s2": 1, "human": 0}


def test_when_the_slow_path_cannot_answer_a_person_gets_it_with_both_candidates_and_the_reason():
    s2, srv = slow_llm(reply="not json at all")
    d = Dispatcher(fast(), SlowPath(s2))
    res = d.ask({"email": "hello"})
    assert (res.action, res.by, res.answer) == ("think", "human", None)
    assert res.candidates["s1"] == "shipping"
    assert any("the slow path did not answer" in r and "invalid LLM output" in r for r in res.reasons)
    assert d.replay(res)["ok"]


def test_no_budget_left_in_total_sends_the_input_to_a_person_without_calling_the_model():
    s2, srv = slow_llm()
    d = Dispatcher(fast(), SlowPath(s2), total=Budget(calls=1))
    assert d.ask({"email": "hello"}).by == "s2"
    res = d.ask({"email": "hello again"})
    assert (res.action, res.by, res.answer) == ("human", "human", None)
    assert any("no budget left in total" in r for r in res.reasons)
    assert len(srv.bodies) == 1
    assert d.replay(res)["ok"]


def test_a_run_expected_to_cost_more_than_the_budget_per_decision_is_not_started():
    s2, srv = slow_llm()
    d = Dispatcher(fast(), SlowPath(s2), price=(1.0, 2.0), budget=Budget(usd=0.0001))
    first = d.ask({"email": "hello"})                              # no estimate yet: it runs, and goes over
    assert first.by == "s2" and first.over_budget and "usd" in first.over_budget
    res = d.ask({"email": "hello again"})
    assert res.by == "human" and any("expected to cost more than the budget per decision" in r for r in res.reasons)
    assert len(srv.bodies) == 1
    assert d.replay(res)["ok"] and d.replay(first)["ok"]


def test_a_budget_in_dollars_needs_a_price():
    s2, _ = slow_llm()
    with pytest.raises(ValueError, match="needs price"):
        Dispatcher(fast(), SlowPath(s2), total=Budget(usd=1.0))
    with pytest.raises(ValueError, match="Budget usd"):
        Budget(usd=-1)


def test_supervision_checks_a_sampled_share_records_the_disagreement_and_keeps_system_1s_answer_by_default():
    s2, srv = slow_llm(reply='{"answer": "shipping", "probabilities": {"billing": 0.1, "shipping": 0.9}, "quote": ""}')
    d = Dispatcher(fast(), SlowPath(s2), supervise=1.0)
    res = d.ask({"email": "I was charged twice"})
    assert (res.action, res.by, res.answer) == ("check", "s1", "billing")
    assert res.disagreement == {"s1": "billing", "s2": "shipping", "s2_accepted": True}
    assert any("sampled for supervision" in r for r in res.reasons)
    assert d.disagreements([res])[0]["s2"] == "shipping"
    assert d.replay(res)["ok"]


@pytest.mark.parametrize("policy, by, answer", [("human", "human", None), ("s2", "s2", "shipping")])
def test_a_disagreeing_check_can_go_to_a_person_or_to_the_slow_paths_accepted_answer(policy, by, answer):
    s2, _ = slow_llm(reply='{"answer": "shipping", "probabilities": {"billing": 0.1, "shipping": 0.9}, "quote": ""}')
    d = Dispatcher(fast(), SlowPath(s2), supervise=1.0, on_disagree=policy)
    res = d.ask({"email": "I was charged twice"})
    assert (res.by, res.answer) == (by, answer)
    assert d.replay(res)["ok"]


def test_the_supervision_draw_is_a_reproducible_hash_and_samples_about_the_share_asked():
    s2, _ = slow_llm()
    d = Dispatcher(fast(), SlowPath(s2), supervise=0.3, seed=7)
    draws = [d.draw(f"h{i}", i) for i in range(2000)]
    assert draws == [d.draw(f"h{i}", i) for i in range(2000)]
    assert 0.25 < sum(x < 0.3 for x in draws) / 2000 < 0.35


def test_think_agree_gives_the_slow_answer_only_when_it_equals_what_system_1_would_have_said():
    s2, _ = slow_llm(reply='{"answer": "billing", "probabilities": {"billing": 0.9, "shipping": 0.1}, "quote": ""}')
    d = Dispatcher(fast(), SlowPath(s2), think="agree")
    res = d.ask({"email": "hello"})                                # System 1 would say shipping
    assert (res.by, res.answer) == ("human", None)
    assert any("a person decides" in r for r in res.reasons)
    assert d.replay(res)["ok"]


def test_a_signal_left_out_of_wake_sends_the_input_to_a_person():
    s2, srv = slow_llm()
    d = Dispatcher(fast(), SlowPath(s2), wake=("openset", "abstain"))
    res = d.ask({"email": "hello"})
    assert res.by == "human" and "guarantee does not wake the slow path" in res.reasons[-1]
    assert srv.bodies == []
    with pytest.raises(ValueError, match="unknown signal"):
        Dispatcher(fast(), SlowPath(s2), wake=("guess",))


def test_without_a_slow_path_the_dispatcher_is_system_1_under_its_guarantee_with_a_person_behind_it():
    d = Dispatcher(fast())
    assert d.ask({"email": "hello"}).by == "human"
    assert d.ask({"email": "charged"}).by == "s1"


def test_a_hard_check_that_forces_the_answer_is_never_rethought_or_supervised():
    cat = Catalog()

    @cat.check(hard=True, then={"team": "billing"})
    def not_an_invoice(email) -> bool:
        return "invoice" not in email

    @cat.rule("team")
    def team(email):
        return Decision("shipping", {"billing": 0.5, "shipping": 0.5})
    s1 = System(cat, [Question("team", "Which team?", Answer.choice(list(TEAMS)), min_confidence=0.8,
                               requires=["not_an_invoice"])])
    s2, srv = slow_llm()
    d = Dispatcher(s1, SlowPath(s2), supervise=1.0)
    res = d.ask({"email": "an invoice"})
    assert (res.by, res.answer, res.action) == ("s1", "billing", "accept") and srv.bodies == []


def test_a_low_agreement_wakes_the_slow_path_and_its_fact_is_computed_in_every_flow():
    cat = Catalog()

    @cat.fn
    def team_agreement(email) -> float:
        return 0.5

    @cat.rule("team")
    def team(email):
        return "billing"
    s1 = System(cat, [Question("team", "Which team?", Answer.choice(list(TEAMS)))])
    s2, _ = slow_llm()
    d = Dispatcher(s1, SlowPath(s2), agreement={"team_agreement": 2 / 3})
    res = d.ask({"email": "hello parcel"})
    assert res.action == "think" and "team_agreement = 0.5 < 0.667" in res.reasons[0]
    assert d.replay(res)["ok"]


def test_a_broken_constraint_between_the_questions_wakes_the_slow_path():
    cat = Catalog()

    @cat.rule("team")
    def team(email):
        return "billing"

    @cat.rule("refund")
    def refund(email):
        return "no"

    @cat.constraint
    def billing_refunds(team, refund):
        return team != "billing" or refund == "yes"
    s1 = System(cat, [Question("team", "Which team?", Answer.choice(list(TEAMS))),
                      Question("refund", "Refund?", Answer.yes_no())])
    s2, _ = slow_llm()
    d = Dispatcher(s1, SlowPath(s2), question="team")
    res = d.ask({"email": "parcel"})
    assert res.action == "think" and "billing_refunds" in res.reasons[0] and res.by == "s2"
    assert d.replay(res)["ok"]


class _Monitor:
    def __init__(self, at):
        self.at, self.seen = at, 0

    def observe(self, res, question=None):
        self.seen += 1
        return {"drift": self.seen >= self.at, "why": ["answered alone 90% → 40%"]}


def test_a_drift_flag_makes_the_slow_path_check_every_later_answer_until_reset():
    s2, _ = slow_llm(reply='{"answer": "shipping", "probabilities": {"billing": 0.1, "shipping": 0.9}, "quote": ""}')
    d = Dispatcher(fast(), SlowPath(s2), monitor=_Monitor(at=2), on_disagree="human")
    assert d.ask({"email": "charged"}).action == "accept"
    res = d.ask({"email": "charged again"})
    assert res.action == "check" and res.by == "human" and "DriftMonitor" in res.reasons[0]
    assert res.drift and d.replay(res)["ok"]
    d.reset_drift()
    d.monitor = None
    assert d.ask({"email": "charged"}).action == "accept"


def test_replay_does_not_call_the_model_and_catches_an_edited_answer_and_a_changed_dispatcher():
    s2, srv = slow_llm()
    d = Dispatcher(fast(), SlowPath(s2), price=(1.0, 2.0))
    res = d.ask({"email": "hello, my parcel"})
    calls = len(srv.bodies)
    assert res.answer == "shipping" and d.replay(res)["ok"] and len(srv.bodies) == calls
    res.answer = "billing"
    assert ("answer" in [w for w, _ in d.replay(res)["mismatches"]])
    res.answer = "shipping"
    res.by = "s1"
    assert not d.replay(res)["ok"]
    res.by = "s2"
    other = Dispatcher(fast(), SlowPath(s2), price=(2.0, 2.0))
    assert ("config" in [w for w, _ in other.replay(res)["mismatches"]])
    res.cost["s2"] = Cost(0.0, 0)
    res.s2.cost = Cost(0.0, 0)
    assert any(w == "cost" for w, _ in d.replay(res)["mismatches"])


def test_every_decision_is_stored_hash_chained_and_replays_from_the_store(tmp_path):
    s2, srv = slow_llm()
    store = JSONLStorage(tmp_path / "d.jsonl")
    d = Dispatcher(fast(), SlowPath(s2), price=(1.0, 2.0), supervise=0.5, storage=store)
    for e in ["charged", "hello", "parcel", "charged twice", "late"]:
        d.ask({"email": e})
    back = d.stored()
    assert [x.n for x in back] == [1, 2, 3, 4, 5] and all(x.stored_id for x in back)
    assert {x.by for x in back} <= {"s1", "s2", "human"}
    calls = len(srv.bodies)
    assert d.replay_all() == [] and len(srv.bodies) == calls
    assert store.verify()["ok"]


def _plan_system():
    """System 1 as the judge of a proposed slot: nobody is busy then."""
    cat = Catalog()

    @cat.check(hard=True, then={"ok": "no"})
    def free(slot) -> bool:
        return slot != "9:00" or Fail("Harold is busy at 9:00")

    @cat.rule("ok")
    def ok(free, slot):
        return "yes"
    return System(cat, [Question("ok", "Is the slot fine?", Answer.yes_no(), requires=["free"])])


def test_refine_as_the_slow_path_reasks_with_the_failed_checks_reasons_and_stops_on_the_budget():
    judge = _plan_system()
    seen = []

    def propose(state, rounds):
        seen.append([r.reasons for r in rounds])
        return "9:00" if not rounds else "10:00"
    slow = SlowPath(judge, propose=propose, into="slot", rounds=3)
    th = slow.run({}, "ok")
    assert th.accepted and th.answer == "yes" and len(th.record.rounds) == 2
    assert seen[1] == [["Harold is busy at 9:00"]]
    assert slow.replay(th)["ok"]
    stop = slow.run({}, "ok", budget=Budget(calls=0), expected_round=Cost(0.0, 1))
    assert not stop.accepted and stop.stopped and "no budget for another round" in stop.stopped


def test_search_as_the_slow_path_walks_the_space_through_the_checks():
    judge = _plan_system()
    slow = SlowPath(judge, space=["9:00", "10:00", "11:00"], into="slot")
    th = slow.run({}, "ok")
    assert th.mode == "search" and th.accepted and th.record.best == "10:00"
    assert slow.replay(th)["ok"]


def test_a_slow_path_is_refused_when_it_cannot_answer_the_question_or_is_misconfigured():
    s2, _ = slow_llm()
    with pytest.raises(ValueError, match="no question"):
        Dispatcher(fast(), SlowPath(s2, question="other"))
    with pytest.raises(ValueError, match="not both"):
        SlowPath(s2, propose=lambda s, r: 1, space=[1], into="x")
    with pytest.raises(ValueError, match="into="):
        SlowPath(s2, propose=lambda s, r: 1)


def test_cost_of_reads_llm_and_generator_records_and_a_price_function():
    s2, _ = slow_llm()
    res = s2.ask({"email": "parcel"})
    c = cost_of([res], lambda model, u: 0.5, ms=3.0, generated=[{"model": "g", "usage": {"input_tokens": 10,
                                                                                         "output_tokens": 5}}])
    assert (c.calls, c.usd, c.input_tokens, c.output_tokens, c.ms) == (2, 1.0, 130, 35, 3.0)
    assert cost_of([res], None).usd is None and cost_of([], None).usd == 0.0


class _Gate:
    """A stand-in open-set gate: threshold 0.9 on the confidence, a flag from the third decision on."""
    stateful, promise, by, min_group = True, "error among the answered ≤ 0.05 with new kinds of inputs", None, None
    report = {}

    def __init__(self):
        self.seen = 0

    def threshold_of(self, group=None):
        return 0.9, None, None

    def state(self):
        out = {"seen": self.seen, "level": 0.1}
        if self.seen >= 2:
            out.update(flag_at=2, why="signals below 0.5: CUSUM 9.0 ≥ 8.0")
        return out

    def record(self):
        return {"method": "open-set", "promise": self.promise, "error": 0.05}

    def fingerprint(self):
        return "gate-1"

    def observe(self, signal, result=None):
        self.seen += 1


def test_an_open_set_gate_below_its_threshold_thinks_and_its_flag_makes_later_answers_checked():
    s1 = fast(min_confidence=None)
    s1.guarantee("team", promise=_Gate())
    s2, _ = slow_llm()
    d = Dispatcher(s1, SlowPath(s2))
    res = d.ask({"email": "hello parcel"})                         # confidence 0.6 < 0.9
    assert res.action == "think" and res.reasons[0].startswith("System 1 is below its guarantee")
    assert dict(d.signals(res.s1))["openset"]
    assert d.ask({"email": "charged"}).action == "accept"          # 0.95: answered, no flag yet
    res = d.ask({"email": "charged twice"})
    assert res.action == "check" and "open-set gate flagged" in res.reasons[0]
    assert d.replay(res)["ok"]


def test_same_says_when_two_answers_agree_and_a_comparison_that_raises_is_not_an_agreement():
    s2, _ = slow_llm(reply='{"answer": "billing", "probabilities": {"billing": 0.9, "shipping": 0.1}, "quote": ""}')
    loose = Dispatcher(fast(), SlowPath(s2), think="agree", same=lambda a, b: True)
    assert loose.ask({"email": "hello"}).by == "s2"
    broken = Dispatcher(fast(), SlowPath(s2), think="agree", same=lambda a, b: 1 / 0)
    assert broken.ask({"email": "hello"}).by == "human"


def test_the_slow_paths_not_stated_is_an_answer_or_with_unknown_human_a_hand_off():
    from typing import Literal

    from solvi import Maybe, Unknown
    srv = FakeLLM(reply='{"answer": "not stated", "confidence": 0.9, "quote": ""}')
    model = llm(URL, "m", api_key="k", opener=srv, sleep=lambda s: None, ask="confidence")
    part = model.decision("team", "Which team?", "email", Maybe[Literal["billing", "shipping"]])
    cat = Catalog()
    s2 = System(cat, [part.question(cat)])
    res = Dispatcher(fast(), SlowPath(s2)).ask({"email": "hello"})
    assert res.by == "s2" and res.answer is Unknown
    d = Dispatcher(fast(), SlowPath(s2), unknown="human")
    res = d.ask({"email": "hello again"})
    assert res.by == "human" and "none of the options" in res.reasons[-1]
    assert d.replay(res)["ok"]
