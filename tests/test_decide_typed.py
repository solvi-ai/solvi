"""solvi.decide, the typed decider contract: questions declared by types (choice, multi, score, noul), text or JSON / pydantic
state (state_text, "kv1"), probabilities + calibrated confidence + act / escalate per question, several questions in one
forward pass (flow.batches), adapt / fit / teach per question type, the checkpoint capability fields and backward
compatibility with the 'l14b_decider v1' format; also the quote source of hand-written extract parts. Stub scorers stand in for the network; a slow test runs the real L14d
checkpoint when it is present."""
import copy
import json
import os
import subprocess
import sys
from datetime import date
from enum import Enum
from typing import Literal, Optional

import numpy as np
import pytest
from pydantic import BaseModel, Field

from solvi import Answer, Catalog, Decision, Question, Quote, Response, System
from solvi.decide import (BlockUnsupported, DecideModel, Item, _basis, _Spec, act_features, block_masks, capabilities, pass_prompt, prompt,
                          state_text)
from solvi.provenance import digest
from solvi.typed import Scale, question_kind

TEAMS = ["billing", "technical", "shipping"]
LEVELS = ["low", "medium", "high", "critical"]
TOPICS = ["refund", "delay", "bug"]
KW = {"billing": ["charged", "refund", "invoice"], "technical": ["crash", "error", "bug"],
      "shipping": ["parcel", "delivery", "tracking"], "refund": ["refund"], "delay": ["late", "delay"], "bug": ["crash", "bug"]}
ANGRY = ("angry", "furious", "terrible")

META2 = {"format": "l14f typed v1", "multi_question": {"layout": "block", "max_questions": 8},
         "temperature": {"choice": 1.0, "multi": 1.0, "score": 1.0, "noul": 1.0}, "act": {"threshold": 0.5}}


class StubScorer:
    """Keyword logits for every question kind; an act logit (negative when the text says "maybe"); logits_pass scores
    several questions per call. Records every call so tests can count forward passes."""
    model_id = "test/stub-decider"

    def __init__(self, act=True, version="1", passes=True, fail_pass=False):
        self.act, self.version, self.fail_pass = act, version, fail_pass
        self.calls = []                     # ("seq", n items) or ("pass", [n questions per pass])
        self.texts = []                     # every input the network read
        if not passes:
            self.logits_pass = None

    def fingerprint(self):
        return f"stub-{self.version}"

    def _z(self, it, text):
        low = text.lower()
        opts = list(it.options)
        if opts in (["yes", "no"], ["true", "false"]):
            y = 2.0 if any(w in low for w in ANGRY) else -2.0
            return [y] if it.mode == "noul" else [y, 0.0]          # native noul ("true" / "false"): one yes log-odds
        if opts == LEVELS or opts == ["1", "2", "3", "4", "5"]:
            u = min(len(opts) - 1, low.count("!") + 2 * ("urgent" in low))
            return [-2.0 * abs(k - u) for k in range(len(opts))]
        z = [2.0 * sum(low.count(k) for k in KW.get(o, [])) for o in opts]
        return [v - 1.0 for v in z] if it.mode == "multi" else z

    def _out(self, it, text):
        z = np.array(self._z(it, text))
        if not self.act:
            return z
        return {"logits": z, "act": -3.0 if "maybe" in text.lower() else 3.0}

    def logits(self, items):
        self.calls.append(("seq", len(items)))
        self.texts += [it.text for it in items]
        return [self._out(it, it.text) for it in items]

    def logits_pass(self, passes):
        if self.fail_pass == "unsupported":
            raise BlockUnsupported("no block inputs")
        if self.fail_pass:
            raise ValueError("the questions do not fit together")
        self.calls.append(("pass", [len(p.items) for p in passes]))
        self.texts += [p.text for p in passes]
        return [[self._out(it, p.text) for it in p.items] for p in passes]


def v2(**kw):
    meta = copy.deepcopy(META2)
    meta.update(kw.pop("meta", {}))
    return DecideModel(StubScorer(**kw), meta=meta)


class Urgency(str, Enum):
    low = "low"
    high = "high"


class Ticket(BaseModel):
    team: Literal["billing", "technical", "shipping"] = Field(description="Which team should handle it?")
    urgency: Scale[Literal["low", "medium", "high", "critical"]] = Field(description="How urgent is it?")
    angry: bool = Field(description="Is the customer angry?")
    topics: list[Literal["refund", "delay", "bug"]] = Field(description="What does it mention?")


# --------------------------------------------------------------------------------------------------- state serialization
class Customer(BaseModel):
    name: str
    tier: Literal["free", "gold"]
    since: date


class Order(BaseModel):
    customer: Customer
    items: list[dict]
    note: Optional[str] = None
    total: float


ORDER = {"customer": {"name": "Anna", "tier": "gold", "since": date(2024, 1, 5)},
         "items": [{"sku": "A-1", "qty": 2}, {"sku": "B 7", "qty": 1}], "note": "two lines\nof text", "total": 25.5}


def test_state_text_is_the_l14f_paths_serialization():
    assert state_text(ORDER) == (
        "customer.name: Anna\ncustomer.tier: gold\ncustomer.since: 2024-01-05\n"
        "items[0].sku: A-1\nitems[0].qty: 2\nitems[1].sku: B 7\nitems[1].qty: 1\n"
        "note: two lines of text\ntotal: 25.5")
    assert state_text(Order(**ORDER)) == state_text(ORDER)                      # a model and the equal dict: same text
    assert state_text("a plain text") == "a plain text" and state_text(Quote("q", 0, 1)) == "q"
    odd = {"b": None, "a": True, "c d": [], "e.f": {}, "g": float("nan"), "h": "", " i": " padded ", "j": Urgency.high,
           "k": {3, 1, 2}, 5: 1e-7, "m": 2.50, "n": 1 / 3}
    assert state_text(odd).splitlines() == [
        "b: null", "a: true", '["c d"]: []', '["e.f"]: {}', "g: nan", "h: ", '[" i"]:  padded ', "j: high", "k[0]: 1",
        "k[1]: 2", "k[2]: 3", "5: 0", "m: 2.5", "n: 0.333333"]
    assert state_text([1, {"x": [2]}]) == "[0]: 1\n[1].x[0]: 2" and state_text({}) == ".: {}" and state_text(3) == ".: 3"
    assert state_text({"a": {"b": [1, {"c": 2}]}}, "tree") == "a:\n  b:\n    - 1\n    - c: 2"
    assert state_text({"a": [1, "x"]}, "json") == '{"a": [1, "x"]}'
    with pytest.raises(ValueError):
        state_text({"a": 1}, "yaml")


def test_a_decision_reads_a_json_or_pydantic_state():
    m = v2(act=False)
    part = m.decision("team", "Which team?", "order", TEAMS)
    cat = Catalog()
    cat.fn(part)
    s = System(cat, [Question("route", "Route", Answer.choice(TEAMS))])

    @cat.rule("route")
    def route(team):
        return team
    state = {"order": {"note": "charged twice, refund please", "id": 7}}
    res = s.ask(state)
    assert res["route"].answer == "billing"
    assert m.scorer.texts == ["note: charged twice, refund please\nid: 7"]      # the network read the "paths" text
    assert part.text_of(state) == "note: charged twice, refund please\nid: 7"
    assert m.logits(part.text_of(state), "Which team?", TEAMS) == m.logits({"note": "charged twice, refund please",
                                                                             "id": 7}, "Which team?", TEAMS)

    class Req(BaseModel):
        order: dict
        channel: str = "email"
    res2 = s.ask(Req(order={"id": 7, "note": "charged twice, refund please"}))
    assert res2["route"].answer == "billing" and res2.trace.replay(cat)["ok"]
    two = m.decision("team2", "Which team?", ["order", "channel"], TEAMS)
    assert two.text_of({"order": {"id": 7}, "channel": "email"}) == "order.id: 7\nchannel: email"
    assert part.text_of({"order": 12.0}) == "12" and part.text_of({"order": date(2026, 1, 2)}) == "2026-01-02"
    assert two.text_of({"order": "text a", "channel": "text b"}) == "text a\ntext b"   # all texts: joined as before


def test_the_state_serialization_follows_the_checkpoint():
    m = v2(meta={"state_serialization": ["json", "paths"]})
    part = m.decision("team", "Which team?", "order", TEAMS)
    assert m.state_format == "json" and part.text_of({"order": {"note": "refund", "n": 1}}) == '{"note": "refund", "n": 1}'
    assert part(order="a refund").value == "billing"
    with pytest.raises(ValueError, match="this solvi writes"):
        v2(meta={"state_serialization": ["yaml"]})
    assert DecideModel(StubScorer(), {"format": "l14b_decider v1"}).state_format == "paths"   # L14d reads the lines as text


# --------------------------------------------------------------------------------------------------- question types
def test_types_declare_the_question_kind():
    assert question_kind(Literal["a", "b"]) == ("choice", ["a", "b"], False)
    assert question_kind(Urgency) == ("choice", ["low", "high"], False)
    assert question_kind(bool) == ("noul", ["yes", "no"], True)
    assert question_kind(Optional[bool]) == ("noul", ["yes", "no"], True)
    assert question_kind(Literal["yes", "no"]) == ("noul", ["yes", "no"], False)
    assert question_kind(Scale["low", "high", "top"]) == ("score", ["low", "high", "top"], False)
    assert question_kind(list[Literal["x", "y"]]) == ("multi", ["x", "y"], True)[0:2] + (False,)
    assert Answer.from_type(Scale[1, 2, 3]).kind == "ordinal" and Answer.from_type(Scale[Urgency]).options == ["low", "high"]
    assert question_kind(Scale[Literal["a", "b"]]) == ("score", ["a", "b"], False)
    with pytest.raises(TypeError):
        question_kind(int)
    m = v2()
    parts = m.decisions(Ticket, text_fact="email")
    assert {n: p.kind for n, p in parts.items()} == {"team": "choice", "urgency": "score", "angry": "noul", "topics": "multi"}
    assert parts["angry"].options == [True, False] and parts["angry"].labels == ["yes", "no"]
    assert parts["urgency"].task == "How urgent is it?"
    cat = Catalog()
    qs = [p.question(cat) for p in parts.values()]
    assert [q.answer.kind for q in qs] == ["choice", "ordinal", "yes_no", "multi"]
    assert qs[1].answer.options == LEVELS
    with pytest.raises(ValueError):
        m.decision("x", "?", "email", Scale[1])                                  # a score has 2–10 levels


def test_every_kind_gives_probabilities_confidence_and_a_value_among_the_options():
    m = v2(act=False)
    team = m.decision("team", "Which team?", "email", Literal["billing", "technical", "shipping"])
    urg = m.decision("urgency", "How urgent?", "email", Scale["low", "medium", "high", "critical"])
    stars = m.decision("stars", "Stars?", "email", Scale[1, 2, 3, 4, 5])
    angry = m.decision("angry", "Angry?", "email", bool)
    yn = m.decision("angry_yn", "Angry?", "email", Literal["yes", "no"])
    topics = m.decision("topics", "Topics?", "email", list[Literal["refund", "delay", "bug"]])
    other = m.decision("team_o", "Which team?", "email", TEAMS + ["other"])
    text = "I am furious!! The parcel is late and I was charged twice, refund now!"
    for p in (team, urg, stars, angry, yn, topics, other):
        d = p(email=text)
        assert isinstance(d, Decision) and 0 <= d.conf <= 1
        vals = d.value if isinstance(d.value, tuple) else (d.value,)
        assert all(v in p.options for v in vals)
        if p.kind != "multi":
            assert abs(sum(d.probs.values()) - 1) < 1e-9
    assert team(email=text).value == "billing"
    d = urg(email=text)
    assert d.value == "critical" and set(d.probs) == set(LEVELS) and d.extra["median"] == "critical"
    assert 0 <= d.extra["expected"] <= 3
    s = stars(email="urgent!")                                       # numeric levels: expected in level units
    assert s.value == 4 and 3.0 < s.extra["expected"] < 5.0
    assert angry(email=text).value is True and angry(email="thanks").value is False
    assert set(angry(email=text).probs) == {"yes", "no"}
    assert yn(email=text).value == "yes"
    assert topics(email=text).value == ("refund", "delay")
    assert other(email="hello there").value in TEAMS + ["other"]


def test_score_median_differs_from_the_mode_and_can_be_chosen():
    sp = _Spec("?", LEVELS, kind="score")
    m = v2(act=False)
    z = np.log(np.array([0.4, 0.05, 0.15, 0.4]))
    d = m._decision(sp, z)
    assert d.value == "high" and d.extra["median"] == "high"         # cumulative 0.4, 0.45, 0.6 → high; the mode is low
    sp2 = _Spec("?", LEVELS, kind="score", score_value="mode")
    assert m._decision(sp2, z).value == "low"


def test_bool_decision_feeds_typed_and_untyped_readers_and_answers_yes_no():
    m = v2(act=False)
    cat = Catalog()
    cat.fn(m.decision("angry", "Is the customer angry?", "email", bool))

    @cat.rule("priority")
    def priority(angry: bool) -> Literal["high", "normal"]:
        return "high" if angry else "normal"
    q2 = m.decision("angry_q", "Is the customer angry?", "email", bool).question(cat, "angry_q")
    s = System(cat, [Question("priority", "Priority?"), q2])
    r = s.ask({"email": "This is terrible service"})
    assert r["priority"].answer == "high" and r["angry_q"].answer == "yes" and set(r["angry_q"].probs) == {"yes", "no"}
    r = s.ask({"email": "all fine, thanks"})
    assert r["priority"].answer == "normal" and r["angry_q"].answer == "no"
    assert r.trace.replay(cat)["ok"]


# --------------------------------------------------------------------------------------------------- act / escalate
def ticket_system(m, **kw):
    cat = Catalog()
    qs = m.questions(cat, Ticket, text_fact="email", **kw)
    return cat, System(cat, qs)


def test_model_escalation_abstains_with_its_own_reason_in_audit_and_stats():
    m = v2()
    cat, s = ticket_system(m)
    res = s.ask({"email": "maybe the parcel is late?"})
    r = res["team"]
    assert r.status == "abstain" and r.guard == "escalated" and r.answer is None
    assert r.why.startswith("model escalated: act 0.05 < 0.50; would have answered") and r.provenance == "decided"
    rec = next(x for x in res.trace.records if x.name == "answer:team")
    assert rec.model["type"] == "DecisionPart" and rec.extra["act"] == pytest.approx(1 / (1 + np.exp(3)))
    au = res.audit("team")
    assert au.safeguard_line() == "model escalated ×1" and "act 0.05" in str(au)
    assert s.stats["model_escalated"] == 4 and s.stats["low_confidence"] == 0 and s.stats["abstained"] == 4
    assert "model escalated" in s.safeguard_report()
    ok = s.ask({"email": "the parcel is late"})
    assert all(x.status == "ok" for x in ok.results.values()) and ok["team"].answer == "shipping"
    assert res.trace.replay(cat)["ok"] and ok.trace.replay(cat)["ok"]


def test_use_act_false_ignores_the_models_signal_and_act_threshold_overrides_it():
    m = v2()
    p = m.decision("team", "Which team?", "email", TEAMS, use_act=False)
    assert p(email="maybe a refund").act and p(email="maybe a refund").extra["act"] < 0.5
    q = m.decision("team", "Which team?", "email", TEAMS, act_threshold=0.01)
    assert q(email="maybe a refund").act
    assert p.fingerprint() != q.fingerprint() != m.decision("team", "Which team?", "email", TEAMS).fingerprint()


def test_escalate_below_uses_calibrated_confidence_as_low_confidence():
    m = v2(act=False)
    cat = Catalog()
    team = m.decision("team", "Which team?", "email", TEAMS, escalate_below=0.9)
    cat.fn(team)

    @cat.rule("route")
    def route(team):
        return team
    s = System(cat, [Question("route", "Route", Answer.choice(TEAMS))])
    r = s.ask({"email": "charged twice and the app crashed"})          # a tie: confidence 0.5
    assert r["route"].status == "abstain" and "team" in r["route"].why
    ev = r.safeguards[0]
    assert ev["kind"] == "low_confidence" and "escalate_below" in ev["detail"] and ev["fact"] == "team"
    assert s.stats["low_confidence"] == 1 and s.stats["model_escalated"] == 0
    assert s.ask({"email": "refund refund, charged twice"})["route"].answer == "billing"
    assert r.trace.replay(cat)["ok"]


def test_an_escalated_decision_falls_back_to_the_next_producer():
    m = v2()
    cat = Catalog()
    cat.fn(provides="team")(m.decision("team_model", "Which team?", "email", TEAMS))

    @cat.fn(provides="team")
    def team_desk(email):
        return "billing"

    @cat.rule("route")
    def route(team):
        return team
    s = System(cat, [Question("route", "Route", Answer.choice(TEAMS))])
    r = s.ask({"email": "maybe a parcel?"})
    assert r["route"].answer == "billing" and s.stats["fallbacks"] == 1 and s.stats["model_escalated"] == 1
    assert r.trace.replay(cat)["ok"]


def test_calibrate_for_a_target_error_rate():
    m = v2(act=False)
    part = m.decision("team", "Which team?", "email", TEAMS)
    ex = ([("charged twice, refund", "billing")] * 6 + [("the app shows an error", "technical")] * 6
          + [("a crash and a refund", "billing")] * 2 + [("a crash and a refund", "technical")] * 2)
    info = part.calibrate_for(ex, error=0.05)
    assert info["signal"] == "confidence" and 0.5 < info["threshold"] <= 1 and info["error"] <= 0.05
    assert info["coverage"] == pytest.approx(12 / 16) and part.escalate_below == info["threshold"]
    assert not part(email="a crash and a refund").act and part(email="refund").act
    ma = v2()
    pa = ma.decision("team", "Which team?", "email", TEAMS)
    info = pa.calibrate_for([("refund", "billing"), ("maybe refund", "technical"), ("crash", "technical")], error=0.0)
    assert info["signal"] == "act" and pa.act_threshold == info["threshold"] and info["coverage"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        part.calibrate_for(ex, signal="act")


# --------------------------------------------------------------------------------------------------- several questions per pass
def test_questions_about_the_same_text_share_one_forward_pass():
    m = v2(act=False)
    cat, s = ticket_system(m)
    assert s.ask({"email": "x"}).flow.batches == [["answer:team", "answer:urgency", "answer:angry", "answer:topics"]]
    m.scorer.calls.clear()
    before = m.passes
    res = s.ask({"email": "I am furious! The parcel is late, refund!"})
    assert m.scorer.calls == [("pass", [4])] and m.passes == before + 1          # one forward pass for four questions
    rec = next(r for r in res.trace.records if r.name == "answer:urgency")
    assert rec.extra["pass"] == {"with": ["answer:team", "answer:urgency", "answer:angry", "answer:topics"], "shared": True}
    assert "one pass with answer:team" in str(res.audit("urgency"))
    # the same questions, one per pass: identical answers and probabilities
    m1 = v2(act=False, meta={"multi_question": False})
    cat1, s1 = ticket_system(m1)
    res1 = s1.ask({"email": "I am furious! The parcel is late, refund!"})
    assert m1.scorer.calls == [("seq", 1)] * 4 and res1.flow.batches == []
    for q in res.results:
        a, b = res[q], res1[q]
        assert (a.answer, a.status, a.probs, a.provenance) == (b.answer, b.status, b.probs, b.provenance)
    assert res.trace.replay(cat)["ok"] and res1.trace.replay(cat1)["ok"]


def test_batching_groups_by_the_fact_read_chunks_and_falls_back():
    m = v2(act=False, meta={"multi_question": {"max_questions": 2}})
    cat = Catalog()
    ps = [m.decision(f"t{i}", "Which team?", "email", TEAMS) for i in range(3)] + [m.decision("n", "Which?", "note", TEAMS)]
    for p in ps:
        cat.fn(p)

    @cat.rule("q")
    def q(t0, t1, t2, n):
        return t0
    s = System(cat, [Question("q", "q", Answer.choice(TEAMS))])
    res = s.ask({"email": "refund", "note": "crash"})
    assert res.flow.batches == [["t0", "t1"]] and res["q"].answer == "billing"
    assert m.scorer.calls == [("pass", [1]), ("pass", [2])]   # block layout: n alone, t0 + t1; t2 asks what t0 asked
    mf = v2(act=False, fail_pass=True)
    cat2, s2 = ticket_system(mf)
    r = s2.ask({"email": "refund!"})
    rec = next(x for x in r.trace.records if x.name == "answer:team")
    assert rec.extra["pass"]["shared"] is False and mf.scorer.calls == [("seq", 4)] and r["team"].answer == "billing"
    assert "own pass" in str(r.audit("team")) and r.trace.replay(cat2)["ok"] and not mf._block_failed
    mu = v2(act=False, fail_pass="unsupported")                   # an export without the block layout: never tried again
    cat3, s3 = ticket_system(mu)
    s3.ask({"email": "refund!"})
    s3.ask({"email": "a crash"})
    assert mu._block_failed and mu.scorer.calls == [("seq", 4), ("seq", 4)]


def test_batched_trace_replays_detects_tampering_and_survives_json():
    m = v2()
    cat, s = ticket_system(m)
    res = s.ask({"email": "furious: the app crashes, bug!!"})
    assert res.trace.replay(cat)["ok"]
    m._cache.clear()                                      # replay re-scores the shared pass
    calls = len(m.scorer.calls)
    rep = res.trace.replay(cat)
    assert rep["ok"] and m.scorer.calls[calls:] == [("pass", [4])]
    back = Response.from_json(res.to_json(), catalog=s)
    assert back.flow.batches == res.flow.batches and back.trace.replay(cat)["ok"]
    assert [r.hash for r in back.trace.records] == [r.hash for r in res.trace.records]
    bad = copy.deepcopy(res.trace)
    rec = next(r for r in bad.records if r.name == "answer:urgency")
    rec.extra["act"] = 0.99
    assert not bad.replay(cat)["ok"]
    m.decisions(Ticket, text_fact="email")["team"].teach("furious: the app crashes, bug!!", "shipping")
    rep = res.trace.replay(cat)                          # the same question on the same model: its adaptation changed
    assert not rep["ok"] and "model changed since this decision" in rep["mismatches"][0][2]


def test_parallel_workers_share_one_pass():
    m = v2(act=False)
    cat = Catalog()
    qs = m.questions(cat, Ticket, text_fact="email")
    s1, s4 = System(cat, qs), System(cat, qs, workers=4)
    a = s1.ask({"email": "refund, late parcel!"})
    m._cache.clear()
    m.scorer.calls.clear()
    b = s4.ask({"email": "refund, late parcel!"})
    assert m.scorer.calls == [("pass", [4])]
    assert [r.hash for r in a.trace.records] == [r.hash for r in b.trace.records]


def test_hard_checks_and_constraints_still_decide_over_the_model():
    m = v2(act=False)
    cat = Catalog()
    qs = m.questions(cat, Ticket, text_fact="email")

    @cat.fn
    def legal(email):
        return "lawyer" in email

    @cat.check(hard=True, then={"urgency": "critical"})
    def no_lawyer(legal):
        return not legal

    @cat.constraint
    def refunds_to_billing(team, topics):
        return "refund" not in (topics or ()) or team == "billing"
    qs[1].checkpoints.append("no_lawyer")
    s = System(cat, qs)
    r = s.ask({"email": "my lawyer says the app crash and the bug need a refund"})
    assert r["urgency"].answer == "critical" and r["urgency"].status == "forced"
    assert r["team"].answer == "technical" and r["topics"].answer == ("bug",)          # the cheaper repair: drop "refund"
    assert r["topics"].repaired == (("refund", "bug"), ["refunds_to_billing"]) and r.feasible
    assert r.trace.replay(cat)["ok"]


# --------------------------------------------------------------------------------------------------- adapt / fit / teach per kind
def test_score_fit_is_ordinal_aware_and_noul_fit_is_one_bias():
    m = v2(act=False)
    urg = m.decision("urgency", "How urgent?", "email", Scale["low", "medium", "high", "critical"])
    ex = [("please help", "medium")] * 5 + [("please help!", "high")] * 5 + [("urgent!", "critical")] * 3 + [("ok", "low")] * 3
    urg.fit(ex)
    B = _basis(urg.spec)
    b = np.asarray(urg.adaptation.shift)
    c = np.linalg.lstsq(B, b, rcond=None)[0]
    assert np.allclose(B @ c, b) and B.shape == (4, 2)                    # the shift is a tilt + spread, not free
    assert urg(email="please help").value == "medium"                    # the tilt moved "no signal" up one level
    ms = urg.teach("please help", "high")
    assert ms < 200 and urg.adaptation.n_labelled == 17
    angry = m.decision("angry", "Angry?", "email", bool)
    angry.fit([("fine", False)] * 4 + [("meh", True)] * 4 + [("terrible", True)] * 4)
    sh = angry.adaptation.shift
    assert sh[0] == pytest.approx(-sh[1]) and sh[0] > 0
    angry.adapt(["fine", "terrible", "ok"])
    assert angry.adaptation.bias is not None


def test_system_teach_routes_to_score_and_bool_decisions():
    m = v2(act=False)
    cat, s = ticket_system(m)
    fp = cat.rules["angry"].func.fingerprint()
    assert s.teach("angry", {"email": "meh"}, True) is not None               # True → the label "yes"
    assert s.teach("urgency", {"email": "meh"}, "high") is not None
    assert cat.rules["angry"].func.fingerprint() != fp


def test_adaptations_of_every_kind_save_and_load(tmp_path):
    m = v2(act=False)
    parts = m.decisions(Ticket, text_fact="email")
    parts["urgency"].fit([("ok", "low"), ("help!", "medium"), ("urgent!", "critical"), ("urgent", "high")] * 2)
    parts["angry"].teach("meh", True)
    f = tmp_path / "a.json"
    m.save_adaptations(f)
    m2 = v2(act=False)
    m2.load_adaptations(f)
    p2 = m2.decisions(Ticket, text_fact="email")
    for n in ("urgency", "angry"):
        assert p2[n].fingerprint() == parts[n].fingerprint()


# --------------------------------------------------------------------------------------------------- format and compatibility
def test_capabilities_defaults_per_format_and_declared_fields():
    c = capabilities({"format": "l14b_decider v1", "temperature": 1.08})
    assert c["version"] == 1 and c["modes"] == ["single", "multi"] and c["columns"] == {"single": 0, "multi": 1}
    assert c["serialization"] == ["text"] and c["act"] is None and c["max_questions"] == 0 and c["noul_labels"] == ["yes", "no"]
    t = capabilities({"format": "l14f typed v1"})
    assert t["modes"] == ["single", "multi", "score", "noul"] and t["columns"]["act"] == 2 and t["columns"]["score"] == 0
    assert t["act"] == {"column": 2, "temperature": 1.0} and t["noul_labels"] == ["true", "false"]
    assert t["state_format"] == "paths" and t["max_questions"] == 0                   # several per pass only when declared
    c2 = capabilities(META2)
    assert c2["multi_question"] == {"layout": "block", "max_questions": 8, "max_len": 1024, "window": 64}
    assert capabilities(META2, multi_question=False)["max_questions"] == 0 and capabilities(META2, act=False)["act"] is None
    assert capabilities({"multi_question": 4})["multi_question"]["layout"] == "block"
    assert capabilities({"format": "solvi_decide v2", "act_head": True})["act"]["column"] == 2
    with pytest.raises(ValueError, match="act calibrator"):
        capabilities({"format": "l14f typed v1", "act": {"calibrator": {"features": ["nope"], "weights": [1.0]}}})


def test_act_calibrator_and_thresholds_for_a_target_error():
    cal = {"features": ["act_logit", "confidence", "kind=choice"], "weights": [1.0, 0.0, 0.5], "bias": -1.5}
    m = v2(meta={"act": {"calibrator": cal, "threshold": 0.5, "threshold_for_error": {"0.05": 0.9, "0.1": 0.7}}})
    d = m.decision("team", "Which team?", "email", TEAMS)(email="refund")      # act logit 3 → σ(3 + 0.5 − 1.5)
    assert d.extra["act"] == pytest.approx(1 / (1 + np.exp(-2.0))) and d.act
    assert m.act_threshold_for(0.1) == 0.7 and m.act_threshold_for(0.07) == 0.9
    strict = m.decision("team", "Which team?", "email", TEAMS, target_error=0.05)
    assert strict.act_threshold == 0.9 and not strict(email="refund").act
    with pytest.raises(ValueError, match="calibrate_for"):
        m.act_threshold_for(0.01)
    x = act_features(_Spec("?", TOPICS, kind="multi"), Decision(("refund",), {"refund": 0.9, "delay": 0.2, "bug": 0.5},
                                                                   confidence=0.5), 1.0)
    assert x["margin"] == pytest.approx(0.0) and x["kind=multi"] == 1.0 and x["n_options"] == 3.0


def test_block_masks_keep_questions_apart():
    blk = np.array([[-1, -1, -1, 0, 0, 1, 1, -2]])
    pids = np.array([[0, 1, 2, 3, 4, 3, 4, 0]])
    full, sliding = block_masks(blk, pids, window=1)
    f = full[0, 0]
    assert f[0, :3].all() and not f[0, 3:].any()                   # the input sees only the input
    assert f[3, :5].all() and not f[3, 5:].any()                   # block 0 sees the input and itself
    assert f[5, :3].all() and not f[5, 3:5].any() and f[5, 5:7].all()
    assert f[7].tolist() == [False] * 7 + [True]                   # padding sees itself
    assert sliding[0, 0, 3].tolist() == [False, False, True, True, True, False, False, False]


def test_legacy_checkpoint_hashes_and_asks_as_before():
    class Legacy:
        model_id = "test/legacy"

        def fingerprint(self):
            return "legacy-1"

        def logits(self, items):
            return [np.stack([np.arange(len(it.options), dtype=float), -np.arange(len(it.options), dtype=float)], 1)
                    for it in items]
    meta = {"format": "l14b_decider v1", "temperature": 1.2, "other_threshold": 0.3, "multi_threshold": 0.4}
    m = DecideModel(Legacy(), meta)
    assert m.weights_fingerprint() == digest("DecideModel", "legacy-1", "Legacy", 1.2, 1.0, 0.3, 0.4)
    stand_in = DecideModel(Legacy(), {"format": "demo stand-in", "temperature": 1.0})       # any other format: as before
    assert stand_in.weights_fingerprint() == digest("DecideModel", "legacy-1", "Legacy", 1.0, 1.0, 0.5, 0.5)
    part = m.decision("team", "Which team?", "email", TEAMS)
    assert part.fingerprint() == digest("DecisionPart", m.weights_fingerprint(),
                                        {"task": "Which team?", "options": TEAMS, "descriptions": {}, "multi": False,
                                         "other": None}, None)
    assert not m.batchable and not m.has_act and m.wire("score") == "single" and m.wire("noul") == "single"
    it = _Spec("?", LEVELS, kind="score").item("t", m.wire("score"))
    assert it.mode == "single" and it.kind == "" and not it.multi
    assert m.decide("x", "?", LEVELS, kind="score").value == "critical"      # column 0, levels as a single choice
    assert m.decide("x", "?", TEAMS, multi=True).value == ("billing",)       # column 1 for multi-label: 0, −1, −2
    cat = Catalog()
    s = System(cat, [part.question(cat, "team"), m.decision("u", "?", "email", Scale["a", "b"]).question(cat, "u")])
    res = s.ask({"email": "hello"})
    recs = {r.name: r for r in res.trace.records}
    assert res.flow.batches == [] and recs["answer:team"].extra is None and "extra" not in recs["answer:team"].body()
    assert set(recs["answer:u"].extra) == {"expected", "median"} and res.trace.replay(cat)["ok"]


def test_prompts_items_and_columns():
    assert prompt("t", ["a", "b"]) == "[unused1] t[unused0] a[unused0] b"
    assert prompt("t", ["a"], mode="score") == "[unused3] t[unused0] a"
    items = (Item("t1", ("a", "b"), None, "x"), Item("t2", ("yes", "no"), None, "x", kind="noul"))
    assert pass_prompt(items) == "[unused1] t1[unused0] a[unused0] b [unused4] t2[unused0] yes[unused0] no"

    class Cols:
        def logits(self, items):
            return [np.tile(np.arange(5, dtype=float), (len(it.options), 1)) * [1, 1, 1, 1, 1]
                    + np.arange(len(it.options))[:, None] * [0, 0, 1, 0, 0] for it in items]
    m = DecideModel(Cols(), {**META2, "columns": {"score": 2}})
    assert m.logits("x", "?", LEVELS, kind="score") == {"low": 2.0, "medium": 3.0, "high": 4.0, "critical": 5.0}
    it = _Spec("?", ["yes", "no"], kind="noul", descriptions={"yes": "it holds"}).item("t", "noul", ["true", "false"])
    assert it.options == ("true", "false") and it.descriptions == ("it holds", "") and it.mode == "noul"


def test_unknown_format_versions_are_refused(tmp_path):
    (tmp_path / "solvi_decide.json").write_text(json.dumps({"format": "l14f typed v2"}))
    with pytest.raises(ValueError, match="unknown decider format"):
        DecideModel.load(tmp_path)


def test_catalogs_without_decisions_do_no_batching_work():
    code = ("import sys\nfrom solvi import Answer, Catalog, Question, System\ncat = Catalog()\n"
            "@cat.fn\ndef a(x):\n    return x + 1\n@cat.rule('q')\ndef q(a):\n    return 'yes' if a > 1 else 'no'\n"
            "r = System(cat, [Question('q', 'q', Answer.yes_no())]).ask({'x': 1})\n"
            "print(r['q'].answer, cat.decisions, r.flow.batches, 'solvi.decide' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["yes", "0", "[]", "False"]


# --------------------------------------------------------------------------------------------------- quote source inference
def test_extract_quote_source_is_inferred_from_its_text_argument():
    cat = Catalog()

    @cat.extract
    def refund_word(message):                                      # no source= in the Quote: it points into `message`
        i = message.find("refund")
        return Quote("refund", i, i + 6) if i >= 0 else None

    @cat.extract
    def amount(message: str, currency):                            # the only str-typed argument
        i = message.find("42")
        return Quote(42, i, i + 2)

    @cat.extract(source="body")
    def word(body, subject):
        return Quote("hi", 0, 2)

    @cat.extract
    def kept(note):
        return Quote("x", 0, 1, source="other")                   # an explicit source is kept

    @cat.rule("q")
    def q(refund_word, amount, word):
        return "yes"
    s = System(cat, [Question("q", "q", Answer.yes_no())])
    r = s.ask({"message": "please refund 42", "currency": "EUR", "body": "hi there", "subject": "s"})
    recs = {x.name: x for x in r.trace.records}
    assert recs["refund_word"].quote == (7, 13, "message") and recs["amount"].quote == (14, 16, "message")
    assert recs["word"].quote == (0, 2, "body") and r["q"].answer == "yes" and r.trace.replay(cat)["ok"]
    assert refund_word("a refund") == Quote("refund", 2, 8)       # the decorated function itself is unchanged
    assert cat.parts["kept"].func(note="n").source == "other"
    with pytest.raises(ValueError, match="source="):
        @cat.extract
        def ambiguous(message, body):
            return None


# --------------------------------------------------------------------------------------------------- example 15
def test_example_15_runs_with_the_stand_in(monkeypatch):
    import io
    import runpy
    from contextlib import redirect_stdout
    from pathlib import Path
    monkeypatch.delenv("SOLVI_DECIDE_MODEL", raising=False)
    out = io.StringIO()
    with redirect_stdout(out):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "15_typed_decisions.py"), run_name="__main__")
    text = out.getvalue()
    assert "forward passes: 1 for 4 decided questions" in text and "urgency='critical' [forced]" in text
    assert "model escalated: act" in text and "model escalated        4" in text and "one pass with answer:team" in text
    assert "replay: first ticket ok=True, legal threat ok=True, unclear ticket ok=True" in text


# --------------------------------------------------------------------------------------------------- the real checkpoint
def _real_dir():
    for p in (os.environ.get("SOLVI_DECIDE_MODEL"), os.path.expanduser("~/.cache/solvi_release/decide-base-clean"),
              os.path.expanduser("~/.cache/solvi_release/decide-base")):
        if p and os.path.isfile(os.path.join(p, "solvi_decide.json")):
            return p
    return None


@pytest.mark.parametrize("backend", ["onnx", "torch"])
def test_real_l14d_checkpoint_typed_questions_if_present(backend):
    pytest.importorskip("tokenizers")
    pytest.importorskip("onnxruntime" if backend == "onnx" else "torch")
    if backend == "torch":
        pytest.importorskip("transformers")
    path = _real_dir()
    if path is None or (backend == "onnx" and not os.path.isdir(os.path.join(path, "onnx"))):
        pytest.skip("no solvi-decide checkpoint (set SOLVI_DECIDE_MODEL)")
    m = DecideModel.load(path, backend=backend, device="cpu")
    assert m.caps["version"] == 1 and not m.batchable and not m.has_act                # the L14d format, unchanged
    assert m.weights_fingerprint() == digest("DecideModel", m._wfp, m.backend, m.temperature, m.temperature_multi,
                                             m.other_threshold, m.multi_threshold)
    cat = Catalog()
    qs = m.questions(cat, Ticket, text_fact="email", other=False)
    s = System(cat, qs)
    text = "I was charged twice for my order and nobody answers. This is terrible, fix it today!"
    res = s.ask({"email": text})
    assert res["team"].answer == "billing" and res["angry"].answer == "yes"
    assert res["urgency"].answer in LEVELS and set(res["urgency"].probs) == set(LEVELS)
    assert res.flow.batches == [] and res.trace.replay(cat)["ok"]
    state = {"ticket": {"subject": "Double charge", "body": "Charged twice for order 5521, please refund.", "tier": "gold"}}
    d = m.decision("team", "Which team should handle this ticket?", "ticket", TEAMS)
    assert d(ticket=state["ticket"]).value == "billing"
    # forced multi-question passes (the L14f contract) run on the L14d weights too: valid answers, replay ok. torch: the
    # block layout (an answer does not depend on the other questions of its pass); ONNX without block inputs: one question
    # per pass (fallback); the concat layout: one pass
    layout = {"layout": "block", "max_questions": 4} if backend == "torch" else {"layout": "concat", "max_questions": 4}
    for mqc in ([layout, 4] if backend == "onnx" else [layout]):
        mq = DecideModel.load(path, backend=backend, device="cpu", multi_question=mqc)
        cat2 = Catalog()
        s2 = System(cat2, mq.questions(cat2, Ticket, text_fact="email", other=False))
        before = mq.passes
        r2 = s2.ask({"email": text})
        shared = next(r for r in r2.trace.records if r.name == "answer:team").extra["pass"]["shared"]
        assert mq.passes == before + (1 if shared else 4) and shared == (mqc is layout)
        assert r2.flow.batches and all(r.status == "ok" for r in r2.results.values())
        assert r2.trace.replay(cat2)["ok"] and mq.weights_fingerprint() != m.weights_fingerprint()
        if backend == "torch":
            specs = [cat2.rules[q].func.spec for q in ("team", "urgency", "angry")]
            mq._cache.clear()
            alone = mq._raw_pass(specs[:1], text)[0][0]
            mq._cache.clear()
            together = mq._raw_pass(specs, text)[0][0]
            assert np.max(np.abs(alone - together)) < 1e-3
