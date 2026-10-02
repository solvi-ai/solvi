"""Jeeves (github.com/PostHog/jeeves) as a solvi decider, end to end over HTTP against a stand-in server that validates
requests and shapes replies as Jeeves's own server does (tests/jeeves_mock.py): its `options` pass through extra_body,
reasoning tokens / latency / reasoning text are recorded, and the reasoning never decides the answer."""
import io
import json
import sys
from pathlib import Path

import pytest

from solvi import Answer, Catalog, Question, System, Unknown
from solvi.multi import Vote
from solvi.systemone import REASONING_CHARS, systemone

sys.path.insert(0, str(Path(__file__).parent))
from jeeves_mock import CHAIN, JeevesMock  # noqa: E402

TEAMS = {"billing": "Charges, invoices, refunds", "shipping": "Delivery, parcels, tracking"}
TEAMS3 = {**TEAMS, "account": "Login, password, profile"}       # a keyword hit is p = 9/11 < 0.9: Jeeves thinks
OPTIONS = {"options": {"max_think": 512, "nothink_threshold": 0.9}}
CALIB = [("I was charged twice", "billing"), ("my parcel is lost", "shipping")] * 30 + [("hello", "shipping")] * 10


@pytest.fixture
def jeeves():
    mock = JeevesMock()
    with mock as url:
        mock.url = url
        yield mock


def standin(req, timeout=None):
    """Another family (a stand-in for solvi-large behind a System One server): sure of billing on "charged"."""
    body = json.loads(req.data.decode())
    text = body["state"].lower()
    answers = {}
    for name, q in body["questions"].items():
        opts = list(q["criteria"])
        raw = [1.0 + 9.0 * (o == "billing" and "charged" in text) + 9.0 * (o == "shipping" and "parcel" in text)
               for o in opts]
        answers[name] = {"type": "choice", "probabilities": {o: r / sum(raw) for o, r in zip(opts, raw)}}
    return io.BytesIO(json.dumps({"model": body["model"], "answers": answers}).encode())


def test_every_question_type_goes_through_jeeves_with_its_options(jeeves):
    m = systemone(jeeves.url, "jeeves-latest", extra_body=OPTIONS)
    cat = Catalog()
    m.decision("team", "Which team should handle this?", "email", TEAMS).question(cat, "route")
    m.decision("urgent", "Does this need urgent attention?", "email", type=bool).question(cat, "urgent")
    m.decision("mood", "How frustrated is the customer?", "email", ["calm", "frustrated", "very angry"],
               kind="score").question(cat, "mood")
    m.decision("refund", "Which team?", "email", TEAMS, not_stated=True).question(cat, "who")
    m.decision("tags", "Which topics?", "email", TEAMS, multi=True).question(cat, "tags")
    s = System(cat, [Question("route", "Route", Answer.choice(list(TEAMS))), Question("urgent", "Urgent", Answer.yes_no()),
                     Question("mood", "Mood", Answer.choice(["calm", "frustrated", "very angry"])),
                     Question("who", "Who", Answer.choice(list(TEAMS))),
                     Question("tags", "Tags", Answer.multi(list(TEAMS)))])
    r = s.ask({"email": "urgent: I was charged twice, furious!!"})
    assert r["route"].answer == "billing" and r["urgent"].answer == "yes" and r["mood"].answer == "very angry"
    assert r["who"].answer == "billing" and r["tags"].answer == ("billing",)
    r = s.ask({"email": "the parcel is late"})
    assert r["route"].answer == "shipping" and r["urgent"].answer == "no" and r["mood"].answer == "frustrated"
    assert s.ask({"email": "charged twice and the parcel is late"})["tags"].answer == ("billing", "shipping")
    assert set(jeeves.statuses) == {200} and jeeves.bodies
    for body in jeeves.bodies:                            # Jeeves's options arrive as given, next to what solvi sets
        assert body["options"] == OPTIONS["options"] and body["model"] == "jeeves-latest"
        assert set(body) == {"options", "model", "state", "questions"}
    assert s.ask({"email": "hello"})["who"].status == "abstain"            # "not stated": the question abstains
    assert m.decision("refund", "Which team?", "email", TEAMS, not_stated=True).decide("hello").value is Unknown


def test_reasoning_tokens_latency_and_the_reasoning_are_recorded_but_never_decide(jeeves):
    m = systemone(jeeves.url, "jeeves-latest",
                  extra_body={"options": {**OPTIONS["options"], "return_reasoning": True}})
    d = m.decision("team", "Which team?", "email", TEAMS3).decide("the parcel is late")
    x = d.extra["systemone"]
    assert x["usage"] == {"input_tokens": 44, "output_tokens": 16, "reasoning_tokens": 512}
    assert x["latency_ms"] == pytest.approx(300.0 + 5 * 512) and x["ms"] >= 0
    why = x["reasoning"]
    assert why["tokens"] == 512 and why["thought"] is True and why["closed"] is False
    full = CHAIN + "So the answer must be billing."        # the chain argues for billing ...
    assert why["text"] == full[:REASONING_CHARS] and why["truncated"] == len(full)
    assert d.value == "shipping"                          # ... the answer is the probabilities'
    assert m.scorer.usage["reasoning_tokens"] == 512
    tags = m.decision("tags", "Which topics?", "email", TEAMS, multi=True, not_stated=True).decide("charged twice")
    assert set(tags.extra["systemone"]["reasoning"]) == {"billing", "shipping", "not stated"}
    assert tags.value == ("billing",)
    plain = systemone(jeeves.url, "jeeves-latest", extra_body=OPTIONS)
    y = plain.decision("team", "Which team?", "email", TEAMS3).decide("the parcel is late").extra["systemone"]
    assert "reasoning" not in y and y["usage"]["reasoning_tokens"] == 512
    sure = plain.decision("team", "Which team?", "email", TEAMS).decide("charged, invoice")   # p ≥ 0.9: no thinking
    assert sure.extra["systemone"]["usage"]["reasoning_tokens"] == 0


def test_return_reasoning_leaves_the_fingerprint_the_thinking_options_do_not(jeeves):
    fp = {name: systemone(jeeves.url, "jeeves-latest", extra_body=x).fingerprint() for name, x in [
        ("plain", None), ("reasoning", {"options": {"return_reasoning": True}}), ("opts", OPTIONS),
        ("opts+reasoning", {"options": {**OPTIONS["options"], "return_reasoning": True}})]}
    assert fp["plain"] == fp["reasoning"] and fp["opts"] == fp["opts+reasoning"] and fp["plain"] != fp["opts"]


def test_an_option_jeeves_does_not_know_escalates_with_its_error_text(jeeves):
    m = systemone(jeeves.url, "jeeves-latest", extra_body={"options": {"max_thinking": 512}},
                  sleep=lambda s: pytest.fail("a 422 is not retried"))
    d = m.decision("team", "Which team?", "email", TEAMS).decide("I was charged twice")
    assert jeeves.statuses == [422] and d.escalate
    assert "the System One service refused the request: HTTP 422 — unknown options: ['max_thinking']" in d.escalate


def test_act_guard_a_vote_with_another_family_and_replay(jeeves):
    j = systemone(jeeves.url, "jeeves-latest", extra_body=OPTIONS)
    team = j.decision("team", "Which team?", "email", TEAMS)
    info = team.act_guard(CALIB, max_risk=0.10)
    assert info["signal"] == "confidence" and info["risk"] <= 0.10
    other = systemone("http://127.0.0.1:9", "solvi-large", opener=standin).decision("team", "Which team?", "email",
                                                                                     TEAMS)
    vote = Vote([team, other], rule="all", name="team")
    assert vote.act_guard(CALIB, max_risk=0.10)["risk"] <= 0.10
    cat = Catalog()
    s = System(cat, [vote.question(cat, "team", "Which team handles it?")])
    r = s.ask({"email": "I was charged twice"})
    assert r["team"].answer == "billing" and r["team"].status != "abstain"
    assert "jeeves-latest" in json.dumps([x.model for x in r.trace.records if x.model])
    j._cache.clear()
    n = len(jeeves.bodies)
    rep = r.trace.replay(s.catalog, r.flow)            # deterministic=False: the recorded output is trusted
    assert rep["ok"] and len(jeeves.bodies) == n and "trusted" in [v for _, _, v in rep["models"]]
    assert s.ask({"email": "hello"})["team"].status == "abstain"
