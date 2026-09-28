"""A System One endpoint (POST /v1/systemone) as a solvi decider, against a fake server."""
import io
import json

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.systemone import systemone

TEAMS = {"billing": "Charges, invoices, refunds", "shipping": "Delivery, parcels, tracking"}


class FakeService:
    """Answers like a System One server: choice → probabilities by keywords, noul → p(true)."""

    def __init__(self):
        self.bodies, self.headers = [], []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.bodies.append(body)
        self.headers.append(dict(req.header_items()))
        text = body["state"].lower()
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.9 if "urgent" in text else 0.2}
            else:
                opts = list(q["criteria"])
                raw = [1.0 + 4.0 * (o == "billing" and "charged" in text) + 4.0 * (o == "shipping" and "parcel" in text)
                       for o in opts]
                z = sum(raw)
                answers[name] = {"type": "choice", "choice": opts[raw.index(max(raw))],
                                 "probabilities": {o: r / z for o, r in zip(opts, raw)}}
        return io.BytesIO(json.dumps({"model": body["model"], "answers": answers}).encode())


def test_questions_about_one_input_go_in_one_request_and_become_decisions():
    svc = FakeService()
    m = systemone("http://localhost:8009/", "kev-latest", api_key="k", opener=svc)
    team = m.decision("team", "Which team should handle this?", "email", TEAMS)
    urgent = m.decision("urgent", "Does this need urgent attention?", "email", type=bool)
    d = team.decide("I was charged twice")
    assert d.value == "billing" and d.probs["billing"] == pytest.approx(5 / 6, abs=1e-6)
    assert svc.bodies[-1]["questions"]["q0"] == {"type": "choice", "instructions": "Which team should handle this?",
                                                 "criteria": TEAMS}
    assert svc.headers[-1]["Authorization"] == "Bearer k" and svc.bodies[-1]["model"] == "kev-latest"
    u = urgent.decide("urgent: the parcel is lost")
    assert u.value is True and u.probs["yes"] == pytest.approx(0.9, abs=1e-6)
    assert svc.bodies[-1]["questions"]["q0"]["type"] == "noul"
    n = len(svc.bodies)
    both = m.decide_pass("urgent, I was charged twice", [team, urgent])
    assert [x.value for x in both] == ["billing", True]
    assert len(svc.bodies) <= n + 2


def test_a_system_one_decision_in_a_catalog_is_traced_and_guarded():
    svc = FakeService()
    m = systemone("http://localhost:8009", "kev-latest", opener=svc)
    part = m.decision("team", "Which team?", "email", TEAMS)
    info = part.act_guard([("charged twice", "billing"), ("my parcel", "shipping")] * 30
                          + [("hello", "shipping")] * 10, risk=0.10)
    assert info["signal"] == "confidence" and info["risk"] <= 0.10
    cat = Catalog()
    part.question(cat, "route")
    s = System(cat, [Question("route", "Route", Answer.choice(list(TEAMS)))])
    r = s.ask({"email": "I was charged twice"})
    assert r["route"].answer == "billing"
    step = next(x for x in r.trace.records if x.model)
    assert step.model and "systemone:kev-latest" in json.dumps(step.model)
    assert s.ask({"email": "hello"})["route"].status == "abstain"


def test_what_the_api_cannot_answer_is_refused_up_front():
    m = systemone("http://localhost:8009", "kev-latest", opener=FakeService())
    with pytest.raises(ValueError, match="multi-label"):
        m.decision("tags", "Tags?", "email", list(TEAMS), multi=True).decide("x")
