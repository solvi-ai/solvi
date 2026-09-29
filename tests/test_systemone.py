"""A System One endpoint (POST /v1/systemone) as a solvi decider, against a fake server."""
import http.client
import io
import json
import urllib.error

import pytest

from solvi import Answer, Catalog, Question, System, Unknown
from solvi.decide import Item
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


def test_what_the_api_cannot_point_at_is_refused():
    m = systemone("http://localhost:8009", "kev-latest", opener=FakeService())
    with pytest.raises(ValueError):
        m.decision("o", "Order number?", "email", kind="span")


class PinnedService(FakeService):
    """A FakeService that reports usage, cost and the served model, and answers "not stated" / multi-label questions."""

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.bodies.append(body)
        self.headers.append(dict(req.header_items()))
        text = body["state"].lower()
        answers = {}
        for name, q in body["questions"].items():
            ins = q["instructions"]
            if q["type"] == "noul":
                if "not stated" in ins:
                    p = 0.9 if text.strip() == "hello" else 0.05
                else:
                    p = 0.95 if ("'billing'" in ins and "charged" in text) or ("'shipping'" in ins and "parcel" in text) \
                        else 0.1
                answers[name] = {"type": "noul", "noul": p}
                continue
            opts = list(q["criteria"])
            if text.strip() == "hello" and "not stated" in opts:
                raw = [8.0 if o == "not stated" else 1.0 for o in opts]
            else:
                raw = [1.0 + 8.0 * (o in ("billing", "yes") and "charged" in text)
                       + 8.0 * (o in ("shipping", "no") and "parcel" in text) for o in opts]
            z = sum(raw)
            answers[name] = {"type": "choice", "choice": opts[raw.index(max(raw))], "confidence": 0.5,
                             "probabilities": {o: round(r / z, 2) for o, r in zip(opts, raw)}}
        resp = {"model": body["model"] + "-20260917", "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 4.2e-06}, "provider": "Someone"}
        return io.BytesIO(json.dumps(resp).encode())


def test_extra_body_is_merged_copied_fingerprinted_and_cannot_override_the_request():
    pin = {"provider": {"only": ["Someone"], "allow_fallbacks": False}, "user": "tenant-7"}
    svc = PinnedService()
    m = systemone("https://router.example/api", "vendor/model", api_key="k", opener=svc, extra_body=pin)
    assert m.decision("team", "Which team?", "email", TEAMS).decide("I was charged twice").value == "billing"
    body = svc.bodies[-1]
    assert body["provider"] == pin["provider"] and body["user"] == "tenant-7" and body["model"] == "vendor/model"
    assert set(body) == {"provider", "user", "model", "state", "questions"}
    pin["provider"]["only"] = ["Other"]                                  # copied: a later edit changes nothing
    m.decision("team", "Which team?", "email", TEAMS).decide("my parcel")
    assert svc.bodies[-1]["provider"]["only"] == ["Someone"]
    plain = systemone("https://router.example/api", "vendor/model", opener=svc)
    assert m.fingerprint() != plain.fingerprint()
    assert systemone("https://router.example/api", "vendor/model", opener=svc,
                     extra_body={"user": "other"}).fingerprint() != m.fingerprint()
    assert plain.scorer.fingerprint() == "systemone|https://router.example/api/v1/systemone|vendor/model"   # as in 0.7.0
    for key in ("model", "state", "questions"):
        with pytest.raises(ValueError, match="extra_body cannot set"):
            systemone("http://localhost:8009", "m", extra_body={key: 1})
    with pytest.raises(ValueError, match="dict"):
        systemone("http://localhost:8009", "m", extra_body=[("provider", {})])
    with pytest.raises(ValueError, match="not JSON"):
        systemone("http://localhost:8009", "m", extra_body={"x": object()})
    assert "k" not in repr(m.scorer).replace("kev", "")


def test_not_stated_is_an_option_of_its_own_and_the_question_abstains():
    svc = PinnedService()
    m = systemone("http://localhost:8009", "kev-latest", opener=svc)
    team = m.decision("team", "Which team?", "email", TEAMS, unknown=True)
    d = team.decide("hello")
    q = svc.bodies[-1]["questions"]["q0"]
    assert q["type"] == "choice" and list(q["criteria"]) == ["billing", "shipping", "not stated"]
    assert q["criteria"]["not stated"].startswith("The input does not state it")
    assert d.value is Unknown and d.probs[Unknown] == pytest.approx(0.8, abs=1e-6)
    assert team.decide("I was charged twice").value == "billing"
    ref = m.decision("refund", "Does the customer want a refund?", "email", type=bool, unknown=True)
    assert ref.decide("hello").value is Unknown
    assert list(svc.bodies[-1]["questions"]["q0"]["criteria"]) == ["yes", "no", "not stated"]
    assert ref.decide("I was charged twice").value is True
    cat = Catalog()
    team.question(cat, "route")
    s = System(cat, [Question("route", "Route", Answer.choice(list(TEAMS)))])
    assert s.ask({"email": "I was charged twice"})["route"].answer == "billing"
    r = s.ask({"email": "hello"})["route"]
    assert r.status == "abstain"


def test_a_multi_label_question_is_one_noul_per_option_thresholded_and_guarded():
    svc = PinnedService()
    m = systemone("http://localhost:8009", "kev-latest", opener=svc)
    tags = m.decision("tags", "Which topics?", "email", TEAMS, multi=True)
    d = tags.decide("charged twice and my parcel is lost")
    qs = svc.bodies[-1]["questions"]
    assert list(qs) == ["q0__0", "q0__1"] and all(q["type"] == "noul" for q in qs.values())
    assert qs["q0__0"]["instructions"] == "Which topics? Does the option 'billing' (Charges, invoices, refunds) apply?"
    assert d.value == ("billing", "shipping") and d.probs["billing"] == pytest.approx(0.95, abs=1e-6)
    assert d.conf == pytest.approx(0.95, abs=1e-6)
    assert tags.decide("I was charged twice").value == ("billing",)
    assert tags.decide("hi there").value == () and tags.decide("hi there").conf == pytest.approx(0.9, abs=1e-6)
    ns = m.decision("tags2", "Which topics?", "email", TEAMS, multi=True, unknown=True)
    assert ns.decide("hello").value is Unknown and "q0__ns" in svc.bodies[-1]["questions"]
    info = tags.act_guard([("charged twice", ("billing",)), ("my parcel", ("shipping",))] * 30
                          + [("hello", ("billing",))] * 10, risk=0.10)
    assert info["signal"] == "confidence" and info["risk"] <= 0.10


def test_cost_latency_and_the_served_model_are_recorded():
    svc = PinnedService()
    m = systemone("http://localhost:8009", "kev-latest", opener=svc)
    x = m.decision("team", "Which team?", "email", TEAMS).decide("I was charged twice").extra["systemone"]
    assert x["endpoint"] == "http://localhost:8009/v1/systemone" and x["model"] == "kev-latest"
    assert x["served_by"] == "kev-latest-20260917" and x["cost"] == pytest.approx(4.2e-06)
    assert x["usage"] == {"input_tokens": 100, "output_tokens": 10} and x["questions"] == 1 and x["ms"] >= 0
    assert m.scorer.cost == pytest.approx(4.2e-06) and m.scorer.usage["input_tokens"] == 100
    items = [Item("Which team?", tuple(TEAMS), None, "my parcel"), Item("Urgent?", ("yes", "no"), None, "my parcel")]
    a, b = m.scorer.logits(items)                                      # one request answers both: its cost is shared
    assert len(svc.bodies) == 2 and a["info"]["systemone"]["questions"] == 2 == b["info"]["systemone"]["questions"]
    assert m.scorer.cost == pytest.approx(8.4e-06)
    plain = systemone("http://localhost:8009", "kev-latest", opener=FakeService())
    y = plain.decision("team", "Which team?", "email", TEAMS).decide("I was charged").extra["systemone"]
    assert "cost" not in y and "usage" not in y and "served_by" not in y and y["ms"] >= 0


def test_a_hosted_model_is_not_replayed_but_a_deterministic_local_one_is():
    for deterministic, calls in ((False, 0), (True, 1)):
        svc = FakeService()
        m = systemone("http://localhost:8009", "kev-latest", opener=svc, deterministic=deterministic)
        assert m.deterministic is deterministic
        part = m.decision("team", "Which team?", "email", TEAMS)
        cat = Catalog()
        part.question(cat, "route")
        s = System(cat, [Question("route", "Route", Answer.choice(list(TEAMS)))])
        r = s.ask({"email": "I was charged twice"})
        m._cache.clear()
        n = len(svc.bodies)
        rep = r.trace.replay(s.catalog, r.flow)
        assert rep["ok"] and len(svc.bodies) - n == calls
        if not deterministic:
            assert [v for _, _, v in rep["models"]] == ["trusted"]


class Flaky:
    """Fails with the given errors first, then answers like FakeService."""

    def __init__(self, *errors):
        self.errors, self.ok, self.calls = list(errors), FakeService(), 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        if self.errors:
            e = self.errors.pop(0)
            raise e() if callable(e) and not isinstance(e, Exception) else e
        return self.ok(req, timeout)


def http_error(code, body=b""):
    return urllib.error.HTTPError("http://localhost:8009/v1/systemone", code, "x", {}, io.BytesIO(body))


def test_transient_errors_are_retried_with_backoff_then_the_decision_escalates_and_is_asked_again():
    waits = []
    svc = Flaky(http_error(429), http_error(503), http.client.IncompleteRead(b"{"))
    m = systemone("http://localhost:8009", "kev-latest", opener=svc, retries=3, backoff=0.5, sleep=waits.append)
    assert m.decision("team", "Which team?", "email", TEAMS).decide("I was charged twice").value == "billing"
    assert svc.calls == 4 and waits == [0.5, 1.0, 2.0]
    down = Flaky(*[TimeoutError("timed out"), ConnectionResetError("reset"), http_error(502)])
    m = systemone("http://localhost:8009", "kev-latest", api_key="sekret", opener=down, retries=2, sleep=lambda s: None)
    part = m.decision("team", "Which team?", "email", TEAMS)
    d = part.decide("I was charged twice")
    assert d.escalate and "did not answer after 3 attempts: HTTP 502" in d.escalate and "sekret" not in d.escalate
    assert down.calls == 3
    again = part.decide("I was charged twice")                        # not cached: the service is asked again
    assert down.calls == 4 and again.escalate is None and again.value == "billing"


def test_a_refused_request_escalates_at_once_with_the_services_error_text():
    body = json.dumps({"error": {"message": "Provider returned error", "code": 400,
                                 "metadata": {"raw": "criteria too long", "provider_name": "Someone"}}}).encode()
    svc = Flaky(http_error(400, body))
    m = systemone("http://localhost:8009", "kev-latest", opener=svc, sleep=lambda s: pytest.fail("no retry"))
    d = m.decision("tags", "Which topics?", "email", TEAMS, multi=True).decide("x")
    assert svc.calls == 1 and d.escalate
    assert "HTTP 400" in d.escalate and "Provider returned error (Someone: criteria too long)" in d.escalate
    cat = Catalog()
    part = m.decision("team", "Which team?", "email", TEAMS)
    part.question(cat, "route")
    s = System(cat, [Question("route", "Route", Answer.choice(list(TEAMS)))])
    svc.errors = [http_error(401, b'{"error": {"message": "bad key"}}')]
    r = s.ask({"email": "I was charged twice"})["route"]
    assert r.status == "abstain"                                       # no exception, no guess


def test_a_reply_that_breaks_the_contract_escalates():
    def broken(req, timeout=None):
        return io.BytesIO(json.dumps({"answers": {"q0": {"type": "choice", "probabilities": {"billing": 1.0}}}}).encode())
    d = systemone("http://localhost:8009", "kev-latest", opener=broken).decision("team", "Which team?", "email",
                                                                                  TEAMS).decide("x")
    assert d.escalate and "invalid System One output" in d.escalate and "shipping" in d.escalate
