"""The identity canary (solvi.core.guarantees.canary) against a fake chat server whose model silently changes behind the
same name: the reference, a confirmed change, a one-off flip, a probability shift, a served-model rename, the mark on
later decisions, escalation, the DriftMonitor flag, accept(), and no probes in a replay."""
import io
import json
import math
import re

import numpy as np
import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.deciders import DecideModel
from solvi.core.deciders.llm import llm
from solvi.core.guarantees.canary import IdentityCanary, no_probes
from solvi.core.guarantees.drift import DriftMonitor

TEAMS = ["billing", "shipping"]
PROBES = [("I was charged twice", "Which team?", TEAMS), ("Where is my parcel", "Which team?", TEAMS),
          ("Refund the charged fee", "Which team?", TEAMS), ("The parcel never came", "Which team?", TEAMS)]


class Server:
    """A chat-completions server: keyword answers with stated probabilities. `swapped` makes it another model (parcel
    texts answered "billing"); `p` is the probability it gives its answer; `name` the model it says answered; `flip`
    answers the next n requests wrongly (a provider's own noise)."""

    def __init__(self):
        self.swapped, self.p, self.name, self.flip, self.requests = False, 0.9, None, 0, 0

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.requests += 1
        text = re.search(r"<text>\n(.*)\n</text>", body["messages"][1]["content"], re.S).group(1).lower()
        best = "billing" if "charged" in text or (self.swapped and "parcel" in text) else "shipping"
        if self.flip:
            self.flip -= 1
            best = "shipping" if best == "billing" else "billing"
        other = TEAMS[1 - TEAMS.index(best)]
        content = json.dumps({"answer": best, "probabilities": {best: self.p, other: round(1 - self.p, 4)}, "quote": ""})
        ch = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        return io.BytesIO(json.dumps({"model": self.name or body["model"], "choices": [ch],
                                      "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode())


def _model(server):
    return llm("http://llm.example/v1", "tiny-chat", opener=server, sleep=lambda s: None, workers=1)


def test_off_by_default_no_probe_calls():
    srv = Server()
    m = _model(srv)
    part = m.decision("team", "Which team?", "email", TEAMS)
    part.decide("charged twice")
    assert m.canary is None and srv.requests == 1 and "canary" not in part.decide("charged twice").extra


def test_reference_then_the_same_model():
    srv = Server()
    m = _model(srv)
    c = IdentityCanary(PROBES).attach(m)
    ev = c.probe()
    assert ev["verdict"] == "reference" and ev["calls"] == len(PROBES) and c.identity == ev["identity"]
    ev = c.probe()
    assert ev["verdict"] == "same" and ev["answers_changed"] == 0 and ev["tv"] == 0 and not c.changed
    assert srv.requests == 2 * len(PROBES)


def test_a_silent_swap_is_found_marked_and_flagged_by_drift():
    srv = Server()
    m = _model(srv)
    c = IdentityCanary(PROBES, every=5).attach(m)
    part = m.decision("team", "Which team?", "email", TEAMS)
    mon = DriftMonitor(window=4, sequential=False)
    texts = [f"charged {i}" for i in range(12)]
    ds = [part.decide(t) for t in texts[:5]]                   # the 5th decision probes: the reference
    assert ds[-1].extra["canary"] == {"probed": 1} and c.events[0]["verdict"] == "reference"
    assert not any(mon.observe(d)["drift"] for d in ds)
    srv.swapped = True                                         # the provider swaps the weights, same name
    ds = [part.decide(t) for t in texts[5:10]]                 # the 10th probes again
    ev = c.events[-1]
    assert ev["verdict"] == "changed" and ev["answers_changed"] == 2 and ev["reference"] != ev["identity"]
    assert ev["calls"] == 2 * len(PROBES)                       # confirmed: asked twice
    d = ds[-1]
    assert d.extra["canary"]["changed"] and d.extra["canary"]["probed"] == ev["n"] and d.escalate is None
    later = part.decide("charged again")
    assert later.extra["canary"]["changed"] and later.extra["canary"]["event"] == ev["n"]
    rep = mon.observe(later)
    assert rep["drift"] and "identity" in rep["flags"]
    c.accept()                                                 # recalibrated on the new model: the mark goes
    assert not c.changed and "canary" not in part.decide("charged once more").extra
    assert c.events[-1]["verdict"] == "accepted"


def test_escalate_on_change():
    srv = Server()
    m = _model(srv)
    c = IdentityCanary(PROBES, on_change="escalate").attach(m)
    part = m.decision("team", "Which team?", "email", TEAMS)
    c.probe()
    srv.swapped = True
    assert c.probe()["verdict"] == "changed"
    d = part.decide("charged twice")
    assert d.escalate.startswith("model identity changed (canary event 2: 2 of 4 probes")
    assert "would have answered 'billing'" in d.escalate


def test_a_one_off_flip_is_not_confirmed():
    srv = Server()
    m = _model(srv)
    c = IdentityCanary(PROBES).attach(m)
    c.probe()
    srv.flip = 1                                               # one noisy reply, then the same model again
    ev = c.probe()
    assert ev["verdict"] == "same" and ev["unconfirmed"]["answers_changed"] == 1 and not c.changed
    srv.flip = 1
    ev = IdentityCanary(PROBES, confirm=False).attach(_model(srv)).probe()   # a fresh canary: its reference has the flip
    assert ev["verdict"] == "reference"


def test_probability_shift_and_served_name():
    srv = Server()
    m = _model(srv)
    c = IdentityCanary(PROBES, tol=0.15).attach(m)
    c.probe()
    srv.p = 0.8                                                # same answers, slightly other probabilities: tv 0.1
    ev = c.probe()
    assert ev["verdict"] == "same" and ev["tv"] == pytest.approx(0.1)
    srv.p = 0.6                                                # tv 0.3 > tol
    ev = c.probe()
    assert ev["verdict"] == "changed" and ev["answers_changed"] == 0 and ev["tv"] == pytest.approx(0.3)
    srv2 = Server()
    c2 = IdentityCanary(PROBES).attach(_model(srv2))
    c2.probe()
    srv2.name = "tiny-chat-int4"                               # the server says another model answered
    ev = c2.probe()
    assert ev["verdict"] == "changed" and ev["served"] == ["tiny-chat-int4"]


def test_unanswered_rounds_give_no_verdict():
    class Down:
        def __call__(self, req, timeout=None):
            raise TimeoutError("down")
    c = IdentityCanary(PROBES).attach(llm("http://llm.example/v1", "m", opener=Down(), sleep=lambda s: None,
                                          retries=0, workers=1))
    assert c.probe()["verdict"] == "unanswered" and c.identity is None


class Scorer:
    """A deterministic local scorer (replay re-runs it)."""
    model_id = "kw"

    def logits(self, items):
        return [np.array([2.0, 0.0]) if "charged" in it.text else np.array([0.0, 2.0]) for it in items]


def test_replay_never_probes():
    m = DecideModel(Scorer(), model_id="kw")
    c = IdentityCanary(PROBES, every=1).attach(m)
    part = m.decision("team", "Which team?", "email", TEAMS)
    cat = Catalog()
    part.question(cat, "route")
    s = System(cat, [Question("route", "Route", Answer.choice(TEAMS))])
    r = s.ask({"email": "I was charged twice"})
    seen, rounds = c.seen, len(c.events)
    assert seen >= 1 and rounds == seen
    rep = r.trace.replay(s.catalog, r.flow)
    assert rep["ok"] and [v for _, _, v in rep["models"]] == ["recomputed"]
    assert (c.seen, len(c.events)) == (seen, rounds)
    with no_probes():
        part.decide("charged")
    assert c.seen == seen


def test_arguments_checked():
    with pytest.raises(ValueError):
        IdentityCanary([])
    with pytest.raises(ValueError):
        IdentityCanary(PROBES, on_change="stop")
    with pytest.raises(ValueError):
        IdentityCanary(PROBES, every=0)
    with pytest.raises(ValueError):
        IdentityCanary(PROBES).probe()
    with pytest.raises(TypeError):
        IdentityCanary(PROBES).attach(object())
    assert math.isfinite(IdentityCanary(PROBES).tol)
