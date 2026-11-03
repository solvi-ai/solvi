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


def test_confirmed_change_discards_cached_outputs_before_recalibration():
    srv = Server()
    m = _model(srv)
    c = IdentityCanary(PROBES).attach(m)
    texts = [f"Where is my parcel {i}" for i in range(4)]
    assert all(m.decide(t, "Which team?", TEAMS).value == "shipping" for t in texts)
    c.probe()
    srv.swapped = True
    assert c.probe()["verdict"] == "changed"
    calls = srv.requests
    part = m.decision("team", "Which team?", "email", TEAMS)
    part.fit([(t, "billing") for t in texts] * 2)
    assert srv.requests > calls, "recalibration must ask the new model, not read old logits"
    c.accept()
    logits = m.logits(texts[0], "Which team?", TEAMS)
    assert logits["billing"] > logits["shipping"]
    assert part.decide(texts[0]).value == "billing"


def test_same_and_unconfirmed_probe_rounds_keep_the_decision_cache():
    srv = Server()
    m = _model(srv)
    c = IdentityCanary(PROBES).attach(m)
    part = m.decision("team", "Which team?", "email", TEAMS)
    part.decide("a parcel")
    c.probe()
    assert c.probe()["verdict"] == "same"
    calls = srv.requests
    part.decide("a parcel")
    assert srv.requests == calls
    srv.flip = 1
    assert c.probe()["verdict"] == "unconfirmed"
    calls = srv.requests
    part.decide("a parcel")
    assert srv.requests == calls


def test_old_inflight_output_cannot_refill_the_cache_after_a_change():
    import concurrent.futures
    import threading

    entered, release = threading.Event(), threading.Event()

    class SlowScorer:
        model_id = "swappable"
        swapped = False

        def logits(self, items):
            old = not self.swapped
            if old and any(it.text == "request" for it in items):
                entered.set()
                assert release.wait(10)
            return [np.array([3.0, -3.0]) if old else np.array([-3.0, 3.0]) for _ in items]

    scorer = SlowScorer()
    model = DecideModel(scorer, meta={})
    canary = IdentityCanary([("probe", "Which?", ["a", "b"])]).attach(model)
    canary.probe()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(model.decide, "request", "Which?", ["a", "b"])
        try:
            assert entered.wait(10)
            scorer.swapped = True
            assert canary.probe()["verdict"] == "changed"
        finally:
            release.set()
        assert old.result(timeout=10).value == "a"
    assert model.decide("request", "Which?", ["a", "b"]).value == "b"


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
    assert ev["verdict"] == "unconfirmed" and ev["unconfirmed"]["answers_changed"] == 1 and not c.changed
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


def _routed(srv, **kw):
    m = _model(srv)
    c = IdentityCanary(PROBES, **kw).attach(m)
    part = m.decision("team", "Which team?", "email", TEAMS)
    cat = Catalog()
    part.question(cat, "route")
    return c, System(cat, [Question("route", "Route", Answer.choice(TEAMS))])


def test_the_mark_reaches_drift_through_responses_and_a_dispatcher():
    import warnings

    from solvi.core.dispatch import Dispatcher
    from solvi.core.guarantees.canary import changed_of
    srv = Server()
    c, s = _routed(srv)
    c.probe()
    mon = DriftMonitor(window=4, sequential=False)
    for i in range(4):
        assert not mon.observe(s.ask({"email": f"charged {i}"}), question="route")["drift"]
    srv.swapped = True
    assert c.probe()["verdict"] == "changed"
    r = s.ask({"email": "charged after"})
    assert changed_of(r)["event"] == 2 and changed_of(r, "route") == changed_of(r)
    rep = mon.observe(r, question="route")
    assert rep["drift"] and "identity" in rep["flags"]
    # the same through a Dispatcher (System 1 only: the drift is warned about, there is no slow path)
    srv2 = Server()
    c2, s2 = _routed(srv2)
    c2.probe()
    disp = Dispatcher(s2, monitor=DriftMonitor(window=4, sequential=False))
    for i in range(4):
        disp.ask({"email": f"charged {i}"})
    assert disp.drift is None
    srv2.swapped = True
    c2.probe()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        disp.ask({"email": "charged after"})
    assert disp.drift is not None and "identity canary event 2" in disp.drift
    assert any("drift flagged" in str(w.message) for w in caught)


def test_a_marked_decision_stays_out_of_the_reference():
    srv = Server()
    c, s = _routed(srv)
    c.probe()
    srv.swapped = True
    c.probe()
    mon = DriftMonitor(window=4, sequential=False)
    rep = mon.observe(s.ask({"email": "charged"}), question="route")
    assert rep["phase"] == "reference" and rep["drift"] and mon.reference == []


class Raising:
    """A local scorer whose probe path breaks: texts of the probe set raise, the real inputs are scored."""
    model_id = "raising"

    def logits(self, items):
        if any("parcel" in it.text.lower() for it in items):
            raise RuntimeError("probe backend exploded")
        return [np.array([2.0, 0.0]) if "charged" in it.text else np.array([0.0, 2.0]) for it in items]


def test_a_failing_scheduled_probe_leaves_the_decision_alone():
    m = DecideModel(Raising(), model_id="raising")
    c = IdentityCanary(PROBES, every=1).attach(m)
    part = m.decision("team", "Which team?", "email", TEAMS)
    d = part.decide("I was charged")
    assert d.value == "billing" and d.escalate is None and "canary" not in d.extra
    ev = c.events[-1]
    assert ev["verdict"] == "error" and ev["error"] == "RuntimeError: probe backend exploded" and ev["at"] == 1
    assert c.seen == 1 and not c.changed
    with pytest.raises(RuntimeError):                          # on demand: raised
        c.probe()


def test_a_probe_round_does_not_hold_up_other_decisions():
    import threading

    gate, started = threading.Event(), threading.Event()

    class Slow:
        model_id = "slow"

        def logits(self, items):
            if any("parcel" in it.text.lower() for it in items):          # the probe round waits at the gate
                started.set()
                assert gate.wait(10)
            return [np.array([2.0, 0.0]) if "charged" in it.text else np.array([0.0, 2.0]) for it in items]

    m = DecideModel(Slow(), model_id="slow")
    c = IdentityCanary(PROBES, every=1).attach(m)
    part = m.decision("team", "Which team?", "email", TEAMS)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("a", part.decide("charged A")))
    t.start()
    assert started.wait(10)
    done = threading.Event()
    u = threading.Thread(target=lambda: (out.setdefault("b", part.decide("charged B")), done.set()))
    u.start()
    assert done.wait(10), "a decision waited for another thread's probe round"
    assert "canary" not in out["b"].extra                      # its round was due but one was running: not started
    gate.set()
    t.join(10)
    u.join(10)
    assert out["a"].extra["canary"] == {"probed": 1} and [e["verdict"] for e in c.events] == ["reference"]
    assert c.seen == 2


def _local_adapter_model(monkeypatch):
    """Faithful active adapter seam: logits see the same mutable state as switching."""
    from solvi.experimental import lora
    from solvi.core.deciders.model import lora_key
    from solvi.core.deciders.adapt import _Spec
    srv = Server()
    m = _model(srv)
    state = {"active": "local"}
    m.scorer._lora = state
    monkeypatch.setattr(lora, "_activate", lambda scorer, st, name: st.update(active=name))
    original = m.scorer.logits
    def logits(items):
        out = original(items)
        if state["active"] is not None:
            for o in out:
                o["logits"] = np.asarray(o["logits"])[::-1].copy()
        return out
    m.scorer.logits = logits
    # Use the real LoRA suspension implementation without loading torch/weights.
    class Adapter:
        provider_using = lora.LoraAdapter.provider_using
        def fingerprint(self):
            return "local"
        def using(self, scorer, active=True):
            return lora.using(scorer, "local" if active else None)
    # _net uses the same gate; state lock is the real adapter lock.
    import threading
    state["lock"] = threading.RLock()
    m.loras[lora_key(_Spec("Which team?", TEAMS))] = Adapter()
    return m, srv, state


def test_local_adapter_does_not_change_provider_identity(monkeypatch):
    m, srv, state = _local_adapter_model(monkeypatch)
    c = IdentityCanary(PROBES).attach(m)
    state["active"] = None
    assert c.probe()["verdict"] == "reference"
    state["active"] = "local"
    assert c.probe()["verdict"] == "same"
    assert state["active"] == "local"
    # The adapter really changes ordinary model scoring.
    assert m.decide("Where is my parcel", "Which team?", TEAMS).value == "billing"


def test_provider_swap_detected_under_active_local_adapter(monkeypatch):
    m, srv, state = _local_adapter_model(monkeypatch)
    c = IdentityCanary(PROBES).attach(m)
    c.probe()
    srv.swapped = True
    assert c.probe()["verdict"] == "changed"
    assert state["active"] == "local"


def test_probe_restores_local_adapter_after_error(monkeypatch):
    m, srv, state = _local_adapter_model(monkeypatch)
    c = IdentityCanary(PROBES).attach(m)
    def fail(items):
        assert state["active"] is None
        raise RuntimeError("probe failure")
    m.scorer.logits = fail
    with pytest.raises(RuntimeError, match="probe failure"):
        c.probe()
    assert state["active"] == "local"
    assert c.reference is None


def test_adapter_without_raw_support_is_explicitly_unsupported(monkeypatch):
    m, srv, state = _local_adapter_model(monkeypatch)
    next(iter(m.loras.values())).provider_using = None
    c = IdentityCanary(PROBES).attach(m)
    with pytest.raises(NotImplementedError, match="raw provider identity unsupported"):
        c.probe()
    assert c.reference is None and state["active"] == "local"


@pytest.mark.parametrize("registry", ["loras", "_lora_extra"])
def test_provider_probe_after_concurrent_last_adapter_removal(monkeypatch, registry):
    import concurrent.futures
    import contextlib
    import threading
    from solvi.core.deciders.adapt import _Spec
    from solvi.core.deciders.model import net_gate

    m, srv, state = _local_adapter_model(monkeypatch)
    adapter = next(iter(m.loras.values()))
    if registry == "_lora_extra":
        m._lora_extra = {"local": adapter}
        m.loras = {}
    original_plain = m._plain
    checked, removed = threading.Event(), threading.Event()

    @contextlib.contextmanager
    def paused_plain():
        with original_plain() as plain:
            assert not plain
            yield plain
        # Reproduce removal precisely after the unprotected False check.
        checked.set()
        assert removed.wait(10)

    monkeypatch.setattr(m, "_plain", paused_plain)
    sp = _Spec("Which team?", TEAMS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        probe = pool.submit(m._score_provider, [(sp, "Where is my parcel")])
        try:
            assert checked.wait(10)
            # Same gate / adapter-lock order as unstage and uninstall, without
            # loading a GPU backend. Removal leaves the plain network active.
            with net_gate(m.scorer), state["lock"]:
                setattr(m, registry, {})
                state["active"] = None
        finally:
            removed.set()
        outputs, used = probe.result(timeout=10)
    assert not used and srv.requests == 1
    np.testing.assert_allclose(outputs[0]["logits"], np.log([0.1, 0.9]))
    assert state["active"] is None
    assert not m.loras and not m._lora_extra
    assert state["lock"].acquire(blocking=False)
    state["lock"].release()
    # Both gate modes remain usable after the race (no leaked/upgrade lock).
    with net_gate(m.scorer):
        with original_plain() as plain:
            assert plain


class ScriptedProbeProvider:
    model_id = 'scripted-fixed-name'

    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def logits(self, items):
        return [np.log([p, 1 - p]) for p in (next(self.probabilities) for _ in items)]


def _scripted_canary(probabilities):
    return IdentityCanary([('fixed probe', 'Which?', ['a', 'b'])]).attach(
        DecideModel(ScriptedProbeProvider(probabilities), meta={}))


def test_s374_partial_route_preserves_both_reports_and_is_unconfirmed():
    c = _scripted_canary([.9, .1, .9])
    c.probe()
    event = c.probe()
    assert event['verdict'] == 'unconfirmed' and not c.changed
    assert event['unconfirmed']['differs'] is True
    assert event['confirmation']['differs'] is False
    assert event['unconfirmed']['answers_changed'] == 1
    assert event['confirmation']['answers_changed'] == 0
    assert event['diagnostics']['repeat_agreement'] == 0
    assert event['diagnostics']['probes'][0]['probability_shift_tv'] == pytest.approx(.8)


def test_s375_fixed_probe_blindness_and_tolerance_boundary_are_reported():
    c = _scripted_canary([.999, .999, .999, .999])
    c.probe()
    assert c.probe()['verdict'] == 'same'
    report = c.diagnose()
    assert report['repeat_agreement'] == 1
    assert report['very_clear_reference_count'] == 1
    assert report['probes'][0]['reference_top_gap'] == pytest.approx(.998)
    assert report['blind_spots'] and c.tol == .15
    assert len(c.events) == 2 and not c.changed
    # No answer flip, and a shift just below the existing TV tolerance.
    boundary = _scripted_canary([.9, .7501])
    boundary.probe()
    event = boundary.probe()
    assert event['verdict'] == 'same'
    assert event['diagnostics']['probes'][0]['probability_shift_tv'] == .1499


def test_s375_boundary_probe_repeat_disagreement_exposes_noise_without_tuning():
    c = _scripted_canary([.51, .51, .49])
    c.probe()
    reference = c.reference
    report = c.diagnose()
    assert report['probes'][0]['reference_top_gap'] == .02
    assert report['repeat_agreement'] == 0
    assert c.reference is reference and not c.changed
    assert len(c.events) == 1 and c.tol == .15
