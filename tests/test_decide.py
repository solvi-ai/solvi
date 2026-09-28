"""solvi.decide: a decider model as a catalog part (closed set, provenance, model identity, safeguards), label-bias
correction without labels, few-shot shift / scale with teach, "other" as a threshold, and solvi.calibration.
A tiny fake scorer stands in for the ModernBERT decider; one optional test runs the real ONNX checkpoint if present."""
import io
import os
import runpy
import zlib
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from solvi import Answer, Catalog, Decision, Question, System
from solvi.calibration import accuracy_at, coverage_at, ece, reliability, summary, threshold_for
from solvi.decide import DecideModel, DecisionPart, Item, decision_of, prompt

KW = {"billing": ["charged", "refund", "invoice"], "technical": ["crash", "error", "bug"],
      "shipping": ["parcel", "delivery", "tracking"]}
TASK = "Which team?"
TEAMS = ["billing", "technical", "shipping"]


class FakeScorer:
    """Keyword logits plus a per-option bias (a label preference the unlabelled correction should remove)."""
    model_id = "test/fake-decider"

    def __init__(self, bias=None, version="1", noise=0.0):
        self.bias, self.version, self.noise = dict(bias or {}), version, noise
        self.calls = []

    def fingerprint(self):
        return f"fake-{self.version}"

    def logits(self, items):
        self.calls.append(list(items))
        out = []
        for it in items:
            low = it.text.lower()
            z = []
            for o in it.options:
                n = (zlib.crc32((o + it.text).encode()) % 1000) / 1000 - 0.5
                z.append(2.0 * sum(low.count(k) for k in KW.get(o, [])) + self.bias.get(o, 0.0) + self.noise * n)
            out.append(np.stack([np.array(z), np.array(z) - 1.0], 1))          # [K, 2]: single, multi
        return out


def texts(team, n, start=0):
    body = {"billing": "I was charged twice, please refund order {i}.", "technical": "The app shows error {i} and I see a crash.",
            "shipping": "My parcel {i} is lost, the tracking has no delivery date.",
            "other": "Do you sponsor football club number {i}?"}[team]
    return [body.format(i=i) for i in range(start, start + n)]


def model(**kw):
    return DecideModel(FakeScorer(**kw), meta={"format": "test", "temperature": 1.0})


def team_catalog(m, options=TEAMS, min_confidence=None, as_rule=False, option_order="given"):
    cat = Catalog()
    part = m.decision("team", TASK, text_fact="email", options=options, option_order=option_order)
    if as_rule:
        q = part.question(cat, "route", min_confidence=min_confidence)
    else:
        cat.fn(part)

        @cat.rule("route")
        def route(team):
            return team
        q = Question("route", "Route", Answer.choice(options), min_confidence=min_confidence)
    return cat, part, q


# --------------------------------------------------------------------------------------------------- scoring
def test_prompt_format_matches_the_decider():
    assert prompt("t", ["a", "b"]) == "[unused1] t[unused0] a[unused0] b"
    assert prompt("t", ["a", "b"], ["first", ""], multi=True) == "[unused2] t[unused0] a: first[unused0] b"


def test_score_softmax_multi_sigmoid_and_batch():
    m = model()
    p = m.score("charged twice, refund please", TASK, TEAMS)
    assert set(p) == set(TEAMS) and abs(sum(p.values()) - 1) < 1e-9 and max(p, key=p.get) == "billing"
    ps = m.score(["a crash and an error", "parcel lost"], TASK, TEAMS)
    assert [max(x, key=x.get) for x in ps] == ["technical", "shipping"]
    pm = m.score("charged twice, and the app shows an error", TASK, TEAMS, multi=True)
    assert pm["billing"] > 0.5 and pm["technical"] > 0.5 and pm["shipping"] < 0.5 and sum(pm.values()) > 1
    d = m.decide("charged twice, and the app shows an error", TASK, TEAMS, multi=True)
    assert d.value == ("billing", "technical")
    assert m.logits("refund", TASK, TEAMS)["billing"] == pytest.approx(2.0)


def test_cache_scores_each_text_once_and_descriptions_reach_the_scorer():
    sc = FakeScorer()
    m = DecideModel(sc)
    m.score("refund", TASK, {"billing": "money", "technical": "bugs", "shipping": ""})
    m.score("refund", TASK, {"billing": "money", "technical": "bugs", "shipping": ""})
    assert len(sc.calls) == 1
    it = sc.calls[0][0]
    assert isinstance(it, Item) and it.descriptions == ("money", "bugs", "") and it.options == ("billing", "technical", "shipping")


def test_fingerprint_and_model_id():
    a, b = model(), model(version="2")
    assert a.fingerprint() == model().fingerprint() and a.fingerprint() != b.fingerprint()
    assert a.model_id == "test/fake-decider"
    part = a.decision("team", TASK, "email", TEAMS)
    assert part.model_id == "test/fake-decider" and len(part.fingerprint()) == 16


# --------------------------------------------------------------------------------------------------- the catalog part
def test_decision_part_in_a_catalog_is_decided_with_model_identity_and_replays():
    m = model()
    cat, part, q = team_catalog(m)
    s = System(cat, [q])
    res = s.ask({"email": "I was charged twice, refund please"})
    assert res["route"].answer == "billing"
    rec = next(r for r in res.trace.records if r.name == "team")
    assert rec.origin == "decided" and rec.model["type"] == "DecisionPart" and rec.model["id"] == "test/fake-decider"
    assert rec.model["fp"] == part.fingerprint() and set(rec.probs) == set(TEAMS)
    assert cat.parts["team"].options == TEAMS                     # the closed set comes with the part
    au = res.audit("route")
    assert au.decided and au.decided[0]["name"] == "team" and au.counts["decided"] == 1
    assert res.trace.replay(cat)["ok"] and s.stats["model_outputs"] == 1


def test_answers_outside_the_options_are_impossible():
    m = model(noise=6.0)
    part = m.decision("team", TASK, "email", TEAMS + ["other"])
    for t in texts("billing", 5) + texts("other", 5) + ["", "x" * 50]:
        d = part(email=t)
        assert isinstance(d, Decision) and d.value in part.options and abs(sum(d.probs.values()) - 1) < 1e-9


def test_part_as_the_question_answer_with_min_confidence():
    m = model()
    cat, part, q = team_catalog(m, as_rule=True, min_confidence=0.9)
    assert q.answer.kind == "choice" and q.answer.options == TEAMS
    s = System(cat, [q])
    r = s.ask({"email": "refund, I was charged twice"})["route"]
    assert r.answer == "billing" and r.provenance == "decided" and r.probs and r.confidence > 0.9
    r = s.ask({"email": "hello"})["route"]                        # no keyword: uniform → abstains
    assert r.status == "abstain" and r.guard == "low_confidence" and "would have answered" in r.why
    assert s.stats["low_confidence"] == 1


def test_part_min_confidence_rejects_unsure_decisions():
    m = model()
    cat = Catalog()
    cat.fn(min_confidence=0.8)(m.decision("team", TASK, "email", TEAMS))

    @cat.rule("route")
    def route(team):
        return team
    s = System(cat, [Question("route", "", Answer.choice(TEAMS))])
    res = s.ask({"email": "hello"})
    assert res["route"].status == "abstain"
    assert "confidence" in next(r for r in res.trace.records if r.name == "team").error


def test_constraint_repairs_a_decided_answer_and_hard_check_overrides_it():
    m = model()
    cat = Catalog()
    part = m.decision("team", TASK, "email", TEAMS)

    @cat.fn
    def wants_refund(email):
        return "refund" in email.lower()

    @cat.rule("refund")
    def refund(wants_refund):
        return "yes" if wants_refund else "no"

    @cat.check(hard=True, then={"route": "shipping"})
    def not_a_parcel_claim(email):
        return "claim" not in email

    @cat.constraint
    def refunds_to_billing(route, refund):
        return refund == "no" or route == "billing"
    q = part.question(cat, "route", checkpoints=["not_a_parcel_claim"])
    s = System(cat, [q, Question("refund", "", Answer.yes_no())])
    res = s.ask({"email": "The app shows an error and a crash; refund me"})
    assert res["route"].answer == "billing" and res["route"].repaired[0] == "technical" and res.feasible
    assert s.stats["constraint_repairs"] == 1
    res = s.ask({"email": "claim: I was charged twice"})
    assert res["route"].answer == "shipping" and res["route"].status == "forced"


def test_multi_label_decision_with_none():
    m = model()
    part = m.decision("issues", "Which issues?", "email", TEAMS + ["none"], multi=True)
    assert part(email="charged twice and a crash").value == ("billing", "technical")
    d = part(email="hello there")
    assert d.value == ("none",) and set(d.probs) == set(TEAMS + ["none"])
    cat = Catalog()
    q = part.question(cat)
    assert q.answer.kind == "multi"
    r = System(cat, [q]).ask({"email": "a parcel and an invoice"})["issues"]
    assert r.answer == ("billing", "shipping")


# --------------------------------------------------------------------------------------------------- adaptation
def test_adapt_removes_label_bias_without_labels_and_changes_the_fingerprint():
    m = model(bias={"technical": 5.0})
    cat, part, q = team_catalog(m, as_rule=True)
    s = System(cat, [q])
    test = [(t, y) for y in TEAMS for t in texts(y, 5, 100)]

    def acc():
        return np.mean([s.ask({"email": t})["route"].answer == y for t, y in test])
    before_acc, fp0 = acc(), part.fingerprint()
    old = s.ask({"email": texts("billing", 1)[0]})
    unlabelled = [t for y in TEAMS for t in texts(y, 10)]
    a = part.adapt(unlabelled)
    assert abs(sum(a.bias)) < 1e-9 and a.bias[1] > 2.5 and a.n_unlabelled == 30
    assert acc() > before_acc and acc() == 1.0
    assert part.fingerprint() != fp0 and m.metadata()["adaptations"][0]["n_unlabelled"] == 30
    rep = old.trace.replay(cat)
    assert not rep["ok"] and "model changed since this decision" in rep["mismatches"][0][2]
    other = m.decision("other_part", "Another task?", "email", ["a", "b"])
    fp_other = other.fingerprint()
    part.adapt(unlabelled[:5])
    assert other.fingerprint() == fp_other                       # adapting one decision does not touch the others


def test_fit_S_shift_scale_and_temperature():
    m = model(bias={"technical": 1.5}, noise=3.0)
    part = m.decision("team", TASK, "email", TEAMS)
    ex = [(t, y) for y in TEAMS for t in texts(y, 6)]
    a = part.fit(ex)
    assert a.n_labelled == 18 and a.scale is not None and len(a.shift) == 3 and 0.25 <= a.temperature <= 8
    assert a.shift[1] < a.shift[0]                                 # the preferred label is shifted down
    test = [(t, y) for y in TEAMS for t in texts(y, 10, 200)]
    assert np.mean([part(email=t).value == y for t, y in test]) == 1.0


def test_teach_updates_the_shift_at_once_and_system_teach_routes_to_the_decision():
    m = model(bias={"technical": 3.0})
    cat, part, q = team_catalog(m, as_rule=True)
    s = System(cat, [q])
    t = "Hello, I have a question about order 7"                   # no keyword: the bias decides → technical
    assert s.ask({"email": t})["route"].answer == "technical"
    fp = part.fingerprint()
    p0 = part.score(t)["shipping"]
    ms = s.teach("route", {"email": t}, "shipping")
    assert ms is not None and ms < 100
    assert part.adaptation.n_labelled == 1 and part.score(t)["shipping"] > p0 and part.fingerprint() != fp
    for _ in range(6):
        s.teach("route", {"email": t}, "shipping")
    assert s.ask({"email": t})["route"].answer == "shipping"
    # a rule that passes a decided fact on is routed too
    cat2, part2, q2 = team_catalog(model(bias={"technical": 3.0}))
    assert decision_of(cat2, "route") is part2
    assert System(cat2, [q2]).teach("route", {"email": t}, "shipping") is not None and part2.adaptation.n_labelled == 1


def test_teach_is_consistent_with_a_later_adapt():
    m = model(bias={"technical": 3.0})
    part = m.decision("team", TASK, "email", TEAMS)
    part.teach(texts("billing", 1)[0], "billing")
    a = part.adapt([t for y in TEAMS for t in texts(y, 4)])
    assert a.n_labelled == 1 and a.bias is not None and a.shift is not None


# --------------------------------------------------------------------------------------------------- "other"
def test_other_is_a_threshold_not_a_scored_label():
    sc = FakeScorer()
    m = DecideModel(sc, meta={"other_threshold": 0.6, "temperature": 1.0})
    part = m.decision("team", TASK, "email", TEAMS + ["other"], option_order="given")
    d = part(email="hello")
    assert d.value == "other" and sc.calls[-1][0].options == tuple(TEAMS)     # the model never sees "other"
    assert d.conf == pytest.approx(1 - 1 / 3)
    d = part(email="refund, I was charged twice")
    assert d.value == "billing" and d.probs["other"] < d.probs["billing"]
    assert max(d.probs, key=d.probs.get) == d.value


def test_other_threshold_is_fitted_on_examples_that_include_other():
    m = model(noise=2.0)
    part = m.decision("team", TASK, "email", TEAMS + ["other"])
    ex = [(t, y) for y in TEAMS + ["other"] for t in texts(y, 8)]
    a = part.fit(ex)
    assert a.other_threshold is not None
    test = [(t, y) for y in TEAMS + ["other"] for t in texts(y, 10, 300)]
    assert np.mean([part(email=t).value == y for t, y in test]) >= 0.9
    part.reset()
    assert part.adaptation is None
    part.fit([(t, y) for y in TEAMS for t in texts(y, 8)])       # no "other" examples: the default threshold stays
    assert part.adaptation.other_threshold is None


def test_other_can_be_named_or_disabled():
    m = model()
    part = m.decision("team", TASK, "email", TEAMS + ["misc"], other="misc")
    assert part.spec.other == "misc" and part(email="hello").value == "misc"
    off = m.decision("team2", TASK, "email", TEAMS + ["other"], other=False, option_order="given")
    assert off.spec.real == TEAMS + ["other"]


def test_save_and_load_adaptations(tmp_path):
    m = model(bias={"technical": 3.0})
    part = m.decision("team", TASK, "email", TEAMS)
    part.adapt([t for y in TEAMS for t in texts(y, 4)])
    part.fit([(t, y) for y in TEAMS for t in texts(y, 4)])
    f = tmp_path / "adapt.json"
    m.save_adaptations(f)
    m2 = model(bias={"technical": 3.0})
    m2.load_adaptations(f)
    part2 = m2.decision("team", TASK, "email", TEAMS)
    assert part2.fingerprint() == part.fingerprint() and part2.score("hello") == pytest.approx(part.score("hello"))
    with pytest.raises(ValueError):
        model(version="9").load_adaptations(f)


# --------------------------------------------------------------------------------------------------- calibration
def test_calibration_helpers():
    conf = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    ok = [1, 1, 1, 0, 1, 0]
    assert coverage_at(conf, ok, 0.9) == pytest.approx(0.5)
    assert threshold_for(conf, ok, 0.9) == pytest.approx(0.8)
    assert coverage_at(conf, ok, 1.0) == pytest.approx(0.5) and coverage_at([0.9], [0], 0.9) == 0.0
    assert threshold_for([0.9], [0], 0.9) is None
    acc, cov = accuracy_at(conf, ok, 0.75)
    assert acc == 1.0 and cov == pytest.approx(0.5)
    assert ece([1.0, 1.0], [1, 1]) == 0.0 and ece([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0)
    assert ece([0.9] * 10, [0] * 10) == pytest.approx(0.9)
    assert sum(b["n"] for b in reliability(conf, ok, 5)) == 6
    s = summary(conf, ok)
    assert s["n"] == 6 and s["accuracy"] == pytest.approx(4 / 6)


# --------------------------------------------------------------------------------------------------- example, real model
def test_example_13_runs_with_the_stand_in(monkeypatch):
    monkeypatch.delenv("SOLVI_DECIDE_MODEL", raising=False)
    out = io.StringIO()
    with redirect_stdout(out):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "13_decide_model.py"), run_name="__main__")
    text = out.getvalue()
    assert "stand-in" in text and "+ bias correction" in text and "+ S on 16 labelled" in text
    assert "decided" in text and "[forced]" in text and "model changed since this decision" in text
    assert "System.teach(team" in text and "low confidence" in text


def _real_model_dir():
    for p in (os.environ.get("SOLVI_DECIDE_MODEL"), os.path.expanduser("~/.cache/solvi_release/decide-base")):
        if p and os.path.isfile(os.path.join(p, "solvi_decide.json")) and os.path.isdir(os.path.join(p, "onnx")):
            return p
    return None


def test_real_onnx_decider_if_present():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    path = _real_model_dir()
    if path is None:
        pytest.skip("no solvi-decide checkpoint (set SOLVI_DECIDE_MODEL)")
    m = DecideModel.load(path, backend="onnx")
    assert m.backend.startswith("onnx") and m.temperature > 0 and len(m.fingerprint()) == 16
    opts = {"billing": "payments, refunds", "technical": "bugs, crashes", "shipping": "delivery, parcels", "other": "anything else"}
    d = m.decide("My card was charged twice for the same order, please refund one payment.", "Which team?", opts)
    assert d.value == "billing" and abs(sum(d.probs.values()) - 1) < 1e-6
    part = m.decision("team", "Which team?", "email", opts)
    assert isinstance(part, DecisionPart)
    cat = Catalog()
    s = System(cat, [part.question(cat)])
    res = s.ask({"email": "The app crashes when I open settings."})
    assert res["team"].answer == "technical" and res.trace.replay(cat)["ok"]
