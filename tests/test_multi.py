"""solvi.multi: several models, one decision — a cascade, a vote and a route over decision parts, every proposal in the
trace; act_guard on the combination keeps P(answered alone and wrong) ≤ risk although a cascade's loss is not monotone in
its threshold (checked by simulation, as test_guarantees does for one part); replay re-runs or verifies the proposals."""
import numpy as np
import pytest

from solvi import Answer, Catalog, Question, System
from solvi.decide import DecideModel
from solvi.multi import Cascade, Route, Vote, _src
from solvi.runtime import Trace
from test_decide import TEAMS, FakeScorer
from test_guarantees import _labelled

HARD = "The app crashed after the refund of order 7."          # both keywords: the models are unsure
CLEAR = "I was charged twice on invoice 3, please refund."                # billing, sure


def _model(name, noise, **kw):
    return DecideModel(FakeScorer(noise=noise, version=name, **kw), meta={"format": "test", "temperature": 1.0},
                       model_id=name)


def _parts(small_below=0.8, large_below=0.6):
    small, large = _model("small", 3.0), _model("large", 1.0)
    return (small.decision("team", "Which team?", "email", TEAMS, escalate_below=small_below),
            large.decision("team", "Which team?", "email", TEAMS, escalate_below=large_below), small, large)


# --------------------------------------------------------------------------------------------------- construction
def test_the_parts_must_answer_the_same_question():
    s, l_, small, _ = _parts()
    with pytest.raises(ValueError, match="same question"):
        Cascade([s, small.decision("team", "Which team?", "email", TEAMS + ["sales"])])
    with pytest.raises(ValueError, match="same question"):
        Vote([s, small.decision("urgent", "Urgent?", "email", bool)])
    with pytest.raises(TypeError):
        Cascade([s, lambda email: "billing"])
    with pytest.raises(ValueError):
        Vote([s, l_], rule="most")
    other_task = small.decision("team2", "Route this email to a team", "body", TEAMS)   # task and facts may differ
    c = Cascade([s, other_task], name="team")
    assert c.facts == ["email", "body"] and c.options == TEAMS and c.__name__ == "team"


# --------------------------------------------------------------------------------------------------- cascade
def test_cascade_asks_the_large_model_only_when_the_small_one_escalates():
    s, l_, _, large = _parts()
    c = Cascade([s, l_], costs=[45, 137])
    d = c(email=CLEAR)
    assert d.value == "billing" and d.escalate is None
    assert d.extra["answered_by"] == 0 and d.extra["calls"] == 1 and len(d.extra["stages"]) == 1
    assert large.scorer.calls == []                                   # the large model was never asked
    st = d.extra["stages"][0]
    assert st["model"] == "small" and st["value"] == "billing" and st["signal"] == "confidence" and st["escalate"] is None
    d = c(email=HARD)
    assert d.escalate.startswith("model escalated: every model of the cascade escalated")
    assert "team (small)" in d.escalate and "team (large)" in d.escalate
    assert d.extra["answered_by"] is None and d.extra["calls"] == 2 and d.value == d.extra["stages"][1]["value"]
    u = c.usage()
    assert u["asked"] == 2 and u["calls"] == {"0:team": 2, "1:team": 1} and u["per_question"] == 1.5
    assert u["cost"] == pytest.approx((2 * 45 + 137) / 2)


def test_cascade_answers_with_the_large_model_when_it_is_sure():
    s, _, _, large = _parts(small_below=0.999)
    l_ = large.decision("team", "Which team?", "email", TEAMS, escalate_below=0.5)
    d = Cascade([s, l_])(email=CLEAR)
    assert d.extra["answered_by"] == 1 and d.extra["stages"][0]["escalate"] and d.escalate is None
    assert d.probs == pytest.approx(d.extra["stages"][1]["probs"], abs=1e-6)


# --------------------------------------------------------------------------------------------------- vote
def test_vote_all_answers_on_agreement_and_escalates_with_the_proposals_on_disagreement():
    s, l_, _, _ = _parts(small_below=0.0, large_below=0.0)
    biased = _model("biased", 0.0, bias={"shipping": 9.0}).decision("team", "Which team?", "email", TEAMS)
    v = Vote([s, l_])
    d = v(email=CLEAR)
    assert d.value == "billing" and d.escalate is None and [x["value"] for x in d.extra["votes"]] == ["billing"] * 2
    assert d.extra["rule"] == "all" and d.extra["calls"] == 2
    assert d.conf == pytest.approx(min(x["confidence"] for x in d.extra["votes"]))
    d = Vote([s, l_, biased])(email=CLEAR)
    assert d.escalate.startswith("model escalated: the models disagree (all)") and "team (biased) 'shipping'" in d.escalate
    d = Vote([s, l_, biased], rule="majority")(email=CLEAR)
    assert d.value == "billing" and d.escalate is None


def test_vote_escalates_when_an_agreeing_model_is_unsure():
    s, l_, _, _ = _parts(small_below=0.9999, large_below=0.0)
    d = Vote([s, l_])(email=CLEAR)
    assert d.escalate and "agree on 'billing'" in d.escalate and "team (small)" in d.escalate


class PassScorer(FakeScorer):
    """A fake that can score several questions about one input in one pass."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.passes = []

    def logits_pass(self, passes):
        self.passes.append(len(passes))
        return [self.logits(list(p.items)) for p in passes]


def test_vote_asks_parts_of_one_model_in_one_pass():
    sc = PassScorer()
    m = DecideModel(sc, meta={"format": "test", "temperature": 1.0, "multi_question": {"layout": "concat"}})
    a = m.decision("team", "Which team?", "email", TEAMS)
    b = m.decision("team", "Route this email to a team", "email", TEAMS)
    d = Vote([a, b])(email="I was charged twice")
    assert d.value == "billing" and sc.passes == [1] and m.passes == 1


# --------------------------------------------------------------------------------------------------- route
def test_route_picks_the_part_by_code_and_only_that_model_runs():
    s, l_, _, large = _parts(small_below=0.0, large_below=0.0)

    def long_email(email):
        return len(email) > 40
    r = Route({long_email: l_, "vip": l_}, default=s)
    assert r.facts == ["email", "vip"]
    d = r(email="Refund, charged twice.", vip=False)
    assert d.extra["route"] == {"to": 2, "part": "team", "by": "default"} and d.extra["routed"]["model"] == "small"
    assert large.scorer.calls == []
    d = r(email="Refund, charged twice.", vip=True)
    assert d.extra["route"]["by"] == "vip" and d.extra["routed"]["model"] == "large"
    d = r(email="I was charged twice for order 5521 and would like a refund please.", vip=False)
    assert d.extra["route"]["by"] == "long_email" and d.extra["calls"] == 1


# --------------------------------------------------------------------------------------------------- catalog, trace, replay
def _system(comb):
    cat = Catalog()
    cat.fn(comb)

    @cat.rule("route")
    def route(team):
        return team
    return cat, System(cat, [Question("route", "Route", Answer.choice(TEAMS))])


@pytest.mark.parametrize("make", [lambda s, l_: Cascade([s, l_]), lambda s, l_: Vote([s, l_]),
                                  lambda s, l_: Route({"vip": l_}, default=s), lambda s, l_: Cascade([s, Vote([s, l_])])])
def test_a_combination_is_a_decided_catalog_part_that_replays(make):
    s, l_, _, _ = _parts()
    comb = make(s, l_)
    cat, sys_ = _system(comb)
    assert cat.parts["team"].options == TEAMS
    for email in (CLEAR, HARD):
        res = sys_.ask({"email": email, "vip": True})
        rec = next(r for r in res.trace.records if r.name == "team")
        assert rec.origin == "decided" and rec.model["type"] == type(comb).__name__ and rec.model["fp"] == comb.fingerprint()
        assert rec.extra["calls"] >= 1
        for tr in (res.trace, Trace.from_json(res.trace.to_json())):
            assert tr.replay(cat)["ok"] and tr.replay(cat)["models"] == [(1, "team", "recomputed")]
            assert tr.replay(cat, trust_models=True)["ok"]
        au = str(res.audit("route"))
        assert {"Cascade": "cascade ", "Vote": "vote ", "Route": "route "}[type(comb).__name__] in au


def test_replay_catches_changed_proposals_and_verifies_recorded_ones():
    s, l_, _, large = _parts()
    c = Cascade([s, l_])
    cat, sys_ = _system(c)
    res = sys_.ask({"email": HARD})
    large.scorer.noise = 0.0                      # the same fingerprint, other outputs: a silently changed model
    large._cache.clear()
    rep = res.trace.replay(cat)
    assert not rep["ok"] and "recorded stages differ" in rep["mismatches"][0][2]
    assert res.trace.replay(cat, trust_models=True)["ok"]              # the recorded proposals are still consistent
    rec = next(r for r in res.trace.records if r.name == "team")
    assert c.check_record(rec) == []
    import copy
    bad = copy.deepcopy(rec)
    bad.extra["stages"][1]["escalate"] = None     # a stage that answered, yet the cascade escalated
    assert c.check_record(bad)
    bad = copy.deepcopy(rec)
    bad.extra["answered_by"], bad.error, bad.value = 0, None, "shipping"
    assert c.check_record(bad)
    s.teach(HARD, "billing")                      # a real change of a part: the combination's fingerprint changes
    rep = res.trace.replay(cat)
    assert not rep["ok"] and "model changed since this decision" in rep["mismatches"][0][2]


def test_system_teach_teaches_every_part_and_question_makes_it_an_answer():
    s, l_, _, _ = _parts()
    c = Cascade([s, l_])
    cat = Catalog()
    q = c.question(cat, "route")
    assert q.answer.kind == "choice" and q.answer.options == TEAMS
    sys_ = System(cat, [q])
    assert sys_.ask({"email": "I was charged twice"})["route"].answer == "billing"
    assert sys_.teach("route", {"email": HARD}, "billing") is not None
    assert s.adaptation.n_labelled == 1 and l_.adaptation.n_labelled == 1


# --------------------------------------------------------------------------------------------------- guarantees
class Table:
    """A scorer reading prepared logits by input text (a synthetic stream with known right answers)."""

    def __init__(self, name):
        self.model_id, self.table = name, {}

    def fingerprint(self):
        return self.model_id

    def logits(self, items):
        return [np.stack([self.table[it.text], self.table[it.text] - 1.0], 1) for it in items]


def _stream(rng, tag, n, S, L, gold_of):
    """n examples; the small model is right with probability 0.35 + 0.6·conf, the large with 0.5 + 0.5·conf — their
    confidences independent, so passing a question on can turn a wrong answer right and back (a non-monotone loss)."""
    out = []
    for i in range(n):
        t, y = f"{tag} {i}", ("a", "b")[rng.integers(2)]
        for tab, lo, gain in ((S, 0.35, 0.6), (L, 0.5, 0.5)):
            c = rng.uniform(0.5, 1.0)
            ok = rng.uniform() < lo + gain * (c - 0.5) * 2 * 0.9     # 0.35–0.89, 0.5–0.95
            pred = y if ok else ("b" if y == "a" else "a")
            tab.table[t] = np.log(np.array([c, 1 - c]) if pred == "a" else np.array([1 - c, c]))
        gold_of[t] = y
        out.append((t, y))
    return out


def test_act_guard_on_a_cascade_and_a_vote_keeps_the_risk_on_new_inputs():
    rng = np.random.default_rng(0)
    S, L = Table("S"), Table("L")
    ms, ml = (DecideModel(x, meta={"format": "test", "temperature": 1.0}) for x in (S, L))
    s, l_ = ms.decision("q", "Q?", "doc", ["a", "b"]), ml.decision("q", "Q?", "doc", ["a", "b"])
    gold = {}
    risks = {"cascade": [], "vote": []}
    nonmono = 0
    for rep in range(25):
        cal = _stream(rng, f"c{rep}", 300, S, L, gold)
        test = _stream(rng, f"t{rep}", 600, S, L, gold)
        for key, comb in (("cascade", Cascade([s, l_])), ("vote", Vote([s, l_], rule="all"))):
            info = comb.act_guard(cal, risk=0.10)
            assert info["risk"] <= 0.10
            wrong = [d.escalate is None and d.value != y for d, (_, y) in zip(comb.decide([t for t, _ in test]), test)]
            risks[key].append(float(np.mean(wrong)))
            if key == "cascade" and rep < 5:          # the runtime agrees with the calibration's vectorized outcome
                for t, _ in cal[:50]:
                    st = comb.state(_src(t))
                    auto, keys, _, vals, _, _ = comb.vec(st, np.array([comb.threshold]))
                    d = comb.decide(t)
                    assert (d.escalate is None) == bool(auto[0]) and d.value == vals[keys[0]]
                grid = np.linspace(0.5, 1.0, 26)
                for t, y in cal:
                    auto, keys, _, vals, _, _ = comb.vec(comb.state(_src(t)), grid)
                    loss = auto & np.array([vals[k] != y for k in keys])
                    nonmono += bool(np.any(np.diff(loss.astype(int)) > 0))
    assert nonmono > 0                                   # the case the monotonization is for does occur
    for key, rs in risks.items():
        assert np.mean(rs) <= 0.10 + 0.01, key
        assert np.mean(rs) >= 0.05, key                  # not trivially conservative: it does answer alone


def test_act_guard_records_the_promise_and_reports_the_cost():
    small, large = _model("small", 3.0), _model("large", 1.0)
    c = Cascade([small.decision("team", "Which team?", "email", TEAMS),
                 large.decision("team", "Which team?", "email", TEAMS)], costs=[45, 137])
    fp = c.fingerprint()
    info = c.act_guard(_labelled(), risk=0.10)
    assert info["risk"] <= 0.10 and 0 < info["answered"] <= 1 and info["n"] == 240
    assert 1 <= info["calls"] <= 2 and 45 <= info["cost"] <= 182 and sum(info["answered_by"]) == pytest.approx(info["answered"])
    assert c.fingerprint() != fp and c.threshold == info["threshold"]
    d = c(email=HARD)
    assert d.extra["guarantee"]["method"] == "crc" and d.extra["threshold"] == info["threshold"]
    assert all("shared threshold" in s["escalate"] for s in d.extra["stages"] if s["escalate"])
    _, sys_ = _system(c)
    assert "guarantee   P(answered alone and wrong) ≤ 0.1" in str(sys_.ask({"email": HARD}).audit("route"))
    with pytest.raises(ValueError):
        c.act_guard([], risk=0.1)


def test_conformal_sets_for_a_vote():
    small, large = _model("small", 2.0), _model("large", 2.0, bias={"technical": 0.5})
    v = Vote([small.decision("team", "Which team?", "email", TEAMS), large.decision("team", "Which team?", "email", TEAMS)])
    v.act_guard(_labelled(), risk=0.10)
    info = v.conformal(_labelled(), coverage=0.90)
    assert info["n"] == 240 and 1 <= info["mean_size"] <= 3
    hits = [y in v(email=t).extra["candidates"] for t, y in _labelled(1000)]
    assert np.mean(hits) >= 0.85


def test_example_18_runs_with_the_stand_ins(monkeypatch):
    import io
    import runpy
    from contextlib import redirect_stdout
    from pathlib import Path
    for env in ("SOLVI_DECIDE_MODEL", "SOLVI_DECIDE_SMALL"):
        monkeypatch.delenv(env, raising=False)
    out = io.StringIO()
    with redirect_stdout(out):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "18_several_models.py"), run_name="__main__")
    text = out.getvalue()
    assert "cascade small → large" in text and "the models disagree" in text and "route by length" in text
    assert "stage 2 answered" in text and "replay: True" in text and "replay: False" not in text
