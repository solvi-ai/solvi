"""solvi.multi: several models, one decision — a cascade, a vote and a route over decision parts, every proposal in the
trace; act_guard on the combination keeps P(answered alone and wrong) ≤ risk although a cascade's loss is not monotone in
its threshold (checked by simulation, as test_guarantees does for one part); replay re-runs or verifies the proposals."""
import numpy as np
import pytest

from solvi import Answer, Catalog, Question, System
from solvi.decide import DecideModel, Facts
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
    return (small.decision("team", "Which team?", "email", TEAMS, min_confidence=small_below),
            large.decision("team", "Which team?", "email", TEAMS, min_confidence=large_below), small, large)


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
    u = c.calls()
    assert u["asked"] == 2 and u["calls"] == {"0:team": 2, "1:team": 1} and u["calls_per_question"] == 1.5
    with pytest.warns(DeprecationWarning, match=r"usage\(\) is deprecated: use calls\(\)"):
        assert c.usage()["calls_per_question"] == 1.5                 # the 0.7 name, one release
    assert u["cost"] == pytest.approx((2 * 45 + 137) / 2)


def test_cascade_answers_with_the_large_model_when_it_is_sure():
    s, _, _, large = _parts(small_below=0.999)
    l_ = large.decision("team", "Which team?", "email", TEAMS, min_confidence=0.5)
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


def test_a_route_keyed_by_a_fact_name_does_not_read_a_raw_input_as_that_fact():
    s, l_, _, large = _parts(small_below=0.0, large_below=0.0)
    r = Route({"vip": l_}, default=s)
    with pytest.raises(ValueError, match="reads the fact"):          # every non-empty text used to go to the vip route
        r.decide("Refund, charged twice.")
    with pytest.raises(ValueError, match="reads the fact"):          # and the shared threshold was calibrated on that
        r.act_guard([(CLEAR, "billing")] * 20, max_risk=0.5)
    assert large.scorer.calls == []
    with pytest.raises(ValueError, match="does not give"):           # a missing fact is not false
        r.decide(Facts(email="Refund, charged twice."))
    assert r.decide(Facts(email="Refund, charged twice.", vip=False)).extra["route"]["by"] == "default"
    assert r.decide({"email": "Refund, charged twice.", "vip": True}).extra["route"]["by"] == "vip"   # a state gives it
    info = r.act_guard([(Facts(email=CLEAR, vip=bool(i % 2)), "billing") for i in range(20)], max_risk=0.5)
    assert info["n"] == 20

    def long_email(email):                                           # a predicate of the input itself: as before
        return len(email) > 40
    assert Route({long_email: l_}, default=s).decide("Refund, charged twice.").extra["route"]["by"] == "default"

    def both(email, vip):
        return vip and len(email) > 40
    with pytest.raises(ValueError, match="reads the facts"):
        Route({both: l_}, default=s).decide("Refund, charged twice.")


def test_a_route_with_two_predicates_over_the_same_fact():
    s, l_, _, _ = _parts(small_below=0.0, large_below=0.0)

    def strict(mode):
        return mode == "strict"

    def stop(mode):
        return mode == "stop"
    r = Route({strict: l_, stop: l_}, default=s)                     # was: ValueError: duplicate parameter name: 'mode'
    assert r.facts == ["email", "mode"]
    assert [r(email="Refund, charged twice.", mode=m).extra["route"]["by"] for m in ("strict", "stop", "other")] == \
        ["strict", "stop", "default"]


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
            info = comb.act_guard(cal, max_risk=0.10)
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


def _scales_stream(rng, tag, n, S, L):
    """The small model's confidence spread over (0.5, 1); the other's packed near 1, as an LLM's from log-probabilities
    (1 − 10^−(3…7)) and weaker. Both informative: right with probability rising in the confidence's rank."""
    out = []
    for i in range(n):
        t, y = f"{tag} {i}", ("a", "b")[rng.integers(2)]
        u, v = rng.uniform(), rng.uniform()
        for tab, c, p in ((S, 0.5 + 0.5 * u, 0.6 + 0.39 * u), (L, 1 - 10 ** -(3 + 4 * v), 0.45 + 0.45 * v)):
            pred = y if rng.uniform() < p else ("b" if y == "a" else "a")
            tab.table[t] = np.log(np.array([c, 1 - c]) if pred == "a" else np.array([1 - c, c]))
        out.append((t, y))
    return out


def _scales_parts():
    S, L = Table("S"), Table("L")
    ms, ml = (DecideModel(x, meta={"format": "test", "temperature": 1.0}) for x in (S, L))
    return S, L, ms.decision("q", "Q?", "doc", ["a", "b"]), ml.decision("q", "Q?", "doc", ["a", "b"])


def test_rank_scale_lets_a_cascade_use_both_models_when_their_signals_differ_in_scale():
    rng = np.random.default_rng(3)
    S, L, s, l_ = _scales_parts()
    cal, test = _scales_stream(rng, "c", 600, S, L), _scales_stream(rng, "t", 3000, S, L)
    raw, rank = Cascade([s, l_]), Cascade([s, l_])
    ir = raw.act_guard(cal, max_risk=0.10)                        # raw is the default
    ik = rank.act_guard(cal, max_risk=0.10, scale="rank")
    assert ir["scale"] == "raw" and ik["scale"] == "rank" and rank.scale == "rank" and len(rank.ranks) == 2
    assert ir["answered_by"][0] < 0.05 and raw.scale == "raw" and raw.ranks is None
    assert any("no better than a single model" in w and 'scale="rank"' in w for w in ir["warnings"])
    assert ik["answered_by"][0] > 0.05 and "warnings" not in ik
    assert ik["answered"] > ir["answered"] + 0.05
    assert rank.guarantee["signal"] == "shared-rank"                  # a name, as a part's "act" / "confidence"
    assert raw.guarantee["signal"] == "shared"
    risks = {"raw": [], "rank": []}                            # the promise holds on new inputs, over calibrations
    for rep in range(10):
        c2, t2 = _scales_stream(rng, f"c{rep}", 300, S, L), _scales_stream(rng, f"t{rep}", 1000, S, L)
        for key in risks:
            comb = Cascade([s, l_])
            comb.act_guard(c2, max_risk=0.10, scale=key)
            ds = comb.decide([t for t, _ in t2])
            risks[key].append(np.mean([d.escalate is None and d.value != y for d, (_, y) in zip(ds, t2)]))
    assert all(np.mean(r) <= 0.10 + 0.01 for r in risks.values()), risks
    for t, _ in cal[:200] + test[:200]:                        # the runtime agrees with the vectorized calibration
        st = rank.state(_src(t))
        auto, keys, _, vals, _, _ = rank.vec(st, np.array([rank.threshold]))
        d = rank.decide(t)
        assert (d.escalate is None) == bool(auto[0]) and d.value == vals[keys[0]]
    with pytest.raises(ValueError, match="scale"):
        rank.act_guard(cal, scale="log")


def test_the_rank_threshold_maps_to_the_same_raw_threshold_per_model():
    from solvi.multi import _rank, _raw_t, _table
    rng = np.random.default_rng(0)
    tbl = _table(np.round(rng.uniform(size=300), 2))           # ties
    assert len(_table(rng.uniform(size=5000))) == 1024 and np.all(np.diff(_table(rng.uniform(size=5000))) >= 0)
    for t in [0.0, -np.inf, np.inf, 1.0, 1e-9] + [_rank(tbl, x) for x in tbl[::7]] + list(rng.uniform(size=50)):
        r = _raw_t(tbl, t)
        for s_ in np.concatenate([tbl, rng.uniform(-0.1, 1.1, 50), [-np.inf]]):   # signals are not +inf
            assert (_rank(tbl, s_) >= t) == (s_ >= r), (t, s_, r)


def test_old_calibration_files_load_on_the_raw_scale_bit_for_bit(tmp_path):
    import json
    from solvi.provenance import digest
    rng = np.random.default_rng(5)
    S, L, s, l_ = _scales_parts()
    cal, test = _scales_stream(rng, "c", 300, S, L), _scales_stream(rng, "t", 300, S, L)
    for make in (lambda: Cascade([s, l_]), lambda: Vote([s, l_])):
        raw = make()
        raw.act_guard(cal, max_risk=0.10)                          # the default scale
        # the fingerprint a combination had before scales existed: no scale in it
        g07 = {**raw.guarantee, "signal": "shared threshold on each model's signal"}   # the signal as 0.7 named it
        old_fp = digest(type(raw).__name__, raw._describe(), [m.fingerprint() for m in raw.members],
                        {"threshold": raw.threshold, "guarantee": g07})
        assert raw.fingerprint() == old_fp
        f = raw.save_calibration(tmp_path / "old.json")
        rec = json.loads(f.read_text())
        assert rec["scale"] == "raw" and "ranks" not in rec
        del rec["scale"]                                       # as written before 0.7
        f.write_text(json.dumps(rec))
        old = make().load_calibration(f)
        assert old.scale == "raw" and old.ranks is None and old.fingerprint() == old_fp
        for a, b in zip(raw.decide([t for t, _ in test]), old.decide([t for t, _ in test])):
            assert (a.value, a.escalate, a.conf, a.extra.get("threshold")) == (b.value, b.escalate, b.conf,
                                                                               b.extra.get("threshold"))
        ranked = make()                                        # a rank-scale file keeps the ranks and decides the same
        ranked.act_guard(cal, max_risk=0.10, scale="rank")
        f2 = ranked.save_calibration(tmp_path / "rank.json")
        rec2 = json.loads(f2.read_text())
        assert rec2["scale"] == "rank" and [len(r) for r in rec2["ranks"]] == [300, 300]
        back = make().load_calibration(f2)
        assert back.fingerprint() == ranked.fingerprint() and back.fingerprint() != old_fp
        for a, b in zip(ranked.decide([t for t, _ in test]), back.decide([t for t, _ in test])):
            assert (a.value, a.escalate) == (b.value, b.escalate)
        del rec2["ranks"]
        f2.write_text(json.dumps(rec2))
        with pytest.raises(ValueError, match="ranks"):
            make().load_calibration(f2)


def test_act_guard_records_the_promise_and_reports_the_cost():
    small, large = _model("small", 3.0), _model("large", 1.0)
    c = Cascade([small.decision("team", "Which team?", "email", TEAMS),
                 large.decision("team", "Which team?", "email", TEAMS)], costs=[45, 137])
    fp = c.fingerprint()
    info = c.act_guard(_labelled(), max_risk=0.10)
    assert info["risk"] <= 0.10 and 0 < info["answered"] <= 1 and info["n"] == 240
    assert 1 <= info["calls_per_question"] <= 2 and 45 <= info["cost"] <= 182 and sum(info["answered_by"]) == pytest.approx(info["answered"])
    assert c.fingerprint() != fp and c.threshold == info["threshold"]
    d = c(email=HARD)
    assert d.extra["guarantee"]["method"] == "crc" and d.extra["threshold"] == info["threshold"]
    assert all("shared threshold" in s["escalate"] for s in d.extra["stages"] if s["escalate"])
    _, sys_ = _system(c)
    assert "guarantee   P(answered alone and wrong) ≤ 0.1" in str(sys_.ask({"email": HARD}).audit("route"))
    with pytest.raises(ValueError):
        c.act_guard([], max_risk=0.1)


def test_conformal_sets_for_a_vote():
    small, large = _model("small", 2.0), _model("large", 2.0, bias={"technical": 0.5})
    v = Vote([small.decision("team", "Which team?", "email", TEAMS), large.decision("team", "Which team?", "email", TEAMS)])
    v.act_guard(_labelled(), max_risk=0.10)
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


def test_example_20_vote_of_two_families_answers_more_than_either_alone():
    import io
    import re
    import runpy
    from contextlib import redirect_stdout
    from pathlib import Path
    out = io.StringIO()
    with redirect_stdout(out):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "20_vote_across_families.py"),
                       run_name="__main__")
    text = out.getvalue()
    got = {m[0]: (int(m[1]), float(m[2])) for m in re.findall(r"  (family A alone|family B alone|vote A \+ B) .*?"
                                                               r"answered alone +(\d+)%.*?risk +([\d.]+)%", text)}
    assert set(got) == {"family A alone", "family B alone", "vote A + B"}
    assert got["vote A + B"][0] > max(got["family A alone"][0], got["family B alone"][0])
    assert all(r <= 10.0 for _, r in got.values())
    assert "the models disagree" in text and "hard check not_locked is false" in text
    assert "replay: True" in text and "replay: False" not in text


def _group_stream(rng, tag, n, S, L):
    """As _stream, in two groups: in "hard" both models are right far less often (0.3–0.8), in "easy" more (0.8–1.0)."""
    from solvi.multi import Facts
    out = []
    for i in range(n):
        grp = "hard" if rng.uniform() < 0.25 else "easy"
        t, y = f"{tag} {i}", ("a", "b")[rng.integers(2)]
        for tab in (S, L):
            c = rng.uniform(0.5, 1.0)
            lo, gain = (0.3, 0.5) if grp == "hard" else (0.8, 0.2)
            ok = rng.uniform() < lo + gain * (c - 0.5) * 2
            pred = y if ok else ("b" if y == "a" else "a")
            tab.table[t] = np.log(np.array([c, 1 - c]) if pred == "a" else np.array([1 - c, c]))
        out.append((Facts(doc=t, domain=grp), y))
    return out


@pytest.mark.parametrize("make", [lambda s, l_: Cascade([s, l_]), lambda s, l_: Vote([s, l_], rule="all")])
def test_act_guard_per_group_on_a_combination_holds_inside_every_group(make):
    rng = np.random.default_rng(1)
    S, L = Table("S"), Table("L")
    ms, ml = (DecideModel(x, meta={"format": "test", "temperature": 1.0}) for x in (S, L))
    comb = make(ms.decision("q", "Q?", "doc", ["a", "b"]), ml.decision("q", "Q?", "doc", ["a", "b"]))
    cal, test = _group_stream(rng, "c", 1600, S, L), _group_stream(rng, "t", 6000, S, L)

    def risk_in(group):
        ex = [(x, y) for x, y in test if x["domain"] == group]
        return float(np.mean([d.escalate is None and d.value != y for d, (_, y) in zip(comb.decide([x for x, _ in ex]), ex)]))
    comb.act_guard(cal, max_risk=0.10)
    plain_hard = risk_in("hard")
    fp = comb.fingerprint()
    info = comb.act_guard(cal, max_risk=0.10, groups="domain", min_group=100)
    assert comb.fingerprint() != fp and comb.facts == ["doc", "domain"]
    assert set(info["groups"]) == {("easy",), ("hard",), ()} and info["groups"][()]["n"] == 0
    assert info["groups"][("hard",)]["threshold"] > info["groups"][("easy",)]["threshold"]
    assert all(v["risk"] <= 0.10 for v in info["groups"].values())
    assert plain_hard > 0.15                                        # one threshold for the stream: over the risk in "hard"
    assert risk_in("hard") <= 0.10 and risk_in("easy") <= 0.10
    d = comb.decide(test[0][0])
    g = d.extra["guarantee"]
    assert g["group"] == [test[0][0]["domain"]] and g["applied"] == g["group"] and d.extra["threshold"] == g["threshold"]
    d = comb.decide("t 3")
    assert d.escalate and "group unknown" in d.escalate
    cat = Catalog()                                                  # in a catalog: the group fact is an input
    cat.fn(comb)

    @cat.rule("answer")
    def answer(q):
        return q
    sys_ = System(cat, [Question("answer", "A?", Answer.choice(["a", "b"]))])
    res = sys_.ask({"doc": "t 5", "domain": test[5][0]["domain"]})
    rec = next(r for r in res.trace.records if r.name == "q")
    assert rec.extra["guarantee"]["group"] == [test[5][0]["domain"]] and res.trace.replay(cat)["ok"]
    assert "within every group at once" in str(res.audit("answer"))


def test_a_cost_for_a_nested_combination_is_refused_and_costs_of_one_are_still_reported():
    """costs=[2, 10] with a Vote as the second member was dropped silently (leaf costs [2, 1, 1]); costs=[1, 10] then
    reported no cost at all."""
    small, mid, large = _model("small", 3.0), _model("mid", 2.0), _model("large", 1.0)
    p = [m.decision("team", "Which team?", "email", TEAMS) for m in (small, mid, large)]
    with pytest.raises(ValueError, match=r"costs: \['team'\] is itself a combination"):
        Cascade([p[0], Vote([p[1], p[2]])], costs=[2, 10])
    c = Cascade([p[0], Vote([p[1], p[2]], costs=[5, 20])])
    assert [lf.cost for lf in c.leaves()] == [1.0, 5.0, 20.0]
    assert "cost" in c.act_guard(_labelled(), max_risk=0.10)
    ones = Cascade([p[0], p[2]], costs=[1, 1])                 # given, even if 1: reported
    assert ones.act_guard(_labelled(), max_risk=0.10)["cost"] >= 1


def test_parts_and_combinations_speak_one_decider_protocol():
    """act_guard(examples, risk, signal, ...) on a part and (examples, risk, groups, ...) on a combination: the third
    positional argument differed, the result keys differed and guarantee["signal"] was a name for a part and a sentence
    for a combination."""
    import inspect

    from solvi.decide import DecisionPart
    from solvi.multi import Combination
    shared = ["examples", "max_risk", "signal", "groups", "min_group", "delta"]
    for cls in (DecisionPart, Combination):
        ps = list(inspect.signature(cls.act_guard).parameters.values())[1:]
        assert [p.name for p in ps][:6] == shared and all(p.kind is p.KEYWORD_ONLY for p in ps[1:])
    rng = np.random.default_rng(5)
    S, L, s, l_ = _scales_parts()
    cal = _scales_stream(rng, "c", 300, S, L)
    keys = {"signal", "threshold", "answered", "error", "risk", "n", "guarantee", "promise", "base_error",
            "must_escalate_at_least"}
    one, both = s.act_guard(cal, max_risk=0.10), Vote([s, l_]).act_guard(cal, max_risk=0.10)
    assert keys <= set(one) and keys <= set(both) and "calls_per_question" in both
    assert one["signal"] in ("act", "confidence") and both["signal"] == "shared"
    with pytest.raises(ValueError, match="a combination's signal is each part's own"):
        Vote([s, l_]).act_guard(cal, max_risk=0.10, signal="act")
    with pytest.warns(DeprecationWarning, match=r"act_guard\(risk=\) is deprecated: use max_risk="):
        Vote([s, l_]).act_guard(cal, risk=0.10)
