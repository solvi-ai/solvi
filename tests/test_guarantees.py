"""Escalation thresholds with a guarantee (conformal risk control, learn-then-test) and conformal answer sets."""
import numpy as np
import pytest

from solvi.calibration import conformal_quantile, crc_threshold, ltt_threshold, set_scores
from test_decide import TEAMS, model, texts


def _stream(rng, n, lo=0.3, gain=0.6):
    """A decider whose confidence is informative but over-confident: P(right) = lo + gain·conf."""
    conf = rng.uniform(0.3, 1.0, n)
    wrong = rng.uniform(size=n) > lo + gain * conf
    return conf, wrong.astype(float)


def test_crc_keeps_the_risk_of_answering_wrongly_alone_on_new_inputs():
    rng = np.random.default_rng(0)
    risks = []
    for _ in range(300):
        c, w = _stream(rng, 300)
        t = crc_threshold(c, w, 0.10)
        c2, w2 = _stream(rng, 2000)
        risks.append(float(((c2 >= t) * w2).mean()))
    assert np.mean(risks) <= 0.10 + 0.005
    assert np.mean(risks) >= 0.07                        # not trivially conservative: it does answer alone


def test_crc_escalates_everything_when_the_risk_cannot_be_met():
    assert crc_threshold([0.9, 0.8, 0.7], [1, 1, 1], 0.10) == float("inf")
    assert crc_threshold([0.9] * 5, [0] * 5, 0.10) == float("inf")      # 5 examples cannot certify 10%: (0 + 1) / 6 > 0.1


def test_ltt_bounds_the_error_among_the_answered_and_is_stricter():
    rng = np.random.default_rng(1)
    c, w = _stream(rng, 3000, 0.2, 0.8)
    t = ltt_threshold(c, w, error=0.10)
    c2, w2 = _stream(rng, 20000, 0.2, 0.8)
    assert t < float("inf") and w2[c2 >= t].mean() <= 0.10
    assert ltt_threshold(c[:50], w[:50], error=0.10) == float("inf")


def test_ordinal_sets_are_intervals_and_the_quantile_needs_enough_examples():
    p = [0.05, 0.1, 0.4, 0.3, 0.15]
    s = set_scores(p, ordinal=True)
    for q in (0.0, 0.3, 0.5, 0.75, 0.9):
        keep = [i for i in range(5) if s[i] <= q]
        assert keep == list(range(min(keep), max(keep) + 1))
    assert conformal_quantile([0.1] * 5, 0.10) == float("inf")
    assert conformal_quantile(list(np.linspace(0, 1, 99)), 0.10) == pytest.approx(0.9, abs=0.02)


def _labelled(n0=0):
    ex = []
    for team in TEAMS:
        ex += [(t, team) for t in texts(team, 60, n0)]
    ex += [(f"The app crashed after the refund of order {i}.", "billing" if i % 2 else "technical")
           for i in range(n0, n0 + 60)]
    return ex


def test_act_guard_sets_a_threshold_and_every_decision_records_its_promise():
    m = model(noise=2.0)
    part = m.decision("team", "Which team?", "email", TEAMS)
    fp = part.fingerprint()
    info = part.act_guard(_labelled(), risk=0.10)
    assert info["signal"] == "confidence" and part.escalate_below == info["threshold"]
    assert info["risk"] <= 0.10 and 0 < info["answered"] < 1 and info["n"] == 240
    assert "P(answered alone and wrong) ≤ 0.1" in info["guarantee"]
    assert part.fingerprint() != fp
    d = part(email="The app crashed after the refund of order 7.")
    assert d.extra["guarantee"]["method"] == "crc" and d.extra["guarantee"]["risk"] == 0.10


def test_calibrate_for_ltt_lets_nothing_through_on_few_examples():
    part = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    info = part.calibrate_for(_labelled()[:40], error=0.02, method="ltt")
    assert info["method"] == "ltt" and info["threshold"] == float("inf") and info["coverage"] == 0
    assert part(email="I was charged twice, please refund order 3.").escalate
    with pytest.raises(ValueError):
        part.calibrate_for(_labelled(), method="magic")


def test_conformal_sets_cover_the_right_answer_and_reach_the_escalation_message():
    part = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    info = part.conformal(_labelled(), coverage=0.90)
    assert info["n"] == 240 and 1 <= info["mean_size"] <= 3
    test = _labelled(1000)
    hits = [y in part(email=t).extra["candidates"] for t, y in test]
    assert np.mean(hits) >= 0.85
    part.escalate_below = 0.99
    d = part(email="The app crashed after the refund of order 5.")
    assert d.escalate and "candidates at 90%" in d.escalate
    assert set(d.extra["candidates"]) <= set(TEAMS)


class PositionBiased:
    """Keyword logits plus a strong preference for whatever option is listed first."""
    model_id = "test/position-biased"

    def fingerprint(self):
        return "position-biased-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            z = [1.0 * ("refund" in low and o == "billing") + 1.0 * ("crash" in low and o == "technical")
                 + (1.5 if i == 0 else 0.0) for i, o in enumerate(it.options)]
            out.append(np.stack([np.array(z), np.array(z) - 1.0], 1))
        return out


def _biased():
    from solvi.decide import DecideModel
    return DecideModel(PositionBiased(), meta={"format": "test", "temperature": 1.0})


def test_option_order_canonical_and_average_remove_the_callers_order_from_the_answer():
    m = _biased()
    text = "A refund please."
    given = [m.decision("t", "Team?", "email", o, option_order="given").decide(text).value
             for o in (["billing", "technical"], ["technical", "billing"])]
    assert given == ["billing", "technical"]                       # the listed order decides: a position bias
    for mode in ("canonical", "average"):
        got = {m.decision("t", "Team?", "email", o, option_order=mode).decide(text).value
               for o in (["billing", "technical"], ["technical", "billing"])}
        assert len(got) == 1, mode
    avg = m.decision("t", "Team?", "email", ["technical", "billing"], option_order="average")
    assert avg.decide(text).probs["billing"] == pytest.approx(1 / (1 + np.exp(-1.0)), abs=1e-9)  # the bias cancels
    assert avg.fingerprint() != m.decision("t", "Team?", "email", ["technical", "billing"]).fingerprint()


def test_a_near_tie_escalates_with_min_margin():
    m = model()
    part = m.decision("team", "Which team?", "email", TEAMS, min_margin=0.2)
    d = part(email="The app crashed after the refund of order 7.")
    assert d.escalate and "margin" in d.escalate and d.extra["margin"] < 0.2
    assert not part(email="I was charged twice, please refund order 3.").escalate


def test_act_guard_reports_how_much_must_escalate_when_the_model_is_often_wrong():
    part = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    info = part.act_guard(_labelled(), risk=0.05)
    assert info["base_error"] > 0.05
    assert info["must_escalate_at_least"] == pytest.approx((info["base_error"] - 0.05) / 0.95)
    assert 1 - info["answered"] >= info["must_escalate_at_least"] - 1e-9


def test_the_audit_says_what_the_thresholds_behind_an_answer_promise():
    from solvi import Answer, Catalog, Question, System
    part = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    cat = Catalog()
    cat.fn(part)

    @cat.rule("route")
    def route(team):
        return team
    s = System(cat, [Question("route", "Route", Answer.choice(TEAMS))])
    email = {"email": "I was charged twice, please refund order 3."}
    assert "guarantee   none for some decisions" in str(s.ask(email).audit("route"))
    part.act_guard(_labelled(), risk=0.10)
    txt = str(s.ask(email).audit("route"))
    assert "guarantee   P(answered alone and wrong) ≤ 0.1" in txt and "(crc, n = 240)" in txt


def test_by_default_the_callers_order_neither_changes_the_answer_nor_how_it_is_shown():
    m = _biased()
    parts = [m.decision("t", "Team?", "email", o) for o in (["billing", "technical"], ["technical", "billing"])]
    ds = [p.decide("A refund please.") for p in parts]
    assert ds[0].value == ds[1].value and ds[0].probs["billing"] == pytest.approx(ds[1].probs["billing"])
    assert list(ds[1].probs) == ["technical", "billing"] and parts[1].options == ["technical", "billing"]


# --------------------------------------------------------------------------------------------------- thresholds per group
def _groups_stream(rng, n, share_hard=0.2):
    """Two groups of different difficulty: P(right) = 0.8–1.0 in "easy", 0.3–0.8 in "hard", rising with the confidence."""
    g = np.where(rng.uniform(size=n) < share_hard, "hard", "easy")
    c = rng.uniform(0.5, 1.0, n)
    p = np.where(g == "hard", 0.3 + 0.5 * (c - 0.5) * 2, 0.8 + 0.2 * (c - 0.5) * 2)
    return c, (rng.uniform(size=n) > p).astype(float), g


def test_plain_crc_breaks_the_risk_inside_a_hard_group_and_thresholds_per_group_keep_it():
    from solvi.calibration import group_thresholds, node_of
    rng = np.random.default_rng(0)
    risk = {"plain": {"easy": [], "hard": []}, "groups": {"easy": [], "hard": []}}
    violated = {"plain": 0, "groups": 0}
    for _ in range(100):
        c, w, g = _groups_stream(rng, 1000)
        t = crc_threshold(c, w, 0.10)
        th = group_thresholds(c, w, list(g), risk=0.10, min_group=100, delta=0.10)
        c2, w2, g2 = _groups_stream(rng, 20000)
        for key, thr in (("plain", {"easy": t, "hard": t}), ("groups", {x: th[node_of(x, th)] for x in ("easy", "hard")})):
            auto = c2 >= np.where(g2 == "hard", thr["hard"], thr["easy"])
            r = {x: float((auto * w2)[g2 == x].mean()) for x in ("easy", "hard")}
            for x in r:
                risk[key][x].append(r[x])
            violated[key] += max(r.values()) > 0.10
    assert np.mean(risk["plain"]["hard"]) > 0.25                   # the promise over the stream hides a hard group
    assert np.mean(risk["plain"]["easy"]) + 0.2 * np.mean(risk["plain"]["hard"]) <= 0.10 / 0.8 + 0.01
    assert violated["plain"] == 100
    assert max(np.mean(risk["groups"][x]) for x in ("easy", "hard")) <= 0.10
    assert violated["groups"] <= 10                               # every group at once, with probability ≥ 90%


def test_small_groups_are_pooled_with_their_parent_and_the_nodes_depend_on_sizes_only():
    from solvi.calibration import group_nodes, loss_budget, node_of
    paths = [("a", "x")] * 120 + [("a", "y")] * 30 + [("a", "z")] * 90 + [("b", "u")] * 50
    nodes, owner = group_nodes(paths, min_group=100)
    assert set(nodes) == {("a", "x"), ("a",), ()}                # a/y + a/z (120) pool into a; b/u (50) into the rest
    assert len(nodes[("a",)]) == 120 and len(nodes[()]) == 50 and owner[130] == ("a",)
    assert node_of(("a", "new"), nodes) == ("a",) and node_of("c", nodes) == () and node_of(("a", "x"), nodes) == ("a", "x")
    assert loss_budget(300, 0.1) == 29 and loss_budget(20, 0.1, 0.1) == -1       # 20 examples cannot certify 10% at 90%
    assert 15 < loss_budget(300, 0.1, 0.1) < 29                                 # the binomial bound is stricter than CRC


class Table:
    """A scorer reading prepared logits by input text (a synthetic stream with known right answers)."""
    model_id = "test/table"

    def __init__(self):
        self.table = {}

    def fingerprint(self):
        return "table"

    def logits(self, items):
        return [np.stack([self.table[it.text], self.table[it.text] - 1.0], 1) for it in items]


def _table_stream(rng, tab, tag, n):
    from solvi.multi import Facts
    c, w, g = _groups_stream(rng, n)
    out = []
    for i in range(n):
        t, y = f"{tag} {i}", ("a", "b")[rng.integers(2)]
        pred = y if not w[i] else ("b" if y == "a" else "a")
        tab.table[t] = np.log(np.array([c[i], 1 - c[i]]) if pred == "a" else np.array([1 - c[i], c[i]]))
        out.append((Facts(doc=t, domain=str(g[i]), task="refunds" if i % 2 else "invoices"), y))
    return out


def test_act_guard_per_group_on_a_decision_part_records_the_group_and_holds_inside_it():
    from solvi import Answer, Catalog, Question, System
    from solvi.decide import DecideModel
    rng = np.random.default_rng(3)
    tab = Table()
    m = DecideModel(tab, meta={"format": "test", "temperature": 1.0})
    part = m.decision("q", "Q?", "doc", ["a", "b"])
    cal, test = _table_stream(rng, tab, "cal", 1500), _table_stream(rng, tab, "test", 6000)
    plain = part.act_guard(cal, risk=0.10)
    fp = part.fingerprint()

    def risk_in(group):
        ds = part.decide([x for x, _ in test if x["domain"] == group])
        ys = [y for x, y in test if x["domain"] == group]
        return float(np.mean([d.escalate is None and d.value != y for d, y in zip(ds, ys)]))
    assert risk_in("hard") > 0.2 and plain["risk"] <= 0.10
    info = part.act_guard(cal, risk=0.10, groups=["domain", "task"], min_group=200)
    assert part.fingerprint() != fp and part.groups is not None
    assert risk_in("hard") <= 0.10 and risk_in("easy") <= 0.10
    g = info["groups"]
    assert ("easy", "refunds") in g and g[("easy", "refunds")]["n"] >= 200
    hard = [k for k in g if k[:1] == ("hard",)]
    assert hard and all(g[k]["threshold"] > g[("easy", "refunds")]["threshold"] for k in hard)
    d = part.decide(test[0][0])
    rec = d.extra["guarantee"]
    assert rec["method"] == "group-bound" and rec["group"] == [test[0][0]["domain"], test[0][0]["task"]]
    assert rec["applied"] == rec["group"][:len(rec["applied"])] and "within every group at once" in rec["promise"]
    assert "here: group" in rec["promise"]
    d = part.decide("cal 1")                                       # no group given: no threshold holds for it
    assert d.escalate.startswith("group unknown")
    # in a catalog, the group facts join the part's inputs and the audit prints the group's promise
    cat = Catalog()
    cat.fn(part)

    @cat.rule("answer")
    def answer(q):
        return q
    s = System(cat, [Question("answer", "A?", Answer.choice(["a", "b"]))])
    x = dict(test[1][0])
    res = s.ask(x)
    assert "within every group at once" in str(res.audit("answer")) and "here: group" in str(res.audit("answer"))
    with pytest.raises(ValueError, match="group"):
        part.act_guard([("cal 1", "a")] * 5, groups="domain")
    part.act_guard(cal, risk=0.10)                                  # without groups again: one threshold, inputs as before
    assert part.groups is None and list(part.__signature__.parameters) == ["doc"]


# --------------------------------------------------------------------------------------------------- fixes before 0.7
def test_conformal_takes_an_iterator_of_examples():
    ex = _labelled()
    a = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    b = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    ia = a.conformal(ex, coverage=0.90)
    ib = b.conformal(zip([t for t, _ in ex], [y for _, y in ex]), coverage=0.90)   # a one-pass iterator
    assert ib["n"] == ia["n"] == 240 and ib["quantile"] == ia["quantile"] < float("inf")


def test_fit_teach_adapt_with_average_order_use_the_averaged_logits():
    m = _biased()
    part = m.decision("t", "Team?", "email", ["technical", "billing"], option_order="average")
    ex = [("A refund please.", "billing"), ("It crashed.", "technical")] * 4
    a = part.fit(ex)
    for (z, _), (t, _) in zip(a.examples, ex):
        assert z == pytest.approx(list(part._raw([t])[0][0]))      # not the single-order logits (position bias 1.5)
    part.teach("A refund, now.", "billing")
    assert a.examples[-1][0] == pytest.approx(list(part._raw(["A refund, now."])[0][0]))
    b = part.adapt(["A refund please.", "It crashed.", "Hello."])
    mean = np.mean([part._raw([t])[0][0] for t in ["A refund please.", "It crashed.", "Hello."]], 0)
    assert b.bias == pytest.approx(list(mean - mean.mean()))
    assert abs(b.bias[0] - b.bias[1]) < 1.0                       # the position bias was averaged away, not learned


class KeywordAct:
    """Keyword logits (test_decide.KW) and an act logit that grows with the number of keywords."""
    model_id = "test/keyword-act"

    def fingerprint(self):
        return "keyword-act-1"

    def logits(self, items):
        from test_decide import KW
        out = []
        for it in items:
            z = np.array([2.0 * sum(it.text.lower().count(k) for k in KW.get(o, [])) for o in it.options])
            out.append({"logits": z, "act": float(z.max()) - 1.0, "unknown": -4.0})
        return out


def test_switching_the_signal_clears_the_other_threshold():
    from solvi.decide import DecideModel
    from test_primitives import L14G
    part = DecideModel(KeywordAct(), L14G).decision("team", "Which team?", "email", TEAMS, act_threshold=0.99)
    ex = [(t, team) for team in TEAMS for t in texts(team, 100)]
    part.calibrate_for(ex, error=0.05, signal="confidence")
    assert part.act_threshold is None and part.guarantee["cleared"] == {"act_threshold": 0.99}
    assert part.escalate_below is not None
    info = part.act_guard(ex, risk=0.10, signal="act")
    assert part.escalate_below is None and part.act_threshold == info["threshold"]
    assert "escalate_below" in part.guarantee["cleared"]
    assert not part(email=texts("billing", 1, 500)[0]).escalate      # the stale confidence threshold no longer applies


@pytest.mark.parametrize("error", [0, 0.0, 1, 1.5, -0.1, "x"])
def test_ltt_and_calibrate_for_refuse_an_error_outside_0_1(error):
    with pytest.raises(ValueError, match="error must be a number strictly between 0 and 1"):
        ltt_threshold([0.9, 0.8], [0, 1], error=error)
    part = model().decision("team", "Which team?", "email", TEAMS)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        part.calibrate_for(_labelled()[:20], error=error, method="ltt")
    if error not in (0, 0.0):                                  # empirical: 0 is a target (no error on the examples)
        with pytest.raises(ValueError, match=r"in \[0, 1\)"):
            part.calibrate_for(_labelled()[:20], error=error, method="empirical")
    with pytest.raises(ValueError, match="delta"):
        ltt_threshold([0.9, 0.8], [0, 1], error=0.1, delta=0)


def test_ltt_default_grid_follows_the_scores_so_confidences_near_1_can_pass():
    from solvi.calibration import ltt_grid
    rng = np.random.default_rng(2)
    v = rng.uniform(size=3000)
    c, w = 1 - 10 ** -(3 + 4 * v), (rng.uniform(size=3000) > 0.6 + 0.39 * v).astype(float)     # an LLM: all ≥ 0.999
    assert ltt_threshold(c, w, error=0.10, grid=np.linspace(0.2, 0.995, 32)) == float("inf")   # the grid before 0.7
    t = ltt_threshold(c, w, error=0.10)
    assert 0.999 < t < 1 and w[c >= t].mean() <= 0.10
    g = ltt_grid(c)
    assert len(g) == 64 and g[0] == c.min() and g[-1] == c.max()        # label-free: quantiles of the scores
    assert list(ltt_grid([0.3, 0.3, 0.9, float("nan")])) == [0.3, 0.9]  # few distinct scores: all of them
    assert ltt_threshold([], [], error=0.1) == float("inf")
