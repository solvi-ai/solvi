"""State the decider must not lose or mix: the logits cache of a question with and without "not stated", the adaptation
of an evidence question across save / load, and a guarantee calibrated on a signal the part does not enforce."""
import math

import numpy as np
import pytest

from solvi import Unknown
from solvi.decide import DecideModel

V2 = {"format": "solvi_decide v2", "subformat": "l14g typed v2"}


class Words:
    """Says "not stated" strongly — when the question allows it, as a model asked in words does (solvi.llm, systemone)."""
    tag = "words"

    def __init__(self):
        self.calls = []

    def fingerprint(self):
        return "w"

    def logits(self, items):
        self.calls += [it.unknown for it in items]
        return [({"logits": np.array([0.2, 0.0]), "unknown": 5.0} if it.unknown else {"logits": np.array([0.2, 0.0])}) for it in items]


def test_a_maybe_question_and_a_plain_one_do_not_share_a_cached_reply():
    """The cache key left "not stated" out: the part asked second got the other question's reply and no call was made."""
    seen = []
    for first in ("maybe", "plain"):
        sc = Words()
        m = DecideModel(sc, {**V2, "act": False})
        plain = m.decision("p", "Is it signed?", "doc", ["yes", "no"])
        maybe = m.decision("q", "Is it signed?", "doc", ["yes", "no"], unknown=True)
        order = (maybe, plain) if first == "maybe" else (plain, maybe)
        out = {p.__name__: (p(doc="some text").value, round(p(doc="some text").conf, 6)) for p in order}
        assert out["q"][0] is Unknown and out["p"][0] != Unknown
        assert sorted(sc.calls) == [False, True]        # each was put to the scorer once, then cached under its own key
        seen.append(out)
    assert seen[0] == seen[1]                           # the answers do not depend on which part asked first


class Act:
    """Text "L|A": the logit of option a and the act logit."""
    tag = "act"

    def fingerprint(self):
        return "a"

    def logits(self, items):
        out = []
        for it in items:
            lg, a = (float(x) for x in it.text.split("|"))
            out.append({"logits": np.array([lg] + [0.0] * (len(it.options) - 1)), "act": a, "unknown": -4.0})
        return out


def stream(n, seed=3):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.5, 0.99, n)
    right = rng.uniform(size=n) < 0.2 + 0.75 * conf
    act = np.where(right, rng.normal(1.0, 1.0, n), rng.normal(-1.0, 1.0, n))
    return [(f"{math.log(c / (1 - c)):.5f}|{a:.5f}", "a" if r else "b") for c, r, a in zip(conf, right, act)]


def test_a_guarantee_on_the_act_signal_needs_a_part_that_uses_it():
    """act_guard(signal="act") on a part made with use_act=False recorded a promise nothing enforced (22.8% wrong alone
    against ≤ 5%)."""
    m, cal = DecideModel(Act(), V2), stream(400)
    off = m.decision("q", "Which?", "x", ["a", "b"], use_act=False)
    with pytest.raises(ValueError, match="use_act=False"):
        off.act_guard(cal, risk=0.05, signal="act")
    assert off.guarantee is None and off.act_threshold is None
    info = off.act_guard(cal, risk=0.2, signal="auto")          # auto takes the signal the part enforces
    assert info["signal"] == "confidence"
    on = m.decision("q2", "Which?", "x", ["a", "b"])
    assert on.act_guard(cal, risk=0.2, signal="act")["signal"] == "act"


def test_the_adaptation_of_an_evidence_question_survives_save_and_load(tmp_path):
    """Its key has a fifth element; the file kept four, so after a reload the fit sat on the plain question with the same
    task and options."""
    class Lean(Act):
        def logits(self, items):
            return [{"logits": np.array([0.1 * len(it.text) % 1.0] + [0.0] * (len(it.options) - 1)), "act": 1.0, "unknown": -4.0}
                    for it in items]

    opts = ["billing", "shipping", "other"]
    ex = [(f"ticket {i} about {'billing' if i % 2 else 'shipping'}", "billing" if i % 2 else "shipping") for i in range(24)]
    m = DecideModel(Lean(), V2)
    ev = m.decision("team", "Which team?", "email", opts, evidence=True)
    plain = m.decision("team_plain", "Which team?", "email", opts)
    ev.fit(ex)
    assert ev.adaptation is not None and plain.adaptation is None
    assert len(ev.spec.key) > 4                          # the case the file used to lose
    f = tmp_path / "ad.json"
    m.save_adaptations(f)
    m2 = DecideModel(Lean(), V2).load_adaptations(f)
    ev2 = m2.decision("team", "Which team?", "email", opts, evidence=True)
    plain2 = m2.decision("team_plain", "Which team?", "email", opts)
    assert ev2.adaptation is not None and plain2.adaptation is None
    assert ev2.fingerprint() == ev.fingerprint() and plain2.fingerprint() == plain.fingerprint()
