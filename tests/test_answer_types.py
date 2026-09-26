"""Answer types: ordinal (median of a learned distribution), multi (subset; rules and learned per-option heads), descriptions."""
import random

import pytest

from solvi import Answer, Catalog, Question, System


def test_multi_rule_normalizes_to_option_order():
    cat = Catalog()

    @cat.rule("tags")
    def tags(text):
        return {t for t in ("urgent", "billing", "legal") if t in text}

    s = System(cat, [Question("tags", "", Answer.multi(["legal", "billing", "urgent"]))])
    assert s.ask({"text": "urgent billing issue"})["tags"].answer == ("billing", "urgent")
    assert s.ask({"text": "hello"})["tags"].answer == ()
    with pytest.raises(ValueError):
        Answer.multi(["a"]).normalize(["b"])


def test_multi_learned_with_fit_fast_and_teach():
    cat = Catalog()

    @cat.fn
    def amount(x):
        return x

    @cat.fn
    def foreign(country):
        return country != "DE"

    q = Question("flags", "", Answer.multi(["big", "foreign"]))
    rng = random.Random(0)

    def ex():
        x, c = rng.uniform(0, 100), rng.choice(["DE", "FR", "US"])
        return {"x": x, "country": c}, [f for f, on in (("big", x > 60), ("foreign", c != "DE")) if on]
    s = System(cat, [q])
    s.fit_fast("flags", [ex() for _ in range(200)])
    test = [ex() for _ in range(200)]
    acc = sum(s.ask(st)["flags"].answer == Answer.multi(["big", "foreign"]).normalize(y) for st, y in test) / len(test)
    assert acc > 0.8
    assert s.teach("flags", *ex()) is not None


def test_ordinal_answers_with_the_median():
    from solvi.system import System as S
    at = Answer.ordinal({"low": "no action", "medium": "watch", "high": "act now"})
    assert at.options == ["low", "medium", "high"] and at.descriptions["high"] == "act now"
    assert at.rank("medium") == 1

    class FakeHead:                          # bimodal distribution: argmax would say "high", the median is "medium"
        features = []

        def predict(self, row):
            return {"low": 0.40, "medium": 0.15, "high": 0.45}

        def contributions(self, row):
            return {}
    cat = Catalog()
    s = S(cat, [Question("risk", "", at)])
    s.heads["risk"] = FakeHead()
    assert s.ask({"x": 1})["risk"].answer == "medium"


class _Head:
    features = []

    def __init__(self, p):
        self.p = p

    def predict(self, row):
        return self.p

    def contributions(self, row):
        return {}


def _guard(p_verdict, p_harm, fixed_verdict=None):
    cat = Catalog()

    @cat.constraint
    def unsafe_if_harm(verdict, harm):
        return harm == "none" or verdict == "unsafe"

    if fixed_verdict:
        @cat.rule("verdict")
        def verdict_rule(x):
            return fixed_verdict
    s = System(cat, [Question("verdict", "", Answer.choice(["safe", "unsafe"])),
                     Question("harm", "", Answer.choice(["none", "prompt_injection", "pii"]))])
    if not fixed_verdict:
        s.heads["verdict"] = _Head(p_verdict)
    s.heads["harm"] = _Head(p_harm)
    return s.ask({"x": 1})


def test_joint_decoding_fixes_contradictory_learned_answers():
    # independently: harm = prompt_injection (0.82), verdict = safe (0.52) — a contradiction
    r = _guard({"safe": 0.52, "unsafe": 0.48}, {"none": 0.10, "prompt_injection": 0.82, "pii": 0.08})
    assert (r["verdict"].answer, r["harm"].answer) == ("unsafe", "prompt_injection")
    assert r.feasible and "unsafe_if_harm" in r["verdict"].why


def test_rule_answers_stay_fixed_and_learned_ones_adapt_or_report():
    r = _guard(None, {"none": 0.30, "prompt_injection": 0.60, "pii": 0.10}, fixed_verdict="safe")
    assert r["verdict"].answer == "safe" and r["harm"].answer == "none"      # the rule wins, the learned answer adapts
    assert r.feasible


def test_consistent_answers_untouched():
    r = _guard({"safe": 0.9, "unsafe": 0.1}, {"none": 0.9, "prompt_injection": 0.05, "pii": 0.05})
    assert (r["verdict"].answer, r["harm"].answer) == ("safe", "none") and r.feasible and not r.violations
