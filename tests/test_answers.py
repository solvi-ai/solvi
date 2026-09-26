"""Answers: a hard check beats the learned head, abstain on errors or missing inputs, yes/no normalization, head training."""
import random

from solvi import Answer, Catalog, Question, System
from examples_loader import load

R = load("04_refunds")


def _refund_system(n=200):
    rng = random.Random(0)
    s = System(R.cat, R.QUESTIONS)
    s.fit("refund", [(a, t["refund"]) for a, t in (R.make(rng, i) for i in range(n))])
    return s


def test_fit_head_learns_policy_facts():
    s = _refund_system()
    head = s.heads["refund"]
    assert {"reason", "within_30_days"} <= set(head.features)
    rng = random.Random(1)
    test = [R.make(rng, 1000 + i) for i in range(200)]
    acc = sum(s.ask(a)["refund"].answer == t["refund"] for a, t in test) / len(test)
    assert acc >= 0.9


def test_hard_check_overrides_head():
    s = _refund_system()
    rng = random.Random(2)
    old = [(a, t) for a, t in (R.make(rng, 2000 + i) for i in range(300)) if not t["within"]]
    assert old
    for a, _ in old:
        r = s.ask(a)["refund"]
        assert r.answer == "no" and r.status == "forced"


def test_abstain_on_function_error_and_missing_input():
    cat = Catalog()

    @cat.fn
    def ratio(a, b):
        return a / b

    @cat.rule("big")
    def big_rule(ratio):
        return ratio > 1

    s = System(cat, [Question("big", "?", Answer.yes_no())])
    r = s.ask({"a": 1.0, "b": 0.0})["big"]
    assert r.status == "abstain" and r.answer is None
    r2 = s.ask({"a": 1.0})["big"]
    assert r2.status == "abstain"
    assert s.ask({"a": 3.0, "b": 1.0})["big"].answer == "yes"


def test_rule_answer_out_of_options_abstains():
    cat = Catalog()

    @cat.rule("pick")
    def pick_rule(x):
        return x

    s = System(cat, [Question("pick", "?", Answer.choice(["a", "b"]))])
    assert s.ask({"x": "a"})["pick"].answer == "a"
    assert s.ask({"x": "zzz"})["pick"].status == "abstain"


def test_unfitted_question_without_rule_abstains():
    cat = Catalog()

    @cat.fn
    def y(x):
        return x + 1

    s = System(cat, [Question("q", "?", Answer.yes_no())])
    assert s.ask({"x": 1})["q"].status == "abstain"


def test_head_abstains_when_its_features_are_missing():
    rng = random.Random(1)
    s = System(R.cat, R.QUESTIONS)
    s.fit("refund", [(init, truth["refund"]) for init, truth in (R.make(rng, i) for i in range(200))])
    r = s.ask({"doc": "hello", "today": R.TODAY})["refund"]      # no order date → nothing to compute the head's features from
    assert r.status in ("abstain", "forced")
