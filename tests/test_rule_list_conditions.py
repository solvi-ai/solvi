"""Readable finite conjunction/exception and ngram capability, no corpus claims."""

import pytest
from solvi.core.deciders.rulelist import RuleList, literals


def test_xor_requires_two_literals_and_negation():
    rows = []
    answers = []
    for a, b in [(False, False), (True, False), (False, True), (True, True)]:
        rows += [{"text": " ".join(x for x, on in [("A", a), ("B", b)] if on)}] * 8
        answers += ["yes" if a != b else "no"] * 8
    rules = RuleList(["text"], min_precision=0.8, max_literals=2, negations=True).fit(rows, answers)
    assert all(rules.predict(row)[0] == answer for row, answer in zip(rows, answers))
    assert any(r["unless"] for r in rules.rules)
    assert any(len(r["all"]) + len(r["unless"]) == 2 for r in rules.rules)
    assert all(len(r["all"]) + len(r["unless"]) <= 2 for r in rules.rules)


def test_specific_fee_exception_does_not_take_generic_transaction_answer():
    rows = [{"text": "international transactions fee"}] * 8 + [{"text": "international transactions"}] * 8
    ys = ["fee"] * 8 + ["transactions"] * 8
    rules = RuleList(["text"], max_literals=3, negations=True).fit(rows, ys)
    assert rules.predict({"text": "fee for international transactions"})[0] == "fee"
    assert rules.predict({"text": "international transactions"})[0] == "transactions"


def test_ngram_keeps_order_and_default_literals_unchanged():
    assert "text has 'MEANING OF'" not in literals({"text": "meaning of life"}, ["text"])
    assert "text has 'MEANING OF'" in literals({"text": "meaning of life"}, ["text"], ngrams=2)
    rows = [{"text": "meaning of"}] * 8 + [{"text": "of meaning"}] * 8
    rules = RuleList(["text"], ngrams=2).fit(rows, ["forward"] * 8 + ["reverse"] * 8)
    assert rules.predict({"text": "meaning of"})[0] == "forward"
    assert rules.predict({"text": "of meaning"})[0] == "reverse"


@pytest.mark.parametrize("kw", [{"max_literals": True}, {"max_literals": 4}, {"ngrams": 0}, {"negations": 1}])
def test_invalid_condition_configuration_refuses(kw):
    with pytest.raises((TypeError, ValueError)):
        RuleList(["text"], **kw)


def test_three_literal_parity_has_no_adequate_shorter_condition():
    import itertools

    rows = []
    answers = []
    for bits in itertools.product([False, True], repeat=3):
        rows += [{"text": " ".join(x for x, on in zip("ABC", bits) if on)}] * 8
        answers += ["odd" if sum(bits) % 2 else "even"] * 8
    rules = RuleList(["text"], max_literals=3, negations=True).fit(rows, answers)
    assert all(rules.predict(row)[0] == answer for row, answer in zip(rows, answers))
    assert any(len(r["all"]) + len(r["unless"]) == 3 for r in rules.rules)


def test_rich_list_warns_when_more_admissible_rules_are_truncated():
    with pytest.warns(UserWarning, match="max_rules"):
        rules = RuleList(["text"], ngrams=2, max_rules=1).fit(
            [{"text": "meaning of"}] * 8 + [{"text": "of meaning"}] * 8, ["a"] * 8 + ["b"] * 8
        )
    assert rules.truncated == 8


def test_system_learn_rule_forwards_condition_options_and_replays():
    from solvi import Answer, Catalog, Question, System

    s = System(Catalog(), [Question("q", "?", Answer.choice(["yes", "no"]))])
    examples = []
    for text, label in [("", "no"), ("A", "yes"), ("B", "yes"), ("A B", "no")]:
        examples += [({"text": text}, label)] * 8
    s.learn_rule("q", examples, features=["text"], max_literals=2, negations=True)
    for state, want in examples[::8]:
        result = s.ask(state, store=False)
        assert result["q"].answer == want
        assert result.trace.replay(s, result.flow)["ok"]
