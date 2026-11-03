import pytest

from solvi import Decision, Unknown
from solvi.testing.conformance import ConformanceError, check_decider


class Decider:
    options = ["a", "b"]

    def __init__(self, decision, **attrs):
        self.decision = decision
        self.__dict__.update(attrs)

    def fingerprint(self):
        return "contract-test"

    def __call__(self, **facts):
        return self.decision


@pytest.mark.parametrize("decision", [
    Decision("a", {"a": .8, "b": .8}),
    Decision("a", {"outside": 1.0}),
    Decision("a", confidence=1.7),
    Decision("a", confidence=float("nan")),
    Decision("outside", escalate="unsure"),
    Decision(Unknown),
    Decision(Unknown, escalate="unsure"),
])
def test_invalid_closed_decisions_are_rejected(decision):
    with pytest.raises(ConformanceError):
        check_decider(Decider(decision), [{}])


@pytest.mark.parametrize("decision,attrs", [
    (Decision("a", {"a": .8, "b": .2}), {}),
    (Decision("a", {"a": .8, "b": .2}, escalate="unsure"), {}),
    (Decision(None, escalate="no proposal"), {}),
    (Decision(Unknown, {Unknown: 1.}), {"options": ["a", Unknown]}),
    (Decision(Unknown, escalate="unsure"), {"options": ["a", Unknown]}),
    (Decision(["a", "b"], {"a": .8, "b": .8}), {"multi": True, "kind": "multi"}),
    (Decision(Unknown, {"a": .8, Unknown: .8}),
     {"multi": True, "kind": "multi", "options": ["a", Unknown]}),
    (Decision(2.5), {"options": None, "kind": "number"}),
    (Decision("passage"), {"options": None, "kind": "span"}),
    (Decision("a", {"a": .2, "b": .8}), {"kind": "score"}),
])
def test_valid_refusals_and_other_decoding_modes(decision, attrs):
    assert check_decider(Decider(decision, **attrs), [{}])["inputs"] == 1


@pytest.mark.parametrize("as_rule", [False, True])
def test_protocol_only_callable_registers_and_records_identity(as_rule):
    from solvi import Answer, Catalog, Question, System
    class Honest:
        options = ["a", "b"]
        def fingerprint(self):
            return "honest-protocol"
        def __call__(self, text):
            return Decision("a", {"a": .8, "b": .2})
    dec = Honest()
    cat = Catalog()
    if as_rule:
        cat.rule("answer")(dec)
    else:
        cat.fn(dec)
        @cat.rule("answer")
        def answer(Honest):
            return Honest
    system = System(cat, [Question("answer", "?", Answer.choice(dec.options))])
    states = [{"text": "hello"}]
    assert check_decider(dec, states, system=system, states=states)["system"]


def test_interleaved_repeat_detects_state_leak():
    class Leaky(Decider):
        def __init__(self):
            self.seen_b = False
        def __call__(self, text):
            value = "b" if text == "A" and self.seen_b else "a"
            self.seen_b |= text == "B"
            return Decision(value)
    with pytest.raises(ConformanceError, match="same input the same way"):
        check_decider(Leaky(), [{"text": "A"}, {"text": "B"}])


def test_check_detects_input_mutation_and_preserves_the_callers_nested_input():
    class Mutating(Decider):
        def __call__(self, payload):
            payload["items"].append("changed")
            return Decision("a")
    rows = [{"payload": {"items": ["original"]}}]
    with pytest.raises(ConformanceError, match="mutat"):
        check_decider(Mutating(Decision("a")), rows)
    assert rows == [{"payload": {"items": ["original"]}}]


def test_repeat_count_can_probe_later_drift_but_is_bounded():
    class Drifting(Decider):
        def __init__(self):
            self.calls = 0
        def __call__(self, **facts):
            self.calls += 1
            return Decision("a" if self.calls <= 3 else "b")
    assert check_decider(Drifting(), [{}])["inputs"] == 1
    with pytest.raises(ConformanceError, match="same input the same way"):
        check_decider(Drifting(), [{}], repeats=4)


@pytest.mark.parametrize("repeats", [1, 0, True, 2.5])
def test_invalid_repeat_count_has_a_clear_contract_error(repeats):
    with pytest.raises(ConformanceError, match="repeats"):
        check_decider(Decider(Decision("a")), [{}], repeats=repeats)
