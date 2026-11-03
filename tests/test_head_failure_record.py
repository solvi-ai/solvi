import pytest
from solvi import Catalog, Question, Answer, System, Response
from solvi.core.runtime import MISSING, _replay_head


class BrokenHead:
    features = ["x"]

    def __init__(self):
        self.calls = 0
        self.finite = False

    def predict(self, values):
        self.calls += 1
        return {"yes": 0.9, "no": 0.1} if self.finite else {"yes": float("nan"), "no": 0.1}

    def contributions(self, values):
        raise AssertionError("failure must not explain")

    def fingerprint(self):
        return "broken-head-fixed"


def make():
    cat = Catalog()
    cat.answer_fact("q")

    @cat.rule("down")
    def down(q: str) -> bool:
        return q == "yes"

    s = System(cat, [Question("q", "Q?", Answer.yes_no(), uses=["x"]), Question("down", "Down?")])
    head = s.heads["q"] = BrokenHead()
    return s, head


def test_nonfinite_head_record_roundtrip_and_independent_replay():
    s, h = make()
    out = s.ask({"x": 3})
    assert out["q"].status == "abstain" and out["q"].guard == "missing_facts"
    assert out["q"].source == "BrokenHead" and out["q"].why == "the answer head gave no finite probabilities"
    assert out["down"].status == "abstain" and h.calls == 1
    record = next(r for r in out.trace.records if r.kind == "head")
    assert record.value is MISSING and record.probs == {} and record.confidence == 0
    derived = s.answers_of(out.trace, ["q"], out.flow)["q"]
    assert (derived.status, derived.guard, derived.source, derived.why) == (
        out["q"].status,
        out["q"].guard,
        out["q"].source,
        out["q"].why,
    )
    assert h.calls == 1
    restored = Response.from_json(out.to_json(), catalog=s)
    assert restored.trace.replay(s, restored.flow)["ok"]
    assert h.calls == 2
    assert restored.trace.replay(s, restored.flow, trust_models=True)["ok"]
    assert h.calls == 2
    h.finite = True
    assert not restored.trace.replay(s, restored.flow)["ok"]
    assert h.calls == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("error", "forged"),
        ("head_status", "forced"),
        ("head_guard", "forged"),
        ("confidence", 0.5),
        ("head_failure", "forged"),
    ],
)
def test_failure_metadata_cannot_be_forged_even_when_trusting_models(field, value):
    s, h = make()
    out = s.ask({"x": 3})
    r = next(r for r in out.trace.records if r.kind == "head")
    if field in ("error", "confidence"):
        setattr(r, field, value)
    else:
        r.extra[field] = value
    assert _replay_head(r, s.heads, {"x": 3}, True, [])
    assert h.calls == 1


class FiniteHead(BrokenHead):
    def predict(self, values):
        self.calls += 1
        return {"yes": 0.9, "no": 0.1}

    def contributions(self, values):
        return {"x": 0.2}


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("prob", float("nan")),
        ("prob", float("inf")),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", 0.99),
    ],
)
def test_normal_head_record_rejects_nonfinite_probs_and_wrong_confidence(trust, field, value):
    s = System(Catalog(), [Question("q", "Q?", Answer.yes_no(), uses=["x"])])
    s.heads["q"] = FiniteHead()
    out = s.ask({"x": 3})
    r = next(r for r in out.trace.records if r.kind == "head")
    if field == "prob":
        r.probs["yes"] = value
    else:
        r.confidence = value
    assert _replay_head(r, s.heads, {"x": 3}, trust, [])


@pytest.mark.parametrize(
    "answer,probs",
    [
        (Answer.yes_no(), {"yes": 0.9, "no": 0.1}),
        (Answer.ordinal(["low", "medium", "high"]), {"low": 0.4, "medium": 0.15, "high": 0.45}),
        (Answer.multi(["a", "b"]), {"a": 0.7, "b": 0.2}),
    ],
)
def test_valid_normal_head_confidence_math_keeps_class_ordinal_multi(answer, probs):
    class Head(FiniteHead):
        def predict(self, values):
            self.calls += 1
            return dict(probs)

    s = System(Catalog(), [Question("q", "Q?", answer, uses=["x"])])
    s.heads["q"] = Head()
    out = s.ask({"x": 3})
    assert out.trace.replay(s, out.flow)["ok"]
    assert out.trace.replay(s, out.flow, trust_models=True)["ok"]
