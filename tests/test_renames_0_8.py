"""The names 0.8 gave one concept each still read under their 0.7 spelling for one release: every old name works, warns
once with the new one, and goes in 0.9. (New code, the docs, the examples and the gallery use the new names: the pytest
configuration turns an old name into an error everywhere else.)"""
import importlib

import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, System


def _cat():
    cat = Catalog()

    @cat.fn
    def big(x: float) -> bool:
        return x > 5

    @cat.rule("q")
    def q(big) -> bool:
        return big
    return cat


def _sys(**kw):
    return System(_cat(), [Question("q", "Big?", Answer.yes_no())], **kw)


class In(BaseModel):
    x: float


@pytest.mark.parametrize("old, new, value", [("inputs", "input_model", In), ("costs", "cost_policy", "declared")])
def test_system_options_renamed_in_0_8_still_work_with_a_warning(old, new, value):
    with pytest.warns(DeprecationWarning, match=rf"System\({old}=\) is deprecated: use {new}="):
        s = _sys(**{old: value})
    assert getattr(s, new) == (value if old == "inputs" else None)
    with pytest.raises(TypeError, match=f"got both {old}= and {new}="):
        _sys(**{old: value, new: value})


def test_system_attributes_renamed_in_0_8():
    s = _sys(input_model=In)
    with pytest.warns(DeprecationWarning, match=r"System.inputs is deprecated: use input_model"):
        assert s.inputs is In
    with pytest.warns(DeprecationWarning, match=r"System.costs is deprecated: use cost_book"):
        assert s.costs is s.cost_book


def test_question_requires_was_checkpoints():
    with pytest.warns(DeprecationWarning, match=r"Question\(checkpoints=\) is deprecated: use requires="):
        q = Question("q", "Big?", Answer.yes_no(), checkpoints=["big"])
    assert q.requires == ["big"]
    with pytest.warns(DeprecationWarning, match=r"Question.checkpoints is deprecated: use requires"):
        assert q.checkpoints == ["big"]
    from solvi.core import question_data
    assert question_data(q)["checkpoints"] == ["big"]          # the stored and hashed form keeps its key: fingerprints hold


def test_response_state_text_was_computed_state():
    res = _sys().ask({"x": 9})
    with pytest.warns(DeprecationWarning, match=r"Response.computed_state is deprecated: use Response.state_text\(\)"):
        assert res.computed_state == res.state_text()
    with pytest.warns(DeprecationWarning, match=r"computed_state_text\(\) is deprecated"):
        assert res.computed_state_text("en") == res.state_text("en")


def test_teach_label_source_and_learn_rule_features(tmp_path):
    s = _sys(storage=str(tmp_path / "s.jsonl"))
    with pytest.warns(DeprecationWarning, match=r"System.teach\(source=\) is deprecated: use label_source="):
        s.teach("q", {"x": 9}, True, source="outcome")
    assert s.storage.corrections()[-1]["source"] == "outcome"
    ex = [({"x": float(i)}, i > 5) for i in range(12)]
    with pytest.warns(DeprecationWarning, match=r"System.learn_rule\(facts=\) is deprecated: use features="):
        s.learn_rule("q", ex, facts=["big"])


@pytest.mark.parametrize("old, new, name", [("solvi.fast", "solvi.heads", "FastHead"),
                                            ("solvi.fast", "solvi.heads", "CandidateHead"),
                                            ("solvi.learned", "solvi.costs", "MeasuredCosts"),
                                            ("solvi.learned", "solvi.strategist", "OrderModel"),
                                            ("solvi.rules", "solvi.rulelist", "RuleList")])
def test_modules_renamed_in_0_8_still_read_with_a_warning(old, new, name):
    mod = importlib.import_module(old)
    with pytest.warns(DeprecationWarning, match=rf"{old}.{name} is deprecated: use {new}.{name}"):
        got = getattr(mod, name)
    assert got is getattr(importlib.import_module(new), name)
    with pytest.raises(AttributeError):
        mod.no_such_name  # noqa: B018


def test_decider_options_take_the_question_level_names():
    from solvi.decide import DecideModel
    from test_decide_plumbing import V2, Words
    m = DecideModel(Words(), {**V2, "act": False})
    for old, new, v in (("escalate_below", "min_confidence", 0.7), ("unknown", "not_stated", True)):
        with pytest.warns(DeprecationWarning, match=rf"decision\({old}=\) is deprecated: use {new}="):
            p = m.decision("p", "Signed?", "doc", ["yes", "no"], **{old: v})
        assert p.fingerprint() == m.decision("p", "Signed?", "doc", ["yes", "no"], **{new: v}).fingerprint()
    p = m.decision("p", "Signed?", "doc", ["yes", "no"], min_confidence=0.7)
    assert p.min_confidence == p.escalate_below == 0.7             # the stored name keeps the calibration files' key
    with pytest.warns(DeprecationWarning, match="has_unknown is deprecated: use has_not_stated"):
        assert m.has_unknown is m.has_not_stated is True


def test_heads_report_one_accuracy_each_and_learn_one_label_by_teach():
    from solvi.heads import FastHead, Head
    rows = [{"x": float(i)} for i in range(20)]
    ans = ["a" if i < 10 else "b" for i in range(20)]
    fh = FastHead(["a", "b"]).fit(rows, ans, ["x"])
    with pytest.warns(DeprecationWarning, match="FastHead.cv_acc is deprecated: use loo_acc"):
        assert fh.cv_acc == fh.loo_acc
    with pytest.warns(DeprecationWarning, match=r"FastHead.update\(\) is deprecated: use FastHead.teach\(\)"):
        fh.update({"x": 3.0}, "a")
    h = Head(["a", "b"]).fit(rows, ans, ["x"])
    assert not hasattr(h, "T") and h.cv_acc is not None
    with pytest.warns(DeprecationWarning, match="Head.loo_acc is deprecated: use cv_acc"):
        assert h.loo_acc == h.cv_acc


def test_the_code_strategist_is_cost_strategist_and_model_strategist_takes_a_model():
    from solvi.strategy import CostStrategist, ModelStrategist
    c = CostStrategist(producers="equivalent", on_failure="abstain", keep_alternatives=False)
    assert (c.on_failure, c.keep_alternatives, c.model) == ("abstain", False, None)
    with pytest.warns(DeprecationWarning, match=r"ModelStrategist\(\) without a model is deprecated: use CostStrategist\(\)"):
        m = ModelStrategist(producers="equivalent")
    assert isinstance(m, CostStrategist)
    with pytest.warns(DeprecationWarning, match=r"CostStrategist\(fallbacks=\) is deprecated: use keep_alternatives="):
        assert CostStrategist(fallbacks=False).keep_alternatives is False
    with pytest.warns(DeprecationWarning, match="solvi.strategy_model.SegmentModel is deprecated"):
        from solvi.strategy_model import SegmentModel
    from solvi.segment_model import SegmentModel as S2
    assert SegmentModel is S2
