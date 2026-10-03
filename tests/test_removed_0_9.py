"""The names 0.8 renamed were kept with a SolviDeprecationWarning for one release and are gone in 0.9: an old keyword is a
TypeError and an old attribute or method an AttributeError, each naming the new name; the old modules are gone. (The
CHANGELOG's 0.9 "Breaking changes" lists every one.)"""
import importlib
import json

import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, SolviDeprecationWarning, System

GONE = "was renamed in 0.8 and removed in 0.9: use "


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


def test_the_warning_class_stays_for_later_renames():
    import solvi
    assert issubclass(SolviDeprecationWarning, FutureWarning) and "SolviDeprecationWarning" in solvi.__all__


@pytest.mark.parametrize("module", ["solvi.fast", "solvi.learned", "solvi.rules", "solvi.strategy_model",
                                    "solvi.extract_model"])
def test_the_modules_renamed_in_0_8_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("old, new", [("inputs", "input_model"), ("costs", "cost_policy"),
                                      ("journal", r"storage=JSONLStorage\(path\)")])
def test_system_options(old, new):
    with pytest.raises(TypeError, match=rf"System\({old}=\) {GONE}{new}"):
        _sys(**{old: None})


def test_system_and_response_attributes(tmp_path):
    s = _sys(input_model=In, storage=str(tmp_path / "s.jsonl"))
    for old, new in (("inputs", "input_model"), ("costs", "cost_book"),
                     ("safeguard_report", r"safeguard_summary\(lang\)"),
                     ("fit_fast", r"fit\(question, examples, features, select=False\)")):
        with pytest.raises(AttributeError, match=rf"System.{old}(\(\))? {GONE}{new}"):
            getattr(s, old)
    res = s.ask({"x": 9})
    for old, new in (("computed_state", r"state_text\(\)"), ("computed_state_text", r"state_text\(lang\)"),
                     ("textin", "read")):
        with pytest.raises(AttributeError, match=rf"Response.{old}(\(\))? {GONE}{new}"):
            getattr(res, old)
    with pytest.raises(AttributeError, match=rf"Trace.value\(name\) {GONE}res.values\[name\]"):
        res.trace.value("big")
    with pytest.raises(TypeError, match=rf"System.ask\(names=\) {GONE}questions="):
        s.ask({"x": 9}, names=["q"])
    with pytest.raises(TypeError, match=rf"System.teach\(source=\) {GONE}label_source="):
        s.teach("q", {"x": 9}, True, source="outcome")
    with pytest.raises(TypeError, match=rf"System.learn_rule\(facts=\) {GONE}features="):
        s.learn_rule("q", [({"x": 9.0}, True)], facts=["big"])
    with pytest.raises(TypeError, match=rf"{GONE}System.calibrate\(question, \[\(state, answer\), \.\.\.\]\)"):
        s.calibrate("q", [{"x": 9}], [True])
    with pytest.raises(TypeError, match="unexpected keyword argument 'harvest_rules'"):
        s.learning(harvest_rules=True)


def test_question_and_answer_type():
    with pytest.raises(TypeError, match=rf"Question\(checkpoints=\) {GONE}requires="):
        Question("q", "Big?", Answer.yes_no(), checkpoints=["big"])
    q = Question("q", "Big?", Answer.yes_no(), requires=["big"])
    with pytest.raises(AttributeError, match=rf"Question.checkpoints {GONE}requires"):
        q.checkpoints  # noqa: B018
    from solvi.core import question_data
    assert question_data(q)["checkpoints"] == ["big"]          # the stored and hashed form keeps its key: fingerprints hold
    with pytest.raises(AttributeError, match=rf"AnswerType.rank\(v\) {GONE}options.index\(v\)"):
        Answer.ordinal(["low", "high"]).rank("low")


def test_decider_options_and_names():
    from solvi.decide import DecideModel
    from test_decide_plumbing import V2, Words
    m = DecideModel(Words(), {**V2, "act": False})
    for old, new in (("escalate_below", "min_confidence"), ("act_threshold", "min_act"),
                     ("target_error", "max_error"), ("unknown", "not_stated")):
        with pytest.raises(TypeError, match=rf"decision\({old}=\) {GONE}{new}="):
            m.decision("p", "Signed?", "doc", ["yes", "no"], **{old: 0.5})
    with pytest.raises(TypeError, match=rf"decide\(escalate_below=\) {GONE}min_confidence="):
        m.decide("a text", "Signed?", ["yes", "no"], escalate_below=0.5)
    with pytest.raises(TypeError, match=rf"decide\(unknown=\) {GONE}not_stated="):
        m.decide("a text", "Signed?", ["yes", "no"], unknown=True)
    p = m.decision("p", "Signed?", "doc", ["yes", "no"], min_confidence=0.7)
    assert p.min_confidence == p.escalate_below == 0.7             # the stored name keeps the calibration files' key
    for obj, owner, old, new in ((m, "DecideModel", "has_unknown", "has_not_stated"),
                                 (m, "DecideModel", "long_len", "max_len_long"),
                                 (p, "DecisionPart", "long_len", "max_len_long")):
        with pytest.raises(AttributeError, match=rf"{owner}.{old} {GONE}{new}"):
            getattr(obj, old)
    with pytest.raises(TypeError, match=rf"act_guard\(risk=\) {GONE}max_risk="):
        p.act_guard([], risk=0.1)
    with pytest.raises(TypeError, match=rf"calibrate_for\(error=\) {GONE}max_error="):
        p.calibrate_for([], error=0.1)
    with pytest.raises(TypeError, match=rf"question\(checkpoints=\) {GONE}requires="):
        p.question(Catalog(), checkpoints=["x"])

    class Contract(BaseModel):
        signed: bool = pydantic_field(description="Is it signed?", json_schema_extra={"escalate_below": 0.6})
    with pytest.raises(ValueError, match=r"Contract.signed: json_schema_extra 'escalate_below' was renamed in 0.8 and "
                                         r"removed in 0.9: use 'min_confidence'"):
        m.decisions(Contract)


def pydantic_field(**kw):
    from pydantic import Field
    return Field(**kw)


def test_combination_names():
    from solvi.decide import Facts
    from solvi.multi import Cascade
    from test_multi import CLEAR, _parts
    s, l_, _, _ = _parts()
    c = Cascade([s, l_])
    with pytest.raises(TypeError, match=rf"decide\(x=\) {GONE}text="):
        c.decide(x=CLEAR)
    with pytest.raises(TypeError, match=rf"teach\(x=\) {GONE}text="):
        c.teach(x=Facts(email=CLEAR), correct="billing")
    with pytest.raises(TypeError, match=rf"adapt\(inputs=\) {GONE}texts="):
        c.adapt(inputs=[CLEAR])
    assert c.text_of(facts={"email": CLEAR, "other": 1}) == Facts(email=CLEAR)
    with pytest.raises(AttributeError, match=rf"Combination.usage\(\) {GONE}calls\(\)"):
        c.usage()
    out = c.calls()
    assert "per_question" not in out and "calls_per_question" in out


def test_heads_and_learned_policies():
    from solvi.heads import FastHead, Head
    from solvi.strategist import Binary
    rows = [{"x": float(i)} for i in range(20)]
    ans = ["a" if i < 10 else "b" for i in range(20)]
    fh = FastHead(["a", "b"]).fit(rows, ans, ["x"])
    with pytest.raises(AttributeError, match=rf"FastHead.cv_acc {GONE}loo_acc"):
        fh.cv_acc  # noqa: B018
    with pytest.raises(AttributeError, match=rf"FastHead.update\(\) {GONE}teach\(\)"):
        fh.update({"x": 3.0}, "a")
    h = Head(["a", "b"]).fit(rows, ans, ["x"])
    assert h.cv_acc is not None
    with pytest.raises(AttributeError, match=rf"Head.loo_acc {GONE}cv_acc"):
        h.loo_acc  # noqa: B018
    with pytest.raises(AttributeError, match=rf"Binary.observe\(\) {GONE}teach\(\)"):
        Binary().observe({"x": 1.0}, True)


def test_strategists():
    from solvi.strategy import CostStrategist, ModelStrategist
    with pytest.raises(TypeError, match=r"ModelStrategist\(\) needs a model: without one it is CostStrategist\(\)"):
        ModelStrategist(None)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'model'"):
        ModelStrategist()  # type: ignore[call-arg]
    for old, new in (("fallback", "on_failure"), ("fallbacks", "keep_alternatives")):
        with pytest.raises(TypeError, match=rf"CostStrategist\({old}=\) {GONE}{new}="):
            CostStrategist(**{old: False})
        with pytest.raises(AttributeError, match=rf"CostStrategist.{old} {GONE}{new}"):
            getattr(CostStrategist(), old)


def test_storage_tooling_and_agents(tmp_path):
    from solvi import JSONLStorage, testing
    from solvi.agents import Guard
    from solvi.memory import CorrectionMemory
    with pytest.raises(TypeError, match=rf"JSONLStorage\(catalog=\) {GONE}system="):
        JSONLStorage(tmp_path / "a.jsonl", catalog=None)
    st = JSONLStorage(tmp_path / "a.jsonl")
    s = _sys(storage=st)
    sid = s.ask({"x": 9}).stored_id
    with pytest.raises(AttributeError, match=rf"TraceStorage.forget\(\) {GONE}where_is\(fact, value\)"):
        st.forget("x", 9)
    with pytest.raises(TypeError, match=rf"query\(catalog=\) {GONE}catalog_fp="):
        st.query(catalog=s.fingerprint()["catalog"])
    with pytest.raises(TypeError, match=rf"get\(catalog=\) {GONE}system="):
        st.get(sid, catalog=s)
    with pytest.raises(TypeError, match=rf"save_correction\(source=\) {GONE}label_source="):
        st.save_correction("q", {"x": 9}, True, source="outcome")
    assert not hasattr(testing, "check") and callable(testing.run_case)
    with pytest.raises(TypeError, match=rf"Guard\(facts=\) {GONE}fact_names="):
        Guard(facts=["user_id"])
    with pytest.raises(TypeError, match=rf"calibrate_authorizer\(risk=\) {GONE}max_risk="):
        Guard().calibrate_authorizer([], risk=0.1)
    with pytest.raises(TypeError, match=rf"calibrate\(risk=\) {GONE}max_risk="):
        CorrectionMemory.calibrate(None, risk=0.1)
    import solvi.serve as serve
    assert not hasattr(serve, "Guard") and serve.AccessGuard
    from solvi.agents.mcp import run_proxy
    for old, new in (("context_messages", "max_messages"), ("context_chars", "max_chars")):
        with pytest.raises(TypeError, match=rf"run_proxy\({old}=\) {GONE}{new}="):
            run_proxy(None, **{old: 5})


def test_files_and_command_lines(tmp_path, monkeypatch, capsys):
    from solvi import honesty, hooks
    p = tmp_path / "set.json"
    p.write_text(json.dumps({"cases": [{"name": "a", "state": {}, "gold": {"q": "yes"}}]}))
    with pytest.raises(ValueError, match=r'has "gold", the key before 0.8 \(removed in 0.9\): name its right answers "expected"'):
        honesty.load_set(p)
    assert hooks.main(["pre-edit", "--model", "m"]) == 1               # 1, not 2: an old installed hook blocks nothing
    assert "--model was renamed in 0.8 and removed in 0.9: use --decider" in capsys.readouterr().err
    monkeypatch.delenv("SOLVI_HOOK_DECIDER", raising=False)
    monkeypatch.setenv("SOLVI_HOOK_MODEL", "m")
    from solvi.loader import LoadError
    with pytest.raises(LoadError, match=r"\$SOLVI_HOOK_MODEL was renamed in 0.8 and removed in 0.9: set \$SOLVI_HOOK_DECIDER"):
        hooks.rules_system()


def test_extractors():
    from solvi.extract_multi import MultiSpanExtractor
    ex = MultiSpanExtractor.__new__(MultiSpanExtractor)
    with pytest.raises(TypeError, match=rf"MultiSpanExtractor.fit\(docs, spans\) {GONE}fit\(\[\(text, spans\), \.\.\.\]\)"):
        ex.fit(["a text"], [{"total": None}])  # type: ignore[call-arg]
    with pytest.raises(AttributeError, match=rf"MultiSpanExtractor.predict_doc\(\) {GONE}predict\(text\[, field\]\)"):
        ex.predict_doc("a text")


def test_the_adapters_take_auto_declare():
    from solvi import _deprecate

    @_deprecate.removed_kwargs(declare="auto_declare")
    def adapter(guard, *, auto_declare=False):
        return auto_declare
    assert adapter(None, auto_declare=True) is True
    with pytest.raises(TypeError, match=rf"adapter\(declare=\) {GONE}auto_declare="):
        adapter(None, declare=True)
    for mod, fn in (("solvi.agents.openai_agents", "guard_tool"), ("solvi.agents.langgraph", "guard_wrappers"),
                    ("solvi.agents.langgraph", "guarded_tool_node"), ("solvi.agents.pydantic_ai", "GuardedToolset")):
        try:
            f = getattr(importlib.import_module(mod), fn)
        except ImportError:                            # the framework is not installed
            continue
        with pytest.raises(TypeError, match=rf"{fn}\(declare=\) {GONE}auto_declare="):
            f(None, None, declare=True)
