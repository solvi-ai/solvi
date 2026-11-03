import pytest

from solvi import Answer, Catalog, Question, System
from solvi.testing import run_case


@pytest.mark.parametrize("value", [None, "abstain", "<not stated>"])
@pytest.mark.parametrize("fast", [False, True])
def test_explicit_ok_status_tests_literal_json_values(value, fast):
    cat = Catalog()
    cat.rule("result")(lambda x: value)
    system = System(cat, [Question("result", "Value?", Answer.value())])
    case = {"name": "literal", "expected": {"result": value}, "status": {"result": "ok"}}
    result = run_case(system, case, {"x": 1}, fast=fast)
    assert result.ok, result.problems


@pytest.mark.parametrize("actual, expected", [(True, 1), (False, 0), ({"x": True}, {"x": 1})])
def test_json_value_testing_distinguishes_booleans_from_numbers(actual, expected):
    cat = Catalog()
    cat.rule("result")(lambda x: actual)
    system = System(cat, [Question("result", "Value?", Answer.value())])
    result = run_case(system, {"expected": {"result": expected}, "status": {"result": "ok"}}, {"x": 1})
    assert not result.ok


def test_null_without_explicit_ok_status_keeps_abstention_contract():
    cat = Catalog()
    cat.rule("result")(lambda x: None)
    system = System(cat, [Question("result", "Ordinary?", Answer.yes_no())])
    assert run_case(system, {"expected": {"result": None}}, {"x": 1}).ok


@pytest.mark.parametrize("record", ["full", "compact"])
@pytest.mark.parametrize("value", [None, "abstain", "<not stated>", {"enabled": True}])
def test_saved_json_values_export_as_runnable_decision_cases(tmp_path, value, record):
    import json

    from solvi.core.store import JSONLStorage
    from solvi.testing import cases_from_store

    cat = Catalog()
    cat.rule("result")(lambda x: value)
    store = JSONLStorage(str(tmp_path / "values.jsonl"), record=record)
    system = System(cat, [Question("result", "Value?", Answer.value())], storage=store)
    system.ask({"x": 1})
    cases = json.loads(json.dumps(cases_from_store(store)))
    assert len(cases) == 1
    result = run_case(system, cases[0], cases[0]["state"])
    assert result.ok, result.problems
