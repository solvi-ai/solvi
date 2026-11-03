import pytest

from solvi import Answer, Catalog, Question, Response, System
from solvi.core.catalog import AnswerType


@pytest.mark.parametrize("value", [None, True, False, 0, 1.25, "", "text", [1, None], {"x": True}])
def test_json_value_keeps_scalars_and_null_as_answers(value):
    cat = Catalog()
    cat.rule("result")(lambda x: value)
    system = System(cat, [Question("result", "Value?", Answer.value())], cache=2)
    response = system.ask({"x": 1})
    assert response["result"].status == "ok" and response["result"].answer == value
    assert response.trace.replay(system, response.flow)["ok"]
    restored = Response.from_json(response.to_json(), catalog=system)
    assert restored["result"].answer == value and restored["result"].status == "ok"
    assert restored.trace.replay(system, restored.flow)["ok"]
    assert AnswerType.from_json(Answer.value().to_json()).normalize(value) == value


def test_typed_bool_value_does_not_turn_into_yes_no():
    cat = Catalog()

    @cat.rule("result")
    def result(x: int) -> bool:
        return x > 0

    system = System(cat, [Question("result", "Value?", Answer.value())])
    assert system.ask({"x": 1})["result"].answer is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), b"bytes", object(), {1: "wrong key"}])
def test_json_value_rejects_non_json(bad):
    cat = Catalog()
    cat.rule("result")(lambda x: bad)
    system = System(cat, [Question("result", "Value?", Answer.value())])
    assert system.ask({"x": 1})["result"].status == "abstain"


def test_default_structured_and_ordinary_none_keep_existing_contracts():
    cat = Catalog()

    @cat.rule("ordinary")
    def ordinary(x) -> bool:
        return None

    system = System(cat, [Question("ordinary", "Ordinary?")])
    assert system.ask({"x": 1})["ordinary"].status == "abstain"
    with pytest.raises((ValueError, TypeError)):
        Answer.structured().normalize(3)


def test_typed_structured_array_stays_a_list_on_load():
    cat = Catalog()
    cat.rule("result")(lambda x: [1, 2])
    system = System(cat, [Question("result", "Array?", Answer.structured(list[int]))])
    response = system.ask({"x": 1})
    restored = Response.from_json(response.to_json(), catalog=system)
    assert restored["result"].answer == [1, 2]
    assert restored.trace.replay(system, restored.flow)["ok"]
