import json
import pytest
from solvi import Answer, Catalog, Question, Response, System


def make():
    c = Catalog()

    @c.fn
    def doubled(x: int) -> int:
        return x * 2

    @c.check(hard=True, then={"q": "no"})
    def safe(x: int) -> bool:
        return x >= 0

    @c.rule("q")
    def q(doubled: int) -> bool:
        return doubled > 3

    return System(c, [Question("q", "Q?", Answer.yes_no(), requires=["safe"])])


@pytest.mark.parametrize("x", [2, -1, 0, "3"])
def test_decide_matches_ask_without_audit_or_observation(x):
    s = make()
    before = dict(s.stats)
    costs = dict(s.cost_book.n)
    out = s.decide({"x": x})
    normal = make().ask({"x": x})
    assert out["q"].model_dump() == normal["q"].model_dump()
    assert out.values == normal.values
    assert out.feasible == normal.feasible and out.violations == normal.violations
    assert not isinstance(out, Response) and out.recorded is False and out.replayable is False
    assert s.stats == before and s.cost_book.n == costs
    assert json.loads(out.to_json())["execution"] == "unrecorded"
    with pytest.raises(ValueError, match="ask"):
        _ = out.trace
    with pytest.raises(ValueError, match="ask"):
        out.replay()


@pytest.mark.parametrize(
    "attr,value",
    [
        ("guards", {"q": object()}),
        ("heads", {"q": object()}),
        ("storage", object()),
        ("learn", True),
        ("workers", 2),
        ("shift_watch", {"q": object()}),
        ("strategist", object()),
        ("_cache", object()),
        ("order", "learned"),
        ("producers", "learned"),
        ("adapter_loop", object()),
    ],
)
def test_preflight_rejects_mutation_before_input_validator(attr, value):
    from pydantic import BaseModel, model_validator

    calls = []

    class Input(BaseModel):
        x: int

        @model_validator(mode="before")
        @classmethod
        def called(cls, v):
            calls.append(v)
            return v

    s = make()
    s.input_model = Input
    s.decide({"x": 2})
    calls.clear()
    setattr(s, attr, value)
    with pytest.raises(ValueError, match="decide"):
        s.decide({"x": 2})
    assert not calls


def test_answer_fact_stages_and_constraint_are_real():
    from typing import Literal

    c = Catalog()
    calls = []

    @c.rule("a")
    def a(x: int) -> bool:
        calls.append(x)
        return x > 0

    c.answer_fact("a")

    @c.rule("b")
    def b(a: str) -> Literal["go", "stop"]:
        return "go" if a == "yes" else "stop"

    s = System(c, [Question("a", "A?"), Question("b", "B?", Answer.choice(["go", "stop"]))])
    out = s.decide({"x": 2}, "b")
    assert out["b"].answer == "go" and calls == [2]
    assert out.recorded is False


def test_model_and_async_parts_rejected_before_call():
    c = Catalog()
    calls = []

    @c.rule("q")
    async def q(x: int) -> bool:
        calls.append(x)
        return True

    s = System(c, [Question("q", "Q?")])
    with pytest.raises(ValueError, match="decide"):
        s.decide({"x": 1})
    assert calls == []


def test_storage_cannot_save_unrecorded_result(tmp_path):
    from solvi.core.store import JSONLStorage

    out = make().decide({"x": 2})
    with pytest.raises((ValueError, TypeError), match="ask|record"):
        JSONLStorage(tmp_path / "decisions.jsonl").save(out)


def test_alias_contract_and_joint_constraints_match_ask():
    c = Catalog()

    @c.rule("a")
    def a(x: int) -> bool:
        return x > 0

    @c.rule("b")
    def b(x: int) -> bool:
        return x < 0

    @c.constraint
    def opposite(a, b):
        return a != b

    c.alias("x", "external_x")
    c.contract("a", applicable=lambda st: st["external_x"] > 0, applicable_inputs=("external_x",))
    system = System(c, [Question("a", "A?"), Question("b", "B?")])
    out = system.decide({"external_x": 2})
    asked = system.ask({"external_x": 2})
    assert {n: r.model_dump() for n, r in out.results.items()} == {n: r.model_dump() for n, r in asked.results.items()}
    assert out.feasible == asked.feasible


def test_calibration_mutation_rejected_before_validator():
    from pydantic import BaseModel, model_validator

    calls = []

    class Input(BaseModel):
        x: int

        @model_validator(mode="before")
        @classmethod
        def called(cls, v):
            calls.append("validator")
            return v

    s = make()
    s.input_model = Input
    s.decide({"x": 2})
    calls.clear()
    s.calib["q"] = (0.0, 0.0)
    with pytest.raises(ValueError, match="calib"):
        s.decide({"x": 2})
    assert calls == []


@pytest.mark.parametrize("kind", ["async_function", "async_object", "sync_object", "sync_partial", "async_partial"])
def test_unsupported_contract_callback_refuses_before_validator(kind, recwarn):
    from functools import partial
    from pydantic import BaseModel, model_validator

    calls = []

    class Input(BaseModel):
        x: int

        @model_validator(mode="before")
        @classmethod
        def called(cls, v):
            calls.append("validator")
            return v

    async def af(state):
        calls.append("applicable")
        return True

    def sf(state):
        calls.append("applicable")
        return True

    class Async:
        async def __call__(self, state):
            calls.append("applicable")
            return True

    class Sync:
        def __call__(self, state):
            calls.append("applicable")
            return True

    fn = {
        "async_function": af,
        "async_object": Async(),
        "sync_object": Sync(),
        "sync_partial": partial(sf),
        "async_partial": partial(af),
    }[kind]
    s = make()
    s.input_model = Input
    s.catalog.contract("q", applicable=fn, applicable_inputs=("x",))
    with pytest.raises(ValueError, match="contract"):
        s.decide({"x": 2})
    assert calls == []
    assert not any("never awaited" in str(w.message) for w in recwarn)


def test_async_then_callable_object_is_rejected_before_validator():
    calls = []

    class AsyncThen:
        async def __call__(self, x):
            calls.append(x)
            return False

    s = make()
    s.catalog.parts["safe"].then["q"] = AsyncThen()
    with pytest.raises(ValueError, match="callback"):
        s.decide({"x": 2})
    assert calls == []
