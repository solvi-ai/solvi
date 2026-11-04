"""Reachability validation does not ask a custom planner to plan nonexistent hard checks."""
import asyncio

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.plan.strategist import plan


def make():
    cat = Catalog()

    @cat.fn
    def score(a: int, b: int) -> int:
        return a * 3 + b

    @cat.rule("act")
    def act(score: int) -> str:
        return "go" if score > 10 else "stop"

    class Planner:
        def __init__(self):
            self.calls = []

        def plan(self, catalog, questions, init_keys, heads):
            self.calls.append(tuple(q.name for q in questions))
            return plan(catalog, questions, init_keys, heads)

    planner = Planner()
    system = System(cat, [Question("act", "?", Answer.choice(["go", "stop"]))], strategist=planner)
    planner.calls.clear()
    return system, planner


@pytest.mark.parametrize("asynchronous", [False, True])
def test_ordinary_ask_builds_one_actual_flow_and_preserves_trace(asynchronous):
    system, planner = make()
    facts = {"a": 3, "b": 2}
    result = (asyncio.run(system.aask(facts, store=False)) if asynchronous
              else system.ask(facts, store=False))
    assert result["act"].answer == "go"
    assert planner.calls == [("act",)]
    assert result.trace.replay(system, result.flow)["ok"]


def test_newly_registered_idle_hard_check_is_still_refused_before_answer():
    system, planner = make()
    assert system.ask({"a": 3, "b": 2}, store=False)["act"].answer == "go"

    @system.catalog.check(hard=True)
    def new_guard(permission: bool) -> bool:
        return permission

    with pytest.raises(ValueError, match="new_guard.*never runs"):
        system.ask({"a": 3, "b": 2}, store=False)


def test_soft_to_hard_edit_is_not_hidden_by_an_earlier_fast_path():
    system, planner = make()

    @system.catalog.check
    def permission(ok: bool) -> bool:
        return ok

    system.ask({"a": 3, "b": 2, "ok": False}, store=False)
    system.catalog.parts["permission"].hard = True
    with pytest.raises(ValueError, match="permission.*never runs"):
        system.ask({"a": 3, "b": 2, "ok": False}, store=False)
