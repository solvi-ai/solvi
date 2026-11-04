import asyncio

import pytest

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.plan.check_policy import CheckPolicy
from solvi.core.store import JSONLStorage
from solvi.stat import NaiveCombiner


def make(*, max_cost=None, hard=False, unavailable=False, intervals=None, min_confidence=None):
    calls = []
    cat = Catalog()

    @cat.rule("q")
    def primary(x: int):
        calls.append("primary")
        return Decision("A", {"A": .6, "B": .4})

    @cat.check
    def cheap(ok: bool) -> bool:
        calls.append("cheap")
        return ok

    if unavailable:
        @cat.check
        def expensive(missing: bool) -> bool:
            calls.append("expensive")
            return missing
    else:
        @cat.check
        def expensive(ok: bool) -> bool:
            calls.append("expensive")
            return ok
    if hard:
        @cat.check(hard=True, then={"q": "B"})
        def permitted(allowed: bool) -> bool:
            calls.append("hard")
            return allowed
    rows, targets = [], []
    for correct in (True, False):
        for i in range(10):
            rows.append({k: "agree" if i < (9 if correct else 1) else "disagree"
                         for k in ("cheap", "expensive")})
            targets.append(correct)
    model = NaiveCombiner(["cheap", "expensive"]).fit(rows, targets)
    policy = CheckPolicy(model, costs={"cheap": .01, "expensive": .9}, max_checks=1, max_cost=max_cost)
    q = Question("q", "Question", Answer.choice(["A", "B"]), requires=["cheap", "expensive"],
                 min_confidence=min_confidence)
    return System(cat, [q], check_policies={"q": policy}, check_intervals=intervals), calls


def test_actual_acquisition_pays_only_selected_check_and_replays():
    system, calls = make()
    response = system.ask({"x": 1, "ok": True}, check_loss=2)
    assert calls == ["primary", "cheap"]
    assert response["q"].answer == "A"
    assert response["q"].confidence > .9
    assert response["q"].extra["check_acquisition"]["used"] == ["cheap"]
    assert response["q"].extra["check_acquisition"]["spent"] == .01
    assert response.trace.replay(system)["ok"]
    assert response.trace.replay(system, response.flow)["ok"]
    assert system.answers_of(response.trace, ["q"])["q"].confidence == response["q"].confidence


def test_bad_outcome_abstains_and_zero_budget_never_buys():
    system, calls = make()
    result = system.ask({"x": 1, "ok": False}, check_loss=2)["q"]
    assert result.status == "abstain" and result.guard == "check_policy"
    assert calls == ["primary", "cheap"]
    blocked, calls = make(max_cost=0)
    result = blocked.ask({"x": 1, "ok": True}, check_loss=2)["q"]
    assert result.status == "abstain" and calls == ["primary"]


def test_hard_forced_answer_never_buys_checks_and_replays():
    system, calls = make(hard=True)
    response = system.ask({"x": 1, "ok": True, "allowed": False}, check_loss=100)
    assert response["q"].status == "forced" and response["q"].answer == "B"
    assert calls == ["hard"]
    assert response.trace.replay(system)["ok"]


def test_async_and_storage_roundtrip(tmp_path):
    system, calls = make()
    system.storage = JSONLStorage(tmp_path / "decisions.jsonl", system=system)
    response = asyncio.run(system.aask({"x": 1, "ok": True}, check_loss=2))
    assert calls == ["primary", "cheap"]
    assert response.trace.replay(system)["ok"]
    restored = system.storage.get(response.stored_id)
    assert restored["q"].confidence == response["q"].confidence
    assert restored.trace.replay(system)["ok"]


def test_unavailable_check_is_not_purchased_or_imputed():
    system, calls = make(unavailable=True)
    response = system.ask({"x": 1, "ok": True}, check_loss=2)
    assert calls == ["primary", "cheap"]
    record = next(r for r in response.trace.records if r.kind == "check_policy")
    assert record.extra["available"] == ["cheap"]
    assert response.trace.replay(system)["ok"]


@pytest.mark.parametrize("loss", [None, -1, True, float("nan"), float("inf")])
def test_invalid_request_loss_fails_before_calls(loss):
    system, calls = make()
    with pytest.raises(ValueError, match="check_loss"):
        system.ask({"x": 1, "ok": True}, check_loss=loss)
    assert calls == []


def test_changed_prices_cannot_replay_old_policy_and_rename_keeps_policy():
    system, _ = make()
    response = system.ask({"x": 1, "ok": True}, check_loss=2)
    changed, _ = make(max_cost=0)
    assert not response.trace.replay(changed)["ok"]
    renamed = system.rename(answers={"q": "new_q"})
    result = renamed.ask({"x": 1, "ok": True}, questions="new_q", check_loss=2)
    assert result["new_q"].answer == "A" and result.trace.replay(renamed)["ok"]


def test_acquisition_cannot_overwrite_a_lower_interval_or_its_threshold():
    system, _ = make(intervals={"q": {"mode": "lower", "n": 256, "seed": 7}}, min_confidence=.87)
    response = system.ask({"x": 1, "ok": True}, check_loss=2)
    result = response["q"]
    lower = result.extra["check_evidence"]["uncertainty"]["lower"]
    assert result.confidence == lower
    assert result.status == "abstain" and result.guard == "low_confidence"
    assert response.trace.replay(system)["ok"]


def test_head_is_called_once_and_recorded_answer_recalculation_does_not_call_it():
    class Head:
        features = ["x"]

        def __init__(self):
            self.calls = 0

        def predict(self, values):
            self.calls += 1
            return {"A": .6, "B": .4}

        def contributions(self, values):
            return {"x": .1}

        def fingerprint(self):
            return "acquisition-head"

    system, calls = make()
    del system.catalog.rules["q"]
    system.questions["q"].uses = ["x"]
    head = system.heads["q"] = Head()
    response = system.ask({"x": 1, "ok": True}, check_loss=2)
    assert head.calls == 1 and calls == ["cheap"]
    assert system.answers_of(response.trace, ["q"])["q"].confidence == response["q"].confidence
    assert head.calls == 1
    assert response.trace.replay(system, trust_models=True)["ok"]
    assert head.calls == 1


def test_recalculation_refuses_a_flow_that_drops_mandatory_hard_checks():
    from solvi.core.runtime import Flow

    system, _ = make(hard=True)
    response = system.ask({"x": 1, "ok": True, "allowed": False}, check_loss=2)
    assert not response.trace.replay(system, Flow([], {"q": []}, {}, {}))["ok"]
