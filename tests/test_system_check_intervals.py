"""Conditional check intervals in decision flow, with no coverage claim."""
import asyncio

import pytest

from solvi import System
from solvi.core._check_evidence import freeze_intervals
from test_system_check_models import catalog, fitted, question


def system(mode="report", **kwargs):
    return System(catalog(), [question(**kwargs)], check_models={"q": fitted()},
                  check_intervals={"q": {"mode": mode, "n": 256, "seed": 7}})


def test_report_retains_point_and_lower_drives_decision_before_threshold():
    report = system()
    result = report.ask({"x": 1, "ok": True})["q"]
    evidence = result.extra["check_evidence"]
    uncertainty = evidence["uncertainty"]
    expected = fitted().posterior_interval(.7, {"weak": "agree"}, n=256, seed=7)
    assert result.confidence == evidence["posterior"]
    assert (uncertainty["lower"], uncertainty["upper"]) == expected
    assert uncertainty["interpretation"] == "model_conditional"
    lower = system("lower").ask({"x": 1, "ok": True})["q"]
    assert lower.confidence == expected[0]
    threshold = (expected[0] + result.confidence) / 2
    assert system("report", min_confidence=threshold).ask({"x": 1, "ok": True})["q"].status == "ok"
    assert system("lower", min_confidence=threshold).ask({"x": 1, "ok": True})["q"].status == "abstain"


def test_settings_are_copied_and_identity_replay_async_rename_hold():
    config = {"q": {"mode": "lower", "seed": 19, "n": 64}}
    obj = System(catalog(), [question()], check_models={"q": fitted()}, check_intervals=config)
    before = obj.fingerprint()
    config["q"]["seed"] = 20
    assert obj.fingerprint() == before
    result = obj.ask({"x": 1, "ok": True})
    assert result.trace.replay(obj)["ok"]
    assert asyncio.run(obj.aask({"x": 1, "ok": True}))["q"].confidence == result["q"].confidence
    assert obj.rename(answers={"q": "new"}).ask({"x": 1, "ok": True})["new"].confidence == result["q"].confidence
    assert system("report").fingerprint() != system("lower").fingerprint()


@pytest.mark.parametrize("config", [
    {"unknown": {}}, {"q": {"unknown": 1}}, {"q": []}, {"q": {"level": True}},
    {"q": {"level": 1}}, {"q": {"level": float("nan")}}, {"q": {"n": True}},
    {"q": {"n": 0}}, {"q": {"seed": -1}}, {"q": {"seed": True}},
    {"q": {"mode": "unsupported"}},
])
def test_bad_settings_rejected_without_input_mutation(config):
    with pytest.raises(ValueError):
        freeze_intervals({"q": question()}, {"q": fitted()}, config)


def test_model_required_and_no_configuration_preserves_identity():
    with pytest.raises(ValueError, match="configured check model"):
        System(catalog(), [question()], check_intervals={"q": {}})
    a = System(catalog(), [question()], check_models={"q": fitted()})
    b = System(catalog(), [question()], check_models={"q": fitted()}, check_intervals={})
    assert a.fingerprint() == b.fingerprint()
    assert "uncertainty" not in a.ask({"x": 1, "ok": True})["q"].extra["check_evidence"]


def test_hard_abstention_never_gets_interval_evidence():
    obj = System(catalog(hard=True), [question()], check_models={"q": fitted()},
                 check_intervals={"q": {"mode": "lower"}})
    result = obj.ask({"x": 1, "ok": True, "allowed": False})["q"]
    assert result.status == "abstain"
    assert "check_evidence" not in (result.extra or {})


def test_legacy_fitted_uncertainty_configuration_fails_at_construction():
    model = fitted()
    del model._log_concentrations
    point = System(catalog(), [question()], check_models={"q": model})
    assert point.ask({"x": 1, "ok": True})["q"].status == "ok"
    with pytest.raises(ValueError, match="refit a legacy"):
        System(catalog(), [question()], check_models={"q": model}, check_intervals={"q": {}})
