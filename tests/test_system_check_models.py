"""Runtime bridge for fitted conditional check likelihoods, not a quality benchmark."""
import asyncio
import pickle

import pytest

from solvi import Answer, Catalog, Decision, Question, System
from solvi.stat import CorrelationAwareCombiner, NaiveCombiner


def fitted(names=("weak",), *, correlated=False):
    rows, targets = [], []
    for correct in (True, False):
        for agree in [True] * (8 if correct else 2) + [False] * (2 if correct else 8):
            rows.append({name: "agree" if agree else "disagree" for name in names})
            targets.append(correct)
    model = (CorrelationAwareCombiner(names, {(a, b): .8 for a in names for b in names if a != b})
             if correlated else NaiveCombiner(names))
    return model.fit(rows, targets)


def catalog(*, hard=False):
    cat = Catalog()

    @cat.check
    def weak(ok: bool) -> bool:
        return ok

    @cat.rule("q")
    def choose(x: int):
        return Decision("A", {"A": .7, "B": .3})

    if hard:
        @cat.check(hard=True, then={"q": None})
        def permitted(allowed: bool) -> bool:
            return allowed
    return cat


def question(**kwargs):
    return Question("q", "Choose", Answer.choice(["A", "B"]), requires=["weak"], **kwargs)


@pytest.mark.parametrize("ok", [True, False])
def test_posterior_matches_the_existing_native_model_and_is_recorded(ok):
    model = fitted()
    system = System(catalog(), [question()], check_models={"q": model})
    result = system.ask({"x": 1, "ok": ok})
    expected = model.posterior(.7, {"weak": "agree" if ok else "disagree"})
    assert result["q"].answer == "A" and result["q"].confidence == pytest.approx(expected)
    assert result["q"].probs == {"A": .7, "B": .3}
    assert result["q"].extra["check_evidence"]["model"] == model.fingerprint()
    assert result.trace.replay(system)["ok"]
    assert system.answers_of(result.trace, ["q"], result.flow)["q"].confidence == result["q"].confidence


def test_hard_check_cannot_be_overridden_and_low_confidence_still_abstains():
    model = fitted()
    system = System(catalog(hard=True), [question()], check_models={"q": model})
    assert system.ask({"x": 1, "ok": True, "allowed": False})["q"].status == "abstain"
    cautious = System(catalog(), [question(min_confidence=.6)], check_models={"q": model})
    assert cautious.ask({"x": 1, "ok": False})["q"].status == "abstain"


def test_supplied_model_is_snapshotted_and_refit_does_not_change_system():
    model = fitted()
    system = System(catalog(), [question()], check_models={"q": model})
    old = system.fingerprint()
    before = system.ask({"x": 1, "ok": True})["q"].confidence
    model.fit([{"weak": "agree"}] * 10, [False] * 10)
    assert system.fingerprint() == old
    assert system.ask({"x": 1, "ok": True})["q"].confidence == before


def test_freeze_resolves_correlation_callbacks_and_is_pickleable():
    rho = {("weak", "other"): .8, ("other", "weak"): .8}
    model = fitted(("weak", "other"), correlated=True)
    model.rho = lambda a, b: 1.0 if a == b else rho[(a, b)]
    frozen = model.freeze()
    initial = frozen.posterior(.7, {"weak": "agree", "other": "agree"})
    rho.update({("weak", "other"): 0.0, ("other", "weak"): 0.0})
    restored = pickle.loads(pickle.dumps(frozen))
    assert restored.posterior(.7, {"weak": "agree", "other": "agree"}) == initial
    assert restored.fingerprint() == frozen.fingerprint()
    with pytest.raises(ValueError, match="frozen"):
        frozen.fit([], [])


def test_freeze_preserves_native_callback_symmetry_tolerance():
    model = fitted(("weak", "other"), correlated=True)
    model.rho = lambda a, b: 1.0 - 1e-14 if a == b else .8 + (1e-14 if a < b else 0)
    outcomes = {"weak": "agree", "other": "agree"}
    assert model.freeze().posterior(.7, outcomes) == model.posterior(.7, outcomes)


def test_async_replay_and_rename_use_the_same_model():
    system = System(catalog(), [question()], check_models={"q": fitted()})
    initial = system.ask({"x": 1, "ok": True})
    other = asyncio.run(system.aask({"x": 1, "ok": True}))
    assert initial["q"].confidence == other["q"].confidence
    assert other.trace.replay(system)["ok"]
    renamed = system.rename(answers={"q": "new_q"})
    assert renamed.ask({"x": 1, "ok": True})["new_q"].confidence == initial["q"].confidence


def test_no_model_preserves_the_old_fingerprint_and_confidence():
    cat = catalog()
    system, empty = System(cat, [question()]), System(cat, [question()], check_models={})
    assert system.fingerprint() == empty.fingerprint()
    assert system.ask({"x": 1, "ok": False})["q"].confidence == .7


def test_modeled_checks_must_be_explicit_posthoc_requirements():
    model = fitted()
    with pytest.raises(ValueError, match="requires"):
        System(catalog(), [Question("q", "Choose", Answer.choice(["A", "B"]))], check_models={"q": model})
    cat = catalog()

    @cat.rule("q")
    def dependent(weak: bool):
        return Decision("A", {"A": .7, "B": .3})

    with pytest.raises(ValueError, match="already be read"):
        System(cat, [question()], check_models={"q": model})


def test_unrecorded_and_cache_paths_refuse_models():
    with pytest.raises(ValueError, match="cache"):
        System(catalog(), [question()], check_models={"q": fitted()}, cache=5)
    system = System(catalog(), [question()], check_models={"q": fitted()})
    with pytest.raises(ValueError):
        system.decide({"x": 1, "ok": True})


def test_correlated_pair_uses_the_existing_combiner_instead_of_boolean_all_any():
    cat = catalog()

    @cat.check
    def other(ok: bool) -> bool:
        return ok

    model = fitted(("weak", "other"), correlated=True)
    q = Question("q", "Choose", Answer.choice(["A", "B"]), requires=["weak", "other"])
    system = System(cat, [q], check_models={"q": model})
    result = system.ask({"x": 1, "ok": True})["q"]
    expected = model.posterior(.7, {"weak": "agree", "other": "agree"})
    assert result.confidence == pytest.approx(expected)
    naive = fitted(("weak", "other"))
    assert result.confidence < naive.posterior(.7, {"weak": "agree", "other": "agree"})


def test_different_fitted_model_changes_identity_and_cannot_replay_old_result():
    first = fitted()
    system = System(catalog(), [question()], check_models={"q": first})
    response = system.ask({"x": 1, "ok": True})
    second = fitted().fit([{"weak": "agree"}] * 10, [False] * 10)
    changed = System(catalog(), [question()], check_models={"q": second})
    assert system.fingerprint()["questions"] != changed.fingerprint()["questions"]
    assert not response.trace.replay(changed)["ok"]


def test_stored_response_roundtrip_preserves_the_posterior(tmp_path):
    from solvi.core.store import JSONLStorage

    path = tmp_path / "responses.jsonl"
    system = System(catalog(), [question()], check_models={"q": fitted()}, storage=str(path))
    result = system.ask({"x": 1, "ok": True})
    restored = JSONLStorage(path, system=system).get(result.stored_id)
    assert restored["q"].confidence == result["q"].confidence
    assert restored["q"].extra == result["q"].extra
    assert restored.trace.replay(system)["ok"]


def test_error_is_a_modeled_failure_but_supplied_check_value_is_not_evidence():
    cat = Catalog()

    @cat.check
    def weak(ok: bool) -> bool:
        raise ValueError("unavailable")

    @cat.rule("q")
    def choose(x: int):
        return Decision("A", {"A": .7, "B": .3})

    model = fitted()
    system = System(cat, [question()], check_models={"q": model})
    result = system.ask({"x": 1, "ok": True})["q"]
    assert result.confidence == pytest.approx(model.posterior(.7, {"weak": "fail"}))
    with pytest.raises(ValueError, match="given fact cannot stand in"):
        system.ask({"x": 1, "weak": True})


def test_unsupported_text_input_fails_before_calling_reader():
    system = System(catalog(), [question()], check_models={"q": fitted()})

    def reader(*args, **kwargs):
        raise AssertionError("reader must not run")

    with pytest.raises(ValueError, match="structured"):
        system.ask_text("A", decider=reader)


def test_validation_rejects_unfitted_models_and_hard_sources():
    with pytest.raises(ValueError, match="fit before"):
        System(catalog(), [question()], check_models={"q": NaiveCombiner(["weak"])})
    with pytest.raises(ValueError, match="declared question"):
        System(catalog(), [question()], check_models={"missing": fitted()})
    cat = catalog(hard=True)
    q = Question("q", "Choose", Answer.choice(["A", "B"]), requires=["permitted"])
    with pytest.raises(ValueError, match="soft checks"):
        System(cat, [q], check_models={"q": fitted(("permitted",))})


def test_joint_constraints_and_declared_self_checks_are_refused():
    cat = catalog()

    @cat.constraint
    def only_a(q):
        return q == "A"

    with pytest.raises(ValueError, match="joint constraints"):
        System(cat, [question()], check_models={"q": fitted()})
    cat = catalog()
    cat.parts["weak"].shares_call_with = "answer:q"
    with pytest.raises(ValueError, match="self-check"):
        System(cat, [question()], check_models={"q": fitted()})


@pytest.mark.parametrize("error", [False, True])
def test_forced_and_failed_hard_checks_do_not_receive_soft_evidence(error):
    cat = catalog()

    @cat.check(hard=True, then={"q": "B"})
    def permitted(allowed: bool) -> bool:
        if error:
            raise ValueError("hard check unavailable")
        return allowed

    system = System(cat, [question()], check_models={"q": fitted()}, early_exit=False)
    result = system.ask({"x": 1, "ok": True, "allowed": False})["q"]
    assert result.status == ("abstain" if error else "forced")
    assert result.confidence == (0.0 if error else 1.0)
    assert not result.extra or "check_evidence" not in result.extra


def test_calibration_runs_after_check_likelihoods_and_replay_agrees():
    from solvi.core.system import _platt

    model = fitted()
    system = System(catalog(), [question()], check_models={"q": model})
    system.calib["q"] = (1.0, -.5)
    result = system.ask({"x": 1, "ok": False})
    posterior = model.posterior(.7, {"weak": "disagree"})
    assert result["q"].extra["check_evidence"]["posterior"] == posterior
    assert result["q"].confidence == pytest.approx(_platt(posterior, 1.0, -.5))
    assert result.trace.replay(system)["ok"]


@pytest.mark.parametrize("chained", [False, True])
def test_shared_primary_dependency_cannot_be_reused_as_evidence(chained):
    cat = Catalog()

    @cat.fn
    def primary_fact(x: int) -> int:
        return x

    @cat.rule("q")
    def choose(primary_fact: int):
        return Decision("A", {"A": .7, "B": .3})

    @cat.check(shares_call_with="primary_fact")
    def other(primary_fact: int) -> bool:
        return primary_fact > 0

    @cat.check(shares_call_with="other" if chained else "primary_fact")
    def weak(ok: bool) -> bool:
        return ok

    with pytest.raises(ValueError, match="self-check"):
        System(cat, [question()], check_models={"q": fitted()})


def test_rescore_revalidates_mutated_configuration_and_recorded_primary_inputs():
    system = System(catalog(), [question()], check_models={"q": fitted()})
    response = system.ask({"x": 1, "ok": True})
    system.questions["q"].requires.clear()
    with pytest.raises(ValueError, match="requires"):
        system.answers_of(response.trace, ["q"], response.flow)
    system.questions["q"].requires.append("weak")
    primary = next(r for r in response.trace.records if r.name == "answer:q")
    primary.inputs["weak"] = "recorded dependency"
    with pytest.raises(ValueError, match="already be read"):
        system.answers_of(response.trace, ["q"], response.flow)


def test_boundary_input_rename_keeps_registered_check_names():
    system = System(catalog(), [question()], check_models={"q": fitted()})
    renamed = system.rename(inputs={"ok": "accepted"}, answers={"q": "new_q"})
    result = renamed.ask({"x": 1, "accepted": True})
    assert result["new_q"].confidence == system.ask({"x": 1, "ok": True})["q"].confidence
    assert result.trace.replay(renamed)["ok"]


def test_shared_call_through_source_alias_is_not_new_evidence():
    cat = Catalog()

    @cat.fn
    def primary_fact(x: int) -> int:
        return x

    cat.alias("reused", "primary_fact", type=int)

    @cat.rule("q")
    def choose(primary_fact: int):
        return Decision("A", {"A": .7, "B": .3})

    @cat.check(shares_call_with="reused")
    def weak(reused: int) -> bool:
        return reused > 0

    with pytest.raises(ValueError, match="self-check"):
        System(cat, [question()], check_models={"q": fitted()})
