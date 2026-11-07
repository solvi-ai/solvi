"""Independent adversarial review probes for decision learning."""
from fractions import Fraction

import pytest

from solvi.experimental.experience._common import WorkBudget, digest
from solvi.experimental.experience.decision_learning import (
    DecisionAction, DecisionContext, DecisionLearner, DecisionObjective, DecisionPolicy,
    LoggedFeedback, OutcomeFeedback, decision_regret, evaluate_logged_policy,
)


def context(name):
    return DecisionContext(name, "v1", (DecisionAction("a", (0,)), DecisionAction("b", (1,))))


def feedback(name, split):
    return OutcomeFeedback("event-" + name, "host", context(name), (("a", 0), ("b", 9)), split)


def admitted():
    learner = DecisionLearner(WorkBudget(1000000))
    candidate = learner.train([feedback("train", "train")])
    receipt = learner.admit(candidate, [feedback("validation", "validation")], DecisionPolicy((0,)))["receipt"]
    learner.activate(receipt)
    return learner, candidate, receipt


def test_review_recomputed_digest_cannot_resurrect_revoked_candidate():
    learner, candidate, receipt = admitted()
    learner.revoke(candidate.fingerprint)
    snapshot = learner.snapshot()
    snapshot["revoked"] = []
    snapshot["active"] = candidate.fingerprint
    snapshot["digest"] = digest({k: v for k, v in snapshot.items() if k != "digest"})
    try:
        restored = DecisionLearner.restore(snapshot, learner.budget, authority=learner.authority)
    except ValueError:
        return
    assert restored.choose(context("new")) is None


def test_review_live_state_must_remain_restorable():
    learner = DecisionLearner(WorkBudget(1000000))
    for i in range(129):
        try:
            learner.train([feedback("train-" + str(i), "train")], epochs=1, steps=(1,))
        except ValueError:
            break
    DecisionLearner.restore(learner.snapshot(), learner.budget, authority=learner.authority)


def test_review_logged_formulas_exact_and_cost_adjusted():
    c = context("logged")
    rows = [LoggedFeedback("a", c, "a", 2, (("a", (1, 2)), ("b", (1, 2))), "host"),
            LoggedFeedback("b", c, "b", 10, (("a", (1, 2)), ("b", (1, 2))), "host")]
    result = evaluate_logged_policy(rows, {c.fingerprint: {"a": (1, 4), "b": (3, 4)}},
                                    WorkBudget(1000), reward_model={c.fingerprint: {"a": 0, "b": 8}},
                                    minimum_ess=1)
    assert result.ips == Fraction(8)
    assert result.snips == Fraction(8)
    assert result.dr == Fraction(8)
    assert result.effective_sample_size == Fraction(8, 5)


def test_review_hard_feasibility_and_actual_decision_loss():
    c = DecisionContext("hard", "1", (DecisionAction("bad", (100,), allowed=False),
                                       DecisionAction("a", (0,)), DecisionAction("b", (1,), cost=3)))
    row = OutcomeFeedback("e", "host", c, (("bad", 1000), ("a", 2), ("b", 10)))
    policy = DecisionPolicy((1,), DecisionObjective())
    assert policy.choose(c, WorkBudget(1000)) == "a"
    assert decision_regret(policy, row, WorkBudget(1000)) == 5


def test_review_leakage_rejected_by_context_even_different_event():
    learner = DecisionLearner(WorkBudget(10000))
    candidate = learner.train([feedback("same", "train")])
    row = OutcomeFeedback("different-event", "host", context("same"), (("a", 0), ("b", 9)), "validation")
    with pytest.raises(ValueError, match="overlaps"):
        learner.admit(candidate, [row], DecisionPolicy((0,)))


def test_review_unknown_propensity_and_unsupported_policy_refuse():
    c = context("logged")
    row = LoggedFeedback("a", c, "a", 2, (("a", (1, 1)), ("b", (0, 1))), "host")
    target = {c.fingerprint: {"a": 0, "b": 1}}
    assert evaluate_logged_policy([row], target, WorkBudget(100), minimum_ess=1).status == "unsupported"
    row = LoggedFeedback("a", c, "a", 2, (("a", (1, 1)), ("b", (0, 1))), "host", False)
    assert evaluate_logged_policy([row], target, WorkBudget(100), minimum_ess=1).status == "unidentified"


def test_review_authority_transfer_invalidates_old_live_copy_and_replay():
    learner, candidate, receipt = admitted()
    snapshot = learner.snapshot()
    restored = DecisionLearner.restore(snapshot, learner.budget, authority=learner.authority)
    assert restored.choose(context("new")) == "b"
    with pytest.raises(ValueError, match="stale"):
        learner.choose(context("new"))
    with pytest.raises(ValueError, match="stale"):
        DecisionLearner.restore(snapshot, restored.budget, authority=restored.authority)
    restored.revoke(candidate.fingerprint)
    snapshot = restored.snapshot()
    current = DecisionLearner.restore(snapshot, restored.budget, authority=restored.authority)
    assert current.choose(context("new")) is None
    with pytest.raises(ValueError):
        current.activate(receipt)
