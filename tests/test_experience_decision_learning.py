"""Known-answer decision loss, explicit admission and logged-policy evaluation."""
from fractions import Fraction
import copy

import pytest

from solvi.experimental.experience._common import WorkBudget, BudgetStop, digest
from solvi.experimental.experience.decision_learning import (
    DecisionAction, DecisionContext, DecisionObjective, DecisionPolicy, DecisionLearner,
    LoggedFeedback, OutcomeFeedback, fit_decision_policy, decision_regret, evaluate_logged_policy,
)


def context(x=1, name="c", forbidden=False):
    return DecisionContext(name, "v1", (DecisionAction("a", (x, 1), allowed=not forbidden), DecisionAction("b", (-x, 1))))


def rows(split="train"):
    return [OutcomeFeedback(f"{split}-{x}", "oracle-v1", context(x, f"{split}-{x}"), (("a", 3*x), ("b", -3*x)), split)
            for x in (-2, -1, 1, 2)]


@pytest.mark.parametrize("loss", ["decision", "prediction"])
def test_real_learning_and_held_out(loss):
    budget = WorkBudget(100000)
    trained = fit_decision_policy(rows(), budget, loss=loss)
    assert trained.policy.weights[0] > 0
    held = OutcomeFeedback("test", "oracle-v1", context(-5, "held"), (("a", -15), ("b", 15)), "evaluation")
    assert decision_regret(trained.policy, held, budget) == 0
    assert trained.policy.choose(context(1, forbidden=True), budget) == "b"
    assert budget.spent > 0


def test_outcome_cost_can_change_preferred_action():
    c = DecisionContext("cost", "v1", (DecisionAction("a", (10,), cost=9), DecisionAction("b", (5,))))
    assert DecisionPolicy((1,)).choose(c, WorkBudget(100)) == "b"
    assert DecisionPolicy((1,), DecisionObjective(cost_weight=0)).choose(c, WorkBudget(100)) == "a"


def test_admission_explicit_activation_and_revocation():
    learner = DecisionLearner(WorkBudget(100000))
    candidate = learner.train(rows())
    assert learner.choose(context()) is None
    receipt = learner.admit(candidate, rows("validation"), DecisionPolicy((0, 0)))
    assert receipt["accepted"] and learner.choose(context()) is None
    learner.activate(receipt["receipt"])
    assert learner.choose(context(-1)) == "b"
    learner.revoke(candidate.fingerprint)
    assert learner.choose(context()) is None
    with pytest.raises(ValueError):
        learner.activate(receipt["receipt"])


def test_rejected_validation_and_split_leak():
    learner = DecisionLearner(WorkBudget(100000))
    candidate = learner.train(rows())
    with pytest.raises(ValueError):
        learner.admit(candidate, rows(), DecisionPolicy((0, 0)))
    leaky = [OutcomeFeedback("new", "oracle", rows()[0].context, rows()[0].outcomes, "validation")]
    with pytest.raises(ValueError):
        learner.admit(candidate, leaky, DecisionPolicy((0, 0)))


def test_snapshot_requires_current_host_authority_and_paid_transfer():
    learner = DecisionLearner(WorkBudget(100000))
    c = learner.train(rows())
    receipt = learner.admit(c, rows("validation"), DecisionPolicy((0, 0)))
    learner.activate(receipt["receipt"])
    state = learner.snapshot()
    before = learner.budget.spent
    restored = DecisionLearner.restore(state, learner.budget, authority=learner.authority)
    assert restored.budget.spent > before and restored.choose(context(-1)) == "b"
    with pytest.raises(ValueError, match="stale"):
        learner.choose(context())
    with pytest.raises(ValueError):
        DecisionLearner.restore(state, restored.budget, authority=restored.authority)
    clean = restored.snapshot()
    bad = copy.deepcopy(clean)
    bad["active"] = None
    bad["digest"] = digest({k:v for k,v in bad.items() if k != "digest"})
    with pytest.raises(ValueError):
        DecisionLearner.restore(bad, restored.budget, authority=restored.authority)


def logs():
    c = context()
    return [LoggedFeedback(str(i), c, action, reward, (("a", (1,2)), ("b", (1,2))), "randomized")
            for i, (action,reward) in enumerate((("a", 4),("b", 2),("a", 4),("b", 2)))]


def test_ips_snips_dr_ess_known_answer():
    data = logs()
    key = data[0].context.fingerprint
    out = evaluate_logged_policy(data, {key:{"a":[1,1],"b":[0,1]}}, WorkBudget(1000),
                                 reward_model={key:{"a":4,"b":2}})
    assert out.status == "estimated"
    assert out.ips == out.snips == out.dr == Fraction(4)
    assert out.effective_sample_size == 2 and out.max_weight == 2
    assert out.to_dict()["dr"] == [4,1]


def test_off_policy_absence_of_support_or_unknown_propensity_is_not_estimate():
    c = context()
    data = [LoggedFeedback("x", c, "a", 4, (("a",(1,1)),("b",(0,1))), "logger")]
    out = evaluate_logged_policy(data, {c.fingerprint:{"a":0,"b":1}}, WorkBudget(100))
    assert out.status == "unsupported" and out.ips is None
    data = [LoggedFeedback("x", c, "a", 4, (("a",(1,1)),("b",(0,1))), "logger", False)]
    out = evaluate_logged_policy(data, {c.fingerprint:{"a":1,"b":0}}, WorkBudget(100))
    assert out.status == "unidentified"


def test_ess_refusal_no_fake_confidence():
    out = evaluate_logged_policy(logs()[:1], {context().fingerprint:{"a":1,"b":0}}, WorkBudget(100))
    assert out.status == "insufficient" and out.ips is out.snips is out.dr is None


@pytest.mark.parametrize("bad", [True, 0.1, "1", 10**10])
def test_strict_features(bad):
    with pytest.raises(ValueError):
        DecisionAction("a", (bad,))


def test_no_execution_at_zero_budget_and_bounded_iterators():
    with pytest.raises(BudgetStop):
        fit_decision_policy(rows(), WorkBudget(0))
    consumed = []
    def infinite():
        while True:
            consumed.append(1)
            yield rows()[0]
    with pytest.raises(ValueError):
        fit_decision_policy(infinite(), WorkBudget(10000))
    assert len(consumed) == 4097


def test_logging_probability_and_hard_constraints():
    c = context(forbidden=True)
    with pytest.raises(ValueError):
        LoggedFeedback("x", c, "a", 10, (("a",(1,2)),("b",(1,2))), "source")
    with pytest.raises(ValueError):
        LoggedFeedback("x", context(), "a", 10, (("a",(0,1)),("b",(1,1))), "source")


def test_trusted_anchor_cross_process_and_no_ledger_refund():
    from solvi.experimental.experience.decision_learning import DecisionAuthority
    learner = DecisionLearner(WorkBudget(100000))
    candidate = learner.train(rows())
    rec = learner.admit(candidate, rows("validation"), DecisionPolicy((0, 0)))
    learner.activate(rec["receipt"])
    snapshot = learner.snapshot()
    anchor = learner.authority.export_trusted_state()
    budget = WorkBudget.restore(learner.budget.snapshot())
    authority = DecisionAuthority.from_trusted_state(anchor, budget=budget)
    with pytest.raises(ValueError, match="refund"):
        DecisionLearner.restore(snapshot, WorkBudget(100000), authority=authority)
    restored = DecisionLearner.restore(snapshot, budget, authority=authority)
    assert restored.choose(context(-2)) == "b"


def test_restore_cannot_refund_post_snapshot_shared_work():
    learner = DecisionLearner(WorkBudget(100000))
    candidate = learner.train(rows())
    state = learner.snapshot()
    old_ledger = WorkBudget.restore(learner.budget.snapshot())
    learner.budget.charge("other-shared-work", 500)
    with pytest.raises(ValueError, match="refund"):
        DecisionLearner.restore(state, old_ledger, authority=learner.authority)
    assert candidate.policy.weights


def test_restore_untrusted_structure_is_paid_before_copy():
    learner = DecisionLearner(WorkBudget(100))
    state = {"huge": [None]*4096}
    with pytest.raises(BudgetStop):
        DecisionLearner.restore(state, WorkBudget(1), authority=learner.authority)
