import copy
import pytest
from benchmarks.compute_score import audit, gold
from benchmarks.compute_trial import trial


@pytest.mark.parametrize("family,amount,flag", [("reconcile", 1, False), ("invoice", 0, True), ("inventory", 0, True)])
def test_literal_anchors(family, amount, flag):
    assert gold(family, dict.fromkeys("abcdef", 1)) == {"amount": amount, "flag": flag}


@pytest.mark.parametrize("family", ["reconcile", "invoice", "inventory"])
@pytest.mark.parametrize("arm", ["DIRECT", "DIRTY", "FIFO", "CHEAP", "DISPATCH", "NO_CACHE", "NO_LEARN"])
def test_full_independent_score(family, arm):
    row = trial(7, family, 100000, arm)
    score = audit(row)
    assert score["correct"] == 24 and score["wrong"] == 0


def test_scorer_does_not_call_actor_or_verifier(monkeypatch):
    row = trial(7, "invoice", 100000, "DISPATCH")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("actor forbidden during audit")
    monkeypatch.setattr("benchmarks.compute_trial.trial", forbidden)
    monkeypatch.setattr("benchmarks.compute_trial.reference", forbidden)
    assert audit(row)["correct"] == 24
    row["epochs"][0]["outputs"]["amount"] += 1
    assert audit(row)["wrong"] == 1


def test_strict_types_and_tamper():
    row = trial(7, "reconcile", 100000, "DIRECT")
    row["epochs"][0]["outputs"]["flag"] = int(row["epochs"][0]["outputs"]["flag"])
    assert audit(row)["wrong"] == 1
    changed = copy.deepcopy(row)
    changed["charges"][0]["amount"] += 1
    with pytest.raises(ValueError, match="debit"):
        audit(changed)
    row["spec"]["nodes"][0]["kind"] = "add"
    with pytest.raises(ValueError, match="fixture"):
        audit(row)


@pytest.mark.parametrize("accepted,proposal,expected", [(False, 1, 0), (True, 2, 0), (True, 1, 1)])
def test_learned_count_requires_independently_correct_accepted_call(accepted, proposal, expected):
    from benchmarks.compute_score import fixture
    from solvi.experimental.experience import WorkBudget
    charges = []
    def sink(s):
        charges.append({"kind": "compute_attempt", "amount": 82, **s})
    budget = WorkBudget(1000, on_charge=sink)
    budget.charge("compute_attempt", 82)
    row = {"seed": 7, "family": "invoice", "limit": 1000, "arm": "DISPATCH", **fixture(7, "invoice"),
           "ledger": budget.snapshot(), "charges": charges,
           "callbackevents": [{"kind": "blend", "args": [0, 1], "proposal": proposal, "accepted": accepted,
                               "opname": "learned:known", "cost": 28, "verify_cost": 52, "phase": "epoch:0"}],
           "epochs": [{"epoch": i, "spent": 82, "outputs": {}, "status": "budget"} for i in range(12)],
           "acquisition": {"spent": 0, "status": "disabled"}}
    assert audit(row)["learned_calls"] == expected
