"""The stable statistics path shares state and arithmetic with legacy users."""
import copy
import pickle
import subprocess
import sys

import pytest

from solvi.stat import Certificate, DeltaBudget, EProcess, MeanBound, MirrorBound, Mixture


def test_legacy_paths_are_the_same_classes():
    from solvi.experimental._evalue import MeanBound as OldMean
    from solvi.experimental.skills._certificate import Certificate as OldCert, EProcess as OldProcess
    from solvi.experimental.stream._eprocess import DeltaBudget as OldBudget, Mixture as OldMixture
    assert (OldMean, OldCert, OldProcess, OldBudget, OldMixture) == (
        MeanBound, Certificate, EProcess, DeltaBudget, Mixture)


def test_stable_import_is_quiet_and_does_not_load_experimental():
    run = subprocess.run([sys.executable, "-Werror", "-c",
                          "import sys,solvi.stat; assert 'solvi.experimental' not in sys.modules"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_checkpoint_continues_identical_bets_and_wealth():
    p = MeanBound(.4)
    for x in [.1, .6, .9, .5]:
        p.update(x)
    state = p.dump()
    restored = MeanBound(.4).restore(state)
    pickled = pickle.loads(pickle.dumps(p))
    for x in [.2, .8, .7, .3]:
        assert p.bet() == restored.bet() == pickled.bet()
        assert p.update(x) == restored.update(x) == pickled.update(x)
    assert p.dump() == restored.dump() == pickled.dump()


def test_invalid_restore_is_atomic_and_does_not_alias_input():
    p = MeanBound(.4)
    p.update(.7)
    state = p.dump()
    for key, bad in [("m", .5), ("n", True), ("n", -1), ("s", 2),
                     ("ss", -1), ("evidence", float("nan"))]:
        invalid = copy.deepcopy(state)
        invalid[key] = bad
        with pytest.raises(ValueError):
            p.restore(invalid)
        assert p.dump() == state
    p.restore(state)
    state["s"] = 10
    assert p.dump()["s"] == .7


def test_finite_delta_budget_refuses_exhaustion_without_spending():
    p = DeltaBudget(.12, schedule="horizon", horizon=3)
    assert [p.next() for _ in range(3)] == [(1, .04), (2, .04), (3, .04)]
    before = p.state()
    assert p.spent == .12
    with pytest.raises(ValueError, match="exhausted"):
        p.next()
    assert p.state() == before
    for kwargs in [{"horizon": 3}, {"schedule": "horizon"},
                   {"schedule": "horizon", "horizon": True}]:
        with pytest.raises(ValueError):
            DeltaBudget(**kwargs)


def test_legacy_budget_pickle_without_horizon_remains_reportable():
    budget = DeltaBudget(.05)
    budget.next()
    del budget.horizon
    restored = pickle.loads(pickle.dumps(budget))
    assert restored.state()["horizon"] is None
    assert restored.state()["v"] == 1
    assert restored.next() == (2, Certificate(delta=.05).delta_at(2))


def test_stream_finite_budget_horizon_is_part_of_unstarted_evidence_identity():
    from solvi.experimental.stream import StreamPromise
    first = StreamPromise(max_risk=.1, grid="fixed", budget=DeltaBudget(.05, "horizon", horizon=3))
    second = StreamPromise(max_risk=.1, grid="fixed", budget=DeltaBudget(.05, "horizon", horizon=4))
    assert first.fingerprint() != second.fingerprint()
    with pytest.raises(ValueError):
        second.restore(first.dump())


def test_legacy_basel_and_mirrored_mean_keep_exact_arithmetic():
    cert = Certificate(delta=.05)
    budget = DeltaBudget(.05)
    for index in range(1, 8):
        assert budget.next() == (index, cert.delta_at(index))
    mirror = MirrorBound(.2)
    mean = MeanBound(.8)
    for loss in [0, .1, .9, .2, .4]:
        assert mirror.update(loss) == mean.update(1 - loss)
    mixture = Mixture(.2)
    for loss in [0, 0, .7]:
        mixture.update(loss)
    assert Mixture(.2).restore(mixture.dump()).log_wealth == mixture.log_wealth
