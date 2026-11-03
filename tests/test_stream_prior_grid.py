"""Past calibration scores fix only a future grid: no transferred labels, wealth or delta."""
import copy
import math
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import StreamPromise

from test_stream_dispatch import _s1, _s2
from solvi.core.dispatch import AskPath, Dispatcher


@pytest.mark.parametrize("bad", [[], (), [True], [math.inf], [math.nan], [".9"], [1j], [0.] * 4097, iter([.9])])
def test_invalid_prior_is_atomic(bad):
    promise = StreamPromise(max_risk=.1)
    before = copy.deepcopy(promise.dump())
    with pytest.raises(ValueError):
        promise.prime_grid(bad, source="past")
    assert promise.dump() == before and promise.budget.v == 0


def test_prior_roundtrip_has_no_bets_and_binds_actual_grid():
    promise = StreamPromise(max_risk=.1, grid_size=2)
    fp = promise.fingerprint()
    promise.prime_grid([.6, .7, .9], source="independent pilot from another model")
    assert promise.fingerprint() == fp and promise.n == promise.seen == promise.dropped == 0
    assert promise.v is None and promise.budget.v == 0 and not promise._buffer
    assert promise.grid == [.9, .6] and all(p.log_wealth == 0 for p in promise._proc)
    restored = StreamPromise(max_risk=.1, grid_size=2).restore(promise.dump())
    assert restored.dump() == promise.dump()
    tampered = copy.deepcopy(promise.dump())
    tampered["grid"][0] = .99
    with pytest.raises(ValueError, match="evidence_fp"):
        StreamPromise(max_risk=.1, grid_size=2).restore(tampered)
    promise.learn(.8, "yes", "yes", sampling_policy="complete")
    restored.learn(.8, "yes", "yes", sampling_policy="complete")
    assert restored.dump() == promise.dump()
    old = StreamPromise(max_risk=.1).dump()
    old.pop("grid_origin")
    assert StreamPromise(max_risk=.1).restore(old).grid_origin is None


def calibrated(count=200):
    system = _s1()
    examples = [({"p": .6 + i / 1000}, "billing") for i in range(count)]
    system.guarantee("intent", examples, max_risk=.1)
    return system


def test_ordinary_calibration_to_dispatcher_uses_first_future_label():
    system = calibrated()
    promise = StreamPromise(max_risk=.1)
    system.guarantee("intent", promise=promise)
    assert promise.grid_origin["count"] == 200
    assert promise.n == promise.seen == promise.dropped == 0 and promise.budget.v == 0
    dispatcher = Dispatcher(system, AskPath(_s2()), audit="adaptive", promise=promise, secret="prior-test")
    dispatcher.ask({"p": .8, "gold": "billing"})
    assert promise.n == 1 and promise.dropped == 0 and promise.budget.v == 1


def test_cache_adds_no_inference_and_unknown_or_active_grid_is_unchanged(monkeypatch):
    calls = []
    for cached in (True, False):
        system = calibrated()
        if not cached:
            del system.guards["intent"].promise._prior_grid_scores
        original = system.ask
        count = []

        def ask(*args, **kw):
            count.append(1)
            return original(*args, **kw)

        monkeypatch.setattr(system, "ask", ask)
        promise = StreamPromise(max_risk=.1)
        system.guarantee("intent", promise=promise)
        calls.append(len(count))
        assert (promise.grid is not None) is cached
    assert calls[0] == calls[1]
    for promise in (StreamPromise(max_risk=.1, grid="fixed"), StreamPromise(max_risk=.1)):
        system = calibrated()
        if promise.grid is None:
            promise.observe(.8)
        before = promise.dump()
        system.guarantee("intent", promise=promise)
        assert promise.dump() == before


def test_empty_source_or_used_auto_grid_refuses_without_changing_state():
    promise = StreamPromise(max_risk=.1)
    before = promise.dump()
    with pytest.raises(ValueError, match="source"):
        promise.prime_grid([.9], source=" ")
    assert promise.dump() == before
    promise.observe(.9)
    before = promise.dump()
    with pytest.raises(ValueError, match="fresh"):
        promise.prime_grid([.9], source="past")
    assert promise.dump() == before


def test_answer_descriptor_mismatch_keeps_live_warmup_fallback():
    system = calibrated()
    promise = StreamPromise(max_risk=.1)
    system.guarantee("intent", promise=promise, answer="billing")
    assert promise.grid is None and promise.grid_origin is None


def test_grouped_ordinary_calibration_does_not_supply_a_prior():
    system = _s1()
    examples = [({"p": .6 + i / 1000}, "billing") for i in range(100)]
    system.guarantee("intent", examples, max_risk=.1, groups="answer")
    assert not hasattr(system.guards["intent"].promise, "_prior_grid_scores")
    promise = StreamPromise(max_risk=.1)
    system.guarantee("intent", promise=promise)
    assert promise.grid is None


@pytest.mark.parametrize("count,grid_from", [(199, 200), (200, 201), (200, 4097)])
def test_automatic_prior_never_shrinks_the_requested_pilot(count, grid_from):
    system = calibrated(count)
    promise = StreamPromise(max_risk=.1, grid_from=grid_from)
    system.guarantee("intent", promise=promise)
    assert promise.grid is None and promise.grid_origin is None
    assert promise.n == promise.seen == promise.dropped == 0
    assert promise.budget.v == 0


def test_automatic_prior_uses_exact_first_requested_scores_in_original_order():
    from solvi.core.calibration import ltt_grid

    system = calibrated(300)
    prior = system.guards["intent"].promise._prior_grid_scores
    promise = StreamPromise(max_risk=.1, grid_from=200)
    system.guarantee("intent", promise=promise)
    expected = sorted(map(float, ltt_grid(prior[:200], promise.grid_size)), reverse=True)
    all_cached = sorted(map(float, ltt_grid(prior, promise.grid_size)), reverse=True)
    assert promise.grid == expected and promise.grid != all_cached
    assert promise.grid_origin["count"] == 200


def test_explicit_prime_grid_is_a_deliberate_prior_not_automatic_pilot():
    promise = StreamPromise(max_risk=.1, grid_from=200)
    promise.prime_grid([.8], source="explicit one-point fixed prior")
    assert promise.grid == [.8] and promise.grid_origin["count"] == 1
    assert promise.v is None and promise.budget.v == 0


def test_fold_calibration_cannot_supply_an_automatic_prior(monkeypatch):
    from solvi.core.guarantees import guarantee

    # Exercise the fold exclusion contract without fitting any real model; row collection and installation are real.
    monkeypatch.setattr(guarantee, "_fold_heads", lambda system, question, examples, folds, seed: [None] * len(examples))
    system = _s1()
    examples = [({"p": .6 + i / 1000}, "billing") for i in range(200)]
    system.guarantee("intent", examples, max_risk=.1, folds=2)
    assert not hasattr(system.guards["intent"].promise, "_prior_grid_scores")
    promise = StreamPromise(max_risk=.1)
    system.guarantee("intent", promise=promise)
    assert promise.grid is None and promise.grid_origin is None
