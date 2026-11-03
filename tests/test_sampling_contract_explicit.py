"""Required sampling declarations prevent accidental biased manual certificate input."""
import json
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from solvi.experimental.stream import StreamPromise


def frozen(p):
    return json.dumps([p.dump(), p.budget.state()], sort_keys=True).encode()


@pytest.mark.parametrize('grid', [[0.0], 'auto'])
@pytest.mark.parametrize('existing', [False, True])
def test_omission_and_input_dependent_are_atomic(grid, existing):
    p = StreamPromise(max_risk=.05, delta=.05, grid=grid)
    if existing:
        p.observe(.9)
        p.learn(.9, 'a', 'a', sampling_policy='uniform')
    before = frozen(p)
    with pytest.raises(TypeError, match='sampling_policy'):
        p.learn(.9, 'a', None)
    assert frozen(p) == before
    with pytest.raises(ValueError, match='input_dependent'):
        p.learn(.9, 'a', None, sampling_policy='input_dependent')
    assert frozen(p) == before


def test_uniform_complete_and_restored_valid_updates_have_identical_state():
    ps = [StreamPromise(max_risk=.05, delta=.05, grid=[.8, .5]) for _ in range(3)]
    for n in range(600):
        score = .9 if n % 3 else .6
        label = None if n % 101 == 0 else ('b' if n % 71 == 0 else 'a')
        floor = .8 if n % 4 == 0 else None
        if n == 300:
            ps[2].restore(ps[0].dump())
            before = frozen(ps[2])
            with pytest.raises(TypeError, match='sampling_policy'):
                ps[2].learn(score, 'a', label)
            assert frozen(ps[2]) == before
        for p, policy in zip(ps, ['uniform', 'complete', 'uniform']):
            p.observe(score)
            p.learn(score, 'a', label, floor, sampling_policy=policy)
        assert frozen(ps[0]) == frozen(ps[1]) == frozen(ps[2])


def test_slice_sampling_witness_has_low_observed_but_high_population_risk():
    # Bounded two-slice S4 mechanism: sampling hard rows 100x less hides their errors.
    easy_n, hard_n = 7000, 3000
    easy_errors, hard_errors = 140, 600
    population_risk = (easy_errors + hard_errors) / (easy_n + hard_n)
    sampled_risk = (easy_errors + hard_errors / 100) / (easy_n + hard_n / 100)
    assert sampled_risk < .05 < population_risk
    p = StreamPromise(max_risk=.05, delta=.05, grid=[0])
    before = frozen(p)
    with pytest.raises(ValueError, match='input_dependent'):
        p.learn(1, 'a', 'a', sampling_policy='input_dependent')
    assert frozen(p) == before


def test_dispatcher_system_owned_sampling_passes_its_declaration():
    from test_stream_handoff_oct30 import _systems
    from solvi.core.dispatch import AskPath, Dispatcher
    system, gold = _systems()
    p = StreamPromise(max_risk=.05, delta=.05, grid=[0.0])
    system.guarantee('intent', promise=p, signal='msp')
    d = Dispatcher(system, AskPath(gold), question='intent', audit='adaptive',
                   promise=p, secret=b'explicit-sampling-contract')
    d.ask({'pred': 'a', 'msp': .9, 'gold': 'a'})
    assert p.n == 1
