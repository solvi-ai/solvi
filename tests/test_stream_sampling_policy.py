"""The direct learn API validates a sampling declaration, not an unobserved caller design."""
import copy
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from solvi.experimental.stream import StreamPromise


@pytest.mark.parametrize('grid', [[0.0], 'auto'])
@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('policy', ['input_dependent', 'unknown', None, [], {}])
def test_rejected_sampling_declaration_is_atomic(grid, existing, policy):
    promise = StreamPromise(max_risk=.05, delta=.05, grid=grid)
    if existing:
        promise.observe(.9)
        promise.learn(.9, 'a', 'a', sampling_policy='complete')
    before = copy.deepcopy(promise.dump())
    budget = copy.deepcopy(promise.budget.state())
    with pytest.raises(ValueError, match='sampling'):
        promise.learn(.9, 'a', None, sampling_policy=policy)
    assert promise.dump() == before
    assert promise.budget.state() == budget


def test_explicit_uniform_and_complete_have_identical_state():
    promises = [StreamPromise(max_risk=.05, delta=.05, grid=[.8, .5]) for _ in range(3)]
    for n in range(600):
        score = .9 if n % 3 else .6
        label = None if n % 101 == 0 else ('b' if n % 71 == 0 else 'a')
        floor = .8 if n % 4 == 0 else None
        for promise in promises:
            promise.observe(score)
        states = [promises[0].learn(score, 'a', label, floor, sampling_policy='complete'),
                  promises[1].learn(score, 'a', label, floor, sampling_policy='uniform'),
                  promises[2].learn(score, 'a', label, floor, sampling_policy='complete')]
        assert states[0] == states[1] == states[2]
        assert promises[0].dump() == promises[1].dump() == promises[2].dump()
        assert promises[0].budget.state() == promises[1].budget.state() == promises[2].budget.state()
