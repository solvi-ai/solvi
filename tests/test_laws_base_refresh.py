"""An explicit unbound set_base call refreshes even a reused mutable callable."""
import pickle
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from solvi.experimental.laws import Buffer, Candidate, Grammar, LawSet, Transition


class MutableBase:
    def __init__(self):
        self.value = 0

    def __call__(self, state, action):
        return self.value


@pytest.mark.parametrize('memo', [False, True])
def test_same_callable_explicit_refresh_shared_buffer_and_pickle(memo):
    g = Grammar([], k=0)
    b = Buffer(g)
    b.add(Transition(0, 'a', 1))
    base = MutableBase()
    first = LawSet(g, base=base, memo=memo)
    second = LawSet(g, base=base, memo=memo)
    candidate = Candidate('a', (), 1)
    assert first.score(candidate, b) == (1, 1, 1)
    assert second.score(candidate, b) == (1, 1, 1)
    before = b.tested, first.cost.copy(), first.laws.copy(), first.retired.copy()
    base.value = 1
    first.set_base(base)
    assert first.score(candidate, b) == (1, 1, 0)
    assert (b.tested, first.cost, first.laws, first.retired) == before
    # Refreshing another owner cannot reuse its prior view of this shared buffer.
    second.set_base(base)
    assert second.score(candidate, b) == (1, 1, 0)
    restored = pickle.loads(pickle.dumps(first))
    restored.base.value = 0
    restored.set_base(restored.base)
    assert restored.score(candidate, b) == (1, 1, 1)
    assert first.score(candidate, b) == (1, 1, 0)
    b.add(Transition(1, 'a', 0))
    assert first.score(candidate, b) == (2, 1, 0)
    assert first.score_many([candidate], b, stable=True) == [first.score(candidate, b)]
