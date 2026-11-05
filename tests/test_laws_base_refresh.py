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


def test_refresh_removes_only_owned_entries_and_does_not_retain_buffers():
    import gc
    import weakref
    g = Grammar([], k=0)
    b = Buffer(g)
    b.add(Transition(0, 'a', 1))
    base = MutableBase()
    ls, other = LawSet(g, base=base), LawSet(g, base=base)
    c = Candidate('a', (), 1)
    other.score(c, b)
    other_key = ('a', other._base_key, None)
    saved = b._cache[other_key]
    for i in range(100):
        base.value = i % 2
        ls.set_base(base)
        ls.score(c, b)
        base_entries = [key for key in b._cache if key[1][0] == 'base']
        assert len(base_entries) == 2
        assert b._cache[other_key] == saved
    restored = pickle.loads(pickle.dumps(ls))
    assert len(restored._base_cache_buffers) == 0
    restored.score(c, b)
    restored.set_base(restored.base)
    assert b._cache[other_key] == saved
    reference = weakref.ref(b)
    del b
    gc.collect()
    assert reference() is None
    assert len(ls._base_cache_buffers) == len(other._base_cache_buffers) == 0


def test_bound_current_revision_keeps_existing_key_and_cache():
    g = Grammar([], k=0)
    b = Buffer(g)
    b.add(Transition(0, 'a', 1))
    base = MutableBase()
    ls = LawSet(g, base=base, lapse_on_base_change=True, base_version='v1')
    ls.score(Candidate('a', (), 1), b)
    key, cache = ls._base_key, b._cache.copy()
    assert ls.set_base(base, version='v1') == []
    assert ls._base_key == key and b._cache == cache
