"""Traversal closure stays exact while each same-key adjacency is expanded once."""
import pytest

from solvi.core.knowledge import KnowledgeStore


class CountedSet(set):
    visits = 0

    def __iter__(self):
        for item in super().__iter__():
            type(self).visits += 1
            yield item


def reference(store, ids, direction):
    """Original traversal, deliberately expanding a key on every visited item."""
    seen, todo = set(), list(ids)
    while todo:
        iid = todo.pop()
        if iid in seen or iid not in store.items:
            continue
        seen.add(iid)
        item = store.items[iid]
        if direction in ('upstream', 'component'):
            for assertion in item.asserts:
                todo.extend(assertion['just'])
        if direction in ('downstream', 'component'):
            todo.extend(store.dependents.get(iid, ()))
        if item.key is not None:
            todo.extend(store.by_key[item.key])
    return seen


@pytest.mark.parametrize('method', ['upstream', 'downstream', 'component'])
@pytest.mark.parametrize('size', [10, 20, 40])
def test_each_key_list_is_expanded_once(method, size):
    store = KnowledgeStore()
    ids = [store.add('fact', {'s': 's', 'r': 'r', 'o': i}, source='spec', by='writer', valid=(i, None))
           for i in range(size)]
    store.by_key = {key: CountedSet(values) for key, values in store.by_key.items()}
    CountedSet.visits = 0
    assert getattr(store, method)([ids[0]]) == set(ids)
    assert CountedSet.visits == size


@pytest.mark.parametrize('method', ['upstream', 'downstream', 'component'])
def test_closure_matches_reference_across_versions_scopes_relations_and_cycles(method):
    # Traversal accepts all stored dependency graphs, including explicit cycles.
    store = KnowledgeStore(gates=[])
    def add(subject, value, relation='r', scope=None, premises=()):
        return store.add('fact', {'s': subject, 'r': relation, 'o': value}, scope=scope,
                         source='spec', by='writer', derived_from=premises, valid=(value, None))
    a = add('s', 0, scope={'map': 'a'})
    b = add('s', 1, scope={'map': 'a'})
    c = add('s', 2, scope={'map': 'b'}, premises=[a])
    d = add('s', 3, relation='other', premises=[b])
    e = add('end', 4, premises=[c, d])
    isolated = add('alone', 5)
    # Reassert an existing item with another support: dependencies must still be
    # traversed on every item even when its key has already been expanded.
    store.add('fact', {'s': 's', 'r': 'r', 'o': 0}, scope={'map': 'a'},
              source='spec', by='writer', derived_from=[e], valid=(0, None))
    for seeds in ([a], [b], [c], [d], [e], [isolated], [a, d, 'missing'], []):
        assert getattr(store, method)(seeds) == reference(store, seeds, method)
    assert isolated not in getattr(store, method)([a])
    assert store.component([a]) == {a, b, c, d, e}
    assert store.upstream([d]) == {a, b, c, d, e}
    assert store.downstream([a]) == {a, b, c, d, e}


class CountedItems(dict):
    reads = 0

    def __getitem__(self, key):
        type(self).reads += 1
        return super().__getitem__(key)


def full_scan_snapshot(store, **query):
    class FullIndex:
        def get(self, key, default=None):
            return set(store.items)
    subjects, relations = store._by_subject, store._by_relation
    try:
        store._by_subject = store._by_relation = FullIndex()
        return store.snapshot(**query)
    finally:
        store._by_subject, store._by_relation = subjects, relations


@pytest.mark.parametrize('size', [10, 20, 40])
@pytest.mark.parametrize('query', [{'about': '0'}, {'relation': 'r0'}, {'about': ['0', '1'], 'relation': 'r0'}])
def test_cached_addressed_snapshot_scans_only_candidates(size, query):
    store = KnowledgeStore()
    for i in range(size):
        store.add('fact', {'s': str(i), 'r': 'r' + str(i), 'o': i}, source='spec', by='writer')
    store.fingerprint()
    store.items = CountedItems(store.items)
    CountedItems.reads = 0
    indexed = store.snapshot(**query)
    assert CountedItems.reads == 2  # one row, one upstream vertex
    CountedItems.reads = 0
    assert full_scan_snapshot(store, **query) == indexed
    assert CountedItems.reads == size + 1


def test_snapshot_indexes_preserve_hints_scopes_taint_retraction_and_replay(tmp_path):
    from solvi.core.store import JSONLStorage
    storage = JSONLStorage(tmp_path / 'knowledge.jsonl')
    store = KnowledgeStore(storage)
    def add(value, *, subject='s', relation='r', scope=None, premises=()):
        return store.add('fact', {'s': subject, 'r': relation, 'o': value}, scope=scope,
                         source='spec', by='writer', valid=(value, None), derived_from=premises)
    first = add(0, scope={'map': 'a'})
    add(1, scope={'map': 'a'})
    other = add(2, scope={'map': 'b'})
    derived = add(3, subject='derived', premises=[other])
    add(4, relation='other')
    add(5, subject='unrelated')
    store.retract(other, why='taint premise')
    store.flag(ids=[first], why='stale historical version')
    queries = [dict(about='s'), dict(relation='r'), dict(about=['s', 'derived'], relation='r'),
               dict(about='s', relation='r', scope={'map': 'a'}, valid_at=0),
               dict(about='derived'), dict(about=[]), dict(about='missing'), dict(relation='missing')]
    for query in queries:
        assert store.snapshot(**query) == full_scan_snapshot(store, **query)
    assert store.status(derived) != 'active'
    replayed = KnowledgeStore(storage)
    for query in queries:
        assert replayed.snapshot(**query) == store.snapshot(**query)
        assert replayed.snapshot(**query) == full_scan_snapshot(replayed, **query)


def test_snapshot_unhashable_relation_filter_keeps_equality_semantics():
    store = KnowledgeStore()
    store.add('fact', {'s': 's', 'r': ['relation'], 'o': 1}, source='spec', by='writer')
    for query in ({'relation': ['relation']}, {'about': 's', 'relation': ['relation']}):
        assert store.snapshot(**query) == full_scan_snapshot(store, **query)
        assert len(store.snapshot(**query)['facts']) == 1
