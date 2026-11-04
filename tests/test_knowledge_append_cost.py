"""Original S384 chain: linear resolver work; general resolver remains the oracle."""
import pytest
import solvi.core.knowledge.store as module


@pytest.mark.parametrize('size', [10, 20, 40])
def test_plain_chain_resolves_at_most_two_vertices_per_append(monkeypatch, size):
    original = module._resolve
    visits = []
    def counted(items, comp, *args, **kwargs):
        visits.append(len(comp))
        return original(items, comp, *args, **kwargs)
    monkeypatch.setattr(module, '_resolve', counted)
    store = module.KnowledgeStore()
    for i in range(size):
        store.add('fact', {'s': 'K', 'r': 'price', 'o': i}, source='outcome', by='w', valid=(i, None))
    assert sum(visits) <= 2 * size
    assert visits == [1] + [2] * (size - 1)


@pytest.mark.parametrize('extra', ['none', 'crosswriter', 'bounded', 'old_since', 'repeat', 'derived', 'retract'])
def test_append_and_fallback_match_full_resolver(extra):
    fast, reference = module.KnowledgeStore(), module.KnowledgeStore()
    def apply(operation):
        operation(fast)
        reference._append_chains.clear()  # force the pre-optimization resolver
        operation(reference)
        assert fast.state == reference.state
        assert fast.fingerprint() == reference.fingerprint()
        for at in [None, -.5] + [i + .5 for i in range(11)]:
            query = {'about': 'K', 'relation': 'price'}
            if at is not None:
                query['valid_at'] = at
            assert fast.snapshot(**query) == reference.snapshot(**query)
    for i in range(5):
        apply(lambda s, i=i: s.add('fact', {'s': 'K', 'r': 'price', 'o': i}, source='outcome', by='w', valid=(i, None)))
    if extra == 'retract':
        first = next(i for i, item in fast.items.items() if item.body['o'] == 0)
        apply(lambda s: s.retract(first, by='audit'))
    elif extra != 'none':
        kw = {'source': 'outcome', 'by': 'other' if extra == 'crosswriter' else 'w',
              'valid': (2 if extra == 'old_since' else 5, 7 if extra == 'bounded' else None)}
        value = 0 if extra == 'repeat' else 9
        if extra == 'derived':
            kw['derived_from'] = [next(iter(fast.items))]
        apply(lambda s: s.add('fact', {'s': 'K', 'r': 'price', 'o': value}, **kw))
    apply(lambda s: s.add('fact', {'s': 'K', 'r': 'price', 'o': 10}, source='outcome', by='w', valid=(10, None)))
    rebuilt = fast.rebuild()
    assert rebuilt.state == fast.state and rebuilt.fingerprint() == fast.fingerprint()


@pytest.mark.parametrize('mode', ['batch', 'branch', 'writer_gate', 'clock'])
def test_nonplain_configuration_uses_general_resolution(mode):
    kwargs = {'writers': {'w': 'outcome'}} if mode == 'writer_gate' else {'now': 2} if mode == 'clock' else {}
    store = module.KnowledgeStore(**kwargs)
    for i in range(4):
        store.add('fact', {'s': 'K', 'r': 'r', 'o': i}, source='outcome', by='w', valid=(i, None))
    if mode in ('writer_gate', 'clock'):
        assert not store._append_chains
    if mode == 'batch':
        with store.batch():
            for i in [4, 5]:
                store.add('fact', {'s': 'K', 'r': 'r', 'o': i}, source='outcome', by='w', valid=(i, None))
    if mode == 'branch':
        store = store.branch()
        store.add('fact', {'s': 'K', 'r': 'r', 'o': 5}, source='outcome', by='w', valid=(5, None))
    assert store.rebuild().state == store.state
    assert store.rebuild().fingerprint() == store.fingerprint()


def test_dependency_on_an_older_version_disables_later_chain_fast_path():
    store = module.KnowledgeStore()
    first = store.add('fact', {'s': 'K', 'r': 'r', 'o': 0}, source='outcome', by='w', valid=(0, None))
    store.add('fact', {'s': 'child', 'r': 'r', 'o': 1}, source='spec', by='w', derived_from=[first])
    assert store.items[first].key not in store._append_chains
    for i in range(1, 4):
        store.add('fact', {'s': 'K', 'r': 'r', 'o': i}, source='outcome', by='w', valid=(i, None))
    assert store.rebuild().state == store.state


def test_date_chains_in_interleaved_scopes_match_replay():
    store = module.KnowledgeStore()
    for i in range(1, 12):
        date = f'2022-01-{i:02d}'
        for scope in [{'region': 'EU'}, {'region': 'US'}]:
            store.add('fact', {'s': 'K', 'r': 'r', 'o': i}, scope=scope,
                      source='outcome', by='w', valid=(date, None))
        replay = store.rebuild()
        assert replay.state == store.state and replay.fingerprint() == store.fingerprint()
        for j in range(1, i + 1):
            query = dict(about='K', relation='r', scope={'region': 'EU'}, valid_at=f'2022-01-{j:02d}')
            assert store.snapshot(**query) == replay.snapshot(**query)
    branch = store.branch()
    assert branch._next_n == store._next_n + 1  # branch appends its lineage note
    assert branch.journal[-1]["op"] == "note"
    assert branch.state == store.state and branch.fingerprint() == store.fingerprint()
    assert branch.verify() and store.verify()


def test_custom_gate_subclass_does_not_establish_plain_chain():
    class Custom(module.ConsistencyGate):
        pass
    store = module.KnowledgeStore(gates=[Custom()])
    for i in range(4):
        store.add('fact', {'s': 'K', 'r': 'r', 'o': i}, source='outcome', by='w', valid=(i, None))
    assert not store._append_chains
