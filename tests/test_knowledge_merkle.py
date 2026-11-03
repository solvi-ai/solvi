"""Independent full-tree reference and state-transition coverage for km2."""
import hashlib
import json
import random

from solvi.core.knowledge import KnowledgeStore
from solvi.core.knowledge import _merkle
from solvi.core.runtime import _canon, vhash


def full_digest(domain, value):
    return hashlib.sha256(json.dumps([domain, _canon(value)], sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def reference(rows):
    if not rows:
        return full_digest('solvi.km2.empty', None)
    key = min(rows, key=lambda key: (full_digest('solvi.km2.priority', key), key))
    return full_digest('solvi.km2.node', [key, full_digest('solvi.km2.leaf', rows[key]),
                      reference({k: v for k, v in rows.items() if k < key}),
                      reference({k: v for k, v in rows.items() if k > key})])


def check(store):
    rows = {}
    for i, st in store.state.items():
        it = store.items[i]
        rows[i] = [i, it.kind, it.body, it.scope, st['status'], st['rank'], st['n_confirm'],
                   st['refuted_by'], st['learned_at'], st['invalidated_at']]
    assert store.fingerprint() == 'km2:sha256:' + reference(rows)
    assert store.legacy_fingerprint() == vhash({'items': [rows[i] for i in sorted(rows)]})[:16]


def test_merkle_permutations_delete_reinsert():
    rows = {str(i): ['payload', i] for i in range(40)}
    for seed in range(4):
        keys = list(rows)
        random.Random(seed).shuffle(keys)
        root = None
        for key in keys:
            root = _merkle.put(root, key, _merkle.digest('solvi.km2.leaf', rows[key]))
        assert root.hash == reference(rows)
        for key in keys[::2]:
            root = _merkle.put(root, key, None)
        remaining = {k: v for k, v in rows.items() if k not in keys[::2]}
        assert root.hash == reference(remaining)
        for key in reversed(keys[::2]):
            root = _merkle.put(root, key, _merkle.digest('solvi.km2.leaf', rows[key]))
        assert root.hash == reference(rows)


def test_state_transitions_batch_restart_expiry_retest(tmp_path):
    store = KnowledgeStore(str(tmp_path / 'journal.jsonl'))
    check(store)
    with store.batch():
        store.add('fact', {'s': 's', 'r': 'r', 'o': 1}, source='spec', by='a', reconfirm_after=2)
        other = store.add('fact', {'s': 't', 'r': 'r', 'o': 2}, source='spec', by='a')
    check(store)
    store.tick(3)
    check(store)
    store.retract(other, why='withdraw')
    check(store)
    store.add('fact', {'s': 't', 'r': 'r', 'o': 2}, source='spec', by='a')
    assert store.pending_retest(other)
    check(store)
    store.retest(other, ok=True)
    check(store)
    flag = store.flag(ids=[other], why='review')
    check(store)
    store.end_flag(flag)
    check(store)
    assert store.rebuild().fingerprint() == store.fingerprint()
    restarted = KnowledgeStore(str(tmp_path / 'journal.jsonl'))
    check(restarted)
    assert restarted.fingerprint() == store.fingerprint()
    assert restarted.verify()


def test_noop_events_do_not_hash_and_update_hash_count_is_bounded(monkeypatch):
    store = KnowledgeStore()
    for i in range(40):
        store.add('fact', {'s': str(i), 'r': 'r', 'o': i}, source='spec', by='a')
    calls = []
    original = _merkle.digest
    def count(domain, value):
        calls.append(domain)
        return original(domain, value)
    monkeypatch.setattr(_merkle, 'digest', count)
    old = store.fingerprint()
    store.note(message='no belief change')
    store.used(list(store.items)[:1])
    store.tick(1)
    assert store.fingerprint() == old
    assert calls == []
    store.add('fact', {'s': 'new', 'r': 'r', 'o': 100}, source='spec', by='a')
    assert calls.count('solvi.km2.leaf') == 1
    assert calls.count('solvi.km2.node') < 20
    check(store)
