"""A local retraction must not traverse witnesses of every unrelated key."""
import solvi.core.knowledge.store as module


class WitnessWork(dict):
    def __init__(self, values):
        super().__init__(values)
        self.visits = 0

    def clear(self):
        self.visits += len(self)
        return super().clear()

    def pop(self, key, *args):
        self.visits += 1
        return super().pop(key, *args)


def test_unrelated_witness_invalidation_is_constant_and_stale_chain_falls_back(monkeypatch):
    store = module.KnowledgeStore()
    ids = []
    for key in range(100):
        for version in range(3):
            ids.append(store.add('fact', {'s': str(key), 'r': 'value', 'o': version},
                                 source='outcome', by='w', valid=(version, None)))
    cache = WitnessWork(store._append_chains)
    store._append_chains = cache
    store.retract(ids[0])
    assert cache.visits <= 2
    visits = []
    resolve = module._resolve

    def counted(items, comp, *args, **kwargs):
        visits.append(len(comp))
        return resolve(items, comp, *args, **kwargs)

    monkeypatch.setattr(module, '_resolve', counted)
    store.add('fact', {'s': '99', 'r': 'value', 'o': 3}, source='outcome', by='w', valid=(3, None))
    assert visits == [4]  # an old witness may not survive global invalidation
    rebuilt = store.rebuild()
    assert store.state == rebuilt.state and store.fingerprint() == rebuilt.fingerprint()
    for at in (0, 1, 2, 3):
        assert store.snapshot(about='99', valid_at=at) == rebuilt.snapshot(about='99', valid_at=at)
