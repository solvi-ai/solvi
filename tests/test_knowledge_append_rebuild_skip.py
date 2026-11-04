"""Rebuild(skip=) legitimately replays assertions whose support was omitted."""
import pytest

from solvi.core.knowledge import KnowledgeStore


@pytest.mark.parametrize('dependency', ['derived_from', 'consulted'])
def test_append_witness_invalidation_tolerates_an_omitted_parent(dependency):
    store = KnowledgeStore()
    parent = store.add('fact', {'s': 'parent', 'r': 'value', 'o': 1},
                       source='outcome', by='writer', valid=(0, None))
    child = store.add('fact', {'s': 'child', 'r': 'value', 'o': 2},
                      source='spec', by='writer', **{dependency: [parent]})
    before = store.head(), store.fingerprint()
    omitted = store.rebuild(skip={parent})
    withdrawn = store.rebuild()
    withdrawn.retract(parent)
    assert parent not in omitted.items and child in omitted.items
    if dependency == 'derived_from':
        assert omitted.state == withdrawn.state
        assert omitted.fingerprint() == withdrawn.fingerprint()
    else:
        # Missing consulted evidence is not an observed withdrawal event.
        assert omitted.status(child) == 'active'
    assert omitted.rebuild().state == omitted.state
    # Reintroducing a previously omitted parent must not certify its key as
    # dependency-free: later writes still need to revisit the old child.
    restored = omitted.add('fact', {'s': 'parent', 'r': 'value', 'o': 1},
                           source='outcome', by='writer', valid=(0, None))
    assert restored == parent
    assert omitted.items[parent].key not in omitted._append_chains
    omitted.add('fact', {'s': 'parent', 'r': 'value', 'o': 3},
                source='outcome', by='writer', valid=(1, None))
    assert omitted.rebuild().state == omitted.state
    assert omitted.rebuild().fingerprint() == omitted.fingerprint()
    assert (store.head(), store.fingerprint()) == before
    assert store.verify()
