import copy

import pytest

from solvi import Knowledge


FACT = {'s': 'AD', 'r': 'P35', 'o': 'Hollande'}
VALID = ('2012-01-01', '2017-01-01')


def setup(capable=True, actor='wikidata'):
    k = Knowledge(writers={'wikidata': 'outcome'}, reassert_writers={'wikidata'} if capable else ())
    k.tell({'s': 'AD', 'r': 'P35', 'o': 'Vives'}, source='outcome', by='wikidata',
           valid=('2003-01-01', None))
    h = k.tell(FACT, source='outcome', by='wikidata', valid=VALID)
    k.retract(h, by=actor, assertion=k.store.journal[-1]['n'])
    return k, h


def test_exact_trusted_original_comeback_and_replay():
    k, h = setup()
    k.tell(FACT, source='outcome', by='wikidata', valid=VALID, reassert=True)
    st = k.store
    snap = st.snapshot(about='AD', relation='P35', valid_at='2014-01-01', kinds=('fact',))
    assert Knowledge.value(snap, 'AD', 'P35', valid_at='2014-01-01') == 'Hollande'
    assert h not in st.pending()
    assert not st.items[h].promoted and not st.items[h].rechecked
    assert st.rebuild().fingerprint() == st.fingerprint()


@pytest.mark.parametrize('capable,actor,valid', [(False, 'wikidata', VALID),
                                               (True, 'audit', VALID),
                                               (True, 'wikidata', ('2012-01-01', None))])
def test_unsafe_reassert_is_atomic(capable, actor, valid):
    k, _ = setup(capable, actor)
    before = copy.deepcopy(k.store.journal)
    with pytest.raises(ValueError, match='reassert'):
        k.tell(FACT, source='outcome', by='wikidata', valid=valid, reassert=True)
    assert k.store.journal == before


@pytest.mark.parametrize('operation', ['global', 'source', 'rollback', 'failed', 'flag'])
def test_external_authority_is_never_waived(operation):
    k, h = setup()
    st = k.store
    if operation == 'flag':
        st.flag(ids=[h], by='audit')
    else:
        k.tell(FACT, source='outcome', by='wikidata', valid=VALID)
        if operation == 'global':
            st.retract(h, by='wikidata')
        elif operation == 'source':
            st.retract_source('wikidata', by='audit')
        elif operation == 'rollback':
            st.rollback(h, by='audit')
        else:
            st.retest(h, ok=False, by='audit')
    before = copy.deepcopy(st.journal), st.fingerprint()
    with pytest.raises(ValueError, match='reassert'):
        k.tell(FACT, source='outcome', by='wikidata', valid=VALID, reassert=True)
    assert (st.journal, st.fingerprint()) == before


def test_unrelated_and_ended_flags_do_not_block_exact_reassert():
    k, h = setup()
    f = k.store.flag(ids=[h], by='audit')
    k.store.end_flag(f, by='audit')
    k.store.flag(ids=['unrelated'], by='audit')
    k.tell(FACT, source='outcome', by='wikidata', valid=VALID, reassert=True)
    assert h not in k.store.pending()


def test_claim_only_vouch_preserves_foreign_pending_and_withdrawal():
    from solvi.core.knowledge.store import KnowledgeStore
    st = KnowledgeStore(writers={'wikidata': 'outcome', 'mirror': 'outcome'}, reassert_writers={'wikidata'})
    h = st.add('fact', FACT, source='outcome', by='wikidata', valid=VALID)
    st.retract(h, assertion=st.journal[-1]['n'], by='wikidata')
    st.add('fact', FACT, source='outcome', by='mirror', valid=VALID)
    st.add('fact', FACT, source='outcome', by='wikidata', valid=VALID, reassert=True)
    new = st.journal[-1]['n']
    claims = st.item(h)['valid']['claims']
    assert any(c.get('pending') for c in claims if c['by'] == 'mirror')
    assert st.rebuild().fingerprint() == st.fingerprint()
    st.retract(h, assertion=new, by='audit')
    assert h in st.pending()


def test_malformed_proof_fails_before_mutation():
    k, h = setup()
    k.tell(FACT, source='outcome', by='wikidata', valid=VALID, reassert=True)
    st = k.store
    rec = copy.deepcopy(st.journal[-1])
    rec['item'] = 'missing-item'
    rec['clock'] = 999
    before = st.fingerprint(), copy.deepcopy(st.journal), set(st.items), st.clock
    with pytest.raises(ValueError, match='reassert'):
        st._apply(rec)
    assert (st.fingerprint(), st.journal, set(st.items), st.clock) == before


def test_disk_persistence_and_asof_snapshot(tmp_path):
    path = tmp_path / 'knowledge'
    k = Knowledge(path, writers={'wikidata': 'outcome'}, reassert_writers={'wikidata'})
    h = k.tell(FACT, source='outcome', by='wikidata', valid=VALID)
    old = k.store.journal[-1]['n']
    k.retract(h, by='wikidata', assertion=old)
    before = len(k.store.journal)
    k.tell(FACT, source='outcome', by='wikidata', valid=VALID, reassert=True)
    loaded = Knowledge(path, writers={'wikidata': 'outcome'})
    assert loaded.store.fingerprint() == k.store.fingerprint()
    assert h not in loaded.store.pending()
    assert h in loaded.store.rebuild(upto=before).pending() or not loaded.store.rebuild(upto=before).usable(h)


@pytest.mark.parametrize('extra', [{'source': 'person'}, {'by': 'impostor'},
                                   {'derived_from': ['missing']}, {'scope': {'other': True}}])
def test_wrong_source_writer_derivation_or_scope_is_atomic(extra):
    k, _ = setup()
    args = dict(source='outcome', by='wikidata', valid=VALID, reassert=True)
    args.update(extra)
    before = copy.deepcopy(k.store.journal), k.store.fingerprint()
    with pytest.raises(ValueError, match='reassert'):
        k.tell(FACT, **args)
    assert (k.store.journal, k.store.fingerprint()) == before


@pytest.mark.parametrize('field,value', [('body', {'s': 'changed', 'r': 'P35', 'o': 'Hollande'}),
                                        ('scope', {'changed': True}), ('item_kind', 'rule')])
def test_replay_cannot_substitute_claim_identity(field, value):
    k, _ = setup()
    k.tell(FACT, source='outcome', by='wikidata', valid=VALID, reassert=True)
    rec = copy.deepcopy(k.store.journal[-1])
    rec[field] = value
    before = copy.deepcopy(k.store.journal), k.store.fingerprint()
    with pytest.raises(ValueError, match='reassert'):
        k.store._apply(rec)
    assert (k.store.journal, k.store.fingerprint()) == before
