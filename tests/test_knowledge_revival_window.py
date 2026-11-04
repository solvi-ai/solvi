import pytest
from solvi.core.knowledge import KnowledgeStore


def tell(st,value=1,valid=(0,10),**kw):
    return st.add('fact',{'s':'x','r':'v','o':value},source='outcome',by='a',valid=valid,**kw)


def test_any_period_inclusive_window_and_fresh_after_expiry():
    st=KnowledgeStore(revival_window=5)
    i=tell(st);st.tick(10);st.retract(i);st.tick(5)
    tell(st,valid=(100,110))
    assert i in st.pending() and not st.usable(i,valid_at=105)
    # Time passing alone cannot revive the held claim.
    st.tick(1)
    assert not st.usable(i,valid_at=105)
    tell(st,valid=(200,210))
    assert st.usable(i,valid_at=205)
    assert not st.usable(i,valid_at=105)
    assert st.verify() and st.rebuild().fingerprint()==st.fingerprint()
    child=st.branch();assert child.revival_window==5 and child.fingerprint()==st.fingerprint()


def test_default_period_policy_unchanged_and_zero_window():
    st=KnowledgeStore();i=tell(st);st.retract(i);st.tick(100);tell(st)
    assert i in st.pending()
    st=KnowledgeStore(revival_window=0);i=tell(st);st.retract(i);tell(st,valid=(30,40))
    assert i in st.pending()
    st.tick(1);tell(st,valid=(50,60));assert st.usable(i,valid_at=55)


@pytest.mark.parametrize('value',[True,-1,float('nan'),float('inf'),'5',10**500])
def test_invalid_window(value):
    with pytest.raises(ValueError):KnowledgeStore(revival_window=value)


def test_reopen_recorded_clocks_and_prefix(tmp_path):
    path=tmp_path/'k.jsonl';st=KnowledgeStore(path,revival_window=2)
    i=tell(st);st.tick(10);st.retract(i);st.tick(2);tell(st,valid=(20,30))
    prefix=len(st.journal);st.tick(1);tell(st,valid=(40,50))
    reopened=KnowledgeStore(path,revival_window=2)
    assert reopened.usable(i,valid_at=45) and not reopened.usable(i,valid_at=25)
    old=st.rebuild(upto=prefix)
    assert not old.usable(i,valid_at=25) and old.revival_window==2 and old.verify()


def test_flags_retirement_and_other_keys_not_waived():
    st=KnowledgeStore(revival_window=1);i=tell(st);st.retract(i);st.tick(2);tell(st,valid=(20,30))
    assert st.usable(i,valid_at=25)
    st.flag(ids=[i],by='audit')
    assert not st.usable(i,valid_at=25)
    st.retire(i)
    assert not st.usable(i,valid_at=25)
    other=st.add('fact',{'s':'other','r':'v','o':1},source='outcome',by='a',valid=(20,30))
    assert st.usable(other,valid_at=25)
    assert st.verify()


def test_native_knowledge_forwards_policy_and_batch_rebuild():
    from solvi import Knowledge
    k=Knowledge(revival_window=1);st=k.store
    with st.batch():
        i=tell(st);st.retract(i);st.tick(1);tell(st,valid=(10,20));st.tick(1);tell(st,valid=(30,40))
    assert not st.usable(i,valid_at=15) and st.usable(i,valid_at=35)
    assert st.verify() and st.rebuild().fingerprint()==st.fingerprint()


def test_fresh_window_claim_does_not_clear_old_derived_support_taint():
    st=KnowledgeStore(revival_window=1)
    i=tell(st,valid=(0,None))
    child=st.add('fact',{'s':'child','r':'v','o':1},source='outcome',by='reader',
                 derived_from=[i],valid=(0,None))
    st.retract(i);st.tick(2);tell(st,valid=(0,None))
    assert st.usable(i,valid_at=10)
    assert child in st.pending() and not st.usable(child,valid_at=10)
    assert st.verify()


def test_standalone_inferred_already_rejected_and_explicit_premises_get_no_waiver():
    st=KnowledgeStore(revival_window=1);i=tell(st);st.retract(i);st.tick(2)
    before=(len(st.journal),st.fingerprint())
    with pytest.raises(ValueError,match='premises'):
        tell(st,inferred=True)
    assert (len(st.journal),st.fingerprint())==before
    p=st.add('fact',{'s':'premise','r':'v','o':1},source='outcome',by='a')
    tell(st,inferred=True,derived_from=[p])
    assert '_revival_cut' not in st.items[i].asserts[-1]
    assert st.verify()


def test_default_resolver_does_not_scan_window_vouches(monkeypatch):
    import solvi.core.knowledge.store as mod
    def forbidden(*_):raise AssertionError('disabled policy must not scan live claims')
    monkeypatch.setattr(mod,'_window_vouched',forbidden)
    st=KnowledgeStore();i=tell(st);st.retract(i);tell(st)
    assert i in st.pending() and st._needs_retest(i)


def test_local_window_eligibility_consistent_with_needs_retest():
    st=KnowledgeStore(revival_window=1);i=tell(st);st.retract(i);st.tick(2);tell(st)
    assert st.usable(i,valid_at=5) and not st._needs_retest(i)
