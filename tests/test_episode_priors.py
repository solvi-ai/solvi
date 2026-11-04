import copy
from dataclasses import FrozenInstanceError
import pytest
from solvi.stat import EpisodePriors,SourceTrust


def add(p,id,**kw):
    args=dict(episode='past',context='war',source_type='AI',source='A',role='origin',loss=0,asserted_at=4,ended_at=5,known_at=6,now=10)
    args.update(kw);p.record(id,**args)

def get(p,**kw):
    args=dict(current_episode='current',context='war',source_type='AI',source='A',role='origin',now=10);args.update(kw);return p.get(**args)

def test_analytic_beta_source_type_context_base_and_current_trust_minimum():
    p=EpisodePriors(window=20,max_strength=4)
    add(p,'a');add(p,'b',loss=1)
    assert get(p).prior==(2,2) and get(p).level=='source'
    assert get(p,source='B').level=='type'
    assert get(p,source_type='human').level=='context'
    assert get(p,context='peace').prior==(1,1) and get(p,context='peace').level=='base'
    assert get(p,role='repeater').level=='base'
    t=SourceTrust(window=20,prior=get(p).prior,min_labels=2)
    assert t.opinion('A',now=10).support=='unknown'  # prior never fabricates current labels
    with pytest.raises(FrozenInstanceError):get(p).level='x'

def test_cap_and_episode_exclusion_no_currentdata_double_count():
    p=EpisodePriors(window=20,max_strength=3)
    for n in range(5):add(p,str(n))
    v=get(p);assert sum(v.prior)==pytest.approx(3)
    assert v.prior[0]/sum(v.prior)==pytest.approx(6/7)
    assert get(p,current_episode='past').prior==(1,1)
    add(p,'current',episode='current',loss=1)
    assert get(p).prior==v.prior
    assert 'current' not in get(p).record_ids

def test_old_late_labels_future_asof_and_identity_windows():
    p=EpisodePriors(window=5)
    add(p,'old',asserted_at=0,ended_at=1,known_at=10)
    assert get(p).level=='base'  # late label does not rejuvenate old assertion
    add(p,'recent',asserted_at=7,ended_at=8,known_at=9)
    assert get(p).record_ids==('recent',)
    assert get(p,now=8).record_ids==()
    assert get(p,now=13).record_ids==()
    before=p.fingerprint()
    with pytest.raises(ValueError):add(p,'future',known_at=11)
    with pytest.raises(ValueError):add(p,'recent')
    assert p.fingerprint()==before

def test_minimum_backoff_and_detached_strict_atomic_restore():
    p=EpisodePriors(window=20,min_records=2)
    add(p,'a');add(p,'b',source='B',loss=1)
    assert get(p).level=='type'
    state=p.snapshot();q=EpisodePriors.restore(state)
    assert q.fingerprint()==p.fingerprint() and get(q)==get(p)
    state['records']['a']['loss']=1
    assert q.snapshot()['records']['a']['loss']==0
    bad=copy.deepcopy(q.snapshot());bad['records']['b']['known_at']=0
    before=q.fingerprint()
    with pytest.raises(ValueError):EpisodePriors.restore(bad)
    assert q.fingerprint()==before
    bad=q.snapshot();bad['records']['a']['extra']=1
    with pytest.raises(ValueError):EpisodePriors.restore(bad)
