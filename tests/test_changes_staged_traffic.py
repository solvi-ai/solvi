import pytest
from solvi.core.store import JSONLStorage
from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.changes import Change,Pipeline,Static,StagedTraffic

def setup(tmp_path):
    p=Pipeline([Static()],JSONLStorage(tmp_path/'rollout.jsonl',fsync=True),store=KnowledgeStore())
    c=Change('person',payload={'kind':'fact','body':{'s':'new','r':'value','o':2}},proposer='owner')
    result=p.propose(c);assert result.promoted
    phases=[dict(share=.5,units=[str(i) for i in range(20)],max_excess_loss=0)]
    return p,result,phases

def test_assign_before_observe_hold_complete_and_restart(tmp_path):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    assert s.checkpoint()['decision']=='hold'
    with pytest.raises(ValueError):s.observe('0',0,by='gold')
    for unit in phases[0]['units']:
        version=s.route(unit)
        assert p.journal.records('rollout_assignment')[-1]['version']==version
        again=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
        assert again.route(unit)==version
        again.observe(unit,0,by='independent')
    assert len(p.journal.records('observed'))==20
    with pytest.raises(ValueError):s.observe('0',0,by='duplicate')
    assert s.checkpoint()['decision']=='advance'
    with pytest.raises(ValueError):s.route('0')
    assert p.journal.verify()['ok']

def test_bad_actual_variant_rolls_back_and_freeze(tmp_path):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    for u in phases[0]['units']:
        version=s.route(u);s.observe(u,int(version==r.version),by='gold')
    assert s.checkpoint()['decision']=='rollback'
    assert StagedTraffic(p,r.version,rollout_id='A',phases=phases)._phase() is None
    with pytest.raises(ValueError):s.route('0')
    changed=[dict(phases[0],share=.6)]
    with pytest.raises(ValueError):StagedTraffic(p,r.version,rollout_id='A',phases=changed)

def test_unsafe_storage_and_mutation_refused(tmp_path):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    s.phases[0]['max_excess_loss']=1
    with pytest.raises(ValueError):s.route('0')
    p.journal.storage.fsync=False
    with pytest.raises(ValueError):StagedTraffic(p,r.version,rollout_id='B',phases=phases)
