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


def test_raw_observe_without_assignment_cannot_advance(tmp_path):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    for unit in phases[0]['units']:
        p.observe(r.change_id,dict(rollout='A',phase=0,unit=unit,version=r.version if s._arm(unit,.5) else s.baseline,loss=0))
    with pytest.raises(ValueError,match='prior assignment'):s.checkpoint()
    assert p.journal.records('rollout_decision')==[]

@pytest.mark.parametrize('corruption',['version','unit','loss','change','assignment','duplicate'])
def test_raw_observation_provenance_is_revalidated(tmp_path,corruption):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    unit='0';version=s.route(unit);assignment=p.journal.records('rollout_assignment')[-1]
    effect=dict(rollout='A',phase=0,unit=unit,version=version,loss=0,assignment=assignment['id'])
    change_id=r.change_id
    if corruption=='version':effect['version']=100
    if corruption=='unit':effect['unit']='outside'
    if corruption=='loss':effect['loss']=float('nan')
    if corruption=='assignment':effect['assignment']='not-the-assignment'
    if corruption=='change':change_id='different-change'
    # Direct raw journal event deliberately bypasses public validation.
    p.journal.write('observed',change=change_id,observed=effect)
    if corruption=='duplicate':p.journal.write('observed',change=change_id,observed=effect)
    with pytest.raises(ValueError):s.checkpoint()
    assert p.journal.records('rollout_decision')==[]

@pytest.mark.parametrize('field,value',[('phase',True),('phase',-1),('phase',10),('phase','0'),('version',100),('version',True),('unit','foreign'),('unit',None)])
def test_recovered_assignment_fields_fail_before_route_and_checkpoint(tmp_path,field,value):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    fields=dict(rollout='A',phase=0,unit='0',version=r.version if s._arm('0',.5) else s.baseline)
    fields[field]=value
    p.journal.write('rollout_assignment',**fields)
    with pytest.raises(ValueError,match='recovered'):s.route('0')
    with pytest.raises(ValueError,match='recovered'):s.checkpoint()

@pytest.mark.parametrize('sequence',[True,-1,10000,'1'])
def test_recovered_assignment_sequence_and_hash_integrity(tmp_path,sequence):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    s.route('0')
    p.journal.records('rollout_assignment')[-1]['seq']=sequence
    with pytest.raises(ValueError):s.route('0')
    with pytest.raises(ValueError):s.checkpoint()


def test_duplicate_recovered_assignment_refused_not_last_wins(tmp_path):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    version=s.route('0')
    p.journal.write('rollout_assignment',rollout='A',phase=0,unit='0',version=version)
    with pytest.raises(ValueError):s.route('0')
    with pytest.raises(ValueError):s.checkpoint()


def test_raw_decision_cannot_skip_unlabelled_phase(tmp_path):
    p,r,phases=setup(tmp_path);s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    p.journal.write('rollout_decision',rollout='A',phase=0,decision='advance',excess_loss=0,labels=20)
    with pytest.raises(ValueError,match='complete prior'):s.route('0')


def test_two_phase_decisions_replay_against_prior_complete_labels(tmp_path):
    p,r,phases=setup(tmp_path)
    phases.append(dict(share=.5,units=[str(i) for i in range(20,40)],max_excess_loss=0))
    s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    for phase in phases:
        for unit in phase['units']:s.route(unit);s.observe(unit,0,by='gold')
        assert s.checkpoint()['decision']=='advance'
        s=StagedTraffic(p,r.version,rollout_id='A',phases=phases)
    with pytest.raises(ValueError):s.route('39')
