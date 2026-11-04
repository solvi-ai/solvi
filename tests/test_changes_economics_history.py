import pytest
from solvi.experimental.changes import Change, ChangeJournal, Context, Economics, ProposerHistory, Pipeline, StageVerdict

def change():return Change('person',payload={'kind':'fact','body':{'s':'a','r':'b','o':1}},proposer='p')
def ctx(j):return Context(None,None,j,'candidate')

def test_economics_stops_before_expensive_callback_and_fingerprints_utility():
    calls=[]
    class Expensive:
        name='expensive'
        def run(self,c,x):calls.append(1);return StageVerdict.refuse('bad')
    e=Economics(lambda c,x:dict(gain=3,loss=1),costs={'expensive':2},units='USD')
    p=Pipeline([e,Expensive()],storage=None)
    assert p.propose(change()).status=='rejected' and calls==[]
    a=e.run(change(),ctx(p.journal)).measured['policy_fingerprint']
    e2=Economics(lambda c,x:dict(gain=5,loss=1),costs={'expensive':2},units='USD')
    assert e2.run(change(),ctx(p.journal)).measured['policy_fingerprint']!=a
    assert Pipeline([e2,Expensive()],storage=None).propose(change()).status=='rejected' and calls==[1]

@pytest.mark.parametrize('value',[True,-1,float('nan'),float('inf')])
def test_invalid_cost(value):
    with pytest.raises(ValueError):Economics(lambda c,x:dict(gain=1,loss=0),costs={'s':value},units='u')

def test_labelled_history_restart_expiry_and_no_admission_as_truth(tmp_path):
    path=tmp_path/'history.jsonl';j=ChangeJournal(path);clock=[10]
    h=ProposerHistory(window=5,clock=lambda:clock[0])
    assert h.run(change(),ctx(j)).ok
    j.write('promoted',proposer='p')
    assert h.run(change(),ctx(j)).measured['labels']==0
    h.label(j,proposer='p',loss=1,observed_at=9,evidence='independent:test1',event_id='e1')
    assert not h.run(change(),ctx(ChangeJournal(path))).ok
    with pytest.raises(ValueError):h.label(j,proposer='p',loss=0,observed_at=11,evidence='x',event_id='future')
    with pytest.raises(ValueError):h.label(j,proposer='p',loss=0,observed_at=9,evidence='x',event_id='e1')
    clock[0]=20
    assert h.run(change(),ctx(j)).ok  # expired bad evidence does not cause eternal cold-start refusal
    assert ProposerHistory(window=5,unknown='hold',clock=lambda:20).run(change(),ctx(j)).decision=='hold'


def test_good_history_never_bypasses_fresh_bad_candidate():
    j=ChangeJournal(None);h=ProposerHistory(window=5,clock=lambda:10)
    h.label(j,proposer='p',loss=0,observed_at=9,evidence='external',event_id='e')
    calls=[]
    class Hard:
        name='hard'
        def run(self,c,x):calls.append(1);return StageVerdict.refuse('fresh bad candidate')
    p=Pipeline([h,Hard()],storage=j.storage)
    assert p.propose(change()).status=='rejected' and calls==[1]
