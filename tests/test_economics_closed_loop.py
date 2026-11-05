"""Offline known-answer accounting, not a historical utility-quality evaluation."""
import pytest

from solvi.core.plan.answer_policy import AnswerPolicy, CheckModel
from solvi.experimental.changes import Economics
from solvi.experimental.regime_memory import RegimeMemory


def model():
    return CheckModel().fit([
        {'answer':1.,'gold':1. if ok else 2.,'correct':ok,
         'checks':{'P':{'status':'agree' if ok else 'disagree','value':1. if ok else 2.,'correct':True}}}
        for ok in [True]*8+[False]*4])


def test_independent_outcome_counts_abstention_not_wrong_action():
    record={'answer':1.,'gold':2.,'correct':False,
            'checks':{'P':{'status':'disagree','value':2.,'correct':True}}}
    policy=AnswerPolicy(model(),checks=['P'],cost=.05,max_checks=1)
    replay=policy.replay(.8,record,3)
    assert replay.used==['P'] and replay.answer==2. and replay.calls==2
    wrong=int(replay.answered and replay.answer!=record['gold'])
    total=.1+.05*len(replay.used)+30*wrong
    assert wrong==0 and total==pytest.approx(.15)
    stopped=AnswerPolicy(model(),checks=['P'],max_checks=0).replay(.2,record,30)
    assert not stopped.answered and not (stopped.answered and stopped.answer!=record['gold'])
    assert stopped.calls==1 and stopped.used==[]
    assert AnswerPolicy(model(),checks=['P'],max_checks=0).decide(.99,{},0).kind=='answer'


def test_explicit_economics_and_delayed_regime_do_not_fabricate_outcomes():
    verdict=Economics(lambda c,x:{'gain':1.,'loss':2.},costs={'check':.05},units='utility').run(None,None)
    assert verdict.measured['expected_net']==pytest.approx(-1.05)
    m=RegimeMemory(epoch='closed-loop-known-answer',model_id='known-table-v1',
        regimes={'A':'laws-a','B':'laws-b'},prior=[.5,.5],transition=[[.9,.1],[.1,.9]],
        probes={'p':{'yes':[.8,.2],'no':[.2,.8]}},max_delay=3)
    ticket=m.issue('one','p')
    m.advance()
    assert m.observe(ticket,'yes')==pytest.approx((.74,.26))
    restored=RegimeMemory.restore(m.snapshot())
    assert restored.weights==m.weights
