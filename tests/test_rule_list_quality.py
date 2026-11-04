"""Independent caller-heldout quality, native Decision/act and stale calibration."""
import copy

import pytest

from solvi import Answer,Catalog,Question,System


def data(prefix,n=8):
    return [({'text':f'{kind} {prefix}{i}'},answer) for kind,answer in [('north','n'),('south','s')] for i in range(n)]


def make(calibration=None,min_confidence=None):
    s=System(Catalog(),[Question('q','?',Answer.choice(['n','s']),min_confidence=min_confidence)])
    rl=s.learn_rule('q',data('train'),features=['text'],calibrated=True,calibration=calibration)
    return s,rl


def test_heldout_disagreements_lower_actual_confidence_without_changing_structure():
    s,rl=make()
    structure=copy.deepcopy(rl.rules)
    assert s.ask({'text':'north fresh'},store=False)['q'].status=='abstain'
    hold=data('hold')
    rl.calibrate([st for st,_ in hold],['s' for _ in hold])
    assert rl.rules==structure
    s.questions['q'].min_confidence=.7
    low=s.ask({'text':'north fresh'},store=False)
    high=s.ask({'text':'south fresh'},store=False)
    assert low['q'].status=='abstain' and low['q'].confidence==pytest.approx(.1)
    assert high['q'].answer=='s' and high['q'].confidence==pytest.approx(.9)
    rec=next(r for r in high.trace.records if r.name=='answer:q')
    assert rec.extra['act']==pytest.approx(.9)
    assert high.trace.replay(s,high.flow)['ok']


def test_calibrated_rule_native_act_guarantee_uses_separate_third_set():
    hold=data('hold')
    s,rl=make([(st,'s') for st,_ in hold])
    report=s.guarantee('q',[(st,'s') for st,_ in data('guarantee')],signal='act',max_error=.1,
                       method='empirical',min_support=1)
    assert report['threshold']>.1
    assert s.ask({'text':'north unseen'},store=False)['q'].status=='abstain'
    assert s.ask({'text':'south unseen'},store=False)['q'].answer=='s'


def test_refit_invalidates_quality_fingerprint_and_old_trace():
    s,rl=make(data('hold'))
    old=s.ask({'text':'north fresh'},store=False)
    fp=rl.fingerprint()
    train=data('refit')
    rl.fit([st for st,_ in train],[a for _,a in train])
    assert rl.quality is None and rl.fingerprint()!=fp
    assert s.ask({'text':'north fresh'},store=False)['q'].status=='abstain'
    assert not old.trace.replay(s,old.flow)['ok']
