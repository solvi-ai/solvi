"""Explicit model proposal fact, no question rebinding or evaluation-label selection."""
import pytest

from solvi import Answer,Catalog,Question,System
from test_rule_list_quality import data


def make():
    cat=Catalog()
    @cat.fn
    def proposal(text, confidence, flipped):
        value='n' if 'north' in text else 's'
        if flipped:value='s' if value=='n' else 'n'
        return {'value':value,'confidence':confidence}
    s=System(cat,[Question('q','?',Answer.choice(['n','s']))])
    train=[({**st,'confidence':.8,'flipped':False},a) for st,a in data('train')]
    hold=[({**st,'confidence':.8,'flipped':False},a) for st,a in data('hold')]
    rl=s.learn_rule('q',train,features=['text'],calibrated=True,calibration=hold,confirm='proposal')
    return s,rl


def test_model_agreement_is_required_and_never_upgrades_source_confidence():
    s,rl=make()
    agree=s.ask({'text':'north fresh','confidence':.2,'flipped':False},store=False)
    assert agree['q'].answer=='n' and agree['q'].confidence==pytest.approx(.2)
    assert agree.trace.replay(s,agree.flow)['ok']
    conflict=s.ask({'text':'north fresh','confidence':.8,'flipped':True},store=False)
    assert conflict['q'].status=='abstain' and 'disagree' in conflict['q'].why
    assert rl.quality[0]['n']>0


def test_disagreeing_or_unscored_calibration_does_not_create_positive_support():
    s,rl=make()
    for proposal in [None, {'value':'s','confidence':.8}, {'value':'n','confidence':float('nan')}]:
        rl.calibrate([{'text':f'north held{i}','proposal':proposal} for i in range(8)],['n']*8)
        assert rl.decision({'text':'north fresh','proposal':{'value':'n','confidence':.8}}).escalate
    train=data('refit')
    rl.fit([st for st,_ in train],[a for _,a in train])
    assert s.ask({'text':'north fresh','confidence':.8,'flipped':False},store=False)['q'].status=='abstain'


def test_no_question_name_is_implicitly_bound_as_confirmation():
    s=System(Catalog(),[Question('q','?',Answer.choice(['n','s']))])
    with pytest.raises(ValueError,match='explicit proposal fact'):
        s.learn_rule('q',data('train'),features=['text'],calibrated=True,confirm='q')
