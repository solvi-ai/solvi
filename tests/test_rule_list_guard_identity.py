"""A learned quality update cannot silently inherit an installed threshold."""

from test_rule_list_quality import data,make


def install_stale_case():
    s,rl=make([(st,'s') for st,_ in data('hold')])
    s.guarantee('q',[(st,'s') for st,_ in data('third')],signal='act',max_error=.1,
                method='empirical',min_support=1)
    assert s.ask({'text':'north fresh'},store=False)['q'].status=='abstain'
    return s,rl


def test_quality_mutation_does_not_silently_reuse_old_native_guarantee():
    s,rl=install_stale_case()
    old=s.guards['q']
    hold=data('newhold')
    rl.calibrate([st for st,_ in hold],[a for _,a in hold])
    result=s.ask({'text':'north fresh'},store=False)
    assert result['q'].status=='abstain' and 'stale learned-rule guarantee' in result['q'].why
    assert s.guards['q'] is old
    s.guarantee('q',data('freshthird'),signal='act',max_error=.1,method='empirical',min_support=1)
    assert s.ask({'text':'north fresh'},store=False)['q'].answer=='n'


def test_refit_and_rule_replacement_require_a_fresh_guarantee():
    s,rl=install_stale_case()
    fresh=data('refit')
    rl.fit([st for st,_ in fresh],[a for _,a in fresh])
    rl.calibrate([st for st,_ in data('quality2')],[a for _,a in data('quality2')])
    assert s.ask({'text':'north fresh'},store=False)['q'].status=='abstain'
    s.learn_rule('q',data('replacement'),features=['text'],calibrated=True,calibration=data('quality3'))
    assert s.ask({'text':'north fresh'},store=False)['q'].status=='abstain'


def test_saved_confidence_guard_preserves_binding_even_when_strict_override_used(tmp_path):
    s,rl=make(data('hold'))
    s.guarantee('q',data('third'),signal='confidence',max_error=.1,method='empirical',min_support=1)
    path=tmp_path/'guard.json'
    s.save_guarantee('q',path)
    s.guarantee('q',False)
    s.load_guarantee(path)
    assert s.ask({'text':'north fresh'},store=False)['q'].answer=='n'
    hold=data('newhold')
    rl.calibrate([st for st,_ in hold],['s' for _ in hold])
    s.guarantee('q',False)
    s.load_guarantee(path,strict=False)
    assert s.ask({'text':'south fresh'},store=False)['q'].status=='abstain'


def test_reattached_shared_promise_keeps_old_binding_until_new_labelled_calibration():
    s,rl=make(data('hold'))
    s.guarantee('q',data('third'),signal='confidence',max_error=.1,method='empirical',min_support=1)
    promise=s.guards['q'].promise
    binding=promise.report['learned_rule_model']
    rows=data('newhold')
    rl.calibrate([st for st,_ in rows],['s' for _ in rows])
    s.guarantee('q',False)
    s.guarantee('q',None,promise=promise,signal='confidence')
    assert promise.report['learned_rule_model']==binding
    assert s.ask({'text':'south fresh'},store=False)['q'].status=='abstain'
    s.guarantee('q',[(st,'s') for st,_ in data('freshthird')],signal='confidence',max_error=.1,
                method='empirical',min_support=1)
    assert s.guards['q'].promise is not promise
    assert s.guards['q'].promise.report['learned_rule_model']!=binding
    assert promise.report['learned_rule_model']==binding


def test_renamed_and_parameter_views_use_the_owned_learned_model_binding():
    s,rl=make(data('hold'))
    s.guarantee('q',data('third'),signal='confidence',max_error=.1,method='empirical',min_support=1)
    renamed=s.rename(answers={'q':'renamed'})
    view=s.with_params(text='south fresh')
    assert renamed.ask({'text':'north fresh'},store=False)['renamed'].answer=='n'
    assert view.ask({},store=False)['q'].answer=='s'
    rows=data('newhold')
    rl.calibrate([st for st,_ in rows],['s' for _ in rows])
    assert renamed.ask({'text':'south fresh'},store=False)['renamed'].status=='abstain'
    assert view.ask({},store=False)['q'].status=='abstain'


def test_cached_shared_catalog_and_model_proposal_identity_stay_fail_closed():
    from solvi import Answer,Catalog,Question,System
    def examples(prefix,version):
        return [({**st,'proposal':{'value':a,'confidence':.8,'identity':version}},a) for st,a in data(prefix)]
    s=System(Catalog(),[Question('q','?',Answer.choice(['n','s']))],cache=8)
    rl=s.learn_rule('q',examples('train','v1'),features=['text'],calibrated=True,
                    calibration=examples('hold','v1'),confirm='proposal')
    s.guarantee('q',examples('third','v1'),signal='act',max_error=.1,method='empirical',min_support=1)
    state={'text':'north fresh','proposal':{'value':'n','confidence':.8,'identity':'v1'}}
    old=s.ask(state,store=False)
    assert old['q'].answer=='n' and s.ask(state,store=False)['q'].answer=='n'
    state['proposal']['identity']='v2'
    assert s.ask(state,store=False)['q'].status=='abstain'
    changed=examples('quality2','v2')
    rl.calibrate([st for st,_ in changed],[a for _,a in changed])
    assert s.ask(state,store=False)['q'].status=='abstain'
    assert not old.trace.replay(s,old.flow)['ok']
    # The native catalog's actual model, not a side table, owns a shared rule.
    shared=System(s.catalog,[Question('q','?',Answer.choice(['n','s']))])
    shared.guarantee('q',examples('third2','v2'),signal='confidence',max_error=.1,method='empirical',min_support=1)
    assert shared.ask(state,store=False)['q'].answer=='n'
    rl.calibrate([st for st,_ in examples('quality3','v3')],[a for _,a in examples('quality3','v3')])
    state['proposal']['identity']='v3'
    assert shared.ask(state,store=False)['q'].status=='abstain'


def test_unbound_native_guard_fingerprint_is_unchanged():
    from solvi.core.guarantees.guarantee import QuestionGuard,calibrate
    from solvi.core.provenance import digest
    promise=calibrate([.2,.9],[False,True],max_error=.1,method='empirical',min_support=1)
    guard=QuestionGuard('q',promise,signal='confidence')
    assert guard.fingerprint()==digest('QuestionGuard','q','confidence',repr(None),None,promise.fingerprint())
