"""Fixed deterministic prototype references, not historical quality evaluation.

Reference source SHA256: 6bd3e7e39ce82d76a5d0dc34d36f20a575e3040a581d501d5e0377a5ee16f30b.
Expectations computed by that archived source on records() below, 2026-11-03.
"""
import pytest

from solvi.core.plan.answer_policy import AnswerPolicy, BudgetPacer, CheckModel


def records():
    return [{'answer': 1., 'gold': 1. if ok else 2., 'correct': ok,
             'checks': {k: {'value': 1. if ok else 2., 'status': 'agree' if ok else 'disagree',
                            'correct': True} for k in ('D', 'P')}}
            for ok in [True] * 8 + [False] * 4]


def policy(**kw):
    return AnswerPolicy(CheckModel().fit(records()), **kw)


def test_original_pooled_conditional_model_and_unseen_check():
    m = policy().m
    assert m.get('D').s == pytest.approx(.980952380952381)
    assert m.get('D').t == pytest.approx(.925)
    assert m.get('D').f == pytest.approx(.025)
    assert m.pooled['wrong']['truth'] == pytest.approx(.85)
    assert m.collision == .2
    assert m.get('new').t == pytest.approx(.85)


def test_original_candidate_posterior_and_dp_reference():
    p = policy()
    state = {'D': ('val', 'x'), 'P': ('val', 'x')}
    a0, candidates, unseen = p.posterior(.8, state)
    assert a0 == pytest.approx(8.478668531090425e-05)
    assert candidates['x'] == pytest.approx(.9997691550013214)
    assert unseen == pytest.approx(.00014605831336761375)
    a = p.decide(.8, state, 30)
    assert (a.kind, a.check) == ('answer_check', 'D')
    assert a.value == pytest.approx(.9928438050409619)
    a = p.decide(.8, {}, 30)
    assert (a.kind, a.check) == ('check', 'P')
    assert a.value == pytest.approx(.8434515306122458)


def test_actual_replay_replacement_and_original_ablation():
    r = policy().replay(.8, records()[-1], 30)
    assert (r.answered, r.answer, r.correct, r.calls, r.used) == (True, 2., True, 3, ['P', 'D'])
    assert r.final.kind == 'answer_check'
    r = policy(allow_replace=False).replay(.8, records()[-1], 30)
    assert (r.answered, r.answer, r.calls, r.used) == (False, None, 2, ['P'])
    assert r.final.kind == 'abstain'


def test_distinct_candidates_failures_and_collision_normalize():
    p = policy()
    for state in ({'D': 'fail'}, {'D': ('val', 'a'), 'P': ('val', 'b')},
                  {'D': ('val', 'a'), 'P': ('val', 'a')}, {'D': 'agree'}):
        a0, pv, unseen = p.posterior(.8, state)
        assert a0 + sum(pv.values()) + unseen == pytest.approx(1)
    assert policy(cost=10).decide(.8, {}, 30).kind == 'abstain'
    assert policy(max_checks=0).decide(.8, {}, 30).kind == 'abstain'
    assert policy(units={'P': 5, 'D': 5}, max_units=4).decide(.8, {}, 30).kind == 'abstain'


@pytest.mark.parametrize('candidate_id', ['agree', 'fail', 'new1:P', 'new1:P:', 'arbitrary'])
def test_candidate_renaming_cannot_change_the_policy(candidate_id):
    p = policy()
    baseline = p.decide(.8, {'D': ('val', 'x')}, 30)
    renamed = p.decide(.8, {'D': ('val', candidate_id)}, 30)
    assert (renamed.kind, renamed.check) == (baseline.kind, baseline.check)
    assert renamed.value == pytest.approx(baseline.value)
    outcomes = p._predictive('P', {'D': ('val', candidate_id)},
                             p.posterior(.8, {'D': ('val', candidate_id)}))
    assert len(outcomes) == 4
    assert sum(probability for probability, _ in outcomes) == pytest.approx(1)


@pytest.mark.parametrize('state', [{'X': 'agree'}, {'D': 'disagree'}, {'D': ('bad', 'x')},
                                  {'D': ('val', [])}, {'D': None}, [], {'D': ('val', '')}])
def test_bad_state_refused_before_weight_callback(state):
    calls = []
    p = policy(weights=lambda obs: calls.append(obs) or {})
    with pytest.raises(ValueError):
        p.decide(.8, state, 30)
    assert calls == []


@pytest.mark.parametrize('name,value', [('q0', float('nan')), ('q0', 1.01), ('q0', None),
                                       ('lam', -1), ('lam', float('inf')), ('lam', None)])
def test_invalid_numeric_inputs_before_callbacks(name, value):
    calls = []
    p = policy(weights=lambda obs: calls.append(obs) or {})
    kw = dict(q0=.8, state={'D': 'agree'}, lam=30)
    kw[name] = value
    with pytest.raises(ValueError):
        p.decide(**kw)
    assert not calls


@pytest.mark.parametrize('kw', [{'cost': -1}, {'cost': float('nan')}, {'units': {'D': -1}},
                               {'max_units': float('inf')}, {'max_checks': 1.2}, {'checks': ['D', 'D']}])
def test_configuration_domains(kw):
    with pytest.raises(ValueError):
        policy(**kw)


def test_invalid_replay_record_before_callback_and_cache_mutation():
    calls = []
    p = policy(weights=lambda obs: calls.append(obs) or {})
    bad = records()[-1]
    bad['checks']['D']['status'] = 'broken'
    with pytest.raises(ValueError):
        p.replay(.8, bad, 30)
    assert not calls
    p = policy()
    assert p.decide(.8, {}, 30).kind == 'check'
    p.max_checks = 0
    assert p.decide(.8, {}, 30).kind == 'abstain'


def test_pacer_accounts_units_and_rejects_before_state_mutation():
    p = policy()
    b = BudgetPacer(budget=1, kappa=.01)
    b.configure(p, {'D': 1, 'P': 1})
    r = p.replay(.8, records()[-1], 30)
    b.charge(r)
    assert b.mean() <= 1
    before = (b.spent, b.t, b.mu, list(b.history))
    with pytest.raises(ValueError):
        b.charge(b.allowance() + 1)
    assert (b.spent, b.t, b.mu, b.history) == before
