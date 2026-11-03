"""Finite functional fixtures; no independence or quality guarantee."""
import math

import pytest

from solvi.stat import Check, CheckRegistry, CorrelationAwareCombiner, NaiveCombiner, error_correlation


def fixture():
    return [{'P': 'agree', 'D': 'agree'}, {'P': 'disagree', 'D': 'fail'},
            {'P': 'agree', 'D': 'disagree'}, {'P': 'fail'}], [1, 0, 1, 0]


def test_registry_ancestry():
    r = CheckRegistry(Check('answer', 'model', 'direct'))
    r.add(Check('A', 'model', 'expr', shares_call_with='answer'))
    r.add(Check('B', 'model', 'expr', shares_call_with='A'))
    r.add(Check('P', 'model', 'code'))
    assert r.independent() == ['P']
    assert r.prior_rho('A', 'B') == .9
    assert len(r.warnings()) == 3
    with pytest.raises(ValueError):
        r.add(Check('P', 'other', 'code'))


def test_pooled_likelihood_and_generators():
    x, y = fixture()
    m = NaiveCombiner(['P', 'D']).fit(iter(x), iter(y))
    assert m.counts['D'][0]['fail'] == 1
    assert m.lik['P'][1]['agree'] == pytest.approx(9 / 11)
    assert m.posterior(.5, {}) == .5
    assert math.isfinite(m.posterior(.8, {'P': 'agree'}))


def test_weights_recomputed_and_returned_dict_owned():
    rho = {('P', 'D'): .5}
    c = CorrelationAwareCombiner(['P', 'D'], rho)
    assert c.weights(['P', 'D']) == pytest.approx({'P': 2/3, 'D': 2/3})
    assert c.n_effective(['P', 'D']) == pytest.approx(4/3)
    w = c.weights(['P'])
    w['P'] = 0
    assert c.weights(['P']) == {'P': 1}
    rho[('P', 'D')] = 0
    assert c.weights(['P', 'D']) == {'P': 1, 'D': 1}


def test_error_correlation_shrink():
    r = error_correlation([{'P': True, 'D': True}, {'P': False, 'D': False}], ['P', 'D'],
                          prior=lambda a, b: .2)
    assert r['rho'][('P', 'D')] == pytest.approx(1/3)
    assert r['raw'][('P', 'D')]['n_inf'] == 1


@pytest.mark.parametrize('x,y', [([{}], []), ([{}], [2]), ([{}], [.5]),
    ([{'P': 'unknown'}], [1]), ([{'X': 'agree'}], [1])])
def test_invalid_fit(x, y):
    with pytest.raises(ValueError):
        NaiveCombiner(['P']).fit(iter(x), iter(y))


@pytest.mark.parametrize('rho', [{('P', 'D'): float('nan')}, {('P', 'D'): 2},
    {('P', 'D'): .2, ('D', 'P'): .3}, lambda a,b: 1 if a == b else (.2 if a == 'P' else .3)])
def test_invalid_rho(rho):
    with pytest.raises(ValueError):
        CorrelationAwareCombiner(['P', 'D'], rho).weights(['P', 'D'])


def test_answer_policy_actual_candidate():
    from solvi.core.plan.answer_policy import AnswerPolicy, CheckModel
    records = [{'answer': 1., 'gold': 1. if ok else 2., 'correct': ok,
        'checks': {k: {'value': 1. if ok else 2., 'status': 'agree' if ok else 'disagree',
                       'correct': True} for k in ('P', 'D')}} for ok in [True]*8 + [False]*4]
    comb = CorrelationAwareCombiner(['P', 'D'], {('P','D'): .5})
    p = AnswerPolicy(CheckModel().fit(records), weights=comb.weights)
    a = p.decide(.8, {'P': ('val','actual'), 'D': ('val','actual')}, 30)
    assert a.kind == 'answer_check' and a.check in ('P','D')
    assert p.replay(.8, records[-1], 30).answer == 2.


@pytest.mark.parametrize('cls,kw,llr,posterior', [
    (NaiveCombiner, {}, 4.727076938828336, .9977918096211402),
    (CorrelationAwareCombiner, {'rho': {('P','D'): .5}}, 3.151384625885558, .9894150317561615)])
def test_archived_source_differential_pins(cls, kw, llr, posterior):
    # Computed from the archived original on fixture(), not a historical quality eval.
    x, y = fixture()
    m = cls(['P','D'], **kw).fit(x,y)
    assert m.log_lr({'P':'agree','D':'agree'}) == pytest.approx(llr, abs=1e-14)
    assert m.posterior(.8, {'P':'agree','D':'agree'}) == pytest.approx(posterior, abs=1e-14)


@pytest.mark.parametrize('alpha,pool', [(1e308, 2.), (1e308, 1e308),
    (1e-300, 1e-300), (5e-324, 5e-324), (1e-300, 1e308)])
@pytest.mark.parametrize('empty', [True, False])
@pytest.mark.parametrize('correlated', [True, False])
def test_finite_extreme_smoothing(alpha, pool, empty, correlated):
    cls = CorrelationAwareCombiner if correlated else NaiveCombiner
    kw = {'rho': {('P','D'): .5}} if correlated else {}
    m = cls(['P','D'], alpha=alpha, pool=pool, **kw)
    x, y = ([], []) if empty else ([{'P':'agree', 'D':'agree'}], [1])
    m.fit(iter(x), iter(y))
    for outcome in m.outcomes:
        assert math.isfinite(m.llr('P', outcome))
        assert math.isfinite(m.log_lr({'P': outcome, 'D': outcome}))
        for prior in (0., .5, 1.):
            q = m.posterior(prior, {'P': outcome, 'D': outcome})
            assert math.isfinite(q) and 0 <= q <= 1
        if empty:
            assert m.lik['P'][1][outcome] == pytest.approx(1/3)
            assert m.llr('P', outcome) == 0
    if not empty and alpha == pool == 1e-300:
        assert m.lik['P'][1]['fail'] == 0  # introspection can underflow; inference does not
        assert m.llr('P', 'fail') < -1000


@pytest.mark.parametrize('parameter', ['pool', 'alpha'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True])
def test_mutated_smoothing_rejected_before_fit(parameter, value):
    m = NaiveCombiner(['P'])
    setattr(m, parameter, value)
    with pytest.raises(ValueError):
        m.fit([], [])
