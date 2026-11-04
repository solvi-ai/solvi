import math
import pytest
from solvi.experimental.laws import GaussianProbes


def model(means=(0.,2.), **kw):
    return GaussianProbes([lambda s,a,m=m:m*s for m in means], sigma=kw.pop('sigma',1.), **kw)


def test_normal_likelihood_analytic_and_repeat_same_point():
    g=model(); p=g.choose(lambda *_:True,[(1,'read')])
    assert .40 < p.info < .55  # binary Normal channel, not categorical flip entropy
    assert g.observe(p,2.)[1]==pytest.approx(1/(1+math.exp(-2)))
    assert g.observe(p,2.)[1]==pytest.approx(1/(1+math.exp(-4)))
    assert sum(g.weights)==pytest.approx(1)


def test_adaptive_loop_stops_confidence_and_preserves_repeat():
    g=model();seen=[]
    rows=g.run(lambda s,a:seen.append((s,a)) or 2.,lambda *_:True,[(1,'r')],
               max_observations=10,confidence=.98)
    assert len(rows)==2 and seen==[(1,'r')]*2 and rows[-1]['spent']==2


def test_no_equal_mean_calls_and_budget_zero():
    def forbidden(*_):raise AssertionError('no measurement')
    assert model((1.,1.)).run(forbidden,lambda *_:True,[(1,'r')],max_observations=5)==[]
    assert model().run(forbidden,lambda *_:True,[(1,'r')],max_observations=0)==[]
    assert model().run(forbidden,lambda *_:True,[(1,'r')],max_observations=5,max_cost=0)==[]
    assert len(model().run(lambda *_:2.,lambda *_:True,[(1,'r')],max_observations=10,max_cost=2.5))==2


def test_safe_readonly_cost_and_order():
    g=model(sigma=lambda s,a: (_ for _ in ()).throw(AssertionError()) if a=='unsafe' else 1.)
    points=[(2,'unsafe'),(2,'write'),(1,'expensive'),(1,'cheap'),(1,'tie')]
    cost=lambda s,a: 10 if a=='expensive' else 1
    p=g.choose(lambda s,a:a!='unsafe',points,read_only=lambda a:a!='write',cost=cost)
    assert p.action=='cheap'
    assert g.choose(lambda *_:True,[(1,'expensive'),(.8,'cheap')],cost=cost,per_cost=True).action=='cheap'


@pytest.mark.parametrize('sigma',[0,-1,float('nan'),float('inf'),True])
def test_invalid_noise(sigma):
    with pytest.raises(ValueError):model(sigma=sigma)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True])
def test_bad_observation_atomic(value):
    g=model();p=g.choose(lambda *_:True,[(1,'r')]);before=g.weights
    with pytest.raises(ValueError):g.observe(p,value)
    assert g.weights==before and g.observations==0


def test_logspace_extreme_and_nonuniform_prior():
    g=model(weights=[1e300,1e300]);p=g.choose(lambda *_:True,[(1,'r')])
    assert g.observe(p,1000)[1]==pytest.approx(1)
    assert sum(g.weights)==pytest.approx(1)
    g=model(weights=[3,1]);p=g.choose(lambda *_:True,[(1,'r')])
    assert g.observe(p,1)==pytest.approx((.75,.25))


def test_invalid_budget_prediction_and_cost():
    for kw in [dict(max_observations=True),dict(max_observations=1,max_cost=float('nan')),
               dict(max_observations=1,confidence=0)]:
        with pytest.raises(ValueError):model().run(lambda *_:0,lambda *_:True,[(1,'r')],**kw)
    with pytest.raises(ValueError):model().choose(lambda *_:True,[(float('nan'),'r')])
    with pytest.raises(ValueError):model().choose(lambda *_:True,[(1,'r')],cost=lambda *_:-1)


@pytest.mark.parametrize('observation,expected',[(1e100,(0.,1.)),(-1e100,(1.,0.))])
def test_huge_common_squared_residual_does_not_erase_logodds(observation,expected):
    g=model();p=g.choose(lambda *_:True,[(1,'r')])
    assert g.observe(p,observation)==pytest.approx(expected)
    assert sum(g.weights)==pytest.approx(1)


def test_centered_means_posterior_and_information_translation_invariant():
    g=model();p=g.choose(lambda *_:True,[(1,'r')]);info=p.info
    shifted=model((1e12,1e12+2));q=shifted.choose(lambda *_:True,[(1,'r')])
    assert q.info==pytest.approx(info,abs=1e-12)
    assert shifted.observe(q,1e12+2)==pytest.approx(g.observe(p,2.))


def test_unsupported_relative_scale_is_atomic():
    g=model(sigma=1e-200)
    # Build a matching probe without an unrepresentable EIG ranking calculation.
    from solvi.experimental.laws import Probe
    p=Probe(1,'r',(0.,2.),1)
    before=(g.weights,g.observations)
    with pytest.raises(ValueError,match='supported finite range'):
        g.observe(p,1e200)
    assert (g.weights,g.observations)==before
