import copy

import pytest
from pydantic import BaseModel

from solvi import Catalog, Question, Response, System


def build():
    cat = Catalog()

    @cat.fn
    def ratio(price: float, limit: float) -> float:
        return price / limit

    @cat.rule('go')
    def go(ratio: float, threshold: float) -> bool:
        return ratio >= threshold

    return System(cat, [Question('go', 'Proceed?')], cache=2)


def test_real_rule_at_intermediate_values_with_why_and_replay():
    sys = build()
    with pytest.raises(ValueError, match='cannot stand in'):
        sys.ask({'ratio': .9, 'threshold': .75})
    with pytest.raises(ValueError, match='cannot stand in'):
        sys.probe('go', {'ratio': .9, 'threshold': .75})
    out = sys.probe('go', {'ratio': .9, 'threshold': .75}, override=True)
    result = out['go']
    assert result.answer == 'yes' and result.status == 'ok'
    assert 'ratio = 0.9' in result.why and 'hypothetical probe' in result.why
    assert [r.name for r in out.trace.records] == ['answer:go']
    assert result.extra['probe'] == {'hypothetical': True, 'overridden': ['ratio'], 'guarantee_applied': False}
    replay = out.trace.replay(sys, out.flow)
    assert replay['ok'], replay
    restored = Response.from_json(out.to_json())
    assert restored['go'].extra['probe'] == result.extra['probe']


def test_probe_does_not_update_stats_costs_cache_or_original_input():
    sys = build()
    state = {'price': 9., 'limit': 10., 'threshold': .75}
    stats = copy.deepcopy(sys.stats)
    cost = copy.deepcopy(sys.cost_book.__dict__)
    out = sys.probe('go', state)
    assert out['go'].answer == 'yes' and sys.stats == stats
    assert sys.cost_book.__dict__ == cost and not sys._cache.items
    assert state == {'price': 9., 'limit': 10., 'threshold': .75}
    assert out._system is None


def test_probe_can_use_direct_facts_despite_raw_input_model():
    class Inputs(BaseModel):
        price: float
        limit: float
        threshold: float
    sys = build()
    sys.input_model = Inputs
    out = sys.probe('go', {'ratio': .5, 'threshold': .75}, override=True)
    assert out['go'].answer == 'no'
    rejected = sys.probe('go', {'ratio': 'bad', 'threshold': .75}, override=True)
    assert rejected['go'].status == 'abstain'


def test_probe_cannot_supply_the_answer_and_validates_names():
    sys = build()
    with pytest.raises(ValueError, match='answer rules'):
        sys.probe('go', {'answer:go': True}, override=True)
    with pytest.raises(KeyError, match='no question'):
        sys.probe('missing', {})
    with pytest.raises(TypeError, match='override'):
        sys.probe('go', {}, override=1)


def test_checks_cannot_be_silently_overridden_and_real_checks_still_force():
    from solvi import Answer
    cat = Catalog()

    @cat.check(hard=True, then={'go': 'no'})
    def safe(y):
        return y < 1

    @cat.rule('go')
    def go(y) -> bool:
        return True

    sys = System(cat, [Question('go', 'Go?', Answer.yes_no(), requires=['safe'])])
    with pytest.raises(ValueError, match='cannot replace checks'):
        sys.probe('go', {'safe': False, 'y': 2}, override=True)
    out = sys.probe('go', {'y': 2})
    assert out['go'].status == 'forced' and out['go'].answer == 'no'
    assert out.trace.replay(sys, out.flow)['ok']


def test_probe_and_replay_skip_guarantees_and_force_serial_context(monkeypatch):
    import solvi.core.runtime as runtime
    import solvi.core.guarantees.guarantee as guarantee
    sys = build()
    sys.workers = 3

    class Guard:
        def fingerprint(self):
            return 'probe-test-guard'

    sys.guards = {'go': Guard()}
    def forbidden(*args, **kwargs):
        raise AssertionError('a hypothetical probe cannot apply a calibrated guarantee')
    monkeypatch.setattr(guarantee, 'apply_guards', forbidden)
    original, workers = runtime.execute, []
    def execute(*args, **kwargs):
        workers.append(kwargs.get('workers'))
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, 'execute', execute)
    out = sys.probe('go', {'ratio': .9, 'threshold': .75}, override=True)
    assert workers == [1] and out['go'].extra['probe']['guarantee_applied'] is False
    assert out.trace.replay(sys, out.flow)['ok']
    restored = Response.from_json(out.to_json())
    assert restored.trace.replay(sys, restored.flow)['ok']


def test_probe_marker_is_reserved_and_hash_bound():
    from solvi.core.system import _PROBE_MARKER as MARKER
    sys = build()
    with pytest.raises(ValueError, match='reserved'):
        sys.ask({'price': 1., 'limit': 1., 'threshold': .5, MARKER: {}})
    out = sys.probe('go', {'ratio': .9, 'threshold': .75}, override=True)
    out.trace.init[MARKER]['overridden'] = []
    assert not out.trace.replay(sys, out.flow)['ok']
