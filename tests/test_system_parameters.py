import asyncio

import pytest

from solvi import Answer, Catalog, Question, System


def make(tmp_path, **options):
    calls = []
    cat = Catalog()

    @cat.fn
    def value(x, config):
        calls.append(x)
        return x + config['offset']

    @cat.check(hard=True, then={'q': 'blocked'})
    def allowed(permission):
        return permission

    @cat.rule('q')
    def result(value):
        return str(value)

    s = System(cat, [Question('q', 'q', Answer.choice(['1', '2', '3', '4', 'blocked']), requires=['allowed'])],
               storage=tmp_path / 'records.jsonl', **options)
    return s, calls


def test_private_defaults_overrides_fingerprint_traces_and_replay(tmp_path):
    s, calls = make(tmp_path)
    defaults = {'config': {'offset': 1}, 'permission': True}
    view = s.with_params(defaults)
    fp = view.fingerprint()
    defaults['config']['offset'] = 99
    view.params['config']['offset'] = 999
    assert view.params['config'] == {'offset': 1}
    a = view.ask({'x': 1})
    b = view.ask({'x': 1, 'config': {'offset': 2}})
    c = view.ask({'x': 1, 'permission': False})
    assert [r['q'].answer for r in (a, b, c)] == ['2', '3', 'blocked']
    assert a.trace.init['config'] == {'offset': 1}
    assert b.trace.init['config'] == {'offset': 2}
    assert a.trace.init_hash != b.trace.init_hash
    assert a.trace.replay(s)['ok']
    assert s.storage.replay_all(s) == []
    assert view.fingerprint() == fp
    assert view.fingerprint() != s.with_params(config={'offset': 2}).fingerprint()
    assert view.shared_system is s
    assert s.stats['asks'] == 3
    with pytest.raises((AttributeError, TypeError)):
        view.params = {}
    assert not hasattr(view, 'guarantee')


@pytest.mark.parametrize('params', [{'value': 100}, {'allowed': True}, {'__solvi_hypothetical_probe__': {}},
    {'x': float('inf')}, {'x': (1, 2)}, {'x': {1: 'bad'}}])
def test_defaults_refuse_computed_hard_facts_and_non_json(tmp_path, params):
    s, calls = make(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        s.with_params(params)
    assert calls == []


def test_input_model_validation_stays_native(tmp_path):
    from pydantic import BaseModel, Field

    class Input(BaseModel):
        x: int = Field(gt=0)
        config: dict
        permission: bool

    s, _ = make(tmp_path, input_model=Input)
    view = s.with_params(config={'offset': 1}, permission=True)
    for x in ('1', -1):
        bound = view.ask({'x': x})
        native = s.ask({'x': x, 'config': {'offset': 1}, 'permission': True})
        assert bound['q'].answer == native['q'].answer
        assert bound.trace.init == native.trace.init
    assert view.ask({'x': '1'})['q'].answer == '2'


def test_batch_sequential_storage_cache_and_view_parity(tmp_path):
    s, calls = make(tmp_path, cache=8)
    view = s.with_params(config={'offset': 1}, permission=True)
    responses = view.ask_many([{'x': 1}, {'x': 2}, {'x': 1}])
    assert [r['q'].answer for r in responses] == ['2', '3', '2']
    assert len({r.stored_id for r in responses}) == 3
    assert len({id(r.trace) for r in responses}) == 3
    assert calls == [1, 2]
    assert responses[-1].cached is not None
    assert s.storage.verify()['ok'] and s.storage.replay_all(s) == []
    plain = s.ask_many([{'x': 2, 'config': {'offset': 1}, 'permission': True}])
    assert plain[0]['q'].answer == '3'
    assert asyncio.run(view.aask({'x': 1}))['q'].answer == '2'


@pytest.mark.parametrize('states,limit', [([{'x': 1}, {'value': 1}], 3),
    ([{'x': 1}, 3], 3), ([{'x': 1}, {1: 2}], 3), ([{'x': 1}, {'x': 2}], 1),
    ([{'x': 1}], True), ([{'x': 1}], 1.0)])
def test_batch_preflight_before_calls(tmp_path, states, limit):
    s, calls = make(tmp_path)
    view = s.with_params(config={'offset': 1}, permission=True)
    with pytest.raises((TypeError, ValueError)):
        view.ask_many(states, max_items=limit)
    assert calls == [] and s.stats['asks'] == 0


def test_batch_runtime_failure_keeps_earlier_record(tmp_path, monkeypatch):
    s, calls = make(tmp_path)
    native = s.ask
    def ask(state, *args, **kwargs):
        if state['x'] == 2:
            raise RuntimeError('backend failed')
        return native(state, *args, **kwargs)
    monkeypatch.setattr(s, 'ask', ask)
    view = s.with_params(config={'offset': 1}, permission=True)
    with pytest.raises(RuntimeError, match='backend failed'):
        view.ask_many([{'x': 1}, {'x': 2}, {'x': 3}])
    assert calls == [1]
    assert s.stats['asks'] == 1 and s.storage.verify()['ok']


def test_batch_empty_generator_failure_and_native_model_parity(tmp_path):
    from pydantic import BaseModel

    class Input(BaseModel):
        x: int
        config: dict
        permission: bool

    s, calls = make(tmp_path, input_model=Input)
    assert s.ask_many([]) == []
    def broken():
        yield {'x': 1, 'config': {'offset': 1}, 'permission': True}
        raise RuntimeError('iterator failed')
    with pytest.raises(RuntimeError, match='iterator failed'):
        s.ask_many(broken())
    assert calls == [] and s.stats['asks'] == 0
    state = Input(x=1, config={'offset': 1}, permission=True)
    batch = s.ask_many([state])[0]
    native = s.ask(state)
    assert batch['q'].answer == native['q'].answer == '2'
    assert batch.trace.init == native.trace.init


def test_process_binding_is_explicitly_unsupported(tmp_path):
    from solvi.experimental.process import Process
    s, _ = make(tmp_path)
    view = s.with_params(x=1)
    with pytest.raises(TypeError, match='parameter views'):
        Process('p').node('s', view, end=True)
    spec = Process('p').node('s', s, end=True).spec()
    with pytest.raises(TypeError, match='parameter views'):
        Process.from_spec(spec, systems={'s': view})
