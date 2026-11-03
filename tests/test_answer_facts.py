import asyncio
import copy
from typing import Literal

import pytest

from solvi import Answer, Catalog, Claim, Question, System


def make_system(*, source=None, threshold=None, cache=None):
    cat, calls = Catalog(), []

    @cat.rule('payable')
    def payable(amount: float) -> bool:
        calls.append(amount)
        return amount > 0 if source is None else source

    cat.answer_fact('payable')

    @cat.rule('go')
    def go(payable: str) -> Literal['pay', 'skip']:
        return 'pay' if payable == 'yes' else 'skip'

    return System(cat, [Question('payable', 'Payable?', min_confidence=threshold),
                        Question('go', 'Next?')], cache=cache), calls


def test_accepted_answer_not_raw_bool_downstream_and_count_once():
    system, calls = make_system()
    result = system.ask({'amount': 2}, 'go')
    assert result['go'].answer == 'pay' and calls == [2]
    assert result.trace.init == {'amount': 2}
    assert result.values['payable'] == 'yes'
    assert system.depends('go') == {'amount'}
    assert system.ask({'amount': -1})['go'].answer == 'skip'


def test_source_hard_check_does_not_force_consumer():
    system, _ = make_system()

    @system.catalog.check(hard=True, then={'payable': 'no'})
    def permitted(amount):
        return False

    result = system.ask({'amount': 2}, 'go')
    assert result.values['payable'] == 'no'
    assert result['go'].answer == 'skip' and result['go'].status == 'ok'


def test_source_refusal_does_not_call_consumer():
    system, _ = make_system(source=None, threshold=1.01)
    result = system.ask({'amount': 2})
    assert result['payable'].status == 'abstain'
    assert result['go'].status == 'abstain'
    assert any(r.kind == 'answer_ref' and r.error for r in result.trace.records)


def test_source_confidence_bounds_downstream():
    system, _ = make_system()

    @system.catalog.fn
    def amount(raw):
        return Claim(raw, confidence=.2)

    result = system.ask({'raw': 2})
    assert result['go'].answer == 'pay'
    assert result['go'].confidence <= .2


def test_source_cycle_rejected_without_calls():
    system, calls = make_system()
    system.catalog.answer_fact('go', name='amount')
    with pytest.raises(ValueError, match='cycle'):
        system.ask({}, 'go')
    assert not calls


def test_joint_source_dependency_rejected_without_calls():
    system, calls = make_system()

    @system.catalog.constraint
    def joint(payable, go):
        return True

    with pytest.raises(ValueError, match='cycle'):
        system.ask({'amount': 2})
    assert not calls


def test_missing_source_and_given_override_rejected():
    system, _ = make_system()
    del system.questions['payable']
    with pytest.raises(ValueError, match='not declared'):
        system.ask({'amount': 2}, 'go')
    system, _ = make_system()
    with pytest.raises(ValueError, match='given fact'):
        system.ask({'amount': 2, 'payable': 'yes'})


def test_trace_replay_json_and_reference_tamper():
    from solvi.core.response import Response
    system, calls = make_system()
    response = system.ask({'amount': 2}, 'go')
    assert response.trace.replay(system, response.flow)['ok']
    assert calls == [2, 2]
    restored = Response.from_json(response.to_json(), system.catalog)
    assert restored.trace.replay(system, restored.flow)['ok']
    damaged = copy.deepcopy(response.trace)
    ref = next(r for r in damaged.records if r.kind == 'answer_ref')
    ref.confidence = .99
    assert not damaged.replay(system, response.flow)['ok']


def test_async_cache_and_lightweight_match():
    system, calls = make_system(cache=4)
    assert asyncio.run(system.aask({'amount': 2}, 'go'))['go'].answer == 'pay'
    assert system.ask({'amount': 2}, 'go')['go'].answer == 'pay'
    assert calls == [2]
    assert system._decide({'amount': -1}, 'go')['go'].answer == 'skip'


def test_shared_computation_and_checks_on_computed_downstream_run_once():
    cat, calls = Catalog(), []

    @cat.fn
    def shared(x):
        calls.append('shared')
        return x

    @cat.rule('a')
    def a(shared):
        return True

    cat.answer_fact('a')

    @cat.check(hard=True, then={'b': 'no'})
    def allowed(shared):
        calls.append('allowed')
        return False

    @cat.rule('b')
    def b(a, shared):
        calls.append('b')
        return a == 'yes'

    sys = System(cat, [Question('a', 'A?', Answer.yes_no()), Question('b', 'B?', Answer.yes_no())])
    result = sys.ask({'x': 3})
    assert result['b'].answer == 'no' and result['b'].status == 'forced'
    assert calls.count('shared') == calls.count('allowed') == 1


def test_structured_source_metadata_and_unknown_are_preserved():
    from solvi import Unknown
    cat = Catalog()

    @cat.rule('a')
    def a(doc):
        return Claim({'n': 2}, confidence=.7, extra={'fields': {'/n': {'why': 'declared count'}}})

    cat.answer_fact('a')

    @cat.rule('b')
    def b(a: dict):
        return a['n'] == 2

    sys = System(cat, [Question('a', 'A?', Answer.structured()), Question('b', 'B?', Answer.yes_no())])
    result = sys.ask({'doc': '2'})
    assert result['b'].answer == 'yes' and result['b'].confidence <= .7
    ref = next(r for r in result.trace.records if r.kind == 'answer_ref')
    assert ref.extra['result']['extra']['fields']['/n']['why'] == 'declared count'
    assert result.trace.replay(sys, result.flow)['ok']

    cat = Catalog()
    cat.rule('a')(lambda doc: Unknown)
    cat.answer_fact('a')
    cat.rule('b')(lambda a: a is Unknown)
    sys = System(cat, [Question('a', 'A?', Answer.maybe(Answer.yes_no())), Question('b', 'B?', Answer.yes_no())])
    result = sys.ask({'doc': 'nothing'})
    assert result['a'].not_stated and result['b'].answer == 'yes'


def test_source_contract_refusal_and_transitive_echo_read():
    cat = Catalog()
    cat.rule('a')(lambda x: True)
    cat.answer_fact('a')
    cat.rule('b')(lambda a: 'yes')
    cat.contract('b', echo=True)
    sys = System(cat, [Question('a', 'A?', Answer.yes_no()), Question('b', 'B?', Answer.yes_no())])
    result = sys.ask({'x': 'yes'})
    assert result['b'].status == 'abstain' and result['b'].guard == 'contract'
    cat.contract('a', applicable=lambda state: False, applicable_inputs=())
    result = sys.ask({'x': 'yes'})
    assert result['a'].guard == 'not_applicable' and result['b'].status == 'abstain'


def test_guards_applied_once_before_publishing(monkeypatch):
    import solvi.core.guarantees.guarantee as guarantee
    from solvi.core.runtime import Result
    system, _ = make_system()
    calls = []

    class Guard:
        def fingerprint(self):
            return 'test-guard'

    system.guards['payable'] = Guard()

    def guarded(system, results, *args, **kwargs):
        calls.append(tuple(results))
        if 'payable' in results:
            results['payable'] = Result(None, 0, 'guard refused', 'abstain', guard='test_guard')

    monkeypatch.setattr(guarantee, 'apply_guards', guarded)
    response = system.ask({'amount': 2})
    assert response['go'].status == 'abstain'
    assert calls == [('payable',), ('go',)]
    ref = next(r for r in response.trace.records if r.kind == 'answer_ref')
    assert ref.extra['result']['guard'] == 'test_guard'


def test_probe_keeps_diagnostic_semantics_and_replays():
    system, _ = make_system()
    stats, costs = copy.deepcopy(system.stats), copy.deepcopy(system.cost_book.__dict__)
    response = system.probe('go', {'amount': 2})
    assert response['go'].answer == 'pay'
    assert response['go'].extra['probe']['hypothetical']
    assert response.trace.replay(system, response.flow)['ok']
    assert system.stats == stats and system.cost_book.__dict__ == costs
    with pytest.raises(ValueError, match='cannot replace accepted'):
        system.probe('go', {'payable': 'yes'}, override=True)


def test_source_applicability_input_is_required_in_downstream_schema_and_skips_call():
    system, calls = make_system()
    system.catalog.contract('payable', applicable=lambda state: state['enabled'], applicable_inputs=('enabled',))
    assert system.depends('go') == {'amount', 'enabled'}
    result = system.ask({'amount': 2, 'enabled': False}, 'go')
    assert result['go'].status == 'abstain' and calls == []
    assert result.trace.replay(system, result.flow)['ok']


def test_async_parts_and_timeout_propagate_without_running_consumer():
    cat, calls = Catalog(), []

    @cat.rule('a')
    async def a(x) -> bool:
        await asyncio.sleep(.02)
        return x > 0

    cat.answer_fact('a')

    @cat.rule('b')
    def b(a) -> bool:
        calls.append(a)
        return a == 'yes'

    sys = System(cat, [Question('a', 'A?'), Question('b', 'B?')])
    result = asyncio.run(sys.aask({'x': 2}, timeout=.001))
    assert result['a'].status == result['b'].status == 'abstain' and calls == []
    assert asyncio.run(sys.aask({'x': 2}))['b'].answer == 'yes'


def test_requested_inapplicable_question_outside_dependency_stages():
    system, _ = make_system()

    @system.catalog.rule('unrelated')
    def unrelated(amount) -> bool:
        raise AssertionError('not applicable question must not execute')

    system = System(system.catalog, [*system.questions.values(), Question('unrelated', 'Unrelated?')])
    system.catalog.contract('unrelated', applicable=lambda state: False, applicable_inputs=())
    for response in [system.ask({'amount': 2}), asyncio.run(system.aask({'amount': 2}))]:
        assert response['go'].answer == 'pay'
        assert response['unrelated'].status == 'abstain'
        assert response['unrelated'].guard == 'not_applicable'
        assert response.trace.replay(system, response.flow)['ok']


def test_unfitted_source_abstains_without_inventing_a_reference_cycle():
    cat = Catalog()
    cat.answer_fact('a')
    cat.rule('b')(lambda a: True)
    system = System(cat, [Question('a', 'A?', Answer.yes_no()), Question('b', 'B?', Answer.yes_no())])
    response = system.ask({'x': 1})
    assert response['a'].status == response['b'].status == 'abstain'
    assert 'not fitted' in response['a'].why
    assert response.trace.replay(system, response.flow)['ok']
    system.questions['a'].uses = ['a']
    with pytest.raises(ValueError, match='cycle'):
        system.ask({'x': 1})


def test_fit_source_then_fit_consumer_explicit_answer_features():
    cat = Catalog()
    cat.answer_fact('a')
    sys = System(cat, [Question('a', 'A?', Answer.yes_no()), Question('b', 'B?', Answer.yes_no())])
    examples = [({'x': x}, x > 0) for x in range(-10, 11)]
    sys.fit('a', examples, features=['x'], select=False)
    assert sys.ask({'x': 10}, 'a')['a'].answer == 'yes'
    sys.fit('b', examples, features=['a'], select=False)
    response = sys.ask({'x': 10})
    assert response['b'].answer == 'yes'
    assert response.trace.replay(sys, response.flow)['ok']
    with pytest.raises(ValueError, match='own accepted answer'):
        sys.fit('a', examples, features=['a'])
    previous_head = sys.heads['a']

    @cat.fn
    def numeric(a):
        return int(a == 'yes')

    with pytest.raises(ValueError, match='own accepted answer'):
        sys.fit('a', examples, features=['numeric'])
    assert sys.heads['a'] is previous_head
    assert sys.ask({'x': 10})['a'].answer == 'yes'


def test_rename_keeps_answer_reference_source_and_original_system():
    system, _ = make_system()
    renamed = system.rename(inputs={'amount': 'total'}, answers={'payable': 'eligible', 'go': 'next'})
    response = renamed.ask({'total': 2}, 'next')
    assert response['next'].answer == 'pay'
    assert renamed.depends('next') == {'total'}
    assert response.trace.replay(renamed, response.flow)['ok']
    assert system.ask({'amount': 2}, 'go')['go'].answer == 'pay'


def test_fit_cannot_depend_on_own_answer_through_another_question():
    cat = Catalog()
    cat.answer_fact('a')

    @cat.rule('b')
    def b(a) -> bool:
        return a == 'yes'

    cat.answer_fact('b')
    sys = System(cat, [Question('a', 'A?', Answer.yes_no()), Question('b', 'B?')])
    examples = [({'x': x}, x > 0) for x in range(-10, 11)]
    sys.fit('a', examples, features=['x'], select=False)
    before = sys.heads['a']
    with pytest.raises(ValueError, match='own accepted answer'):
        sys.fit('a', examples, features=['b'], select=False)
    assert sys.heads['a'] is before
    assert sys.ask({'x': 10})['a'].answer == 'yes'
