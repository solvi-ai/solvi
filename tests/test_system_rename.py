import pytest
from solvi import Catalog, Question, System, Response


def make():
    cat = Catalog()
    @cat.rule('enough')
    def enough(car_worth: float) -> bool:
        return car_worth >= 100
    @cat.check(hard=True, then={'enough': False})
    def valid(car_worth: float) -> bool:
        return car_worth >= 0
    return System(cat, [Question('enough', 'Enough?', requires=['valid'])])


def test_rename_boundary_trace_json_and_fingerprint():
    original = make()
    renamed = original.rename(inputs={'car_worth': 'market_value'}, answers={'enough': 'eligible'})
    out = renamed.ask({'market_value': 120})
    assert out['eligible'].answer == 'yes'
    assert 'car_worth = 120' in out['eligible'].why
    assert out.trace.replay(renamed, out.flow)['ok']
    restored = Response.from_json(out.to_json(), catalog=renamed)
    assert restored['eligible'].answer == 'yes'
    assert renamed.ask({'market_value': -1})['eligible'].answer == 'no'
    assert original.ask({'car_worth': 120})['enough'].answer == 'yes'
    assert renamed.fingerprint() == make().rename(inputs={'car_worth': 'market_value'}, answers={'enough': 'eligible'}).fingerprint()
    assert renamed.fingerprint() != original.fingerprint()


@pytest.mark.parametrize('kwargs', [
    {'answers': {'missing': 'new'}}, {'answers': {'enough': ''}},
    {'inputs': {'missing': 'external'}}, {'inputs': {'car_worth': 'valid'}},
    {'inputs': {'car_worth': 'car_worth'}},
])
def test_invalid_rename_is_atomic(kwargs):
    original = make()
    before = original.fingerprint()
    with pytest.raises(ValueError):
        original.rename(**kwargs)
    assert original.fingerprint() == before


def test_answer_swap_and_constraint_keep_function_arguments():
    cat = Catalog()
    @cat.rule('left')
    def left(x: int) -> bool:
        return x > 0
    @cat.rule('right')
    def right(x: int) -> bool:
        return x < 0
    @cat.constraint
    def opposite(left, right):
        return left != right
    sys = System(cat, [Question('left', 'L'), Question('right', 'R')])
    renamed = sys.rename(answers={'left': 'right', 'right': 'left'})
    out = renamed.ask({'x': 3})
    assert out['right'].answer == 'yes'
    assert out['left'].answer == 'no'
    c = renamed.catalog.constraints['opposite']
    assert c.inputs == ['right', 'left']
    assert c.func(right='yes', left='no')
    assert not c.func(right='yes', left='yes')
    with pytest.raises(ValueError, match='collides'):
        sys.rename(answers={'left': 'right'})


def test_contract_initial_state_and_explicit_echo_names_are_adapted():
    cat = Catalog(sensitive=['car_worth'])
    @cat.rule('enough')
    def enough(car_worth: float) -> bool:
        return car_worth > 0
    def applies(state):
        return state['car_worth'] > 0
    cat.contract('enough', applicable=applies, applicable_inputs=('car_worth',), echo=['car_worth'])
    original = System(cat, [Question('enough', 'Enough?')])
    renamed = original.rename(inputs={'car_worth': 'market_value'}, answers={'enough': 'eligible'})
    assert renamed.ask({'market_value': 120})['eligible'].answer == 'yes'
    c = renamed.catalog.contracts['eligible']
    assert c.applicable_inputs == ('market_value',)
    assert c.echo_inputs({'market_value': 120}) == {'market_value': 120}
    assert 'market_value' in renamed.catalog.sensitive
    assert original.catalog.contracts['enough'].applicable_inputs == ('car_worth',)


def test_untyped_input_and_span_grounding_source():
    from solvi import Answer
    cat = Catalog()
    @cat.rule('copy')
    def copy(doc):
        return doc
    cat.contract('copy', echo=False)
    sys = System(cat, [Question('copy', 'Copy', answer=Answer.span())])
    renamed = sys.rename(inputs={'doc': 'external_doc'})
    assert renamed.ask({'external_doc': 'hello'})['copy'].answer == 'hello'


def test_compatible_different_consumer_types_preserve_each_validation():
    cat = Catalog()

    @cat.rule('a')
    def a(x: int) -> bool:
        return x > 0

    @cat.rule('b')
    def b(x: float) -> bool:
        return x > 0

    system = System(cat, [Question('a', 'A?'), Question('b', 'B?')])
    renamed = system.rename(inputs={'x': 'external_x'})
    response = renamed.ask({'external_x': 2})
    assert response['a'].answer == response['b'].answer == 'yes'
    assert response.trace.replay(renamed, response.flow)['ok']


def test_guard_act_uses_renamed_question_without_changing_original_guard():
    from solvi import Claim
    from solvi.core.guarantees.guarantee import QuestionGuard, calibrate
    cat = Catalog()

    @cat.rule('a')
    def a(x) -> bool:
        return Claim(True, extra={'act': .9})

    system = System(cat, [Question('a', 'A?')])
    guard = QuestionGuard('a', calibrate([.8] * 30, [True] * 30, max_risk=.1), signal='act')
    system.guards['a'] = guard
    original_fingerprint = guard.fingerprint()
    assert system.ask({'x': 1})['a'].status == 'ok'
    renamed = system.rename(answers={'a': 'renamed'})
    response = renamed.ask({'x': 1})
    assert response['renamed'].status == 'ok'
    assert response.trace.replay(renamed, response.flow)['ok']
    assert renamed.guards['renamed'].question == 'renamed'
    assert guard.question == 'a' and guard.fingerprint() == original_fingerprint


def test_uses_fallback_remains_an_initial_fact_after_rename():
    cat = Catalog()
    @cat.rule('go')
    def go(x: int) -> bool:
        return x > 0
    cat.contract('go', applicable=lambda state: state['x'] > 0)
    sys = System(cat, [Question('go', 'Go?', uses=['x'])])
    renamed = sys.rename(inputs={'x': 'external_x'})
    assert renamed.depends('go') == {'external_x'}
    assert set(renamed.entry_points()[0].fields) == {'external_x'}
    assert renamed.ask({'external_x': 1})['go'].answer == 'yes'


@pytest.mark.parametrize('cost', [False, True])
def test_boundary_alias_keeps_given_check_selection(cost):
    from solvi.core.plan.cost import CostStrategist
    cat = Catalog()
    @cat.rule('go')
    def go(x: int) -> bool:
        return x > 0
    @cat.check(hard=True)
    def safe(x: int) -> bool:
        return False
    strategy = CostStrategist() if cost else None
    with pytest.warns(UserWarning, match='idle|never runs'):
        sys = System(cat, [Question('go', 'Go?')], strategist=strategy)
    with pytest.warns(UserWarning, match='idle|never runs'):
        renamed = sys.rename(inputs={'x': 'external_x'})
    original = sys.ask({'x': 1})
    adapted = renamed.ask({'external_x': 1})
    assert original['go'].answer == adapted['go'].answer == 'yes'
    assert 'safe' not in [r.name for r in adapted.trace.records]
    # Ordinary user-declared aliases continue to be computed facts.
    direct = Catalog()
    @direct.rule('go')
    def direct_go(x: int) -> bool:
        return x > 0
    @direct.check(hard=True)
    def direct_safe(x: int) -> bool:
        return False
    direct.alias('x', 'external_x')
    ordinary = System(direct, [Question('go', 'Go?')])
    assert ordinary.ask({'external_x': 1})['go'].answer != 'yes'
    assert ordinary.fingerprint()['catalog'] != renamed.fingerprint()['catalog']


def test_boundary_marker_fingerprint_cache_and_required_checks():
    from copy import deepcopy
    from solvi.core.provenance import catalog_fingerprint
    cat = Catalog()
    @cat.rule('go')
    def go(x: int) -> bool:
        return x > 0
    @cat.check(hard=True)
    def safe(x: int) -> bool:
        return False
    sys = System(cat, [Question('go', 'Go?', requires=['safe'])])
    renamed = sys.rename(inputs={'x': 'external_x'})
    adapted = renamed.ask({'external_x': 1})
    assert adapted['go'].guard == 'hard_check'
    assert 'safe' in [r.name for r in adapted.trace.records]
    copy = deepcopy(renamed.catalog)
    before = catalog_fingerprint(copy)['fp']
    copy._boundary_given_aliases = frozenset()
    assert catalog_fingerprint(copy)['fp'] != before


def test_checks_on_actual_computations_still_run_after_boundary_rename():
    cat = Catalog()
    @cat.fn
    def doubled(x: int) -> int:
        return x * 2
    @cat.rule('go')
    def go(doubled: int) -> bool:
        return doubled > 0
    @cat.check(hard=True)
    def safe(doubled: int) -> bool:
        return False
    sys = System(cat, [Question('go', 'Go?')])
    renamed = sys.rename(inputs={'x': 'external_x'})
    assert sys.ask({'x': 1})['go'].guard == renamed.ask({'external_x': 1})['go'].guard == 'hard_check'
