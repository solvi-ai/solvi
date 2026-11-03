import pytest

from solvi import Catalog, Question, System


def test_dependencies_are_initial_inputs_through_computations_and_checks():
    cat = Catalog()

    @cat.fn
    def doubled(a: float):
        return 2 * a

    @cat.check(hard=True)
    def allowed(enabled: bool):
        return enabled

    @cat.rule('go')
    def go(doubled, threshold: float) -> bool:
        return doubled >= threshold

    sys = System(cat, [Question('go', 'Go?', requires=['allowed'])])
    assert sys.depends('go') == {'a', 'threshold', 'enabled'}
    first = sys.depends('go')
    first.add('unrelated')
    assert 'unrelated' not in sys.depends('go')


def test_aliases_report_external_inputs():
    cat = Catalog()
    cat.alias('internal', 'external', type=float)

    @cat.rule('go')
    def go(internal) -> bool:
        return internal > 0

    assert System(cat, [Question('go', 'Go?')]).depends('go') == {'external'}


def test_opaque_applicability_requires_explicit_inputs():
    cat = Catalog()

    @cat.rule('go')
    def go(value) -> bool:
        return bool(value)

    cat.contract('go', applicable=lambda state: state['enabled'])
    sys = System(cat, [Question('go', 'Go?')])
    with pytest.raises(ValueError, match='declare applicable_inputs'):
        sys.depends('go')
    cat.contract('go', applicable=lambda state: state['enabled'], applicable_inputs=['enabled'])
    assert sys.depends('go') == {'value', 'enabled'}


def test_unknown_question_and_multiple_names_refused():
    sys = System(Catalog(), [])
    with pytest.raises(KeyError, match='no question'):
        sys.depends('missing')
    with pytest.raises(TypeError, match='one question'):
        sys.depends(['a', 'b'])
