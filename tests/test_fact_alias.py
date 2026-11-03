import pytest
from pydantic import BaseModel

from solvi import Catalog, Question, System
from solvi.core._inputs import input_schema


def make(source='market_value'):
    cat = Catalog()

    @cat.rule('enough')
    def enough(car_worth: float) -> bool:
        return car_worth >= 100

    cat.alias('car_worth', source, description='Vehicle value in EUR')
    return System(cat, [Question('enough', 'Enough collateral?')])


def test_alias_adapts_once_without_wrapping_original_rule():
    sys = make()
    out = sys.ask({'market_value': 120})
    assert out['enough'].answer == 'yes' and 'car_worth = 120' in out['enough'].why
    assert [r.name for r in out.trace.records] == ['car_worth', 'answer:enough']
    assert out.trace.replay(sys, out.flow)['ok']
    with pytest.raises(ValueError, match='cannot stand in'):
        sys.ask({'car_worth': 120})


def test_alias_retains_types_description_and_identity():
    first, same, different = make(), make(), make('assessed_value')
    assert first.fingerprint() == same.fingerprint()
    assert first.fingerprint() != different.fingerprint()
    ep = first.entry_points()[0]
    assert set(ep.fields) == {'market_value'}
    assert ep.fields['market_value'].type is float
    assert ep.fields['market_value'].description == 'Vehicle value in EUR'
    schema = input_schema(first, 'enough')
    assert schema['properties']['market_value']['description'] == 'Vehicle value in EUR'
    assert first.ask({'market_value': 'bad'})['enough'].status == 'abstain'


def test_alias_before_consumers_with_explicit_type_and_external_input_model():
    cat = Catalog()
    cat.alias('car_worth', 'market_value', type=float)

    @cat.rule('enough')
    def enough(car_worth: float) -> bool:
        return car_worth >= 100

    class Inputs(BaseModel):
        market_value: float

    sys = System(cat, [Question('enough', 'Enough?')], input_model=Inputs)
    assert sys.ask({'market_value': '120'})['enough'].answer == 'yes'


def test_alias_refuses_replacement_self_reference_and_cycles():
    cat = Catalog()
    with pytest.raises(ValueError, match='itself'):
        cat.alias('a', 'a')
    cat.alias('a', 'b', type=float)
    before = dict(cat.parts)
    with pytest.raises(ValueError, match='cycle'):
        cat.alias('b', 'a')
    assert cat.parts == before
    with pytest.raises(ValueError, match='replace'):
        cat.alias('a', 'c')


@pytest.mark.parametrize('name,source', [('x', 'bad-name'), ('class', 'x'), ('x', ''), ('answer:q', 'x')])
def test_invalid_alias_names(name, source):
    with pytest.raises(ValueError, match='identifier'):
        Catalog().alias(name, source)


def test_conflicting_source_descriptions_do_not_register_partial_alias():
    cat = Catalog()
    cat.alias('a', 'external', type=float, description='one')
    with pytest.raises(ValueError, match='another description'):
        cat.alias('b', 'external', type=float, description='two')
    assert 'b' not in cat.parts


def test_missing_or_disagreeing_consumer_types_require_explicit_type():
    from typing import Any
    cat = Catalog()
    with pytest.raises(ValueError, match='declare type='):
        cat.alias('x', 'external')
    assert cat.parts == {}

    @cat.fn
    def as_int(x: int):
        return x

    @cat.fn
    def as_string(x: str):
        return x

    before = dict(cat.parts)
    with pytest.raises(ValueError, match='declare type='):
        cat.alias('x', 'external')
    assert cat.parts == before
    cat.alias('x', 'external', type=Any)
    assert 'x' in cat.parts
