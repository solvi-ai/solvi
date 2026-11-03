import json
from typing import Literal

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from solvi import Answer, Catalog, Claim, Question, Response, System
from solvi.core import AnswerType
from solvi.core.schema import response_schema


class QuoteAnswer(BaseModel):
    price: float
    currency: Literal['EUR', 'USD']


class NestedIntegerKeys(BaseModel):
    rates: dict[int, int]


class Reservation(TypedDict):
    seats: list[int]


def make(value, at=None):
    cat = Catalog()

    @cat.rule('quote')
    def quote(doc):
        return value

    return System(cat, [Question('quote', 'Quote?', at or Answer.structured())])


@pytest.mark.parametrize('t', [dict, dict[str, float], list[dict[str, int]], QuoteAnswer, Reservation])
def test_infer_and_answer_type_roundtrip(t):
    at = Answer.from_type(t)
    assert at.kind == 'structured' and at.primitive
    restored = AnswerType.from_json(at.to_json())
    assert restored.kind == 'structured' and restored.type == t


def test_model_rule_inference_and_response_schema():
    cat = Catalog()

    @cat.rule('quote')
    def quote(doc: str) -> QuoteAnswer:
        return QuoteAnswer(price=12, currency='EUR')

    sys = System(cat, [Question('quote', 'Quote?')])
    out = sys.ask({'doc': '12 EUR'})
    assert out['quote'].answer == {'price': 12., 'currency': 'EUR'}
    assert out['quote'].confidence == 1 and out['quote'].why
    assert 'QuoteAnswer' in json.dumps(response_schema(sys))
    restored = Response.from_json(out.to_json())
    assert restored['quote'].answer == out['quote'].answer


def test_explicit_field_why_confidence_and_grounded_evidence():
    value = Claim({'price': 12., 'currency': 'EUR'}, confidence=.9, extra={'fields': {
        '/price': {'why': 'tariff table', 'confidence': .8, 'evidence': ['12']},
        '/currency': {'why': 'contract currency', 'confidence': .95, 'evidence': ['EUR']},
    }})
    out = make(value).ask({'doc': '12 EUR'})['quote']
    assert out.status == 'ok' and out.confidence == .8
    assert out.extra['fields']['/price']['why'] == 'tariff table'
    assert out.extra['fields']['/currency']['why'] == 'contract currency'
    assert out.extra['fields']['/price']['evidence'][0]['value'] == '12'


@pytest.mark.parametrize('value', [{1: 'bad key'}, {'x': float('nan')}, {'x': object()}, {'x': float('inf')}])
def test_reject_non_json(value):
    out = make(value).ask({'doc': ''})['quote']
    assert out.status == 'abstain' and out.guard == 'type_rejected'


@pytest.mark.parametrize('fields', [
    {'/missing': {'why': 'wrong path'}},
    {'/x': {'why': ''}},
    {'/x': {'why': 'bad', 'confidence': float('nan')}},
    {'/x': {'why': 'bad', 'evidence': ['fabricated']}},
])
def test_reject_invalid_field_explanations(fields):
    out = make(Claim({'x': 1}, extra={'fields': fields})).ask({'doc': '1'})['quote']
    assert out.status == 'abstain' and out.guard == 'type_rejected'


def test_nested_schema_violation_abstains():
    sys = make({'seats': ['bad']}, Answer.from_type(Reservation))
    out = sys.ask({'doc': ''})['quote']
    assert out.status == 'abstain' and out.guard == 'type_rejected'


def test_literal_collection_remains_closed_answer():
    assert Answer.from_type(list[Literal['a', 'b']]).kind == 'multi'


def test_hard_check_forces_validated_object_and_contract_applies():
    cat = Catalog()

    @cat.check(hard=True, then={'quote': {'price': 0., 'currency': 'EUR'}})
    def allowed(doc):
        return False

    @cat.rule('quote')
    def quote(doc):
        return {'price': 12., 'currency': 'EUR'}

    sys = System(cat, [Question('quote', 'Quote?', Answer.structured(QuoteAnswer), requires=['allowed'])])
    out = sys.ask({'doc': ''})['quote']
    assert out.status == 'forced' and out.answer['price'] == 0
    sys.catalog.contract('quote', returns=list, echo=False)
    other = make({'price': 12., 'currency': 'EUR'}, Answer.structured(QuoteAnswer))
    other.catalog.contract('quote', returns=list, echo=False)
    sys = System(other.catalog, [Question('quote', 'Quote?', Answer.structured(QuoteAnswer))])
    rejected = sys.ask({'doc': ''})['quote']
    assert rejected.status == 'abstain' and rejected.guard == 'contract'


def test_cache_nested_answer_and_metadata_are_isolated():
    cat = Catalog()

    @cat.rule('quote')
    def quote(doc):
        return Claim({'items': [{'price': 12}]}, extra={'fields': {'/items/0/price': {'why': 'tariff'}}})

    sys = System(cat, [Question('quote', 'Quote?', Answer.structured())], cache=2)
    first = sys.ask({'doc': ''})
    first['quote'].answer['items'][0]['price'] = 999
    first['quote'].extra['fields']['/items/0/price']['why'] = 'corrupted'
    first.values['answer:quote']['items'][0]['price'] = 888
    first.trace.records[-1].extra['fields']['/items/0/price']['why'] = 'trace corruption'
    second = sys.ask({'doc': ''})
    assert second['quote'].answer['items'][0]['price'] == 12
    assert second['quote'].extra['fields']['/items/0/price']['why'] == 'tariff'
    assert second.values['answer:quote']['items'][0]['price'] == 12
    assert second.trace.records[-1].extra['fields']['/items/0/price']['why'] == 'tariff'


def test_field_pointer_escaping_and_unknown_paths():
    out = make(Claim({'a/b': {'~x': [1]}}, extra={'fields': {'/a~1b/~0x/0': {'why': 'nested'}}})).ask({'doc': ''})['quote']
    assert out.status == 'ok' and out.extra['fields']['/a~1b/~0x/0']['why'] == 'nested'


def test_local_type_cannot_silently_lose_schema_on_serialization():
    class Local(BaseModel):
        x: int
    with pytest.raises(ValueError, match='importable'):
        Answer.structured(Local)


def test_numeric_key_schema_cannot_merge_distinct_fields():
    with pytest.raises(ValueError, match='string keys'):
        Answer.structured(dict[int, int])


def test_typed_rule_cannot_coerce_non_json_bytes_before_check():
    cat = Catalog()

    @cat.rule('quote')
    def quote(doc: str) -> dict[str, str]:
        return {'x': b'abc'}

    sys = System(cat, [Question('quote', 'Quote?')])
    out = sys.ask({'doc': ''})['quote']
    assert out.status == 'abstain' and out.guard == 'type_rejected'


def test_variable_tuple_schema_roundtrips():
    t = list[tuple[int, ...]]
    at = Answer.structured(t)
    assert AnswerType.from_json(at.to_json()).type == t


def test_nested_model_integer_keys_cannot_silently_merge():
    out = make({'rates': {'01': 1, '1': 2}}, Answer.structured(NestedIntegerKeys)).ask({'doc': ''})['quote']
    assert out.status == 'abstain' and out.guard == 'type_rejected'


def test_cyclic_json_is_rejected():
    from solvi.core.types import _normalize_structured as normalize
    value = {}
    value['loop'] = value
    with pytest.raises(ValueError, match='cycles'):
        normalize(Answer.structured(), value)


def test_explicit_weaker_answer_schema_keeps_rule_return_validation():
    cat = Catalog()

    @cat.rule('quote')
    def quote(doc: str) -> dict[str, int]:
        return {'x': 'bad'}

    sys = System(cat, [Question('quote', 'Quote?', Answer.structured())])
    out = sys.ask({'doc': ''})['quote']
    assert out.status == 'abstain' and 'type rejected' in out.why


def test_variable_tuple_values_become_json_arrays():
    sys = make([[1, 2]], Answer.structured(list[tuple[int, ...]]))
    assert sys.ask({'doc': ''})['quote'].answer == [[1, 2]]


@pytest.mark.parametrize('annotation,value', [
    (tuple[dict[str, str], ...], ({'x': b'abc'},)),
    (dict[str, str] | list[str], {'x': b'abc'}),
])
def test_tuple_union_return_annotations_cannot_erase_non_json_values(annotation, value):
    cat = Catalog()

    def quote(doc):
        return value

    quote.__annotations__['return'] = annotation
    cat.rule('quote')(quote)
    sys = System(cat, [Question('quote', 'Quote?', Answer.structured(annotation))])
    out = sys.ask({'doc': ''})['quote']
    assert out.status == 'abstain' and out.guard == 'type_rejected'


def test_schema_changes_identity_and_rejects_drift_on_load():
    left = make({'price': 1}, Answer.structured(dict[str, int]))
    right = make({'price': '1'}, Answer.structured(dict[str, str]))
    assert left.fingerprint() != right.fingerprint()
    from solvi.core.types import _load_structured_type as load_type, _structured_type_data as type_data
    data = type_data(QuoteAnswer)
    data['schema']['properties']['price']['type'] = 'string'
    with pytest.raises(ValueError, match='schema changed'):
        load_type(data)


def test_explicit_field_evidence_confidence_bounds_its_field():
    evidence = {'value': '12', 'start': 0, 'end': 2, 'source': 'doc', 'confidence': .3}
    result = make(Claim({'price': 12}, extra={'fields': {
        '/price': {'why': 'tariff', 'confidence': .9, 'evidence': [evidence]},
    }})).ask({'doc': '12'})['quote']
    assert result.status == 'ok' and result.confidence == .3
    assert result.extra['fields']['/price']['confidence'] == .3
