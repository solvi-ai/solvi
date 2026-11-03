"""S8: typed dates are supplied values to ground, not implicit exemptions."""
import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest
from pydantic import AnyHttpUrl, AnyUrl, BaseModel

from solvi import Guard


def context(text):
    return [{'role': 'user', 'content': text}]


@pytest.mark.parametrize('annotation, value, written', [
    (dt.date, '2026-11-03', '2026-11-03'),
    (dt.datetime, '2026-11-03T12:30:00', '2026-11-03T12:30:00'),
    (dt.datetime, '2026-11-03T12:30:00+00:00', '2026-11-03T12:30:00+00:00'),
])
def test_typed_date_values_need_canonical_iso_in_request(annotation, value, written):
    guard = Guard()
    def book(when):
        raise AssertionError('check must not execute a real tool')
    book.__annotations__ = {'when': annotation}
    guard.tool(book)
    guard.firewall()
    call = {'name': 'book', 'arguments': {'when': value}}
    good = guard.check(call, context('Book at ' + written + '.'), store=False)
    assert good.outcome == 'allow'
    assert good.slots['when']['status'] == 'found'
    assert good.slots['when']['value'] == written
    assert good.slots['when']['where'] == 'request'
    start, end = good.slots['when']['at']
    assert good.response.values['conversation'][start:end] == written
    bad = guard.check(call, context('Book at 2050-01-01T01:00:00+00:00.'), store=False)
    assert bad.outcome == 'escalate'
    assert bad.slots['when']['status'] == 'missing'
    assert bad.failed == ['slots_grounded']
    assert good.replay()['ok'] and bad.replay()['ok']


def test_nested_models_and_lists_keep_each_date_slot_grounded():
    class Leg(BaseModel):
        day: dt.date
        departure: dt.datetime
    guard = Guard()
    @guard.tool
    def itinerary(legs: list[Leg]):
        raise AssertionError('not executed')
    guard.firewall()
    call = {'name': 'itinerary', 'arguments': {'legs': [
        {'day': '2026-11-03', 'departure': '2026-11-03T12:00:00'},
        {'day': '2026-11-04', 'departure': '2026-11-04T13:00:00'}]}}
    request = context('2026-11-03 2026-11-03T12:00:00; 2026-11-04 2026-11-04T13:00:00')
    good = guard.check(call, request, store=False)
    assert good.outcome == 'allow'
    assert set(good.slots) == {'legs[0].day', 'legs[0].departure', 'legs[1].day', 'legs[1].departure'}
    assert all(slot['status'] == 'found' for slot in good.slots.values())
    bad_call = {'name': 'itinerary', 'arguments': {'legs': [
        call['arguments']['legs'][0], {'day': '2050-01-01', 'departure': '2026-11-04T13:00:00'}]}}
    bad = guard.check(bad_call, request, store=False)
    assert bad.outcome == 'escalate'
    assert bad.slots['legs[1].day']['status'] == 'missing'
    assert bad.slots['legs[0].day']['status'] == 'found'
    assert good.replay()['ok'] and bad.replay()['ok']


@pytest.mark.parametrize('annotation, value, approved', [
    (dt.date, '2026-11-03', dt.date(2026, 11, 3)),
    (dt.datetime, '2026-11-03T12:00:00+00:00', dt.datetime(2026, 11, 3, 12, tzinfo=dt.timezone.utc)),
])
def test_typed_date_facts_are_compared_by_iso_value_not_implicitly_exempt(annotation, value, approved):
    guard = Guard()
    def book(day):
        raise AssertionError('not executed')
    book.__annotations__ = {'day': annotation}
    guard.tool(book)
    guard.firewall()
    call = {'name': 'book', 'arguments': {'day': value}}
    good = guard.check(call, context('Book it.'), facts={'approved_day': approved}, store=False)
    assert good.outcome == 'allow' and good.slots['day']['where'] == 'fact approved_day'
    bad = guard.check(call, context('Book it.'), facts={'approved_day': dt.date(2050, 1, 1)}, store=False)
    assert bad.outcome == 'escalate' and bad.slots['day']['status'] == 'missing'
    assert good.replay()['ok'] and bad.replay()['ok']


def test_date_grounding_does_not_invent_a_natural_language_date_parser():
    guard = Guard()
    @guard.tool
    def book(day: dt.date):
        raise AssertionError('not executed')
    guard.firewall()
    decision = guard.check({'name': 'book', 'arguments': {'day': '2026-11-03'}},
                           context('Book on November 3, 2026.'), store=False)
    assert decision.outcome == 'escalate' and decision.slots['day']['status'] == 'missing'


def test_explicit_date_enum_and_boolean_exemptions_stay_explicit():
    class Seat(str, Enum):
        economy = 'economy'
        business = 'business'
    guard = Guard()
    @guard.tool
    def book(day: dt.date, seat: Seat, tier: Literal['one', 'two'], refundable: bool):
        raise AssertionError('not executed')
    guard.firewall(exempt=['day'])
    decision = guard.check({'name': 'book', 'arguments': {
        'day': '2050-01-01', 'seat': 'business', 'tier': 'two', 'refundable': True}},
        context('Book it.'), store=False)
    assert decision.outcome == 'allow'
    assert decision.slots['day']['why'] == 'declared'
    assert decision.slots['seat']['why'] == decision.slots['tier']['why'] == 'enumeration'
    assert decision.slots['refundable']['why'] == 'boolean'


def test_a_date_valued_enum_keeps_its_declared_enumeration_exemption():
    import json
    from solvi.solutions.guard import _slot_grounding
    class Day(dt.date, Enum):
        selected = (2026, 11, 3)
    report = _slot_grounding(json.dumps({'sources': ['user'], 'exempt': [], 'enums': [],
                                        'match': 'nocase', 'locale': None}), False)
    evidence = report({'when': '2026-11-03'}, {'when': Day.selected}, '', [])
    assert evidence['missing'] == []
    assert evidence['slots']['when']['status'] == 'exempt'
    assert evidence['slots']['when']['why'] == 'enumeration'


@pytest.mark.parametrize('annotation,value,request_value', [
    (Path, '/private/not-authorized.txt', '/public/allowed.txt'),
    (AnyHttpUrl, 'https://attacker.example/private', 'https://trusted.example/public'),
    (AnyUrl, 's3://attacker-bucket/private', 's3://trusted-bucket/public'),
    (UUID, '123e4567-e89b-12d3-a456-426614174000', '123e4567-e89b-12d3-a456-426614174999'),
    (dt.time, '23:59:00', '09:00:00'),
])
@pytest.mark.parametrize('on_ungrounded', ['escalate', 'deny'])
def test_unsupported_typed_scalars_fail_closed_even_with_matching_text(annotation, value, request_value, on_ungrounded):
    def setup(type_, **firewall):
        guard = Guard()
        def fake(value):
            raise AssertionError('Guard.check must not execute a tool')
        fake.__annotations__ = {'value': type_}
        guard.tool(fake)
        guard.firewall(**firewall)
        return guard
    call = {'name': 'fake', 'arguments': {'value': value}}
    string_guard = setup(str, on_ungrounded=on_ungrounded)
    string_bad = string_guard.check(call, context('Use ' + request_value), store=False)
    assert string_bad.outcome == on_ungrounded and string_bad.slots['value']['status'] == 'missing'
    string_good = string_guard.check(call, context('Use ' + value), store=False)
    assert string_good.outcome == 'allow' and string_good.slots['value']['status'] == 'found'
    typed_guard = setup(annotation, on_ungrounded=on_ungrounded)
    for request in ('Use ' + request_value, 'Use ' + value):
        typed = typed_guard.check(call, context(request), store=False)
        assert typed.outcome == on_ungrounded
        assert typed.slots['value']['status'] == 'missing'
        assert 'unsupported grounding scalar type:' in typed.slots['value']['why']
        assert typed.failed == ['slots_grounded']
        assert any('unsupported grounding scalar type:' in reason for reason in typed.reasons)
        assert typed.response.values['slot_grounding']['missing']
        assert typed.replay()['ok']
    explicit = setup(annotation, exempt=['value'])
    allowed = explicit.check(call, context('Use ' + request_value), store=False)
    assert allowed.outcome == 'allow' and allowed.slots['value']['why'] == 'declared'
    assert allowed.replay()['ok']
    off = setup(annotation, grounded=False)
    unchecked = off.check(call, context('Use ' + request_value), store=False)
    assert unchecked.outcome == 'allow'


def test_unknown_object_is_rejected_without_using_its_string_representation():
    import json
    from solvi.solutions.guard import _slot_grounding
    class Opaque:
        def __str__(self):
            raise AssertionError('opaque str() must not define grounding or report evidence')
    report = _slot_grounding(json.dumps({'sources': ['user'], 'exempt': [], 'enums': [],
                                        'match': 'nocase', 'locale': None}), False)
    evidence = report({'value': 'trusted text'}, {'value': Opaque()}, 'trusted text', [(0, 12, 'user')])
    assert evidence['slots']['value'] == {
        'value': {'unsupported_type': 'Opaque'}, 'status': 'missing',
        'why': 'unsupported grounding scalar type: Opaque'}
    assert evidence['missing'] == ['value: unsupported grounding scalar type: Opaque']
