"""Opaque applicability predicates declare initial inputs for exported schemas."""
import pytest
from pydantic import BaseModel, ValidationError

from solvi import Answer, Catalog, Question, System
from solvi.core._inputs import input_model, input_schema
from solvi.core.contract import Contract


class Inputs(BaseModel):
    text: str
    enabled: bool


def enabled(state):
    return state['enabled']


def build(declared=('enabled',), uses=None, model=Inputs):
    cat = Catalog()

    @cat.rule('q')
    def answer(text: str) -> int:
        return len(text)

    options = {} if declared is None else {'applicable_inputs': declared}
    cat.contract('q', applicable=enabled, **options)
    return System(cat, [Question('q', 'Length?', Answer.estimate(), uses=uses)], input_model=model)


def test_declared_hidden_fact_is_required_typed_and_client_schema_input_works():
    system = build()
    schema = input_schema(system, 'q')
    assert set(schema['properties']) == {'text', 'enabled'}
    assert set(schema['required']) == {'text', 'enabled'}
    assert schema['properties']['enabled']['type'] == 'boolean'
    client = input_model(system, 'q')
    payload = client.model_validate({'text': 'hello', 'enabled': True}).model_dump(by_alias=True)
    result = system.ask(payload)
    assert result['q'].answer == 5 and result.trace.replay(system)['ok']
    with pytest.raises(ValidationError):
        client.model_validate({'text': 'hello'})
    entry = system.entry_points()[0]
    assert entry.fields['enabled'].type is bool and entry.fields['enabled'].required


def test_undeclared_opaque_reads_fail_schema_but_local_ask_is_unchanged():
    system = build(declared=None)
    assert system.ask({'text': 'hi', 'enabled': True})['q'].answer == 2
    for export in (lambda: input_schema(system, 'q'), system.entry_points):
        with pytest.raises(ValueError, match='declare applicable_inputs'):
            export()
    assert build(declared=None, uses=['enabled']).ask({'text': 'hi', 'enabled': False})['q'].guard == 'not_applicable'
    assert 'enabled' in input_schema(build(declared=None, uses=['enabled']), 'q')['required']


def test_empty_declaration_explicitly_requires_no_extra_initial_facts():
    system = build(declared=())
    system.catalog.contract('q', applicable=lambda state: True, applicable_inputs=())
    assert set(input_schema(system, 'q')['properties']) == {'text'}
    assert system.ask({'text': 'hi'})['q'].answer == 2


def test_declaration_is_initial_fact_not_computed_dependency():
    system = build()

    @system.catalog.fn
    def enabled(text: str) -> bool:
        return bool(text)

    with pytest.raises(ValueError, match='initial given facts, not computed'):
        input_schema(system, 'q')


@pytest.mark.parametrize('bad', ['enabled', 1, True, (1,), {'enabled': bool}])
def test_bad_declaration_types_refused(bad):
    with pytest.raises(TypeError, match='applicable_inputs'):
        Contract(applicable=enabled, applicable_inputs=bad)


@pytest.mark.parametrize('bad', [('',), ('bad name',), ('enabled', 'enabled'), ('a:b',)])
def test_bad_declaration_names_refused(bad):
    with pytest.raises(ValueError, match='applicable_inputs'):
        Contract(applicable=enabled, applicable_inputs=bad)


def test_fingerprint_changes_on_declaration_and_reconstructed_contract_replays():
    first, same, different = build(), build(), build(declared=())
    assert first.fingerprint() == same.fingerprint()
    assert first.fingerprint() != different.fingerprint()
    result = first.ask({'text': 'hello', 'enabled': True})
    assert result.trace.replay(same)['ok']
    # A declaration changes recorded identity, not the callable's local verdict.
    assert result.trace.fingerprint != different.ask({'text': 'hello', 'enabled': True}).trace.fingerprint
    assert Contract(applicable=enabled).fingerprint() != Contract(applicable=enabled, applicable_inputs=()).fingerprint()


def test_declared_fact_type_also_comes_from_existing_typed_readers():
    system = build(model=None)

    @system.catalog.fn
    def audit_enabled(enabled: bool) -> bool:
        return enabled

    schema = input_schema(system, 'q')
    assert schema['properties']['enabled']['type'] == 'boolean'
    assert system.entry_points()[0].fields['enabled'].type is bool
