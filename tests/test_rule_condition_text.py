"""Declared condition text works even for rules with no inspectable source."""
import pytest
from solvi import Answer, Catalog, Question, System
from solvi.core import because
from solvi.core.provenance import part_fingerprint


def make(text=None, own=False):
    cat = Catalog()
    fn = eval('lambda dm_ratio, kb_tl_threshold: dm_ratio >= kb_tl_threshold')
    if own:
        def fn(dm_ratio, kb_tl_threshold):
            return because(dm_ratio >= kb_tl_threshold, 'comparison evaluated')
    cat.rule('approve', **({'text': text} if text is not None else {}))(fn)
    return System(cat, [Question('approve', 'Approve?', Answer.yes_no())])


@pytest.mark.parametrize('ratio, expected', [(0.9, 'yes'), (0.5, 'no')])
def test_condition_text_and_values_on_both_outcomes(ratio, expected):
    s = make('dm_ratio >= kb_tl_threshold')
    response = s.ask({'dm_ratio': ratio, 'kb_tl_threshold': 0.75})
    assert response['approve'].answer == expected
    assert response['approve'].why == f'condition: dm_ratio >= kb_tl_threshold; dm_ratio = {ratio}; kb_tl_threshold = 0.75'
    assert response.trace.replay(s.catalog)['ok']
    assert s.depends('approve') == {'dm_ratio', 'kb_tl_threshold'}


def test_condition_preserves_dynamic_reason_and_plain_compatibility():
    assert make('a >= b', own=True).ask({'dm_ratio': 0.9, 'kb_tl_threshold': 0.75})['approve'].why.startswith(
        'because comparison evaluated; condition: a >= b;')
    assert make().ask({'dm_ratio': 0.9, 'kb_tl_threshold': 0.75})['approve'].why == 'dm_ratio = 0.9; kb_tl_threshold = 0.75'


@pytest.mark.parametrize('text', ['', '  ', 42, []])
def test_invalid_condition_text_rejected(text):
    with pytest.raises(ValueError, match='text'):
        make(text)


def test_condition_text_is_audited_in_rule_fingerprint():
    one, two = make('first condition'), make('second condition')
    assert part_fingerprint(one.catalog.rules['approve']) != part_fingerprint(two.catalog.rules['approve'])
