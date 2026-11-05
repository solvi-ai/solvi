import copy
import pytest
from solvi.stat import SourceTrust


@pytest.mark.parametrize('field', ['window', 'prior', 'loss', 'now', 'asserted_at', 'snapshot'])
def test_unrepresentable_integer_is_value_error_and_does_not_mutate(field):
    huge = 10 ** 1000
    trust = SourceTrust(window=10)
    trust.assert_statement('a', statement='s', source='lab', asserted_at=0, now=0)
    before = trust.snapshot()
    with pytest.raises(ValueError, match='finite number'):
        if field == 'window':
            SourceTrust(window=huge)
        elif field == 'prior':
            SourceTrust(window=10, prior=(huge, 1))
        elif field == 'loss':
            trust.label('a', loss=huge, now=1)
        elif field == 'now':
            trust.opinion('lab', now=huge)
        elif field == 'asserted_at':
            trust.assert_statement('b', statement='s', source='lab', asserted_at=huge, now=1)
        else:
            state = copy.deepcopy(before)
            state['assertions']['a']['known_at'] = huge
            SourceTrust.restore(state)
    assert trust.snapshot() == before


def test_future_read_does_not_advance_clock_or_rejuvenate_late_label():
    trust = SourceTrust(window=10)
    trust.assert_statement('a', statement='s', source='lab', asserted_at=0, now=0)
    before = trust.fingerprint()
    assert trust.opinion('lab', now=100).expired == 1
    assert trust.fingerprint() == before
    trust.label('a', loss=0, now=1)
    assert trust.opinion('lab', now=9).n == 1
    assert trust.opinion('lab', now=10).n == 0
    assert SourceTrust.restore(trust.snapshot()).fingerprint() == trust.fingerprint()
