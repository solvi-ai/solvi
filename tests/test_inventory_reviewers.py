"""S295: separate reviewers retain their own pre-outcome judgement."""
from solvi.core.store.sysreport import _people


def test_each_reviewer_keeps_last_own_label_before_outcome():
    pairs = {('decision', 'q'): [
        ('human', 'yes', 'ann', 1),
        ('human', 'no', 'bob', 2),
        ('outcome', 'no', 'ledger', 3),
        ('human', 'no', 'ann', 4),
    ]}
    result = _people(pairs)
    assert result['paired'] == 1 and result['wrong'] == 0
    assert result['by']['ann']['paired'] == 1 and result['by']['ann']['wrong'] == 1
    assert result['by']['bob']['paired'] == 1 and result['by']['bob']['wrong'] == 0
    assert result['after_outcome'] == 0
