import json
import pytest
from solvi.stat import SourceTrust


def test_delayed_labels_expire_by_assertion_and_repeat_is_separate():
    trust = SourceTrust(window=10)
    trust.assert_statement('origin', statement='s', source='lab', asserted_at=0, now=0)
    trust.assert_statement('repeat', statement='s', source='publisher', asserted_at=9, now=9, parent='origin')
    trust.label('origin', loss=1, now=11)
    assert trust.opinion('lab', now=11).n == 0
    assert trust.opinion('lab', now=11).expired == 1
    assert trust.opinion('publisher', now=11).pending == 1
    trust.label('repeat', loss=0.25, now=12)
    opinion = trust.opinion('publisher', now=12, role='repeater')
    assert opinion.n == 1 and opinion.mean_loss == .25
    assert opinion.reliability == pytest.approx(1.75 / 3)
    assert trust.opinion('publisher', now=12, role='origin').n == 0


def test_unknown_and_boundary():
    trust = SourceTrust(window=10, min_labels=2)
    assert trust.opinion('new', now=0).support == 'unknown'
    trust.assert_statement('a', statement='s', source='lab', asserted_at=0, now=0)
    trust.label('a', loss=0, now=1)
    assert trust.opinion('lab', now=9).support == 'unknown'
    assert trust.opinion('lab', now=10).n == 0


def test_snapshot_and_invalid_inputs_are_atomic():
    trust = SourceTrust(window=10)
    trust.assert_statement('z', statement='s', source='lab', asserted_at=0, now=0)
    trust.assert_statement('a', statement='s', source='pub', asserted_at=0, now=0, parent='z')
    trust.label('a', loss=.5, now=2)
    state = json.loads(json.dumps(trust.snapshot(), sort_keys=True))
    restored = SourceTrust.restore(state)
    assert restored.fingerprint() == trust.fingerprint()
    for loss in (-1, 2, float('nan'), True):
        before = trust.fingerprint()
        with pytest.raises(ValueError):
            trust.label('z', loss=loss, now=3)
        assert trust.fingerprint() == before
    before = trust.fingerprint()
    with pytest.raises(ValueError):
        trust.assert_statement('bad', statement='different', source='x', asserted_at=3, now=3, parent='z')
    with pytest.raises(ValueError):
        trust.label('z', loss=0, now=1)
    assert trust.fingerprint() == before
    state['labels']['a']['loss'] = 5
    with pytest.raises(ValueError):
        SourceTrust.restore(state)


def test_large_finite_prior_does_not_overflow_posterior_mean():
    trust = SourceTrust(window=10, prior=(1e308, 1e308))
    assert trust.opinion('new', now=0).reliability == .5
    trust.assert_statement('a', statement='s', source='lab', asserted_at=0, now=0)
    trust.label('a', loss=0, now=1)
    assert trust.opinion('lab', now=1).reliability == .5
