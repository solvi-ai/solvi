import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from solvi.experimental.audit import AuditWatch


class CusumStub:
    def reset(self):
        pass

    def step(self, value):
        pass

    def top(self):
        return ('failures', 0, 0)


def watch(check=None):
    w = AuditWatch(check)
    w.cusum, w.h, w._up, w._dn = CusumStub(), 10, 1, -1
    return w


def test_missing_labels_and_lifetime_coverage():
    w = watch(lambda _: None)
    assert w.observe(audited=False)['missing'] is False
    assert w.observe(audited=True)['missing'] is True
    assert w.observe(audited=True, passed=float('nan'))['missing'] is True
    assert w.observe(audited=True, passed=False)['failed'] is True
    r = w.report()
    assert (r['seen'], r['audit_requests'], r['audited'], r['missing'], r['failed']) == (4, 3, 1, 2, 1)
    assert r['observed_coverage'] == 1 / 3
    w.reset()
    assert w.report() == r


def test_exceptions_are_not_missing():
    def broken(_):
        raise RuntimeError('callback defect')
    w = watch(broken)
    with pytest.raises(RuntimeError, match='callback defect'):
        w.observe(audited=True)
    assert w.report()['missing'] == 0
    assert w.report()['audit_errors'] == 1
    assert w.report()['observed_coverage'] == 0
    assert watch().report()['observed_coverage'] is None


def test_latched_detector_still_counts_labels_and_report_is_detached():
    w = watch(lambda _: True)
    w._flagged = True
    w._why = ['already flagged']
    assert w.observe(audited=True)['failed'] is False
    r = w.report()
    assert r['audit_requests'] == r['audited'] == 1
    assert r['observed_coverage'] == 1
    r['missing'] = 100
    assert w.report()['missing'] == 0
    w.reset()
    assert not w.flagged and w.seen == w.audited == 1


def test_missing_configuration_remains_error():
    w = watch()
    with pytest.raises(ValueError, match='needs passed'):
        w.observe(audited=True)
    assert w.report()['audit_errors'] == 1 and w.missing == 0


@pytest.mark.parametrize('kind', ['float', 'float16', 'float32', 'float64'])
def test_floating_nan_missing_without_detector_update(kind):
    import numpy as np
    value = float('nan') if kind == 'float' else getattr(np, kind)('nan')
    w = watch(lambda _: value)
    def forbidden(_):
        raise AssertionError('missing label reached detector')
    w.cusum.step = forbidden
    assert w.observe(audited=True)['missing'] is True
    assert w.audited == 0 and w.missing == 1
    assert w.report()['observed_coverage'] == 0


def test_reference_uses_same_floating_nan_predicate_without_calibration(monkeypatch):
    import numpy as np
    from solvi.core.guarantees.drift import Cusum
    stub = CusumStub()
    stub.h = 10
    monkeypatch.setattr(Cusum, 'calibrate', lambda *args, **kwargs: stub)
    w = AuditWatch().set_reference([True] * 20 + [float('nan'), np.float16('nan'), np.float32('nan'), np.float64('nan')])
    assert w.n_ref == 20 and w.k_ref == 0


def test_array_is_error_not_missing_or_observed():
    import numpy as np
    w = watch(lambda _: np.array([True, False]))
    with pytest.raises(ValueError, match='audit outcome'):
        w.observe(audited=True)
    assert w.missing == w.audited == 0 and w.audit_errors == 1


@pytest.mark.parametrize('bad', ['yes', '', 2, -1, float('inf'), object()])
def test_unsupported_scalar_labels_are_errors(bad):
    w = watch(lambda _: bad)
    with pytest.raises(ValueError, match='audit outcome'):
        w.observe(audited=True)
    assert w.audited == w.failed == w.missing == 0
    assert w.audit_requests == w.audit_errors == w.seen == 1


@pytest.mark.parametrize('shape', [(), (1,), (2,)])
def test_all_array_labels_invalid_in_observation_and_atomic_reference(monkeypatch, shape):
    import numpy as np
    from solvi.core.guarantees.drift import Cusum
    stub = CusumStub()
    stub.h = 10
    monkeypatch.setattr(Cusum, 'calibrate', lambda *args, **kwargs: stub)
    w = AuditWatch().set_reference(np.array([True] * 20))
    state = w.n_ref, w.k_ref, w.p0, w.p1, w.h, w.cusum
    bad = np.full(shape, np.nan)
    with pytest.raises(ValueError, match='audit outcome'):
        w.set_reference([True] * 20 + [bad])
    assert (w.n_ref, w.k_ref, w.p0, w.p1, w.h, w.cusum) == state
    w.check = lambda _: bad
    with pytest.raises(ValueError, match='audit outcome'):
        w.observe(audited=True)
    assert w.audited == w.missing == 0 and w.audit_errors == 1


def test_explicit_numeric_scalar_domain():
    import numpy as np
    w = watch()
    for value in (True, np.bool_(True), 1, np.int64(1), np.float32(1), 0.0, np.int32(0)):
        w.observe(audited=True, passed=value)
    assert w.audited == 7 and w.failed == 2 and w.audit_errors == 0


@pytest.mark.parametrize('bad', ['yes', 2, float('inf')])
def test_invalid_reference_scalar_preserves_fitted_state(bad):
    w = watch()
    w.n_ref, w.k_ref, w.p0, w.p1 = 25, 2, .1, .2
    before = w.__dict__.copy()
    with pytest.raises(ValueError, match='audit outcome'):
        w.set_reference([True] * 20 + [bad])
    assert w.__dict__ == before


def test_calibration_failure_preserves_previous_fitted_detector(monkeypatch):
    from solvi.core.guarantees.drift import Cusum

    w = watch(lambda _: True)
    w.observe(audited=True)
    before = dict(w.__dict__)

    def fail(*_args, **_kwargs):
        raise RuntimeError("calibration unavailable")

    monkeypatch.setattr(Cusum, "calibrate", fail)
    with pytest.raises(RuntimeError, match="calibration unavailable"):
        w.set_reference([True] * 20)
    assert w.__dict__ == before
