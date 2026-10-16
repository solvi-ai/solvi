"""solvi.experimental.audit: AuditWatch (a check bought on a random share of decisions, followed by a Bernoulli
CUSUM)."""
import warnings

import pytest

np = pytest.importorskip("numpy")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.audit import AuditWatch  # noqa: E402


def _ref(rng, n, fail):
    return list(rng.random(n) >= fail)


def test_auditwatch_false_alarms_and_detection():
    rng = np.random.default_rng(5)
    alarms, delays = 0, []
    for s in range(60):
        w = AuditWatch(share=0.1, alpha=0.05, horizon=600, seed=s).set_reference(_ref(rng, 800, 0.2))
        for t in range(600):
            if w.observe(passed=bool(rng.random() >= 0.2), key=f"r{s}")["drift"]:
                alarms += 1
                break
    assert alarms <= 8                                                # 60 × 0.05 = 3 expected
    for s in range(30):
        w = AuditWatch(share=0.2, alpha=0.05, horizon=600, seed=s).set_reference(_ref(rng, 800, 0.2))
        for t in range(600):
            fail = 0.2 if t < 100 else 0.6
            if w.observe(passed=bool(rng.random() >= fail), key=f"d{s}")["drift"]:
                delays.append(t - 100)
                break
    assert len(delays) >= 27 and np.median(delays) < 120


def test_auditwatch_reads_the_dispatchers_audit_flag_and_calls_the_check_only_when_audited():
    class D:
        def __init__(self, audit, ok):
            self.audit, self.ok = audit, ok
    calls = []

    def check(d):
        calls.append(d)
        return d.ok
    w = AuditWatch(check, share=0.1, alpha=0.05, horizon=100).set_reference([True] * 40 + [False] * 10)
    r = w.observe(D(False, True))
    assert not r["audited"] and not calls
    r = w.observe(D(True, False))
    assert r["audited"] and r["failed"] and len(calls) == 1 and w.audited == 1
    for _ in range(200):
        w.observe(D(True, False))
    assert w.flagged and w.alarms and "fails more often" in w.observe(D(False, True))["why"][0]
    w.reset()
    assert not w.flagged
    with pytest.raises(ValueError):
        AuditWatch(share=0.0)
    with pytest.raises(ValueError):
        AuditWatch().set_reference([True] * 5)
    with pytest.raises(RuntimeError):
        AuditWatch().observe(passed=True)
