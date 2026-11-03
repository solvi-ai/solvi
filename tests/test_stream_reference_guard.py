"""S298 guarded reference re-takes; actual Dispatcher with fake labelled answers."""
import warnings
import pytest
from test_stream_handoff_oct30 import _systems
from solvi.core.dispatch import AskPath, Dispatcher
from solvi.core.guarantees.drift import DriftMonitor
from solvi.experimental.stream import RiskDetector, StreamPromise


def make_dispatcher(detector=None, **kw):
    s1, gold = _systems()
    sp = StreamPromise(max_risk=0.05, delta=0.05, grid="fixed")
    s1.guarantee("intent", promise=sp, signal="msp")
    mon = DriftMonitor(window=20, sequential=False)
    d = Dispatcher(s1, AskPath(gold), question="intent", audit="adaptive", promise=sp,
                   detector=detector, monitor=mon, secret=b"reference-guard",
                   sampling={"floor": 1.0}, **kw)
    return d, sp, mon


def test_missing_label_detector_rejected_before_first_decision():
    with pytest.raises(ValueError, match="concurrent shift"):
        make_dispatcher()
    make_dispatcher(allow_reference_retake=True)


def test_label_risk_survives_reference_retake_and_detects_shift():
    det = RiskDetector(alpha=0.05, arl=1000)
    d, sp, mon = make_dispatcher(det)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(120):
            d.ask({"pred": "a", "msp": 0.9, "gold": "a"})
        assert sp.threshold < float("inf")
        before = det.n
        assert before > 0 and not det.alarms and d.drift is None
        mon.adapted()  # exact dispatch action at certified / loosened
        assert not mon.calibrated and det.n == before
        for i in range(20):
            d.ask({"pred": "a", "msp": 0.9, "gold": "b"})
            if det.alarms:
                break
        assert det.alarms == 1 and d.drift is not None
