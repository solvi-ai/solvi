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


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_waiver_requires_an_actual_bool_before_constructor_access(value):
    class Untouched:
        @property
        def questions(self):
            raise AssertionError("constructor touched the system before validating the waiver")

    with pytest.raises(TypeError, match="allow_reference_retake must be a bool"):
        Dispatcher(Untouched(), allow_reference_retake=value)


def test_waiver_is_bound_into_config_and_default_keeps_legacy_digest(monkeypatch):
    import solvi.core.provenance as provenance

    d, _, _ = make_dispatcher(RiskDetector(alpha=0.05))
    default = d.config()
    real_digest = provenance.digest
    calls = []

    def capture(*args):
        calls.append(args)
        return real_digest(*args)

    monkeypatch.setattr(provenance, "digest", capture)
    assert d.allow_reference_retake is False
    assert d.config() == default
    baseline_args = calls[-1]
    assert ("allow_reference_retake", True) not in baseline_args
    d.allow_reference_retake = True
    assert d.config() != default
    waived_args = calls[-1]
    assert ("allow_reference_retake", True) in waived_args
    assert real_digest(*(a for a in waived_args if a != ("allow_reference_retake", True))) == default


def test_explicit_waiver_stream_replay_keeps_config_compatible(tmp_path):
    from solvi.core.store import JSONLStorage

    d, _, _ = make_dispatcher(allow_reference_retake=True, storage=JSONLStorage(tmp_path / "waived.jsonl"))
    fresh, _, _ = make_dispatcher(allow_reference_retake=True)
    assert d.allow_reference_retake is True and fresh.config() == d.config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(5):
            d.ask({"pred": "a", "msp": 0.9, "gold": "a"})
    assert d.replay_stream()["ok"]
    assert all(record.config == d.config() for record in d.stored())
    assert d.replay_all() == []
    assert fresh.replay_stream(d.stored())["ok"]


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
