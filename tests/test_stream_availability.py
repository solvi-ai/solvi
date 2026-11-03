"""Availability diagnostics never replace worst-case missing-label evidence."""
import copy
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from solvi.experimental.stream import RiskDetector, StreamPromise
    from solvi.experimental.stream.sampling import StreamDriver


def feed(driver, n, *, lost=False, label='a', alone=True):
    driver.issue(n, True)
    return driver.after(score=.9, would='a', alone=alone, label=label,
                        label_source='human', sampled=True, free=False, rho=1,
                        in_force=.5, probation=False, lost=lost, n=n)


def test_clean_received_labels_do_not_hide_the_missing_worst_case_losses():
    detector = RiskDetector(alpha=.05, arl=1000)
    losses = []
    update = detector.update
    def remember(loss):
        losses.append(loss)
        return update(loss)
    detector.update = remember
    driver = StreamDriver(StreamPromise(max_risk=.05, grid=[.5]), detector)
    for n in range(1, 101):
        missing = n % 10 == 0
        feed(driver, n, lost=missing, label=None if missing else 'a')
    info = driver.availability()
    assert info['observed_error_rate'] == 0
    assert info['missing_rate'] == .1
    assert info['observed_labels'] == info['observed_deployed'] == 90
    assert sum(losses) == 10 and len(losses) == 100  # detector still sees every expired label as loss one
    assert driver.promise.missing == 10


def test_only_fifo_consumed_samples_change_the_diagnostics():
    driver = StreamDriver(StreamPromise(max_risk=.05, grid=[.5]))
    for n in (1, 2, 3):
        driver.issue(n, True)
    feed(driver, 3, label='b')
    feed(driver, 2, alone=False)
    assert driver.availability()['resolved_samples'] == 0
    feed(driver, 1, label=None, lost=True)
    info = driver.availability()
    assert info['resolved_samples'] == 3 and info['missing_labels'] == 1
    assert info['observed_labels'] == 2 and info['observed_deployed'] == 1
    assert info['observed_deployed_wrong'] == 1 and info['observed_error_rate'] == 1
    info['missing_labels'] = -100
    assert driver.availability()['missing_labels'] == 1


def test_legacy_driver_does_not_fabricate_historical_availability():
    driver = StreamDriver(StreamPromise(max_risk=.05, grid=[.5]))
    feed(driver, 1)
    del driver._availability  # a driver pickled before diagnostics existed
    driver = copy.deepcopy(driver)
    assert driver.availability()['unknown_prefix'] is True
    assert driver.availability()['missing_rate'] is None
    feed(driver, 2, label=None)
    assert driver.availability()['unknown_prefix'] is True
    assert driver.availability()['resolved_samples'] == 1
    assert driver.availability()['missing_rate'] == 1


def test_dispatcher_replay_rebuilds_the_same_availability():
    from test_stream_handoff_oct30 import _systems
    from solvi.core.dispatch import AskPath, Dispatcher
    system, gold = _systems()
    promise = StreamPromise(max_risk=.05, grid=[0.0])
    system.guarantee('intent', promise=promise, signal='msp')
    dispatcher = Dispatcher(system, AskPath(gold), question='intent', promise=promise,
                            audit='adaptive', secret=b'availability')
    records = [dispatcher.ask({'pred': 'a', 'msp': .9, 'gold': 'a'}) for _ in range(5)]
    replay = dispatcher.replay_stream(records)
    assert replay['ok'] and replay['availability'] == dispatcher.availability()
    assert dispatcher.availability()['observed_labels'] == 5
