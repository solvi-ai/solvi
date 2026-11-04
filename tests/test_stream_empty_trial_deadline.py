"""Zero-occupancy trials already have a durable decision-age deadline in DefaultPolicy."""
from test_stream_additive_b3 import (GOOD, KEY, _Hand, _clock, _events, _loop, _policy,
                                     _run, _until, _xs)

from solvi.experimental.stream import AdapterLoop


def test_empty_trial_expires_on_decisions_across_checkpoint_without_lowering_floors(tmp_path):
    a = _loop([GOOD], policy=_policy(stale=40, min_novel=10 ** 9), alarm_cascade="downstream")
    _until(a, _xs(0, 6000), lambda lp: lp.candidate is not None)
    c = a.candidate
    assert not a._vs[c]["samples"]
    a.pause()  # pause stops new training, not the existing trial deadline
    empty_zone = [("a", 1.0, 1.0, 1.0)] * 10
    _run(a, empty_zone)
    assert a.candidate == c and not a._vs[c]["samples"]
    a.save(tmp_path / "ck")
    b = AdapterLoop.load(tmp_path / "ck", _Hand(list(a.live.models)), secret=KEY, clock=_clock)
    assert b.policy.stale == 40 and b.policy.min_novel == 10 ** 9
    assert b.candidate == c and not b._vs[c]["samples"]
    tail = empty_zone * 5
    _run(a, tail)
    _run(b, tail)
    assert a.log == b.log and a.status() == b.status()
    assert a.candidate is None and not a._route
    discard = [e for e in _events(a, "discard") if e["version"] == c]
    assert len(discard) == 1 and "not certified 40 decisions" in discard[0]["why"]
    assert discard[0]["samples"] == 0 and not _events(a, "promote")
    assert any(e["event"] == "unfreeze" and e["decision"] == discard[0]["decision"]
               for e in _events(a))
    assert b.replay()["ok"]
