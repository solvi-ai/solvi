"""solvi.experimental.stream, the adapter loop (stage 4): the checkpoint. StreamPromise and RiskDetector dump / restore;
AdapterLoop.save / load — a loop loaded from a checkpoint continues with the same decisions and records as the loop it
was saved from (a promotion and a rollback after the load included) and replays exactly; a tampered or foreign
checkpoint is refused; a crash in the middle of a save leaves the previous checkpoint usable; a training in flight
refuses the save unless cancelled. No torch: the LoRA end to end is in test_stream_lora.py."""
import json
import random
import threading
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy, RiskDetector, StreamPromise
    from solvi.experimental.stream import checkpoint as ckpt_mod
    from solvi.experimental.stream.serving import HeadVersion

KEY = b"checkpoint-test-key"
SHARES = [0.6, 0.75, 0.3, 0.9, 0.97, 0.995, 0.4, 0.998, 0.999]


def _events(loop, *kinds):
    return [e for k, e in loop._items() if k == "stream" and (not kinds or e["event"] in kinds)]


def _plain(keep=10, **kw):
    base = dict(calib=0.0, holdout=0.0, pre_freeze=False, probe=False)
    base.update(kw)
    return DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6, max_versions_kept=keep), **base)


def _fake(seed, n):
    r = random.Random(seed)
    return [(r.choice("ab"), r.random()) for _ in range(n)]


class _Model:
    """x = (label, u): right with score 0.95 when u < share, else a wrong answer with score 0.05."""

    def __init__(self, name, share):
        self.name, self.share = name, share

    def score(self, x):
        return (x[0], 0.95) if x[1] < self.share else ("wrong", 0.05)

    def fingerprint(self):
        return f"model-{self.name}-{self.share}"


class _Det:
    """A batch learner whose training depends only on the job (the version it would become): the same in the loop that
    was saved and in the one loaded from it. gate: an Event the training waits for (None: none)."""

    def __init__(self, gate=None):
        self._seen, self.gate = 0, gate

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        if self.gate is not None:
            while not self.gate.wait(0.01):
                if job.cancel is not None and job.cancel.is_set():
                    from solvi.experimental.stream.runner import TrainingCancelled
                    raise TrainingCancelled(job.id)
        return _Model(f"v{job.version}", SHARES[(job.version - 1) % len(SHARES)])


class _Other(_Det):
    pass


def _loop(learner=None, **kw):
    base = dict(question="q", max_risk=0.05, secret=KEY, initial=_Model("v0", 0.5), wait=True, clock=lambda: 1000.0,
                detector=RiskDetector(0.05, arl=50.0), policy=_plain(keep=3, growth=1.0, trial=300, first=200))
    base.update(kw)
    return AdapterLoop(_Det() if learner is None else learner, **base)


def _load(path, learner=None, **kw):
    base = dict(secret=KEY, clock=lambda: 1000.0)
    base.update(kw)
    return AdapterLoop.load(path, _Det() if learner is None else learner, **base)


def _norm(log):
    """The records without the one measured field, a training's wall time (report.seconds)."""
    out = []
    for k, e in log:
        if k == "stream" and isinstance(e.get("report"), dict):
            e = {**e, "report": {**e["report"], "seconds": None}}
        out.append((k, e))
    return out


def _run(loop, data):
    return [loop.step(x, lambda x=x: x[0]) for x in data]


# ───────────────────────────── dump / restore
@pytest.mark.parametrize("bets", ["mixture", "agrapa"])
def test_a_stream_promise_restored_from_its_dump_continues_exactly(bets):
    r = random.Random(3)
    a = StreamPromise(max_risk=0.1, delta=0.05, bets=bets, grid_from=50)
    for _ in range(400):
        s = r.random()
        a.observe(s)
        a.learn(s, "x", "x" if r.random() < 0.97 else "y")
    d = json.loads(json.dumps(a.dump()))
    b = StreamPromise(max_risk=0.1, delta=0.05, bets=bets, grid_from=50).restore(d)
    assert b.state() == a.state() and b.events == a.events
    for _ in range(300):
        s = r.random()
        for p in (a, b):
            p.observe(s)
            p.learn(s, "x", "x" if s > 0.2 else "y")
    assert b.state() == a.state() and b.threshold == a.threshold
    with pytest.raises(ValueError, match="another configuration"):
        StreamPromise(max_risk=0.2, delta=0.05, bets=bets, grid_from=50).restore(d)
    d["procs"][0][next(k for k in d["procs"][0] if k not in ("bet", "n"))] = [0.5] if bets == "mixture" else 0.5
    with pytest.raises(ValueError):
        StreamPromise(max_risk=0.1, delta=0.05, bets=bets, grid_from=50).restore(d)


def test_a_risk_detector_restored_from_its_dump_continues_exactly():
    r = random.Random(4)
    a = RiskDetector(0.05, arl=100.0)
    for _ in range(300):
        a.update(1.0 if r.random() < 0.08 else 0.0)
    b = RiskDetector(**a.config()).restore(json.loads(json.dumps(a.dump())))
    assert b.state() == a.state() and b.flagged == a.flagged and b.alarms == a.alarms
    for _ in range(300):
        x = 1.0 if r.random() < 0.2 else 0.0
        assert a.update(x) == b.update(x)
    assert b.state() == a.state()
    with pytest.raises(ValueError, match="another configuration"):
        RiskDetector(0.05, arl=200.0).restore(a.dump())


# ───────────────────────────── save / load: the same continuation
def test_a_loaded_loop_continues_exactly_as_the_loop_it_was_saved_from(tmp_path):
    data = _fake(7, 9000)
    a = _loop()
    _run(a, data[:2500])
    assert a.served != 0 or a.candidate is not None, a.versions()
    info = a.save(tmp_path / "ck")
    assert info["n"] == 2500 and (tmp_path / "ck" / "CURRENT").exists()
    b = _load(tmp_path / "ck")
    assert b.n == a.n and b.served == a.served and b.candidate == a.candidate and b.versions() == a.versions()
    assert b.log == a.log and b.delta_spent == a.delta_spent and b.status() == a.status()
    rb_at = 6000
    for i, x in enumerate(data[2500:], start=2501):
        oa, ob = a.step(x, lambda x=x: x[0]), b.step(x, lambda x=x: x[0])
        assert oa == ob, i
        if i == rb_at:                               # a rollback after the load, on both
            target = next(v["version"] for v in reversed(a.versions()) if v["state"] == "retired")
            assert a.rollback(target, by="test") == b.rollback(target, by="test")
    assert _norm(a.log) == _norm(b.log) and a.versions() == b.versions() and a.status() == b.status()
    after = [e for e in _events(b, "promote", "rollback", "freeze", "alarm") if e["decision"] > 2500]
    kinds = {e["event"] for e in after}
    assert {"promote", "rollback", "freeze"} <= kinds, kinds
    r = b.replay()
    assert r["ok"] and r["decisions"] == 9000, r["mismatches"][:5]


def test_a_loop_with_a_trace_storage_loads_from_its_chain_head(tmp_path):
    from solvi.core.store import JSONLStorage
    store = JSONLStorage(tmp_path / "records.jsonl")
    data = _fake(9, 3000)
    a = _loop(storage=store)
    _run(a, data[:1500])
    a.save(tmp_path / "ck")
    b = _load(tmp_path / "ck", storage=JSONLStorage(tmp_path / "records.jsonl"))
    assert b.versions() == a.versions() and b.n == 1500
    _run(b, data[1500:])
    r = b.replay()
    assert r["ok"] and r["decisions"] == 3000, r["mismatches"][:5]
    with pytest.raises(ValueError, match="records written after the save"):   # b went on in the same store
        _load(tmp_path / "ck", storage=JSONLStorage(tmp_path / "records.jsonl"))
    with pytest.raises(ValueError, match="TraceStorage"):
        _load(tmp_path / "ck")


def test_versions_dir_gets_the_files_and_a_rescored_replay_is_exact(tmp_path):
    data = _fake(11, 5000)
    a = _loop(versions_dir=tmp_path / "va")
    _run(a, data[:3000])
    a.save(tmp_path / "ck")
    b = _load(tmp_path / "ck", versions_dir=tmp_path / "vb")
    assert sorted(p.name for p in (tmp_path / "vb").iterdir()) == sorted(p.name for p in (tmp_path / "va").iterdir())
    _run(b, data[3000:])
    r = b.replay(rescore=True)
    assert r["ok"], r["mismatches"][:5]


# ───────────────────────────── integrity
def test_a_tampered_or_foreign_checkpoint_is_refused(tmp_path):
    a = _loop()
    _run(a, _fake(12, 2500))
    info = a.save(tmp_path / "ck")
    folder = info["path"]
    assert any(f.startswith("versions/") for f in info["files"])
    with pytest.raises(ValueError, match="secret"):
        _load(tmp_path / "ck", secret=None)
    with pytest.raises(ValueError, match="signature"):
        _load(tmp_path / "ck", secret=b"another key")
    with pytest.raises(ValueError, match="learner"):
        _load(tmp_path / "ck", learner=_Other())
    _load(tmp_path / "ck")                           # untouched: loads
    for rel in ("state.json", "labels.jsonl", "records.jsonl", "policy.pkl",
                next(f for f in info["files"] if f.startswith("versions/"))):
        p = folder / rel
        good = p.read_bytes()
        p.write_bytes(good[:-2] + bytes([good[-2] ^ 1]) + good[-1:])
        with pytest.raises(ValueError, match="tampered"):
            _load(tmp_path / "ck")
        p.write_bytes(good)
    m = folder / "manifest.json"
    good = m.read_bytes()
    man = json.loads(good)
    man["n"] += 1
    m.write_bytes(json.dumps(man, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="signature"):
        _load(tmp_path / "ck")
    m.write_bytes(good)
    (folder / "labels.jsonl").unlink()
    with pytest.raises(ValueError, match="missing"):
        _load(tmp_path / "ck")


def test_a_learner_that_is_not_fresh_is_refused(tmp_path):
    a = _loop()
    _run(a, _fake(13, 600))
    a.save(tmp_path / "ck")
    used = _Det()
    used.update("x", "a")
    with pytest.raises(ValueError, match="as built"):
        _load(tmp_path / "ck", learner=used)


# ───────────────────────────── atomic save
def test_a_crash_in_the_middle_of_a_save_leaves_the_previous_checkpoint_usable(tmp_path, monkeypatch):
    data = _fake(14, 4000)
    a = _loop()
    _run(a, data[:1500])
    a.save(tmp_path / "ck")
    _run(a, data[1500:2500])

    def crash(*_, **__):
        raise OSError("power cut")
    with monkeypatch.context() as mp:                # after every file is written, before the switch
        mp.setattr(ckpt_mod, "_switch", crash)
        with pytest.raises(OSError, match="power cut"):
            a.save(tmp_path / "ck")
    assert _load(tmp_path / "ck").n == 1500
    with monkeypatch.context() as mp:                # while the version files are written
        mp.setattr(HeadVersion, "save", crash)
        with pytest.raises(OSError, match="power cut"):
            a.save(tmp_path / "ck")
    b = _load(tmp_path / "ck")
    assert b.n == 1500
    assert not [p for p in (tmp_path / "ck").iterdir() if p.name.startswith(".tmp-")]
    a.save(tmp_path / "ck")                          # the next save works and replaces it
    assert _load(tmp_path / "ck").n == 2500
    assert len([p for p in (tmp_path / "ck").iterdir() if p.name.startswith("ckpt-")]) == 1


# ───────────────────────────── a training in flight
def test_a_training_in_flight_refuses_the_save_unless_cancelled(tmp_path):
    gate = threading.Event()
    a = _loop(learner=_Det(gate), wait=False, policy=_plain(keep=3, growth=1.0, trial=300, first=10 ** 6))
    data = _fake(15, 3000)
    _run(a, data[:300])
    a.train_now()
    _run(a, data[300:350])
    with pytest.raises(RuntimeError, match="in flight"):
        a.save(tmp_path / "ck")
    a.save(tmp_path / "ck", cancel=True)
    ev = _events(a, "cancel")
    assert len(ev) == 1 and ev[0]["cancelled"] == "1.1" and a.status()["job"] is None
    assert a.status()["trainings"] == 1               # it stays counted
    b = _load(tmp_path / "ck", learner=_Det(gate))
    assert b.status() == a.status() and b.log == a.log
    gate.set()
    for x in data[350:]:
        assert a.step(x, lambda x=x: x[0]) == b.step(x, lambda x=x: x[0])
    assert _norm(a.log) == _norm(b.log)
    for lp in (a, b):
        r = lp.replay()
        assert r["ok"], r["mismatches"][:5]
