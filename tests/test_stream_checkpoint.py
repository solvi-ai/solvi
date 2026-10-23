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
    with pytest.warns(ckpt_mod.CheckpointTrustWarning), pytest.raises(ValueError, match="saved with a secret"):
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


# ───────────────────────────── held-out and calib labels, an alarm after the load
def _shifted(seed, n, lo, hi=1.0):
    r = random.Random(seed)
    return [(r.choice("ab"), lo + (hi - lo) * r.random()) for _ in range(n)]


def test_a_loop_with_held_out_and_calib_labels_continues_through_an_alarm_after_the_load(tmp_path):
    """holdout and calib shares, the pre-freeze gates on: at the save a promoted version is served with a held-out
    window of its own; after the load the scores shift, the served version alarms and is recertified. The state the
    policy reads (held-out window, served_since, alarms, ...) equals the saved loop's, and so does everything after."""
    pol = _plain(keep=3, growth=1.0, trial=300, first=200, calib=0.2, holdout=0.3, pre_freeze=True)
    a = _loop(policy=pol)
    save_at = 5300
    data = _fake(21, 1500) + _shifted(22, 4000, 0.7) + _shifted(23, 400, 0.0, 0.3) + _shifted(24, 2600, 0.7)
    _run(a, data[:save_at])
    assert a.served != 0 and a._served_since > 0 and len(a._holdout) > 0, (a.served, a._served_since)
    assert a.status()["labels"]["calib"] > 0 and a.status()["labels"]["holdout"] > 0, a.status()["labels"]
    served = a.served
    a.save(tmp_path / "ck")
    b = _load(tmp_path / "ck")
    assert b._state() == a._state()                  # everything the policy reads, the held-out window included
    for i, x in enumerate(data[save_at:], start=save_at + 1):
        assert a.step(x, lambda x=x: x[0]) == b.step(x, lambda x=x: x[0]), i
    assert _norm(a.log) == _norm(b.log) and a.versions() == b.versions() and a._state() == b._state()
    after = [e for e in _events(b) if e["decision"] > save_at]
    alarms = [e for e in after if e["event"] == "alarm"]
    assert alarms and alarms[0]["version"] == served, [e["event"] for e in after]
    assert {"alarm", "recertify"} <= {e["event"] for e in after}, {e["event"] for e in after}
    r = b.replay()
    assert r["ok"] and r["decisions"] == len(data), r["mismatches"][:5]


# ───────────────────────────── trust: unlisted files, the secret
_PLANTED = []


def _planted_ran():
    _PLANTED.append(1)


class _Boom:
    def __reduce__(self):
        return (_planted_ran, ())


def test_an_unlisted_file_in_the_checkpoint_is_refused_and_never_run(tmp_path):
    import pickle
    a = _loop(policy=_plain(keep=1, growth=1.0, trial=300, first=200))   # no versions_dir: an evicted version has no file
    _run(a, _fake(31, 9000))
    info = a.save(tmp_path / "ck")
    listed = {f for f in info["files"] if f.startswith("versions/")}
    gone = [v["version"] for v in a.versions() if not any(f.startswith(f"versions/{v['version']}.") for f in listed)]
    assert gone, (a.versions(), listed)
    planted = info["path"] / "versions" / f"{gone[0]}.pkl"
    planted.write_bytes(pickle.dumps(_Boom()))
    _PLANTED.clear()
    with pytest.raises(ValueError, match="unlisted"):
        _load(tmp_path / "ck", versions_dir=tmp_path / "vb")
    assert not _PLANTED
    planted.unlink()
    (info["path"] / "notes.txt").write_text("x")      # anywhere in the folder
    with pytest.raises(ValueError, match="unlisted"):
        _load(tmp_path / "ck")
    (info["path"] / "notes.txt").unlink()
    b = _load(tmp_path / "ck", versions_dir=tmp_path / "vb")   # versions_dir gets the listed files only, verified bytes
    got = sorted(p.name for p in (tmp_path / "vb").iterdir())
    assert got == sorted(f[len("versions/"):] for f in listed if not f.startswith("versions/0."))
    for name in got:
        assert (tmp_path / "vb" / name).read_bytes() == (info["path"] / "versions" / name).read_bytes()
    r = b.replay(rescore=True)
    assert not _PLANTED and r["decisions"] == 9000


def _resign(folder, secret=None, **changes):
    """Rewrite the manifest with changes and sign it as a writer of the folder would (secret: an HMAC, else SHA-256)."""
    m = json.loads((folder / "manifest.json").read_bytes())
    m.update(changes)
    mb = ckpt_mod._dumps(m)
    (folder / "manifest.json").write_bytes(mb)
    (folder / "manifest.sig").write_bytes(ckpt_mod._dumps(ckpt_mod._sig(secret, mb)))


def test_a_keyed_checkpoint_needs_its_secret_and_a_load_without_one_warns(tmp_path):
    a = _loop()
    _run(a, _fake(12, 1200))
    folder = a.save(tmp_path / "ck")["path"]
    m = json.loads((folder / "manifest.json").read_bytes())
    assert m["keyed"] is True and json.loads((folder / "manifest.sig").read_bytes())["kind"] == "hmac-sha256"
    with pytest.warns(ckpt_mod.CheckpointTrustWarning), pytest.raises(ValueError, match="saved with a secret"):
        _load(tmp_path / "ck", secret=None)
    good_m, good_s = (folder / "manifest.json").read_bytes(), (folder / "manifest.sig").read_bytes()
    _resign(folder, None, keyed=True)                  # keyed kept, a plain SHA-256 signature: still needs the secret
    with pytest.warns(ckpt_mod.CheckpointTrustWarning), pytest.raises(ValueError, match="saved with a secret"):
        _load(tmp_path / "ck", secret=None)
    with pytest.raises(ValueError, match="signature"):
        _load(tmp_path / "ck")
    _resign(folder, None, keyed=False)                 # a downgrade by a writer of the folder: the secret's holder
    with pytest.raises(ValueError, match="without a secret"):   # refuses it
        _load(tmp_path / "ck")
    with pytest.warns(ckpt_mod.CheckpointTrustWarning):   # without the secret it is indistinguishable from an unkeyed
        with warnings.catch_warnings():                    # checkpoint: the documented limit, hence the warning
            warnings.filterwarnings("ignore", message=".*calibration coin")
            _load(tmp_path / "ck", secret=None)
    (folder / "manifest.json").write_bytes(good_m)
    (folder / "manifest.sig").write_bytes(good_s)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ckpt_mod.CheckpointTrustWarning)
        _load(tmp_path / "ck")                         # with the secret: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = _loop(secret=None)
    _run(u, _fake(13, 600))
    u.save(tmp_path / "cu")
    with pytest.warns(ckpt_mod.CheckpointTrustWarning, match="not tampering"):
        b = _load(tmp_path / "cu", secret=None)
    assert b.versions() == u.versions()


# ───────────────────────────── concurrent saves
def _current_ok(root):
    name = (root / "CURRENT").read_text().strip()
    return (root / name / "manifest.json").is_file(), name


def test_two_saves_of_one_loop_never_leave_current_naming_a_removed_folder(tmp_path, monkeypatch):
    a = _loop()
    _run(a, _fake(16, 800))
    root = tmp_path / "ck"
    # forced interleaving: save A pauses in its cleanup while save B is started; B must wait for A (the cleanup is
    # under the loop's lock), else A's cleanup would remove B's folder after B switched CURRENT to it
    real, in_cleanup, b_done, first = ckpt_mod._cleanup, threading.Event(), threading.Event(), []

    def slow(r, name):
        if not first and threading.current_thread().name == "save-a":
            first.append(1)
            in_cleanup.set()
            b_done.wait(1.0)
        real(r, name)
    monkeypatch.setattr(ckpt_mod, "_cleanup", slow)
    errs = []

    def save_b():
        try:
            in_cleanup.wait(10)
            a.save(root)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
        finally:
            b_done.set()
    ta = threading.Thread(target=lambda: a.save(root), name="save-a")
    tb = threading.Thread(target=save_b)
    ta.start(), tb.start()
    ta.join(), tb.join()
    assert not errs, errs
    ok, name = _current_ok(root)
    assert ok, (name, sorted(p.name for p in root.iterdir()))
    monkeypatch.setattr(ckpt_mod, "_cleanup", real)
    # and free-running: two threads, many saves; after every round CURRENT names a complete folder
    bar = threading.Barrier(2)

    def many():
        try:
            for _ in range(40):
                a.save(root)
                bar.wait()
                ok, name = _current_ok(root)
                if not ok:
                    errs.append(name)
                bar.wait()
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
            bar.abort()
    th = [threading.Thread(target=many) for _ in range(2)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert not errs, errs[:3]
    assert _load(root).n == 800
    assert [p.name for p in root.iterdir() if p.name.startswith("ckpt-")] == [_current_ok(root)[1]]


def test_a_version_file_swapped_after_the_check_is_never_read(tmp_path, monkeypatch):
    """The version loaders read the verified bytes (a private copy), not the checkpoint folder: a file replaced between
    the hash check and the load is never read, its pickle never run."""
    import pickle
    a = _loop()
    _run(a, _fake(12, 2500))
    info = a.save(tmp_path / "ck")
    vfiles = sorted(f for f in info["files"] if f.startswith("versions/") and f.endswith(".pkl"))
    assert vfiles, info["files"]
    real = ckpt_mod._verified

    def swap_after_check(folder, secret):
        out = real(folder, secret)
        for rel in vfiles:                         # every version file replaced once its hash was checked
            (folder / rel).write_bytes(pickle.dumps(_Boom()))
        return out

    monkeypatch.setattr(ckpt_mod, "_verified", swap_after_check)
    _PLANTED.clear()
    b = _load(tmp_path / "ck")
    assert not _PLANTED
    assert b.versions() == a.versions() and b.status() == a.status()
    data = _fake(13, 1500)
    assert _run(a, data) == _run(b, data)
