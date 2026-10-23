"""solvi.experimental.stream, the adapter loop (stage 3b): deferred apply — a training runs in the background, the loop
keeps deciding on the served version, the result is applied on a later decision at a recorded position and replays
exactly; close() cancels a training in flight; Budget.max_versions_kept evicts old retired versions; the lock order
(the part's version lock, then the loop's) has no deadlock with a decision that reaches the loop under the part's lock.
No torch: the LoRA end to end is in test_stream_lora.py."""
import os
import random
import threading
import time
import warnings

import pytest

from test_stream_runner import _sub, helper  # noqa: F401 — the subprocess helper module and its runner

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.core.deciders.part import version_lock
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy, HashedTextTrainer
    from solvi.experimental.stream.runner import RunnerError, TrainingCancelled, TrainingRefused

KEY = b"deferred-apply-test-key"
STEP_BOUND_S = 0.5                                   # no decision waits for a training


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
    """x = (label, u): right with score 0.95 when u < share, else a wrong answer with score 0.05. released: how many
    times the loop called release()."""

    def __init__(self, name, share):
        self.name, self.share, self.released = name, share, 0

    def score(self, x):
        return (x[0], 0.95) if x[1] < self.share else ("wrong", 0.05)

    def fingerprint(self):
        return f"model-{self.name}-{self.share}"

    def release(self):
        self.released += 1


class _Learner:
    """A batch learner: each training takes the next outcome (a model, or an exception to raise), after `gate` is set
    (or `delay` seconds); it gives up when the job is cancelled."""

    def __init__(self, outcomes, gate=None, delay=0.0):
        self.outcomes, self.gate, self.delay = list(outcomes), gate, delay
        self._seen, self.jobs, self.cancelled = 0, [], []

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        self.jobs.append(job)
        t_end = time.monotonic() + (self.delay() if callable(self.delay) else self.delay)
        while (self.gate is not None and not self.gate.is_set()) or time.monotonic() < t_end:
            if job.cancel is not None and job.cancel.wait(0.01):
                self.cancelled.append(job.id)
                raise TrainingCancelled(job.id)
            if self.gate is None and time.monotonic() >= t_end:
                break
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def _step(loop, x):
    t0 = time.monotonic()
    out = loop.step(x, lambda: x[0])
    return out, time.monotonic() - t0


# ───────────────────────────── decisions go on while a training runs; the apply is at a recorded position
def test_decisions_go_on_while_a_slow_training_runs_and_the_apply_is_recorded_where_it_happened():
    gate = threading.Event()
    learner = _Learner([_Model("m", 0.9)], gate=gate)
    loop = AdapterLoop(learner, question="q", max_risk=0.05, secret=KEY, policy=_plain(first=20))
    data = iter(_fake(1, 5000))
    while not learner.jobs:
        _step(loop, next(data))
    start = _events(loop, "train_start")
    assert len(start) == 1 and start[0]["inline"] is False and start[0]["trigger"] == "schedule"
    s = start[0]["decision"]
    worst = 0.0
    for _ in range(300):                             # the training waits for the gate: every decision goes on
        out, dt = _step(loop, next(data))
        worst = max(worst, dt)
        assert out["candidate"] is None and out["served"] == 0
    assert worst < STEP_BOUND_S, worst
    st = loop.status()
    assert st["state"] == "training" and st["job"]["state"] == "running" and st["job"]["started_at"] == s
    assert loop.train_now().why == "a training is in flight"
    gate.set()
    assert loop.wait_training(10)
    assert loop.status()["job"]["state"] == "ready"
    out, _ = _step(loop, next(data))
    tr = [e for e in _events(loop, "train") if e.get("applied")]     # the refused train_now above: a control event
    assert len(tr) == 1 and tr[0]["state"] == "done" and tr[0]["frozen"]
    assert tr[0]["decision"] == out["n"] > s and tr[0]["started_at"] == s and tr[0]["applied"] == "decision"
    assert "train" in out["events"] and "freeze" in out["events"] and loop.candidate == 1
    assert loop.versions()[1]["frozen_at"] == out["n"] and loop.status()["job"] is None
    for _ in range(200):
        _step(loop, next(data))
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_train_now_returns_at_once_and_wait_true_blocks():
    gate = threading.Event()
    loop = AdapterLoop(_Learner([_Model("m", 0.9), _Model("m2", 0.9)], gate=gate), question="q", max_risk=0.05,
                       secret=KEY, policy=_plain(first=10 ** 6))
    data = iter(_fake(2, 20000))
    for _ in range(50):
        _step(loop, next(data))
    t0 = time.monotonic()
    job = loop.train_now()
    assert job.state == "running" and time.monotonic() - t0 < STEP_BOUND_S and loop.candidate is None
    gate.set()
    loop.wait_training(10)
    out, _ = _step(loop, next(data))
    assert loop.candidate == 1 and _events(loop, "train")[0]["decision"] == out["n"]
    while loop.candidate is not None:                # promoted or discarded by the policy
        _step(loop, next(data))
    job2 = loop.train_now(wait=True)                 # blocking: applied inside the control
    assert job2.state == "done" and loop.candidate == 2
    tr = _events(loop, "train")[-1]
    assert tr["applied"] == "train_now" and tr["decision"] == loop.n
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_a_paused_loop_keeps_a_ready_result_until_resume():
    loop = AdapterLoop(_Learner([_Model("m", 0.9)]), question="q", max_risk=0.05, secret=KEY, policy=_plain(first=20))
    data = iter(_fake(3, 2000))
    while not loop.status()["job"]:
        _step(loop, next(data))
    loop.pause()
    loop.wait_training(10)
    for _ in range(30):
        _step(loop, next(data))
    assert loop.candidate is None and loop.status()["job"]["state"] == "ready"
    loop.resume()
    out, _ = _step(loop, next(data))
    assert loop.candidate == 1 and _events(loop, "train")[0]["decision"] == out["n"]
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


# ───────────────────────────── replay exact across apply, failure, refusal, discard and eviction
def test_replay_is_exact_across_apply_failure_refusal_discard_and_eviction():
    r0 = random.Random(7)
    outcomes = [_Model("s60", 0.6), ValueError("a bug in the training"), TrainingRefused("estimated 99 s > 1"),
                _Model("s75", 0.75), _Model("bad", 0.1), _Model("s90", 0.9), _Model("s97", 0.97),
                _Model("s99", 0.995)] + [_Model(f"w{i}", 0.3) for i in range(60)]
    models = [o for o in outcomes if isinstance(o, _Model)]
    loop = AdapterLoop(_Learner(outcomes, delay=lambda: r0.random() * 0.02), question="q", max_risk=0.05, secret=KEY,
                       policy=_plain(keep=2, growth=1.0, trial=300, first=200))
    keep = loop.policy.budget.max_versions_kept
    data = iter(_fake(8, 30000))
    for _ in range(14000):
        _step(loop, next(data))
        held = [v for v in loop.versions() if v["state"] in ("served", "retired")]
        protect = {loop.served, loop._rollback_target(), 0}
        assert len(held) <= keep or all(v["version"] in protect for v in held), held
    states = {e["state"] for e in _events(loop, "train")}
    assert {"done", "failed", "refused"} <= states, states
    assert _events(loop, "discard"), "no candidate was discarded"
    ev = _events(loop, "evict")
    assert ev, [(v["version"], v["state"]) for v in loop.versions()]
    for e in ev:                                     # released once, never the served version or the rollback target
        m = next(m for m in models if m.fingerprint() == loop.versions()[e["version"]]["fingerprint"])
        assert m.released == 1 and loop.versions()[e["version"]]["state"] == "evicted"
        with pytest.raises(ValueError, match="not a retired version"):
            loop.rollback(e["version"])
    assert all(e["decision"] > e["started_at"] for e in _events(loop, "train"))
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_a_runner_error_in_the_background_stops_the_loop_where_it_is_applied_and_replays():
    loop = AdapterLoop(_Learner([RunnerError("the training process 1: no log line for 300 s")]), question="q",
                       max_risk=0.05, secret=KEY, policy=_plain(first=20))
    data = iter(_fake(4, 2000))
    while not loop.status()["job"]:
        _step(loop, next(data))
    loop.wait_training(10)
    with pytest.raises(RunnerError):
        _step(loop, next(data))
    st = loop.status()
    assert st["state"] == "error" and st["error"]["where"] == "learner.train" and st["job"] is None
    err = _events(loop, "error")
    assert len(err) == 1 and err[0]["train"]["job"] == "1.1" and err[0]["train"]["applied"] == "decision"
    with pytest.raises(RuntimeError, match="stopped on an error"):
        _step(loop, next(data))
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


# ───────────────────────────── close cancels the training in flight
def test_close_cancels_a_running_training_in_a_thread():
    learner = _Learner([_Model("m", 0.9)], gate=threading.Event())
    loop = AdapterLoop(learner, question="q", max_risk=0.05, secret=KEY, policy=_plain(first=20))
    data = iter(_fake(5, 2000))
    while not learner.jobs:
        _step(loop, next(data))
    th = loop._run["thread"]
    e = loop.close(timeout=10)
    assert e["event"] == "close" and e["cancelled"] == "1.1" and not th.is_alive()
    assert learner.cancelled == ["1.1"] and loop.status()["state"] == "closed" and loop.close() is None
    with pytest.raises(RuntimeError, match="closed"):
        _step(loop, next(data))
    with pytest.raises(RuntimeError, match="closed"):
        loop.train_now()
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


class _ProcSlow(HashedTextTrainer):
    """A training in a subprocess that runs for ~50 s unless cancelled."""

    def __init__(self, runner, dir):
        super().__init__(list("ab"))
        self.runner, self.dir = runner, dir

    def train(self, job):
        payload = {"out": str(self.dir / "x.txt"), "estimate": 50.0, "steps": 1000, "sleep": 0.05}
        return self.runner.run("lt_runner_helper:echo", payload, name=f"train_{job.id}", workdir=self.dir,
                               budget=job.budget, status=job.status, cancel=job.cancel)


def test_close_kills_a_running_training_subprocess(helper, tmp_path):  # noqa: F811
    loop = AdapterLoop(_ProcSlow(_sub(helper), tmp_path), question="q", max_risk=0.05, secret=KEY,
                       policy=DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6, cpu_threads=1, device="cpu",
                                                   max_train_s=120.0), calib=0.0, holdout=0.0, pre_freeze=False,
                                            probe=False, first=20))
    data = [(f"text {c} {i}", c) for i, c in enumerate("ab" * 200)]
    i = 0
    while not loop.status()["job"]:
        x, y = data[i]
        loop.step(x, lambda y=y: y)
        i += 1
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not ((tmp_path / "train_1.1.log").exists()
                                               and (loop.status()["job"] or {}).get("pid")):
        time.sleep(0.05)
    pid = loop.status()["job"]["pid"]
    os.kill(pid, 0)                                  # alive
    loop.close(timeout=20)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not (tmp_path / "x.txt").exists()


# ───────────────────────────── lock order: no deadlock with a decision that reaches the loop under the part's lock
class _Part:
    """Stands for a DecisionPart: only its version lock matters here."""


class _Shadow(_Model):
    """Scores under the part's version lock, as LoraVersion.score does (solvi.experimental.lora.scoring)."""

    def __init__(self, part, name, share):
        super().__init__(name, share)
        self.part = part

    def score(self, x):
        with version_lock(self.part):
            time.sleep(0.0005)
            return super().score(x)


def test_a_decision_under_the_part_lock_and_a_loop_step_do_not_deadlock():
    """Thread A is a decision of the part that reaches the loop (a Dispatcher hook under the part's version lock);
    thread B steps the loop, whose shadow scoring takes the part's lock. With the loop's lock taken first in B this
    deadlocks; the loop takes the part's lock first."""
    part = _Part()
    learner = _Learner([_Shadow(part, f"m{i}", 0.9) for i in range(30)])
    learner.part = part
    loop = AdapterLoop(learner, question="q", max_risk=0.05, secret=KEY, initial=_Shadow(part, "v0", 0.6),
                       policy=_plain(first=20, growth=1.0, trial=100))
    assert loop._part_lock is version_lock(part)
    errors = []

    def a():
        try:
            for x in _fake(10, 400):
                with version_lock(part):
                    time.sleep(0.0002)
                    loop.step(x, lambda x=x: x[0])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def b():
        try:
            for x in _fake(11, 400):
                loop.step(x, lambda x=x: x[0])
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=a, daemon=True), threading.Thread(target=b, daemon=True)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in ts), "deadlock: a thread is still waiting after 60 s"
    assert not errors, errors
    assert loop.n == 800 and _events(loop, "train_start")
    loop.close()


# ───────────────────────────── stage-3b review fixes
class _Counted(_Model):
    """A model that counts its score calls (the served side of a training's report)."""

    def __init__(self, name, share, delay=0.0):
        super().__init__(name, share)
        self.calls, self.delay = 0, delay

    def score(self, x):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return super().score(x)


class _Gated(_Learner):
    """A batch learner with one gate per training (None: no wait)."""

    def __init__(self, outcomes, gates):
        super().__init__(outcomes)
        self.gates = list(gates)

    def train(self, job):
        self.gate = self.gates.pop(0) if self.gates else None
        return super().train(job)


def test_a_rollback_during_a_training_makes_its_result_stale_discarded_and_released():
    """S1 (the review's probe 1): version 1 served, a training starts, rollback(0), 400 decisions, the result comes in.
    The report is not gated on the stale comparison with version 1: the "train" event has state "stale" and names the
    served version at the start, the candidate is discarded and released, version 1 is never scored again, replay
    exact."""
    g1, g2 = threading.Event(), threading.Event()
    g1.set()
    m2 = _Counted("m2", 0.95)
    v1 = _Counted("m1", 0.9)
    learner = _Gated([v1, m2], gates=[g1, g2])
    loop = AdapterLoop(learner, question="q", max_risk=0.05, secret=KEY, initial=_Model("v0", 0.6),
                       policy=_plain(first=20, growth=1.0, trial=300, holdout=0.5))
    data = iter(_fake(1, 60000))
    while loop.served != 1:
        _step(loop, next(data))
        assert loop.n < 50000, "version 1 was never served"
    job = loop.train_now(wait=False)
    assert job.state == "running"
    assert _events(loop, "train_start")[-1]["served_at_start"] == 1
    loop.rollback(0)
    for _ in range(400):
        _step(loop, next(data))
    calls = v1.calls
    g2.set()
    assert loop.wait_training(10)
    assert v1.calls == calls                          # the report stopped: version 1 (no longer served) not scored
    out, _ = _step(loop, next(data))
    tr = _events(loop, "train")[-1]
    assert tr["job"] == job.id and tr["state"] == "stale" and tr["served_at_start"] == 1 and not tr["frozen"]
    assert "v1 → v0" in tr["why"] and tr["decision"] == out["n"]
    dis = _events(loop, "discard")[-1]
    assert dis["version"] == 2 and dis["why"].startswith("stale:")
    assert loop.candidate is None and loop.versions()[2]["state"] == "discarded" and m2.released == 1
    assert job.state == "stale"
    for _ in range(200):
        _step(loop, next(data))
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_the_report_stops_when_the_training_is_cancelled():
    """S1: close() during a long held-out pass does not wait for the whole pass."""
    gate = threading.Event()
    slow = _Counted("slow", 0.9, delay=0.01)
    loop = AdapterLoop(_Gated([slow], gates=[gate]), question="q", max_risk=0.05, secret=KEY,
                       initial=_Model("v0", 0.6), policy=_plain(first=10 ** 6, holdout=0.9))
    for x in _fake(2, 1500):
        _step(loop, x)
    assert loop.status()["labels"]["holdout"] > 300
    loop.train_now()
    th = loop._run["thread"]
    gate.set()
    deadline = time.monotonic() + 10
    while slow.calls < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    t0 = time.monotonic()
    loop.close(timeout=10)
    assert not th.is_alive() and time.monotonic() - t0 < 2.0
    assert slow.calls < 200


def test_the_rollback_target_is_never_evicted():
    """S2: after every "evict" event the version served before the served one is "retired", a rollback to it succeeds
    (then the loop rolls forward again), and version 0 is "served" or "retired" throughout."""
    outcomes = [_Model(f"s{i}", min(0.6 + 0.05 * i, 0.995)) for i in range(40)]
    loop = AdapterLoop(_Learner(outcomes), question="q", max_risk=0.05, secret=KEY, wait=True,
                       policy=_plain(keep=2, growth=1.0, trial=300, first=200))
    data = iter(_fake(8, 60000))
    checked = 0
    seen_evicts = 0
    while checked < 2:
        _step(loop, next(data))
        assert loop.n < 55000, [(v["version"], v["state"]) for v in loop.versions()]
        assert loop.versions()[0]["state"] in ("served", "retired")
        ev = _events(loop, "evict")
        if len(ev) == seen_evicts:
            continue
        seen_evicts = len(ev)
        served = [e["to"] for e in _events(loop, "promote", "rollback")]
        cur = loop.served
        before = [v for v in [0] + served if v != cur]
        target = before[-1]
        assert loop.versions()[target]["state"] == "retired", (target, loop.versions())
        loop.rollback(target)
        assert loop.served == target
        loop.rollback(cur)
        assert loop.served == cur
        checked += 1
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_a_candidate_refused_at_the_freeze_is_released_at_once():
    """S3: with the pre-freeze gates on, a candidate that fails them is released when it is discarded at birth."""
    bad = _Model("bad", 0.1)
    loop = AdapterLoop(_Learner([bad]), question="q", max_risk=0.05, secret=KEY, initial=_Model("v0", 0.9),
                       policy=_plain(first=10 ** 6, pre_freeze=True, holdout=0.3, calib=0.2))
    for x in _fake(3, 3000):
        _step(loop, x)
    job = loop.train_now()
    assert job.state == "running"
    assert loop.wait_training(10)
    data = iter(_fake(4, 100))
    _step(loop, next(data))
    tr = _events(loop, "train")[-1]
    assert tr["state"] == "done" and tr["frozen"] is False
    dis = _events(loop, "discard")[-1]
    assert dis["version"] == 1 and dis["why"].startswith("freeze refused")
    assert bad.released == 1 and loop.versions()[1]["state"] == "discarded" and loop.candidate is None
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_an_error_stop_names_the_cancelled_training():
    class _Failing(_Learner):
        fail_at = None

        def update(self, x, y, **_):
            if self.fail_at is not None and self._seen + 1 == self.fail_at:
                raise RuntimeError("the label store is gone")
            self._seen += 1
    learner = _Failing([_Model("m", 0.9)], gate=threading.Event())
    loop = AdapterLoop(learner, question="q", max_risk=0.05, secret=KEY, policy=_plain(first=20))
    data = iter(_fake(3, 5000))
    while not learner.jobs:
        _step(loop, next(data))
    learner.fail_at = learner.seen() + 3
    with pytest.raises(RuntimeError, match="label store"):
        for _ in range(200):
            _step(loop, next(data))
    err = _events(loop, "error")[-1]
    assert err["cancelled"] == "1.1" and err["train"] is None
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_train_now_without_wait_on_an_online_learner_records_the_wait_asked():
    class _Online:
        def __init__(self):
            self._seen = 0

        def update(self, x, y, **_):
            self._seen += 1

        def seen(self):
            return self._seen

        def snapshot(self):
            return _Model(f"snap{self._seen}", 0.9)
    loop = AdapterLoop(_Online(), question="q", max_risk=0.05, secret=KEY, policy=_plain(first=10 ** 6))
    for x in _fake(2, 30):
        _step(loop, x)
    job = loop.train_now(wait=False)
    ts = _events(loop, "train_start")[-1]
    assert job.state == "done" and ts["wait"] is False and ts["inline"] is True and loop.candidate == 1
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_a_keyboard_interrupt_in_the_training_thread_is_a_failed_training():
    loop = AdapterLoop(_Learner([KeyboardInterrupt()]), question="q", max_risk=0.05, secret=KEY,
                       policy=_plain(first=10 ** 6))
    data = iter(_fake(5, 200))
    for _ in range(30):
        _step(loop, next(data))
    loop.train_now()
    assert loop.wait_training(10)
    _step(loop, next(data))
    tr = _events(loop, "train")[-1]
    assert tr["state"] == "failed" and tr["why"].startswith("KeyboardInterrupt") and loop.status()["state"] != "error"
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]
