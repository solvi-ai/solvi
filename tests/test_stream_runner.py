"""solvi.experimental.stream.runner (the adapter loop, stage 3): where a training runs — the subprocess runner (nice,
capped threads, the device picked, the time estimate against max_train_s, a heartbeat log, the result file) and the
thread runner, and how the adapter loop records what a run ends with. No torch: the targets are tiny functions in a
helper module the test writes; the LoRA target itself is tested in test_stream_lora.py."""
import os
import pickle
import random
import textwrap
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy, HashedTextTrainer
    from solvi.experimental.stream.runner import (RunnerError, SubprocessRunner, ThreadRunner, TrainingFailed,
                                                  TrainingRefused, make_runner, pick_device)

KEY = b"runner-test-key"

HELPER = '''
import os, pickle, signal, time, types

def echo(payload, ctx):
    ctx.estimate(payload.get("estimate", 0.01))
    for i in range(payload.get("steps", 1)):
        ctx.progress(i + 1, payload.get("steps", 1))
        time.sleep(payload.get("sleep", 0.0))
    with open(payload["out"], "w") as f:
        f.write(payload.get("text", "trained"))
    return {"file": payload["out"], "device": ctx.device, "threads": ctx.threads, "pid": os.getpid(),
            "nice": os.nice(0), "omp": os.environ.get("OMP_NUM_THREADS")}

def boom(payload, ctx):
    raise ValueError("a bug in the training")

def die(payload, ctx):
    os._exit(3)

def killed(payload, ctx):
    os.kill(os.getpid(), signal.SIGKILL)

def late(payload, ctx):
    """Loads for `pre` s, estimates, trains for `post` s: the overhead before the estimate is not in it."""
    time.sleep(payload["pre"])
    ctx.estimate(payload["estimate"])
    time.sleep(payload["post"])
    return {"ok": True}

def silent(payload, ctx):
    time.sleep(30)

def hashed(payload, ctx):
    """HashedTextTrainer in the training process: the frozen model pickled to a file."""
    from solvi.experimental.stream import HashedTextTrainer
    from solvi.experimental.stream.policy import LabelRow
    ctx.estimate(0.1)
    rows = tuple(LabelRow(*r) for r in payload["rows"])
    job = types.SimpleNamespace(rows=rows, hyper=payload["hyper"], version=payload["version"])
    frozen = HashedTextTrainer(payload["options"]).train(job)
    with open(payload["out"], "wb") as f:
        pickle.dump(frozen, f)
    return {"file": payload["out"], "fingerprint": frozen.fingerprint()}
'''


@pytest.fixture
def helper(tmp_path):
    d = tmp_path / "helper"
    d.mkdir()
    (d / "lt_runner_helper.py").write_text(textwrap.dedent(HELPER))
    import sys
    sys.path.insert(0, str(d))                       # the thread runner imports it in this process
    yield d
    sys.path.remove(str(d))
    sys.modules.pop("lt_runner_helper", None)


def _budget(**kw):
    base = dict(min_interval_s=0.0, per_day=10 ** 6, cpu_threads=2, device="cpu", max_train_s=60.0)
    base.update(kw)
    return Budget(**base)


def _sub(helper, **kw):
    kw.setdefault("heartbeat_s", 0.2)
    kw.setdefault("poll_s", 0.05)
    return SubprocessRunner(pythonpath=[helper], **kw)


# ───────────────────────────── the device
def test_auto_picks_a_gpu_with_enough_free_memory_and_falls_back_to_the_cpu():
    yes = lambda: True                                                       # noqa: E731
    assert pick_device("auto", 3000, cuda=lambda: False)[0] == "cpu"
    dev, why = pick_device("auto", 3000, cuda=yes, probe=lambda: [1200])
    assert dev == "cpu" and "1200 MB" in why
    assert pick_device("auto", 3000, cuda=yes, probe=lambda: [])[0] == "cpu"
    assert pick_device("auto", 3000, cuda=yes, probe=lambda: [100, 5000])[0] == "cuda:1"
    assert pick_device("cpu", 3000, cuda=yes, probe=lambda: [9000]) == ("cpu", "asked")
    assert pick_device("cuda:0", 3000, cuda=lambda: False) == ("cuda:0", "asked")
    with pytest.raises(ValueError):
        pick_device("tpu")


def test_auto_returns_torchs_ordinal_under_cuda_visible_devices():
    yes = lambda: True                                                       # noqa: E731
    smi = lambda: [100, 5000]                                                # noqa: E731 — nvidia-smi sees both GPUs
    assert pick_device("auto", 3000, cuda=yes, probe=smi, env={"CUDA_VISIBLE_DEVICES": "1"})[0] == "cuda:0"
    assert pick_device("auto", 3000, cuda=yes, probe=smi, env={"CUDA_VISIBLE_DEVICES": "1,0"})[0] == "cuda:0"
    dev, why = pick_device("auto", 3000, cuda=yes, probe=smi, env={"CUDA_VISIBLE_DEVICES": "0"})
    assert dev == "cpu" and "100 MB" in why                                  # GPU 1 is hidden from torch
    assert pick_device("auto", 3000, cuda=yes, probe=lambda: [5000, 100], env={"CUDA_VISIBLE_DEVICES": "0,1"})[0] \
        == "cuda:0"
    assert pick_device("auto", 3000, cuda=yes, probe=lambda: [100, 100, 5000],
                       env={"CUDA_VISIBLE_DEVICES": "0,2"})[0] == "cuda:1"
    assert pick_device("auto", 3000, cuda=yes, probe=smi, env={"CUDA_VISIBLE_DEVICES": ""})[0] == "cpu"
    assert pick_device("auto", 3000, cuda=yes, probe=smi, env={})[0] == "cuda:1"           # not set: the same index
    uuid = {"CUDA_VISIBLE_DEVICES": "GPU-8f2a"}                              # by UUID: torch's own free memory
    assert pick_device("auto", 3000, cuda=yes, probe=smi, env=uuid, torch_probe=lambda: [6000])[0] == "cuda:0"


def test_a_subprocess_with_auto_and_too_little_gpu_memory_trains_on_the_cpu(helper, tmp_path):
    out = _sub(helper).run("lt_runner_helper:echo", {"out": str(tmp_path / "m.txt")}, name="t_auto", workdir=tmp_path,
                           budget=_budget(device="auto", free_gpu_mb=10 ** 9))
    assert out["device"] == "cpu"
    assert "auto:" in (tmp_path / "t_auto.log").read_text()


# ───────────────────────────── the subprocess runner
def test_a_subprocess_training_writes_its_file_log_and_heartbeat_in_a_niced_capped_process(helper, tmp_path):
    status = {"eta": None}
    out = _sub(helper).run("lt_runner_helper:echo", {"out": str(tmp_path / "m.txt"), "steps": 4, "sleep": 0.2},
                           name="t1", workdir=tmp_path, budget=_budget(cpu_threads=3), status=status)
    assert out["state"] == "done" and (tmp_path / "m.txt").read_text() == "trained"
    assert out["pid"] != os.getpid() and out["pid"] == status["pid"] and status["runner"] == "subprocess"
    assert out["nice"] >= 10 and out["omp"] == "3" and out["threads"] == 3
    log = (tmp_path / "t1.log").read_text()
    assert "start device=cpu" in log and "estimate 0.0 s" in log and "heartbeat" in log and log.rstrip().endswith("done")
    assert status["log"] == str(tmp_path / "t1.log") and status["eta"] is not None and status["device"] == "cpu"


def test_the_estimate_refuses_a_training_above_max_train_s(helper, tmp_path):
    for runner in (_sub(helper), ThreadRunner(heartbeat_s=0.2)):
        with pytest.raises(TrainingRefused, match="max_train_s 5"):
            runner.run("lt_runner_helper:echo", {"out": str(tmp_path / "x.txt"), "estimate": 50.0}, name=f"t_{runner.kind}",
                       workdir=tmp_path, budget=_budget(max_train_s=5.0))
        assert not (tmp_path / "x.txt").exists()                # refused before it trained


def test_a_failing_target_is_a_failed_training_a_dying_process_a_runner_error(helper, tmp_path):
    with pytest.raises(TrainingFailed, match="a bug in the training"):
        _sub(helper).run("lt_runner_helper:boom", {}, name="t_boom", workdir=tmp_path, budget=_budget())
    with pytest.raises(RunnerError, match="exited with 3 and wrote no result"):
        _sub(helper).run("lt_runner_helper:die", {}, name="t_die", workdir=tmp_path, budget=_budget())
    with pytest.raises(RunnerError, match="memory cap"):
        _sub(helper).run("lt_runner_helper:killed", {}, name="t_kill", workdir=tmp_path, budget=_budget())


def test_a_silent_process_is_killed_and_one_past_its_time_is_stopped(helper, tmp_path):
    with pytest.raises(RunnerError, match="no log line"):
        _sub(helper, heartbeat_s=60.0, stall_s=1.0).run("lt_runner_helper:silent", {}, name="t_silent",
                                                         workdir=tmp_path, budget=_budget())
    with pytest.raises(TrainingFailed, match="ran past max_train_s"):
        _sub(helper, grace_s=0.5).run("lt_runner_helper:silent", {}, name="t_over", workdir=tmp_path,
                                      budget=_budget(max_train_s=0.5))


def test_a_training_within_its_estimate_is_not_killed_for_the_time_before_the_estimate(helper, tmp_path):
    """The deadline runs from the estimate line: 1 s of loading + 2 s of training against an estimate of 2 s and
    max_train_s 2.5 + grace 0.3 is done (from the start it would be 3 s > 2.8 s); 3 s against an estimate of 2 s is
    stopped."""
    late = {"pre": 1.0, "estimate": 2.0, "post": 2.0}
    for runner in (_sub(helper, grace_s=0.3), ThreadRunner(heartbeat_s=0.2, grace_s=0.3)):
        out = runner.run("lt_runner_helper:late", late, name=f"t_late_{runner.kind}", workdir=tmp_path,
                         budget=_budget(max_train_s=2.5))
        assert out["state"] == "done" and out["ok"]
    with pytest.raises(TrainingFailed, match="past its estimate"):
        _sub(helper, grace_s=0.3).run("lt_runner_helper:late", {**late, "post": 3.0}, name="t_late_over",
                                      workdir=tmp_path, budget=_budget(max_train_s=2.5))


def test_the_thread_runner_runs_the_same_target_in_this_process(helper, tmp_path):
    status = {}
    out = ThreadRunner(heartbeat_s=0.1).run("lt_runner_helper:echo", {"out": str(tmp_path / "m.txt"), "steps": 3,
                                                                      "sleep": 0.1},
                                             name="t_th", workdir=tmp_path, budget=_budget(), status=status)
    assert out["pid"] == os.getpid() and status["runner"] == "thread" and (tmp_path / "m.txt").exists()
    with pytest.raises(TrainingFailed, match="a bug"):
        ThreadRunner().run("lt_runner_helper:boom", {}, name="t_th2", workdir=tmp_path, budget=_budget())
    assert isinstance(make_runner("subprocess"), SubprocessRunner) and isinstance(make_runner("thread"), ThreadRunner)
    with pytest.raises(ValueError):
        make_runner("cluster")


# ───────────────────────────── the adapter loop over a runner
class _ProcTrainer(HashedTextTrainer):
    """HashedTextTrainer trained in a runner: the frozen model comes back as a file."""

    def __init__(self, options, runner, dir, target="lt_runner_helper:hashed"):
        super().__init__(options)
        self.runner, self.dir, self.target, self.jobs = runner, dir, target, []

    def train(self, job):
        self.jobs.append(job)
        out = str(self.dir / f"train_{job.id}.pkl")
        payload = {"rows": [(r.key, r.x, r.y, r.source, r.t) for r in job.rows], "hyper": job.hyper,
                   "version": job.version, "options": self._kw["options"], "out": out}
        got = self.runner.run(self.target, payload, name=f"train_{job.id}", workdir=self.dir, budget=job.budget,
                              status=job.status)
        with open(got["file"], "rb") as f:
            frozen = pickle.load(f)  # noqa: S301 — the test's own file
        assert frozen.fingerprint() == got["fingerprint"]
        return frozen


class _Slot:
    """A Served that records what the loop installed (the test's System)."""

    def __init__(self, frozen, sink):
        self.frozen, self.sink = frozen, sink

    def score(self, x):
        return self.frozen.score(x)

    def fingerprint(self):
        return self.frozen.fingerprint()

    def install(self, system):
        before = system.fp
        system.fp = self.frozen.fingerprint()
        self.sink.append(system.fp)
        return {"from": before, "to": system.fp}

    def uninstall(self, system):
        pass

    def save(self, dir, version):
        return []


def _plain(**kw):
    base = dict(calib=0.0, holdout=0.0, pre_freeze=False, probe=False)
    base.update(kw)
    return DefaultPolicy(_budget(), **base)


def _data(seed, n):
    r = random.Random(seed)
    return [(f"text {c} {r.random():.4f}", c) for c in (r.choice("ab") for _ in range(n))]


def _events(loop, *kinds):
    return [e for k, e in loop._items() if k == "stream" and (not kinds or e["event"] in kinds)]


def test_a_subprocess_training_writes_a_file_the_loop_installs(helper, tmp_path):
    import types
    sink, system = [], types.SimpleNamespace(heads={}, fp=None)
    learner = _ProcTrainer(list("ab"), _sub(helper), tmp_path)
    loop = AdapterLoop(learner, question="q", max_risk=0.05, secret=KEY, policy=_plain(first=100), system=system,
                       served=lambda f: _Slot(f, sink))
    for x, y in _data(1, 1500):
        loop.step(x, lambda y=y: y)
        if loop.served != 0:
            break
    assert learner.jobs and loop.served != 0, loop.status()
    job = learner.jobs[loop.served - 1]
    assert job.budget is loop.policy.budget and job.alpha == 0.05
    f = tmp_path / f"train_{job.id}.pkl"
    with open(f, "rb") as fh:
        frozen = pickle.load(fh)  # noqa: S301
    assert system.fp == frozen.fingerprint() == loop.versions()[loop.served]["fingerprint"] and sink == [system.fp]
    assert (tmp_path / f"train_{job.id}.log").read_text().rstrip().endswith("done")
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


class _Crash(HashedTextTrainer):
    def __init__(self, runner, dir, target):
        super().__init__(list("ab"))
        self.runner, self.dir, self.target = runner, dir, target

    def train(self, job):
        return self.runner.run(self.target, {"out": str(self.dir / "x.txt"), "estimate": 1000.0}, name=f"train_{job.id}",
                               workdir=self.dir, budget=job.budget, status=job.status)


def test_a_runner_crash_stops_the_loop_with_status_error_and_the_replay_stays_exact(helper, tmp_path):
    loop = AdapterLoop(_Crash(_sub(helper), tmp_path, "lt_runner_helper:die"), question="q", max_risk=0.05, secret=KEY,
                       policy=_plain(first=50))
    with pytest.raises(RunnerError):
        for x, y in _data(2, 400):
            loop.step(x, lambda y=y: y)
    st = loop.status()
    assert st["state"] == "error" and st["error"]["where"] == "learner.train" and st["job"] is None
    assert "exited with 3" in st["error"]["why"]
    with pytest.raises(RuntimeError, match="stopped on an error"):
        loop.step("text a 0.5", lambda: "a")
    err = _events(loop, "error")
    assert len(err) == 1 and err[0]["where"] == "learner.train"
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]
    # the same through a control
    loop2 = AdapterLoop(_Crash(_sub(helper), tmp_path, "lt_runner_helper:killed"), question="q", max_risk=0.05,
                        secret=KEY, policy=_plain(first=10 ** 6))
    for x, y in _data(3, 60):
        loop2.step(x, lambda y=y: y)
    with pytest.raises(RunnerError):
        loop2.train_now()
    assert loop2.status()["state"] == "error"
    r = loop2.replay()
    assert r["ok"], r["mismatches"][:5]


def test_a_training_refused_by_its_estimate_is_recorded_and_replays(helper, tmp_path):
    clock = [1000.0]
    loop = AdapterLoop(_Crash(_sub(helper), tmp_path, "lt_runner_helper:echo"), question="q", max_risk=0.05, secret=KEY,
                       policy=DefaultPolicy(_budget(max_train_s=10.0, min_interval_s=600.0), calib=0.0, holdout=0.0,
                                            pre_freeze=False, probe=False, first=50), clock=lambda: clock[0])
    for x, y in _data(4, 120):
        loop.step(x, lambda y=y: y)
    tr = _events(loop, "train")
    assert len(tr) == 1 and tr[0]["state"] == "refused" and "max_train_s 10" in tr[0]["why"] and tr[0]["job"]
    assert loop.status()["state"] == "running" and loop.served == 0 and loop.candidate is None
    assert loop.status()["trainings"] == 1                 # a refused training counts against the budget
    assert not (tmp_path / "x.txt").exists()
    clock[0] += 601
    job = loop.train_now()                                 # a control refused by the estimate: one "train" event
    assert job.state == "refused" and "max_train_s" in job.why
    assert sum(1 for e in _events(loop, "train") if e.get("control")) == 1
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]
