"""The LoRA learner of the adapter loop (stage 3): solvi.experimental.lora.swap (an adapter and its thresholds change
together), LoraVersion (score in shadow, install / rollback bit-exact, files), LoraLearner in the thread and the
subprocess runner (the device, the time estimate), and the loop installing a version trained in a subprocess. On the
tiny random checkpoint of test_lora.py; skipped without torch, transformers, tokenizers and peft."""
import threading
import warnings

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

from test_lora import TEAMS, _threads, _toy, ckpt  # noqa: E402,F401 — the tiny checkpoint and its two threads

from solvi.core.deciders import DecideModel  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental import lora  # noqa: E402
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy  # noqa: E402
    from solvi.experimental.stream.learner import TrainJob  # noqa: E402
    from solvi.experimental.stream.lora_learner import LoraLearner, LoraVersion  # noqa: E402
    from solvi.experimental.stream.policy import LabelRow  # noqa: E402
    from solvi.experimental.stream.runner import SubprocessRunner, ThreadRunner, TrainingRefused  # noqa: E402

KEY = b"lora-loop-test-key"


def _part(ckpt):
    m = DecideModel.load(ckpt, backend="torch", device="cpu")
    return m, m.decision("team", "Which team?", "email", TEAMS)


def _rows(data, start=0):
    return tuple(LabelRow(f"k{start + i}", x, y, "human", start + i) for i, (x, y) in enumerate(data))


def _budget(**kw):
    base = dict(min_interval_s=0.0, per_day=10 ** 6, cpu_threads=2, device="cpu", max_train_s=600.0)
    base.update(kw)
    return Budget(**base)


def _job(i=1, n=80, budget=None, alpha=0.2, updates=60):
    return TrainJob(id=f"{i}.1", version=i, rows=_rows(_toy(n, 1)), calib=_rows(_toy(60, 5), 10_000),
                    hyper={"r": 8, "lr": 1e-2, "updates": updates, "seed": 0, "warm_start": False},
                    budget=budget or _budget(), alpha=alpha, status={})


@pytest.fixture(scope="module")
def trained(ckpt, tmp_path_factory):
    """A part calibrated without an adapter (version 0) and a version trained by the thread runner."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
        part.act_guard(_toy(60, 7), max_risk=0.3)
        learner = LoraLearner(part, tmp_path_factory.mktemp("lora_loop"), runner=ThreadRunner(heartbeat_s=0.5))
        v0 = learner.version0()
        v1 = learner.train(_job())
    return m, part, learner, v0, v1


def test_a_version_trained_in_a_thread_has_an_adapter_a_calibration_and_scores_in_shadow(trained):
    m, part, learner, v0, v1 = trained
    assert v1.adapter is not None and v0.adapter is None and v1.fingerprint() != v0.fingerprint()
    assert v1.calibration["guarantee"]["risk"] == 0.2 and v0.calibration["guarantee"]["risk"] == 0.3
    assert v1.act_threshold == v1.calibration["escalate_below"] and v1.info["device"] == "cpu"
    log = (learner.dir / "train_1.1.log").read_text()
    assert "estimate" in log and "phase calibrate" in log and log.rstrip().endswith("done")
    fp, mfp = part.fingerprint(), m.fingerprint()
    t = _toy(6, 9)
    before = [part.decide(x).probs for x, _ in t]
    s1 = [v1.score(x) for x, _ in t]                       # in shadow: the served part is untouched
    assert part.fingerprint() == fp and m.fingerprint() == mfp and part.lora is None
    assert [part.decide(x).probs for x, _ in t] == before
    assert [v0.score(x) for x, _ in t] != s1
    v1.install()
    try:
        for (x, _), (a, s) in zip(t, s1):                  # the shadow score is the served one, once installed
            d = part.decide(x)
            assert d.value == a and abs(float(d.conf) - s) < 1e-12 and d.extra["lora"]["adapter"] == v1.adapter.hash
    finally:
        v0.install()


def test_swap_then_rollback_is_bit_exact(trained):
    m, part, learner, v0, v1 = trained
    t = [x for x, _ in _toy(8, 11)]
    fp0, mfp0, probs0 = part.fingerprint(), m.fingerprint(), [part.decide(x).probs for x in t]
    g0 = (part.escalate_below, part.guarantee)
    out = v1.install()
    assert out["from"] == fp0 and out["to"] == part.fingerprint() != fp0 and m.fingerprint() != mfp0
    fp1, probs1 = part.fingerprint(), [part.decide(x).probs for x in t]
    assert probs1 != probs0 and part.lora.hash == v1.adapter.hash and part.guarantee["risk"] == 0.2
    v0.install()                                          # the rollback
    assert part.fingerprint() == fp0 and m.fingerprint() == mfp0 and part.lora is None
    assert [part.decide(x).probs for x in t] == probs0 and (part.escalate_below, part.guarantee) == g0
    v1.install()                                          # and forward again: the same version to the bit
    assert part.fingerprint() == fp1 and [part.decide(x).probs for x in t] == probs1
    v1.uninstall()                                        # back to what that install replaced
    assert part.fingerprint() == fp0 and [part.decide(x).probs for x in t] == probs0


def test_a_concurrent_scorer_never_sees_an_adapter_without_its_thresholds(trained, swaps=60):
    """Two threads decide while a third swaps version 1 in and out: every decision's probabilities, adapter and
    guarantee are one version's (the adapter it records answered it, with that adapter's own thresholds)."""
    m, part, learner, v0, v1 = trained
    want = {None: v0.calibration["guarantee"]["risk"], v1.adapter.hash: v1.calibration["guarantee"]["risk"]}
    texts = [x for x, _ in _toy(10, 13)]
    expected = {}
    for v, ad in ((v1, v1.adapter.hash), (v0, None)):
        v.install()
        expected[ad] = {x: part.decide(x).probs for x in texts}
    seen, bad, stop = [], [], threading.Event()

    def score():
        i = 0
        while not stop.is_set():
            x = texts[i % len(texts)]
            d = part.decide(x)
            i += 1
            ad = (d.extra.get("lora") or {}).get("adapter")
            g = (d.extra.get("guarantee") or {}).get("risk")
            seen.append(ad)
            if want.get(ad, "?") != g or d.probs != expected[ad][x]:
                bad.append((ad, g))
            assert v1.score(texts[i % len(texts)])[0] in TEAMS     # shadow scoring on the same threads mixes in

    th = [threading.Thread(target=score) for _ in range(2)]
    for x in th:
        x.start()
    try:
        for _ in range(swaps):
            v1.install()
            v0.install()
    finally:
        stop.set()
        for x in th:
            x.join()
    v0.install()
    assert not bad, bad[:5]
    assert len(seen) > 20 and {None, v1.adapter.hash} <= set(seen), set(seen)


def test_a_sibling_question_decides_unharmed_while_the_part_swaps(trained, swaps=400):
    """Part B (another question on the same model, never an adapter) decides in two threads while part A swaps
    versions 0 and 1: B never raises (the slots are copy-on-write, so no reader sees a dict change size), its
    probabilities equal the reference and no adapter answers it."""
    import sys
    m, part, learner, v0, v1 = trained
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = m.decision("mood", "Is the writer angry?", "email", ["yes", "no"])
    texts = [x for x, _ in _toy(10, 41)]
    v0.install()
    ref = {x: b.decide(x).probs for x in texts}
    stop, errors, bad, n = threading.Event(), [], [], [0]

    def score():
        i = 0
        try:
            while not stop.is_set():
                x = texts[i % len(texts)]
                d = b.decide(x)
                i += 1
                if d.probs != ref[x] or d.extra.get("lora"):
                    bad.append((x, d.extra.get("lora")))
        except Exception as e:  # noqa: BLE001 — the failure under test is any exception in a decision
            errors.append(repr(e))
        n[0] += i

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    th = [threading.Thread(target=score) for _ in range(2)]
    for x in th:
        x.start()
    try:
        for _ in range(swaps):
            v1.install()
            v0.install()
    finally:
        stop.set()
        for x in th:
            x.join()
        sys.setswitchinterval(old)
        v0.install()
    assert not errors and not bad and n[0] > 0, (errors[:3], bad[:3])


def test_a_version_saves_loads_and_is_found_by_its_hash(trained, tmp_path):
    m, part, learner, v0, v1 = trained
    files = v1.save(tmp_path, 3)
    assert [f.name for f in files] == ["3.lora.safetensors", "3.calib.json"]
    back = LoraVersion.load(tmp_path, 3, expect_fp=v1.fingerprint(), part=part)
    assert back.fingerprint() == v1.fingerprint() and back.adapter.hash == v1.adapter.hash
    x = _toy(1, 17)[0][0]
    assert back.score(x) == v1.score(x)
    with pytest.raises(ValueError, match="fingerprint"):
        LoraVersion.load(tmp_path, 3, expect_fp="0" * 16)
    assert lora.adapter_by_hash(tmp_path, v1.adapter.hash).hash == v1.adapter.hash
    assert lora.adapter_by_hash(learner.dir, v1.adapter.hash).hash == v1.adapter.hash    # the runner's own file
    with pytest.raises(FileNotFoundError):
        lora.adapter_by_hash(tmp_path, "f" * 16)
    unbound = LoraVersion.load(tmp_path, 3, expect_fp=v1.fingerprint())
    with pytest.raises(ValueError, match="not bound"):
        unbound.score(x)


def test_version0_with_an_adaptation_and_thresholds_per_group_round_trips_its_files(ckpt, tmp_path):
    """Version 0 of a part that was fitted (an adaptation) and calibrated per group saves and loads with the same
    fingerprint and answers; grouped by a function, load needs the function again."""
    from solvi.core.deciders import Facts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
        part.fit(_toy(40, 3))
        ex = [(Facts(email=t, domain="a" if i % 3 else "b"), y) for i, (t, y) in enumerate(_toy(120, 7))]
        part.act_guard(ex, max_risk=0.3, groups="domain", min_group=30)
        v0 = LoraVersion.of_part(part)
        assert v0.calibration["adaptation"] is not None and v0.calibration["groups"] is not None
        v0.save(tmp_path, 0)
        back = LoraVersion.load(tmp_path, 0, expect_fp=v0.fingerprint(), part=part)
        assert back.fingerprint() == v0.fingerprint()
        assert back.calibration["groups"]["nodes"] == v0.calibration["groups"]["nodes"]
        fp = part.fingerprint()
        x = Facts(email=_toy(1, 17)[0][0], domain="a")
        d0 = part.decide(x)
        back.install()                                    # the loaded calibration serves exactly as the live one
        assert part.fingerprint() == fp and part.decide(x).probs == d0.probs
        assert list(part.__signature__.parameters) == ["email", "domain"]

        def dom(email):
            return "long" if len(email) > 45 else "short"
        part.act_guard([(t, y) for t, y in _toy(120, 7)], max_risk=0.3, groups=dom, min_group=30)
        vf = LoraVersion.of_part(part)
        vf.save(tmp_path, 1)
        with pytest.raises(ValueError, match="pass it again"):
            LoraVersion.load(tmp_path, 1, expect_fp=vf.fingerprint())
        assert LoraVersion.load(tmp_path, 1, expect_fp=vf.fingerprint(), groups=dom).fingerprint() == vf.fingerprint()


def test_a_released_version_leaves_the_scorer_and_staged_adapters_stay_bounded(ckpt, tmp_path):
    """Each trained version is staged on the scorer; the loop releases a discarded candidate (here by hand: release),
    so after N trainings and N discards the scorer holds no extra adapter."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
        learner = LoraLearner(part, tmp_path, runner=ThreadRunner(heartbeat_s=0.5))
        st = lora._state(m.scorer)
        base = len(st["heads"])
        for i in range(1, 4):
            v = learner.train(_job(i=i, updates=10, n=40 + i))
            assert v.adapter.name in m._lora_extra and len(st["heads"]) == base + 1
            v.release()
            v.release()                                   # idempotent
            assert v.adapter.name not in m._lora_extra and len(st["heads"]) == base
        assert not m._lora_extra and not m._has_loras()
        v.install()                                       # a released version can still be served: swap installs it
        assert part.lora.hash == v.adapter.hash
        v.release()                                       # a slot holds it: unstaged, not uninstalled
        assert part.lora.hash == v.adapter.hash and v.adapter.name in st["heads"]


def test_the_loop_releases_a_discarded_candidate(trained):
    """AdapterLoop._discard calls Served.release: the discarded version's adapter leaves the scorer."""
    m, part, learner, v0, v1 = trained

    class FakeLoop:
        _release = staticmethod(AdapterLoop._release)
        _event = staticmethod(lambda *a, **k: {"event": a[0]})
        _split_n = {"train": 0}

    v2 = learner.train(_job(i=9, updates=10, n=50))
    fake = FakeLoop()
    fake._vs = {2: {"state": "candidate", "model": v2, "served_as": v2, "samples": []}}
    fake.candidate = 2
    assert v2.adapter.name in m._lora_extra
    AdapterLoop._discard(fake, 2, 5, "test")
    assert v2.adapter.name not in m._lora_extra and fake._vs[2]["model"] is None


def test_the_time_estimate_refuses_a_training_above_max_train_s(trained):
    m, part, learner, v0, v1 = trained
    job = _job(i=7, budget=_budget(max_train_s=0.0))
    with pytest.raises(TrainingRefused, match="max_train_s 0"):
        learner.train(job)
    assert not (learner.dir / "train_7.1.lora.safetensors").exists()
    with pytest.raises(ValueError, match="warm_start"):
        learner.train(TrainJob(id="8.1", version=8, rows=_job().rows, calib=_job().calib, hyper={"warm_start": True},
                               budget=_budget(), alpha=0.2, status={}))


def test_gpu_auto_falls_back_to_the_cpu(ckpt, tmp_path, monkeypatch):
    """Too little free GPU memory → the CPU (in a subprocess); a GPU out of memory → trained again on the CPU (a thread,
    with the GPU only pretended: nothing is put on a GPU)."""
    import torch
    from solvi.experimental.stream import runner as R
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
        sub = LoraLearner(part, tmp_path / "sub", runner=SubprocessRunner(heartbeat_s=1.0))
        v = sub.train(_job(budget=_budget(device="auto", free_gpu_mb=10 ** 9), updates=40))
        assert v.info["device"] == "cpu" and "auto:" in (tmp_path / "sub" / "train_1.1.log").read_text()
        monkeypatch.setattr(R, "pick_device", lambda device, free: ("cuda:0", "pretended"))
        real_load, real_train = DecideModel.load, lora.train
        monkeypatch.setattr(DecideModel, "load", classmethod(lambda cls, p, device=None, **kw:
                                                             real_load.__func__(cls, p, device="cpu", **kw)))

        def oom_on_cuda(part, rows, device=None, **kw):
            if str(device).startswith("cuda"):
                raise torch.cuda.OutOfMemoryError("pretended")
            return real_train(part, rows, device=device, **kw)
        monkeypatch.setattr(lora, "train", oom_on_cuda)
        th = LoraLearner(part, tmp_path / "th", runner=ThreadRunner(heartbeat_s=1.0))
        v2 = th.train(_job(updates=40))
    assert v2.info["device"] == "cpu" and "out of GPU memory" in (tmp_path / "th" / "train_1.1.log").read_text()
    assert v2.adapter.hash == v.adapter.hash                  # the same training on the CPU (fixed seed, two threads)


def test_the_loop_installs_a_version_trained_in_a_subprocess_and_rolls_back(ckpt, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
        fp0 = part.fingerprint()
        learner = LoraLearner(part, tmp_path / "loop", runner=SubprocessRunner(heartbeat_s=1.0))
        policy = DefaultPolicy(_budget(), calib=0.3, holdout=0.0, pre_freeze=False, probe=False, novel_risk=False,
                               first=10 ** 6, lr=1e-2, min_updates=60, max_updates=60)
        loop = AdapterLoop(learner, question="team", max_risk=0.2, secret=KEY, policy=policy,
                           initial=learner.version0(), system=part, versions_dir=tmp_path / "versions")
        data = _toy(1200, 21)
        for x, y in data[:200]:
            loop.step(x, lambda y=y: y)
        job = loop.train_now()
        assert job.state == "done" and job.frozen, job
        k = loop.candidate
        assert (tmp_path / "loop" / f"train_{job.id}.lora.safetensors").exists()
        assert (tmp_path / "versions" / f"{k}.lora.safetensors").exists()
        for x, y in data[200:]:
            loop.step(x, lambda y=y: y)
            if loop.served == k:
                break
        if loop.served != k:                                  # certified but not better: promote it by hand
            assert loop.versions()[k]["certified"], loop.versions()
            loop.promote(k, force=True)
        assert loop.served == k and part.lora is not None and part.fingerprint() != fp0
        assert part.lora.hash == loop._vs[k]["model"].adapter.hash
        assert part.guarantee == loop._vs[k]["model"].calibration["guarantee"]
        loop.rollback(0)
        assert part.fingerprint() == fp0 and part.lora is None
        r = loop.replay(rescore=True)
        assert r["ok"], r["mismatches"][:5]
