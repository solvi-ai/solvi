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


def test_a_version_scores_with_its_own_adaptation_whatever_version_is_served(ckpt, tmp_path):
    """Version 0 of a fitted part (an adaptation) and a trained version (none of its own): each scores with its own
    calibration, so version 0 rescored after a promotion, and the candidate scored in shadow, give the served scores
    to the bit; the rollback serves version 0's scores again."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
        part.fit(_toy(40, 3))
        part.act_guard(_toy(60, 7), max_risk=0.3)
        learner = LoraLearner(part, tmp_path / "lora_loop", runner=ThreadRunner(heartbeat_s=0.5))
        v0 = learner.version0()
        v1 = learner.train(_job())
    assert v0.calibration["adaptation"] is not None
    assert v1.calibration["adaptation"] != v0.calibration["adaptation"]
    t = [x for x, _ in _toy(8, 11)]

    def live():
        return [(d.value, part._signal(d)[1]) for d in (part.decide(x) for x in t)]
    s0, s1 = [v0.score(x) for x in t], [v1.score(x) for x in t]     # v1 in shadow while version 0 is served
    assert s0 == live()
    v1.install()                                                     # the promotion
    try:
        assert live() == s1                                          # the shadow score was the served one
        assert [v0.score(x) for x in t] == s0                        # version 0 rescored while v1 is served
    finally:
        v0.install()                                                 # the rollback
    assert live() == s0 and [v1.score(x) for x in t] == s1


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


def test_a_pass_in_flight_on_a_model_without_adapters_is_untouched_by_a_staging_thread(ckpt, trained, stages=40):
    """A model with no adapter scores without the adapter lock. Another thread staging an adapter (what a training's
    thread does when its version is bound) waits for the passes in flight at the scorer's network gate, and passes
    that start meanwhile wait for the change: every pass gives the logits of the plain checkpoint, bit for bit (before
    the gate, a pass that overlapped the install ran partly through the trained adapter)."""
    import sys

    import numpy as np
    ad = trained[4].adapter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
    texts = [x for x, _ in _toy(40, 43)]

    def raw(x):
        with m._lock:
            m._cache.clear()                                # every call a real forward pass
        z, act = part._raw([x])[0]
        return np.asarray(z, float), act

    ref = {}
    t = threading.Thread(target=lambda: ref.update({x: raw(x) for x in texts}))   # the scoring threads' own setting
    t.start()
    t.join()
    stop, errors, bad, n = threading.Event(), [], [], [0]

    def score():
        i = 0
        try:
            while not stop.is_set():
                x = texts[i % len(texts)]
                z, act = raw(x)
                if not np.array_equal(z, ref[x][0]) or act != ref[x][1]:
                    bad.append(x)
                i += 1
        except Exception as e:  # noqa: BLE001 — the failure under test is any exception in a pass
            errors.append(repr(e))
        n[0] += i

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    th = threading.Thread(target=score)
    th.start()
    try:
        for _ in range(stages):
            lora.stage(part, ad)                            # installed on the scorer while a pass may be in flight
            lora.unstage(part, ad)                          # and gone again: the next stage installs it anew
    finally:
        stop.set()
        th.join()
        sys.setswitchinterval(old)
    assert not errors and not bad and n[0] > 0, (errors[:3], len(bad), n[0])


def test_a_pass_held_mid_way_is_not_changed_by_a_staging_thread(ckpt, trained):
    """The deterministic form of the test above: a forward pre-hook on the model's second layer holds the scoring
    thread's pass mid-way; the main thread then stages the adapter and switches it on. With the network gate, stage()
    waits for the pass (the hook gives up after 1 s and the pass ends plain); without it the second half of the pass
    would run through the active adapter and the logits would differ from the plain checkpoint's."""
    import re

    import numpy as np
    ad = trained[4].adapter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
    x = _toy(1, 43)[0][0]

    def raw():
        with m._lock:
            m._cache.clear()
        z, act = part._raw([x])[0]
        return np.asarray(z, float), act
    ref = {}
    t = threading.Thread(target=lambda: ref.update(r=raw()))       # the scoring thread's own setting
    t.start()
    t.join()
    net = m.scorer.model
    layer = next(mod for name, mod in net.named_modules() if re.fullmatch(r".*layers\.1", name))
    mid, go, tid = threading.Event(), threading.Event(), [None]

    def hook(mod, args):
        if threading.get_ident() == tid[0] and not mid.is_set():
            mid.set()
            go.wait(1.0)                     # with the gate nothing can install meanwhile: this times out
    h = layer.register_forward_pre_hook(hook)
    out = {}

    def score():
        tid[0] = threading.get_ident()
        out["r"] = raw()
    th = threading.Thread(target=score)
    th.start()
    try:
        assert mid.wait(10)
        lora.stage(part, ad)                 # waits at the gate for the pass in flight; without the gate, goes on
        with lora.using(m.scorer, ad.name):  # the adapter on: the held pass would finish through it
            go.set()
            th.join(5)
    finally:
        go.set()
        th.join(5)
        h.remove()
        lora.unstage(part, ad)
    assert not th.is_alive()
    z, act = out["r"]
    assert np.array_equal(z, ref["r"][0]) and act == ref["r"][1], (np.abs(z - ref["r"][0]).max(), act, ref["r"][1])


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
                           initial=learner.version0(), system=part, versions_dir=tmp_path / "versions", additive=False)
        data = _toy(1200, 21)
        for x, y in data[:200]:
            loop.step(x, lambda y=y: y)
        job = loop.train_now(wait=True)
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


def test_deferred_apply_end_to_end_with_the_subprocess_runner(ckpt, tmp_path):
    """Stage 3b: the training runs in a subprocess while the loop goes on deciding on version 0 (no decision waits for
    it); its result is applied on a later decision, recorded there, served, and the replay (rescored) is exact."""
    import time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
        fp0 = part.fingerprint()
        learner = LoraLearner(part, tmp_path / "loop", runner=SubprocessRunner(heartbeat_s=1.0))
        policy = DefaultPolicy(_budget(), calib=0.3, holdout=0.0, pre_freeze=False, probe=False, novel_risk=False,
                               first=10 ** 6, lr=1e-2, min_updates=60, max_updates=60)
        loop = AdapterLoop(learner, question="team", max_risk=0.2, secret=KEY, policy=policy,
                           initial=learner.version0(), system=part, versions_dir=tmp_path / "versions", additive=False)
        assert loop._part_lock is lora.version_lock(part)
        data = iter(_toy(4000, 31))
        for _ in range(200):
            x, y = next(data)
            loop.step(x, lambda y=y: y)
        job = loop.train_now()
        assert job.state == "running" and loop.candidate is None
        worst = 0.0
        for _ in range(3):                                  # decisions go on while it trains (none waits for it)
            x, y = next(data)
            t0 = time.monotonic()
            out = loop.step(x, lambda y=y: y)
            worst = max(worst, time.monotonic() - t0)
            assert out["served"] == 0 and out["candidate"] is None and "train" not in out["events"]
        info = loop.status()["job"]
        assert info["state"] == "running" and worst < 5.0, (info, worst)   # still training after those decisions
        assert info["pid"] and info["runner"] == "subprocess" and info["started_at"] == 200
        # the apply position pinned: wait for the result, then it is applied on the next decision (204), so the
        # stream after it — and with it the candidate's certificate — does not depend on how long the training took
        assert loop.wait_training(timeout=300)
        x, y = next(data)
        out = loop.step(x, lambda y=y: y)
        assert "train" in out["events"] and out["n"] == 204, out
        tr = [e for k, e in loop._items() if k == "stream" and e["event"] == "train"]
        assert len(tr) == 1 and tr[0]["state"] == "done" and tr[0]["decision"] == out["n"] == 204
        assert tr[0]["applied"] == "decision" and tr[0]["started_at"] == 200 and loop.candidate == 1
        for x, y in data:
            loop.step(x, lambda y=y: y)
            if loop.served == 1:
                break
        if loop.served != 1:
            assert loop.versions()[1]["certified"], loop.versions()
            loop.promote(1, force=True)
        assert loop.served == 1 and part.lora is not None and part.fingerprint() != fp0
        loop.rollback(0)
        assert part.fingerprint() == fp0
        r = loop.replay(rescore=True)
        assert r["ok"], r["mismatches"][:5]
        loop.close()
        assert loop.status()["state"] == "closed"


CHILD = r'''
import json, sys, warnings
warnings.simplefilter("ignore")
import torch
a = json.loads(open(sys.argv[1]).read())
torch.set_num_threads(a["threads"])
from solvi.core.deciders import DecideModel
from solvi.experimental.stream import AdapterLoop
from solvi.experimental.stream.lora_learner import LoraLearner
from solvi.experimental.stream.runner import SubprocessRunner
m = DecideModel.load(a["ckpt"], backend="torch", device="cpu")
part = m.decision("team", "Which team?", "email", a["teams"])
fp0 = part.fingerprint()
learner = LoraLearner(part, a["loopdir"], runner=SubprocessRunner(heartbeat_s=1.0))
loop = AdapterLoop.load(a["ck"], learner, system=part, secret=a["key"].encode(), versions_dir=a["vdir"],
                        clock=lambda: 1000.0)
loaded_fp = part.fingerprint()
outs = [loop.step(x, lambda y=y: y) for x, y in a["data"]]
r = loop.replay(rescore=True)
json.dump({"outs": outs, "log": loop.log, "ok": r["ok"], "mm": [list(map(str, x)) for x in r["mismatches"][:5]],
           "decisions": r["decisions"], "fp0": fp0, "loaded_fp": loaded_fp, "served": loop.served},
          open(a["out"], "w"))
'''


def test_checkpoint_end_to_end_train_promote_save_load_in_a_new_process_continue_replay(ckpt, tmp_path):
    """Stage 4 on the tiny checkpoint: a version trained in a subprocess is promoted, the loop is saved, a new process
    loads the model and the checkpoint (the served adapter is installed), continues on the same inputs as the loop it
    was saved from, with the same decisions and records, and its rescored replay is exact."""
    import json
    import os
    import subprocess
    import sys

    import torch

    import solvi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, part = _part(ckpt)
        fp0 = part.fingerprint()
        learner = LoraLearner(part, tmp_path / "loop", runner=SubprocessRunner(heartbeat_s=1.0))
        policy = DefaultPolicy(_budget(), calib=0.3, holdout=0.0, pre_freeze=False, probe=False, novel_risk=False,
                               first=10 ** 6, lr=1e-2, min_updates=60, max_updates=60)
        loop = AdapterLoop(learner, question="team", max_risk=0.2, secret=KEY, policy=policy,
                           initial=learner.version0(), system=part, versions_dir=tmp_path / "versions",
                           clock=lambda: 1000.0, additive=False)
        data = _toy(1700, 41)
        for x, y in data[:200]:
            loop.step(x, lambda y=y: y)
        job = loop.train_now(wait=True)
        assert job.state == "done" and job.frozen, job
        k = loop.candidate
        for x, y in data[200:1300]:
            loop.step(x, lambda y=y: y)
            if loop.served == k:
                break
        if loop.served != k:
            assert loop.versions()[k]["certified"], loop.versions()
            loop.promote(k, force=True)
        assert loop.served == k and part.fingerprint() != fp0
        info = loop.save(tmp_path / "ck")
        saved_fp, n = part.fingerprint(), loop.n
        rest = data[n:n + 300]
        outs = [loop.step(x, lambda y=y: y) for x, y in rest]
        r = loop.replay(rescore=True)
        assert r["ok"], r["mismatches"][:5]
        assert any(f.startswith(f"versions/{k}.lora") for f in info["files"])
    args = {"threads": torch.get_num_threads(), "ckpt": str(ckpt), "teams": TEAMS, "loopdir": str(tmp_path / "loop2"),
            "ck": str(tmp_path / "ck"), "key": KEY.decode(), "vdir": str(tmp_path / "versions2"),
            "data": [list(d) for d in rest], "out": str(tmp_path / "child.json")}
    (tmp_path / "args.json").write_text(json.dumps(args))
    (tmp_path / "child.py").write_text(CHILD)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(solvi.__file__)))
    p = subprocess.run([sys.executable, str(tmp_path / "child.py"), str(tmp_path / "args.json")], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    got = json.loads((tmp_path / "child.json").read_text())
    assert got["fp0"] == fp0 and got["loaded_fp"] == saved_fp and got["served"] == k
    assert got["outs"] == json.loads(json.dumps(outs))
    assert got["log"] == json.loads(json.dumps(loop.log))
    assert got["ok"] and got["decisions"] == n + 300, got["mm"]


def test_attach_end_to_end_dispatcher_background_training_promote_rollback_save_load_replay(ckpt, tmp_path):
    """Stage 5 on the tiny checkpoint: a loop attached to a Dispatcher over a System whose question is the LoRA part.
    The slow path's labels flow into the loop; a training runs in the background (thread runner) while decisions go
    on; the candidate is promoted; System 1's trace records the served adapter (the part's fingerprint, extra["lora"]);
    a rollback through the handle puts version 0 back; the loop is saved, loaded on a new System and Dispatcher over a
    fresh copy of the model, continues with the same decisions; every decision replays exactly, each under its own
    version, and the loop's rescored replay is exact."""
    from solvi import Answer, Catalog, Decision, Question, System
    from solvi.core.dispatch import AskPath, Dispatcher
    from solvi.core.store import JSONLStorage
    data = _toy(1800, 43)
    gold = {x: y for x, y in data}

    def s2():
        cat = Catalog()

        @cat.rule("team")
        def team(email):
            return Decision(gold[email], {gold[email]: 1.0})
        return System(cat, [Question("team", "Which team?", Answer.choice(TEAMS))])

    def host(storage=None):
        m, part = _part(ckpt)
        cat = Catalog()
        sysm = System(cat, [part.question(cat)])
        return part, sysm, Dispatcher(sysm, AskPath(s2()), secret=KEY, storage=storage)

    def policy():
        return DefaultPolicy(_budget(), calib=0.3, holdout=0.0, pre_freeze=False, probe=False, novel_risk=False,
                             first=10 ** 6, lr=1e-2, min_updates=60, max_updates=60, s2_labels=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        part, sysm, disp = host(JSONLStorage(tmp_path / "d.jsonl"))
        fp0 = part.fingerprint()
        loop = AdapterLoop.attach(disp, LoraLearner(part, tmp_path / "loop", runner=ThreadRunner(heartbeat_s=0.5)),
                                  question="team", dir=tmp_path / "loop", policy=policy(), max_risk=0.2, secret=KEY,
                                  clock=lambda: 1000.0, additive=False)
        assert loop.part is part and loop.mode == "stream" and disp.loop is loop
        xs = [{"email": x} for x, _ in data]
        out = [disp.ask(x) for x in xs[:200]]
        assert loop.status()["labels"]["train"] > 0
        job = disp.loop.train_now()                  # in the background: the call returns at once
        assert job.state == "running" and loop.status()["job"] is not None
        during = disp.ask(xs[200])                   # a decision while the training runs
        assert during.learning["served"] == 0
        assert loop.wait_training(240)
        i = 201
        while loop.candidate is None and i < 260:
            out.append(disp.ask(xs[i]))
            i += 1
        k = loop.candidate
        assert k is not None, loop.versions()
        while loop.served != k and i < 1500:
            out.append(disp.ask(xs[i]))
            i += 1
            if loop.served != k and loop.versions()[k]["certified"] and i % 50 == 0:
                loop.promote(k, force=True)
        assert loop.served == k and part.fingerprint() != fp0, loop.versions()
        d = disp.ask(xs[i])
        i += 1
        adapter = loop._vs[k]["served_as"].adapter.fingerprint()
        rec = [r for r in d.s1.trace.records if r.model is not None]
        assert rec and rec[-1].model["fp"] == part.fingerprint()            # the trace records the served version
        assert ((rec[-1].extra or {}).get("lora") or {}).get("adapter") == adapter and d.learning["served"] == k
        for x in xs[i:i + 100]:
            disp.ask(x)
        i += 100
        disp.loop.rollback(0, by="test")
        assert part.fingerprint() == fp0
        d0 = disp.ask(xs[i])
        i += 1
        assert d0.learning["served"] == 0 and [r for r in d0.s1.trace.records if r.model][-1].model["fp"] == fp0
        disp.loop.rollback(k, by="test")             # k is retired now: served again
        assert loop.served == k and part.fingerprint() != fp0
        info = loop.save(tmp_path / "ck")
        assert any(f.startswith(f"versions/{k}.lora") for f in info["files"])
        # a new System and Dispatcher over a fresh copy of the model; the loaded loop installs version k on it
        part2, sysm2, disp2 = host()
        loop2 = AdapterLoop.load(tmp_path / "ck", LoraLearner(part2, tmp_path / "loop2", runner=ThreadRunner()),
                                 system=sysm2, part=part2, secret=KEY, clock=lambda: 1000.0)
        assert part2.fingerprint() == part.fingerprint()
        loop2.attach_to(disp2)
        for x in xs[i:i + 150]:
            a, b = disp.ask(x), disp2.ask(x)
            assert (a.answer, a.by, a.action) == (b.answer, b.by, b.action) and a.learning == b.learning
        assert disp.replay_all() == []             # every decision under its own version: 0, k, 0 again, k
        r = loop.replay(rescore=True)
        assert r["ok"], r["mismatches"][:5]
        assert loop2.replay()["ok"]
        loop.detach()
        assert part.fingerprint() == fp0 and disp.loop is None
        loop2.detach()


def test_the_training_reads_the_question_at_the_parts_own_max_len(ckpt, tmp_path):
    # a part loaded with max_len=N trains at N: the payload carried the multi-question layout's length before, so a
    # training rebuilt the model at the default 512 and refused a question that the part itself reads (77 intents
    # with descriptions at 768 tokens)
    for n in (256, 768):
        m = DecideModel.load(ckpt, backend="torch", device="cpu", max_len=n)
        part = m.decision("team", "Which team?", "email", TEAMS)
        learner = LoraLearner(part, tmp_path / str(n), runner=ThreadRunner(heartbeat_s=0.5))
        assert learner._payload(_job())["model"]["max_len"] == n
