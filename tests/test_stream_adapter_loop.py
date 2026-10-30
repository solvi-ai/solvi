"""solvi.experimental.stream, the adapter loop (stage 2): AdapterLoop driven by a LoopPolicy — the lane's LearningLoop
as DefaultPolicy at its constants, the novel-only comparison and the memorisation probe, controls, two alarm sources."""
import gzip
import hashlib
import json
import random
import threading
import warnings
from pathlib import Path

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import (AdapterLoop, Budget, DefaultPolicy, HashedTextLearner, HashedTextTrainer,
                                           LearningLoop, RiskDetector, lane_policy)

KEY = b"adapter-loop-test-key"


def _events(loop, *kinds):
    return [e for k, e in loop._items() if k == "stream" and (not kinds or e["event"] in kinds)]


class _Model:
    """x = (label, u): right with score 0.95 when u < share, else a wrong answer with score 0.05."""

    def __init__(self, name, share):
        self.name, self.share = name, share

    def score(self, x):
        return (x[0], 0.95) if x[1] < self.share else ("wrong", 0.05)

    def fingerprint(self):
        return f"model-{self.name}-{self.share}"


class _Hand:
    """A batch learner that hands out the given models in turn, one per training."""

    def __init__(self, models):
        self.models, self._seen, self.jobs = list(models), 0, []

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        self.jobs.append(job)
        return self.models.pop(0)


def _fake(seed, n):
    r = random.Random(seed)
    return [(r.choice("ab"), r.random()) for _ in range(n)]


def _run(loop, xs):
    return [loop.step(x, lambda x=x: x[0]) for x in xs]


def _plain(**kw):
    """A DefaultPolicy with only the stream rule: every label trains, every training freezes, no wall-clock budget."""
    base = dict(calib=0.0, holdout=0.0, pre_freeze=False, probe=False)
    base.update(kw)
    return DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6), **base)


class _Flag:
    """A score monitor that flags when told to (a solvi Monitor: observe, reset)."""

    def __init__(self):
        self.fire = False

    def observe(self, score):
        return {"drift": self.fire, "why": ["test flag"]}

    def reset(self):
        self.fire = False


# ───────────────────────────── the lane's LearningLoop is AdapterLoop + DefaultPolicy at its constants
def _lazy(g, o, masked):
    """g with o's v_index / delta_v / state_fp wherever g has no index yet and o has one (the lazy δ charge, B4: the
    only fields of the old log it changes); the indices masked are collected in `masked`."""
    if isinstance(g, dict) and isinstance(o, dict):
        out = {k: _lazy(v, o.get(k), masked) for k, v in g.items()}
        if "v_index" in g and g["v_index"] is None and o.get("v_index") is not None:
            masked.add(o["v_index"])
            for k in ("v_index", "delta_v", "state_fp"):
                if k in o:
                    out[k] = o[k]
        return out
    if isinstance(g, list) and isinstance(o, list) and len(g) == len(o):
        return [_lazy(x, y, masked) for x, y in zip(g, o)]
    return g


def test_learning_loop_is_the_adapter_loop_with_the_default_policy_at_the_lanes_constants():
    loop = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, growth=1.5, trial=700, additive=False)
    assert isinstance(loop, AdapterLoop) and type(loop.policy) is DefaultPolicy
    p = loop.policy
    assert (p.first, p.growth, p.refreeze, p.stale, p.trial, p.min_share, p.margin) == (300, 1.5, 1.5, 3000, 700,
                                                                                      0.05, 0.01)
    assert (p.calib, p.holdout, p.probe, p.pre_freeze, p.novel_risk, p.probation) == (0.0, 0.0, False, False, False, 0)
    # the same stream gives the records the old LearningLoop class (before AdapterLoop) wrote: the fixture is its log,
    # recorded at the stage-1 commit on a stream without repeats (every input unique, so the novel-only comparison is
    # the old one); the new fields (content key, split, wall clock, the "train" event, counts added to events) are
    # dropped, every other field must be equal (the symmetric comparison's fields of 1.4.0 too: version 0 here has no
    # model, so l_s is 0 under either comparison and the decisions are the old ones)
    r = random.Random(1)
    voc = {c: [f"{c}w{i}" for i in range(12)] for c in "abcde"}
    data = []
    for i in range(1500):
        c = r.choice("abcde")
        data.append((" ".join(r.sample(voc[c], 3) + [f"u{i}"]), c))
    a = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, additive=False)
    for x, y in data:
        a.step(x, lambda y=y: y)
    old = json.loads(gzip.decompress((Path(__file__).parent / "fixtures" / "learning_loop_04c7e18.json.gz")
                                     .read_bytes()))
    new_fields = {"loop": {"key", "split", "t"}, "stream": {"train_keys", "n_train", "n_novel", "n_seen", "install",
                                                            "l_s_own", "thr_s_cmp", "thr_s_own", "thr_s"}}
    got = []
    for k, d in a.log:
        if k == "stream":
            if d["event"] in ("train", "train_start"):
                continue
            got.append([k, {f: v for f, v in d.items() if f not in new_fields["stream"]}])
        else:
            got.append([k, {**d, "stream": {f: v for f, v in d["stream"].items() if f not in new_fields["loop"]}}])
    assert len(got) == len(old) == 1506

    def _canon(x):
        """Floats rounded to 12 decimals: the replay is pinned per event, but a 1-ulp drift of a shadow score
        across numpy/BLAS builds is not behavior (the CI runner differed from the fixture's machine at index 311
        in exactly one last digit)."""
        if isinstance(x, float):
            return round(x, 12)
        if isinstance(x, list):
            return [_canon(v) for v in x]
        if isinstance(x, dict):
            return {k: _canon(v) for k, v in x.items()}
        return x

    unscored, masked = 0, set()
    for n, (g, o) in enumerate(zip(got, old)):
        if g[0] == "loop" and g[1]["stream"]["shadow"] is None and o[1]["stream"]["shadow"] is not None:
            # the candidate is scored in shadow only while its grid is not fixed and on coin-sampled decisions now;
            # the old class scored it on every decision: only that value (and its would-be answer) may differ
            unscored += 1
            assert not g[1]["stream"]["sampled"], n
            o = [o[0], {**o[1], "stream": {**o[1]["stream"], "shadow": None},
                        "loop": {**o[1]["loop"], "shadow_would": None}}]
        g = _lazy(g, o, masked)
        assert _canon(g) == _canon(o), n
    assert unscored > 0
    # the lazy δ charge (stage B4): a promise has no index between its freeze and its first sample; the index it then
    # takes is the one the old class gave it at the freeze (the same issue order)
    taken = {d.get("v_index") for _, d in a.log} | {s.get("v_index") for _, d in a.log if "stream" in d
                                                     for s in (d["stream"].get("served_state"),
                                                               d["stream"].get("candidate_state")) if s}
    assert masked and masked <= taken
    assert _events(a, "promote") and a.replay()["ok"]


def test_trace_candidate_records_the_candidates_score_on_every_trial_decision_and_changes_nothing_else(tmp_path):
    r = random.Random(1)
    voc = {c: [f"{c}w{i}" for i in range(12)] for c in "abcde"}
    data = []
    for i in range(1500):
        c = r.choice("abcde")
        data.append((" ".join(r.sample(voc[c], 3) + [f"u{i}"]), c))
    runs = {}
    for on in (False, True):
        loop = AdapterLoop(HashedTextLearner(), question="q", policy=lane_policy(), max_risk=0.05, secret=KEY,
                           monitor=None, _detector_alone=True, _batch=False, trace_candidate=on, clock=lambda: 0.0, additive=False)
        for x, y in data:
            loop.step(x, lambda y=y: y)
        runs[on] = loop
    off, on = runs[False], runs[True]
    assert len(off.log) == len(on.log)
    trial = 0
    for (ko, do), (kn, dn) in zip(off.log, on.log):
        assert ko == kn
        if ko == "stream":
            def timed(e):                                  # a training's measured seconds are wall time
                return {f: (timed(v) if isinstance(v, dict) else v) for f, v in e.items() if f != "seconds"}
            assert timed(do) == timed(dn)                  # the same events: the protocol never reads the trace
            continue
        so, sn, lo, ln = do["stream"], dn["stream"], do["loop"], dn["loop"]
        assert "candidate_score" not in so and "candidate_would" not in lo
        if sn["candidate"] is None:                        # no trial: the keys are absent
            assert "candidate_score" not in sn and "candidate_would" not in ln
        else:
            trial += 1
            assert sn["candidate_score"] is not None and ln["candidate_would"] is not None
            if sn["shadow"] is not None:                   # where the protocol read the shadow, the same values
                assert (sn["candidate_score"], ln["candidate_would"]) == (sn["shadow"], ln["shadow_would"])
        assert {k: v for k, v in sn.items() if k != "candidate_score"} == so
        assert {k: v for k, v in ln.items() if k != "candidate_would"} == lo
    unread = sum(1 for k, d in on.log if k == "loop" and d["stream"]["candidate"] is not None
                 and d["stream"]["shadow"] is None)
    assert trial > 0 and unread > 0 and _events(on, "promote")
    assert on.replay()["ok"] and off.replay()["ok"]
    # a record whose trace was dropped no longer replays
    i = next(i for i, (k, d) in enumerate(on.log) if k == "loop" and "candidate_score" in d["stream"])
    k, d = on.log[i]
    on.log[i] = (k, {**d, "stream": {f: v for f, v in d["stream"].items() if f != "candidate_score"}})
    assert not on.replay()["ok"]
    on.log[i] = (k, d)
    on.save(tmp_path / "ck")                               # the option is carried by the checkpoint
    assert AdapterLoop.load(tmp_path / "ck", HashedTextLearner(), secret=KEY).trace_candidate is True


# ───────────────────────────── the memorisation probe on a live loop
class _Memo:
    """Frozen: right with score 0.99 on the keys it trained on; elsewhere on 8 % of keys a random answer with score
    0.99, else a random answer with score 0.1."""

    def __init__(self, table):
        self.table = dict(table)

    def score(self, x):
        k = f"k{x[0]}"
        if k in self.table:
            return self.table[k], 0.99
        h = int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) / 2 ** 32
        guess = "abcde"[int(h * 1e6) % 5]
        return guess, (0.99 if h < 0.08 else 0.1)

    def fingerprint(self):
        return "memo-" + hashlib.sha256(repr(sorted(self.table.items())).encode()).hexdigest()[:16]


class _MemoLearner:
    def __init__(self):
        self._seen = 0

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        return _Memo({r.key: r.y for r in job.rows})


def _memo_run(seed=21, learner=None, n=3000, repeat=0.7, **policy):
    loop = AdapterLoop(learner or _MemoLearner(), question="q", wait=True, max_risk=0.05, secret=KEY, key=lambda x: f"k{x[0]}",
                       policy=_plain(**policy), promise_options={"grid": [0.95, 0.5]}, additive=False)
    r, fresh = random.Random(seed), seed * 10 ** 6         # independent streams: the novel keys differ per seed
    first = []
    for i in range(n):
        if i < 300 or r.random() >= repeat:
            kid, fresh = fresh, fresh + 1                 # a key nobody labelled yet
            if i < 300:
                first.append(kid)
        else:
            kid = first[r.randrange(300)]                  # a key the first candidate trained on
        x = (kid,)
        loop.step(x, lambda kid=kid: "abcde"[kid % 5])
    return loop


def test_a_memoriser_is_refused_by_the_probe_and_promoted_with_it_off():
    on = _memo_run(probe=True)
    assert not _events(on, "promote") and on.served == 0
    assert any(e["why"].startswith("memorisation probe") for e in _events(on, "discard"))
    off = _memo_run(probe=False, novel_risk=False)
    promo = _events(off, "promote")
    assert promo and off.served >= 1 and promo[0]["n_seen"] > 0     # certified on the pool, the memorised keys in it
    assert on.replay()["ok"]


def test_repeated_held_out_labels_do_not_start_a_training_the_freeze_refuses():
    """Each key labelled twice: the label counts run at twice the distinct keys. The training waits for 100 distinct
    held-out keys (the report's count), so the first one is not refused at the freeze for "< 100 held-out label(s)"."""
    loop = AdapterLoop(_Hand([_Model(f"v{i}", 0.9) for i in range(1, 9)]), question="q", wait=True, max_risk=0.05,
                       secret=KEY, key=lambda x: f"k{x[2]}",
                       policy=DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6), probe=False, first=50), additive=False)
    r = random.Random(5)
    for kid in range(900):
        x = (r.choice("ab"), r.random(), kid)
        for _ in range(2):
            loop.step(x, lambda x=x: x[0])
    st = loop._state()
    assert st.labels["holdout"] > st.keys["holdout"] >= 100        # repeats counted in labels, once in keys
    trains = [e for e in _events(loop, "train") if e.get("state") == "done"]
    assert trains, "no training started"
    assert trains[0]["report"]["n_holdout"] >= 100 and trains[0]["frozen"], trains[0]["why"]
    assert not any("held-out label(s) <" in (e.get("why") or "") for e in _events(loop, "discard"))
    assert loop.replay()["ok"]


class _PureMemoLearner:
    """Freezes a pure memoriser: right with score 0.99 on its trained keys, score 0.1 (never alone) on every other key —
    on novel keys alone on none, on the pool alone on every repeat of a trained key."""

    def __init__(self):
        self._seen = 0

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        return _Risky({r.key: r.y for r in job.rows}, 0.0, 0.5)


def _thin_run(seed=21, **policy):
    return _memo_run(seed, _PureMemoLearner(), **policy)


def test_novel_only_off_promotes_a_memoriser_the_defaults_refuse():
    """C-probe control: the memorisation probe, the novel-error bound and the novel-only comparison all off promote a
    memoriser on its trained items; the defaults refuse it, and probe / novel_risk off alone still refuse it (the
    novel-only comparison holds share_k at 0)."""
    on = _thin_run()
    assert _events(on, "freeze") and not _events(on, "promote") and on.served == 0
    half = _thin_run(probe=False, novel_risk=False)
    assert not _events(half, "promote") and half.served == 0
    off = _thin_run(probe=False, novel_risk=False, novel_only=False)
    promo = _events(off, "promote")
    assert promo and off.served >= 1 and promo[0]["n_seen"] > 0
    assert promo[0]["n_novel"] >= promo[0]["n_seen"]              # the comparison counted the seen samples too
    assert off.policy.novel_only is False and DefaultPolicy().novel_only is True
    assert off.replay()["ok"]


def test_novel_only_is_carried_by_the_checkpoint(tmp_path):
    a = _thin_run(n=1200, probe=False, novel_risk=False, novel_only=False)
    a.save(tmp_path / "ck")
    b = AdapterLoop.load(tmp_path / "ck", _PureMemoLearner(), secret=KEY, key=lambda x: f"k{x[0]}")
    assert b.policy.novel_only is False and b.status() == a.status()
    if a.candidate is not None:
        assert b._paired(b.candidate) == a._paired(a.candidate)


class _Risky:
    """Frozen: right with score 0.99 on its trained keys; on a novel key alone (0.99) on a share `alone` of keys, and
    there wrong on a share `wrong` of them (two key hashes), else score 0.1 — a true novel risk of alone × wrong."""

    def __init__(self, table, alone, wrong):
        self.table, self.alone, self.wrong = dict(table), alone, wrong

    def score(self, x):
        k = f"k{x[0]}"
        if k in self.table:
            return self.table[k], 0.99
        h = hashlib.sha256(k.encode()).hexdigest()
        h1, h2 = int(h[:8], 16) / 2 ** 32, int(h[8:16], 16) / 2 ** 32
        return ("wrong" if h2 < self.wrong else "abcde"[x[0] % 5]), (0.99 if h1 < self.alone else 0.1)

    def fingerprint(self):
        return f"risky-{self.alone}-{self.wrong}-" + hashlib.sha256(repr(sorted(self.table.items())).encode()
                                                                    ).hexdigest()[:16]


class _RiskyLearner(_MemoLearner):
    def __init__(self, alone, wrong):
        super().__init__()
        self.alone, self.wrong = alone, wrong

    def train(self, job):
        return _Risky({r.key: r.y for r in job.rows}, self.alone, self.wrong)


def test_the_novel_risk_bound_refuses_a_memoriser_above_alpha_that_the_point_estimate_promoted():
    """Stage-2 review B1, author's decision 6: a memoriser right on its trained keys, alone on 8 % of novel keys and
    wrong on 80 % of those (a true novel risk of 6.4 %), with the probe off. The point estimate looked at on every
    decision promoted it in 8 of 10 independent streams; the Wilson upper bound promotes it in none of 5."""
    for seed in range(21, 26):
        loop = _memo_run(seed, _RiskyLearner(0.08, 0.80), probe=False)
        assert _events(loop, "freeze")
        assert not _events(loop, "promote") and loop.served == 0, seed


def test_the_novel_risk_bound_promotes_a_candidate_at_3_percent_with_enough_novel_samples():
    """A candidate alone on 30 % of novel keys and wrong on 10 % of those (a true novel risk of 3 %) is promoted once
    the bound is under α — after a few hundred novel samples, not at the first 100."""
    for seed in (21, 22, 23):
        loop = _memo_run(seed, _RiskyLearner(0.30, 0.10), repeat=0.3, probe=False)
        promo = _events(loop, "promote")
        assert promo and loop.served >= 1, seed
        assert promo[0]["n_novel"] > 150, promo[0]["n_novel"]
    assert loop.replay()["ok"]


# ───────────────────────────── controls
def test_pause_blocks_training_and_promotion_but_not_labels():
    learner = _Hand([_Model("good", 1.0)] + [_Model(f"w{i}", 0.3) for i in range(5)])
    loop = AdapterLoop(learner, question="q", wait=True, max_risk=0.05, secret=KEY, policy=_plain(first=100, trial=300), additive=False)
    data = _fake(2, 3000)
    assert loop.pause()["event"] == "pause" and loop.pause() is None
    _run(loop, data[:400])
    assert not _events(loop, "train") and loop.candidate is None
    assert loop.status()["labels"]["train"] == 400 and learner.seen() == 400       # labels go on
    job = loop.train_now()
    assert job.state == "refused" and job.why == "paused"
    loop.resume()
    _run(loop, data[400:401])
    assert loop.candidate == 1 and len(learner.jobs) == 1
    loop.pause()
    _run(loop, data[401:1500])
    assert loop.versions()[1]["certified"] and not _events(loop, "promote") and loop.served == 0
    with pytest.raises(ValueError):
        loop.promote()
    loop.resume()
    out = loop.step(data[1500], lambda: data[1500][0])
    assert "promote" in out["events"] and loop.served == 1
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_train_now_ignores_the_schedule_but_not_the_budget():
    clock = [1000.0]
    learner = HashedTextTrainer(list("ab"))
    policy = DefaultPolicy(Budget(min_interval_s=600.0), first=10 ** 6)
    loop = AdapterLoop(learner, question="q", wait=True, max_risk=0.05, secret=KEY, policy=policy, clock=lambda: clock[0], additive=False)
    for x in _fake(3, 50):
        loop.step(f"text {x[0]} {x[1]:.3f}", lambda x=x: x[0])
    assert not policy.want_train(loop._state()) and "train label(s)" in policy.why_not_train(loop._state())
    job = loop.train_now()
    assert job.state == "done" and job.trigger == "train_now" and job.rows
    assert job.frozen is False                          # 50 labels: the held-out gate refuses the freeze ...
    assert loop.versions()[1]["state"] == "discarded" and loop.candidate is None   # ... a version discarded at birth
    again = loop.train_now()
    assert again.state == "refused" and again.why.startswith("budget:") and "min_interval_s" in again.why
    clock[0] += 601
    assert loop.train_now().state == "done"
    assert loop.status()["trainings"] == 2
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_status_answers_from_another_thread_during_a_training_and_shows_the_job():
    """Stage-2 review S1: status() takes no lock, so another thread reads it while step() runs the in-process
    training, and sees the job. The training waits for the watcher: a blocking status() would time out here."""
    started, read = threading.Event(), threading.Event()
    seen = {}

    class _Wait(_Hand):
        def train(self, job):
            started.set()
            seen["released"] = read.wait(timeout=20)
            return super().train(job)
    loop = AdapterLoop(_Wait([_Model("m", 0.5)]), question="q", wait=True, max_risk=0.05, secret=KEY, policy=_plain(first=20), additive=False)

    def watcher():
        started.wait(timeout=20)
        seen["status"] = loop.status()
        read.set()
    th = threading.Thread(target=watcher)
    th.start()
    assert loop.status()["job"] is None and loop.status()["state"] == "running"
    _run(loop, _fake(4, 20))
    th.join(timeout=30)
    assert seen["released"], "status() blocked for the whole training"
    assert seen["status"]["job"]["id"] == "1.1" and seen["status"]["job"]["trigger"] == "schedule"
    assert seen["status"]["state"] == "training"
    assert loop.status()["job"] is None and loop.candidate == 1


# ───────────────────────────── two alarm sources, rollback
def test_the_second_alarm_source_restarts_the_promise():
    with pytest.raises(ValueError):                     # never the detector alone
        AdapterLoop(_Hand([]), question="q", wait=True, max_risk=0.05, secret=KEY, detector=RiskDetector(0.05), monitor=None, additive=False)
    auto = AdapterLoop(_Hand([]), question="q", wait=True, max_risk=0.05, secret=KEY, detector=RiskDetector(0.05), additive=False)
    assert type(auto.monitor).__name__ == "NumericDrift"
    flag = _Flag()
    loop = AdapterLoop(_Hand([]), question="q", wait=True, max_risk=0.05, secret=KEY, initial=_Model("s1", 0.9),
                       detector=RiskDetector(0.05, arl=10 ** 6), monitor=flag, policy=_plain(first=10 ** 6), additive=False)
    data = _fake(5, 1500)
    _run(loop, data[:1000])
    assert loop.versions()[0]["certified"] and not _events(loop, "alarm")
    flag.fire = True
    out = loop.step(data[1000], lambda: data[1000][0])
    assert "alarm" in out["events"] and "restart" in out["events"]
    alarm = _events(loop, "alarm")[0]
    assert alarm["who"] == "monitor" and alarm["action"] == "recertify" and alarm["R"] is None
    assert loop.versions()[0]["threshold"] is None and loop.left == 300
    assert [e for e in _events(loop, "restart") if e["version"] == 0 and e["why"] == "alarm"]
    assert not flag.fire                                # re-referenced at the alarm


def test_the_auto_monitor_alarms_on_a_score_shift_of_the_served_version():
    """The auto monitor (NumericDrift beside the detector) stepped on a live loop: quiet for 1 500 decisions, then the
    served version's scores move and it alarms; the replay is exact."""
    class _Shift:
        def __init__(self):
            self.share = 0.9

        def score(self, x):
            return (x[0], 0.95) if x[1] < self.share else ("wrong", 0.05)

        def fingerprint(self):
            return "shift"
    m = _Shift()
    loop = AdapterLoop(_Hand([]), question="q", wait=True, max_risk=0.05, secret=KEY, initial=m, detector=RiskDetector(0.05),
                       policy=_plain(first=10 ** 6), additive=False)
    assert type(loop.monitor).__name__ == "NumericDrift"
    xs = _fake(5, 3000)
    _run(loop, xs[:1500])
    assert not _events(loop, "alarm")
    m.share = 0.3
    _run(loop, xs[1500:])
    al = _events(loop, "alarm")
    assert al and al[0]["who"] in ("monitor", "detector+monitor") and 1500 < al[0]["decision"] < 1600
    assert loop.replay()["ok"]


def test_rollback_follows_rollback_target():
    flag = _Flag()
    models = [_Model("half", 0.5), _Model("good", 1.0)] + [_Model(f"w{i}", 0.3) for i in range(40)]
    loop = AdapterLoop(_Hand(models), question="q", wait=True, max_risk=0.05, secret=KEY, monitor=flag, probation=0,
                       policy=_plain(growth=1.0, trial=300, probation=300), additive=False)
    data = iter(_fake(6, 20000))

    def go(k):
        for _ in range(k):
            x = next(data)
            loop.step(x, lambda x=x: x[0])

    def alarm():
        flag.fire = True
        go(1)
    go(4000)
    assert loop.served == 2 and [e["to"] for e in _events(loop, "promote")] == [1, 2]
    alarm()
    go(10)
    alarm()                                             # a second alarm on version 2 within 300 decisions
    rb = _events(loop, "rollback")
    assert loop.served == 1 and len(rb) == 1 and rb[0]["by"] == "policy" and rb[0]["from"] == 2
    alarm()
    go(10)
    alarm()                                             # twice on version 1: back to 0, never to the alarmed 2
    rb = _events(loop, "rollback")
    assert loop.served == 0 and [e["to"] for e in rb] == [1, 0]
    assert [a["action"] for a in _events(loop, "alarm")] == ["recertify", "rollback:1", "recertify", "rollback:0"]


# ───────────────────────────── the stand-in batch learner end to end
def test_the_stand_in_batch_learner_runs_the_default_policy_end_to_end():
    r = random.Random(7)
    voc = {c: [f"{c}w{i}" for i in range(12)] for c in "abcde"}
    data = []
    for _ in range(2500):
        c = r.choice("abcde")
        data.append((" ".join(r.sample(voc[c], 3) + [f"n{r.randrange(40)}"]), c))
    loop = AdapterLoop(HashedTextTrainer(list("abcde")), question="q", wait=True, max_risk=0.05, secret=KEY,
                       policy=DefaultPolicy(Budget(min_interval_s=0.0)), additive=False)
    for x, y in data:
        loop.step(x, lambda y=y: y)
    st = loop.status()
    assert st["labels"]["calib"] > 0 and st["labels"]["holdout"] > 0
    trains = _events(loop, "train")
    assert trains and all(t["report"]["n_holdout"] >= 100 for t in trains)
    # stage-2 review S2: no training is refused at birth for want of held-out or calib labels, and none follows a
    # promotion before the new served version was measured
    assert all(t["frozen"] for t in trains), [t["why"] for t in trains]
    assert _events(loop, "freeze")
    for v in loop._vs.values():                         # it never trained on a held-out or calib key
        assert all(loop._split_of[k] == "train" for k in v["keys"])
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


# ───────────────────────────── a hook that raises stops the loop (stage-2 review B2)
_FIRED = [False]                                       # module state, not in the pickled policy: fires once, live only


class _RaisesOnce(DefaultPolicy):
    def _maybe(self, st):
        if st.n >= 700 and not _FIRED[0]:
            _FIRED[0] = True
            raise RuntimeError(f"a policy bug at decision {st.n}")

    def want_train(self, st):
        self._maybe(st)
        return super().want_train(st)

    def promote(self, st, paired):
        self._maybe(st)
        return super().promote(st, paired)


def test_a_hook_that_raises_stops_the_loop_and_the_replay_stays_exact():
    """The reviewer's case: a policy hook raises once at decision 700 (a transient bug). Before the fix the loop went on
    unrecorded and replay broke for good ("order": decision 701 after 699). Now the decision is recorded with an
    "error" event, the loop stops (status "error"; step and controls refuse) and replay is exact."""
    _FIRED[0] = False
    loop = AdapterLoop(_Hand([_Model("good", 1.0)]), question="q", wait=True, max_risk=0.05, secret=KEY,
                       policy=_RaisesOnce(Budget(min_interval_s=0.0, per_day=10 ** 6), calib=0.0, holdout=0.0,
                                          pre_freeze=False, probe=False, first=100), additive=False)
    data = _fake(2, 1200)
    _run(loop, data[:699])
    with pytest.raises(RuntimeError, match="a policy bug at decision 700"):
        loop.step(data[699], lambda: data[699][0])
    assert loop.n == 700 and sum(1 for k, _ in loop._items() if k == "loop") == 700
    rec = [d for k, d in loop._items() if k == "loop"][-1]["stream"]
    assert rec["n"] == 700 and rec["error"]["where"] in ("policy.promote", "policy.want_train")
    err = _events(loop, "error")
    assert len(err) == 1 and err[0]["why"] == "RuntimeError: a policy bug at decision 700"
    st = loop.status()
    assert st["state"] == "error" and st["error"]["n"] == 700 and st["paused"]
    for call in (lambda: loop.step(data[700], lambda: data[700][0]), loop.train_now, loop.resume,
                 lambda: loop.rollback(0)):
        with pytest.raises(RuntimeError, match="the loop stopped on an error at decision 700"):
            call()
    raised = 0
    for x in data[700:]:                                   # the reviewer's driver: catch and step on
        try:
            loop.step(x, lambda x=x: x[0])
        except RuntimeError:
            raised += 1
    assert raised == 500 and loop.n == 700                # every later step refused: nothing unrecorded
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]
    assert r["decisions"] == 700 and not any(m[1] == "order" for m in r["mismatches"])


class _ScoreFails:
    def score(self, x):
        raise MemoryError("out of memory scoring the held-out labels")

    def fingerprint(self):
        return "fails"


class _KeysFail(_Hand):
    def keys(self):
        raise RuntimeError("the label store is gone")


def test_a_learner_failing_after_the_training_stops_the_loop_in_a_step_and_in_a_control():
    """The held-out scoring raising inside a step (the decision is cut at "report"), and the learner's keys() raising
    inside train_now (the control is cut at "learner.keys"): both stop the loop and both replay exactly."""
    a = AdapterLoop(_Hand([_ScoreFails()]), question="q", wait=True, max_risk=0.05, secret=KEY,
                    policy=DefaultPolicy(Budget(min_interval_s=0.0), pre_freeze=False, first=20, calib=0.0,
                                         holdout=0.5), additive=False)
    data = _fake(8, 400)
    with pytest.raises(MemoryError):
        _run(a, data)
    assert a.status()["state"] == "error" and a.error["where"] == "report" and a.status()["job"] is None
    r = a.replay()
    assert r["ok"], r["mismatches"][:5]
    b = AdapterLoop(_KeysFail([_Model("m", 0.9)]), question="q", wait=True, max_risk=0.05, secret=KEY,
                    policy=_plain(first=10 ** 6), additive=False)
    _run(b, data[:50])
    with pytest.raises(RuntimeError, match="the label store is gone"):
        b.train_now()
    assert b.error["where"] == "learner.keys" and _events(b, "error")[0]["of"] == "train_now"
    with pytest.raises(RuntimeError, match="stopped"):
        b.step(data[50], lambda: data[50][0])
    r = b.replay()
    assert r["ok"], r["mismatches"][:5]


def test_a_refused_control_or_label_does_not_stop_the_loop():
    loop = AdapterLoop(_Hand([]), question="q", wait=True, max_risk=0.05, secret=KEY, policy=_plain(first=10 ** 6), additive=False)
    _run(loop, _fake(9, 10))
    with pytest.raises(ValueError):
        loop.rollback(3)                                   # refused before anything changed
    with pytest.raises(ValueError):
        loop.promote()
    from solvi.core.sources import UntrustedLabel
    with pytest.raises(UntrustedLabel):
        loop.step(("a", 0.1), lambda: "a", source="model")
    assert loop.status()["state"] == "running" and loop.error is None
    _run(loop, _fake(10, 5))
    assert loop.n == 15


# ───────────────────────────── a learner's own keys (stage-2 review S3), its update (S4), a refused train_now
class _Keeps(_Hand):
    """A batch learner that also trained on a key the loop never handed it (warm start, a label store)."""

    def keys(self):
        return {"extra-key"}


def test_a_learners_own_keys_are_training_keys():
    loop = AdapterLoop(_Keeps([_Model("m", 1.0)]), question="q", wait=True, max_risk=0.05, secret=KEY, policy=_plain(first=20), additive=False)
    data = _fake(11, 400)
    _run(loop, data[:20])
    assert loop.candidate == 1 and "extra-key" in loop._vs[1]["keys"]
    assert _events(loop, "train")[0]["extra_keys"] == ["extra-key"]
    loop.step(data[20], lambda: data[20][0], key="extra-key")
    assert loop._vs[1]["samples"][-1][4] is True          # its sample is "seen", not novel
    loop.step(data[21], lambda: data[21][0], key="fresh-key")
    assert loop._vs[1]["samples"][-1][4] is False
    _run(loop, data[22:])
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_update_is_called_once_by_its_signature_and_its_type_error_is_not_masked():
    calls = []

    class _Plain(_Hand):
        def update(self, x, y):                            # no key= / source=
            calls.append((x, y))

    loop = AdapterLoop(_Plain([]), question="q", wait=True, max_risk=0.05, secret=KEY, policy=_plain(first=10 ** 6), additive=False)
    _run(loop, _fake(12, 5))
    assert len(calls) == 5

    class _Bad(_Hand):
        def update(self, x, y, *, key=None, source=None):
            calls.append(key)
            raise TypeError("a bad label type")

    calls.clear()
    bad = AdapterLoop(_Bad([]), question="q", wait=True, max_risk=0.05, secret=KEY, policy=_plain(first=10 ** 6), additive=False)
    with pytest.raises(TypeError, match="a bad label type"):
        bad.step(("a", 0.5), lambda: "a")
    assert len(calls) == 1 and calls[0] is not None      # called once, with key=; no retry without it
    assert bad.status()["state"] == "error" and bad.error["where"] == "learner.update"
    assert bad.replay()["ok"]


def test_a_refused_train_now_writes_its_event_and_replays():
    clock = [1000.0]
    loop = AdapterLoop(_Hand([_Model("m", 0.4), _Model("m2", 0.4)]), question="q", wait=True, max_risk=0.05, secret=KEY,
                       policy=DefaultPolicy(Budget(min_interval_s=600.0), first=10 ** 6, pre_freeze=False),
                       clock=lambda: clock[0], additive=False)
    _run(loop, _fake(13, 30))
    loop.pause()
    assert loop.train_now().why == "paused"
    loop.resume()
    assert loop.train_now().state == "done"
    ev = [e for e in _events(loop, "train") if e.get("state") == "refused"]
    assert len(ev) == 1 and ev[0]["control"] and ev[0]["why"] == "paused"
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_versions_and_replay_take_the_lock():
    """Stage-2 review S5: versions() waits for a step in progress (it reads the table under the lock)."""
    loop = AdapterLoop(_Hand([]), question="q", wait=True, max_risk=0.05, secret=KEY, policy=_plain(first=10 ** 6), additive=False)
    _run(loop, _fake(14, 10))
    got = []
    with loop._lock:
        th = threading.Thread(target=lambda: got.append(loop.versions()))
        th.start()
        th.join(timeout=0.3)
        assert th.is_alive() and not got                  # blocked while the lock is held
    th.join(timeout=10)
    assert got and got[0][0]["version"] == 0
