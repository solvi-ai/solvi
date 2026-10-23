"""solvi.experimental.stream, the adapter loop (stage 2): AdapterLoop driven by a LoopPolicy — the lane's LearningLoop
as DefaultPolicy at its constants, the novel-only comparison and the memorisation probe, controls, two alarm sources."""
import hashlib
import random
import warnings

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
def test_learning_loop_is_the_adapter_loop_with_the_default_policy_at_the_lanes_constants():
    loop = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, growth=1.5, trial=700)
    assert isinstance(loop, AdapterLoop) and type(loop.policy) is DefaultPolicy
    p = loop.policy
    assert (p.first, p.growth, p.refreeze, p.stale, p.trial, p.min_share, p.margin) == (300, 1.5, 1.5, 1500, 700,
                                                                                      0.05, 0.01)
    assert (p.calib, p.holdout, p.probe, p.pre_freeze, p.novel_risk, p.probation) == (0.0, 0.0, False, False, False, 0)
    # the same stream through AdapterLoop(policy=lane_policy(...)) gives the same records
    r = random.Random(1)
    voc = {c: [f"{c}w{i}" for i in range(12)] for c in "abcde"}
    data = []
    for _ in range(1500):
        c = r.choice("abcde")
        data.append((" ".join(r.sample(voc[c], 3)), c))
    a = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY)
    b = AdapterLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, policy=lane_policy())
    for x, y in data:
        a.step(x, lambda y=y: y)
        b.step(x, lambda y=y: y)
    def strip(loop):                                    # the wall clock and the training's wall time differ
        out = []
        for k, d in loop.log:
            if k == "stream":
                d = {f: v for f, v in d.items() if f != "t"}
                if d.get("report"):
                    d["report"] = {f: v for f, v in d["report"].items() if f != "seconds"}
            else:
                d = {**d, "stream": {f: v for f, v in d["stream"].items() if f != "t"}}
            out.append((k, d))
        return out
    assert strip(a) == strip(b)
    assert _events(a, "promote") and a.replay()["ok"] and b.replay()["ok"]


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


def _memo_run(**policy):
    loop = AdapterLoop(_MemoLearner(), question="q", max_risk=0.05, secret=KEY, key=lambda x: f"k{x[0]}",
                       policy=_plain(**policy), promise_options={"grid": [0.95, 0.5]})
    r, fresh = random.Random(21), 0
    for i in range(3000):
        if i < 300 or r.random() >= 0.7:
            kid, fresh = fresh, fresh + 1                 # a key nobody labelled yet
        else:
            kid = r.randrange(300)                         # a key the first candidate trained on
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


# ───────────────────────────── controls
def test_pause_blocks_training_and_promotion_but_not_labels():
    learner = _Hand([_Model("good", 1.0)] + [_Model(f"w{i}", 0.3) for i in range(5)])
    loop = AdapterLoop(learner, question="q", max_risk=0.05, secret=KEY, policy=_plain(first=100, trial=300))
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
    loop = AdapterLoop(learner, question="q", max_risk=0.05, secret=KEY, policy=policy, clock=lambda: clock[0])
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


def test_the_job_flag_is_set_during_a_training():
    seen = {}

    class _Spy(_Hand):
        def train(self, job):
            seen["status"], seen["state"] = loop.status()["job"], loop._state().job
            return super().train(job)
    loop = AdapterLoop(_Spy([_Model("m", 0.5)]), question="q", max_risk=0.05, secret=KEY, policy=_plain(first=20))
    assert loop.status()["job"] is None
    _run(loop, _fake(4, 20))
    assert seen["status"]["id"] == "1.1" and seen["status"]["trigger"] == "schedule"
    assert seen["state"] == seen["status"]
    assert loop.status()["job"] is None and loop.candidate == 1


# ───────────────────────────── two alarm sources, rollback
def test_the_second_alarm_source_restarts_the_promise():
    with pytest.raises(ValueError):                     # never the detector alone
        AdapterLoop(_Hand([]), question="q", max_risk=0.05, secret=KEY, detector=RiskDetector(0.05), monitor=None)
    auto = AdapterLoop(_Hand([]), question="q", max_risk=0.05, secret=KEY, detector=RiskDetector(0.05))
    assert type(auto.monitor).__name__ == "NumericDrift"
    flag = _Flag()
    loop = AdapterLoop(_Hand([]), question="q", max_risk=0.05, secret=KEY, initial=_Model("s1", 0.9),
                       detector=RiskDetector(0.05, arl=10 ** 6), monitor=flag, policy=_plain(first=10 ** 6))
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


def test_rollback_follows_rollback_target():
    flag = _Flag()
    models = [_Model("half", 0.5), _Model("good", 1.0)] + [_Model(f"w{i}", 0.3) for i in range(40)]
    loop = AdapterLoop(_Hand(models), question="q", max_risk=0.05, secret=KEY, monitor=flag, probation=0,
                       policy=_plain(growth=1.0, trial=300, probation=300))
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
    loop = AdapterLoop(HashedTextTrainer(list("abcde")), question="q", max_risk=0.05, secret=KEY,
                       policy=DefaultPolicy(Budget(min_interval_s=0.0)))
    for x, y in data:
        loop.step(x, lambda y=y: y)
    st = loop.status()
    assert st["labels"]["calib"] > 0 and st["labels"]["holdout"] > 0
    trains = _events(loop, "train")
    assert trains and all(t["report"]["n_holdout"] > 0 for t in trains)
    assert _events(loop, "freeze")
    for v in loop.versions()[1:]:                       # it never trained on a held-out or calib key
        assert v["train_keys"] <= st["labels"]["train"]
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]
