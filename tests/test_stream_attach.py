"""solvi.experimental.stream, the adapter loop (stage 5): attached to a Dispatcher (stream mode) and to a System alone
(label mode). No torch: a stand-in learner whose versions install on the System through `Served`, a real System and a
real Dispatcher. The decisions go through the loop outside the part's call; the served version is what System 1
decides with; every Dispatched records the loop's view of it; `replay_all` re-installs the version each decision was
made under; a training runs in the background while decisions go on; controls through the handle; save / load of the
loop and a new Dispatcher; detach; refusals (a combination or a member of one, a Dispatcher with its own certificate);
the lock order holds under threads (a timeout catches a deadlock). The LoRA end to end is in test_stream_lora.py."""
import hashlib
import json
import threading
import warnings
from pathlib import Path

import pytest
from test_llm import FakeLLM

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.deciders.combine import Cascade
from solvi.core.deciders.llm import llm
from solvi.core.dispatch import AskPath, Dispatcher, SlowPath, Thought
from solvi.core.store import JSONLStorage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy, LoopPolicy, StreamPromise, Watch

KEY = b"attach-test-key"
TEAMS = ("billing", "shipping")
ACCS = [0.9, 0.97, 0.99, 0.6, 0.995]
URL = "http://127.0.0.1:9/v1"


def _gold(text):
    return "billing" if "charged" in text else "shipping"


def _u(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 2 ** 32


def _text(x):
    return x["email"] if isinstance(x, dict) else x


class Ver:
    """A stand-in version (Frozen and Served): right with score 0.95 on a share `acc` of the inputs, else wrong with
    0.05; installed by putting itself in the System's holder, which System 1's rule reads. lock: a part whose version
    lock score takes (shadow scoring under the part's lock, as the LoRA version's)."""

    lock = None

    def __init__(self, name, acc):
        self.name, self.acc = name, acc

    def score(self, x):
        if Ver.lock is not None:
            from solvi.core.deciders.part import version_lock
            with version_lock(Ver.lock):
                pass
        t = _text(x)
        g = _gold(t)
        return (g, 0.95) if _u(t) < self.acc else ([o for o in TEAMS if o != g][0], 0.05)

    def fingerprint(self):
        return f"ver-{self.name}-{self.acc}"

    def install(self, system):
        prev = system.holder.ver
        system.holder.ver = self
        return {"from": None if prev is None else prev.fingerprint(), "to": self.fingerprint()}

    def uninstall(self, system):
        pass

    def save(self, dir, v):
        p = Path(dir) / f"{v}.ver.json"
        p.write_text(json.dumps([self.name, self.acc]))
        return [p]

    @classmethod
    def load(cls, dir, v, *, expect_fp):
        name, acc = json.loads((Path(dir) / f"{v}.ver.json").read_text())
        out = cls(name, acc)
        if out.fingerprint() != expect_fp:
            raise ValueError("not the recorded version")
        return out


class Learner:
    """A batch learner: the version a training gives depends only on its number. gate: an Event the training waits for."""

    def __init__(self, gate=None):
        self._seen, self.gate = 0, gate

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        if self.gate is not None:
            self.gate.wait(30)
        return Ver(f"v{job.version}", ACCS[(job.version - 1) % len(ACCS)])


class Holder:
    def __init__(self, ver):
        self.ver, self.used = ver, []


def _s1(storage=None):
    """System 1: the version in the System's holder answers (its score as the probability)."""
    h = Holder(Ver("v0", 0.5))
    cat = Catalog()

    @cat.rule("team")
    def team(email):
        a, s = h.ver.score(email)
        h.used.append(h.ver.fingerprint())
        return Decision(a, {a: s, [o for o in TEAMS if o != a][0]: 1 - s})
    s = System(cat, [Question("team", "Which team?", Answer.choice(list(TEAMS)))], storage=storage)
    s.holder = h
    return s


def _s2():
    cat = Catalog()

    @cat.rule("team")
    def team(email):
        return Decision(_gold(email), {_gold(email): 1.0})
    return System(cat, [Question("team", "Which team?", Answer.choice(list(TEAMS)))])


def _policy(**kw):
    """Stream mode's test policy: no split, the stream certificate alone decides (pre_freeze=False)."""
    base = dict(calib=0.0, holdout=0.0, pre_freeze=False, probe=False, growth=1.0, trial=300, first=200,
                s2_labels=True)
    base.update(kw)
    return DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6, max_versions_kept=10), **base)


def _label_policy(**kw):
    """Label mode's test policy: the pre-freeze gates on (calib and held-out labels; label mode refuses them off)."""
    base = dict(first=100, growth=1.0, calib=0.15, holdout=0.15, pre_freeze=True, min_holdout=20)
    base.update(kw)
    return _policy(**base)


def _attach(disp, learner=None, **kw):
    base = dict(question="team", max_risk=0.05, secret=KEY, initial=disp.system.holder.ver, policy=_policy(),
                wait=True, clock=lambda: 1000.0, additive=False)
    base.update(kw)
    return AdapterLoop.attach(disp, Learner() if learner is None else learner, **base)


def _emails(n, start=0):
    return [{"email": f"order {i}: " + ("I was charged twice" if i % 3 else "my parcel is late")}
            for i in range(start, start + n)]


def _disp(s1=None, storage=None, **kw):
    return Dispatcher(_s1(storage) if s1 is None else s1, AskPath(_s2()), secret=KEY, storage=storage, **kw)


# ───────────────────────────── stream mode, end to end
def test_a_dispatcher_with_a_loop_decides_with_the_served_version_records_it_and_replays_exactly(tmp_path):
    store = JSONLStorage(tmp_path / "d.jsonl")
    disp = _disp(storage=store)
    loop = _attach(disp)
    assert disp.loop is loop and disp.system.adapter_loop is loop and loop.status()["attached"] == "dispatcher"
    out = [disp.ask(x) for x in _emails(3000)]
    used = disp.system.holder.used
    assert len(used) == len(out)
    for d, fp in zip(out, used):                      # System 1 decided with the version the loop says was served
        assert d.learning["fingerprint"] == fp, d.learning
        assert d.learning["n"] == d.n                 # attached from the first decision: one numbering
    served = [d.learning["served"] for d in out]
    assert len(set(served)) >= 2, loop.versions()    # a promotion happened during the run
    ev = [e for d in out for e in d.learning["events"]]
    assert "promote" in ev and "freeze" in ev
    holds = [d for d in out if d.action == "think" and any("learning loop" in r for r in d.reasons)]
    assert holds                                     # before its certificate the loop's rule escalated
    assert loop.replay()["ok"]
    assert disp.replay_all() == []                   # every decision, each under its own version
    assert disp.system.holder.ver.fingerprint() == loop.versions()[loop.served]["fingerprint"]
    # teach in stream mode: stored by the System, not a label of the loop (its labels come with its decisions)
    n = loop.n
    disp.system.teach("team", {"email": "order x: I was charged"}, "billing")
    assert loop.n == n
    # a rollback through the handle is served at once and replays
    target = next(v["version"] for v in reversed(loop.versions()) if v["state"] == "retired")
    disp.loop.rollback(target, by="test")
    d = disp.ask(_emails(1, 5000)[0])
    assert d.learning["served"] == target and disp.system.holder.used[-1] == d.learning["fingerprint"]
    assert loop.replay()["ok"] and disp.replay_all() == []


def test_a_training_runs_in_the_background_while_decisions_go_on():
    gate = threading.Event()
    disp = _disp()
    loop = _attach(disp, Learner(gate), wait=False)
    xs = _emails(1200)
    for x in xs[:260]:
        disp.ask(x)
    st = loop.status()
    assert st["job"] is not None and st["state"] == "training", st
    started = loop.n
    for x in xs[260:300]:                            # the training waits on the gate; decisions do not
        d = disp.ask(x)
        assert d.learning["served"] == 0
    gate.set()
    assert loop.wait_training(30)
    d = disp.ask(xs[300])
    assert "train" in d.learning["events"] and "freeze" in d.learning["events"], d.learning
    train = [e for k, e in loop.log if k == "stream" and e["event"] == "train"][0]
    assert train["started_at"] <= started < train["decision"]
    for x in xs[301:]:
        disp.ask(x)
    assert loop.replay()["ok"]


def test_controls_through_the_handle_and_the_status():
    disp = _disp()
    loop = _attach(disp)
    for x in _emails(100):
        disp.ask(x)
    h = disp.loop
    assert h.pause(by="op")["event"] == "pause" and h.status()["state"] == "paused"
    assert h.train_now().state == "refused"           # never while paused
    assert h.resume(by="op")["event"] == "resume"
    job = h.train_now(wait=True)
    assert job.state == "done" and h.status()["candidate"] is not None
    for x in _emails(50, 100):
        disp.ask(x)
    st = h.status()
    assert st["mode"] == "stream" and st["attached"] == "dispatcher" and "anytime" in st["guarantee"]
    assert loop.replay()["ok"]


def test_save_load_and_a_new_dispatcher_continue_as_the_unbroken_run(tmp_path):
    xs = _emails(3200)
    da = _disp()
    la = _attach(da)
    for x in xs[:2400]:
        da.ask(x)
    assert la.served != 0 or la.candidate is not None, la.versions()
    la.save(tmp_path / "ck")
    db = _disp()                                      # a new System (version 0 in its holder) and Dispatcher
    lb = AdapterLoop.load(tmp_path / "ck", Learner(), system=db.system, secret=KEY, clock=lambda: 1000.0)
    assert db.system.holder.ver.fingerprint() == da.system.holder.ver.fingerprint()   # the served version installed
    lb.attach_to(db)
    for x in xs[2400:]:
        a, b = da.ask(x), db.ask(x)
        assert (a.answer, a.by, a.action) == (b.answer, b.by, b.action)
        assert a.learning == b.learning
    assert la.versions() == lb.versions() and lb.replay()["ok"]


def test_detach_puts_version_0_back_and_removes_the_hooks(tmp_path):
    disp = _disp(storage=JSONLStorage(tmp_path / "d.jsonl"))
    loop = _attach(disp)
    for x in _emails(2500):
        disp.ask(x)
    assert loop.served != 0, loop.versions()
    e = loop.detach(by="test")
    assert e["event"] == "detach" and e["from"] == loop.served
    assert disp.loop is None and disp.system.adapter_loop is None and loop.host is None
    assert disp.system.holder.ver.fingerprint() == "ver-v0-0.5"
    assert loop.status()["state"] == "detached"
    with pytest.raises(RuntimeError, match="detached"):
        loop.step({"email": "x"}, lambda: "billing")
    d = disp.ask(_emails(1, 9000)[0])
    assert d.learning is None and disp.system.holder.used[-1] == "ver-v0-0.5"
    assert loop.replay()["ok"]                        # the detach is a recorded control
    bad = disp.replay_all()                           # decisions made under the loop need it to replay
    assert bad and all(any(w == "learning" for w, _ in mm) for _, _, mm in bad)
    assert loop.detach() is None


def test_a_control_from_another_thread_waits_for_the_decision_in_flight(tmp_path):
    """A rollback sent while a decision is between the loop's read and System 1's call: it waits for that decision
    (the Dispatcher's per-decision lock), so the decision is recorded under the version System 1 decided with and the
    replay is exact."""
    disp = _disp(storage=JSONLStorage(tmp_path / "d.jsonl"))
    loop = _attach(disp)
    for x in _emails(2500):
        disp.ask(x)
    assert loop.served != 0 and disp.replay_all() == []
    target = next(v["version"] for v in reversed(loop.versions()) if v["state"] == "retired")
    real, state = disp.system.ask, {}

    def ask_with_control(*a, **k):                   # runs between _pre and System 1's call
        if not state:
            th = threading.Thread(target=lambda: state.update(e=loop.rollback(target, by="ops")))
            state["th"] = th
            th.start()
            th.join(0.5)
            state["blocked"] = th.is_alive()         # waiting for this decision
        return real(*a, **k)
    disp.system.ask = ask_with_control
    try:
        d = disp.ask(_emails(1, 7000)[0])
    finally:
        disp.system.ask = real
    state["th"].join(10)
    assert state["blocked"] and state["e"]["event"] == "rollback"
    assert d.learning["fingerprint"] == disp.system.holder.used[-1] and "step" not in d.learning
    assert loop.served == target
    d2 = disp.ask(_emails(1, 7001)[0])
    assert d2.learning["served"] == target
    assert disp.replay_all() == [] and loop.replay()["ok"]


def test_with_a_loop_a_dispatcher_serves_one_decision_at_a_time():
    """The per-decision lock: two threads with a slow path; each decision's loop number is the Dispatcher's (no two
    decisions read the same number between the loop's read and its step)."""
    import time

    def slow():
        cat = Catalog()

        @cat.rule("team")
        def team(email):
            time.sleep(0.02)
            return Decision(_gold(email), {_gold(email): 1.0})
        return System(cat, [Question("team", "Which team?", Answer.choice(list(TEAMS)))])
    disp = Dispatcher(_s1(), AskPath(slow()), secret=KEY)
    loop = _attach(disp)
    out, errs = [], []

    def go(k):
        try:
            for x in _emails(15, 100 * k):
                out.append(disp.ask(x))
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
    th = [threading.Thread(target=go, args=(k,)) for k in range(2)]
    for t in th:
        t.start()
    for t in th:
        t.join(60)
    assert not errs, errs
    assert len(out) == 30 and loop.n == 30
    assert all(d.learning["n"] == d.n and "step" not in d.learning for d in out), \
        [(d.n, d.learning.get("n")) for d in out]
    assert sorted(d.learning["n"] for d in out) == list(range(1, 31))


def test_the_candidate_is_scored_in_shadow_only_on_grid_and_sampled_decisions():
    calls = {"n": 0}

    class Counted(Ver):
        def score(self, x):
            calls["n"] += 1
            return super().score(x)

    class L(Learner):
        def train(self, job):
            return Counted(f"c{job.version}", 0.97)
    disp = _disp()
    loop = _attach(disp, L(), policy=_policy(trial=10 ** 6, first=200))
    xs = _emails(1600)
    for x in xs[:300]:
        disp.ask(x)
    c = loop.candidate
    assert c is not None, loop.versions()
    trial, scored = [], 0
    for x in xs[300:1500]:
        if loop.candidate != c:                      # promoted: System 1 itself scores with it from here
            break
        before = calls["n"]
        trial.append(disp.ask(x))
        scored += calls["n"] - before
    sampled = sum(1 for d in trial if d.learning["sampled"])
    assert len(trial) > 400 and all(d.learning["candidate"] == c for d in trial)
    assert scored <= 200 + sampled, (scored, sampled, len(trial))   # not once per decision
    assert loop.replay()["ok"]


def test_stream_mode_without_a_label_source_is_refused_with_the_options():
    pol = DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6))      # s2_labels False: the default
    with pytest.raises(ValueError, match=r"s2_labels=True.*trusted oracle.*label mode") as e:
        _attach(_disp(), policy=pol)
    assert "every decision would escalate" in str(e.value)

    assert _attach(_disp(), policy=DefaultPolicy(s2_labels=True)).host is not None


class Trusted(SlowPath):
    """A trusted oracle in the process: says label_source = "human" and its Thoughts carry the oracle record (the
    labels' source at run time). mark=False: the declaration without the record (a wrong declaration)."""
    mode = "trusted_attach_test"
    label_source = "human"

    def __init__(self, system, mark=True, **kw):
        super().__init__(system, **kw)
        self.mark = mark

    def think(self, state, question, **kw):
        rec = {"oracle": {"label_source": "human"}} if self.mark else {"read": _text(state)}
        return Thought(self.mode, _gold(_text(state)), True, record=rec)

    def fingerprint(self):
        return f"trusted-attach-test-{self.mark}"


def test_a_trusted_oracle_as_the_slow_path_attaches_and_its_labels_flow():
    disp = Dispatcher(_s1(), Trusted(_s2()), secret=KEY)
    loop = _attach(disp, policy=DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6)))
    assert loop.host is disp
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = [disp.ask(x) for x in _emails(600)]
    assert not any("refused the last" in str(w.message) for w in rec)          # labels flow: no warning
    st = loop.status()
    assert st["labels"]["refused"] == 0 and sum(st["labels"][k] for k in ("train", "calib", "holdout")) > 100, st
    assert any(d.learning["label_source"] == "human" for d in out)
    assert loop.versions()[0]["certified"], loop.versions()[0]      # version 0 certified on the oracle's labels
    assert loop.replay()["ok"]


def test_a_declared_label_source_whose_thoughts_are_not_marked_warns_and_counts_the_refusals():
    disp = Dispatcher(_s1(), Trusted(_s2(), mark=False), secret=KEY)
    loop = _attach(disp, policy=DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6)), warn_refused=50)
    assert loop.host is disp                         # the declaration passes the attach
    with pytest.warns(UserWarning, match=r"refused the last 50 labels \(source 's2'\).*learns nothing") as rec:
        for x in _emails(120):
            disp.ask(x)
    assert sum("refused the last" in str(w.message) for w in rec) == 1          # once
    st = loop.status()
    assert st["labels"]["refused"] == 120 and st["labels"]["train"] == st["labels"]["calib"] == 0, st
    assert not loop.versions()[0]["certified"]


def test_a_refused_attach_leaves_nothing_attached():
    s1 = _s1()
    loop = AdapterLoop(Learner(), question="team", max_risk=0.05, secret=KEY, initial=s1.holder.ver,
                       policy=DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6)), wait=True,
                       clock=lambda: 1000.0, additive=False)
    disp = Dispatcher(s1, AskPath(_s2()), secret=KEY)
    with pytest.raises(ValueError, match="refuses the slow path's labels"):
        loop.attach_to(disp)
    assert loop.host is None and loop.system is None and loop.part is None and loop._part_lock is None
    assert getattr(disp, "loop", None) is None and getattr(s1, "adapter_loop", None) is None
    other = Dispatcher(_s1(), Trusted(_s2()), secret=KEY)                     # another System: not "serves elsewhere"
    assert loop.attach_to(other).host is other and other.system.adapter_loop is loop


def test_teach_in_stream_mode_without_storage_warns_that_the_correction_is_lost():
    disp = _disp()
    loop = _attach(disp)
    with pytest.warns(UserWarning, match="correction is lost"):
        assert disp.system.teach("team", {"email": "order x: I was charged"}, "billing") is None
    assert loop.n == 0


def test_a_combination_over_an_attached_part_is_refused_until_detach():
    s, (part, other), _ = _llm_system(2)
    disp = Dispatcher(s, AskPath(_s2()), secret=KEY)
    loop = AdapterLoop.attach(disp, Learner(), question="team", max_risk=0.05, secret=KEY, policy=_policy(), additive=False)
    assert loop.part is part
    with pytest.raises(ValueError, match="served by an adapter loop"):
        Cascade([part, other])
    from solvi.core.deciders.combine import combinations_of
    assert combinations_of(part) == []
    Cascade([other])                                  # another part: built
    loop.detach()
    keep = Cascade([part, other])                     # detached: built
    assert keep in combinations_of(part)


# ───────────────────────────── label mode
def test_a_system_alone_learns_from_teach_and_outcome_and_promotes_on_its_pre_freeze_gates(tmp_path):
    s1 = _s1(storage=JSONLStorage(tmp_path / "s.jsonl"))
    loop = AdapterLoop.attach(s1, Learner(), question="team", max_risk=0.05, secret=KEY, initial=s1.holder.ver,
                              policy=_label_policy(), wait=True, clock=lambda: 1000.0, additive=False)
    assert loop.mode == "label" and s1.adapter_loop is loop
    st = loop.status()
    assert st["guarantee"] == "fixed-set (act_guard on calib labels); no anytime certificate" and st["mode"] == "label"
    for x in _emails(700):
        s1.teach("team", x, _gold(x["email"]))
    assert loop.n == 700
    ev = [e for k, e in loop.log if k == "stream"]
    train = [e for e in ev if e["event"] == "train"]
    assert train and train[0]["frozen"] and train[0]["report"]["n_holdout"] >= 20, train
    assert [e["event"] for e in ev].count("promote") == 1, [e["event"] for e in ev]
    promote = next(e for e in ev if e["event"] == "promote")
    assert promote["l_k"] is None and promote["n_novel"] is None  # label mode has no paired samples: not zeros
    assert loop.served == 1 and loop.versions()[1]["certified"]   # measured on held-out labels, then served
    assert s1.holder.ver.fingerprint() == loop.versions()[loop.served]["fingerprint"]
    assert loop.status()["watch"]["target"] == 0                  # the watch runs from the promotion
    res = s1.ask({"email": "order 7: I was charged twice"})
    s1.outcome(res, "billing")
    assert loop.n == 701
    assert len(s1.storage.corrections()) == 701       # every label is stored too
    with pytest.raises(ValueError, match="label mode"):
        AdapterLoop(Learner(), question="team", mode="stream", secret=KEY, max_risk=0.05, additive=False).label({"email": "x"}, "billing")
    assert loop.replay()["ok"]
    loop.detach()
    assert s1.holder.ver.fingerprint() == "ver-v0-0.5" and s1.adapter_loop is None


class Sure(Ver):
    """Every answer with score 0.95, right or wrong."""

    def score(self, x):
        return super().score(x)[0], 0.95


class Worse(Learner):
    """Every training gives a version that is sure and right on 30 % only: the gates refuse it."""

    def train(self, job):
        return Sure(f"w{job.version}", 0.3)


def test_label_mode_never_serves_a_training_its_gates_refuse_and_refuses_the_gates_off():
    s1 = _s1()
    loop = AdapterLoop.attach(s1, Worse(), question="team", max_risk=0.05, secret=KEY, initial=s1.holder.ver,
                              policy=_label_policy(), wait=True, clock=lambda: 1000.0, additive=False)
    for x in _emails(700):
        s1.teach("team", x, _gold(x["email"]))
    ev = [e for k, e in loop.log if k == "stream"]
    train = [e for e in ev if e["event"] == "train"]
    assert train and not any(e["frozen"] for e in train), train
    assert any(e["event"] == "discard" and "freeze refused" in e["why"] for e in ev)
    assert "promote" not in [e["event"] for e in ev] and loop.served == 0 and s1.holder.ver.name == "v0"
    assert loop.replay()["ok"]
    for pol in (_policy(), _label_policy(pre_freeze=False)):       # label mode with the gates off: refused
        with pytest.raises(ValueError, match="pre_freeze=False measures nothing"):
            AdapterLoop.attach(_s1(), Learner(), question="team", max_risk=0.05, secret=KEY, policy=pol, additive=False)


class Uncalibrated(Ver):
    """Every answer with the same score and right on 70 %: no calib threshold exists (no certificate in label mode)."""

    def score(self, x):
        a, _ = super().score(x)
        return a, 0.5


class PassAll(DefaultPolicy):
    """The default policy with a freeze that lets every training through."""

    def freeze(self, st, report):
        return True


def test_label_mode_automatic_promotion_needs_the_certificate_as_the_promote_control():
    class L(Learner):
        def train(self, job):
            return Uncalibrated(f"u{job.version}", 0.7)
    s1 = _s1()
    pol = PassAll(Budget(min_interval_s=0.0, per_day=10 ** 6), **{**dict(first=100, growth=1.0, calib=0.15,
                                                                         holdout=0.15, min_holdout=20, probe=False)})
    loop = AdapterLoop.attach(s1, L(), question="team", max_risk=0.05, secret=KEY, initial=s1.holder.ver, policy=pol,
                              wait=True, clock=lambda: 1000.0, additive=False)
    for x in _emails(700):
        s1.teach("team", x, _gold(x["email"]))
    ev = [e["event"] for k, e in loop.log if k == "stream"]
    assert "freeze" in ev and "promote" not in ev, ev
    assert loop.served == 0 and loop.candidate is not None and not loop.versions()[loop.candidate]["certified"]
    with pytest.raises(ValueError, match="not certified"):
        loop.promote()
    assert loop.replay()["ok"]


class LetThrough(DefaultPolicy):
    """The default policy whose freeze serves any training with a calib threshold (the held-out gates skipped): what
    a freeze that measured well and a version that degraded after it look like to the loop once it is served."""

    def freeze(self, st, report):
        return report.threshold is not None


class Seq(Learner):
    """Version 1 right on 99 % of the inputs, version 2 on 60 % (probe 7a's pair)."""

    def train(self, job):
        return Ver(f"v{job.version}", {1: 0.99, 2: 0.6}[job.version])


def test_label_mode_watch_rolls_back_a_promoted_version_that_is_worse_than_the_one_before(tmp_path):
    s1 = _s1()
    pol = LetThrough(Budget(min_interval_s=0.0, per_day=10 ** 6), first=100, growth=1.0, calib=0.15, holdout=0.15,
                     min_holdout=20, probe=False)
    loop = AdapterLoop.attach(s1, Seq(), question="team", max_risk=0.05, secret=KEY, initial=s1.holder.ver,
                              policy=pol, wait=True, clock=lambda: 1000.0, storage=JSONLStorage(tmp_path / "l.jsonl"), additive=False)
    xs = _emails(3000)
    i = 0
    while loop.served != 1 and i < 1000:
        s1.teach("team", xs[i], _gold(xs[i]["email"]))
        i += 1
    assert loop.served == 1 and loop.status()["watch"]["target"] == 0
    for x in xs[i:i + 400]:                          # version 1 is better than version 0: the watch stays quiet
        s1.teach("team", x, _gold(x["email"]))
    i += 400
    assert loop.served == 1 and loop.status()["watch"]["n"] >= 100, loop.status()["watch"]
    assert loop.train_now(wait=True).state == "done" and loop.candidate == 2
    s1.teach("team", xs[i], _gold(xs[i]["email"]))  # promoted on the next label: 0.60 after 0.99
    i += 1
    assert loop.served == 2 and s1.holder.ver.name == "v2" and loop.status()["watch"]["target"] == 1
    while loop.served == 2 and i < len(xs):
        s1.teach("team", xs[i], _gold(xs[i]["email"]))
        i += 1
    ev = [e for _, e in loop._items() if e.get("event") == "rollback"]
    assert len(ev) == 1 and ev[0]["to"] == 1 and ev[0]["by"] == "policy" and ev[0]["why"].startswith("watch:"), ev
    w = ev[0]["watch"]
    assert w["n"] >= 100 and w["better"] > w["worse"] and w["target"] == 1, w
    assert loop.served == 1 and s1.holder.ver.name == "v1" and loop.status()["watch"] is None
    assert loop.replay()["ok"]                       # the watch and its rollback replay from the records


def test_label_mode_checkpoint_continues_as_the_unbroken_run(tmp_path):
    xs = _emails(1200)
    sa = _s1()
    la = AdapterLoop.attach(sa, Learner(), question="team", max_risk=0.05, secret=KEY, initial=sa.holder.ver,
                            policy=_label_policy(), wait=True, clock=lambda: 1000.0, additive=False)
    for x in xs[:700]:
        sa.teach("team", x, _gold(x["email"]))
    assert la.served == 1 and la.versions()[1]["certified"], la.versions()
    la.save(tmp_path / "ck")
    sb = _s1()
    lb = AdapterLoop.load(tmp_path / "ck", Learner(), system=sb, secret=KEY, clock=lambda: 1000.0)
    lb.attach_to(sb)
    assert lb.mode == "label" and sb.holder.ver.fingerprint() == sa.holder.ver.fingerprint()
    assert lb.versions() == la.versions()            # the fixed-set thresholds restored (certified, threshold)
    sta, stb = la.status(), lb.status()
    assert {k: sta[k] for k in ("served", "n", "labels", "watch", "guarantee")} == \
        {k: stb[k] for k in ("served", "n", "labels", "watch", "guarantee")}
    for x in xs[700:]:
        sa.teach("team", x, _gold(x["email"]))
        sb.teach("team", x, _gold(x["email"]))
    assert la.n == lb.n == 1200
    ra = [e["stream"] for k, e in la.log if k == "loop"][-500:]
    rb = [e["stream"] for k, e in lb.log if k == "loop"][-500:]
    assert ra == rb                                  # the same records: thresholds, held-out outcomes, the watch
    assert la.versions() == lb.versions() and lb.replay()["ok"]


def test_label_mode_checkpoint_saves_and_restores_the_watch_e_process(tmp_path):
    xs = _emails(1000)
    sa = _s1()
    la = AdapterLoop.attach(sa, Learner(), question="team", max_risk=0.05, secret=KEY, initial=sa.holder.ver,
                            policy=_label_policy(), wait=True, clock=lambda: 1000.0, additive=False)
    for x in xs[:700]:
        sa.teach("team", x, _gold(x["email"]))
    w = la.status()["watch"]
    assert la.served == 1 and w is not None and w["better"] + w["worse"] > 0, w
    assert any(lw != 0.0 for lw in w["lw"]), w      # the e-process has bet on discordant labels
    la.save(tmp_path / "ck")
    sb = _s1()
    lb = AdapterLoop.load(tmp_path / "ck", Learner(), system=sb, secret=KEY, clock=lambda: 1000.0)
    lb.attach_to(sb)
    assert lb.status()["watch"] == w and lb._watch["lw"] == la._watch["lw"]
    for x in xs[700:]:
        sa.teach("team", x, _gold(x["email"]))
        sb.teach("team", x, _gold(x["email"]))
    assert lb.status()["watch"] == la.status()["watch"]
    ra = [e["stream"]["watch"] for k, e in la.log if k == "loop"][-300:]
    rb = [e["stream"]["watch"] for k, e in lb.log if k == "loop"][-300:]
    assert ra == rb and any(r.get("counted") and r.get("log_e") not in (None, 0.0) for r in ra)
    assert lb.replay()["ok"]


# ───────────────────────────── a custom policy
class OnlyHuman(LoopPolicy):
    """A minimal policy: human labels only, a third of the keys held out and a third calib, train at 100 labels,
    promote any candidate (the loop still needs it certified: a threshold on the calib labels)."""

    def accept(self, st, x, y, source):
        return source == "human"

    def split(self, st, key):
        return ("holdout", "calib", "train")[int(hashlib.sha256(key.encode()).hexdigest(), 16) % 3]

    def want_train(self, st):
        return st.labels["train"] >= 100 and st.served == 0

    def select(self, st, rows):
        return list(rows)

    def hyper(self, st):
        return {"seed": st.next_version}

    def freeze(self, st, report):
        return True

    def promote(self, st, paired):
        return True

    def discard(self, st, paired):
        return None

    def alarm(self, st, who, report):
        return "recertify"


def test_a_custom_policy_and_learner_attach():
    s1 = _s1()
    loop = AdapterLoop.attach(s1, Learner(), policy=OnlyHuman(), initial=s1.holder.ver, secret=KEY, max_risk=0.05,
                              wait=True, clock=lambda: 1000.0, additive=False)
    for x in _emails(400):
        s1.teach("team", x, _gold(x["email"]))
    assert loop.served == 1 and s1.holder.ver.name == "v1"
    assert loop.status()["watch"] is not None         # the watch's counts are kept ...
    hi = Watch(0, n=10 ** 4, better=10 ** 4, worse=0, total=10 ** 4, log_e=100.0)
    assert OnlyHuman().watch(None, hi) is False       # ... but a policy without its own watch hook never rolls back
    from solvi.core.sources import UntrustedLabel
    with pytest.raises(UntrustedLabel):
        s1.teach("team", _emails(1, 900)[0], "billing", label_source="outcome")
    assert loop.replay()["ok"]


# ───────────────────────────── refusals
def _llm_system(n=1, cascade=False):
    model = llm(URL, "m", api_key="k", opener=FakeLLM(), sleep=lambda s: None)
    parts = [model.decision("team", "Which team?", "email", {"billing": "b", "shipping": "s"}) for _ in range(n)]
    cat = Catalog()
    if cascade:
        c = Cascade(parts)
        return System(cat, [c.question(cat)]), parts, c
    return System(cat, [parts[0].question(cat)]), parts, None


def test_a_combination_or_a_member_of_one_is_refused():
    s, parts, _ = _llm_system(2, cascade=True)
    with pytest.raises(ValueError, match="combination"):
        AdapterLoop.attach(s, Learner(), question="team", secret=KEY, max_risk=0.05, additive=False)
    s1, ps, _ = _llm_system(2)
    keep = Cascade(ps)                                # the System's own part, also a member of a cascade elsewhere
    with pytest.raises(ValueError, match="member of a combination"):
        AdapterLoop.attach(s1, Learner(), question="team", secret=KEY, max_risk=0.05, additive=False)
    del keep


def test_other_refusals():
    sp = StreamPromise(max_risk=0.05)
    s1 = _s1()
    s1.guarantee("team", promise=sp, signal="confidence")
    d = Dispatcher(s1, AskPath(_s2()), audit="adaptive", promise=sp, secret=KEY)
    with pytest.raises(ValueError, match="own stream certificate"):
        _attach(d)
    disp = _disp()
    loop = _attach(disp)
    with pytest.raises(ValueError, match="attached already"):
        loop.attach_to(_disp())
    with pytest.raises(ValueError, match="loop attached already"):
        _attach(Dispatcher(disp.system, AskPath(_s2()), secret=KEY))
    with pytest.raises(ValueError, match="use ask"):
        disp.start(_emails(1)[0])
    with pytest.raises(ValueError, match="label-mode loop"):
        AdapterLoop(Learner(), question="team", secret=KEY, max_risk=0.05, additive=False).attach_to(_s1())
    with pytest.raises(ValueError, match="model-backed decision part"):
        AdapterLoop.attach(_s1(), question="team", dir="unused", additive=False)


# ───────────────────────────── the lock order under threads
def test_no_deadlock_between_decisions_controls_a_background_training_and_the_parts_own_calls():
    """Decisions through the Dispatcher in three threads, the part's own decisions (System 1 asked directly), controls
    and a background training whose report scores under the part's lock: everything ends (a deadlock would hang;
    the timeout catches it)."""
    s1, (part,), _ = _llm_system()
    s1.holder = Holder(Ver("v0", 0.5))
    Ver.lock = part
    try:
        disp = Dispatcher(s1, AskPath(_s2()), secret=KEY)
        loop = AdapterLoop.attach(disp, Learner(), question="team", max_risk=0.05, secret=KEY,
                                  initial=s1.holder.ver, policy=_policy(first=60), wait=False, additive=False)
        assert loop.part is part
        errs, stop = [], threading.Event()

        def asks(k):
            try:
                for x in _emails(150, 1000 * k):
                    disp.ask(x)
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

        def direct():
            try:
                while not stop.is_set():
                    s1.ask({"email": "my parcel is late"})
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

        def controls():
            try:
                while not stop.is_set():
                    loop.status()
                    loop.versions()
                    loop.pause()
                    loop.resume()
                    loop.train_now()
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))
        th = [threading.Thread(target=asks, args=(k,)) for k in range(3)]
        side = [threading.Thread(target=direct), threading.Thread(target=controls)]
        for t in th + side:
            t.start()
        for t in th:
            t.join(120)
        stop.set()
        for t in side:
            t.join(30)
        assert not any(t.is_alive() for t in th + side), "deadlock"
        assert not errs, errs[:3]
        assert loop.n == 450 and loop.replay()["ok"]
        loop.detach()
    finally:
        Ver.lock = None
