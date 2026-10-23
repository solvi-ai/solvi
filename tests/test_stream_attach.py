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
from solvi.core.dispatch import AskPath, Dispatcher
from solvi.core.store import JSONLStorage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy, LoopPolicy, StreamPromise

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
    base = dict(calib=0.0, holdout=0.0, pre_freeze=False, probe=False, growth=1.0, trial=300, first=200,
                s2_labels=True)
    base.update(kw)
    return DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6, max_versions_kept=10), **base)


def _attach(disp, learner=None, **kw):
    base = dict(question="team", max_risk=0.05, secret=KEY, initial=disp.system.holder.ver, policy=_policy(),
                wait=True, clock=lambda: 1000.0)
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


# ───────────────────────────── label mode
def test_a_system_alone_learns_from_teach_and_outcome_and_promotes_on_its_pre_freeze_gates(tmp_path):
    s1 = _s1(storage=JSONLStorage(tmp_path / "s.jsonl"))
    loop = AdapterLoop.attach(s1, Learner(), question="team", max_risk=0.05, secret=KEY, initial=s1.holder.ver,
                              policy=_policy(first=150), wait=True, clock=lambda: 1000.0)
    assert loop.mode == "label" and s1.adapter_loop is loop
    st = loop.status()
    assert st["guarantee"] == "fixed-set (act_guard on calib labels); no anytime certificate" and st["mode"] == "label"
    for x in _emails(160):
        s1.teach("team", x, _gold(x["email"]))
    assert loop.n == 160
    ev = [e["event"] for k, e in loop.log if k == "stream"]
    assert "freeze" in ev and "promote" in ev, ev
    assert loop.served != 0 and s1.holder.ver.fingerprint() == loop.versions()[loop.served]["fingerprint"]
    res = s1.ask({"email": "order 7: I was charged twice"})
    s1.outcome(res, "billing")
    assert loop.n == 161
    assert len(s1.storage.corrections()) == 161       # every label is stored too
    with pytest.raises(ValueError, match="label mode"):
        AdapterLoop(Learner(), question="team", mode="stream", secret=KEY, max_risk=0.05).label({"email": "x"}, "billing")
    assert loop.replay()["ok"]
    loop.detach()
    assert s1.holder.ver.fingerprint() == "ver-v0-0.5" and s1.adapter_loop is None


# ───────────────────────────── a custom policy
class OnlyHuman(LoopPolicy):
    """A minimal policy: human labels only, every third key held out, train at 100 labels, promote any candidate."""

    def accept(self, st, x, y, source):
        return source == "human"

    def split(self, st, key):
        return "holdout" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 3 == 0 else "train"

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
                              wait=True, clock=lambda: 1000.0)
    for x in _emails(200):
        s1.teach("team", x, _gold(x["email"]))
    assert loop.served == 1 and s1.holder.ver.name == "v1"
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
        AdapterLoop.attach(s, Learner(), question="team", secret=KEY, max_risk=0.05)
    s1, ps, _ = _llm_system(2)
    keep = Cascade(ps)                                # the System's own part, also a member of a cascade elsewhere
    with pytest.raises(ValueError, match="member of a combination"):
        AdapterLoop.attach(s1, Learner(), question="team", secret=KEY, max_risk=0.05)
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
        AdapterLoop(Learner(), question="team", secret=KEY, max_risk=0.05).attach_to(_s1())
    with pytest.raises(ValueError, match="model-backed decision part"):
        AdapterLoop.attach(_s1(), question="team", dir="unused")


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
                                  initial=s1.holder.ver, policy=_policy(first=60), wait=False)
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
