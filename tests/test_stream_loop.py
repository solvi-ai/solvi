"""solvi.experimental.stream, stage 3b: the closed learning loop — frozen versions, each certified only on samples
drawn after its freeze, promoted under δ-spending, every event in the chain and replayed exactly."""
import math
import pickle
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import FastHeadLearner, FrozenError, HashedTextLearner, LearningLoop

from solvi import Answer, Catalog, Question, System
from solvi.core.sources import UntrustedLabel
from solvi.core.store import JSONLStorage
from solvi.core.versions import serve_head

KEY = b"loop-test-key"
VOC = {c: [f"{c}w{i}" for i in range(12)] for c in "abcde"}
NOISE = [f"n{i}" for i in range(40)]


def _stream(seed, n, shuffle=False):
    """Five intents, each text three words of its own vocabulary and two shared noise words: learnable. shuffle: the
    labels permuted over the stream (the shuffled-label control: nothing to learn)."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        c = r.choice("abcde")
        words = r.sample(VOC[c], 3) + r.sample(NOISE, 2)
        r.shuffle(words)
        out.append((" ".join(words), c))
    if shuffle:
        labs = [y for _, y in out]
        r.shuffle(labs)
        out = [(x, y) for (x, _), y in zip(out, labs)]
    return out


def _run(loop, data):
    outs = []
    for x, y in data:
        outs.append(loop.step(x, lambda y=y: y))
    return outs


def _events(loop, *kinds):
    return [e for k, e in loop._items() if k == "stream" and (not kinds or e["event"] in kinds)]


# ───────────────────────────── fake versions: exact control over what each frozen model does
class _Model:
    """x = (label, u): right with score 0.95 when u < share, else a wrong answer with score 0.05."""

    def __init__(self, name, share):
        self.name, self.share = name, share

    def score(self, x):
        y, u = x[0], x[1]
        return (y, 0.95) if u < self.share else ("wrong", 0.05)

    def probs(self, x):
        a, s = self.score(x)
        return {a: s}

    def fingerprint(self):
        return f"model-{self.name}-{self.share}"


class _Fake:
    """A live learner that hands out the given models in turn at each freeze."""

    def __init__(self, models):
        self.models, self._seen = list(models), 0

    def update(self, x, y):
        self._seen += 1

    def seen(self):
        return self._seen

    def snapshot(self):
        return self.models.pop(0)


def _fake_data(seed, n):
    r = random.Random(seed)
    return [((r.choice("ab"), r.random()), None) for _ in range(n)]


def _run_fake(loop, data):
    return [loop.step(x, lambda x=x: x[0]) for x, _ in data]


def test_a_version_reads_only_samples_drawn_after_its_freeze():
    loop = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY)
    _run(loop, _stream(0, 1300))
    frozen = [v for v in loop.versions() if v["version"] >= 1]
    assert len(frozen) >= 2
    for v in frozen:
        fed = loop.fed(v["version"])
        assert fed, v
        assert min(fed) > v["frozen_at"]                   # never a sample from before or at its freeze
        assert v["frozen_at"] not in fed                   # the freeze decision was sampled (ρ = 1) and is not read
    # the decision a version was frozen at was itself a sample (ρ = 1 under version 0): it reached version 0's labels
    first = frozen[0]
    rec = next(d["stream"] for k, d in loop._items() if k == "loop" and d["stream"]["n"] == first["frozen_at"])
    assert rec["sampled"] and rec["candidate"] is None


def test_a_better_certified_candidate_is_promoted_a_worse_one_discarded():
    half, good, worse = _Model("half", 0.5), _Model("good", 1.0), _Model("worse", 0.3)
    loop = LearningLoop(_Fake([half, good, worse] + [_Model(f"w{i}", 0.3) for i in range(5)]), question="q",
                        max_risk=0.05, secret=KEY, growth=1.0, trial=300)
    _run_fake(loop, _fake_data(1, 9000))
    ev = [(e["event"], e["version"]) for e in _events(loop, "freeze", "promote", "discard")]
    assert ev[:4] == [("freeze", 1), ("promote", 1), ("freeze", 2), ("promote", 2)]
    promos = _events(loop, "promote")
    assert promos[0]["from"] == 0 and promos[0]["l_s"] == 0.0 and promos[0]["l_k"] > 0.3
    assert promos[1]["from"] == 1 and promos[1]["l_k"] >= promos[1]["l_s"] + 0.01 and promos[1]["l_k"] > 0.9
    # version 3 (right on 30 %) certifies but answers alone less often than version 2 on the same samples: discarded
    disc = _events(loop, "discard")
    assert disc and disc[0]["version"] == 3 and disc[0]["why"].startswith("certified, not better")
    # trial counts decisions since the freeze (decision 14), not samples: version 2 serves alone, so ρ < 1
    v3 = next(e for e in _events(loop, "freeze") if e["version"] == 3)
    assert disc[0]["decision"] >= v3["frozen_at"] + loop.trial and loop.trial == 300   # at the first certified one
    assert disc[0]["samples"] < loop.trial
    assert loop.served == 2


def test_an_uncertified_candidate_is_discarded_after_stale_decisions_and_its_delta_stays_spent():
    loop = LearningLoop(_Fake([_Model("wrong", 0.0), _Model("wrong2", 0.0)]), question="q", max_risk=0.05,
                        secret=KEY, stale=1500)
    _run_fake(loop, _fake_data(2, 1900))
    disc = _events(loop, "discard")
    assert len(disc) == 1 and disc[0]["version"] == 1
    assert disc[0]["why"] == "not certified 1500 decisions after its freeze (1500 samples)"
    assert disc[0]["decision"] == 300 + 1500                # frozen at 300, every decision sampled (ρ = 1)
    assert loop.refreeze_floor == 1.5 * 1800
    assert loop.candidate is None and loop.budget.v == 1   # its δ_v stays spent; the next freeze takes v = 2
    v1 = loop.versions()[1]
    assert v1["state"] == "discarded" and not v1["certified"]


def _served_loop(models, **kw):
    """Version 0 is a model that answers alone on 90 %: once it certifies the coin samples about 1 decision in 18,
    so the decisions since a freeze run far ahead of the candidate's samples."""
    return LearningLoop(_Fake(models), question="q", max_risk=0.05, secret=KEY, initial=_Model("s1", 0.9),
                        freeze_min=320, growth=1.0, **kw)


def test_a_certified_candidate_never_promotable_is_discarded_after_trial_decisions_and_training_resumes():
    """Decision 14: `trial` counts decisions since the freeze, not samples. Version 1 (right on 85 %) certifies but
    never answers alone more often than version 0 (90 %); with trial=3000 it is discarded exactly 3 000 decisions
    after its freeze with ~200 samples (counted in samples it would have stayed in trial to the end and blocked every
    later training), and the next training freezes version 2."""
    loop = _served_loop([_Model(f"c{i}", 0.85) for i in range(6)], stale=10 ** 6, trial=3000)
    _run_fake(loop, _fake_data(21, 9000))
    fr, disc = _events(loop, "freeze"), _events(loop, "discard")
    assert fr[0]["version"] == 1 and disc and disc[0]["version"] == 1
    assert disc[0]["why"].startswith("certified, not better 3000 decisions after its freeze")
    assert disc[0]["decision"] == fr[0]["frozen_at"] + 3000
    assert disc[0]["samples"] < 3000 // 5                  # far fewer samples than decisions: ρ well below 1
    assert not _events(loop, "promote") and loop.served == 0
    assert len(fr) >= 2 and fr[1]["version"] == 2 and fr[1]["decision"] > disc[0]["decision"]   # training resumed
    assert loop.replay()["ok"]


def test_an_uncertified_candidate_is_discarded_after_stale_decisions_not_samples():
    loop = _served_loop([_Model(f"z{i}", 0.0) for i in range(6)], stale=1500, trial=10 ** 6)
    _run_fake(loop, _fake_data(21, 2500))
    fr, disc = _events(loop, "freeze"), _events(loop, "discard")
    assert disc and disc[0]["version"] == 1 and disc[0]["why"].startswith("not certified 1500 decisions after its freeze")
    assert disc[0]["decision"] == fr[0]["frozen_at"] + 1500 and disc[0]["samples"] < 1500 // 5
    assert loop.replay()["ok"]


def test_version_zero_never_certifies_and_spends_no_delta():
    loop = LearningLoop(_Fake([]), question="q", max_risk=0.05, secret=KEY, freeze_min=10 ** 6)
    outs = _run_fake(loop, _fake_data(3, 500))
    assert all(o["by"] == "oracle" and o["served"] == 0 for o in outs)
    v0 = loop.versions()[0]
    assert v0["state"] == "served" and not v0["certified"] and v0["delta_v"] is None
    assert loop.budget.v == 0 and loop.delta_spent == 0.0


def test_delta_spent_over_fifty_versions_is_at_most_delta():
    models = [_Model(f"w{i}", 0.0) for i in range(100)]
    loop = LearningLoop(_Fake(models), question="q", max_risk=0.05, delta=0.05, secret=KEY, freeze_min=1, stale=5,
                        refreeze=1.0)
    _run_fake(loop, _fake_data(4, 300))
    vs = [v for v in loop.versions() if v["version"] >= 1]
    assert len(vs) >= 50
    for v in vs:
        assert v["delta_v"] == pytest.approx(6 * 0.05 / (math.pi ** 2 * v["v_index"] ** 2), rel=1e-12)
    assert loop.delta_spent <= 0.05
    assert len({v["v_index"] for v in vs}) == len(vs)      # one index per version


def test_slow_path_labels_are_refused_without_the_opt_in_and_recorded_as_s2_with_it():
    data = _stream(5, 20)
    loop = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY)
    for src in ("s2", "verified", "model"):
        with pytest.raises(UntrustedLabel):
            loop.step(data[0][0], lambda: data[0][1], source=src)
    assert loop.n == 0 and loop.live.seen() == 0 and not loop.log    # refused before anything changed
    loop.step(data[0][0], lambda: data[0][1], source="human")
    assert loop.log[-1][1]["stream"]["label_source"] == "human"

    loop = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, s2_labels=True)
    for (x, y), src in zip(data[:3], ("s2", "verified", "outcome")):
        loop.step(x, lambda y=y: y, source=src)
    assert [d["stream"]["label_source"] for k, d in loop.log if k == "loop"] == ["s2", "s2", "outcome"]
    assert loop.live.seen() == 3
    assert loop.replay()["ok"]


def test_replay_rebuilds_every_version_and_decision_from_the_chain(tmp_path):
    store = JSONLStorage(tmp_path / "loop.jsonl")
    vd = tmp_path / "versions"
    loop = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, storage=store, versions_dir=vd)
    data = _stream(6, 1400)
    _run(loop, data[:900])
    assert loop.served == 1
    thr1 = loop.versions()[1]["threshold"]
    loop.rollback(0, by="ops", why="test")                 # a person's call, in the chain
    assert loop.served == 0 and loop.candidate is None
    _run(loop, data[900:1000])
    loop.rollback(1, by="ops", why="back")                 # served again with the threshold it left with
    assert loop.served == 1 and loop.versions()[1]["threshold"] == thr1
    _run(loop, data[1000:])
    assert store.verify()["ok"]
    r = loop.replay(rescore=True)
    assert r["ok"], r["mismatches"][:5]
    assert r["decisions"] == 1400 and r["versions"] >= 2
    served = [d["stream"]["served"] for k, d in loop._items() if k == "loop"]
    assert served[899] == 1 and served[950] == 0 and served[1100] == 1

    # another key: the coin is not the recorded one
    other = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=b"other", storage=store,
                         versions_dir=vd)
    assert not other.replay()["ok"]
    # a version file replaced: its fingerprint is not the recorded one
    with open(vd / "1.pkl", "wb") as f:
        pickle.dump(HashedTextLearner(["a"]).snapshot(), f)
    bad = loop.replay()
    assert not bad["ok"] and any(m[1] == "version file" for m in bad["mismatches"])


@pytest.mark.parametrize("kind", ["max_risk", "max_error"])
def test_c3_shuffled_labels_never_promote_under_either_promise(kind):
    """The shuffled-label control, both promise kinds, 3 seeds. Under the joint promise a model of noise can certify at a
    strict threshold that almost nothing passes; the promotion floor (ℓ_k ≥ min_share = 0.05, and ℓ_k ≥ ℓ_s + 0.01)
    keeps it out (author's rule, 2026-10-23)."""
    for seed in range(3):
        loop = LearningLoop(HashedTextLearner(), question="q", secret=KEY, seed=seed, **{kind: 0.05})
        data = _stream(seed, 2000, shuffle=True)
        outs = _run(loop, data)
        assert not _events(loop, "promote"), (kind, seed)
        assert loop.served == 0 and all(o["by"] != "local" for o in outs)
        assert _events(loop, "freeze")                     # candidates were frozen and tried


def test_min_share_and_margin_are_the_promotion_bar():
    """A certified candidate that answers alone on ~4 % (< min_share 0.05) is never promoted; with min_share=0 and
    margin=0 the same stream promotes it (the 3b rule)."""
    def run(**kw):
        models = [_Model(f"thin{i}", 0.04) for i in range(10)]
        loop = LearningLoop(_Fake(models), question="q", max_risk=0.05, secret=KEY, growth=1.0, trial=600, **kw)
        _run_fake(loop, _fake_data(11, 3000))
        return loop
    strict = run()
    assert strict.min_share == 0.05 and strict.margin == 0.01
    assert not _events(strict, "promote") and strict.served == 0
    certified = [e for e in _events(strict, "discard") if e["why"].startswith("certified, not better")]
    assert certified                                       # it did certify: only the floor kept it out
    loose = run(min_share=0.0, margin=0.0)
    assert _events(loose, "promote")
    with pytest.raises(ValueError):
        LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, min_share=1.5)


def test_end_to_end_the_local_share_grows_and_the_alone_error_stays_within_the_promise():
    data = _stream(7, 2000)
    loop = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY)
    outs = _run(loop, data)
    local = [o["by"] == "local" for o in outs]
    assert sum(local[:300]) == 0                           # version 0 answers nothing alone
    assert sum(local[-500:]) / 500 > 0.5                   # a promoted version answers most alone
    wrong = sum(o["by"] == "local" and o["answer"] != y for o, (_, y) in zip(outs, data))
    assert wrong / max(1, sum(local)) <= 0.05
    assert loop.replay()["ok"]


def test_the_served_version_is_the_systems_answer_head():
    cat = Catalog()
    system = System(cat, [Question("intent", "Which intent?", Answer.choice(list("abcde")))])
    loop = LearningLoop(HashedTextLearner(list("abcde")), question="intent", max_risk=0.05, secret=KEY,
                        system=system, feature="text")
    fp0 = system.fingerprint()["models"].get("answer:intent")
    data = _stream(8, 900)
    _run(loop, data)
    assert loop.served >= 1
    head = system.heads["intent"]
    assert head.fingerprint() == loop.versions()[loop.served]["fingerprint"] != fp0
    assert system.fingerprint()["models"]["answer:intent"] == head.fingerprint()
    x, y = data[-1]
    r = system.ask({"text": x})["intent"]
    assert r.answer == loop._vs[loop.served]["model"].score(x)[0]
    with pytest.raises(FrozenError):
        head.teach({"text": x}, y)
    loop.rollback(0)
    assert "intent" not in system.heads

    ruled = Catalog()

    @ruled.rule("intent")
    def intent(text: str):
        return "a"
    with pytest.raises(ValueError):
        serve_head(System(ruled, [Question("intent", "?", Answer.choice(list("abcde")))]), "intent", head)


def test_a_fast_head_learner_over_facts_is_frozen_certified_and_promoted():
    r = random.Random(9)
    data = []
    for _ in range(1500):
        f1, f2 = r.uniform(-1, 1), r.uniform(-1, 1)
        data.append(({"f1": f1, "f2": f2}, "hi" if f1 + 0.5 * f2 > 0 else "lo"))
    learner = FastHeadLearner(["hi", "lo"], ["f1", "f2"], min_fit=20)
    loop = LearningLoop(learner, question="q", max_risk=0.05, secret=KEY)
    outs = _run(loop, data)
    assert _events(loop, "promote") and loop.served >= 1
    wrong = sum(o["by"] == "local" and o["answer"] != y for o, (_, y) in zip(outs, data))
    assert wrong / len(data) <= 0.05
    assert loop.replay()["ok"]


# ───────────────────────────── version 0 = an existing model, the detector on the served version, tickets
def test_version_zero_can_be_the_fitted_system_one_with_its_own_promise():
    s1 = _Model("s1", 0.5)
    loop = LearningLoop(_Fake([_Model("good", 1.0)] + [_Model(f"w{i}", 0.3) for i in range(5)]), question="q",
                        max_risk=0.05, secret=KEY, initial=s1,
                        growth=1.0)
    outs = _run_fake(loop, _fake_data(12, 2500))
    ev = _events(loop)
    assert ev[0]["event"] == "start" and ev[0]["version"] == 0 and ev[0]["fingerprint"] == s1.fingerprint()
    v0 = loop.versions()[0]
    assert v0["v_index"] == 1 and v0["certified"]          # its own certificate, the first δ_v of the budget
    assert loop.fed(0) and min(loop.fed(0)) == 1           # fed from decision 1
    first_local = next(i for i, o in enumerate(outs) if o["by"] == "local")
    assert first_local < 600 and outs[first_local]["served"] == 0   # it answers alone before any freeze
    promo = _events(loop, "promote")
    assert promo and promo[0]["from"] == 0 and promo[0]["l_s"] > 0.3 and promo[0]["l_k"] >= promo[0]["l_s"] + 0.01
    assert loop.served == 1 and loop.versions()[1]["v_index"] == 2
    assert loop.delta_spent <= 0.05
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]


def test_frozen_fast_of_system_starts_the_loop_from_the_fitted_head():
    from solvi.experimental.stream import FrozenFast
    r = random.Random(13)
    rows = []
    for _ in range(1200):
        f1, f2 = r.uniform(-1, 1), r.uniform(-1, 1)
        rows.append(({"f1": f1, "f2": f2}, "hi" if f1 + 0.5 * f2 > 0 else "lo"))
    system = System(Catalog(), [Question("q", "hi or lo?", Answer.choice(["hi", "lo"]))])
    system.fit("q", rows[:200], ["f1", "f2"], select=False)
    v0 = FrozenFast.of_system(system, "q")
    head0 = system.heads["q"]
    assert v0.head is not head0                            # a copy: teaching the System never moves version 0
    loop = LearningLoop(FastHeadLearner(["hi", "lo"], ["f1", "f2"]), question="q", max_risk=0.05, secret=KEY,
                        initial=v0, system=system, freeze_min=10 ** 6)
    outs = _run(loop, rows[200:])
    assert loop.versions()[0]["certified"] and any(o["by"] == "local" for o in outs)
    assert system.heads["q"] is head0
    with pytest.raises(ValueError):
        FrozenFast.of_system(System(Catalog(), [Question("z", "?", Answer.choice(["a", "b"]))]), "z")


def test_an_alarm_restarts_the_served_promise_and_opens_probation():
    from solvi.experimental.stream import RiskDetector
    det = RiskDetector(0.05, arl=100)
    loop = LearningLoop(_Fake([]), question="q", max_risk=0.05, secret=KEY, initial=_Model("s1", 0.9),
                        freeze_min=10 ** 6, detector=det, probation=300)
    data = _fake_data(14, 3000)
    outs = []
    for i, (x, _) in enumerate(data):                      # from decision 1201 the world moves: the label changes
        outs.append(loop.step(x, (lambda x=x: x[0]) if i < 1200 else (lambda: "moved")))
    alarms = _events(loop, "alarm")
    assert alarms and alarms[0]["version"] == 0 and alarms[0]["decision"] > 1200 and alarms[0]["v_index"] == 1
    a = alarms[0]["decision"]
    assert not any(o["by"] == "local" for o in outs[a:a + 300])   # probation: nothing alone for 300 decisions
    rec = [d["stream"] for k, d in loop._items() if k == "loop"]
    assert rec[a]["probation"] and rec[a]["left"] == 299 and rec[a]["rho"] == 1.0
    assert not rec[a - 1]["probation"]
    rest = [e for e in _events(loop, "recertify") if e["decision"] == a]
    assert rest and rest[0]["v_index"] == 2 and rest[0]["threshold"] is None
    assert loop.delta_spent <= 0.05
    assert sum(o["by"] == "local" and o["answer"] != "moved" for o in outs[1200:]) < 100
    r = loop.replay()
    assert r["ok"], r["mismatches"][:5]
    with pytest.raises(ValueError):
        LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, detector=RiskDetector(0.1))


class _Knows(_Model):
    """x = (label, u, moved): answers "moved" on a moved input, else the label — right everywhere, score 0.95."""

    def score(self, x):
        return ("moved" if x[2] else x[0]), 0.95


def test_a_promoted_candidate_waits_out_probation():
    from solvi.experimental.stream import RiskDetector
    loop = LearningLoop(_Fake([_Knows("knows", 1.0)] + [_Model(f"w{i}", 0.3) for i in range(5)]), question="q", max_risk=0.05, secret=KEY,
                        initial=_Model("s1", 0.9), freeze_min=320, detector=RiskDetector(0.05, arl=50), probation=300,
                        promise_options={"grid": [0.9, 0.5]})
    r = random.Random(15)
    outs = []
    for i in range(3000):                                  # from decision 1001 the world moves; version 0 does not
        x = (r.choice("ab"), r.random(), i >= 1000)
        outs.append(loop.step(x, lambda x=x: "moved" if x[2] else x[0]))
    alarm, promo = _events(loop, "alarm"), _events(loop, "promote")
    assert alarm and promo
    a, p = alarm[0]["decision"], promo[0]["decision"]
    assert a < p < a + 300                                 # certified and promoted during probation ...
    assert promo[0]["l_s"] == 0.0                          # (the served promise restarted: +inf)
    assert not any(o["by"] == "local" for o in outs[a:a + 300])   # ... but nothing alone until it ends
    assert any(o["by"] == "local" and o["served"] == 1 for o in outs[a + 300:])
    assert loop.replay()["ok"]


def _ticket_desk(store):
    from solvi import Decision
    from solvi.core.dispatch import Dispatcher
    from solvi.experimental.stream import RiskDetector, StreamPromise
    from solvi.experimental.stream.tickets import AgentOracle, TicketDesk

    cat = Catalog()

    @cat.rule("intent")
    def intent(case: int):
        return Decision("a", {"a": 0.6, "b": 0.4}, confidence=0.6)
    s1 = System(cat, [Question("intent", "a or b?", Answer.choice(["a", "b"]))])
    sp = StreamPromise(max_risk=0.05, grid=[0.9, 0.5])
    s1.guarantee("intent", promise=sp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = Dispatcher(s1, AgentOracle(s1), audit="adaptive", promise=sp, detector=RiskDetector(0.05),
                       on_drift="recertify", secret="k", storage=store)
    return TicketDesk(d, ttl=600, clock=lambda: 1000.0)


def test_a_resolved_tickets_verdict_is_a_label_of_the_loop(tmp_path):
    from solvi.experimental.stream import Verdict, ticket_verdict
    desk = _ticket_desk(JSONLStorage(tmp_path / "desk.jsonl"))
    t = desk.decide({"case": 1})
    assert t["status"] == "needs_verdict"
    res = desk.resolve(t["ticket"], "b")
    v = ticket_verdict(desk, res)
    pending_id = next(s.id for s in desk.dispatcher.storage.iter(kind="pending"))
    assert v == Verdict("b", "s2", pending_id)

    loop = LearningLoop(HashedTextLearner(["a", "b"]), question="intent", max_risk=0.05, secret=KEY)
    with pytest.raises(UntrustedLabel):                    # the agent's verdict is a slow-path label
        loop.step("case one", lambda: v)
    assert loop.n == 0
    loop = LearningLoop(HashedTextLearner(["a", "b"]), question="intent", max_risk=0.05, secret=KEY, s2_labels=True)
    out = loop.step("case one", lambda: v)
    assert out["label"] == "b" and out["label_source"] == "s2"
    rec = loop.log[0][1]["stream"]
    assert rec["label_of"] == pending_id and rec["label_source"] == "s2"
    assert loop.replay()["ok"]

    t2 = desk.decide({"case": 2})                          # a ticket that expires: no label, a person decides
    gone = desk.expire(t2["ticket"])
    lost = ticket_verdict(desk.dispatcher.storage, gone.stored_id)
    assert lost.label is None and lost.label_of is None
    out = loop.step("case two", lambda: lost)
    assert out["by"] == "person" and loop.log[-1][1]["stream"]["missing"]
    with pytest.raises(ValueError):
        ticket_verdict(desk, desk.dispatcher.storage.iter(kind="pending").__iter__().__next__().id)


def _shift_loop(stale=10 ** 6):
    from solvi.experimental.stream import RiskDetector
    return LearningLoop(_Fake([_Model("c", 0.85)] + [_Model(f"w{i}", 0.0) for i in range(5)]), question="q", max_risk=0.05, secret=KEY, initial=_Model("s1", 0.9),
                        freeze_min=320, stale=stale, trial=10 ** 6, detector=RiskDetector(0.05, arl=50),
                        probation=300, promise_options={"grid": [0.9, 0.5]})


def _shift_run(loop, n=7000, at=5000):
    """Before decision `at` the candidate is good but worse than version 0; from there the label moves and neither
    is ever right again."""
    r = random.Random(16)
    outs = []
    for i in range(n):
        x = (r.choice("ab"), r.random())
        outs.append(loop.step(x, (lambda x=x: x[0]) if i < at else (lambda: "moved")))
    return outs


def test_an_alarm_restarts_every_candidates_certificate():
    """A candidate certified before the shift is not better than the served version, so it waits in trial; the alarm
    restarts the served promise (+inf, ℓ_s = 0). Before the fix the candidate was promoted at once on its pre-shift
    threshold and served the moved world; now its promise restarts too and it is never promoted on old evidence."""
    loop = _shift_loop()
    outs = _shift_run(loop)
    alarm = _events(loop, "alarm")
    assert alarm and alarm[0]["decision"] > 5000
    a = alarm[0]["decision"]
    frz = _events(loop, "freeze")
    assert frz and frz[0]["decision"] < 5000
    cert = [e for e in _events(loop, "certified") if e["version"] == 1]
    assert cert and cert[0]["decision"] < a                # certified on the old regime
    assert not _events(loop, "promote")                    # pre-fix: promoted at decision a
    rs = [e for e in _events(loop, "restart") if e["decision"] == a]
    assert {e["version"] for e in rs} == {0, 1} and all(e["why"] == "alarm" and e["threshold"] is None for e in rs)
    assert loop.versions()[1]["threshold"] is None
    assert not any(o["by"] == "local" for o in outs[a:])
    assert len(loop._vs[1]["samples"]) <= len(outs) - a   # the paired samples: only those after the alarm
    assert loop.delta_spent <= 0.05
    assert loop.replay()["ok"]


def test_a_rollback_target_certified_before_an_alarm_starts_over():
    """A retired version's certificate predates the alarm: rolled back after it, it serves nothing alone until it is
    certified again on post-alarm samples."""
    from solvi.experimental.stream import RiskDetector
    loop = LearningLoop(_Fake([_Model("c", 1.0)] + [_Model(f"w{i}", 0.0) for i in range(20)]), question="q", max_risk=0.05, secret=KEY, initial=_Model("s1", 0.6),
                        freeze_min=320, detector=RiskDetector(0.05, arl=50), probation=0,
                        promise_options={"grid": [0.9, 0.5]})
    r = random.Random(17)
    data = [(r.choice("ab"), r.random()) for _ in range(4000)]
    for x in data[:2000]:
        loop.step(x, lambda x=x: x[0])
    assert loop.served == 1 and loop.versions()[0]["certified"]
    for x in data[2000:]:
        loop.step(x, lambda: "moved")
    alarm = _events(loop, "alarm")
    assert alarm and alarm[0]["version"] == 1
    assert loop.versions()[0]["certified"]                 # retired: untouched until it is a rollback target
    ev = loop.rollback(0, by="ops", why="test")
    assert ev["event"] == "rollback"
    assert loop.versions()[0]["threshold"] is None
    rs = [e for e in _events(loop, "restart") if e["version"] == 0]
    assert rs and rs[-1]["why"] == "alarm" and rs[-1]["decision"] == ev["decision"]
    assert loop.step(data[0], lambda: "a")["by"] != "local"
    assert loop.delta_spent <= 0.05
    assert loop.replay()["ok"]
