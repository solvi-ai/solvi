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
    labels permuted over the stream (C3: nothing to learn)."""
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
        y, u = x
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
    assert promos[1]["from"] == 1 and promos[1]["l_k"] >= promos[1]["l_s"] + 0.0 and promos[1]["l_k"] > 0.9
    # version 3 (right on 30 %) certifies but answers alone less often than version 2 on the same samples: discarded
    disc = _events(loop, "discard")
    assert disc and disc[0]["version"] == 3 and disc[0]["why"].startswith("certified, not better")
    assert disc[0]["samples"] == loop.trial == 300
    assert loop.served == 2


def test_an_uncertified_candidate_is_discarded_after_stale_samples_and_its_delta_stays_spent():
    loop = LearningLoop(_Fake([_Model("wrong", 0.0), _Model("wrong2", 0.0)]), question="q", max_risk=0.05,
                        secret=KEY)
    _run_fake(loop, _fake_data(2, 1900))
    disc = _events(loop, "discard")
    assert len(disc) == 1 and disc[0]["version"] == 1 and disc[0]["why"] == "not certified after 1500 samples"
    assert disc[0]["decision"] == 300 + 1500                # frozen at 300, every decision sampled (ρ = 1)
    assert loop.refreeze_floor == 1.5 * 1800
    assert loop.candidate is None and loop.budget.v == 1   # its δ_v stays spent; the next freeze takes v = 2
    v1 = loop.versions()[1]
    assert v1["state"] == "discarded" and not v1["certified"]


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


def test_c3_shuffled_labels_never_promote_under_the_selective_promise():
    for seed in range(3):
        loop = LearningLoop(HashedTextLearner(), question="q", max_error=0.05, secret=KEY, seed=seed)
        outs = _run(loop, _stream(seed, 2000, shuffle=True))
        assert not _events(loop, "promote"), seed
        assert loop.served == 0 and all(o["by"] != "local" for o in outs)
        assert _events(loop, "discard")                    # not certified after 1 500 samples


def test_c3_under_the_joint_promise_a_shuffled_candidate_can_promote_but_answers_almost_nothing_alone():
    """The joint promise (P(alone and wrong) ≤ α) can be certified at a strict threshold that almost nothing passes,
    so a model of noise can be promoted with margin 0 (design §9.3's C3 bar assumed it cannot certify). What it
    answers alone stays within the promise."""
    promoted = []
    for seed in range(2):
        loop = LearningLoop(HashedTextLearner(), question="q", max_risk=0.05, secret=KEY, seed=seed)
        data = _stream(seed, 2000, shuffle=True)
        outs = _run(loop, data)
        promoted.append(bool(_events(loop, "promote")))
        alone_wrong = sum(o["by"] == "local" and o["answer"] != y for o, (_, y) in zip(outs, data))
        assert alone_wrong / len(data) <= 0.05
        assert sum(o["by"] == "local" for o in outs) <= 0.02 * len(data)
    assert any(promoted)                                   # the C3 bar as drafted fails for the joint kind


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
