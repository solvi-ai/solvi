"""Public loop API contracts; tiny online learner, no external environment/model."""
import copy
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, LearningLoop


class MutableLearner:
    def __init__(self):
        self.labels = []

    def update(self, x, y, **kwargs):
        self.labels.append(y)

    def seen(self):
        return len(self.labels)

    def snapshot(self):
        return copy.deepcopy(self)

    def score(self, x):
        return (self.labels[-1], .9)

    def fingerprint(self):
        return "mutable-" + ",".join(self.labels)


def make_loop(tmp_path):
    live = MutableLearner()
    loop = LearningLoop(live, question="q", max_risk=.1, secret=b"loop-api",
                        freeze_min=1000, additive=False, versions_dir=tmp_path)
    return live, loop


def test_learning_loop_inherits_train_now_and_freezes_an_independent_candidate(tmp_path):
    live, loop = make_loop(tmp_path)
    assert LearningLoop.train_now is AdapterLoop.train_now
    loop.step("first", lambda: "a")
    assert loop.candidate is None
    job = loop.train_now(wait=True)
    assert job.state == "done" and job.frozen and job.trigger == "train_now"
    frozen = loop._vs[loop.candidate]["model"]
    fp, prediction = frozen.fingerprint(), frozen.score("probe")
    loop.step("second", lambda: "b")
    assert live.score("probe") == ("b", .9)
    assert frozen is not live and frozen.labels == ["a"]
    assert frozen.fingerprint() == fp and frozen.score("probe") == prediction == ("a", .9)
    events = [e for kind, e in loop._items() if kind == "stream"]
    assert any(e["event"] == "train_start" and e["trigger"] == "train_now" for e in events)
    assert any(e["event"] == "freeze" and e["fingerprint"] == fp for e in events)
    replay = loop.replay(rescore=True)
    assert replay["ok"], replay["mismatches"]


def test_replay_all_is_full_protocol_replay_and_forwards_rescore(tmp_path, monkeypatch):
    _, loop = make_loop(tmp_path)
    loop.step("first", lambda: "a")
    loop.train_now(wait=True)
    loop.step("second", lambda: "b")
    expected = loop.replay(rescore=True)
    assert expected["ok"]
    assert loop.replay_all(rescore=True) == expected
    calls = []
    def replay(*, rescore=False):
        calls.append(rescore)
        return {"checked": rescore}
    monkeypatch.setattr(loop, "replay", replay)
    assert loop.replay_all() == {"checked": False}
    assert loop.replay_all(rescore=True) == {"checked": True}
    assert calls == [False, True]
