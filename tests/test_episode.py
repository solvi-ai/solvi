"""solvi.episode: an agent's memory as a given fact of its decisions — counts since the last progress, loop detectors,
a chooser whose stored decisions replay, and outcomes kept across episodes."""
import json

import numpy as np
import pytest

from solvi import Catalog, Question, SQLiteStorage, System
from solvi.decide import DecideModel
from solvi.episode import Chooser, Episode, EpisodeView, LongMemory


def test_events_counts_and_progress():
    ep = Episode("ticket")
    ep.note("act", "restart").note("act", "restart").note("act", {"tool": "ping", "host": "a"}).set("customer", "c17")
    v = EpisodeView(ep.snapshot())
    assert v.count("act") == 3 and v.count("act", "restart") == 2 and v.seen("act", {"host": "a", "tool": "ping"})
    assert v.repeated("act", "restart") and not v.repeated("act", "restart", limit=3) and v.get("customer") == "c17"
    assert v.last("act", 2)[-1][1:] == ["act", '{"host": "a", "tool": "ping"}'] and v.since_progress == 3
    ep.progress("the customer confirmed")
    v = EpisodeView(ep.snapshot())
    assert v.count("act", "restart") == 0 and v.count("act", "restart", since="total") == 2 and v.since_progress == 0
    assert ep.progress_log == [[3, "the customer confirmed"]]
    assert ep.snapshot() == json.loads(json.dumps(ep.snapshot())) and len(ep.digest()) == 16     # plain data
    same = Episode("ticket")
    same.note("act", "restart").note("act", "restart").note("act", {"host": "a", "tool": "ping"}).set("customer", "c17")
    same.progress("the customer confirmed")
    assert same.digest() == ep.digest()
    assert EpisodeView(None).count("act") == 0 and EpisodeView({}).ping_pong() == set()


def test_loop_detectors():
    ep = Episode()
    for place in ["2F", "3F", "2F", "2F", "3F", "2F", "3F"]:
        ep.note("place", place)
    assert EpisodeView(ep.snapshot()).ping_pong(round_trips=2) == {"2F", "3F"}
    assert EpisodeView(ep.snapshot()).ping_pong(round_trips=3) == set()                 # five stays: two round trips
    ep.note("place", "2F")
    ep.note("place", "3F")
    v = EpisodeView(ep.snapshot())
    assert v.ping_pong() == {"2F", "3F"} and v.stalled(9) and not v.stalled(10) and v.revisits("place", "2F") == 5
    assert v.looping(stalled=9) and not v.looping(stalled=50)                           # stalled AND a ping-pong
    walk = Episode()
    for i in range(12):
        walk.note("place", f"room {i}").note("act", "walk")
    w = EpisodeView(walk.snapshot())
    assert not w.ping_pong() and w.looping(stalled=20, repeat=("act", None, 8)) and not w.looping(stalled=20, repeat=("act", "talk", 8))
    ep.progress("a new floor")
    assert EpisodeView(ep.snapshot()).ping_pong() == set() and not EpisodeView(ep.snapshot()).looping(stalled=1)


class Stubborn:
    """Always proposes "restart" when it is an option (a model that would go in circles)."""
    model_id = "test/stubborn"

    def fingerprint(self):
        return "stubborn-1"

    def logits(self, items):
        return [np.stack([np.array([5.0 if o == "restart" else 0.0 for o in it.options])] * 2, 1) for it in items]


ACTIONS = {"restart": "restart the router", "cable": "check the cable", "replace": "send a new router"}


def test_the_chooser_turns_down_what_was_done_without_progress_and_replays(tmp_path):
    m = DecideModel(Stubborn(), meta={"format": "test", "temperature": 1.0})
    store = SQLiteStorage(tmp_path / "steps.db")
    ch = Chooser(m, storage=store, escalate_below=0.0)
    ep = Episode("ticket")
    done = []
    for rule in ("restart", "cable", "replace"):
        action, who, info = ch.choose("next", "What should support do next?", ACTIONS, context="no internet", rule=rule,
                                      episode=ep)
        done.append((action, who))
        ep.note("act", action)
    assert done == [("restart the router", "model"), ("check the cable", "rule"), ("send a new router", "rule")]
    assert info["tried"][0] == ["pick_by_model", "rejected by validate"] and ch.asked == 3
    assert ch.replay() == (3, 3, []) and store.verify()["ok"]
    ep.progress("the customer rebooted the modem")                    # progress: the first action may be tried again
    assert ch.choose("next", "What should support do next?", ACTIONS, rule="cable", episode=ep)[:2] == ("restart the router", "model")
    strict = Chooser(m, escalate_below=0.0, check=lambda v, q, episode: v != "restart")     # your own check
    assert strict.choose("next", "?", ACTIONS, rule="cable", episode=Episode())[:2] == ("check the cable", "rule")
    assert ch.choose("only", "?", {"wait": "wait"}) == ("wait", "only", {"choice": "wait"}) and strict.replay() is None
    with pytest.raises(ValueError):
        ch.choose("none", "?", {})
    cat = Catalog()                                                   # another system's records in the store are skipped

    @cat.rule("ok")
    def ok(x) -> bool:
        return x > 0
    System(cat, [Question("ok", "Ok?")], storage=store).ask({"x": 1})
    assert ch.replay()[:2] == (4, 4)


def test_long_memory_scores_decay_and_persist(tmp_path):
    lm = LongMemory(tmp_path / "m" / "memory.json", decay=0.5)
    lm.begin("t1").record(("customer", "c17"), "restart", +1, why="solved").record(("customer", "c17"), "cable", -1)
    assert lm.scores(("customer", "c17")) == {"cable": -1.0, "restart": 1.0} and lm.scores("nobody") == {}
    lm.begin("t2").record(("customer", "c17"), "restart", +1)
    assert lm.scores(("customer", "c17")) == {"cable": -0.5, "restart": 1.5}
    for _ in range(20):
        lm.record("outage", "status page", +1)
    assert lm.scores("outage") == {"status page": 5.0}                 # capped
    lm.save()
    again = LongMemory(tmp_path / "m" / "memory.json", decay=0.5)
    assert again.scores(("customer", "c17")) == lm.scores(("customer", "c17")) and again.stats()["episodes"] == 2
    assert again.data["items"]["customer|c17"]["restart"]["last"] == "2:t2"
    assert again.forget(("customer", "c17")) == 2 and again.scores(("customer", "c17")) == {} and again.forget() == 1
