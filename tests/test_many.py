"""solvi.many: a choice among more options than one pass reads — direct, shortlist, tournament, auto; what was considered
and every model call are in the decision."""
import numpy as np
import pytest

from solvi.decide import DecideModel
from solvi.many import Many, bm25_scores, decide_many, fit


class Overlap:
    """Scores an option by the words it shares with the input; refuses more than `limit` options (a pass that is full)."""
    model_id = "test/overlap"

    def __init__(self, limit=12):
        self.limit, self.calls = limit, []

    def fingerprint(self):
        return f"overlap-{self.limit}"

    def logits(self, items):
        out = []
        for it in items:
            if len(it.options) > self.limit:
                raise ValueError(f"task and options do not fit: {len(it.options)} options")
            self.calls.append(list(it.options))
            words = set(it.text.lower().split())
            z = np.array([3.0 * len(words & set(str(o).lower().split())) for o in it.options])
            out.append(np.stack([z, z - 1.0], 1))
        return out


ACTIONS = [f"open the {c} door" for c in ("red", "blue", "green", "grey", "white", "black", "brown", "pink")] + \
          [f"take the {t}" for t in ("lamp", "rope", "key", "map", "coin", "book", "cup", "hat")] + \
          [f"talk to the {p}" for p in ("guard", "cook", "smith", "monk", "child", "king", "thief", "bard")]
TEXT = "Goal: leave by the green door. Room: a guard, a lamp, eight doors."
TASK = "Which action achieves the goal?"


def model(limit=12):
    return DecideModel(Overlap(limit), meta={"format": "test", "temperature": 1.0})


def test_direct_when_the_options_fit_and_an_error_that_names_the_other_modes():
    m = model(limit=100)
    d = decide_many(m, TEXT, TASK, ACTIONS, many=Many(mode="direct"))
    assert d.value == "open the green door"
    assert d.extra["many"]["mode"] == "direct" and d.extra["many"]["considered"] == 24 and d.extra["many"]["unconsidered"] == 0
    assert len(d.extra["many"]["calls"]) == 1
    with pytest.raises(ValueError, match="shortlist"):
        decide_many(model(limit=12), TEXT, TASK, ACTIONS, many=Many(mode="direct"))


def test_shortlist_records_what_was_left_out_and_escalates_on_a_close_cut():
    m = model()
    d = decide_many(m, TEXT, TASK, ACTIONS, many=Many(mode="shortlist", k=4, query="green door"))
    info = d.extra["many"]
    assert d.value == "open the green door" and info["mode"] == "shortlist" and info["selector"] == "bm25"
    assert info["considered"] == 4 and info["unconsidered"] == 20 and info["of"] == 24
    assert info["shortlist"][0][0] == "open the green door" and m.scorer.calls == [[o for o, _ in info["shortlist"]]]
    assert "left out 20 option(s)" in d.escalate                  # seven more doors match "door" as well as the 4th kept
    sure = decide_many(model(), TEXT, TASK, ACTIONS, many=Many(mode="shortlist", k=8, query="green door"))
    assert sure.escalate is None and sure.extra["many"]["gap"] is None      # nothing that matched the query was left out
    by_fn = decide_many(model(), TEXT, TASK, ACTIONS,
                        many=Many(mode="shortlist", k=3, selector=lambda q, labels, texts: [float("green" in t) for t in texts]))
    assert by_fn.value == "open the green door" and by_fn.extra["many"]["selector"] == "callable"
    assert bm25_scores("green door", ACTIONS)[2] == max(bm25_scores("green door", ACTIONS))


def test_tournament_considers_every_option_in_blocks():
    m = model()
    d = decide_many(m, TEXT, TASK, ACTIONS, many=Many(mode="tournament", block=10))
    info = d.extra["many"]
    assert d.value == "open the green door" and info["considered"] == 24 and info["unconsidered"] == 0
    assert info["rounds"] == 2 and [len(b["options"]) for b in info["blocks"]] == [10, 10, 4, 3]
    assert len(info["calls"]) == 4 and all(len(c) <= 10 for c in m.scorer.calls)
    assert info["blocks"][-1]["winner"] == d.value and set(d.probs) == set(info["blocks"][-1]["options"])


def test_auto_goes_direct_then_shortlist_then_tournament_and_is_deterministic():
    assert decide_many(model(limit=100), TEXT, TASK, ACTIONS).extra["many"]["mode"] == "direct"
    tight = model(limit=100)
    tight.max_len_override = None
    d = decide_many(tight, TEXT, TASK, ACTIONS, many=Many(min_input=10_000))          # the input would be left too little
    assert d.extra["many"]["mode"] == "shortlist" and d.extra["many"]["fit"]["fits"] is False
    d = decide_many(model(), TEXT, TASK, ACTIONS, many=Many(min_input=10_000, selector=None))
    assert d.extra["many"]["mode"] == "tournament"
    d = decide_many(model(), TEXT, TASK, ACTIONS)                                    # direct does not fit after all
    assert d.extra["many"]["mode"] == "shortlist" and d.extra["many"]["requested"] == "auto"
    again = decide_many(model(), TEXT, TASK, ACTIONS)
    assert again.extra["many"] == d.extra["many"] and again.value == d.value
    f = fit(model(), TEXT, TASK, ACTIONS)
    assert f["question_tokens"] + f["input_budget"] == f["max_len"]


def test_options_as_a_dict_and_bad_arguments():
    opts = {"a1": "open the green door", "a2": "take the lamp", "a3": "talk to the guard"}
    d = decide_many(model(), TEXT, TASK, opts, many=Many(mode="shortlist", k=2, query="green door"))
    assert d.extra["many"]["shortlist"][0][0] == "a1" and set(d.probs) <= set(opts)
    for bad in (dict(mode="score"), dict(k=1), dict(selector="tfidf"), dict(mode="shortlist", selector=None)):
        with pytest.raises(ValueError):
            Many(**bad)
    with pytest.raises(ValueError, match="at least two"):
        decide_many(model(), TEXT, TASK, ["only"])
    with pytest.raises(ValueError, match="max_options"):
        decide_many(model(), TEXT, TASK, ACTIONS, many=Many(max_options=10))
    with pytest.raises(ValueError, match="duplicate"):
        decide_many(model(), TEXT, TASK, ["a", "a", "b"])


def test_a_close_cut_escalates_whatever_the_sign_of_a_selectors_scores():
    """With a selector whose scores are not positive (cosines, log-probabilities, negative distances) the close-cut
    escalation never fired: an exact tie at the cut left `gap` None."""
    m = DecideModel(Overlap(limit=12), meta={"format": "test", "temperature": 1.0})
    for sel in (lambda q, labels, texts: [-1.0 if "door" in t else -5.0 for t in texts],
                lambda q, labels, texts: [-0.1 if "door" in t else -0.9 for t in texts],
                lambda q, labels, texts: [1.0 if "door" in t else 0.2 for t in texts],
                lambda q, labels, texts: [0.0 for t in texts]):
        d = decide_many(m, TEXT, TASK, ACTIONS, many=Many(mode="shortlist", k=4, query="door", selector=sel))
        assert d.extra["many"]["gap"] == 0.0 and d.escalate and "left out" in d.escalate
    clear = decide_many(m, TEXT, TASK, ACTIONS, many=Many(mode="shortlist", k=8, query="door", selector=lambda q, labels, texts: [
        -1.0 if "door" in t else -5.0 for t in texts]))
    assert clear.extra["many"]["gap"] is None and clear.escalate is None       # what is left out is the worst there is


def test_a_tournament_block_that_escalates_makes_the_final_decision_escalate():
    m = DecideModel(Overlap(limit=12), meta={"format": "test", "temperature": 1.0})
    d = decide_many(m, "Goal: get out. Room: nothing here matches.", TASK, ACTIONS, many=Many(mode="tournament", block=8),
                    escalate_below=0.5)
    calls = d.extra["many"]["calls"]
    assert any(c["escalate"] for c in calls[:-1]) and d.escalate and "tournament call(s) before the last escalated" in d.escalate
    sure = decide_many(m, TEXT, TASK, ACTIONS, many=Many(mode="tournament", block=8))
    assert sure.escalate is None
