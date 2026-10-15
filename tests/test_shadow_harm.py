"""Shadow with labels: a paired, anytime-valid harm stop, overall and per slice."""
import random

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.store.diff import Shadow, _PairedHarm


def gate(limit, worse_in=None, worse_limit=None):
    """approve = amount < limit; in region `worse_in` the limit is `worse_limit` instead."""
    cat = Catalog()

    @cat.rule("approve")
    def approve(amount, region) -> bool:
        lim = worse_limit if region == worse_in else limit
        return amount < lim

    return System(cat, [Question("approve", "Approve?", Answer.yes_no())])


def truth_of(st):
    return "yes" if st["amount"] < 100 else "no"


def states(n, seed=0, regions=("north", "south")):
    rng = random.Random(seed)
    return [{"amount": rng.randrange(0, 200), "region": rng.choice(regions)} for _ in range(n)]


def test_a_worse_candidate_is_halted_overall():
    shadow = Shadow(gate(100), gate(60))                   # the candidate refuses 60..99, which the truth approves
    event = None
    for i, st in enumerate(states(400)):
        shadow.ask(st, rid=i)
        event = shadow.label(i, truth_of(st)) or event
        if event:
            break
    assert event is not None and event["event"] == "halted" and event["scope"] == "overall"
    assert shadow.halted is event and event["e_value"] >= 1 / 0.025
    assert shadow.labels["candidate_wrong"] > shadow.labels["current_wrong"] == 0
    assert "halted" in shadow.summary()
    assert shadow.label(10_000, "yes") is None             # unknown request: nothing


def test_harm_in_one_slice_is_found_where_the_overall_test_is_blind():
    # current is wrong on 100..129 everywhere; the candidate fixes that, except in "south" where it refuses 40..99
    shadow = Shadow(gate(130), gate(100, worse_in="south", worse_limit=40), slice_of=lambda st: st["region"])
    event = None
    for i, st in enumerate(states(3000, seed=1)):
        shadow.ask(st, rid=i)
        event = shadow.label(i, truth_of(st)) or event
    assert event is not None and event["scope"] == "slice" and event["slice"] == "south"
    assert shadow.harm["overall"]["e_value"] < 1 / 0.025


def test_an_equal_candidate_is_rarely_halted():
    # Ville: under "the candidate is no more often the wrong one", the chance of ever halting is ≤ alpha
    rng = random.Random(7)
    halts = 0
    for _ in range(300):
        h = _PairedHarm(alpha=0.05)
        for _ in range(400):
            if h.update(rng.random() < 0.5):
                halts += 1
                break
    assert halts / 300 <= 0.05


def test_labels_default_to_the_stored_id_and_pick_the_question():
    shadow = Shadow(gate(100), gate(100))
    r = shadow.ask({"amount": 5, "region": "north"})
    assert r.stored_id is None
    with pytest.raises(ValueError, match="rid"):
        shadow.label(None, "yes")
    shadow.ask({"amount": 5, "region": "north"}, rid="a")
    assert shadow.label("a", {"approve": "yes"}) is None
    assert shadow.labels == {"labels": 1, "discordant": 0, "current_wrong": 0, "candidate_wrong": 0}
    assert shadow.label("a", "yes") is None                # labelled once: gone from the pending requests


def test_slices_beyond_the_budget_are_pooled_and_pending_is_bounded():
    shadow = Shadow(gate(100), gate(100), slice_of=lambda st: st["region"], max_slices=3, max_pending=5)
    for i, st in enumerate(states(20, regions=tuple("abcdef"))):
        shadow.ask(st, rid=i)
    assert len(shadow._pending) == 5 and 0 not in shadow._pending
    for i in range(15, 20):
        shadow.label(i, "yes")
    assert len(shadow.harm["slices"]) <= 3
