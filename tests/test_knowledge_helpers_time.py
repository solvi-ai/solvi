import datetime

import pytest

from solvi import Knowledge
from solvi.core.knowledge import KnowledgeStore


def indexed(tmp_path, *, dates=False, now=None, derived=False):
    path = tmp_path / "knowledge.jsonl"
    store = KnowledgeStore(path)
    begin = datetime.date(2026, 1, 1) if dates else 0
    end = datetime.date(2026, 2, 1) if dates else 10
    kwargs = {"valid": (begin, end)}
    if derived:
        p = store.add("fact", {"s": "proof", "r": "value", "o": 1}, source="outcome", by="proof", valid=(begin, None))
        store.add("fact", {"s": "proof", "r": "value", "o": 2}, source="outcome", by="proof", valid=(end, None))
        kwargs = {"derived_from": [p]}
    for at, accepted in [("p1", True), ("p0", True), ("p2", False)]:
        store.add(
            "fact",
            {"s": at, "r": "accepts move", "o": accepted},
            {"map": "m"},
            source="outcome",
            by="environment",
            **kwargs,
        )
    edge = store.add(
        "fact",
        {"s": "p0", "r": "leads_to move", "o": "destination"},
        {"map": "m"},
        source="outcome",
        by="environment",
        **kwargs,
    )
    store.add(
        "fact",
        {"s": "p0", "r": "accepts move", "o": True},
        {"map": "other"},
        source="outcome",
        by="environment",
        valid=(begin, None),
    )
    for action in ["z-last", "a-first"]:
        store.add(
            "skill", {"s": "goal:done", "r": "via " + action, "o": True}, source="outcome", by="environment", **kwargs
        )
    return Knowledge(path, now=now), edge


def assert_available(k, at):
    assert k.edges("m", "p0", valid_at=at) == {"move": "destination"}
    assert k.accepted_at("m", "p0", "move", valid_at=at) is True
    assert k.accepted_at("m", "p2", "move", valid_at=at) is False
    assert k.skills("done", valid_at=at) == ["a-first", "z-last"]
    assert k.where("m", "move", valid_at=at) == ["p0", "p1"]


def assert_unavailable(k, at):
    assert k.edges("m", "p0", valid_at=at) == {}
    assert k.accepted_at("m", "p0", "move", valid_at=at) is None
    assert k.skills("done", valid_at=at) == []
    assert k.where("m", "move", valid_at=at) == []


@pytest.mark.parametrize("derived", [False, True])
def test_numeric_boundary_and_derived_versions(tmp_path, derived):
    k, _ = indexed(tmp_path, derived=derived)
    assert_available(k, 5)
    assert_unavailable(k, 10)
    assert k.edges("m", "p0") == {"move": "destination"}
    assert k.skills("done") == ["a-first", "z-last"]
    assert k.where("m", "move") == ["p0", "p1"]
    assert k.accepted_at("m", "p2", "move") is False
    assert k.accepted_at("unknown", "p0", "move", valid_at=5) is None
    assert k.edges("unknown", "p0", valid_at=5) == {}


def test_dates_use_existing_normalization(tmp_path):
    k, _ = indexed(tmp_path, dates=True)
    assert_available(k, datetime.date(2026, 1, 20))
    assert_unavailable(k, "2026-02-01")


def test_default_clock_and_explicit_override(tmp_path):
    clock = [5]
    k, _ = indexed(tmp_path, now=lambda: clock[0])
    assert k.edges("m", "p0") and k.skills("done") and k.where("m", "move")
    clock[0] = 15
    assert k.edges("m", "p0") == {} and k.skills("done") == [] and k.where("m", "move") == []
    assert k.accepted_at("m", "p0", "move") is None
    assert_available(k, 5)
    assert_unavailable(k, None)


def test_unknown_item_index_entries_remain_unusable():
    k = Knowledge()
    k._edges = {"m": {"p": {"go": ("unknown", "next")}}}
    k._accepts = {"m": {"p": {"go": ("unknown", True)}}}
    k._skills = {"g": {"go": "unknown"}}
    assert k.edges("m", "p", valid_at=5) == {}
    assert k.accepted_at("m", "p", "go", valid_at=5) is None
    assert k.skills("g", valid_at=5) == []
    assert k.where("m", "go", valid_at=5) == []


@pytest.mark.parametrize("helper", ["edges", "accepted_at", "skills", "where"])
def test_incompatible_units_refuse_without_state_changes(tmp_path, helper):
    k, _ = indexed(tmp_path)
    args = {"edges": ("m", "p0"), "accepted_at": ("m", "p0", "move"), "skills": ("done",), "where": ("m", "move")}
    before = k.store.fingerprint(), len(k.store.journal)
    with pytest.raises(ValueError, match="valid_at=.*does not compare"):
        getattr(k, helper)(*args[helper], valid_at="2026-01-01")
    assert (k.store.fingerprint(), len(k.store.journal)) == before
