"""The writers registry (opt-in): a knowledge source is a rank (outcome > person > spec > verified wins a contradiction)
that, without a registry, is whatever the writer declares — a writer that says "outcome" silently refutes a written
spec. Knowledge(writers={writer: source}) / KnowledgeStore(writers=) binds the rank to the writer: an unregistered
writer, or one declaring a source it is not registered with, is refused with the reason in the journal."""
import pytest

from solvi import Knowledge
from solvi.core.environment import Outcome
from solvi.core.knowledge import KnowledgeStore

SPEC = {"s": "pump7", "r": "max_pressure", "o": 10}
LIE = {"s": "pump7", "r": "max_pressure", "o": 99}


def _refusals(store):
    return [r["why"] for r in store.journal if r["op"] == "refused"]


def test_without_a_registry_a_declared_rank_is_taken_as_before():
    km = Knowledge()
    spec = km.tell(SPEC, source="spec", by="manual")
    lie = km.tell(LIE, source="outcome", by="liar")              # outranks the spec: it is refuted, no dispute
    assert lie is not None and km.store.status(spec) == "refuted" and not km.store.disputes()


def test_with_a_registry_a_rank_the_writer_is_not_registered_for_is_refused():
    km = Knowledge(writers={"manual": "spec", "liar": "verified", "sensor": ["outcome", "person"]})
    spec = km.tell(SPEC, source="spec", by="manual")
    assert km.tell(LIE, source="outcome", by="liar") is None        # registered as "verified", declares "outcome"
    assert km.tell(LIE, source="outcome", by="stranger") is None    # not registered at all
    assert km.tell(LIE, source="outcome") is None                   # no writer named
    assert km.store.status(spec) == "active"
    why = _refusals(km.store)
    assert len(why) == 3 and all(w.startswith("writers: ") for w in why)
    assert "'liar' declares source 'outcome', but is registered with verified" in why[0]
    assert "'stranger' is not in the writers registry" in why[1] and "None is not in the writers registry" in why[2]
    reading = km.tell({"s": "pump7", "r": "pressure", "o": 7}, source="outcome", by="sensor")   # one of its sources
    assert reading is not None and km.store.status(reading) == "active"


def test_the_registry_refuses_rules_too_not_holds_them():
    ks = KnowledgeStore(writers={"miner": "spec"})
    rule = {"s": "x", "r": "implies", "o": "y"}
    assert ks.add("rule", rule, source="outcome", by="miner") is None
    assert ks.add("rule", rule, source="spec", by="miner") is not None
    assert _refusals(ks) and not [r for r in ks.journal if r["op"] == "held" and r.get("by") == "miner"
                                  and r.get("source") == "outcome"]


def test_the_registry_is_checked_when_given():
    with pytest.raises(ValueError, match="sources from"):
        KnowledgeStore(writers={"bot": "system"})
    with pytest.raises(ValueError, match="sources from"):
        Knowledge(writers={"bot": []})


def test_a_registry_still_lets_an_agent_write_what_it_observes_when_the_environment_is_registered():
    km = Knowledge(vocabulary={"here": lambda s, a: s["here"]}, writers={"environment": "outcome"})
    km.observe({"here": "a"}, "go", Outcome({"here": "b"}), at="a", to="b", map="m1")
    assert km.store.items and not _refusals(km.store)
    blocked = Knowledge(vocabulary={"here": lambda s, a: s["here"]}, writers={"crm": "person"})
    blocked.observe({"here": "a"}, "go", Outcome({"here": "b"}), at="a", to="b", map="m1")
    assert _refusals(blocked.store) and all("'environment' is not in the writers registry" in w
                                            for w in _refusals(blocked.store))
