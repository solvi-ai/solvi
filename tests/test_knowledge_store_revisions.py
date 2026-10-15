"""solvi.core.knowledge.store: reads inside a batch, validity intervals (`valid=`), re-asserted retracted content
waiting for a re-test, retraction of one writer's support, re-deciding on the corrected journal, and the cost of a
load."""
import time

from solvi.core.knowledge import KnowledgeStore


def fact(s, o, r="r"):
    return {"s": s, "r": r, "o": o}


# --- reads inside a batch
def test_an_item_added_in_a_batch_is_read_as_present_inside_the_batch():
    ks = KnowledgeStore()
    with ks.batch():
        a = ks.add("fact", fact("a", 1), source="spec")
        assert ks.status(a) == "active" and ks.usable(a)
        d = ks.add("fact", fact("d", 2), source="spec", derived_from=[a])
        assert d is not None and ks.status(d) == "active"
    assert ks.status(a) == ks.status(d) == "active"
    assert ks.verify()
