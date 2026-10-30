"""The revival guard after the 2026-10-30 handoff (S262, S265), premise cycles (S261) and the opt-in world clock
(S263 / the old S25): an erratum of content the writer no longer stands behind withdraws it (a mirror copy is a
copy pending a re-test, not a silent revival), a passed re-test vouches for its own validity only, support through
a premise cycle is no support, and `KnowledgeStore(now=)` lets `usable()` read the effective intervals without an
explicit `valid_at`."""
import warnings

from solvi.core.knowledge.store import KnowledgeStore


def test_an_erratum_of_content_the_writer_no_longer_stands_behind_withdraws_it():
    ks = KnowledgeStore()
    i = ks.add("fact", {"s": "w", "r": "warranty", "o": 36}, source="person", by="primary", valid=(0, None))
    ks.add("fact", {"s": "w", "r": "warranty", "o": 36}, source="person", by="primary", valid=(100, None))
    ks.add("fact", {"s": "w", "r": "warranty", "o": 36}, source="person", by="vendor", valid=(161, None))
    echo = max(c["at"] for c in ks.item(i)["valid"]["claims"] if c["by"] == "primary")   # the echo at 161
    ks.retract(i, by="primary", why="erratum", assertion=echo)
    ks.add("fact", {"s": "w", "r": "warranty", "o": 24}, source="person", by="primary", valid=(161, None))
    mirror = ks.add("fact", {"s": "w", "r": "warranty", "o": 36}, source="person", by="mirror", valid=(161, None))
    ks.retract_source("vendor", by="audit")                              # the fabrication's source is exposed
    assert mirror in ks.pending(), "the copy of withdrawn content waits for a re-test, it does not revive"
    assert any(c.get("pending") for c in ks.item(mirror)["valid"]["claims"] if c["by"] == "mirror")
    ks.retest(mirror, ok=True, by="qa")                                  # a person vouches: the copy holds again
    assert mirror not in ks.pending() and ks.status(mirror) == "disputed"   # and takes its side against the correction


def test_a_retest_on_one_validity_does_not_clear_a_withdrawal_on_another():
    ks = KnowledgeStore()
    i = ks.add("fact", {"s": "d", "r": "limit", "o": 1000}, source="person", by="policy", valid=(108, None))
    ks.retract(i, by="policy", why="erratum", only_from="policy")
    ks.add("fact", {"s": "d", "r": "limit", "o": 20000}, source="person", by="policy", valid=(108, None))
    ks.add("fact", {"s": "d", "r": "limit", "o": 1000}, source="person", by="policy", valid=(149, None))
    ks.retest(i, ok=True, by="person")                                   # vouches for 1000 on [149, ∞) only
    mirror = ks.add("fact", {"s": "d", "r": "limit", "o": 1000}, source="person", by="mirror", valid=(108, None))
    assert mirror in ks.pending(), "the re-test on [149, ∞) did not vouch for the withdrawal on [108, ∞)"
    ks.retest(mirror, ok=True, by="person")
    assert mirror not in ks.pending() and ks.status(mirror) == "disputed"


def test_support_through_a_premise_cycle_is_no_support():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks = KnowledgeStore()
        p = ks.add("fact", {"s": "p", "r": "is", "o": 1}, source="outcome")
        a = ks.add("fact", {"s": "a", "r": "is", "o": 1}, source="spec", derived_from=[p], inferred=True)
        b = ks.add("fact", {"s": "b", "r": "is", "o": 1}, source="spec", derived_from=[a], inferred=True)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            ks.add("fact", {"s": "a", "r": "is", "o": 1}, source="spec", derived_from=[b], inferred=True)
        assert any("cycle" in str(w.message) for w in seen), "the write warns about the premise cycle"
        assert ks.status(a) == "active" and ks.status(b) == "active"    # the loop still holds while p does
        ks.retract(p, by="audit", why="fabricated")
        assert ks.status(a) == "hypothesis" and ks.status(b) == "hypothesis"
        assert not ks.usable(a) and not ks.usable(b) and {a, b} <= set(ks.pending())
        ks.retest(a, ok=True, by="person")                               # a person vouches for a again: the
        assert ks.status(a) in ("active", "disputed")                    # inferred re-derivation of b follows


def test_the_opt_in_world_clock_reads_effective_intervals():
    ks = KnowledgeStore()
    a = ks.add("fact", {"s": "a", "r": "price", "o": 100}, source="person", valid=(0, None))
    ks.add("fact", {"s": "a", "r": "price", "o": 120}, source="person", valid=(50, None))
    assert ks.usable(a) is True, "no clock, no valid_at: status only (unchanged default)"
    now = [0]
    ks2 = KnowledgeStore(now=lambda: now[0])
    b = ks2.add("fact", {"s": "a", "r": "price", "o": 100}, source="person", valid=(0, None))
    ks2.add("fact", {"s": "a", "r": "price", "o": 120}, source="person", valid=(50, None))
    assert ks2.usable(b) is True
    now[0] = 60
    assert ks2.usable(b) is False, "the version closed at 50 is not usable at the clock's time"
    ks3 = KnowledgeStore(now=60)
    c = ks3.add("fact", {"s": "a", "r": "price", "o": 100}, source="person", valid=(0, None))
    ks3.add("fact", {"s": "a", "r": "price", "o": 120}, source="person", valid=(50, None))
    assert ks3.usable(c) is False, "a fixed clock reads the same"
