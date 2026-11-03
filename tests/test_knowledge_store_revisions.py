"""solvi.core.knowledge.store: reads inside a batch, validity intervals (`valid=`), re-asserted retracted content
waiting for a re-test, retraction of one writer's support, re-deciding on the corrected journal, and the cost of a
load."""
import time

import pytest

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


# --- validity intervals
def test_values_on_disjoint_validity_intervals_do_not_dispute():
    ks = KnowledgeStore()
    a = ks.add("fact", fact("drugX", "40mg", "max_dose"), source="spec", by="label-2023",
               valid=("2023-01-01", "2024-06-01"))
    b = ks.add("fact", fact("drugX", "20mg", "max_dose"), source="spec", by="label-2024", valid=("2024-06-01", None))
    assert ks.status(a) == ks.status(b) == "active" and ks.disputes() == []
    assert ks.usable(a, valid_at="2023-05-01") and not ks.usable(a, valid_at="2025-01-01")
    assert ks.usable(b, valid_at="2025-01-01") and not ks.usable(b, valid_at="2023-05-01")
    snap = ks.snapshot(about="drugX", valid_at="2025-01-01")
    assert [f["body"]["o"] for f in snap["facts"]] == ["20mg"] and snap["query"]["valid_at"] == "2025-01-01"
    assert [h["why"] for h in snap["hints"]] == ["not valid at 2025-01-01"]
    assert ks.verify()


def test_an_overlapping_later_since_closes_the_earlier_value_instead_of_disputing_it():
    # rewritten: under the old rule the two overlapping values were a dispute; a later since now closes the earlier
    ks = KnowledgeStore()
    a = ks.add("fact", fact("x", 1), source="spec", by="p", valid=(None, 10))
    b = ks.add("fact", fact("x", 2), source="spec", by="q", valid=(5, None))
    assert ks.status(a) == ks.status(b) == "active" and ks.disputes() == []
    assert ks.periods(a) == [[None, 5]] and ks.periods(b) == [[5, None]]
    assert ks.usable(a, valid_at=4) and not ks.usable(a, valid_at=7) and ks.usable(b, valid_at=7)


def test_a_later_valid_since_closes_the_earlier_version_whatever_order_the_notes_arrive_in():
    # rewritten: the earlier version was refuted under the old rule; it is now closed at the later since and stays
    ks = KnowledgeStore()
    new = ks.add("fact", fact("anna", "manager", "role"), source="outcome", by="hr", valid=(50, None))
    old = ks.add("fact", fact("anna", "engineer", "role"), source="outcome", by="wiki", valid=(0, None))
    assert ks.status(new) == ks.status(old) == "active"
    assert ks.item(old)["confidence"]["n_refute"] == 0 and ks.item(old)["valid"]["effective"] == [[0, 50]]
    assert [f["body"]["o"] for f in ks.snapshot(about="anna", valid_at=10)["facts"]] == ["engineer"]
    assert [f["body"]["o"] for f in ks.snapshot(about="anna", valid_at=60)["facts"]] == ["manager"]


def test_the_same_valid_since_from_two_writers_is_a_dispute():
    ks = KnowledgeStore()
    x = ks.add("fact", fact("bob", "Oslo", "city"), source="outcome", by="A", valid=(10, None))
    y = ks.add("fact", fact("bob", "Rome", "city"), source="outcome", by="B", valid=(10, None))
    assert ks.status(x) == ks.status(y) == "disputed" and len(ks.disputes()) == 1


def test_an_expired_item_is_not_usable_at_a_later_time_and_items_without_valid_are_unchanged():
    ks = KnowledgeStore()
    z = ks.add("fact", fact("c", 1), source="spec", valid=(2020, 2021))
    assert ks.usable(z) and ks.usable(z, valid_at=2020) and not ks.usable(z, valid_at=2021)
    w = ks.add("fact", fact("w", 1), source="spec")
    assert ks.usable(w, valid_at=1999)
    p = ks.add("fact", fact("k", 1), source="spec")
    q = ks.add("fact", fact("k", 2), source="outcome")            # no valid: an observation refutes, as before
    assert ks.status(p) == "refuted" and ks.status(q) == "active"


# --- re-asserting retracted content
def test_re_asserting_retracted_content_waits_for_a_retest_and_so_do_its_derivations():
    ks = KnowledgeStore()
    f = ks.add("fact", fact("w", "kg", "unit"), source="spec", by="v1")
    d = ks.add("rule", fact("auto", "kg", "uses"), source="person", derived_from=[f])
    assert ks.status(d) == "active"
    ks.retract(f, why="wrong")
    assert ks.status(f) == ks.status(d) == "retracted"
    assert ks.add("fact", fact("w", "kg", "unit"), source="spec", by="v6") == f
    assert ks.status(f) == ks.status(d) == "hypothesis" and ks.pending_retest(f) and ks.pending_retest(d)
    assert not ks.usable(f) and [h["why"] for h in ks.snapshot(about="w")["hints"]] == ["pending retest"]
    ks.retest(f, ok=True, by="qa")
    assert ks.status(f) == "active" and ks.status(d) == "hypothesis"       # a derivation waits for its own re-test
    ks.retest(d, ok=True, by="qa")
    assert ks.status(d) == "active" and not ks.pending_retest(d)
    assert ks.verify()


def test_a_failed_retest_retracts_and_a_rule_re_added_after_its_retraction_is_held():
    ks = KnowledgeStore()
    f = ks.add("fact", fact("a", 1), source="spec")
    ks.retract(f)
    ks.add("fact", fact("a", 1), source="spec")
    ks.retest(f, ok=False, why="still wrong")
    assert ks.status(f) == "retracted"
    r = ks.add("rule", fact("rule", 1), source="person")
    ks.retract(r)
    assert ks.add("rule", fact("rule", 1), source="person") == r
    assert ks.status(r) == "hypothesis" and ks.journal[-1]["op"] == "held"
    assert "re-test" in ks.journal[-1]["verdicts"][-1]["reason"]


def test_content_written_after_a_retraction_of_its_premise_is_not_pending():
    ks = KnowledgeStore()
    f = ks.add("fact", fact("p", 1), source="spec")
    ks.retract(f)
    ks.add("fact", fact("p", 1), source="spec")
    ks.retest(f, ok=True)
    d = ks.add("fact", fact("q", 2), source="spec", derived_from=[f])     # derived after the re-test: fresh
    assert ks.status(d) == "active" and not ks.pending_retest(d)
    g = ks.add("fact", fact("g", 1), source="person")
    ks.add("fact", fact("g", 1), source="outcome")                        # a confirmation, never retracted
    assert ks.status(g) == "active"


def test_an_item_a_rule_re_infers_after_its_retraction_follows_its_premises():
    ks = KnowledgeStore()
    p = ks.add("fact", fact("p", 1), source="spec")
    q = ks.add("fact", fact("q", 1), source="spec", derived_from=[p], inferred=True)
    ks.retract(q, why="the negative condition no longer holds")
    assert ks.add("fact", fact("q", 1), source="spec", derived_from=[p], inferred=True) == q
    assert ks.status(q) == "active" and not ks.pending_retest(q)


# --- retraction of one writer's support
def test_retracting_one_writers_copy_keeps_the_item_on_the_others_support():
    ks = KnowledgeStore()
    a = ks.add("fact", fact("x", 1), source="spec", by="honest")
    assert ks.add("fact", fact("x", 1), source="spec", by="liar") == a
    changes = ks.retract(a, why="liar exposed", only_from="liar")
    assert changes == {} and ks.status(a) == "active" and ks.item(a)["confidence"]["n_confirm"] == 0
    ks.retract(a, only_from="honest")
    assert ks.status(a) == "retracted"
    try:
        ks.retract(a, only_from="nobody")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert ks.verify()


def test_retract_source_takes_back_only_what_that_writer_alone_supports():
    ks = KnowledgeStore()
    shared = ks.add("fact", fact("s", 1), source="spec", by="honest")
    ks.add("fact", fact("s", 1), source="spec", by="liar")
    lie = ks.add("fact", fact("l", 2), source="spec", by="liar")
    d = ks.add("fact", fact("d", 3), source="spec", by="ann", derived_from=[lie])
    changes = ks.retract_source("liar", why="exposed")
    assert ks.status(shared) == "active" and ks.status(lie) == ks.status(d) == "retracted"
    assert changes == {lie: ("active", "retracted"), d: ("active", "retracted")}
    assert ks.add("fact", fact("l", 2), source="spec", by="honest") == lie
    assert ks.pending_retest(lie)                      # its whole support was taken back: told again, it waits


def test_knowledge_value_reads_the_fact_valid_at_a_world_time():
    import solvi
    km = solvi.Knowledge()
    km.tell(("drugX", "max_dose", "10mg"), source="spec", by="label-2023", valid=("2023-01-01", "2024-06-01"))
    km.tell(("drugX", "max_dose", "20mg"), source="spec", by="label-2024", valid=("2024-06-01", None))
    km.tell(("drugX", "form", "tablet"), source="spec", by="label")                # no valid: holds at any time
    snap = km.snapshot(about="drugX")                  # both intervals are facts in a snapshot without valid_at
    with pytest.raises(ValueError, match="Ambiguous"):
        solvi.Knowledge.value(snap, "drugX", "max_dose")
    assert solvi.Knowledge.value(snap, "drugX", "max_dose", valid_at="2023-05-01") == "10mg"
    assert solvi.Knowledge.value(snap, "drugX", "max_dose", valid_at="2025-01-01") == "20mg"
    assert solvi.Knowledge.value(snap, "drugX", "max_dose", "none", valid_at="2020-01-01") == "none"
    assert solvi.Knowledge.value(snap, "drugX", "form", valid_at="2020-01-01") == "tablet"
    rows = {r["body"]["o"]: r for r in snap["facts"]}
    assert rows["20mg"]["valid"] == ["2024-06-01", None] and "valid" not in rows["tablet"]   # rows carry their valid


# --- re-deciding after a correction
def test_redecide_on_the_current_journal_gives_the_corrected_answer(tmp_path):
    import solvi
    from solvi import Answer, Catalog, Question, System
    from solvi.core.store import JSONLStorage
    st = JSONLStorage(tmp_path / "d.jsonl")
    cat = Catalog()

    @cat.rule("dose")
    def dose(knowledge) -> str:
        return solvi.Knowledge.value(knowledge, "drugX", "max_dose", "unknown")
    s = System(cat, [Question("dose", "Dose?", Answer.choice(["10mg", "20mg", "40mg", "unknown"]))], storage=st)
    km = solvi.Knowledge()
    wrong = km.tell(("drugX", "max_dose", "40mg"), source="spec", by="typo")
    s.ask({"knowledge": km.snapshot(about="drugX")})
    km.store.retract(wrong, why="typo")
    km.tell(("drugX", "max_dose", "20mg"), source="spec", by="fix")
    ans = lambda init: s.ask(init, store=False).results["dose"].answer   # noqa: E731
    ch, _ = km.store.redecide(st, {wrong}, ans)
    assert [(c["old"], c["new"]) for c in ch] == [("40mg", "unknown")]   # the default: the journal at the decision
    ch, _ = km.store.redecide(st, {wrong}, ans, upto="now")
    assert [(c["old"], c["new"]) for c in ch] == [("40mg", "20mg")]


# --- the cost of a load
def test_a_write_re_resolves_only_what_it_can_change(monkeypatch):
    import solvi.core.knowledge.store as store_mod
    sizes, real = [], store_mod._resolve
    monkeypatch.setattr(store_mod, "_resolve", lambda items, comp, *a, **k: sizes.append(len(comp)) or real(items, comp, *a, **k))
    ks, prev = KnowledgeStore(), None
    for i in range(300):                                 # one connected chain: each add used to re-resolve all of it
        prev = ks.add("fact", fact(f"s{i}", i), source="spec", derived_from=[prev] if prev else ())
    assert max(sizes) == 1 and ks.verify()
    t0 = time.perf_counter()
    for i in range(300, 2000):
        prev = ks.add("fact", fact(f"s{i}", i), source="spec", derived_from=[prev])
    assert time.perf_counter() - t0 < 5.0                # 2,000 chained facts: ~15 s before, ~0.1 s now


def test_incremental_resolution_equals_the_rebuilt_store():
    import random
    rng = random.Random(3)
    ks, ids = KnowledgeStore(), []
    for step in range(600):
        op = rng.random()
        if op < 0.7 or not ids:
            prem = rng.sample(ids, min(len(ids), rng.choice([0, 0, 1, 2])))
            i = ks.add("fact", fact(f"e{rng.randrange(40)}", rng.randrange(3)),
                       source=rng.choice(["person", "outcome", "spec"]), by=rng.choice("ab"), derived_from=prem,
                       valid=rng.choice([None, None, (rng.randrange(5), None)]))
            if i:
                ids.append(i)
        elif op < 0.8:
            ks.retract(rng.choice(ids))
        elif op < 0.85:
            x = rng.choice(ids)
            ks.retest(x, ok=True)
        elif op < 0.9 and ks.disputes():
            d = ks.disputes()[0]
            ks.resolve(d["key"], ks.items[d["items"][0]].value, by="ann")
        elif op < 0.95:
            ks.reconfirm(rng.choice(ids))
        if step % 50 == 0:
            assert ks.fingerprint() == ks.rebuild().fingerprint()
    assert ks.verify()


# --- the cost of the revival guard (S266)
def test_a_guard_active_write_does_not_rescan_the_whole_key():
    """A corpus with withdrawals and copies (every 5th write: an only_from erratum, then the withdrawn content
    re-told by the other writer — a copy) used to re-read the key's every assertion on every write: 27.7 ms/write
    at 300 writes, 6.8× the plain versioned pattern and doubling with the corpus, because each withdrawal was
    matched against each live assertion again and the walk re-ran per bad copy. The guard now keeps each item's
    withdrawals, re-test clears and the live assertions' matches against them between resolutions, so a write
    re-reads only what moved (0.6 ms/write, cheaper than the plain pattern, whose live set nothing prunes)."""
    import time


    def run(m, guard):
        ks, t0, last = KnowledgeStore(), time.perf_counter(), None
        for n in range(m):
            v = 8 if n % 3 else 5
            w = "a" if n % 2 else "b"
            last = ks.add("fact", fact("K", v), source="outcome", by=w, valid=(n, None))
            if guard and n % 5 == 4:
                ks.retract(last, only_from=w, by="x", why="erratum")
                ks.add("fact", fact("K", v), source="outcome", by="b" if w == "a" else "a", valid=(n, None))
        assert ks.verify()                                # the incrementally kept state is the journal's, exactly
        return time.perf_counter() - t0

    t150 = run(150, guard=True)
    t300, p300 = run(300, guard=True), run(300, guard=False)
    assert t300 < 5.0                                     # was 8.3 s before the guard kept its per-key state
    assert t300 < 2.5 * p300                              # the guard must not cost multiples of the plain pattern
    assert t300 < 4.5 * t150                              # and its per-write cost stays ~flat as the corpus doubles


def test_guard_histories_read_incrementally_match_the_rebuild():
    """The guard's per-key cache must read exactly what a from-scratch resolution reads: random histories over
    withdrawals (plain, only_from, one assertion), copies of withdrawn content re-told by another writer and
    re-tests — resolved write by write, each write its own flush — are compared with their cold rebuild on the
    fingerprint, the pending items and the full item records (the claims' pending/unchecked markers the guard
    decides)."""
    import random
    rng = random.Random(11)
    ks, ids = KnowledgeStore(), []
    for step in range(600):
        op = rng.random()
        if op < 0.55 or not ids:
            i = ks.add("fact", fact(f"e{rng.randrange(5)}", rng.randrange(3)), source=rng.choice(["person", "outcome"]),
                       by=rng.choice("ab"), valid=(rng.randrange(4), None) if rng.random() < 0.85 else None)
            if i:
                ids.append(i)
        elif op < 0.68:
            try:
                ks.retract(rng.choice(ids), only_from=rng.choice("ab"), by="x", why="erratum")
            except ValueError:
                pass                                       # that writer has no standing assertion left there
        elif op < 0.78:
            x = rng.choice(ids)
            claims = [c["at"] for c in (ks.item(x)["valid"].get("claims") or ())]
            if claims:
                ks.retract(x, assertion=rng.choice(claims), by="x", why="erratum")
        elif op < 0.86:
            ks.retest(rng.choice(ids), ok=rng.random() < 0.6, by="qa")
        elif op < 0.92:
            ks.retract(rng.choice(ids), by="x", why="wrong")
        elif op < 0.96:                                    # a copy: withdrawn content re-told by another writer
            x = rng.choice(ids)
            it = ks.items[x]
            if it.key is not None and it.value is not None:
                ks.add("fact", {"s": it.key[1], "r": it.key[2], "o": it.value}, source="person", by="c",
                       valid=(rng.randrange(4), None))
        if step % 60 == 59:
            r = ks.rebuild()
            assert ks.fingerprint() == r.fingerprint()
            assert ks.pending() == r.pending()
            assert {i: ks.item(i) for i in sorted(ks.items)} == {i: r.item(i) for i in sorted(r.items)}
    assert ks.verify()


@pytest.mark.parametrize("order", [("Zanotti", "Terenzi"), ("Terenzi", "Zanotti")])
def test_knowledge_reader_requires_explicit_disambiguation_across_scopes(order):
    import solvi
    km = solvi.Knowledge(writers={"wikidata": "outcome"})
    for value in order:
        km.tell(("SM", "P35", value), source="outcome", by="wikidata",
                valid=("2014-10-01", "2015-04-01"), scope={"holder": value, "realm": "SM"})
    snap = km.snapshot(about="SM", relation="P35", valid_at="2015-01-01")
    for rows in (snap["facts"], list(reversed(snap["facts"]))):
        snapshot = {"facts": rows}
        with pytest.raises(ValueError, match="Ambiguous"):
            solvi.Knowledge.value(snapshot, "SM", "P35", "missing", valid_at="2015-01-01")
        assert set(solvi.Knowledge.values(snapshot, "SM", "P35")) == set(order)
        assert solvi.Knowledge.value(snapshot, "SM", "P35", scope={"holder": order[0]}) == order[0]
        assert solvi.Knowledge.values(snapshot, "SM", "P35", scope={"holder": "absent"}) == []


def test_knowledge_reader_manual_snapshot_preserves_periods_and_equal_values():
    import solvi
    rows = [
        {"body": fact("s", {"v": [1]}), "scope": {"map": "a"}, "valid": [0, 10]},
        {"body": fact("s", {"v": [1]}), "scope": {"map": "b"}, "periods": [[0, 5], [8, 10]]},
        {"body": fact("s", "other"), "valid": [0, 10], "periods": [[5, 8]]},
    ]
    for order in (rows, list(reversed(rows))):
        snap = {"facts": order}
        assert solvi.Knowledge.value(snap, "s", "r", valid_at=2) == {"v": [1]}
        assert solvi.Knowledge.values(snap, "s", "r", valid_at=2) == [{"v": [1]}]
        with pytest.raises(ValueError, match="Ambiguous"):
            solvi.Knowledge.value(snap, "s", "r", valid_at=6)
        assert solvi.Knowledge.value(snap, "s", "r", valid_at=6, scope={"map": "b"}) is None
        assert solvi.Knowledge.value(snap, "missing", "r", "absent") == "absent"
    assert solvi.Knowledge.values(None, "s", "r") == []
