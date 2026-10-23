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
    assert solvi.Knowledge.value(snap, "drugX", "max_dose") in ("10mg", "20mg")   # unchanged: whichever row is first
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
