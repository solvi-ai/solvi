"""solvi.core.knowledge.store: the knowledge store — source check, consistency and disputes keyed on the dispute event,
the retraction cascade (exact: the store after a retraction is the store rebuilt without it), staleness flags read where
staleness is asked, a rollback during a pending flag restoring only a hint (the two R10 defects), persistence on the
TraceStorage backends, snapshots in decisions, and the re-decide list split into answer changes and justification
only."""
import random
import warnings

import pytest

from solvi import Catalog, Question, System
from solvi.core.knowledge import ConsistencyGate, KnowledgeStore, SourceGate
from solvi.core.store import JSONLStorage, SQLiteStorage


def fact(s, o, r="r"):
    return {"s": s, "r": r, "o": o}


# --- the source check
@pytest.mark.parametrize("bad", ["system", "model", "self", "s1", "human", None])
def test_the_systems_own_answers_and_unknown_sources_are_refused(bad):
    ks = KnowledgeStore()
    assert ks.add("fact", fact("a", 1), source=bad) is None
    last = ks.journal[-1]
    assert last["op"] == "refused" and last["why"].startswith("source:") and ks.items == {}


def test_a_verified_item_names_its_decision_and_is_checked_in_the_store(tmp_path, monkeypatch):
    ks = KnowledgeStore()
    assert ks.add("fact", fact("x", "yes", "q"), source="verified", of="abc") is None
    assert "TraceStorage" in ks.journal[-1]["why"]
    st = JSONLStorage(tmp_path / "d.jsonl")
    ks = KnowledgeStore(st)
    assert ks.add("fact", fact("x", "yes", "q"), source="verified") is None and "of=" in ks.journal[-1]["why"]
    assert ks.add("fact", fact("x", "yes", "q"), source="verified", of="nope") is None
    assert "not a stored decision" in ks.journal[-1]["why"]
    seen = []
    monkeypatch.setattr(st, "_check_vouched", lambda q, a, of: seen.append((q, a, of)))
    i = ks.add("fact", fact("x", "yes", "q"), source="verified", of="d1", level=0.05)
    assert i is not None and seen == [("q", "yes", "d1")] and ks.item(i)["confidence"]["level"] == 0.05
    assert ks.item(i)["valid"]["reconfirm_after"] == 2000        # weaker evidence: re-confirmed after 2,000 ticks


def test_the_source_gate_is_always_first():
    ks = KnowledgeStore(gates=[ConsistencyGate()])
    assert isinstance(ks.gates[0], SourceGate) and ks.add("fact", fact("a", 1), source="model") is None
    with pytest.raises(TypeError):
        KnowledgeStore(gates=[object()])


def test_consistency_gate_refuses_missing_retracted_and_self_defeating_premises():
    ks = KnowledgeStore()
    assert ks.add("fact", fact("b", 1), source="person", derived_from=["nope"]) is None
    a = ks.add("fact", fact("a", 1), source="person")
    assert ks.add("fact", fact("a", 2), source="person", derived_from=[a]) is None
    assert "self-defeating" in ks.journal[-1]["why"]
    ks.retract(a)
    assert ks.add("fact", fact("c", 1), source="person", derived_from=[a]) is None
    assert "retracted" in ks.journal[-1]["why"]


# --- consistency and disputes
def test_outcome_refutes_rank_refutes_supersedes_refutes():
    ks = KnowledgeStore()
    a = ks.add("fact", fact("k", 1), source="person")
    b = ks.add("fact", fact("k", 2), source="outcome")              # the world changed
    assert (ks.status(a), ks.status(b)) == ("refuted", "active") and ks.item(a)["valid"]["known"][1] is not None
    c = ks.add("fact", fact("j", 1), source="spec")
    d = ks.add("fact", fact("j", 2), source="person")               # a higher rank
    assert (ks.status(c), ks.status(d)) == ("refuted", "active")
    e = ks.add("fact", fact("j", 3), source="person", supersedes=d)  # the producer's own new version
    assert (ks.status(d), ks.status(e)) == ("refuted", "active") and ks.item(e)["version"] == 3
    same = ks.add("fact", fact("j", 3), source="outcome")             # the same claim again: a confirmation
    assert same == e and ks.item(e)["confidence"]["n_confirm"] == 1


def test_a_dispute_goes_to_a_person_keyed_on_the_event_and_is_never_a_fact():
    ks = KnowledgeStore()
    a = ks.add("fact", fact("k", 1), source="person", by="ann")
    b = ks.add("fact", fact("k", 2), source="person", by="bob")
    assert ks.status(a) == ks.status(b) == "disputed"
    (q,) = ks.disputes()
    assert set(q["items"]) == {a, b} and ks.snapshot(about="k")["facts"] == [] and ks.snapshot(about="k")["hints"] == []
    c = ks.add("fact", fact("k", 3), source="spec")                   # a third item joins the open dispute: same question
    assert len(ks.questions) == 1 and ks.status(c) == "disputed"
    ks.resolve(ks.items[a].key, 1, by="lead")
    assert (ks.status(a), ks.status(b), ks.status(c)) == ("active", "refuted", "refuted") and ks.disputes() == []
    ks.add("fact", fact("k", 2), source="person", by="bob")          # disputed again after the resolution: asked again
    assert len(ks.questions) == 2 and len(ks.disputes()) == 1
    ks.add("fact", fact("k", 2), source="outcome")                     # an observation ends a dispute
    assert ks.disputes() == [] and ks.status(b) == "active"


def test_no_disputes_on_a_clean_stream():
    ks = KnowledgeStore()
    rng = random.Random(1)
    for n in range(300):
        ks.add("fact", fact(f"x{n}", rng.choice("ab")), source=rng.choice(["person", "outcome", "spec"]))
    assert ks.questions == [] and ks.counts() == {"active": 300}


# --- retraction
def test_the_retraction_cascade_and_what_comes_back():
    ks = KnowledgeStore()
    a = ks.add("fact", fact("x", "y", "same"), source="person")
    b = ks.add("fact", fact("x", "acme", "brand"), source="person", derived_from=[a])
    c = ks.add("fact", fact("x", "z", "same"), source="spec")          # lower rank against a person: a dispute
    assert ks.status(c) == "disputed" and ks.status(a) == "disputed"
    changes = ks.retract(a, by="bob", why="wrong label")
    assert changes[a] == ("disputed", "retracted") and changes[b][1] == "retracted" and ks.status(c) == "active"
    assert ks.verify() and ks.fingerprint() == ks.rebuild(skip={a}).fingerprint()
    assert [r["op"] for r in ks.journal].count("retract") == 1        # nothing deleted: the chain stays whole
    again = ks.add("fact", fact("x", "y", "same"), source="person")   # asserted again later: present again
    assert again == a and ks.status(a) in ("active", "disputed")


def test_random_retractions_are_exact():
    rng = random.Random(7)
    ks = KnowledgeStore()
    ids = []
    with ks.batch():
        for _ in range(400):
            prem = rng.sample(ids, min(len(ids), rng.choice([0, 0, 1, 2]))) if ids else []
            i = ks.add("fact", fact(f"e{rng.randrange(60)}", rng.randrange(3)), source=rng.choice(["person", "outcome", "spec"]),
                       derived_from=prem)
            if i:
                ids.append(i)
    for _ in range(40):
        x = rng.choice(ids)
        r = ks.rebuild()
        r.retract(x)
        assert r.fingerprint() == ks.rebuild(skip={x}).fingerprint()


# --- staleness, flags, rollback (the R10 defects)
def test_a_rollback_during_a_pending_flag_restores_the_previous_version_only_as_a_hint():
    ks = KnowledgeStore()
    v0 = ks.add("rule", fact("s1", "v0", "version"), {"q": "intent"}, source="person")
    v1 = ks.add("rule", fact("s1", "v1", "version"), {"q": "intent"}, source="person", supersedes=v0)
    assert (ks.status(v0), ks.status(v1)) == ("refuted", "active")
    flag = ks.reconfirm(v1, why="drift monitor")                       # the flag is on the new version
    assert ks.stale(v1) and ks.status(v1) == "hypothesis"
    changes = ks.rollback(v1, by="supervision", why="Q_lo loss")      # supervision rolls the new version back
    assert changes[v0] == ("refuted", "hypothesis") and ks.status(v1) == "retracted"
    assert ks.stale(v0) and not ks.usable(v0)                         # the stale version is a hint, never a fact
    snap = ks.snapshot(about="s1")
    assert snap["facts"] == [] and [h["id"] for h in snap["hints"]] == [v0]
    ks.end_flag(flag, by="person", why="re-confirmed")
    assert ks.usable(v0) and ks.verify()


def test_a_rollback_without_a_pending_flag_restores_a_fact():
    ks = KnowledgeStore()
    v0 = ks.add("rule", fact("s1", "v0", "version"), source="person")
    v1 = ks.add("rule", fact("s1", "v1", "version"), source="person", supersedes=v0)
    f = ks.reconfirm(v1)
    ks.end_flag(f)
    ks.rollback(v1)
    assert ks.usable(v0)


def test_a_scope_flag_covers_items_learned_before_it_and_the_stale_check_reads_the_flag():
    ks = KnowledgeStore()
    old = ks.add("fact", fact("thr", 0.9), {"q": "intent"}, source="person")
    other = ks.add("fact", fact("thr", 0.9), {"q": "other"}, source="person")
    ks.flag(scope={"q": "intent"}, why="open-set CUSUM")
    new = ks.add("fact", fact("thr2", 0.7), {"q": "intent"}, source="person")    # learned after the flag: a fact
    assert ks.stale(old) and ks.status(old) == "hypothesis" and ks.usable(other) and ks.usable(new)
    ks.add("fact", fact("thr", 0.9), {"q": "intent"}, source="spec")              # a spec renewal does not lift it
    assert ks.stale(old)
    ks.add("fact", fact("thr", 0.9), {"q": "intent"}, source="person", by="ann")  # a person re-confirms it
    assert ks.usable(old)


def test_reconfirm_after_expires_on_the_clock_and_a_renewal_restores():
    ks = KnowledgeStore()
    v = ks.add("fact", fact("x", "yes", "q"), source="person", reconfirm_after=10)
    keep = ks.add("fact", fact("y", "yes", "q"), source="outcome")
    ks.tick(5)
    assert ks.usable(v)
    ks.tick(6)
    assert ks.status(v) == "expired" and ks.stale(v) and ks.usable(keep)
    assert [h["why"] for h in ks.snapshot(about="x")["hints"]] == ["expired"]
    ks.add("fact", fact("x", "yes", "q"), source="person")
    assert ks.usable(v)


def test_behaviour_changing_items_are_promoted_by_the_gates_or_held():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental.learning import ShadowGate
    ks = KnowledgeStore()
    r = ks.add("rule", fact("refund", "original method", "only_to"), source="spec")
    assert ks.status(r) == "active" and any(e["op"] == "promote" for e in ks.journal)
    ks = KnowledgeStore(gates=[SourceGate(), ConsistencyGate(), ShadowGate(min_gain=0.01, max_changed=0.3)])
    held = ks.add("rule", fact("a", 1), source="person")
    assert ks.status(held) == "hypothesis" and ks.journal[-1]["op"] == "held"
    ok = ks.add("rule", fact("b", 1), source="person", shadow={"gain": 0.02, "changed": 0.1, "honest": True})
    assert ks.status(ok) == "active"
    bad = ks.add("skill", fact("c", 1), source="person", shadow=lambda store, item: {"gain": 0.05, "changed": 0.5})
    assert ks.status(bad) == "hypothesis" and "changes 0.5" in str(ks.journal[-1]["verdicts"])
    assert ks.status(ks.add("fact", fact("d", 1), source="outcome")) == "active"   # an observation needs no shadow
    ks.promote(held, measured={"by": "a person"})
    assert ks.status(held) == "active"


def test_the_item_record_has_the_schema():
    ks = KnowledgeStore()
    a = ks.add("fact", fact("x", 1), {"q": "t"}, source="person", by="ann", evidence=["d1"], valid=("2026-01-01", None))
    ks.used([a])
    ks.used([a], wrong=True)
    rec = ks.item(a)
    assert set(rec) == {"id", "kind", "body", "scope", "source", "by", "evidence", "derived_from", "confidence", "status",
                        "stale", "version", "supersedes", "valid", "of"}
    assert rec["confidence"] == {"n_confirm": 0, "n_refute": 0, "n_used": 2, "n_used_wrong": 1, "level": None}
    assert rec["valid"]["world"] == ["2026-01-01", None] and rec["valid"]["known"] == [rec["valid"]["known"][0], None]
    assert ks.find(kind="fact", s="x", scope={"q": "t"})[0]["id"] == a and ks.find(status="refuted") == []
    rep = ks.report()
    assert rep["by_source"] == {"person": 1} and rep["status"] == {"active": 1}


# --- persistence
@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_the_journal_lives_in_a_trace_storage_and_replays(tmp_path, backend):
    path = tmp_path / ("k.jsonl" if backend == "jsonl" else "k.db")
    ks = KnowledgeStore(path)
    a = ks.add("fact", fact("a", 1), source="person")
    b = ks.add("fact", fact("b", 2), source="outcome", derived_from=[a])
    ks.reconfirm(b, why="drift")
    ks.tick(3)
    fp = ks.fingerprint()
    assert ks.verify() and ks.storage.verify()["ok"]
    st = JSONLStorage(path) if backend == "jsonl" else SQLiteStorage(path)
    again = KnowledgeStore(st)
    assert again.fingerprint() == fp and again.stale(b) and again.clock == 3 and len(again) == len(ks)


def test_knowledge_and_decisions_share_one_chain_and_a_decision_records_the_snapshot(tmp_path):
    store = JSONLStorage(tmp_path / "decisions.jsonl")
    ks = KnowledgeStore(store)
    vip = ks.add("fact", fact("c17", "vip", "tier"), source="person", by="crm")
    cat = Catalog()

    @cat.rule("priority")
    def priority(knowledge: dict) -> bool:
        return any(f["body"]["o"] == "vip" for f in knowledge["facts"])
    s = System(cat, [Question("priority", "Priority?")], storage=store)
    res = s.ask({"knowledge": ks.snapshot(about="c17")})
    assert res["priority"].answer == "yes"
    rec = store.record(res.stored_id)
    assert rec["response"]["trace"]["init"]["knowledge"]["fp"] == ks.fingerprint()
    assert store.verify()["ok"] and len(list(store.iter())) == 1 and vip in ks.snapshot(about="c17")["rests_on"]


def test_an_edited_journal_entry_is_caught(tmp_path):
    ks = KnowledgeStore(tmp_path / "k.jsonl")
    ks.add("fact", fact("a", 1), source="person")
    ks.add("fact", fact("b", 1), source="person")
    lines = (tmp_path / "k.jsonl").read_text().splitlines()
    (tmp_path / "k.jsonl").write_text("\n".join([lines[0].replace('"o": 1', '"o": 2')] + lines[1:]) + "\n")
    assert not KnowledgeStore(JSONLStorage(tmp_path / "k.jsonl")).verify()
    mem = KnowledgeStore()
    mem.add("fact", fact("a", 1), source="person")
    mem.journal[0]["body"]["o"] = 5
    assert not mem.verify()


# --- re-deciding
def test_redecide_splits_answer_changes_from_justification_only(tmp_path):
    store = JSONLStorage(tmp_path / "d.jsonl")
    ks = KnowledgeStore(store)
    vip = ks.add("fact", fact("c1", "vip", "tier"), source="person")
    note = ks.add("fact", fact("c2", "polite", "note"), source="person")
    ks.add("fact", fact("c2", "vip", "tier"), source="outcome")
    cat = Catalog()

    @cat.rule("priority")
    def priority(knowledge: dict) -> bool:
        return any(f["body"]["o"] == "vip" for f in knowledge["facts"])
    s = System(cat, [Question("priority", "Priority?")], storage=store)
    s.ask({"knowledge": ks.snapshot(about="c1")})      # rests on vip: retracting it changes the answer
    s.ask({"knowledge": ks.snapshot(about="c2")})      # rests on note and the c2 tier: only the justification changes
    s.ask({"knowledge": ks.snapshot(about="c3")})      # rests on nothing
    ks.retract(vip, why="wrong")
    ks.retract(note, why="wrong")
    changes, same = ks.redecide(store, {vip, note}, s)
    assert [c["old"]["priority"][0] for c in changes] == ["yes"] and [c["new"]["priority"][0] for c in changes] == ["no"]
    assert len(same) == 1 and same[0]["old"] == same[0]["new"]
    assert len(changes) + len(same) == 2
