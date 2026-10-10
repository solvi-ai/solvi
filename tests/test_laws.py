"""solvi.experimental.laws: laws of an environment from prediction mismatches, accepted only by a held-out test.

A tiny world used throughout: a door is "open" or "locked"; "push" opens an unlocked door; a locked door opens only
with a key; "wait" changes nothing. The base predictor knows nothing (it predicts "same")."""
from __future__ import annotations

import warnings

import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.core.store import JSONLStorage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, Buffer, CallableProposer, Candidate, Enumerate, Grammar, Law, LawSet,
                                         RandomProposer, Transition, discriminate, split)


def door_grammar(k=3):
    return Grammar([
        Atom("locked", lambda s: s["locked"], feature="lock"),
        Atom("unlocked", lambda s: not s["locked"], feature="lock"),
        Atom("key", lambda s: s["key"], feature="key"),
        Atom("no_key", lambda s: not s["key"], feature="key"),
        Atom("light", lambda s: s["light"], feature="light"),
        Atom("dark", lambda s: not s["light"], feature="light"),
    ], k=k)


def door(state, action):
    """The written mechanics: what really happens."""
    if action == "push" and (not state["locked"] or state["key"]):
        return "opens"
    return "same"


def states():
    return [{"locked": lo, "key": k, "light": li} for lo in (False, True) for k in (False, True) for li in (False, True)]


def rows(group, actions=("push", "wait"), repeat=1, env=door):
    out = []
    for r in range(repeat):
        for s in states():
            for a in actions:
                out.append(Transition(s, a, env(s, a), group=group, id=f"{group}-{r}-{len(out)}"))
    return out


def buffer_of(*groups, repeat=1, env=door):
    b = Buffer(door_grammar())
    for g in groups:
        b.extend(rows(g, repeat=repeat, env=env))
    return b


def lawset(**kw):
    kw.setdefault("base", lambda s, a: "same")
    kw.setdefault("min_support", 2)
    return LawSet(door_grammar(), **kw)


# ───────────────────────────────────────────── the grammar
def test_an_atom_may_read_the_action_too():
    g = Grammar([Atom("pushing", lambda s, a: a == "push"), Atom("locked", lambda s: s["locked"])])
    assert g.true({"locked": True}, "push") == ("locked", "pushing")
    assert g.true({"locked": False}, "wait") == ()


def test_conditions_are_conjunctions_of_at_most_k_atoms_one_per_feature():
    g = door_grammar(k=2)
    true = g.true({"locked": True, "key": False, "light": True}, "push")
    assert true == ("light", "locked", "no_key")
    conds = g.conditions(true)
    assert () in conds and ("locked",) in conds and ("locked", "no_key") in conds
    assert all(len(c) <= 2 for c in conds) and ("light", "locked", "no_key") not in conds
    assert len(conds) == 1 + 3 + 3
    assert g.conditions(("locked", "unlocked")) == [(), ("locked",), ("unlocked",)]   # one atom per feature
    assert len(door_grammar(k=3).conditions(true)) == 1 + 3 + 3 + 1


def test_a_grammar_refuses_unknown_or_repeated_atoms():
    g = door_grammar()
    with pytest.raises(ValueError):
        g.check(("locked", "nonsense"))
    with pytest.raises(ValueError):
        g.check(("locked", "unlocked"))                    # one atom per feature
    with pytest.raises(ValueError):
        Grammar([Atom("a", lambda s: True), Atom("a", lambda s: False)])
    assert g.check(["no_key", "locked"]) == ("locked", "no_key")    # sorted: one spelling per condition


# ───────────────────────────────────────────── prediction and ordering
def test_without_laws_the_base_predicts():
    ls = lawset()
    assert ls.predict({"locked": True, "key": False, "light": True}, "push") == "same"
    assert ls.predict({"locked": True, "key": False, "light": True}, "push", base=lambda s, a: "x") == "x"


def test_the_most_specific_matching_law_applies_first():
    ls = lawset()
    general = ls.add(Candidate("push", (), "opens"))
    special = ls.add(Candidate("push", ("locked", "no_key"), "same"))
    s_locked = {"locked": True, "key": False, "light": True}
    s_open = {"locked": False, "key": False, "light": True}
    assert ls.applied(s_locked, "push") is special and ls.predict(s_locked, "push") == "same"
    assert ls.applied(s_open, "push") is general and ls.predict(s_open, "push") == "opens"
    assert [law.id for law in ls.laws] == [special.id, general.id]
    assert ls.predict(s_open, "wait") == "same"                     # a law speaks for its own action only


def test_ties_in_specificity_go_to_the_earlier_law():
    ls = lawset()
    first = ls.add(Candidate("push", ("light",), "opens"))
    ls.add(Candidate("push", ("unlocked",), "same"))
    assert ls.applied({"locked": False, "key": False, "light": True}, "push") is first


# ───────────────────────────────────────────── mismatch → candidates → the test
def test_no_mismatch_no_candidates():
    ls = lawset()
    assert ls.mismatch({"locked": True, "key": False, "light": True}, "push", "same") == []


def test_a_mismatch_proposes_every_condition_true_in_the_state_with_the_observed_effect():
    ls = lawset()
    s = {"locked": False, "key": False, "light": True}
    cands = ls.mismatch(s, "push", "opens")
    assert all(c.action == "push" and c.effect == "opens" for c in cands)
    assert {c.condition for c in cands} == set(door_grammar().conditions(door_grammar().true(s, "push")))
    assert all(c.proposer == "enumerate" for c in cands)


def test_the_test_accepts_the_candidate_that_lowers_held_out_error_most():
    ls = lawset()
    buf = buffer_of("w1", "w2")
    s = {"locked": False, "key": False, "light": True}
    acc = ls.test(ls.mismatch(s, "push", "opens"), buf, exclude="w0")
    assert len(acc) == 1
    law = acc[0]
    assert isinstance(law, Law) and law.status == "active" and law.born == "w0"
    assert law.action == "push" and law.effect == "opens" and law.precision == 1.0
    # the most general condition that is right everywhere it applies on the buffer: "unlocked" (or with the key)
    assert law.condition in (("unlocked",), ("key",))
    assert law.support >= 2 and len(law.sources) == law.support


def test_the_test_never_sees_the_proposers_group():
    """Every held-out row comes from the proposing group: nothing can be accepted."""
    ls = lawset()
    buf = buffer_of("w0", repeat=3)
    s = {"locked": False, "key": False, "light": True}
    assert ls.test(ls.mismatch(s, "push", "opens"), buf, exclude="w0") == []
    assert ls.test(ls.mismatch(s, "push", "opens"), buf, exclude="w9") != []


def test_support_and_precision_bars():
    s = {"locked": False, "key": False, "light": True}
    strict = lawset(min_support=100)
    assert strict.test(strict.mismatch(s, "push", "opens"), buffer_of("w1"), exclude="w0") == []
    # "push opens" everywhere is right on 6 of 8 push rows: 0.75 < 0.9
    ls = lawset()
    assert ls.test([Candidate("push", (), "opens")], buffer_of("w1"), exclude="w0") == []
    loose = lawset(min_precision=0.7)
    assert [law.condition for law in loose.test([Candidate("push", (), "opens")], buffer_of("w1"), exclude="w0")] == [()]


def test_a_candidate_must_reduce_errors_net():
    """A candidate that only repeats what the set already predicts is not accepted (net 0)."""
    ls = lawset()
    buf = buffer_of("w1", "w2")
    assert ls.test([Candidate("wait", (), "same")], buf, exclude="w0") == []       # the base already says "same"
    support, hits, net = ls.score(Candidate("push", ("unlocked",), "opens"), buf, exclude="w0")
    assert (support, hits, net) == (8, 8, 8)


def test_the_cap_stops_acceptance():
    ls = lawset(cap=1)
    buf = buffer_of("w1", "w2")
    assert ls.test([Candidate("push", ("unlocked",), "opens")], buf, exclude="w0")
    assert ls.test([Candidate("push", ("locked", "key"), "opens")], buf, exclude="w0") == []
    assert ls.capped == 1 and len(ls.laws) == 1


# ───────────────────────────────────────────── retirement
def test_a_law_retires_when_its_held_out_precision_falls():
    ls = lawset(retire_below=0.8, retire_min=4)
    buf = buffer_of("w1", "w2")
    law = ls.test([Candidate("push", ("unlocked",), "opens")], buf, exclude="w0")[0]
    assert law.status == "active"
    jammed = lambda s, a: "same"                                          # the world changed: doors jam
    retired = []
    for t in rows("w3", actions=("push",), env=jammed):
        retired += ls.observe(t)
    assert retired == [law] and law.status == "retired" and law in ls.retired and law not in ls.laws
    assert "precision" in law.why


def test_observations_from_the_birth_group_do_not_count():
    ls = lawset(retire_below=0.8, retire_min=1)
    law = ls.test([Candidate("push", ("unlocked",), "opens")], buffer_of("w1", "w2"), exclude="w0")[0]
    jammed = lambda s, a: "same"
    for t in rows("w0", actions=("push",), env=jammed):
        assert ls.observe(t) == []
    assert law.status == "active" and law.n == 0


def test_manual_retirement():
    ls = lawset()
    law = ls.add(Candidate("push", (), "opens"))
    ls.retire(law, why="a person said so")
    assert law.status == "retired" and law.why == "a person said so" and ls.laws == []


# ───────────────────────────────────────────── the loop
def test_the_loop_learns_the_door_from_a_stream():
    ls = lawset()
    buf = buffer_of("v1", "v2", "v3")
    steps = ls.learn(rows("t1") + rows("t2"), buf)
    assert sum(st.mismatch for st in steps) >= 1 and sum(len(st.accepted) for st in steps) >= 1
    for s in states():
        for a in ("push", "wait"):
            assert ls.predict(s, a) == door(s, a), (s, a)


def test_remember_puts_each_transition_into_the_buffer_after_its_test():
    ls = lawset()
    buf = Buffer(door_grammar())
    ls.learn(rows("t1"), buf, remember=True)
    assert len(buf) == len(rows("t1")) and buf.groups() == ["t1"]


# ───────────────────────────────────────────── the buffer
def test_the_buffer_keys_rows_by_group():
    buf = buffer_of("a", "b")
    assert buf.groups() == ["a", "b"] and len(buf) == 32
    assert len(buf.rows("push")) == 16 and len(buf.rows("push", exclude="a")) == 8
    assert all(t.group == "b" for t in buf.rows("push", exclude="a"))


def test_a_transition_without_an_id_gets_a_content_id():
    t = Transition({"locked": True}, "push", "same", group="g")
    assert t.id and t.id == Transition({"locked": True}, "push", "same", group="g").id


# ───────────────────────────────────────────── proposers
def test_a_callable_proposer_proposes_and_the_test_decides():
    seen = {}

    def model(state, action, observed, true_atoms, grammar):
        seen["true"] = true_atoms
        return [["dark"], ["unlocked"], ["nonsense"], ["locked", "unlocked"]]   # two invalid ones

    p = CallableProposer(model, name="model:x")
    ls = lawset(proposer=p)
    s = {"locked": False, "key": False, "light": False}
    cands = ls.mismatch(s, "push", "opens")
    assert seen["true"] == ("dark", "no_key", "unlocked")
    assert [c.condition for c in cands] == [("dark",), ("unlocked",)] and p.dropped == 2
    acc = ls.test(cands, buffer_of("w1", "w2"), exclude="w0")
    assert [law.condition for law in acc] == [("unlocked",)] and acc[0].proposer == "model:x"


def test_a_random_proposer_draws_from_the_whole_grammar():
    ls = lawset(proposer=RandomProposer(seed=3, n=50))
    cands = ls.mismatch({"locked": False, "key": False, "light": True}, "push", "opens")
    assert len(cands) == 50 and all(c.proposer == "random" for c in cands)
    g = door_grammar()
    assert all(g.check(c.condition) == c.condition for c in cands)
    assert any(set(c.condition) - {"unlocked", "no_key", "light"} for c in cands)    # not only true atoms
    again = LawSet(door_grammar(), proposer=RandomProposer(seed=3, n=50))
    assert [c.condition for c in again.mismatch({"locked": False, "key": False, "light": True}, "push", "opens")] \
        == [c.condition for c in cands]


def test_the_enumerating_proposer_is_the_default():
    assert isinstance(lawset().proposer, Enumerate)


def test_effect_variants_and_apply():
    """The effect may be a change applied to the state (a counter + 1) rather than the observation itself."""
    g = Grammar([Atom("full", lambda s: s["n"] >= 3, feature="n"), Atom("room", lambda s: s["n"] < 3, feature="n")])
    ls = LawSet(g, base=lambda s, a: s["n"], apply=lambda eff, s: s["n"] + eff,
                effects=lambda s, observed: [observed - s["n"]], min_support=2)
    buf = Buffer(g)
    for i, n in enumerate([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
        buf.add(Transition({"n": n}, "add", min(n + 1, 3) if n < 3 else n, group=f"g{i % 2}"))
    acc = ls.test(ls.mismatch({"n": 0}, "add", 1), buf, exclude="g9")
    assert [(law.condition, law.effect) for law in acc] == [(("room",), 1)]
    assert ls.predict({"n": 2}, "add") == 3 and ls.predict({"n": 3}, "add") == 3


# ───────────────────────────────────────────── knowledge export
def _learned():
    ls = lawset(retire_below=0.8, retire_min=4)
    buf = buffer_of("w1", "w2")
    law = ls.test([Candidate("push", ("unlocked",), "opens")], buf, exclude="w0")[0]
    return ls, law


def test_every_accepted_law_becomes_a_knowledge_item_with_its_transitions():
    ls, law = _learned()
    ks = KnowledgeStore()
    ids = ls.to_knowledge(ks, {"env": "door"})
    item = ks.item(ids[law.id])
    assert item["kind"] == "rule" and item["body"]["kind"] == "law"
    assert item["body"]["action"] == "push" and item["body"]["if"] == ["unlocked"] and item["body"]["then"] == "opens"
    assert item["scope"] == {"env": "door"} and item["source"] == "outcome"
    assert ks.status(ids[law.id]) == "active"
    ev = item["evidence"][-1]                                          # one per assertion
    assert ev["support"] == law.support and ev["precision"] == law.precision and ev["born"] == "w0"
    assert sorted(ev["transitions"]) == sorted(law.sources) and len(ev["transitions"]) == 8
    assert ks.verify()


def test_a_retired_law_is_retired_in_the_store_and_the_journal_replays(tmp_path):
    ls, law = _learned()
    path = tmp_path / "knowledge.jsonl"
    ks = KnowledgeStore(JSONLStorage(path))
    ids = ls.to_knowledge(ks, {"env": "door"})
    assert ls.to_knowledge(ks, {"env": "door"}) == ids and len(ks) == 2       # again: nothing new is written
    for t in rows("w3", actions=("push",), env=lambda s, a: "same"):
        ls.observe(t)
    assert law.status == "retired"
    ls.to_knowledge(ks, {"env": "door"})
    assert ks.status(ids[law.id]) == "expired"
    assert any(r["op"] == "retire" and "precision" in r["why"] for r in ks.journal)
    assert ks.verify()
    again = KnowledgeStore(JSONLStorage(path))
    assert again.verify() and again.fingerprint() == ks.fingerprint() and again.status(ids[law.id]) == "expired"


def test_a_law_retired_before_export_is_written_then_retired():
    ls, law = _learned()
    ls.retire(law, why="superseded")
    ks = KnowledgeStore()
    ids = ls.to_knowledge(ks)
    assert ks.status(ids[law.id]) == "expired" and ks.verify()


# ───────────────────────────────────────────── the probe planner
def test_discriminate_finds_the_cheapest_safe_probe_where_two_law_sets_differ():
    a = lawset()
    a.add(Candidate("push", ("unlocked",), "opens"))
    b = lawset()
    b.add(Candidate("push", (), "opens"))
    probes = [(s, act) for s in states() for act in ("wait", "push")]
    p = discriminate(a, b, safe=lambda s, act: True, probes=probes)
    assert p is not None and p.action == "push" and p.state["locked"] and p.predictions == ("same", "opens")
    assert discriminate(a, b, safe=lambda s, act: act != "push", probes=probes) is None
    assert discriminate(a, a, safe=lambda s, act: True, probes=probes) is None


def test_discriminate_tries_read_only_actions_before_writes():
    a = lambda s, act: "x" if act in ("read", "write") else "y"
    b = lambda s, act: "z"
    probes = [({}, "write"), ({}, "read")]
    p = discriminate(a, b, safe=lambda s, act: True, probes=probes, read_only=lambda act: act == "read")
    assert p.action == "read"
    p = discriminate(a, b, safe=lambda s, act: True, probes=probes, read_only=lambda act: act == "read",
                     cost=lambda s, act: 0)
    assert p.action == "read"                                     # read-only first, cost second


def test_split_prefers_the_probe_that_halves_the_hypotheses():
    hyps = [lambda s, a, i=i: "denied" if s["x"] <= i else "ok" for i in range(4)]
    probes = [({"x": x}, "read") for x in range(5)]
    p = split(hyps, safe=lambda s, a: True, probes=probes)
    assert p.state == {"x": 2} and p.score == 2                     # 2 against 2


def test_a_lawset_offers_the_hypothesis_with_one_more_law():
    ls = lawset()
    h = ls.hypothesis(Candidate("push", ("unlocked",), "opens"))
    s = {"locked": False, "key": False, "light": True}
    assert h(s, "push") == "opens" and ls.predict(s, "push") == "same" and ls.laws == []
