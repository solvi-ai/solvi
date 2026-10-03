"""solvi.core.knowledge.memory: a memory of corrected cases next to a decider — nearest neighbours with an abstain threshold, only trusted
sources, a second signal that escalates (check) or answers where the decider escalated (answer), the cases it rests on in
the trace and the audit, its fingerprint in the part's, deterministic and replayable."""
import random

import pytest

from solvi import Catalog, System
from solvi.core.deciders import Facts
from solvi.core.knowledge.memory import CorrectionMemory, attach, words
from solvi.core.deciders.combine import Cascade
from solvi.core.store import JSONLStorage, UntrustedLabel
from test_decide import TASK, TEAMS, model, texts


def setup(noise=0.3, **kw):
    m = model(noise=noise)
    cat = Catalog()
    part = m.decision("team", TASK, "email", TEAMS, option_order="given", **kw)
    q = part.question(cat, "route")
    return m, cat, part, System(cat, [q])


def test_check_mode_escalates_when_similar_corrections_disagree_and_shows_the_cases():
    _, cat, part, s = setup()
    mem = attach(part)
    for i, t in enumerate(texts("billing", 4)):
        mem.add(t, "shipping", source="human", by="ann", stored_id=f"s{i}")
    r = s.ask({"email": texts("billing", 1, start=40)[0]})["route"]
    assert r.status == "abstain" and r.guard == "memory" and "say 'shipping'" in r.why and "'billing'" in r.why
    res = s.ask({"email": texts("billing", 1, start=40)[0]})
    au = res.audit("route")
    (m,) = au.memory
    assert m["action"] == "escalated" and m["proposal"] == "shipping" and m["n"] == 4
    assert {c["stored_id"] for c in m["neighbours"]} == {"s0", "s1", "s2", "s3"}
    assert all(c["source"] == "human" and c["by"] == "ann" for c in m["neighbours"])
    text = str(au)
    assert "memory" in text and "proposes 'shipping'" in text and "#s0" in text
    assert "память" in au.render(lang="ru") and "память исправлений не согласна" in au.render(lang="ru")
    assert res.trace.replay(cat)["ok"]
    # far from every stored case: the memory abstains and the decider answers
    r2 = s.ask({"email": texts("technical", 1)[0]})
    assert r2["route"].answer == "technical" and r2.audit("route").memory[0]["action"] == "abstained"
    assert "no corrected case within" in r2.audit("route").memory[0]["abstain"]


def test_agreement_is_recorded_and_the_answer_is_kept():
    _, _, part, s = setup()
    mem = attach(part)
    for t in texts("billing", 3):
        mem.add(t, "billing", source="outcome")
    res = s.ask({"email": texts("billing", 1, start=9)[0]})
    assert res["route"].answer == "billing" and res.audit("route").memory[0]["action"] == "agrees"


def test_only_trusted_sources_and_never_the_systems_own_answers():
    _, _, part, s = setup()
    mem = attach(part)
    for bad in ("model", "system", "self", "shadow", None):
        with pytest.raises(UntrustedLabel):
            mem.add(texts("billing", 1)[0], "billing", source=bad)
    assert len(mem) == 0


def test_learn_from_storage_takes_trusted_corrections_and_skips_the_rest(tmp_path):
    _, cat, part, _ = setup()
    store = JSONLStorage(tmp_path / "d.jsonl")
    s = System(cat, [part.question(Catalog(), "route")], storage=store)
    rid = s.ask({"email": texts("billing", 1)[0]}).stored_id
    s.teach("route", {"email": texts("billing", 1)[0]}, "shipping", by="ann", of=rid)
    s.teach("route", {"email": texts("billing", 1, start=1)[0]}, "shipping", label_source="outcome")
    store.save_correction("route", {"email": texts("billing", 1, start=2)[0]}, "nonsense")   # not an option: skipped
    store._append({"v": 1, "kind": "teach", "teach": "route", "init": {"email": "x"}, "answer": "billing",
                   "source": "model"})                                            # written around save_correction
    mem = attach(part)
    got = mem.learn_from(store, question="route")
    assert got["added"] == 2 and len(got["skipped"]) == 2
    assert any("UntrustedLabel" in why for _, why in got["skipped"])
    assert {c.source for c in mem.cases} == {"human", "outcome"} and {c.by for c in mem.cases} == {"ann", None}
    assert any(c.stored_id and store.record(c.stored_id)["kind"] == "teach" for c in mem.cases)
    # the stored decisions (the system's own answers) are never read as cases
    assert all(store.record(c.stored_id)["kind"] == "teach" for c in mem.cases)


def test_answer_mode_answers_where_the_decider_escalated_and_says_so():
    _, cat, part, s = setup(min_confidence=0.999)
    mem = attach(part, mode="answer")
    for t in texts("billing", 4):
        mem.add(t, "billing", source="human")
    res = s.ask({"email": texts("billing", 1, start=30)[0]})
    r = res["route"]
    assert r.status == "ok" and r.answer == "billing"
    m = res.audit("route").memory[0]
    assert m["action"] == "answered" and m["replaced"].startswith("confidence ") and "guarantee" in m
    assert "answered in place of the model" in str(res.audit("route"))
    assert res.trace.replay(cat)["ok"]
    # no stored case near: the decider's escalation stands
    assert s.ask({"email": texts("technical", 1)[0]})["route"].status == "abstain"


def test_inside_a_cascade_the_memory_only_checks():
    m1, m2 = model(noise=0.3), model(noise=0.3, version="2")
    small = m1.decision("team", TASK, "email", TEAMS, option_order="given")
    large = m2.decision("team", TASK, "email", TEAMS, option_order="given")
    mem = attach(small, mode="answer")
    for t in texts("billing", 4):
        mem.add(t, "shipping", source="human")
    casc = Cascade([small, large])
    d = casc.decide(texts("billing", 1, start=50)[0])
    first = d.extra["stages"][0]
    assert first["escalate"].startswith("memory of corrections disagrees") and d.extra["answered_by"] == 1
    assert first["memory"]["action"] == "escalated" and d.value == "billing"
    # the small model escalates by its own threshold; the memory agrees but does not answer inside a combination
    small.escalate_below = 0.999
    mem.remove([c.id for c in mem.cases])
    for t in texts("billing", 4):
        mem.add(t, "billing", source="human")
    d = casc.decide(texts("billing", 1, start=50)[0])
    first = d.extra["stages"][0]
    assert first["escalate"].startswith("confidence") and first["memory"]["action"] == "agrees (already escalated)"
    assert d.extra["answered_by"] == 1


def test_deterministic_order_independent_and_fingerprinted():
    _, _, part, _ = setup()
    rows = [(t, "billing") for t in texts("billing", 5)] + [(t, "shipping") for t in texts("shipping", 5)]
    a = CorrectionMemory(part)
    for t, y in rows:
        a.add(t, y, source="human")
    random.Random(3).shuffle(rows)
    b = CorrectionMemory(part)
    for t, y in rows:
        b.add(t, y, source="human")
    assert a.fingerprint() == b.fingerprint()
    x = texts("billing", 1, start=77)[0]
    assert a.propose(x).to_dict() == b.propose(x).to_dict() == a.propose(x).to_dict()
    fp0 = part.fingerprint()
    attach(part, a)
    fp1 = part.fingerprint()
    a.add(texts("shipping", 1, start=99)[0], "shipping", source="rule")
    assert len({fp0, fp1, part.fingerprint()}) == 3
    attach(part, False)
    assert part.fingerprint() == fp0


def test_a_memory_change_is_seen_by_replay():
    _, cat, part, s = setup()
    mem = attach(part)
    mem.add(texts("billing", 1)[0], "billing", source="human")
    res = s.ask({"email": texts("billing", 1, start=3)[0]})
    assert res.trace.replay(cat)["ok"]
    mem.add(texts("billing", 1, start=4)[0], "billing", source="human")
    rep = res.trace.replay(cat)
    assert not rep["ok"] and "model changed" in rep["mismatches"][0][2]


def test_disagreeing_neighbours_abstain():
    _, _, part, _ = setup(noise=0.0)
    mem = attach(part)
    same = texts("billing", 1)[0]
    mem.add(same, "billing", source="human", by="a")
    mem.add(same, "shipping", source="human", by="b")
    p = mem.propose(same)
    assert p.label is None and "similar cases disagree" in p.abstain


def test_calibrate_sets_the_abstain_threshold_leave_one_out():
    _, _, part, _ = setup()
    mem = attach(part)
    for team in TEAMS:
        for t in texts(team, 12):
            mem.add(t, team, source="human")
    got = mem.calibrate(max_risk=0.1)
    assert got["n"] == 36 and got["risk"] <= 0.1 and got["proposed"] > 0.8 and mem.guarantee["method"] == "crc-loo"
    # a memory whose labels are noise proposes nothing it cannot back
    noisy = CorrectionMemory(part)
    rng = random.Random(0)
    for team in TEAMS:
        for t in texts(team, 12):
            noisy.add(t, rng.choice(TEAMS), source="human")
    g = noisy.calibrate(max_risk=0.05)
    assert g["risk"] <= 0.05
    assert got["min_strength"] > -1e8 and got["radius"] == mem.radius and 0 <= got["nearest"]["min"] <= got["nearest"]["max"]
    assert "note" not in got


def test_calibrate_on_cases_that_are_all_out_of_each_others_reach_says_so(tmp_path):
    """No case has another within the radius: the leave-one-out run proposes nothing. The threshold used to come back as
    -1e9 (the mark of "no proposal") with a guarantee line, which let any later proposal through unchecked."""
    _, _, part, _ = setup()
    mem = CorrectionMemory(part, radius=1e-9)
    for team in TEAMS:
        for t in texts(team, 6):
            mem.add(t, team, source="human")
    got = mem.calibrate(max_risk=0.1)
    assert got["min_strength"] == float("inf") and mem.min_strength == float("inf") and got["proposed"] == 0.0
    assert "no stored case has another within the radius 1e-09" in got["note"] and got["nearest"]["min"] > 1e-9
    assert mem.propose(texts("billing", 1)[0]).label is None
    mem.save(tmp_path / "m.json")                                          # an infinite floor is saved and loaded
    again = CorrectionMemory(part, radius=1e-9)
    again.load(tmp_path / "m.json")
    assert again.min_strength == float("inf") and again.fingerprint() == mem.fingerprint()


def test_save_and_load_refuse_another_checkpoint(tmp_path):
    _, _, part, _ = setup()
    mem = attach(part, k=5, radius=0.2)
    for t in texts("billing", 3):
        mem.add(t, "billing", source="human", by="ann", time=1.0)
    mem.save(tmp_path / "m.json")
    again = CorrectionMemory(part).load(tmp_path / "m.json")
    assert again.fingerprint() == mem.fingerprint() and again.k == 5
    other = model(version="9").decision("team", TASK, "email", TEAMS, option_order="given")
    with pytest.raises(ValueError, match="checkpoint"):
        CorrectionMemory(other).load(tmp_path / "m.json")


def test_text_words_and_multi_label_and_facts():
    assert words("The Parcel parcel is lost") == words("lost parcel, the")
    m = model(noise=0.3)
    part = m.decision("teams", TASK, "email", TEAMS, multi=True, option_order="given")
    mem = attach(part, text=True)
    t = "I was charged twice and the app shows an error"
    mem.add(Facts(email=t), ["billing", "technical"], source="human")
    p = mem.propose(Facts(email=t))
    assert p.label == ["billing", "technical"] and p.neighbours[0]["distance"] == 0
    d = part.decide(t)
    assert d.value == ("billing", "technical") and d.extra["memory"]["action"] == "agrees"
    with pytest.raises(ValueError):
        attach(m.decision("where", "Where?", "email", kind="span"))


# --------------------------------------------------------------------------------------------------- fixes before 0.7
def test_calibrate_does_not_change_the_live_min_strength_while_it_runs():
    _, _, part, _ = setup()
    mem = attach(part, min_strength=2.5)
    for t in texts("billing", 20) + texts("technical", 20):
        mem.add(t, "billing" if "charged" in t else "technical")
    seen = []
    orig = mem._propose

    def spy(*a, **kw):                         # what a concurrent decision would read meanwhile
        seen.append(mem.min_strength)
        return orig(*a, **kw)
    mem._propose = spy
    mem.calibrate(max_risk=0.2)
    assert seen and set(seen) == {2.5}


def test_a_case_added_during_a_proposal_does_not_break_it():
    _, _, part, _ = setup()
    mem = attach(part, radius=0.9, min_strength=0.1, min_agreement=0.5)
    for t in texts("billing", 5):
        mem.add(t, "billing")
    orig, extra = mem._distances, iter(texts("technical", 50))

    def racing(*a, **kw):                      # another thread adds a case between the distances and the ranking
        d = orig(*a, **kw)
        mem.add(next(extra), "technical")
        return d
    mem._distances = racing
    p = mem.propose("I was charged twice, please refund order 99.")
    assert p.label == "billing" and len(p.neighbours) == 5


def test_leave_one_out_leaves_out_the_cases_twins():
    _, _, part, _ = setup()
    mem = attach(part, radius=0.05, min_strength=0.0, min_agreement=0.5)
    far = [texts("billing", 1)[0], texts("technical", 1)[0], texts("shipping", 1)[0]]
    for t, y in zip(far, ["billing", "technical", "shipping"]):
        mem.add(t, y, by="ann")
        mem.add(t, y, by="bob")               # the same correction stored twice: a twin, not independent evidence
    got = mem.calibrate(max_risk=0.3)
    assert got["proposed"] == 0.0                # each case's only neighbour is its twin: nothing left to propose


def test_calibrate_when_one_input_was_corrected_twice_says_so_instead_of_raising():
    """Every stored case has the same features: each is left out with its twins, so nothing is proposed and no case has
    a nearest other case. The note used to format those missing distances (TypeError), after the threshold and the
    guarantee had already been set."""
    _, _, part, _ = setup()
    mem = attach(part)
    t = texts("billing", 1)[0]
    mem.add(t, "shipping", stored_id="a")
    mem.add(t, "shipping", stored_id="b")
    got = mem.calibrate(max_risk=0.05)
    assert got["min_strength"] == float("inf") and got["proposed"] == 0.0
    assert got["nearest"] == {"min": None, "median": None, "max": None}
    assert "every stored case has the same features" in got["note"] and "radius" not in got["note"]
    assert mem.propose(t).label is None


def test_a_memory_file_of_another_question_is_refused(tmp_path):
    m, _, part, _ = setup()
    mem = attach(part)
    for i, t in enumerate(texts("billing", 3)):
        mem.add(t, "shipping", stored_id=f"s{i}")
    f = mem.save(tmp_path / "m.json")
    assert CorrectionMemory(part).load(f).fingerprint() == mem.fingerprint()     # the same question: as before
    product = m.decision("product", "Which product?", "email", ["phone", "laptop", "tablet"], option_order="given")
    with pytest.raises(ValueError, match="another question"):                    # as many options, the same checkpoint
        CorrectionMemory(product).load(f)
    urgent = m.decision("urgent", "Urgent?", "email", ["yes", "no"], option_order="given")
    with pytest.raises(ValueError, match="another question"):
        CorrectionMemory(urgent).load(f)
    # strict=False accepts another checkpoint and question, never a label the question cannot give
    with pytest.raises(ValueError, match="not an answer of this question"):
        CorrectionMemory(product).load(f, strict=False)
    queue = m.decision("queue", "Which queue?", "email", TEAMS, option_order="given")
    assert len(CorrectionMemory(queue).load(f, strict=False)) == 3
    four = m.decision("team4", TASK, "email", TEAMS + ["sales"], option_order="given")
    loose = attach(four, CorrectionMemory(four).load(f, strict=False))
    with pytest.raises(ValueError, match="stored for another question"):         # not a numpy broadcast error
        loose.propose(texts("billing", 1)[0])


def test_a_memory_given_answer_keeps_the_models_probability_as_its_confidence():
    """mode="answer": the answer used to carry the neighbours' agreement as its confidence — 1.00 from three agreeing
    cases, next to probs that say billing 0.79 — so min_confidence=0.99 let it through, and it was missing from its own
    conformal candidates."""
    m, _, _, _ = setup()
    part = m.decision("team", TASK, "email", TEAMS, option_order="given", min_confidence=0.9999)
    part.conformal([(t, k) for k in TEAMS for t in texts(k, 30)], coverage=0.9)
    mem = attach(part, mode="answer", min_strength=0.5)
    t = texts("billing", 1, start=70)[0]
    for i in range(3):
        mem.add(t, "shipping", source="human", stored_id=f"c{i}")           # people corrected it to shipping
    d = part.decide(t)
    rec = d.extra["memory"]
    assert d.value == "shipping" and rec["action"] == "answered" and rec["agreement"] == 1.0 and rec["model_answer"] == "billing"
    assert d.conf == pytest.approx(d.probs["shipping"]) and d.conf < 0.5     # the model's own probability of it
    assert "shipping" in d.extra["candidates"] and "billing" in d.extra["candidates"]
    cat = Catalog()
    r = System(cat, [part.question(cat, "route", min_confidence=0.99)]).ask({"email": t})["route"]
    assert r.status == "abstain" and r.guard == "low_confidence"             # a threshold meant for the model holds
    cat2 = Catalog()
    r2 = System(cat2, [part.question(cat2, "route")]).ask({"email": t})["route"]
    assert r2.status == "ok" and r2.answer == "shipping"


@pytest.mark.parametrize("bad, why", [({"mode": "overwrite"}, "mode must be one of"), ({"k": "seven"}, "k must be"),
                                      ({"radius": None}, "radius must be"), ({"text": "yes"}, "text must be"),
                                      ({"guarantee": "always"}, "guarantee is not a record")])
def test_a_memory_file_with_a_bad_setting_is_refused_and_the_memory_left_as_it_was(bad, why):
    """load_dict copied settings without the constructor's checks: "mode": "overwrite" or "k": "seven" loaded, and every
    later decision failed (TypeError: slice indices must be integers)."""
    _, _, part, _ = setup()
    mem = attach(part, k=5, radius=0.2)
    for t in texts("billing", 3):
        mem.add(t, "billing", source="human", by="ann", time=1.0)
    data = mem.to_dict()
    data["settings"] = {**data["settings"], **bad}
    fresh = CorrectionMemory(part, k=3)
    with pytest.raises(ValueError, match=why):
        fresh.load_dict(data)
    assert fresh.k == 3 and fresh.mode == "check" and len(fresh) == 0 and fresh.guarantee is None
