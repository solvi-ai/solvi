"""solvi.memory: a memory of corrected cases next to a decider — nearest neighbours with an abstain threshold, only trusted
sources, a second signal that escalates (check) or answers where the decider escalated (answer), the cases it rests on in
the trace and the audit, its fingerprint in the part's, deterministic and replayable."""
import random

import pytest

from solvi import Catalog, System
from solvi.decide import Facts
from solvi.memory import CorrectionMemory, words
from solvi.multi import Cascade
from solvi.storage import JSONLStorage, UntrustedLabel
from test_decide import TASK, TEAMS, model, texts


def setup(noise=0.3, **kw):
    m = model(noise=noise)
    cat = Catalog()
    part = m.decision("team", TASK, "email", TEAMS, option_order="given", **kw)
    q = part.question(cat, "route")
    return m, cat, part, System(cat, [q])


def test_check_mode_escalates_when_similar_corrections_disagree_and_shows_the_cases():
    _, cat, part, s = setup()
    mem = part.memory()
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
    mem = part.memory()
    for t in texts("billing", 3):
        mem.add(t, "billing", source="outcome")
    res = s.ask({"email": texts("billing", 1, start=9)[0]})
    assert res["route"].answer == "billing" and res.audit("route").memory[0]["action"] == "agrees"


def test_only_trusted_sources_and_never_the_systems_own_answers():
    _, _, part, s = setup()
    mem = part.memory()
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
    s.teach("route", {"email": texts("billing", 1, start=1)[0]}, "shipping", source="outcome")
    store.save_correction("route", {"email": texts("billing", 1, start=2)[0]}, "nonsense")   # not an option: skipped
    store._append({"v": 1, "kind": "teach", "teach": "route", "init": {"email": "x"}, "answer": "billing",
                   "source": "model"})                                            # written around save_correction
    mem = part.memory()
    got = mem.learn_from(store, question="route")
    assert got["added"] == 2 and len(got["skipped"]) == 2
    assert any("UntrustedLabel" in why for _, why in got["skipped"])
    assert {c.source for c in mem.cases} == {"human", "outcome"} and {c.by for c in mem.cases} == {"ann", None}
    assert any(c.stored_id and store.record(c.stored_id)["kind"] == "teach" for c in mem.cases)
    # the stored decisions (the system's own answers) are never read as cases
    assert all(store.record(c.stored_id)["kind"] == "teach" for c in mem.cases)


def test_answer_mode_answers_where_the_decider_escalated_and_says_so():
    _, cat, part, s = setup(escalate_below=0.999)
    mem = part.memory(mode="answer")
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
    mem = small.memory(mode="answer")
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
    part.memory(a)
    fp1 = part.fingerprint()
    a.add(texts("shipping", 1, start=99)[0], "shipping", source="rule")
    assert len({fp0, fp1, part.fingerprint()}) == 3
    part.memory(False)
    assert part.fingerprint() == fp0


def test_a_memory_change_is_seen_by_replay():
    _, cat, part, s = setup()
    mem = part.memory()
    mem.add(texts("billing", 1)[0], "billing", source="human")
    res = s.ask({"email": texts("billing", 1, start=3)[0]})
    assert res.trace.replay(cat)["ok"]
    mem.add(texts("billing", 1, start=4)[0], "billing", source="human")
    rep = res.trace.replay(cat)
    assert not rep["ok"] and "model changed" in rep["mismatches"][0][2]


def test_disagreeing_neighbours_abstain():
    _, _, part, _ = setup(noise=0.0)
    mem = part.memory()
    same = texts("billing", 1)[0]
    mem.add(same, "billing", source="human", by="a")
    mem.add(same, "shipping", source="human", by="b")
    p = mem.propose(same)
    assert p.label is None and "similar cases disagree" in p.abstain


def test_calibrate_sets_the_abstain_threshold_leave_one_out():
    _, _, part, _ = setup()
    mem = part.memory()
    for team in TEAMS:
        for t in texts(team, 12):
            mem.add(t, team, source="human")
    got = mem.calibrate(risk=0.1)
    assert got["n"] == 36 and got["risk"] <= 0.1 and got["proposed"] > 0.8 and mem.guarantee["method"] == "crc-loo"
    # a memory whose labels are noise proposes nothing it cannot back
    noisy = CorrectionMemory(part)
    rng = random.Random(0)
    for team in TEAMS:
        for t in texts(team, 12):
            noisy.add(t, rng.choice(TEAMS), source="human")
    g = noisy.calibrate(risk=0.05)
    assert g["risk"] <= 0.05


def test_save_and_load_refuse_another_checkpoint(tmp_path):
    _, _, part, _ = setup()
    mem = part.memory(k=5, radius=0.2)
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
    mem = part.memory(text=True)
    t = "I was charged twice and the app shows an error"
    mem.add(Facts(email=t), ["billing", "technical"], source="human")
    p = mem.propose(Facts(email=t))
    assert p.label == ["billing", "technical"] and p.neighbours[0]["distance"] == 0
    d = part.decide(t)
    assert d.value == ("billing", "technical") and d.extra["memory"]["action"] == "agrees"
    with pytest.raises(ValueError):
        m.decision("where", "Where?", "email", kind="span").memory()
