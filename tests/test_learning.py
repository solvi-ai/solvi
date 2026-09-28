"""System.learning (experimental): labels only from trusted sources, the ladder, the gates (consistency, held-out, honesty,
act_guard, shadow size), promotion recorded in the changelog, rejection undone, rollback to any promoted version — with a
stand-in decider whose label bias corrections can fix."""
import warnings

import pytest

from solvi import Catalog, SQLiteStorage, System
from solvi.decide import DecideModel
from solvi.learning import ExperimentalWarning, Learning, split_of
from solvi.storage import UntrustedLabel
from test_decide import TASK, TEAMS, FakeScorer, texts

BIAS = {"shipping": 5.0}                     # the stand-in prefers "shipping": billing and technical emails go wrong
ROTATE = {"billing": "technical", "technical": "shipping", "shipping": "billing"}


def build(tmp_path, name="d.db"):
    m = DecideModel(FakeScorer(bias=BIAS, noise=0.3), meta={"format": "test", "temperature": 1.0})
    cat = Catalog()
    part = m.decision("team", TASK, "email", TEAMS, option_order="given")
    q = part.question(cat, "route")
    store = SQLiteStorage(tmp_path / name)
    return part, System(cat, [q], storage=store), store


def loop_of(s, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return s.learning(**kw)


def stream(s, n, start=0, label=lambda team: team, source="human"):
    """n emails per team: each asked (stored), then corrected."""
    for i in range(start, start + n):
        for team in TEAMS:
            email = texts(team, 1, start=i)[0]
            res = s.ask({"email": email})
            s.teach("route", {"email": email}, label(team), source=source, by="ann", of=res.stored_id)


def accuracy(s, start=500, n=10):
    got = [s.ask({"email": texts(t, 1, start=i)[0]}, store=False)["route"].answer == t
           for t in TEAMS for i in range(start, start + n)]
    return sum(got) / len(got)


def test_learning_is_experimental_and_off_until_asked(tmp_path):
    part, s, store = build(tmp_path)
    assert getattr(s, "_learning", None) is None
    s.teach("route", {"email": texts("billing", 1)[0]}, "billing")
    assert part.adaptation is not None                              # without a loop teach learns at once, as before
    part.reset()
    with pytest.warns(ExperimentalWarning):
        loop = s.learning()
    assert isinstance(loop, Learning) and loop.experimental
    s.teach("route", {"email": texts("billing", 1, start=1)[0]}, "billing")
    assert part.adaptation is None                                 # with it teach only stores: the gates decide
    loop.detach()
    assert s._learning is None


def test_a_stream_where_corrections_help_is_promoted_and_recorded(tmp_path):
    part, s, store = build(tmp_path)
    loop = loop_of(s, gates={"max_change": 0.9})
    stream(s, 10)
    base = accuracy(s)
    rep = loop.run()
    assert rep.promoted, str(rep)
    assert rep.questions["route"]["rung"] == "fit" and rep.version == 1 and loop.current == 1
    assert all(g["ok"] for g in rep.gates.values())
    assert rep.gates["heldout"]["gain"] > 0.3 and rep.gates["size"]["changed"] > 0
    assert accuracy(s) > base + 0.4
    vs = loop.versions()
    assert [v["version"] for v in vs] == [0, 1] and vs[0]["action"] == "baseline"
    rec = loop.history()[-1]
    assert rec["kind"] == "update" and rec["promoted"] and rec["state"]["route"]["adaptation"]["n_labelled"] > 0
    held = {lab.id for lab in loop.labels()["labels"] if lab.split == "holdout"}
    assert held and not held & set(rec["labels"]["route"])          # held-out labels are never trained on
    assert store.verify()["ok"]
    # nothing new: nothing to do
    assert loop.run().action == "none"
    # every decision records the part's fingerprint, which names the version
    res = s.ask({"email": texts("billing", 1, start=900)[0]})
    fp = next(r for r in res.trace.records if r.name == "answer:route").model["fp"]
    assert loop.version_of(fp) == [1]


def test_a_poisoned_batch_is_rejected_by_the_gates_and_undone(tmp_path):
    part, s, store = build(tmp_path)
    loop = loop_of(s, gates={"max_change": 0.9})
    stream(s, 10)
    assert loop.run().promoted
    fp1, acc1 = part.fingerprint(), accuracy(s)
    stream(s, 10, start=100, label=ROTATE.get)                      # every label wrong, all of them "human"
    rep = loop.run()
    assert rep.action == "rejected" and rep.current == 1
    assert not rep.gates["consistency"]["ok"] and rep.gates["consistency"]["share"] > 0.8
    assert part.fingerprint() == fp1 and accuracy(s) == acc1       # undone
    rec = loop.history()[-1]
    assert rec["promoted"] is False and rec["state"] is None and rec["gates"]["consistency"]["ok"] is False
    assert "✗ consistency" in str(rep)


def test_poison_without_history_is_caught_by_the_held_out_labels_and_an_honesty_set(tmp_path):
    part, s, store = build(tmp_path)
    clean = [{"name": f"c{i}", "state": {"email": texts(t, 1, start=700 + i)[0]}, "gold": {"route": t}}
             for t in TEAMS for i in range(4)]
    loop = loop_of(s, gates={"max_change": 0.9, "honesty": clean})
    stream(s, 10, label=ROTATE.get)                                 # a batch of wrong labels and nothing learned before
    rep = loop.run()
    assert rep.action == "rejected" and rep.gates["consistency"]["ok"]           # nothing earlier to contradict them
    assert not rep.gates["honesty"]["ok"] and rep.gates["honesty"]["honesty_set"]["regressions"]
    assert part.adaptation is None
    # labels that only repeat the model's bias teach nothing: no gain on the held-out labels, nothing promoted
    part3, s3, _ = build(tmp_path, "f.db")
    loop3 = loop_of(s3, gates={"max_change": 0.9})
    stream(s3, 10, label=lambda t: "shipping")
    rep3 = loop3.run()
    assert rep3.action == "rejected" and not rep3.gates["heldout"]["ok"] and rep3.gates["heldout"]["gain"] <= 0


def test_rollback_to_any_promoted_version_also_from_another_process(tmp_path):
    part, s, store = build(tmp_path)
    loop = loop_of(s, gates={"max_change": 0.9})
    fp0, acc0 = part.fingerprint(), accuracy(s)
    stream(s, 10)
    assert loop.run().promoted
    fp1, acc1 = part.fingerprint(), accuracy(s)
    assert loop.rollback(0) == 0 and loop.current == 0
    assert part.fingerprint() == fp0 and accuracy(s) == acc0
    loop.rollback(1)
    assert part.fingerprint() == fp1 and accuracy(s) == acc1
    assert [h["action"] for h in loop.history()] == ["baseline", "update", "rollback", "rollback"]
    with pytest.raises(KeyError):
        loop.rollback(7)
    # a new process: a fresh system on the same store restores version 1 from the changelog alone
    store.close()
    part2, s2, store2 = build(tmp_path)
    loop2 = loop_of(s2)
    assert loop2.current == 0 and part2.adaptation is None
    loop2.rollback(1)
    assert part2.fingerprint() == fp1 and loop2.current == 1 and store2.verify()["ok"]


def test_self_labels_are_impossible(tmp_path):
    part, s, store = build(tmp_path)
    loop = loop_of(s, gates={"max_change": 0.9})
    for t in TEAMS:                                                 # the system's own answers, stored
        for i in range(10):
            s.ask({"email": texts(t, 1, start=i)[0]})
    assert loop.labels()["labels"] == [] and loop.run().action == "none"
    with pytest.raises(UntrustedLabel):
        s.teach("route", {"email": texts("billing", 1)[0]}, "billing", source="model")
    with pytest.raises(UntrustedLabel):
        store.save_correction("route", {"email": texts("billing", 1)[0]}, "billing", source="system")
    for i in range(3):                                              # records written around save_correction
        store._append({"v": 1, "kind": "teach", "teach": "route", "init": {"email": texts("billing", 1, start=i)[0]},
                       "answer": "shipping", "source": "model"})
    got = loop.labels()
    assert got["labels"] == [] and len(got["rejected"]) == 3 and all("UntrustedLabel" in w for _, w in got["rejected"])
    assert loop.run().action == "none" and part.adaptation is None


def test_the_memory_rung_keeps_the_provenance_of_each_case(tmp_path):
    part, s, store = build(tmp_path)
    loop = loop_of(s, ladder={"fit_below": 5}, gates={"max_change": 0.9})
    stream(s, 10)
    rep = loop.run()
    assert rep.promoted and rep.questions["route"]["rung"] == "memory"
    mem = part.correction_memory
    assert mem is not None and len(mem) == rep.questions["route"]["train"]
    teach_ids = {c["id"] for c in store.corrections()}
    assert {c.stored_id for c in mem.cases} <= teach_ids and {c.source for c in mem.cases} == {"human"}
    res = s.ask({"email": texts("billing", 1, start=600)[0]})
    assert res.audit("route").memory[0]["action"] in ("agrees", "abstained")
    loop.rollback(0)
    assert part.correction_memory is None


def test_act_guard_is_recalibrated_on_fresh_labels(tmp_path):
    part, s, store = build(tmp_path)
    part.act_guard([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], risk=0.3)
    old = dict(part.guarantee)
    loop = loop_of(s, gates={"max_change": 0.9, "min_calibration": 8})
    stream(s, 20)
    rep = loop.run()
    g = rep.gates["act_guard"]
    assert g["ok"] and g["questions"]["route"]["n"] == rep.questions["route"]["calibration"] >= 8
    assert rep.promoted and part.guarantee["n"] == rep.questions["route"]["calibration"] != old["n"]
    # too few calibration labels: the update cannot keep its promise, so it is rejected
    part2, s2, _ = build(tmp_path, "e.db")
    part2.act_guard([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], risk=0.3)
    loop2 = loop_of(s2, gates={"max_change": 0.9})
    stream(s2, 10)
    rep2 = loop2.run()
    assert rep2.action == "rejected" and not rep2.gates["act_guard"]["ok"] and part2.guarantee == old


def test_too_large_an_update_is_rejected_by_the_size_limit(tmp_path):
    part, s, store = build(tmp_path)
    loop = loop_of(s)                                               # max_change 30%: this update moves two thirds
    stream(s, 10)
    rep = loop.run()
    assert rep.action == "rejected" and not rep.gates["size"]["ok"] and rep.gates["size"]["share"] > 0.3
    assert rep.gates["size"]["by_question"]["route"]["transitions"]


def test_split_is_stable():
    ids = [f"id{i}" for i in range(2000)]
    sp = [split_of(i) for i in ids]
    assert sp == [split_of(i) for i in ids]
    share = sp.count("holdout") / len(sp)
    assert 0.25 < share < 0.35 and 0.15 < sp.count("calibration") / len(sp) < 0.25
