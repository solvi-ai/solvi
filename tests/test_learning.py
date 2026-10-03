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
            s.teach("route", {"email": email}, label(team), label_source=source, by="ann", of=res.stored_id)


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
        s.teach("route", {"email": texts("billing", 1)[0]}, "billing", label_source="model")
    with pytest.raises(UntrustedLabel):
        store.save_correction("route", {"email": texts("billing", 1)[0]}, "billing", label_source="system")
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
    part.act_guard([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], max_risk=0.3)
    old = dict(part.guarantee)
    loop = loop_of(s, gates={"max_change": 0.9, "min_calibration": 8})
    stream(s, 20)
    rep = loop.run()
    g = rep.gates["act_guard"]
    assert g["ok"] and g["questions"]["route"]["n"] == rep.questions["route"]["calibration"] >= 8
    assert rep.promoted and part.guarantee["n"] == rep.questions["route"]["calibration"] != old["n"]
    # too few calibration labels: the update cannot keep its promise, so it is rejected
    part2, s2, _ = build(tmp_path, "e.db")
    part2.act_guard([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], max_risk=0.3)
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


# --------------------------------------------------------------------------------------------------- fixes before 0.7
def test_the_candidate_is_gated_on_a_shadow_and_the_live_part_changes_only_on_promotion(tmp_path):
    part, s, store = build(tmp_path)
    loop = loop_of(s, gates={"max_change": 0.9})
    stream(s, 10)
    fp0 = part.fingerprint()
    seen = []
    gate = loop._gate_heldout

    def spy(before, after, n):                     # what a concurrent ask sees while the gates run
        seen.append((part.fingerprint(), part.adaptation, s.ask({"email": texts("billing", 1, 800)[0]},
                                                                store=False)["route"].answer))
        return gate(before, after, n)
    loop._gate_heldout = spy
    rep = loop.run()
    assert rep.promoted and seen and seen[0][0] == fp0 and seen[0][1] is None and seen[0][2] == "shipping"
    assert loop.fingerprint() == rep.fp_after and part.fingerprint() != fp0 and part.adaptation is not None
    assert s.ask({"email": texts("billing", 1, 800)[0]}, store=False)["route"].answer == "billing"
    # a rejected candidate never reaches the live part
    fp1 = part.fingerprint()
    stream(s, 10, start=100, label=ROTATE.get)
    seen.clear()
    rep = loop.run()
    assert rep.action == "rejected" and seen[0][0] == fp1 and part.fingerprint() == fp1


def test_the_memory_rung_is_promoted_onto_the_live_part(tmp_path):
    part, s, store = build(tmp_path)
    loop = loop_of(s, ladder={"fit_below": 5}, gates={"max_change": 0.9})
    stream(s, 10)
    rep = loop.run()
    assert rep.promoted and part.correction_memory is not None and part.correction_memory.part is part
    assert loop.fingerprint() == rep.fp_after and loop.current == rep.version


def test_conformal_sets_are_recalibrated_or_dropped_after_an_update(tmp_path):
    part, s, store = build(tmp_path)
    part.conformal([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], coverage=0.8)
    loop = loop_of(s, gates={"max_change": 0.9, "min_calibration": 8})
    stream(s, 20)
    rep = loop.run()
    c = rep.gates["act_guard"]["questions"]["route"]["conformal"]
    assert rep.promoted and c["dropped"] is False and c["n"] == rep.questions["route"]["calibration"]
    assert part.conformal_set["n"] == c["n"] and part.conformal_set["coverage"] == 0.8
    part2, s2, _ = build(tmp_path, "c.db")
    part2.conformal([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], coverage=0.8)
    loop2 = loop_of(s2, gates={"max_change": 0.9})              # min_calibration 30: too few to recalibrate
    stream(s2, 10)
    rep2 = loop2.run()
    assert rep2.promoted and rep2.gates["act_guard"]["questions"]["route"]["conformal"]["dropped"]
    assert part2.conformal_set is None and "conformal sets dropped" in rep2.gates["act_guard"]["why"]


def test_the_shadow_set_uses_the_labels_split(tmp_path):
    from solvi.learning import content_key
    part, s, store = build(tmp_path)
    loop = loop_of(s, gates={"max_change": 0.9})
    stream(s, 30)
    rows = loop._shadow_set()
    assert rows
    for row in rows:
        st = row[0] if isinstance(row, tuple) else row
        init = st.data["response"]["trace"]["init"]
        assert split_of(content_key("route", init), loop.holdout, loop.calibration) == "holdout"
    n_held = sum(split_of(content_key("route", s2.data["response"]["trace"]["init"])) == "holdout"
                 for s2 in store.iter() if (s2.data.get("response") or {}).get("trace"))
    assert len(rows) == n_held


def test_the_act_guard_gate_reports_a_recalibration_with_errors(tmp_path):
    from solvi.learning import Label
    part, s, store = build(tmp_path)
    part.act_guard([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], max_risk=0.5)
    loop = loop_of(s, gates={"min_calibration": 8})
    cal = [Label(f"l{t}{i}", "route", {"email": texts(t, 1, start=400 + i)[0]}, t, "human", split="calibration")
           for t in TEAMS for i in range(10)]
    g = loop._recalibrate({"route": {}}, {"route": {"calibration": cal}})
    x = g["questions"]["route"]
    assert g["ok"] and x["error"] > 0 and f"threshold {x['threshold']:.3f} on 30" in g["why"]


def test_open_answers_are_left_out_of_the_loop(tmp_path):
    part, s, _ = build(tmp_path)
    part.spec.kind = "span"                       # stand-in for a span question (the stub decider has no pointer)
    with pytest.raises(ValueError, match="closed-list"):
        loop_of(s, parts=["route"])
    with pytest.raises(ValueError, match="nothing to learn"):
        loop_of(s)                                # chosen automatically: open questions are skipped


class MarginHook:
    """An adapter hook that changes something in the part's fingerprint; with state / restore it can be carried over."""

    def __init__(self):
        self.parts = []

    def __call__(self, part, examples):
        self.parts.append(part)
        part.min_margin = 0.01
        return {"set": "min_margin", "n": len(examples)}

    def state(self, part):
        return {"min_margin": part.min_margin}

    def restore(self, part, state):
        part.min_margin = state["min_margin"]


def test_the_adapter_rung_runs_on_the_shadow_and_reaches_the_live_part_through_state_and_restore(tmp_path):
    part, s, store = build(tmp_path)
    hook = MarginHook()
    loop = loop_of(s, ladder={"fit_below": 3, "memory_below": 6, "adapter": hook}, gates={"max_change": 0.95})
    stream(s, 10)
    rep = loop.run()
    assert rep.promoted and rep.questions["route"]["rung"] == "adapter"
    assert rep.questions["route"]["applied"]["adapter"]["set"] == "min_margin"
    assert hook.parts and all(p is not part for p in hook.parts)        # the hook saw the shadow part only
    assert part.min_margin == 0.01 and part.correction_memory is not None and loop.current == 1
    loop.rollback(0)
    assert part.min_margin != 0.01 and part.correction_memory is None and part.adaptation is None and loop.current == 0


def test_a_promotion_that_cannot_be_carried_over_leaves_the_live_parts_as_they_were(tmp_path):
    """An adapter hook without state / restore changed the shadow part: the candidate's fingerprint cannot be reached on
    the live part. run() used to raise with the adaptation, thresholds and memory already copied onto the live part, no
    record of the update and no version in force."""
    part, s, store = build(tmp_path)

    def hook(p, examples):
        p.min_margin = 0.01
        return {"set": "min_margin"}
    loop = loop_of(s, ladder={"fit_below": 3, "memory_below": 6, "adapter": hook}, gates={"max_change": 0.95})
    stream(s, 10)
    fp0, base = part.fingerprint(), accuracy(s)
    with pytest.raises(RuntimeError, match="put back as they were"):
        loop.run()
    assert part.fingerprint() == fp0 and part.adaptation is None and part.correction_memory is None
    assert accuracy(s) == base and loop.current == 0                    # the answers did not change
    rec = loop.history()[-1]
    assert rec["action"] == "update" and not rec["promoted"] and rec["version"] is None
    assert rec["gates"]["promotion"] == {"ok": False, "restored": True, "why": rec["gates"]["promotion"]["why"]}
    assert "did not give its fingerprint" in rec["gates"]["promotion"]["why"]
    assert [v["version"] for v in loop.versions()] == [0] and store.verify()["ok"]


@pytest.mark.parametrize("kw, why", [
    ({"ladder": {"fit_bellow": 10}}, "unknown ladder settings: \\['fit_bellow'\\]"),
    ({"ladder": {"memory": {"mode": "overwrite"}}}, "mode must be one of"),
    ({"ladder": {"memory": {"kk": 3}}}, "unknown memory settings"),
    ({"holdout": 0.9, "calibration": 0.5}, "leave no label to train on"),
    ({"holdout": -1}, "holdout must be a share"),
    ({"gates": {"risk": 7}}, "gate risk must be a number strictly between 0 and 1"),
    ({"gates": {"max_change": 2}}, "gate max_change must be a share"),
])
def test_a_learning_setting_that_would_be_ignored_or_leave_nothing_to_train_is_refused(tmp_path, kw, why):
    """Only gate names were checked: a misspelt ladder key was ignored, holdout=0.9 with calibration=0.5 left no
    training label, and holdout=-1, a gate risk of 7 or a bad memory mode were accepted."""
    _, s, _ = build(tmp_path)
    with pytest.raises(ValueError, match=why):
        loop_of(s, **kw)


def test_a_combination_question_is_left_out_by_default_instead_of_making_learning_raise(tmp_path):
    from solvi.multi import Cascade
    m = DecideModel(FakeScorer(bias=BIAS, noise=0.3), meta={"format": "test", "temperature": 1.0})
    cat = Catalog()
    q = m.decision("team", TASK, "email", TEAMS).question(cat, "route")
    q2 = Cascade([m.decision("team", TASK, "email", TEAMS)], name="team2").question(cat, "route2")
    s = System(cat, [q, q2], storage=SQLiteStorage(tmp_path / "d.db"))
    assert list(loop_of(s).parts) == ["route"]
