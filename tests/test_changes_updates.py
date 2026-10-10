"""A labelled update as a candidate change (solvi.experimental.changes): `Change(artifact="update")` promoted through
`LearningTarget`, which runs the ladder (fit → memory of corrected cases → the adapter hook) and the gates the removed
learning loop had (consistency, held-out, honesty, act_guard, size) — the gates' numbers are the change's stage
measurements ("update:<gate>"), promotion and rollback go through the pipeline. Labels only from trusted sources, the
candidate built on a shadow of the system, every update recorded (kind "update") with the state to roll back to — with
a stand-in decider whose label bias corrections can fix."""
import warnings

import pytest

from solvi import Catalog, System
from solvi.core.deciders import DecideModel
from solvi.core.store import SQLiteStorage, UntrustedLabel
from test_decide import TASK, TEAMS, FakeScorer, texts

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import Change, LearningTarget, Pipeline
    from solvi.experimental.changes._updates import Label, content_key, split_of

BIAS = {"shipping": 5.0}                     # the stand-in prefers "shipping": billing and technical emails go wrong
ROTATE = {"billing": "technical", "technical": "shipping", "shipping": "billing"}


def build(tmp_path, name="d.db"):
    m = DecideModel(FakeScorer(bias=BIAS, noise=0.3), meta={"format": "test", "temperature": 1.0})
    cat = Catalog()
    part = m.decision("team", TASK, "email", TEAMS, option_order="given")
    q = part.question(cat, "route")
    store = SQLiteStorage(tmp_path / name)
    return part, System(cat, [q], storage=store), store


def pipe_of(tmp_path, target, journal=None):
    """A pipeline with no stages of its own: the update's gates are the target's."""
    return Pipeline([], journal if journal is not None else str(tmp_path / "changes.jsonl"), targets=[target],
                    approval={"update": "gate"})


def propose(pipe, source="person"):
    return pipe.propose(Change(source, artifact="update", proposer="ann", reason="the corrections of the week"))


def gates(res):
    """The update's gates as the change's stage verdicts: {gate: StageVerdict}."""
    return {v.stage.split(":", 1)[1]: v for v in res.verdicts if v.stage.startswith("update:")}


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


def test_updates_are_off_until_a_learning_target_is_made_and_system_learning_names_the_pipeline(tmp_path):
    part, s, store = build(tmp_path)
    assert getattr(s, "_learning", None) is None
    s.teach("route", {"email": texts("billing", 1)[0]}, "billing")
    assert part.adaptation is not None                              # without a target teach learns at once, as before
    part.reset()
    target = LearningTarget(s)
    assert target.artifact == "update" and list(target.parts) == ["route"]
    s.teach("route", {"email": texts("billing", 1, start=1)[0]}, "billing")
    assert part.adaptation is None                                 # with it teach only stores: the gates decide
    target.detach()
    assert s._learning is None
    with pytest.raises(AttributeError, match=r"System.learning\(\) was removed in 1.0: use solvi.experimental.changes"):
        s.learning()


def test_an_update_where_corrections_help_is_promoted_with_its_gates_as_stage_measurements(tmp_path):
    part, s, store = build(tmp_path)
    target = LearningTarget(s, gates={"max_change": 0.9})
    pipe = pipe_of(tmp_path, target, journal=store)                # one chain with the decisions
    stream(s, 10)
    base = accuracy(s)
    res = propose(pipe)
    assert res.promoted, str(res)
    assert res.version == 1 and pipe.current == 1 and res.ref["version"] == 1 and res.ref["before"] == 0
    g = gates(res)
    assert list(g) == ["ladder", "consistency", "act_guard", "heldout", "honesty", "size"]
    assert all(v.ok for v in g.values())
    assert g["ladder"].measured["questions"]["route"]["rung"] == "fit" == res.ref["questions"]["route"]["rung"]
    assert g["heldout"].measured["gain"] > 0.3 and g["size"].measured["changed"] > 0
    assert accuracy(s) > base + 0.4
    # the journal holds a verdict per gate, with its numbers
    recorded = {d["stage"]: d for d in pipe.history("verdict")}
    assert recorded["update:heldout"]["measured"]["gain"] == g["heldout"].measured["gain"]
    vs = target.versions()
    assert [v["version"] for v in vs] == [0, 1] and vs[0]["action"] == "baseline"
    rec = target.history()[-1]
    assert rec["kind"] == "update" and rec["promoted"] and rec["state"]["route"]["adaptation"]["n_labelled"] > 0
    held = {lab.id for lab in target.labels()["labels"] if lab.split == "holdout"}
    assert held and not held & set(rec["labels"]["route"])          # held-out labels are never trained on
    assert store.verify()["ok"]
    # nothing new: the change is refused, nothing recorded as an update
    again = propose(pipe)
    assert again.status == "rejected" and "nothing new to learn" in again.reason and pipe.current == 1
    # every decision records the part's fingerprint, which names the update's version
    asked = s.ask({"email": texts("billing", 1, start=900)[0]})
    fp = next(r for r in asked.trace.records if r.name == "answer:route").model["fp"]
    assert target.version_of(fp) == [1]


def test_a_poisoned_batch_is_refused_at_the_consistency_gate_and_undone(tmp_path):
    part, s, store = build(tmp_path)
    target = LearningTarget(s, gates={"max_change": 0.9})
    pipe = pipe_of(tmp_path, target)
    stream(s, 10)
    assert propose(pipe).promoted
    fp1, acc1 = part.fingerprint(), accuracy(s)
    stream(s, 10, start=100, label=ROTATE.get)                      # every label wrong, all of them "human"
    res = propose(pipe)
    assert res.status == "rejected" and res.stage == "update:consistency" and pipe.current == 1
    c = gates(res)["consistency"]
    assert c.decision == "refuse" and c.measured["share"] > 0.8 and "contradicted by earlier corrections" in c.reason
    assert part.fingerprint() == fp1 and accuracy(s) == acc1       # undone
    rec = target.history()[-1]
    assert rec["promoted"] is False and rec["state"] is None and rec["gates"]["consistency"]["ok"] is False
    assert pipe.history("rejected")[-1]["stage"] == "update:consistency"


def test_poison_without_history_is_caught_by_the_held_out_labels_and_an_honesty_set(tmp_path):
    part, s, store = build(tmp_path)
    clean = [{"name": f"c{i}", "state": {"email": texts(t, 1, start=700 + i)[0]}, "gold": {"route": t}}
             for t in TEAMS for i in range(4)]
    pipe = pipe_of(tmp_path, LearningTarget(s, gates={"max_change": 0.9, "honesty": clean}))
    stream(s, 10, label=ROTATE.get)                                 # a batch of wrong labels and nothing learned before
    res = propose(pipe)
    g = gates(res)
    assert res.status == "rejected" and g["consistency"].ok                     # nothing earlier to contradict them
    assert not g["honesty"].ok and g["honesty"].measured["honesty_set"]["regressions"]
    assert part.adaptation is None
    # labels that only repeat the model's bias teach nothing: no gain on the held-out labels, nothing promoted
    part3, s3, _ = build(tmp_path, "f.db")
    pipe3 = pipe_of(tmp_path, LearningTarget(s3, gates={"max_change": 0.9}), journal=str(tmp_path / "c3.jsonl"))
    stream(s3, 10, label=lambda t: "shipping")
    g3 = gates(propose(pipe3))
    assert not g3["heldout"].ok and g3["heldout"].measured["gain"] <= 0


def test_the_pipeline_rolls_an_update_back_and_a_recorded_version_is_restored_in_another_process(tmp_path):
    part, s, store = build(tmp_path)
    target = LearningTarget(s, gates={"max_change": 0.9})
    pipe = pipe_of(tmp_path, target)
    fp0, acc0 = part.fingerprint(), accuracy(s)
    stream(s, 10)
    assert propose(pipe).promoted
    fp1, acc1 = part.fingerprint(), accuracy(s)
    assert pipe.rollback(0) == [1] and pipe.current == 0 and target.current == 0
    assert part.fingerprint() == fp0 and accuracy(s) == acc0
    assert target.restore(1) == 1 and part.fingerprint() == fp1 and accuracy(s) == acc1
    assert [h["action"] for h in target.history()] == ["baseline", "update", "rollback", "rollback"]
    with pytest.raises(KeyError):
        target.restore(7)
    # a new process: a fresh system on the same store restores version 1 from the records alone
    store.close()
    part2, s2, store2 = build(tmp_path)
    target2 = LearningTarget(s2)
    assert target2.current == 0 and part2.adaptation is None
    target2.restore(1)
    assert part2.fingerprint() == fp1 and target2.current == 1 and store2.verify()["ok"]


def test_self_labels_are_impossible(tmp_path):
    part, s, store = build(tmp_path)
    target = LearningTarget(s, gates={"max_change": 0.9})
    pipe = pipe_of(tmp_path, target)
    for t in TEAMS:                                                 # the system's own answers, stored
        for i in range(10):
            s.ask({"email": texts(t, 1, start=i)[0]})
    assert target.labels()["labels"] == [] and propose(pipe).status == "rejected"
    with pytest.raises(UntrustedLabel):
        s.teach("route", {"email": texts("billing", 1)[0]}, "billing", label_source="model")
    with pytest.raises(UntrustedLabel):
        store.save_correction("route", {"email": texts("billing", 1)[0]}, "billing", label_source="system")
    for i in range(3):                                              # records written around save_correction
        store._append({"v": 1, "kind": "teach", "teach": "route", "init": {"email": texts("billing", 1, start=i)[0]},
                       "answer": "shipping", "source": "model"})
    got = target.labels()
    assert got["labels"] == [] and len(got["rejected"]) == 3 and all("UntrustedLabel" in w for _, w in got["rejected"])
    assert "nothing new to learn" in propose(pipe).reason and part.adaptation is None


def test_the_memory_rung_keeps_the_provenance_of_each_case(tmp_path):
    part, s, store = build(tmp_path)
    pipe = pipe_of(tmp_path, LearningTarget(s, ladder={"fit_below": 5}, gates={"max_change": 0.9}))
    stream(s, 10)
    res = propose(pipe)
    assert res.promoted and res.ref["questions"]["route"]["rung"] == "memory"
    mem = part.correction_memory
    assert mem is not None and len(mem) == res.ref["questions"]["route"]["train"]
    teach_ids = {c["id"] for c in store.corrections()}
    assert {c.stored_id for c in mem.cases} <= teach_ids and {c.source for c in mem.cases} == {"human"}
    asked = s.ask({"email": texts("billing", 1, start=600)[0]})
    assert asked.audit("route").memory[0]["action"] in ("agrees", "abstained")
    pipe.rollback(0)
    assert part.correction_memory is None


def test_act_guard_is_recalibrated_on_fresh_labels(tmp_path):
    part, s, store = build(tmp_path)
    part.act_guard([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], max_risk=0.3)
    old = dict(part.guarantee)
    res = propose(pipe_of(tmp_path, LearningTarget(s, gates={"max_change": 0.9, "min_calibration": 8})))
    assert res.status == "rejected" and "nothing new to learn" in res.reason and not gates(res)   # no labels: no gate
    stream(s, 20)
    res = propose(pipe_of(tmp_path, LearningTarget(s, gates={"max_change": 0.9, "min_calibration": 8}),
                          journal=str(tmp_path / "c2.jsonl")))
    g = gates(res)["act_guard"]
    n_cal = res.ref["questions"]["route"]["calibration"]
    assert g.ok and g.measured["questions"]["route"]["n"] == n_cal >= 8
    assert res.promoted and part.guarantee["n"] == n_cal != old["n"]
    # too few calibration labels: the update cannot keep its promise, so it is refused at that gate
    part2, s2, _ = build(tmp_path, "e.db")
    part2.act_guard([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], max_risk=0.3)
    pipe2 = pipe_of(tmp_path, LearningTarget(s2, gates={"max_change": 0.9}), journal=str(tmp_path / "c3.jsonl"))
    stream(s2, 10)
    res2 = propose(pipe2)
    assert res2.status == "rejected" and not gates(res2)["act_guard"].ok and part2.guarantee == old


def test_too_large_an_update_is_refused_at_the_size_gate(tmp_path):
    part, s, store = build(tmp_path)
    pipe = pipe_of(tmp_path, LearningTarget(s))                     # max_change 30%: this update moves two thirds
    stream(s, 10)
    res = propose(pipe)
    size = gates(res)["size"]
    assert res.status == "rejected" and not size.ok and size.measured["share"] > 0.3
    assert size.measured["by_question"]["route"]["transitions"]


def test_split_is_stable():
    ids = [f"id{i}" for i in range(2000)]
    sp = [split_of(i) for i in ids]
    assert sp == [split_of(i) for i in ids]
    share = sp.count("holdout") / len(sp)
    assert 0.25 < share < 0.35 and 0.15 < sp.count("calibration") / len(sp) < 0.25


def test_the_candidate_is_gated_on_a_shadow_and_the_live_part_changes_only_on_promotion(tmp_path):
    part, s, store = build(tmp_path)
    target = LearningTarget(s, gates={"max_change": 0.9})
    pipe = pipe_of(tmp_path, target)
    stream(s, 10)
    fp0 = part.fingerprint()
    seen = []
    gate = target.updates._gate_heldout

    def spy(before, after, n):                     # what a concurrent ask sees while the gates run
        seen.append((part.fingerprint(), part.adaptation, s.ask({"email": texts("billing", 1, 800)[0]},
                                                                store=False)["route"].answer))
        return gate(before, after, n)
    target.updates._gate_heldout = spy
    res = propose(pipe)
    assert res.promoted and seen and seen[0][0] == fp0 and seen[0][1] is None and seen[0][2] == "shipping"
    assert target.fingerprint() == res.ref["fp_after"] and part.fingerprint() != fp0 and part.adaptation is not None
    assert s.ask({"email": texts("billing", 1, 800)[0]}, store=False)["route"].answer == "billing"
    # a refused candidate never reaches the live part
    fp1 = part.fingerprint()
    stream(s, 10, start=100, label=ROTATE.get)
    seen.clear()
    res = propose(pipe)
    assert res.status == "rejected" and seen[0][0] == fp1 and part.fingerprint() == fp1


def test_the_memory_rung_is_promoted_onto_the_live_part(tmp_path):
    part, s, store = build(tmp_path)
    target = LearningTarget(s, ladder={"fit_below": 5}, gates={"max_change": 0.9})
    pipe = pipe_of(tmp_path, target)
    stream(s, 10)
    res = propose(pipe)
    assert res.promoted and part.correction_memory is not None and part.correction_memory.part is part
    assert target.fingerprint() == res.ref["fp_after"] and target.current == res.ref["version"]


def test_conformal_sets_are_recalibrated_or_dropped_after_an_update(tmp_path):
    part, s, store = build(tmp_path)
    part.conformal([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], coverage=0.8)
    pipe = pipe_of(tmp_path, LearningTarget(s, gates={"max_change": 0.9, "min_calibration": 8}))
    stream(s, 20)
    res = propose(pipe)
    c = gates(res)["act_guard"].measured["questions"]["route"]["conformal"]
    assert res.promoted and c["dropped"] is False and c["n"] == res.ref["questions"]["route"]["calibration"]
    assert part.conformal_set["n"] == c["n"] and part.conformal_set["coverage"] == 0.8
    part2, s2, _ = build(tmp_path, "c.db")
    part2.conformal([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], coverage=0.8)
    pipe2 = pipe_of(tmp_path, LearningTarget(s2, gates={"max_change": 0.9}), journal=str(tmp_path / "c2.jsonl"))
    stream(s2, 10)                                                  # min_calibration 30: too few to recalibrate
    res2 = propose(pipe2)
    ag = gates(res2)["act_guard"]
    assert res2.promoted and ag.measured["questions"]["route"]["conformal"]["dropped"]
    assert part2.conformal_set is None and "conformal sets dropped" in ag.reason


def test_the_shadow_set_uses_the_labels_split(tmp_path):
    part, s, store = build(tmp_path)
    target = LearningTarget(s, gates={"max_change": 0.9})
    stream(s, 30)
    rows = target.updates._shadow_set()
    assert rows
    for row in rows:
        st = row[0] if isinstance(row, tuple) else row
        init = st.data["response"]["trace"]["init"]
        assert split_of(content_key("route", init), target.updates.holdout, target.updates.calibration) == "holdout"
    n_held = sum(split_of(content_key("route", s2.data["response"]["trace"]["init"])) == "holdout"
                 for s2 in store.iter() if (s2.data.get("response") or {}).get("trace"))
    assert len(rows) == n_held


def test_the_act_guard_gate_reports_a_recalibration_with_errors(tmp_path):
    part, s, store = build(tmp_path)
    part.act_guard([(texts(t, 1, start=300 + i)[0], t) for t in TEAMS for i in range(10)], max_risk=0.5)
    target = LearningTarget(s, gates={"min_calibration": 8})
    cal = [Label(f"l{t}{i}", "route", {"email": texts(t, 1, start=400 + i)[0]}, t, "human", split="calibration")
           for t in TEAMS for i in range(10)]
    g = target.updates._recalibrate({"route": {}}, {"route": {"calibration": cal}})
    x = g["questions"]["route"]
    assert g["ok"] and x["error"] > 0 and f"threshold {x['threshold']:.3f} on 30" in g["why"]


def test_open_answers_are_left_out(tmp_path):
    part, s, _ = build(tmp_path)
    part.spec.kind = "span"                       # stand-in for a span question (the stub decider has no pointer)
    with pytest.raises(ValueError, match="closed-list"):
        LearningTarget(s, parts=["route"])
    with pytest.raises(ValueError, match="nothing to learn"):
        LearningTarget(s)                         # chosen automatically: open questions are skipped


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
    target = LearningTarget(s, ladder={"fit_below": 3, "memory_below": 6, "adapter": hook}, gates={"max_change": 0.95})
    pipe = pipe_of(tmp_path, target)
    stream(s, 10)
    res = propose(pipe)
    assert res.promoted and res.ref["questions"]["route"]["rung"] == "adapter"
    assert res.ref["questions"]["route"]["applied"]["adapter"]["set"] == "min_margin"
    assert hook.parts and all(p is not part for p in hook.parts)        # the hook saw the shadow part only
    assert part.min_margin == 0.01 and part.correction_memory is not None and target.current == 1
    pipe.rollback(0)
    assert part.min_margin != 0.01 and part.correction_memory is None and part.adaptation is None and target.current == 0


def test_a_promotion_that_cannot_be_carried_over_is_refused_and_leaves_the_live_parts_as_they_were(tmp_path):
    """An adapter hook without state / restore changed the shadow part: the candidate's fingerprint cannot be reached on
    the live part. The live parts are put back and the change is refused at the "update:promotion" stage."""
    part, s, store = build(tmp_path)

    def hook(p, examples):
        p.min_margin = 0.01
        return {"set": "min_margin"}
    target = LearningTarget(s, ladder={"fit_below": 3, "memory_below": 6, "adapter": hook}, gates={"max_change": 0.95})
    pipe = pipe_of(tmp_path, target)
    stream(s, 10)
    fp0, base = part.fingerprint(), accuracy(s)
    res = propose(pipe)
    assert res.status == "rejected" and res.stage == "update:promotion" and "put back as they were" in res.reason
    assert part.fingerprint() == fp0 and part.adaptation is None and part.correction_memory is None
    assert accuracy(s) == base and target.current == 0 and pipe.current == 0     # the answers did not change
    rec = target.history()[-1]
    assert rec["action"] == "update" and not rec["promoted"] and rec["version"] is None
    assert rec["gates"]["promotion"] == {"ok": False, "restored": True, "why": rec["gates"]["promotion"]["why"]}
    assert "did not give its fingerprint" in rec["gates"]["promotion"]["why"]
    assert [v["version"] for v in target.versions()] == [0] and store.verify()["ok"]


@pytest.mark.parametrize("kw, why", [
    ({"ladder": {"fit_bellow": 10}}, "unknown ladder settings: \\['fit_bellow'\\]"),
    ({"ladder": {"memory": {"mode": "overwrite"}}}, "mode must be one of"),
    ({"ladder": {"memory": {"kk": 3}}}, "unknown memory settings"),
    ({"holdout": 0.9, "calibration": 0.5}, "leave no label to train on"),
    ({"holdout": -1}, "holdout must be a share"),
    ({"gates": {"risk": 7}}, "gate risk must be a number strictly between 0 and 1"),
    ({"gates": {"max_change": 2}}, "gate max_change must be a share"),
])
def test_a_setting_that_would_be_ignored_or_leave_nothing_to_train_is_refused(tmp_path, kw, why):
    _, s, _ = build(tmp_path)
    with pytest.raises(ValueError, match=why):
        LearningTarget(s, **kw)


def test_a_combination_question_is_left_out_by_default(tmp_path):
    from solvi.core.deciders.combine import Cascade
    m = DecideModel(FakeScorer(bias=BIAS, noise=0.3), meta={"format": "test", "temperature": 1.0})
    cat = Catalog()
    q = m.decision("team", TASK, "email", TEAMS).question(cat, "route")
    q2 = Cascade([m.decision("team", TASK, "email", TEAMS)], name="team2").question(cat, "route2")
    s = System(cat, [q, q2], storage=SQLiteStorage(tmp_path / "d.db"))
    assert list(LearningTarget(s).parts) == ["route"]


def test_decisions_made_while_a_target_is_attached_record_the_experimental_piece(tmp_path):
    part, s, store = build(tmp_path)
    LearningTarget(s)
    s.ask({"email": texts("billing", 1)[0]})
    rec = list(store.iter())[-1].data
    assert rec["meta"]["experimental"] == ["changes"]
