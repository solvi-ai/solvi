"""solvi.experimental.changes.RefusalRegression: refusal is the first casualty of self-change (a chain of model
merges has been reported to keep its skills and lose much of its refusal), so every change whose target can answer is
checked on a held should-abstain set — the share refused before and after — and refused when it answers what was
refused before. The stage is not switchable: the pipeline inserts it for such a change whether or not it is among the
stages, and journals its measurement for every one."""
from __future__ import annotations

import inspect
import warnings

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.knowledge import KnowledgeStore

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import (Change, Confirmation, LearningTarget, Pipeline, RefusalRegression,
                                            StageVerdict, VersionsTarget, is_refusal)


class Threshold:
    """A target that answers: a score threshold below which the system hands over (None). Its update payload is the
    new threshold."""
    artifact = "update"

    def __init__(self, t=0.5):
        self.t = t

    def answerer(self, change, ctx):
        t = self.t if change is None else change.payload["t"]
        return lambda x: None if x["score"] < t else "yes"

    def promote(self, change, measured):
        before, self.t = self.t, change.payload["t"]
        return {"t": self.t, "before": before}

    def rollback(self, ref, by=None, why=None):
        self.t = ref["before"]


OUTSIDE = [{"id": f"o{i}", "score": s} for i, s in enumerate((0.1, 0.2, 0.3, 0.4))]


def move(t):
    return Change("outcome", artifact="update", payload={"t": t, "fingerprint": f"t={t}"}, proposer="self",
                  reason=f"threshold to {t}")


def pipe_of(tmp_path, target, stages=(), name="c.jsonl", **kw):
    return Pipeline(list(stages), str(tmp_path / name), targets=[target], approval={"update": "gate"}, **kw)


def rr_records(pipe):
    return [d for d in pipe.history("verdict") if d["stage"] == "refusal_regression"]


def test_a_change_that_answers_the_outside_set_is_refused_with_the_set_and_the_counts(tmp_path):
    target = Threshold(0.5)
    pipe = pipe_of(tmp_path, target, refusals=OUTSIDE)
    res = pipe.propose(move(0.05))                   # answers every outside example
    assert res.status == "rejected" and res.stage == "refusal_regression"
    assert "should-abstain" in res.reason and "4 of 4" in res.reason and "pipeline" in res.reason
    m = next(v for v in res.verdicts if v.stage == "refusal_regression").measured
    assert m["n"] == 4 and m["refused_before"] == 4 and m["refused_after"] == 0 and m["lost"] == 4
    assert m["share_before"] == 1.0 and m["share_after"] == 0.0 and m["sources"] == {"pipeline": 4}
    assert m["lost_examples"][:2] == ["o0", "o1"]
    assert target.t == 0.5                          # nothing promoted
    # one lost refusal is a regression too (tolerance 0 by default)
    assert pipe.propose(move(0.35)).stage == "refusal_regression"
    # more refusals is not a regression
    ok = pipe.propose(move(0.6))
    assert ok.promoted and target.t == 0.6


def test_the_stage_is_inserted_for_an_answering_target_and_journaled_for_every_change(tmp_path):
    target = Threshold(0.5)
    pipe = pipe_of(tmp_path, target, refusals=OUTSIDE)            # constructed without the stage
    assert not any(isinstance(s, RefusalRegression) for s in pipe.stages)
    results = [pipe.propose(move(t)) for t in (0.05, 0.6, 0.7, 0.2)]
    recs = rr_records(pipe)
    assert len(recs) == len(results) == 4
    assert all(r["measured"]["n"] == 4 and "share_after" in r["measured"] for r in recs)
    assert [r.status for r in results] == ["rejected", "promoted", "promoted", "rejected"]
    # it runs before the approval stage (a person is not asked about a change it refuses)
    order = [v.stage for v in pipe.propose(move(0.8)).verdicts]
    assert order.index("refusal_regression") < order.index("approval")
    # the pipeline offers no switch to remove it
    params = set(inspect.signature(Pipeline).parameters)
    assert params == {"stages", "storage", "store", "approval", "targets", "refusals", "answer_of",
                      "raise_stage_errors", "order", "costs"}


def test_its_refusal_is_final_even_where_the_other_stages_are_advisory(tmp_path):
    target = Threshold(0.5)
    pipe = Pipeline([Confirmation(lambda c, x: False)], str(tmp_path / "c.jsonl"), targets=[target],
                    approval={"update": "person"}, refusals=OUTSIDE)
    res = pipe.propose(move(0.05))
    assert res.status == "rejected" and res.stage == "refusal_regression"
    adv = {d["stage"]: d["advisory"] for d in pipe.history("verdict")}
    assert adv == {"confirmation": True, "refusal_regression": False}


def test_tolerances_and_a_stage_of_ones_own_in_its_place(tmp_path):
    target = Threshold(0.5)
    own = RefusalRegression(count_tolerance=1)
    pipe = pipe_of(tmp_path, target, stages=[own], refusals=OUTSIDE)
    assert pipe.propose(move(0.35)).promoted                      # one lost refusal within the count tolerance
    assert pipe.propose(move(0.15)).stage == "refusal_regression"  # two lost
    share = pipe_of(tmp_path, Threshold(0.5), stages=[RefusalRegression(tolerance=0.5)], name="c2.jsonl",
                    refusals=OUTSIDE)
    assert share.propose(move(0.25)).promoted                     # 2 of 4 within half
    with pytest.raises(ValueError):
        RefusalRegression(tolerance=1.5)


def test_examples_kept_by_other_stages_and_by_the_stage_join_the_set(tmp_path):
    target = Threshold(0.5)
    conf = Confirmation(lambda c, x: True, refusals=OUTSIDE[:1])
    pipe = pipe_of(tmp_path, target, stages=[conf, RefusalRegression(refusals=OUTSIDE[1:2])], refusals=OUTSIDE[2:])
    res = pipe.propose(move(0.05))
    m = next(v for v in res.verdicts if v.stage == "refusal_regression").measured
    assert m["sources"] == {"pipeline": 2, "confirmation": 1, "refusal_regression": 1} and m["n"] == 4
    # a set given as a callable is read per change
    p2 = pipe_of(tmp_path, Threshold(0.5), name="c2.jsonl", refusals=lambda change, ctx: OUTSIDE[:2])
    assert p2.propose(move(0.05)).verdicts[-1].stage == "refusal_regression"


def test_without_a_should_abstain_set_the_change_is_held_with_the_reason(tmp_path):
    target = Threshold(0.5)
    pipe = pipe_of(tmp_path, target)
    res = pipe.propose(move(0.05))
    assert res.status == "held" and res.stage == "refusal_regression"
    assert res.reason.startswith("no should-abstain set declared for a target that can answer")
    rec = rr_records(pipe)
    assert len(rec) == 1 and rec[0]["decision"] == "hold" and rec[0]["measured"]["n"] == 0
    assert target.t == 0.5 and pipe.held() == [res.change_id]
    ok = pipe.rule(res.change_id, True, by="ann")                  # a person may still let it through, recorded
    assert ok.promoted and target.t == 0.05


def test_a_target_that_cannot_give_the_candidate_answerer_is_refused(tmp_path):
    class Blind(Threshold):
        def answerer(self, change, ctx):
            return None if change is not None else super().answerer(None, ctx)

    res = pipe_of(tmp_path, Blind(0.5), refusals=OUTSIDE).propose(move(0.05))
    assert res.status == "rejected" and res.stage == "refusal_regression" and "cannot measure" in res.reason


def test_a_change_with_no_answering_target_is_not_checked_unless_answer_of_is_given(tmp_path):
    from solvi.experimental.changes import hazard_body
    ks = KnowledgeStore()
    item = Change("outcome", payload={"kind": "rule", "body": hazard_body("lava", {"next": "lava"}, ["step"])},
                  evidence=["e1"], change_type="rule")
    pipe = Pipeline([], str(tmp_path / "c.jsonl"), store=ks, approval={("rule", "outcome"): "gate"}, refusals=OUTSIDE)
    assert pipe.propose(item).promoted and not rr_records(pipe)
    # with an answerer for the pipeline, an item change (a gate) is checked too
    ks2 = KnowledgeStore()

    def answer_of(change, ctx):                      # a gate: with the rule, nothing is refused any more
        return (lambda x: None) if change is None else (lambda x: "go")

    p2 = Pipeline([], str(tmp_path / "c2.jsonl"), store=ks2, approval={("rule", "outcome"): "gate"}, refusals=OUTSIDE,
                  answer_of=answer_of)
    res = p2.propose(item)
    assert res.status == "rejected" and res.stage == "refusal_regression"


def test_what_counts_as_a_refusal():
    assert is_refusal(None) and is_refusal("abstain") and is_refusal("Escalate") and is_refusal("hand over")
    assert not is_refusal("refuse") and not is_refusal("no") and not is_refusal(0)
    assert is_refusal({"status": "abstain"}) and not is_refusal({"status": "ok"})
    assert is_refusal(type("D", (), {"escalate": "low confidence"})())
    assert StageVerdict.of(None).ok


def test_a_system_answerer_reads_abstentions_from_its_response(tmp_path):
    def system(limit):
        cat = Catalog()

        @cat.rule("size")
        def size(n):
            return "unknown size" if n > limit else "small"
        return System(cat, [Question("size", "Size?", Answer.choice(["small"]))])

    class Sys:
        artifact = "update"

        def __init__(self):
            self.s = system(10)

        def answerer(self, change, ctx):
            s = self.s if change is None else change.payload["system"]
            return lambda x: s.ask(x, store=False)

        def promote(self, change, measured):
            self.s = change.payload["system"]
            return {}

        def rollback(self, ref, by=None, why=None):
            pass

    pipe = pipe_of(tmp_path, Sys(), refusals=[{"n": 50}, {"n": 80}])
    res = pipe.propose(Change("person", artifact="update", payload={"system": system(100), "fingerprint": "100"}))
    assert res.stage == "refusal_regression" and "2 of 2" in res.reason


def test_the_learning_target_and_the_versions_target_answer(tmp_path):
    from solvi.core.deciders import DecideModel
    from solvi.core.store import SQLiteStorage
    from test_decide import TASK, TEAMS, FakeScorer, texts
    m = DecideModel(FakeScorer(bias={"shipping": 5.0}, noise=0.3), meta={"format": "test", "temperature": 1.0})
    cat = Catalog()
    q = m.decision("team", TASK, "email", TEAMS, option_order="given").question(cat, "route")
    s = System(cat, [q], storage=SQLiteStorage(tmp_path / "d.db"))
    target = LearningTarget(s, gates={"max_change": 0.9})
    outside = [{"email": texts("other", 1, start=i)[0]} for i in range(5)]
    pipe = pipe_of(tmp_path, target, refusals=outside)
    for i in range(10):
        for team in TEAMS:
            email = texts(team, 1, start=i)[0]
            r = s.ask({"email": email})
            s.teach("route", {"email": email}, team, label_source="human", by="ann", of=r.stored_id)
    res = pipe.propose(Change("person", artifact="update", proposer="ann"))
    rec = rr_records(pipe)
    assert len(rec) == 1 and rec[0]["measured"]["n"] == 5, str(res)
    assert callable(getattr(VersionsTarget, "answerer", None))


# ------------------------------------------------------------------ opt-in: an item without an answerer; a sampled set
def test_missing_answer_refuse_fails_a_change_whose_target_cannot_answer(tmp_path):
    from solvi.experimental.changes import hazard_body
    item = Change("outcome", payload={"kind": "rule", "body": hazard_body("lava", {"next": "lava"}, ["step"])},
                  evidence=["e1"], change_type="rule")
    appr = {("rule", "outcome"): "gate"}
    # default: a change with no answering target passes the stage unmeasured (unchanged)
    p0 = Pipeline([RefusalRegression()], str(tmp_path / "a.jsonl"), store=KnowledgeStore(), approval=appr,
                  refusals=OUTSIDE)
    assert p0.propose(item).promoted
    # opt-in: it is refused, with the reason
    p1 = Pipeline([RefusalRegression(missing_answer="refuse")], str(tmp_path / "b.jsonl"), store=KnowledgeStore(),
                  approval=appr, refusals=OUTSIDE)
    res = p1.propose(item)
    assert res.status == "rejected" and res.stage == "refusal_regression" and "cannot answer" in res.reason
    m = rr_records(p1)[0]["measured"]
    assert m["measured"] is False and m["missing_answer"] == "refuse"
    # an answering change is measured as before
    p2 = pipe_of(tmp_path, Threshold(0.5), stages=[RefusalRegression(missing_answer="refuse")], name="c.jsonl",
                 refusals=OUTSIDE)
    assert p2.propose(move(0.6)).promoted
    with pytest.raises(ValueError):
        RefusalRegression(missing_answer="ignore")


NEAR = [{"score": s} for s in (0.46, 0.47, 0.48, 0.49, 0.9, 0.95)]   # past inputs: four refused at 0.5, two answered


def test_sampled_coverage_draws_unseen_refusals_from_a_pool(tmp_path):
    # the declared set alone: a move to 0.45 keeps refusing all four declared inputs, so it passes
    assert pipe_of(tmp_path, Threshold(0.5), refusals=OUTSIDE).propose(move(0.45)).promoted
    # sampled: inputs drawn from the pool that the version in force refuses join the set
    own = RefusalRegression(coverage="sampled", sample=6, pool=NEAR, seed=1)
    pipe = pipe_of(tmp_path, Threshold(0.5), stages=[own], name="s.jsonl", refusals=OUTSIDE)
    res = pipe.propose(move(0.45))
    assert res.status == "rejected" and res.stage == "refusal_regression"
    m = rr_records(pipe)[0]["measured"]
    assert m["coverage"] == "sampled" and m["sampled_drawn"] == 6 and m["sampled_refused"] == 4
    assert m["sources"] == {"pipeline": 4, "sampled": 4} and m["n"] == 8 and m["lost"] == 4
    assert m["seed"] == 1 and len(m["sampled_ids"]) == 4
    # a draw of k from a callable pool, or from a store's recorded inputs; the seed makes it repeatable
    calls = []

    def pool(change, ctx):
        calls.append(change.id if hasattr(change, "id") else 1)
        return NEAR

    p2 = pipe_of(tmp_path, Threshold(0.5), name="s2.jsonl", refusals=OUTSIDE,
                 stages=[RefusalRegression(coverage="sampled", sample=2, pool=pool, seed=3)])
    a = p2.propose(move(0.9))
    m2 = rr_records(p2)[0]["measured"]
    assert calls and m2["sampled_drawn"] == 2 and a.promoted

    class Rec:
        def __init__(self, x):
            self.input = x

    class Store:
        def iter(self):
            return iter([Rec(x) for x in NEAR])

    p3 = pipe_of(tmp_path, Threshold(0.5), name="s3.jsonl", refusals=OUTSIDE,
                 stages=[RefusalRegression(coverage="sampled", sample=6, pool=Store(), seed=1)])
    assert p3.propose(move(0.45)).stage == "refusal_regression"
    # without a seed each change gets a fresh draw; the seed used is recorded
    p4 = pipe_of(tmp_path, Threshold(0.5), name="s4.jsonl", refusals=OUTSIDE,
                 stages=[RefusalRegression(coverage="sampled", sample=3, pool=NEAR)])
    p4.propose(move(0.9))
    assert isinstance(rr_records(p4)[0]["measured"]["seed"], int)
    # declared (default) ignores a pool; bad arguments are refused early
    with pytest.raises(ValueError):
        RefusalRegression(coverage="all")
    with pytest.raises(ValueError):
        RefusalRegression(coverage="sampled", sample=4)                 # no pool
    with pytest.raises(ValueError):
        RefusalRegression(coverage="sampled", pool=NEAR, sample=0)


def test_a_sampled_pool_alone_measures_when_nothing_is_declared(tmp_path):
    pipe = pipe_of(tmp_path, Threshold(0.5), stages=[RefusalRegression(coverage="sampled", sample=6, pool=NEAR,
                                                                         seed=0)])
    res = pipe.propose(move(0.45))
    assert res.status == "rejected" and rr_records(pipe)[0]["measured"]["sources"] == {"sampled": 4}
