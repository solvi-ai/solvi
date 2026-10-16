"""Opt-in closings of holes in the change gates that a proposer adapting to them could use:
- RefusalRegression(cumulative=True): refusals lost are counted against the original version too, so a chain of
  changes each within the tolerance ("salami") cannot add up;
- Consistency(probes=): the library measures whether a change widens or narrows behaviour (the caller's classify is
  only a claim, recorded and overruled toward "widens");
- Canaries.drawn: the canary cases are drawn per change from a pool with the stage's secret seed, not a fixed set the
  proposer can tune to;
- Confirmation.on_evidence(exists=): evidence ids must be in the record, and each counts once;
- FreshEnvironments(handed_over=): a world the candidate hands over is not "lower progress"."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import (Canaries, Change, Confirmation, Consistency, FreshEnvironments, Pipeline,
                                            RefusalRegression)


class Threshold:
    """A target that answers: below the threshold it hands over (None)."""
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


def move(t, **kw):
    return Change("outcome", artifact="update", payload={"t": t, "fingerprint": f"t={t}"}, proposer="self", **kw)


def pipe_of(tmp_path, target, stages=(), name="c.jsonl", approval=None, **kw):
    return Pipeline(list(stages), str(tmp_path / name), targets=[target], approval=approval or {"update": "gate"}, **kw)


# ─────────────────────────────────────────────── refusal regression against the original version
def test_salami_refusal_loss_is_caught_against_the_original(tmp_path):
    def chain(name, **kw):
        p = pipe_of(tmp_path, Threshold(0.5), [RefusalRegression(count_tolerance=1, **kw)], name=name, refusals=OUTSIDE)
        return [p.propose(move(t)).status for t in (0.35, 0.25, 0.15)]
    assert chain("a.jsonl") == ["promoted", "promoted", "promoted"]              # each step loses one: all pass
    assert chain("b.jsonl", cumulative=True) == ["promoted", "rejected", "rejected"]


# ─────────────────────────────────────────────── widens / narrows measured by the library
def test_a_widening_change_claimed_as_narrowing_is_measured_and_held(tmp_path):
    claim = lambda change, ctx: "narrows"                                          # noqa: E731
    p = pipe_of(tmp_path, Threshold(0.5), [Consistency(classify=claim)], refusals=OUTSIDE[:1])
    assert p.propose(move(0.3)).status == "promoted"                               # the caller's word was trusted
    p = pipe_of(tmp_path, Threshold(0.5), [Consistency(classify=claim, probes=OUTSIDE)], name="b.jsonl",
                refusals=OUTSIDE[:1])
    res = p.propose(move(0.3))
    assert res.status == "held" and res.stage == "approval"
    m = next(v for v in res.verdicts if v.stage == "consistency").measured
    assert m["direction"] == "widens" and m["claimed"] == "narrows" and m["widened"] == 2


def test_a_narrowing_change_is_measured_as_narrowing(tmp_path):
    p = pipe_of(tmp_path, Threshold(0.3), [Consistency(probes=OUTSIDE)], refusals=OUTSIDE[:1])
    res = p.propose(move(0.5))
    assert res.promoted
    assert next(v for v in res.verdicts if v.stage == "consistency").measured["direction"] == "narrows"


# ─────────────────────────────────────────────── canaries drawn per change
POOL = [{"id": f"q{i}", "gold": i % 7} for i in range(40)]
SHOWN = {c["id"] for c in POOL[:5]}


def cheat(cases):
    """A candidate tuned to the canaries it saw: right on those, wrong elsewhere."""
    wrong = sum(1 for c in cases if c["id"] not in SHOWN)
    return {"n": len(cases), "confident_error_rate": wrong / len(cases), "confident_errors": wrong, "escalated": 1}


def test_static_canaries_are_passed_by_a_tuned_candidate_drawn_ones_are_not(tmp_path):
    static = Canaries(lambda change, ctx: cheat(POOL[:5]), baseline={"confident_error_rate": 0.0})
    assert pipe_of(tmp_path, Threshold(), [static], refusals=OUTSIDE[:1]).propose(move(0.6)).promoted
    drawn = Canaries.drawn(lambda change, ctx, cases: cheat(cases), POOL, 5, baseline={"confident_error_rate": 0.0})
    res = pipe_of(tmp_path, Threshold(), [drawn], name="b.jsonl", refusals=OUTSIDE[:1]).propose(move(0.6))
    assert res.status == "rejected" and res.stage == "canaries"
    m = res.verdicts[0].measured
    assert m["drawn"] == 5 and m["pool"] == 40 and isinstance(m["commit"], str) and m["commit"]


def test_the_draw_depends_on_the_change_and_the_secret_not_on_the_caller():
    st = Canaries.drawn(lambda change, ctx, cases: {}, POOL, 5, secret=b"s")
    a, b = st.draw(SimpleNamespace(change_id="c1")), st.draw(SimpleNamespace(change_id="c2"))
    assert a == st.draw(SimpleNamespace(change_id="c1")) and a != b
    other = Canaries.drawn(lambda change, ctx, cases: {}, POOL, 5, secret=b"t")
    assert other.draw(SimpleNamespace(change_id="c1")) != a


# ─────────────────────────────────────────────── evidence ids checked against the record
RECORDED = {"d1", "d2"}


def test_invented_or_repeated_evidence_ids_do_not_confirm(tmp_path):
    replay = lambda eid, change: eid != "d2"                                      # noqa: E731 — d2 does not confirm
    def run(evidence, name, **kw):
        st = Confirmation.on_evidence(replay, **kw)
        ch = Change("search", artifact="update", payload={"t": 0.6, "fingerprint": name}, proposer="m", evidence=evidence)
        return pipe_of(tmp_path, Threshold(), [st], name=name, refusals=OUTSIDE[:1]).propose(ch)
    assert run(["fake-1"], "a.jsonl").promoted                                     # an invented id was trusted
    res = run(["fake-1"], "b.jsonl", exists=lambda eid, ctx: eid in RECORDED)
    assert res.status == "rejected" and "not in the record" in res.reason
    assert run(["d1", "d1", "d1", "d2"], "c.jsonl", min_share=0.75).promoted       # one good id repeated: 3 of 4
    res = run(["d1", "d1", "d1", "d2"], "d.jsonl", min_share=0.75, exists=lambda eid, ctx: eid in RECORDED)
    assert res.status == "rejected" and res.verdicts[0].measured["evidence"] == 2


# ─────────────────────────────────────────────── refusal is not lower progress
class Quiz:
    """A world is one question; hard ones (seed % 3 == 0) the answerer gets wrong. Answering: progress 1 (failed when
    wrong); handing over: progress 0, not failed."""
    def reset(self, seed):
        self.hard = seed % 3 == 0
        return {"hard": self.hard}

    def actions(self, state):
        return ["answer", "hand_over"]

    def step(self, a):
        if a == "answer":
            return SimpleNamespace(state={}, done=True, info={"failed": self.hard, "progress": 1.0})
        return SimpleNamespace(state={}, done=True, info={"failed": False, "progress": 0.0})


def agent(change, ctx):
    if change is None:
        return lambda s, acts: "answer"
    return lambda s, acts: "hand_over" if s["hard"] else "answer"


def test_handing_over_is_not_lower_progress_when_declared(tmp_path):
    def run(name, **kw):
        st = FreshEnvironments(Quiz, agent, select=range(30), confirm=range(100, 160), boot=500, **kw)
        return pipe_of(tmp_path, Threshold(), [st], name=name, refusals=OUTSIDE[:1]).propose(move(0.6))
    res = run("a.jsonl")
    assert res.status == "rejected" and "progress lower" in res.reason            # the honest refusal is penalised
    res = run("b.jsonl", handed_over=lambda row: row["actions"][-1:] == ["hand_over"])
    assert res.promoted, str(res)
    assert res.verdicts[0].measured["confirm"]["handed_over"] == 20
