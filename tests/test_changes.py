"""solvi.experimental.changes: one change pipeline over a toy corridor with a hazard."""
from __future__ import annotations

import json
import random
import warnings

import pytest

from solvi.core.environment import Outcome
from solvi.core.knowledge import KnowledgeStore
from solvi.core.store import JSONLStorage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import (Approval, Canaries, Change, CleanCases, Consistency, EvalLocked,
                                            EvaluatorLock, FreshEnvironments, Impact, KnownFailures, LearningTarget,
                                            Pipeline, Static, StageVerdict, VersionsTarget, allowed, gate_of, hazard_body,
                                            hazard_gate, paired_bootstrap)


# ───────────────────────────────────────────── a toy world: walk right to the goal; one cell may be lava
class Corridor:
    """Cells 0..L; "step" moves 1, "jump" moves 2; landing on the lava cell ends the episode failed. Deterministic by
    seed: the lava cell (or none) is drawn from the seed."""
    L = 8

    def reset(self, seed=None):
        rng = random.Random(seed)
        self.lava = rng.randint(2, self.L - 1) if rng.random() < 0.7 else None
        self.pos = 0
        return self._state()

    def _state(self):
        nxt = self.pos + 1
        return {"pos": self.pos, "next": "goal" if nxt >= self.L else "lava" if nxt == self.lava else "floor"}

    def actions(self, state):
        return ["step", "jump"]

    def step(self, action):
        n = min(self.L, self.pos + (1 if action == "step" else 2))
        if n == self.lava:
            return Outcome(self._state(), True, {"pos": n}, True, {"failed": True, "progress": n / self.L})
        self.pos = n
        return Outcome(self._state(), True, {"pos": n}, n >= self.L, {"failed": False, "progress": n / self.L})


def agent(change, ctx):
    """Prefers "step"; obeys every active hazard rule of the store (and the candidate's): a forbidden step → jump."""
    gates = [hazard_gate(r["body"]) for r in ctx.store.find(kind="rule", status="active")
             if (r["body"] or {}).get("kind") == "hazard"]
    if change is not None:
        gates.append(gate_of(change))

    def policy(state, actions):
        for a in ("step", "jump"):
            if allowed(state, a, gates):
                return a
        return "step"
    return policy


def successes(seeds=range(1000, 1030)):
    """(state, action) pairs of the current agent that did not end in a failure (clean cases for dry_run)."""
    out = []
    for s in seeds:
        env = Corridor()
        st = env.reset(s)
        for _ in range(20):
            out_ = env.step("step")
            if out_.info["failed"]:
                break
            out.append((st, "step"))
            st = out_.state
            if out_.done:
                break
    return out


def lava_rule(**kw):
    return Change("outcome", payload={"kind": "rule", "body": hazard_body("lava", {"next": "lava"}, ["step"])},
                  proposer="model:test", reason="three deaths stepping onto lava", evidence=["ep-3", "ep-9", "ep-12"],
                  expected={"failed_share": "lower"}, change_type="rule", **kw)


def pipeline(tmp_path, ks, approval=None, name="changes.jsonl"):
    return Pipeline([KnownFailures(), Static(), Consistency(), CleanCases(successes()),
                     FreshEnvironments(Corridor, agent, select=range(100, 130), confirm=range(200, 260), boot=500)],
                    str(tmp_path / name), store=ks, approval=approval or {("rule", "outcome"): "gate"})


# ───────────────────────────────────────────── the end-to-end path
def test_a_hazard_rule_from_outcomes_is_promoted_listed_and_rolled_back(tmp_path):
    ks = KnowledgeStore()
    pipe = pipeline(tmp_path, ks)
    res = pipe.propose(lava_rule())
    assert res.promoted, str(res)
    assert [v.stage for v in res.verdicts] == ["failure_memory", "static", "consistency", "clean_cases",
                                               "fresh_environments", "approval"]
    fresh = res.verdicts[4].measured
    assert fresh["confirm"]["failed"][1] == 0 < fresh["confirm"]["failed"][0] and fresh["confirm"]["gain"][1] > 0
    assert fresh["deterministic"] and fresh["confirm"]["progress"][0] > 0
    iid = res.ref
    item = ks.item(iid)
    assert item["status"] == "active" and item["kind"] == "rule" and item["source"] == "outcome"
    assert item["body"]["kind"] == "hazard" and item["body"]["change"] == "rule" and item["by"] == "outcome:model:test"
    assert pipe.current == 1 and pipe.versions()[0]["version"] == 1 and pipe.versions()[0]["in_force"]
    assert pipe.lineage(res.change_id) == [{"change": res.change_id, "version": 1, "parent": 0}]
    assert pipe.observe(res.change_id, {"failed_share": 0.0})["expected"] == {"failed_share": "lower"}

    assert pipe.rollback(0, by="ann", why="test") == [1]
    assert ks.status(iid) == "retracted" and pipe.current == 0 and not pipe.versions()[0]["in_force"]
    assert pipe.journal.verify()["ok"] and ks.verify()
    acts = [d["action"] for d in pipe.history()]
    assert acts[0] == "propose" and acts.count("verdict") == 6 and acts[-3:] == ["promoted", "observed", "rollback"]

    # a second change rests on the version in force: its lineage goes back through it
    res2 = pipe.propose(lava_rule())                  # the same rule again after the rollback: a new version
    assert res2.promoted and res2.version == 2 and pipe.current == 2
    other = Change("person", payload={"kind": "fact", "body": {"s": "corridor", "r": "length", "o": 8}}, proposer="ann")
    res3 = Pipeline([Static(), Consistency()], pipe.journal.storage, store=ks, approval={"fact": "gate"}).propose(other)
    assert res3.promoted and [x["change"] for x in pipe.lineage(res3.change_id)] == [res3.change_id, res2.change_id]


def test_a_rejected_candidate_is_recorded_with_its_stage_and_refused_when_proposed_again(tmp_path):
    ks = KnowledgeStore()
    pipe = pipeline(tmp_path, ks)
    bad = Change("outcome", payload={"kind": "rule", "body": hazard_body("floor", {"next": "floor"}, ["step"])},
                 proposer="model:test", reason="a phantom rule", evidence=["ep-1"])
    res = pipe.propose(bad)
    assert res.status == "rejected" and res.stage == "clean_cases" and "recorded successes" in res.reason
    rej = pipe.history("rejected")
    assert rej[-1]["stage"] == "clean_cases" and rej[-1]["fingerprint"] == bad.fingerprint()
    assert ks.find(kind="rule") == [] and pipe.versions() == []
    again = pipe.propose(bad)
    assert again.status == "rejected" and again.stage == "failure_memory" and "clean_cases" in again.reason


def test_approval_person_holds_until_a_ruling(tmp_path):
    ks = KnowledgeStore()
    pipe = pipeline(tmp_path, ks, approval={("rule", "outcome"): "person"})
    res = pipe.propose(lava_rule())
    assert res.status == "held" and res.stage == "approval" and pipe.held() == [res.change_id]
    assert ks.find(kind="rule") == [] and pipe.current == 0
    done = pipe.rule(res.change_id, True, by="ann")
    assert done.promoted and ks.item(done.ref)["status"] == "active" and pipe.held() == []
    assert pipe.history("ruling")[-1]["by"] == "ann"

    ks2 = KnowledgeStore()
    pipe2 = pipeline(tmp_path, ks2, approval=Approval({("rule", "outcome"): "gate+person"}), name="c2.jsonl")
    held = pipe2.propose(lava_rule())
    no = pipe2.rule(held.change_id, "reject", by="bob")
    assert no.status == "rejected" and no.stage == "approval" and ks2.find(kind="rule") == []


def test_a_widening_change_needs_a_person_even_where_the_table_says_gate(tmp_path):
    ks = KnowledgeStore()
    pipe = Pipeline([Static(), Consistency(classify=lambda ch, ctx: "widens")], str(tmp_path / "c.jsonl"), store=ks,
                    approval={("rule", "outcome"): "gate"})
    assert pipe.propose(lava_rule()).status == "held"


def test_a_search_proposal_needs_evidence_and_an_unknown_source_is_refused(tmp_path):
    ks = KnowledgeStore()
    pipe = Pipeline([Static()], str(tmp_path / "c.jsonl"), store=ks, approval={"rule": "gate"})
    body = hazard_body("lava", {"next": "lava"}, ["step"])
    res = pipe.propose(Change("search", payload={"kind": "rule", "body": body}, proposer="evolve"))
    assert res.status == "rejected" and res.stage == "static" and "evidence" in res.reason
    ok = pipe.propose(Change("search", payload={"kind": "rule", "body": body}, proposer="evolve", evidence=["ep-1"]))
    assert ok.promoted and ks.item(ok.ref)["source"] == "outcome" and ks.item(ok.ref)["by"] == "search:evolve"
    res = pipe.propose(Change("model", payload={"kind": "rule", "body": body}))
    assert res.status == "rejected" and "source" in res.reason


def test_rules_that_crash_and_rules_that_do_not_help_are_refused_on_fresh_worlds(tmp_path):
    ks = KnowledgeStore()
    fresh = FreshEnvironments(Corridor, agent, select=range(100, 130), confirm=range(200, 260), boot=300)
    pipe = Pipeline([fresh], str(tmp_path / "c.jsonl"), store=ks, approval={"rule": "gate"})

    def boom(state):
        raise ZeroDivisionError("bad rule")
    crash = lava_rule(check=boom)
    res = pipe.propose(crash)
    assert res.status == "rejected" and "crashed" in res.reason
    useless = Change("outcome", payload={"kind": "rule", "body": hazard_body("goal", {"next": "nowhere"}, ["step"])},
                     evidence=["ep-1"])
    res = pipe.propose(useless)
    assert res.status == "rejected" and "no gain" in res.reason
    same = FreshEnvironments(Corridor, agent, select=[1, 2], confirm=[2, 3])
    assert same.run(lava_rule(), None).decision == "refuse"


def test_the_journal_is_hash_chained_and_a_tampered_record_is_caught(tmp_path):
    ks = KnowledgeStore()
    path = tmp_path / "changes.jsonl"
    pipe = pipeline(tmp_path, ks)
    pipe.propose(lava_rule())
    assert JSONLStorage(str(path)).verify()["ok"]
    lines = path.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if json.loads(line).get("action") == "promoted")
    d = json.loads(lines[i])
    d["version"] = 7
    lines[i] = json.dumps(d)
    path.write_text("\n".join(lines) + "\n")
    rep = JSONLStorage(str(path)).verify()
    assert not rep["ok"] and rep["problems"]


# ───────────────────────────────────────────── the other stages and pieces
def test_verdicts_impact_canaries_and_confirmation(tmp_path):
    assert StageVerdict.of(True).ok and StageVerdict.of({"ok": False, "reason": "x", "n": 3}).measured == {"n": 3}
    assert StageVerdict.of(None).ok
    with pytest.raises(ValueError):
        StageVerdict("maybe")
    ch = lava_rule()
    assert Impact(lambda c, x: {"changed": 4, "total": 10}, max_share=0.3).run(ch, None).decision == "refuse"
    assert Impact(lambda c, x: {"changed": 1, "total": 10}, max_share=0.3).run(ch, None).measured["share"] == 0.1
    base = {"confident_error_rate": 0.05, "coverage_at_risk": 0.6}
    worse = Canaries(lambda c, x: {"confident_error_rate": 0.2, "coverage_at_risk": 0.6, "n": 5, "confident_errors": 1,
                                   "escalated": 0}, baseline=base)
    assert worse.run(ch, None).decision == "refuse"
    perfect = Canaries(lambda c, x: {"confident_error_rate": 0.0, "coverage_at_risk": 1.0, "n": 5, "confident_errors": 0,
                                     "escalated": 0}, baseline=base)
    assert perfect.run(ch, None).decision == "hold"
    from solvi.experimental.changes import Confirmation
    on_ev = Confirmation.on_evidence(lambda e, c: e != "ep-12")
    v = on_ev.run(ch, None)
    assert v.decision == "refuse" and v.measured["confirmed"] == 2
    acc = type("C", (), {"accepted": True, "fingerprint": "f", "reason": ""})()
    assert Confirmation.compiled().run(Change("document", artifact="compiled", payload=acc), None).ok
    assert paired_bootstrap([0, 0, 0, 0], [1, 1, 1, 1], boot=100)[1] == 1.0


def test_targets_for_compiled_versions_and_learning_updates(tmp_path):
    class FakeVersions:
        def __init__(self):
            self.v = []

        def numbers(self):
            return list(range(1, len(self.v) + 1))

        def add(self, compiled, label=None):
            self.v.append(compiled)
            return len(self.v)

    class FakeLoop:
        version = 0

        def _baseline(self):
            return self.version

        def run(self):
            self.version += 1
            return type("R", (), {"promoted": True, "version": self.version})()

        def rollback(self, v):
            self.version = v

    vt, loop = VersionsTarget(FakeVersions()), FakeLoop()
    acc = type("C", (), {"accepted": True, "fingerprint": "f1", "reason": ""})()
    from solvi.experimental.changes import Confirmation
    pipe = Pipeline([Confirmation.compiled()], str(tmp_path / "c.jsonl"), targets=[vt, LearningTarget(loop)],
                    approval={"compiled": "gate", "update": "gate"})
    r1 = pipe.propose(Change("document", artifact="compiled", payload=acc, reason="v2 of the policy"))
    assert r1.promoted and vt.in_force == 1
    r2 = Pipeline([], pipe.journal.storage, targets=[vt, LearningTarget(loop)], approval={"update": "gate"}).propose(
        Change("person", artifact="update"))
    assert r2.promoted and loop.version == 1
    pipe.rollback(0)
    assert vt.in_force is None and loop.version == 0


def test_the_evaluator_lock(tmp_path, monkeypatch):
    seeds = tmp_path / "eval_seeds.json"
    pin = EvaluatorLock.create(seeds, n=20)
    with pytest.raises(FileExistsError):
        EvaluatorLock.create(seeds, n=20)

    def metric(rows):
        return sum(rows)
    lock = EvaluatorLock(seeds, pinned=pin, code={"metric": metric}, pins={}, access_log=str(tmp_path / "access.jsonl"),
                         env="SOLVI_TEST_EVAL_LOCKED")
    with pytest.raises(EvalLocked, match="not pinned"):
        lock.verify()
    lock.pins = dict(lock.manifest()["code"])
    assert lock.verify()
    got = lock.seeds("final evaluation", by="test")
    assert len(got) == 20 and lock.access_count() == 1
    (tmp_path / "journal.txt").write_text(f"candidate tried world {got[3]}")
    assert lock.leak_scan([tmp_path])[0][0].endswith("journal.txt")
    monkeypatch.setenv("SOLVI_TEST_EVAL_LOCKED", "1")
    with pytest.raises(EvalLocked, match="locked out"):
        lock.seeds("peek")
    monkeypatch.delenv("SOLVI_TEST_EVAL_LOCKED")
    with pytest.raises(EvalLocked, match="changed"):
        EvaluatorLock(seeds, pinned="0" * 64).verify()
    assert lock.access_count() == 1
