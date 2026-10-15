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
    from solvi.experimental.changes import (Approval, Canaries, Change, CleanCases, Confirmation, Consistency,
                                            EvalLocked, EvaluatorLock, FreshEnvironments, Impact, KnownFailures,
                                            Pipeline, Static, StageVerdict, VersionsTarget, allowed, gate_of,
                                            gates_of, hazard_body, paired_bootstrap)


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
    gates = gates_of(ctx.store)
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

    gates = gates_of(ks)
    assert len(gates) == 1 and not allowed({"next": "lava"}, "step", gates) and allowed({"next": "floor"}, "step", gates)
    assert len(ks.find(kind="rule")) == 1                     # the view wrote nothing
    assert pipe.rollback(0, by="ann", why="test", failed=False) == [1]
    assert gates_of(ks) == []
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
    unconfirmed = pipe.propose(Change("search", payload={"kind": "rule", "body": body}, proposer="evolve",
                                      evidence=["ep-1"]))
    assert unconfirmed.status == "rejected" and unconfirmed.stage == "confirmation"
    assert unconfirmed.reason == "search proposal: confirmation on its evidence did not run"
    other = Pipeline([Static(), Confirmation(lambda c, x: True)], str(tmp_path / "c2.jsonl"), store=ks,
                     approval={"rule": "gate"})               # a confirmation that is not on the evidence: not enough
    assert other.propose(Change("search", payload={"kind": "rule", "body": body}, proposer="evolve",
                                evidence=["ep-1"])).stage == "confirmation"
    pipe = Pipeline([Static(), Confirmation.on_evidence(lambda e, c: True)], str(tmp_path / "c3.jsonl"), store=ks,
                    approval={"rule": "gate"})
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
    on_ev = Confirmation.on_evidence(lambda e, c: e != "ep-12")
    v = on_ev.run(ch, None)
    assert v.decision == "refuse" and v.measured["confirmed"] == 2
    acc = type("C", (), {"accepted": True, "fingerprint": "f", "reason": ""})()
    assert Confirmation.compiled().run(Change("document", artifact="compiled", payload=acc), None).ok
    assert paired_bootstrap([0, 0, 0, 0], [1, 1, 1, 1], boot=100)[1] == 1.0


def test_targets_for_compiled_versions_and_updates(tmp_path):
    class FakeVersions:
        def __init__(self):
            self.v = []

        def numbers(self):
            return list(range(1, len(self.v) + 1))

        def add(self, compiled, label=None):
            self.v.append(compiled)
            return len(self.v)

    class UpdateTarget:                           # any object with artifact, promote and rollback is a target
        artifact = "update"
        version = 0

        def promote(self, change, measured):
            self.version += 1
            return {"version": self.version, "before": self.version - 1}

        def rollback(self, ref, by=None, why=None):
            self.version = ref["before"]

    vt, loop = VersionsTarget(FakeVersions()), UpdateTarget()
    acc = type("C", (), {"accepted": True, "fingerprint": "f1", "reason": ""})()
    pipe = Pipeline([Confirmation.compiled()], str(tmp_path / "c.jsonl"), targets=[vt, loop],
                    approval={"compiled": "gate", "update": "gate"}, refusals=[{"text": "outside"}])
    r1 = pipe.propose(Change("document", artifact="compiled", payload=acc, reason="v2 of the policy"))
    assert r1.promoted and vt.in_force == 1
    r2 = Pipeline([], pipe.journal.storage, targets=[vt, loop], approval={"update": "gate"}).propose(
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


def test_update_candidates_differ_by_state_records_carry_the_level_and_callbacks_measure(tmp_path):
    class Target:
        artifact = "update"

        def __init__(self):
            self.state = None

        def promote(self, change, measured):
            self.state = change.payload["fingerprint"]
            return {"state": self.state}

        def rollback(self, ref, by=None, why=None):
            self.state = None

    a, b = Change("person", artifact="update", payload={"fingerprint": "s1"}), Change("person", artifact="update",
                                                                                          payload={"fingerprint": "s2"})
    assert a.fingerprint() != b.fingerprint() and Change("person", artifact="update").to_dict()["content"] is None
    t = Target()
    only_one = Impact(lambda c, x: {"changed": 2, "total": 10, "ok": False, "reason": "changes another decision"})
    pipe = Pipeline([KnownFailures(), Consistency(classify=lambda c, x: {"direction": "narrows", "conflicts": 0}),
                     only_one], str(tmp_path / "c.jsonl"), targets=[t], approval={("update", "person"): "gate"})
    res = pipe.propose(a)
    assert res.status == "rejected" and res.stage == "impact" and res.reason == "changes another decision"
    assert res.verdicts[1].measured == {"direction": "narrows", "conflicts": 0}
    assert pipe.history("rejected")[-1]["level"] == "gate" and pipe.propose(b).stage == "impact"
    widen = Pipeline([Consistency(classify=lambda c, x: {"direction": "widens"})], str(tmp_path / "c2.jsonl"),
                     targets=[t], approval={("update", "person"): "gate"})
    held = widen.propose(a)
    assert held.status == "held" and widen.history("held")[-1]["level"] == "gate+person"
    done = widen.rule(held.change_id, True, by="ann")
    assert done.promoted and t.state == "s1" and widen.history("promoted")[-1]["level"] == "gate+person"


# ───────────────────────────────────────────── a candidate catalog through Static: sensitive fields
class StateTarget:
    """A target for "update" changes whose payload names a state (and carries its catalog)."""
    artifact = "update"

    def __init__(self):
        self.state = None

    def promote(self, change, measured):
        self.state = change.payload["fingerprint"]
        return {"state": self.state}

    def rollback(self, ref, by=None, why=None):
        self.state = None


def _credit_catalog(young=False, sensitive=()):
    from solvi import Catalog
    cat = Catalog(sensitive=sensitive)

    @cat.fn
    def amount_points(amount):
        return 2 if amount > 8000 else 0

    if young:
        @cat.fn
        def young_points(age):
            return 1 if age <= 25 else 0

    @cat.rule("decision")
    def decision(amount_points):
        return "refuse" if amount_points >= 2 else "approve"
    return cat


def test_static_refuses_a_candidate_catalog_that_reads_a_sensitive_field(tmp_path):
    t = StateTarget()
    pipe = Pipeline([Static(sensitive={"age"})], str(tmp_path / "c.jsonl"), targets=[t],
                    approval={("update", "person"): "gate"})
    bad = pipe.propose(Change("person", artifact="update", payload={"fingerprint": "young", "catalog": _credit_catalog(True)}))
    assert bad.status == "rejected" and bad.stage == "static"
    assert "reads_sensitive" in bad.reason and "young_points" in bad.reason and "'age'" in bad.reason
    ok = pipe.propose(Change("person", artifact="update", payload={"fingerprint": "plain", "catalog": _credit_catalog()}))
    assert ok.promoted and t.state == "plain"
    # the catalog's own declaration is enough: Static() lints with it
    own = Pipeline([Static()], str(tmp_path / "c2.jsonl"), targets=[t], approval={("update", "person"): "gate"})
    res = own.propose(Change("person", artifact="update",
                             payload={"fingerprint": "young2", "catalog": _credit_catalog(True, sensitive={"age"})}))
    assert res.status == "rejected" and res.stage == "static" and "reads_sensitive" in res.reason


# ───────────────────────────────────────────── exact exceptions: Impact refuses a change that moves other decisions
def _exception_system(cond=None):
    from solvi import Answer, Catalog, Question, System
    cat = Catalog()
    if cond is not None:
        @cat.check(hard=True, then={"decision": "approve"})
        def exception_k07(purpose, points) -> bool:
            return not all({"purpose": purpose, "points": points}[k] == v for k, v in cond.items())

    @cat.rule("decision")
    def decision(points):
        return "refuse" if points >= 4 else "approve"
    return System(cat, [Question("decision", "Approve?", Answer.choice(["approve", "refuse"]))])


def test_impact_refuses_an_exception_that_moves_other_decisions_and_admits_the_exact_one(tmp_path):
    store = JSONLStorage(str(tmp_path / "d.jsonl"))
    v0 = _exception_system()
    v0.storage, store.catalog = store, v0
    apps = {"a1": ("car", 5), "a2": ("tv", 5), "a3": ("car", 6), "a4": ("car", 1)}
    sid = {k: v0.ask({"purpose": p, "points": n}).stored_id for k, (p, n) in apps.items()}
    t = StateTarget()
    pipe = Pipeline([Impact.of_system(store, lambda ch, ctx: ch.payload["system"])], str(tmp_path / "c.jsonl"),
                    targets=[t], approval={("update", "person"): "gate"})
    broad = pipe.propose(Change("person", artifact="update", expected={"only": [sid["a1"]]},
                                payload={"fingerprint": "broad", "system": _exception_system({"purpose": "car"})}))
    assert broad.status == "rejected" and broad.stage == "impact"
    # a3 moves (refuse → approve); a4 keeps "approve" but is now forced by the exception: a changed decision too
    assert broad.reason == f"changes 2 decision(s) other than the allowed ones: {[sid['a3'], sid['a4']]}"
    assert broad.verdicts[-1].measured["extraneous"] == [sid["a3"], sid["a4"]]
    exact = pipe.propose(Change("person", artifact="update", expected={"only": [sid["a1"]]},
                                payload={"fingerprint": "exact",
                                         "system": _exception_system({"purpose": "car", "points": 5})}))
    assert exact.promoted and t.state == "exact"
    m = {v.stage: v for v in exact.verdicts}["impact"].measured           # (the approval stage ran after it)
    assert m["extraneous"] == [] and m["changed"] == 1 and m["ids"] == [sid["a1"]]
