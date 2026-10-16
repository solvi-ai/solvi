"""Pipeline(order="cheap_first"): stages run in the order of their cost, so evidence that is expensive to collect (fresh
worlds, a sandbox, a honesty run) is not paid for a change a cheap stage refuses. The default order is the given one."""
from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import STAGE_COSTS, Approval, Change, Pipeline, StageVerdict, stage_cost


class Stage:
    def __init__(self, name, decision="admit", cost=None):
        self.name, self.decision, self.calls = name, decision, 0
        if cost is not None:
            self.cost = cost

    def run(self, change, ctx):
        self.calls += 1
        return StageVerdict(self.decision, self.name)


class Keep:
    artifact = "update"

    def promote(self, change, measured):
        return {}

    def rollback(self, ref, by=None, why=None):
        pass


def ch(i=0):
    return Change("outcome", artifact="update", payload={"fingerprint": f"u{i}"})


def pipe(tmp_path, stages, name="c.jsonl", **kw):
    return Pipeline(stages, str(tmp_path / name), targets=[Keep()], approval={"update": "gate"}, **kw)


def test_the_expensive_stage_is_not_paid_before_a_cheap_refusal(tmp_path):
    big, cheap = Stage("sandbox_run", cost=1000), Stage("lint", "refuse", cost=1)
    res = pipe(tmp_path, [big, cheap]).propose(ch())                       # the given order: paid
    assert res.stage == "lint" and big.calls == 1
    big2, cheap2 = Stage("sandbox_run", cost=1000), Stage("lint", "refuse", cost=1)
    res = pipe(tmp_path, [big2, cheap2], name="c2.jsonl", order="cheap_first").propose(ch())
    assert res.stage == "lint" and big2.calls == 0


def test_order_costs_and_approval_last(tmp_path):
    a, b, c = Stage("a"), Stage("b", cost=5), Stage("c")
    p = pipe(tmp_path, [a, b, c], order="cheap_first", costs={"a": 50})
    names = [s.name for s in p.stages_for(ch())]
    assert names == ["b", "c", "a", "approval"]                         # c: the default cost 10; approval stays last
    assert p.propose(ch()).promoted
    assert p.history("propose")[0]["order"] == names
    assert stage_cost(a) == STAGE_COSTS.get("a", 10.0) and stage_cost(b) == 5


def test_the_default_order_is_unchanged_and_a_bad_order_is_refused(tmp_path):
    a, b = Stage("a", cost=100), Stage("b", cost=1)
    p = pipe(tmp_path, [a, b])
    assert [s.name for s in p.stages_for(ch())] == ["a", "b", "approval"]
    assert "order" not in p.history("propose")[0] if p.history("propose") else True
    with pytest.raises(ValueError):
        pipe(tmp_path, [a], name="c3.jsonl", order="random")


def test_a_held_change_resumes_in_the_cheap_first_order(tmp_path):
    big, cheap = Stage("big", cost=1000), Stage("cheap", cost=1)
    p = Pipeline([big, cheap, Approval({"update": "person"})], str(tmp_path / "c.jsonl"), targets=[Keep()],
                 order="cheap_first")
    res = p.propose(ch())
    assert res.status == "held" and res.stage == "approval" and big.calls == 1 and cheap.calls == 1
    assert p.rule(res.change_id, True, by="ann").promoted
