"""solvi.core.slow.search decides its candidates lean (System._decide: no hashes, fingerprint, audit, stats or storage): every
candidate's verdict — accepted or not, its objective, what rejected it, which checks failed, every fact's value — equals
the verdict of a full ask of the same state; the winner is still asked in full, and its trace is byte for byte the trace
of an ordinary ask. The NATURAL PLAN check (every candidate of the dev problems, both paths) runs with `-m stand` when
the stand's data is there (STAND_DATA)."""
import json
import os
import sys
from pathlib import Path

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage
from solvi.core.slow.refine import causes, failed_checks
from solvi.core.runtime import vhash
from solvi.core.slow.search import Tree, _verdict, search

ROOT = Path(__file__).resolve().parents[1]


def _seen(res, questions):
    """Everything a search could read from a candidate's response, hashed where it is a value."""
    out = []
    for q in questions:
        r = res[q]
        out.append((q, r.status, vhash(r.answer), r.guard, r.why, r.confidence, _verdict(res, q, "checks", None, None),
                    [(f.check, f.hard, f.reasons) for f in failed_checks(res, q)], causes(res, q)))
    out.append(sorted((k, vhash(v)) for k, v in res.values.items()))
    out.append([(x.name, x.kind, x.error, vhash(x.value), x.confidence, x.producer, x.extra) for x in res.trace.records])
    out.append(list(res.trace.skipped))
    return out


@pytest.fixture
def both(monkeypatch):
    """Every lean candidate is asked in full too; the differences are collected (the lean response is returned)."""
    lean = System._decide
    seen = {"n": 0, "diffs": []}

    def decide(self, init_state, questions=None, early_exit=None):
        a = lean(self, init_state, questions, early_exit)
        b = self.ask(init_state, questions=questions, store=False, early_exit=early_exit)
        seen["n"] += 1
        if _seen(a, questions) != _seen(b, questions):
            seen["diffs"].append(init_state)
        return a
    monkeypatch.setattr(System, "_decide", decide)
    return seen


def _trace_bytes(res):
    d = res.trace.model_dump("json")
    d.pop("timings", None)                            # run times: not hashed, never the same twice
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _trip():
    cat = Catalog()
    flights = {("A", "B"), ("B", "C"), ("C", "D"), ("B", "D"), ("D", "E"), ("A", "C")}

    @cat.fn
    def legs(order: list) -> int:
        return max(len(order) - 1, 0)

    @cat.check(hard=True, then={"ok": "no"})
    def direct(order: list) -> bool:
        return all((a, b) in flights or (b, a) in flights for a, b in zip(order, order[1:]))

    @cat.check(hard=True, then={"ok": "no"})
    def starts_at_a(order: list) -> bool:
        return not order or order[0] == "A"

    @cat.check
    def short(legs: int) -> bool:                     # a soft check: recorded, never decides
        return legs < 3

    @cat.rule("ok")
    def ok(direct, starts_at_a, short) -> bool:
        return True
    return System(cat, [Question("ok", "A valid trip?", Answer.yes_no(), requires=["direct", "starts_at_a", "legs"])])


CITIES = ["A", "B", "C", "D", "E"]
TREE = Tree([], lambda o: [o + [c] for c in CITIES if c not in o], complete=lambda o: len(o) == 5)


def _team():
    cat = Catalog()

    @cat.fn
    def roster(problem: str) -> list:                 # read from the state: held for every candidate
        return problem.split()

    @cat.fn
    def size(team: list) -> int:
        return len(team)

    @cat.check(hard=True, then={"ok": "no"})
    def known(roster: list, team: list) -> bool:
        return set(team) <= set(roster)

    @cat.check(hard=True, then={"ok": "no"})
    def no_rivals(team: list) -> bool:
        return not ({"ann", "bob"} <= set(team))

    @cat.fn
    def lead(team: list) -> str:
        if not team:
            raise ValueError("an empty team has no lead")
        return team[0]

    @cat.rule("ok")
    def ok(no_rivals, known, lead) -> bool:
        return True
    return System(cat, [Question("ok", "A valid team?", Answer.yes_no(), requires=["no_rivals", "known", "size"])])


TEAMS = [[], ["ann"], ["ann", "bob"], ["ann", "cy", "dee"], ["bob", "zed"], ["bob", "cy"], ["ann", "bob", "cy", "dee"]]


def test_every_candidate_decided_lean_has_the_verdict_and_the_values_of_a_full_ask(both):
    search(_trip(), {}, "ok", TREE, into="order", keep=10, prune=["direct", "starts_at_a"])
    search(_trip(), {}, "ok", TREE, into="order", keep=10)
    search(_team(), {"problem": "ann bob cy dee"}, "ok", TEAMS, into="team", objective="size", keep=2)
    search(_team(), {"problem": "ann bob cy dee"}, "ok", TEAMS, into="team", objective=len, hold=False)
    assert both["n"] > 150 and both["diffs"] == []


def _plain(run):
    d = run.to_dict()
    d.pop("seconds")
    d["response"].pop("ms")
    d["response"]["trace"].pop("timings")
    return d


@pytest.mark.parametrize("hold", [True, False])
def test_a_lean_search_and_a_search_of_full_asks_return_the_same_record(hold):
    for system, args, kw in [(_trip(), ({}, "ok", TREE), {"into": "order", "keep": 3, "prune": ["direct", "starts_at_a"]}),
                             (_team(), ({"problem": "ann bob cy dee"}, "ok", TEAMS), {"into": "team", "objective": "size"})]:
        lean = search(system, *args, hold=hold, **kw)
        full = search(system, *args, hold=hold, lean=False, **kw)
        assert lean.asked == full.asked > 5 and lean.rejected and _plain(lean) == _plain(full)


def test_the_winner_is_asked_in_full_and_its_trace_is_byte_for_byte_an_ordinary_ask(tmp_path):
    system = _team()
    system.storage = JSONLStorage(str(tmp_path / "t.jsonl"))
    run = search(system, {"problem": "ann bob cy dee"}, "ok", TEAMS, into="team", objective="size")
    ordinary = system.ask({"problem": "ann bob cy dee", "team": run.best}, questions=["ok"], store=False)
    assert run.best == ["ann", "cy", "dee"] and run.response.stored_id is not None
    assert all(r.hash for r in run.response.trace.records) and run.response.trace.init_hash
    assert _trace_bytes(run.response) == _trace_bytes(ordinary)
    assert run.replay(system)["ok"] and len(system.storage) == 1 and system.stats["asks"] == 2


def test_a_lean_decision_hashes_nothing_records_nothing_on_the_system_and_does_not_replay(tmp_path):
    system = _team()
    system.storage = JSONLStorage(str(tmp_path / "t.jsonl"))
    stats, costs = dict(system.stats), dict(system.cost_book.ms)
    state = {"problem": "ann bob", "team": ["ann", "bob"]}
    res = system._decide(state, ["ok"])
    assert res["ok"].answer == "no" and res["ok"].guard == "hard_check"
    assert res.trace.init_hash == "" and all(r.hash == "" and r.inputs == {} for r in res.trace.records)
    assert res.safeguards is None and res.trace.fingerprint == {} and res.stored_id is None
    assert system.stats == stats and len(system.storage) == 0 and system.cost_book.ms == costs
    assert not res.trace.replay(system)["ok"]          # no hashes: nothing to replay a kept decision from
    full = system.ask(state, ["ok"], store=False)
    assert _seen(res, ["ok"]) == _seen(full, ["ok"])


def test_an_accept_function_sees_ordinary_responses_unless_lean_is_asked_for():
    hashed = []

    def accept(res):
        hashed.append(bool(res.trace.init_hash))
        return res["ok"].answer == "yes"
    search(_team(), {"problem": "ann bob cy dee"}, "ok", TEAMS, into="team", accept=accept)
    assert hashed and all(hashed)
    hashed.clear()
    run = search(_team(), {"problem": "ann bob cy dee"}, "ok", TEAMS, into="team", accept=accept, lean=True)
    assert hashed[:-1] and not any(hashed[:-1]) and hashed[-1]       # the candidates lean, the winner's full ask not
    assert run.best == ["ann"]


# ------------------------------------------------------------------------------------------------ NATURAL PLAN (stand)
def _naturalplan():
    data = Path(os.environ.get("STAND_DATA") or ROOT / "benchmarks" / "tasks" / "data") / "naturalplan" / "prepared"
    if not (data / "trip_planning_dev.jsonl").exists():
        pytest.skip("the NATURAL PLAN data is not here (STAND_DATA)")
    for p in (ROOT / "benchmarks" / "tasks", ROOT / "benchmarks" / "tasks" / "naturalplan"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import plans
    import solution
    return data, solution, plans


@pytest.mark.stand
@pytest.mark.parametrize("kind", ["calendar_scheduling", "meeting_planning", "trip_planning"])
def test_naturalplan_dev_every_candidate_lean_and_full_agree_and_every_winner_is_an_ordinary_ask(kind, both):
    data, solution, plans = _naturalplan()
    system, space, opts = solution.SETUP[kind]()
    rows = [json.loads(line) for line in (data / f"{kind}_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    asked = 0
    for r in rows:
        state = {"problem": plans.problem_text(r)}
        run = search(system, state, solution.QUESTION, space, budget=200_000, **opts)
        asked += run.asked
        assert run.best is not None and run.exact
        cand = run.best if opts.get("into") is None else {opts["into"]: run.best}
        ordinary = system.ask({**state, **cand}, questions=[solution.QUESTION], store=False)
        assert _trace_bytes(run.response) == _trace_bytes(ordinary)
    print(f"\n{kind}: {len(rows)} problems, {asked} candidates asked both ways, {len(both['diffs'])} differences")
    assert both["n"] == asked and both["diffs"] == []
