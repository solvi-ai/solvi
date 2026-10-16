"""solvi.testing.mutants: small faults made in a catalog (a comparison flipped or moved by its edge, a threshold
constant nudged, a rule branch swapped, a check dropped) and run against its cases — a mutation score for the cases."""
from __future__ import annotations

import json

import pytest

from solvi.testing import mutants

TASK = '''
from solvi import Answer, Catalog, Question

cat = Catalog()
LIMIT = 100


@cat.fn
def big(amount):
    if amount is None:
        raise ValueError("no amount")
    return amount >= LIMIT


@cat.check(hard=True, then={"action": "block"})
def not_sanctioned(country):
    """a sanctioned country is blocked"""
    return country != "XX"


@cat.rule("action")
def action(big, vip):
    if big:
        return "review" if not vip else "pay"
    else:
        return "pay"


QUESTIONS = [Question("action", "What to do?", Answer.choice(["pay", "review", "block"]))]
'''


def case(name, amount, vip, country, want):
    return {"name": name, "state": {"amount": amount, "vip": vip, "country": country}, "expected": {"action": want}}


WEAK = [case("small", 5, False, "DE", "pay"), case("huge", 1000, False, "DE", "review")]
STRONG = WEAK + [case("at the limit", 100, False, "DE", "review"), case("just under", 99, False, "DE", "pay"),
                 case("vip", 1000, True, "DE", "pay"), case("sanctioned", 5, False, "XX", "block")]


def suite(tmp_path, cases, name="t"):
    d = tmp_path / name
    d.mkdir()
    (d / "task.py").write_text(TASK)
    (d / "cases.json").write_text(json.dumps(cases))
    return d / "cases.json"


def test_the_four_kinds_are_generated_with_line_and_description():
    ms = mutants.generate(TASK)
    kinds = {m.kind for m in ms}
    assert kinds == set(mutants.KINDS)
    d = {(m.kind, m.description) for m in ms}
    assert ("flip", ">= → <") in d and ("boundary", ">= → >") in d and ("flip", "!= → ==") in d
    assert ("constant", "100 → 101") in d and ("constant", "100 → 99") in d
    assert ("drop", "check not_sanctioned() always passes") in d
    assert ("drop", "guard 'if amount is None: raise' never raises") in d
    assert ("branch", "if / else bodies swapped") in d and ("branch", "a if c else b → b if c else a") in d
    assert all(m.line > 0 and m.source != TASK for m in ms)
    assert {m.kind for m in mutants.generate(TASK, kinds=("flip",))} == {"flip"}


def test_weak_cases_let_faults_through_and_strong_cases_kill_them(tmp_path):
    weak = mutants.run(suite(tmp_path, WEAK, "w"))
    strong = mutants.run(suite(tmp_path, STRONG, "s"))
    assert weak.total == strong.total > 0 and weak.score < strong.score
    surv = {(m.kind, m.description) for m in weak.survivors()}
    assert ("boundary", ">= → >") in surv                         # nobody tested the edge
    assert ("drop", "check not_sanctioned() always passes") in surv
    kept = {(m.kind, m.description) for m in strong.survivors()}
    assert ("boundary", ">= → >") not in kept and ("drop", "check not_sanctioned() always passes") not in kept
    killed = next(m for m in strong.mutants if m.description == "check not_sanctioned() always passes")
    assert killed.status == "killed" and killed.killed_by == "sanctioned"
    d = strong.to_dict()
    assert d["score"] == strong.score and set(d["by_kind"]) <= set(mutants.KINDS) and d["seconds"] > 0
    # the original task file is never written
    assert (tmp_path / "s" / "task.py").read_text() == TASK


def test_a_failing_suite_is_refused_and_limit_spreads_the_mutants(tmp_path):
    bad = suite(tmp_path, [case("wrong", 5, False, "DE", "review")], "b")
    with pytest.raises(ValueError, match="fail on the catalog as it is"):
        mutants.run(bad)
    rep = mutants.run(suite(tmp_path, STRONG, "l"), limit=4)
    assert len(rep.mutants) == 4 and len({m.line for m in rep.mutants}) > 1


def test_a_mutant_that_cannot_be_built_is_stillborn_not_scored(tmp_path):
    p = suite(tmp_path, STRONG, "x")
    (p.parent / "task.py").write_text(TASK + "\n_PINNED = {100: 1}[LIMIT]\n")   # a KeyError at load
    rep = mutants.run(p, kinds=("constant",))
    still = [m for m in rep.mutants if m.status == "stillborn"]
    assert {m.description for m in still} == {"100 → 101", "100 → 99"} and all("KeyError" in m.killed_by for m in still)
    assert rep.stillborn == 2 and rep.total == len(rep.mutants) - 2


def test_the_command_line(tmp_path, capsys):
    p = suite(tmp_path, WEAK, "c")
    assert mutants.main([str(p), "--kinds", "boundary,drop"]) == 0
    out = capsys.readouterr().out
    assert "score" in out and "survived" in out
    assert mutants.main([str(p), "--kinds", "nope"]) == 2
    assert mutants.main([str(p), "--json", "--kinds", "flip"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["score"] is not None
    from solvi.cli import main as solvi_main
    assert solvi_main(["mutants", str(p), "--kinds", "flip"]) == 0
    assert "score" in capsys.readouterr().out
