"""Versioned active planning windows and pinned durable recovery."""

from pathlib import Path
import subprocess
import sys

import pytest

from solvi.experimental.experience import CommitteeSearchResult, select_observation_bounded, synthesize_committee
from solvi.experimental.experience._common import BudgetStop, WorkBudget
from solvi.experimental.experience.observations import ObservationOption
from solvi.experimental.experience.programs import synthesize
from solvi.experimental.experience.refinement import RefinementSession
from test_experience_refinement import fixture


def capped(tmp_path, cap):
    session, oracle, journal = fixture(tmp_path)
    session.config["active_work_limit"] = cap
    session._save()
    return session, oracle, journal


def test_new_exports_and_zero_cap_falls_back_to_check(tmp_path):
    session, oracle, _ = capped(tmp_path, 0)
    assert session.config["active_policy"] == "onepass-bounded-v1"
    assert CommitteeSearchResult and callable(select_observation_bounded)
    session.step()  # initial read
    session.step()  # one-pass first proposal survives the continuation cap
    assert session.state["phase"] == "observe" and len(session.state["committee"]) == 1
    assert session.state["planning_remaining"] == 0
    session.step()
    assert session.state["phase"] == "check"
    assert session.run()["status"] == "verified"
    assert session.state["observations"] == 0
    assert "review:observation:cheap" not in oracle.calls


def test_combined_extra_search_and_selection_are_bounded(tmp_path):
    session, _, _ = capped(tmp_path, 100)
    session.step()
    rows = [(row["inputs"], row["expected"]) for row in session.state["examples"]]
    first = WorkBudget(100000)
    synthesize(rows, session.grammar, first)
    before = session.budget.spent
    session.step()
    search_extra = session.budget.spent - before - first.spent
    before_selection = session.budget.spent
    session.step()
    # Any actual observation purchase is outside the planning window.
    read = 2 if session.state["observations"] else 0
    selector_spent = session.budget.spent - before_selection - read
    assert 0 <= search_extra + selector_spent <= 100
    assert session.state["planning_remaining"] == 100 - search_extra - selector_spent


def test_third_member_cap_retains_first_two(tmp_path):
    session, _, _ = fixture(tmp_path)
    session.step()
    rows = [(row["inputs"], row["expected"]) for row in session.state["examples"]]
    pair = synthesize_committee(rows, session.grammar, WorkBudget(100000), size=2)
    session.config["active_work_limit"] = pair.continuation_spent
    session.step()
    assert len(session.state["committee"]) == 2
    assert session.state["candidate"] == pair.candidates[0].to_dict()
    assert session.state["status"] == "running"
    session.step()
    assert session.state["phase"] == "check"


def test_huge_unused_read_fee_reserves_check_and_initial_fee_is_excluded(tmp_path):
    base, oracle, _ = fixture(tmp_path)
    opts = [base.options[0], ObservationOption("cheap", {"x": 1}, 10**100)]
    session = RefinementSession(base.grammar, base.domain, oracle, opts, WorkBudget(100000),
                                session_id="huge", initial=["initial"])
    session.step()
    session.step()
    assert len(session.state["committee"]) == 1
    session.step()
    assert session.state["phase"] == "check"
    assert session.run()["status"] == "verified"
    assert session.state["observations"] == 0


@pytest.mark.parametrize("cap", [True, -1, 100001, 1.0])
def test_constructor_rejects_bad_cap_before_reads(tmp_path, cap):
    base, oracle, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        RefinementSession(base.grammar, base.domain, oracle, base.options, WorkBudget(1000),
                          session_id="bad", active_work_limit=cap)
    assert oracle.calls == []


def test_old_checkpoint_pins_legacy_selector_and_search(tmp_path, monkeypatch):
    session, _, _ = fixture(tmp_path)
    session.config.pop("active_policy")
    session.config.pop("active_work_limit")
    session._save()
    restored, oracle, _ = fixture(tmp_path, reopen=True)

    def forbidden(*args, **kwargs):
        pytest.fail("legacy journal silently switched to one-pass search")

    monkeypatch.setattr("solvi.experimental.experience.refinement.synthesize_committee", forbidden)
    monkeypatch.setattr("solvi.experimental.experience.observations.select_observation_bounded", forbidden)
    assert restored.run()["status"] == "verified"
    assert "active_policy" not in restored.config
    assert "review:observation:cheap" in oracle.calls
    assert restored.budget.counts["observation_compare"] > 0


@pytest.mark.parametrize("field,value", [("active_policy", "future-v9"), ("active_work_limit", True),
                                         ("committee_size", True), ("max_rounds", 65), ("seed", "1")])
def test_restore_rejects_unrecognized_or_malformed_config_before_oracle(tmp_path, field, value):
    session, oracle, journal = fixture(tmp_path)
    session.config[field] = value
    session._save()
    with pytest.raises(ValueError):
        RefinementSession.restore(journal=journal, grammar=session.grammar, domain=session.domain,
                                  oracle=oracle, options=session.options)
    assert oracle.calls == []


def test_observe_reopen_does_not_refill_window_after_durable_sink_failure(tmp_path):
    session, _, journal = capped(tmp_path, 100)
    session.step()
    session.step()
    original_remaining = session.state["planning_remaining"]
    if original_remaining == 0:
        pytest.fail("fixture must leave selector work")
    original = journal.debit
    failure = BudgetStop("persisted selector debit then lost writer")

    def fail_after_debit(snapshot):
        original(snapshot)
        if snapshot["counts"].get("observation_plan"):
            raise failure

    journal.debit = fail_after_debit
    with pytest.raises(BudgetStop) as caught:
        session.step()
    assert caught.value is failure
    restored, _, _ = fixture(tmp_path, reopen=True)
    assert restored.config["active_work_limit"] == 100
    remaining = restored.state["planning_remaining"] - (
        restored.budget.spent - restored.state["planning_ledger_spent"]
    )
    assert remaining < original_remaining
    assert restored.run()["status"] == "verified"


def test_search_continuation_sink_failure_restores_first_without_refilling_cap(tmp_path, monkeypatch):
    session, _, journal = capped(tmp_path, 1)
    session.step()
    original = journal.debit
    failure = BudgetStop("persisted first continuation debit")

    def fail_after_debit(snapshot):
        original(snapshot)
        if snapshot["counts"].get("search_exclusion"):
            raise failure

    journal.debit = fail_after_debit
    with pytest.raises(BudgetStop) as caught:
        session.step()
    assert caught.value is failure
    assert session.state["phase"] == "search" and session.state["rounds"] == 0
    assert session.state["candidate"] is None and session.state["search_first"] is not None
    assert session.budget.spent - session.state["search_plan_mark"] == 1
    restored, oracle, _ = fixture(tmp_path, reopen=True)
    saved = restored.state["search_first"]
    before = restored.budget.snapshot()

    def forbidden(*args, **kwargs):
        pytest.fail("recovery replayed synthesis after a durable first match")

    monkeypatch.setattr("solvi.experimental.experience.refinement.synthesize_committee", forbidden)
    restored.step()
    assert restored.state["phase"] == "observe" and restored.state["rounds"] == 1
    assert restored.state["candidate"] == saved and restored.state["committee"] == [saved]
    assert restored.state["planning_remaining"] == 0
    assert restored.budget.snapshot() == before and oracle.calls == []
    restored.step()
    assert restored.state["phase"] == "check" and restored.budget.snapshot() == before


def test_process_exit_after_durable_continuation_debit_keeps_window(tmp_path):
    script = r'''
import os
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[2])
from test_experience_refinement import fixture
session, oracle, journal = fixture(Path(sys.argv[1]))
session.config["active_work_limit"] = 1
session._save()
session.step()
original = journal.debit
def crash(snapshot):
    original(snapshot)
    if snapshot["counts"].get("search_exclusion"):
        os._exit(73)
journal.debit = crash
session.step()
raise RuntimeError("crash boundary was not reached")
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(Path(__file__).parent)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 73, process.stdout + process.stderr
    restored, oracle, journal = fixture(tmp_path, reopen=True)
    assert restored.state["phase"] == "search" and restored.state["rounds"] == 0
    saved = restored.state["search_first"]
    assert restored.budget.spent - restored.state["search_plan_mark"] == 1
    spent = restored.budget.spent
    restored.step()
    assert restored.state["candidate"] == saved and restored.state["rounds"] == 1
    assert restored.state["planning_remaining"] == 0 and restored.budget.spent == spent
    restored.step()
    assert restored.state["phase"] == "check" and restored.budget.spent == spent
    assert oracle.calls == [] and journal.verify()


@pytest.mark.parametrize("error_type", [ValueError, BudgetStop])
def test_first_match_checkpoint_failure_poisons_writer_and_preserves_prefix(tmp_path, error_type):
    session, _, journal = capped(tmp_path, 1)
    session.step()
    checkpoint = journal.checkpoint
    failure = error_type("first-match checkpoint failed after persistence")

    def fail_after_checkpoint(state):
        checkpoint(state)
        if state.get("search_first") is not None:
            raise failure

    journal.checkpoint = fail_after_checkpoint
    with pytest.raises(error_type) as caught:
        session.step()
    assert caught.value is failure and session._failed
    with pytest.raises(RuntimeError, match="failed writer"):
        session.step()
    restored, oracle, _ = fixture(tmp_path, reopen=True)
    assert restored.budget.spent == restored.state["search_plan_mark"]
    saved = restored.state["search_first"]
    restored.step()
    assert restored.state["candidate"] == saved and restored.state["rounds"] == 1
    assert restored.state["planning_remaining"] == 1 and oracle.calls == []
