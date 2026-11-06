"""Integration review of paid CEGIS, active observations and durable recovery."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience._common import BudgetStop, WorkBudget, digest
from solvi.experimental.experience.domain_checks import DomainReceipt, FiniteDomain
from solvi.experimental.experience.journal import ExperienceJournal
from solvi.experimental.experience.observations import ObservationOption
from solvi.experimental.experience.programs import Grammar, synthesize
from solvi.experimental.experience.refinement import RefinementSession


class CachedOracle:
    source_id = "audit"
    source_revision = "review-v1"

    def __init__(self, path, *, offset=False):
        self.path, self.offset = path, offset
        self.cache = json.loads(path.read_text()) if path.exists() else {}
        self.calls = []
        self.reconciliations = []
        self.interrupt_observation = False
        self.invalid = False
        self.interrupt_verification = False

    def read(self, request_id, revision, point):
        assert request_id not in self.cache, "oracle read replayed"
        assert revision == self.source_revision
        self.calls.append(request_id)
        value = point["x"] + 1 if self.offset else 2 * point["x"]
        receipt = DomainReceipt(
            request_id,
            "external:" + request_id,
            self.source_id,
            revision,
            digest(point),
            True if self.invalid else value,
        )
        self.cache[request_id] = receipt.to_dict()
        with self.path.open("w") as stream:
            json.dump(self.cache, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if self.interrupt_verification and ":round:" in request_id:
            raise RuntimeError("verification reply interrupted")
        if self.interrupt_observation and ":observation:cheap" in request_id:
            raise RuntimeError("reply interrupted after external read")
        return receipt

    def reconcile(self, request_id):
        self.reconciliations.append(request_id)
        record = self.cache.get(request_id)
        return None if record is None else DomainReceipt(**record)


def fixture(tmp_path, *, mode="active", offset=False, budget=None, reopen=False):
    grammar = Grammar(("x",), operators=("add",), constants=(0, 1), max_depth=2, max_nodes=3, max_candidates=100)
    domain = FiniteDomain(
        "review",
        "v1",
        "scalar",
        grammar,
        "audit",
        "review-v1",
        [{"x": 0}, {"x": 1}, {"x": 2}],
        budget=WorkBudget(10000),
    )
    options = [
        ObservationOption("initial", {"x": 0}, 1),
        ObservationOption("cheap", {"x": 1}, 2),
        ObservationOption("expensive", {"x": 2}, 100),
    ]
    oracle = CachedOracle(tmp_path / "oracle.json", offset=offset)
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    journal = ExperienceJournal(store, "refinement-review", {})
    if reopen:
        session = RefinementSession.restore(
            journal=journal, grammar=grammar, domain=domain, oracle=oracle, options=options
        )
    else:
        session = RefinementSession(
            grammar,
            domain,
            oracle,
            options,
            budget or WorkBudget(100000),
            session_id="review",
            initial=["initial"],
            mode=mode,
            max_rounds=6,
            committee_size=3,
            max_observations=1,
            journal=journal,
        )
    return session, oracle, journal


def test_true_counterexample_loop_changes_program_and_terminal_replay(tmp_path):
    session, oracle, journal = fixture(tmp_path, mode="none", offset=True)
    result = session.run()
    assert result["status"] == "verified"
    assert session.state["rounds"] >= 2
    assert session.state["rejected"]
    assert session.state["rejected"][0] != result["program"]
    assert any(row["origin"] == "counterexample" for row in session.state["examples"])
    assert session.state["checks"][0]["reason"] == "counterexample"
    before = dict(session.budget.counts)
    reads = dict(oracle.cache)
    assert session.run() == result and session.budget.counts == before
    restored, fresh_oracle, fresh_journal = fixture(tmp_path, offset=True, reopen=True)
    assert restored.run() == result
    assert restored.budget.counts == before and fresh_oracle.calls == [] and fresh_oracle.cache == reads
    assert fresh_journal.verify() and journal.verify()


def test_active_uses_real_informative_cheap_read_not_gold_for_selection(tmp_path):
    session, oracle, _ = fixture(tmp_path)
    result = session.run()
    assert result["status"] == "verified"
    assert session.state["observations"] == 1
    active = [row for row in session.state["examples"] if row["origin"] == "observation"]
    assert len(active) == 1 and active[0]["inputs"] == {"x": 1} and active[0]["expected"] == 2
    assert "review:observation:cheap" in oracle.calls
    assert "review:observation:expensive" not in oracle.calls
    assert session.budget.counts["observation_compare"] > 0
    assert session.budget.counts["refinement_observation"] == 3


@pytest.mark.parametrize("boundary", ["selection", "intent", "receipt", "read"])
def test_fresh_reopen_selection_intent_read_receipt_never_replays_oracle(tmp_path, boundary):
    session, oracle, journal = fixture(tmp_path)
    checkpoint = journal.checkpoint
    hit = []

    def injected(state):
        if boundary == "selection":
            target = state["phase"] == "observe" and state["rounds"] == 1
        else:
            target = state.get("selection", {}).get("origin") == "observation"
            target = target and (
                state["pending"] is not None
                if boundary == "intent"
                else state["pending"] is None and "review:observation:cheap" in state["receipts"]
            )
        result = checkpoint(state)
        if target and not hit and boundary != "read":
            hit.append(True)
            raise RuntimeError("checkpoint fault " + boundary)
        return result

    journal.checkpoint = injected
    if boundary == "read":
        oracle.interrupt_observation = True
        assert session.run()["status"] == "blocked"
    else:
        with pytest.raises(RuntimeError, match="checkpoint fault"):
            session.run()
        assert hit
    reads = set(oracle.cache)
    spent = session.budget.spent
    restored, reopened_oracle, _ = fixture(tmp_path, reopen=True)
    if restored.state["pending"] is not None and boundary == "intent":
        # Intent preceded the read. Host reconciliation can confirm no receipt;
        # the session must block, not silently perform that ambiguous request.
        assert restored.run()["status"] == "blocked"
        assert reopened_oracle.calls == [] and set(reopened_oracle.cache) == reads
    else:
        assert restored.run()["status"] == "verified"
        assert not reads.intersection(reopened_oracle.calls)
    assert restored.budget.spent >= spent


def test_unknown_read_without_reconciliation_and_source_drift_fail_closed(tmp_path):
    session, oracle, _ = fixture(tmp_path)
    oracle.interrupt_observation = True
    assert session.run()["status"] == "blocked"
    pending = session.state["pending"]["request_id"]
    restored, fresh_oracle, _ = fixture(tmp_path, reopen=True)
    fresh_oracle.reconcile = None
    assert restored.run()["status"] == "blocked"
    assert fresh_oracle.calls == [] and restored.state["pending"]["request_id"] == pending
    restored, fresh_oracle, _ = fixture(tmp_path, reopen=True)
    fresh_oracle.source_revision = "review-v2"
    assert restored.run()["status"] == "unknown"
    assert restored.state["reason"] == "source_changed" and fresh_oracle.calls == []


def test_invalid_receipt_never_becomes_an_example(tmp_path):
    session, oracle, _ = fixture(tmp_path)
    oracle.invalid = True
    with pytest.raises(ValueError, match="invalid refinement oracle receipt"):
        session.run()
    assert not session.state["examples"] and session.state["pending"] is not None
    restored, fresh_oracle, _ = fixture(tmp_path, reopen=True)
    with pytest.raises(ValueError, match="invalid refinement oracle receipt"):
        restored.run()
    assert fresh_oracle.calls == [] and not restored.state["examples"]


def test_budget_and_post_debit_sink_failure(tmp_path):
    session, _, _ = fixture(tmp_path, budget=WorkBudget(0))
    assert session.run()["reason"] == "budget"
    other = tmp_path / "sink"
    other.mkdir()
    session, _, journal = fixture(other)
    original = journal.debit

    def durable_then_fail(snapshot):
        result = original(snapshot)
        if snapshot["counts"].get("program_node"):
            raise BudgetStop("durable sink lost after append")
        return result

    journal.debit = durable_then_fail
    with pytest.raises(BudgetStop, match="durable sink lost after append"):
        session.run()
    spent = session.budget.spent
    counts = session.budget.counts.copy()
    assert counts["program_node"] > 0
    with pytest.raises(RuntimeError, match="failed writer"):
        session.step()
    restored, oracle, _ = fixture(other, reopen=True)
    assert restored.budget.spent >= spent
    for category, value in counts.items():
        assert restored.budget.counts.get(category, 0) >= value
    assert restored.run()["status"] == "verified"
    assert "review:observation:initial" not in oracle.calls


def test_excluded_first_match_changes_committee_without_claiming_all_programs():
    grammar = Grammar(("x",), operators=("add",), max_depth=2, max_nodes=3)
    rows = [({"x": 0}, 0)]
    first = synthesize(rows, grammar, WorkBudget(10000))
    budget = WorkBudget(10000)
    second = synthesize(rows, grammar, budget, exclude=(first.candidate.fingerprint,))
    assert first.candidate.fingerprint != second.candidate.fingerprint
    assert second.candidate.evaluate({"x": 0}, budget, grammar) == 0
    assert budget.counts["search_exclusion"] > 0


@pytest.mark.parametrize("boundary", ["intent", "receipt", "read"])
def test_verification_read_boundaries_reopen_without_oracle_replay(tmp_path, boundary):
    session, oracle, journal = fixture(tmp_path, mode="none")
    checkpoint = journal.checkpoint
    hit = []

    def injected(state):
        pending = state.get("pending")
        target = (
            (pending is not None and pending["purpose"] == "verification")
            if boundary == "intent"
            else (state["phase"] == "check" and pending is None and any(":round:" in key for key in state["receipts"]))
        )
        result = checkpoint(state)
        if target and not hit and boundary != "read":
            hit.append(True)
            raise RuntimeError("verification checkpoint failed")
        return result

    journal.checkpoint = injected
    if boundary == "read":
        oracle.interrupt_verification = True
        assert session.run()["status"] == "blocked"
    else:
        with pytest.raises(RuntimeError):
            session.run()
        assert hit
    paid = session.budget.spent
    reads = set(oracle.cache)
    restored, reopened_oracle, _ = fixture(tmp_path, mode="none", reopen=True)
    result = restored.run()
    if boundary == "intent":
        assert result["status"] == "blocked" and reopened_oracle.calls == []
    else:
        assert result["status"] == "verified"
        assert not reads.intersection(reopened_oracle.calls)
    assert restored.budget.spent >= paid


@pytest.mark.parametrize("excluded", [(None,), (True,), (1,), ("x",) * 33, "fingerprint"])
def test_exclusion_rejects_malformed_or_oversized_inputs(excluded):
    grammar = Grammar(("x",))
    with pytest.raises(ValueError):
        synthesize([({"x": 0}, 0)], grammar, WorkBudget(1000), exclude=excluded)


def test_exclusion_refuses_unbounded_iterable_without_consuming_and_pays_cap():
    visited = []

    def unbounded():
        while True:
            visited.append(True)
            yield "fingerprint"

    grammar = Grammar(("x",))
    with pytest.raises(ValueError, match="32"):
        synthesize([({"x": 0}, 0)], grammar, WorkBudget(1000), exclude=unbounded())
    assert not visited
    budget = WorkBudget(1)
    result = synthesize([({"x": 0}, 0)], grammar, budget, exclude=("first", "second"))
    assert result.reason == "budget_exhausted" and result.tried == 0
    assert budget.spent == 1 and budget.counts == {"search_exclusion": 1}


def test_durable_constructor_rejects_external_sink_before_any_state_write(tmp_path):
    invoked = []

    def sink(snapshot):
        invoked.append(snapshot)
        raise RuntimeError("must not execute")

    budget = WorkBudget(1000, on_charge=sink)
    with pytest.raises(ValueError, match="durable refinement owns"):
        fixture(tmp_path, budget=budget)
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    journal = ExperienceJournal(store, "refinement-review", {})
    assert journal.records() == []
    assert not (tmp_path / "oracle.json").exists()
    assert budget.spent == 0 and invoked == []


@pytest.mark.parametrize("exception", [ValueError, BudgetStop])
def test_volatile_setup_sink_failure_preserves_debit_and_poisons(tmp_path, exception):
    reference, oracle, _ = fixture(tmp_path)

    def sink(snapshot):
        if snapshot["counts"].get("refinement_setup"):
            raise exception("volatile setup sink failed")

    session = RefinementSession(
        reference.grammar,
        reference.domain,
        oracle,
        reference.options,
        WorkBudget(1000, on_charge=sink),
        session_id="volatile",
        initial=["initial"],
        journal=None,
    )
    assert session.budget.spent == 0
    with pytest.raises(exception, match="volatile setup sink failed"):
        session.step()
    assert session.budget.counts["refinement_setup"] == 7 and not oracle.calls
    with pytest.raises(RuntimeError, match="failed writer"):
        session.step()


@pytest.mark.skipif(os.name != "posix", reason="actual SIGKILL requires POSIX")
def test_sigkill_after_fsynced_external_receipt_reconciles_fresh_without_replay(tmp_path):
    test_directory = Path(__file__).resolve().parent
    code = (
        "import json,os,signal,sys\n"
        f"sys.path.insert(0,{str(test_directory)!r})\n"
        "from pathlib import Path\n"
        "from test_experience_refinement import fixture\n"
        f"directory=Path({str(tmp_path)!r})\n"
        "session,oracle,journal=fixture(directory)\n"
        "original=oracle.read\n"
        "def killed_read(request,revision,point):\n"
        "    receipt=original(request,revision,point)\n"
        "    if ':observation:cheap' in request:\n"
        "        with (directory/'sigkill-marker.json').open('w') as stream:\n"
        "            json.dump(dict(request_id=request,spent=session.budget.spent,counts=session.budget.counts),stream)\n"
        "            stream.flush();os.fsync(stream.fileno())\n"
        "        os.kill(os.getpid(),signal.SIGKILL)\n"
        "    return receipt\n"
        "oracle.read=killed_read\n"
        "session.run()\n"
        "raise AssertionError('SIGKILL boundary not reached')\n"
    )
    with (tmp_path / "child.log").open("w") as stream:
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        try:
            child.wait(timeout=60)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    assert child.returncode == -signal.SIGKILL, (tmp_path / "child.log").read_text()
    marker = json.loads((tmp_path / "sigkill-marker.json").read_text())
    cached_before = json.loads((tmp_path / "oracle.json").read_text())
    assert marker["request_id"] in cached_before
    restored, oracle, journal = fixture(tmp_path, reopen=True)
    assert restored.state["pending"]["request_id"] == marker["request_id"]
    assert restored.budget.spent >= marker["spent"]
    for category, value in marker["counts"].items():
        assert restored.budget.counts.get(category, 0) >= value
    assert restored.run()["status"] == "verified"
    assert marker["request_id"] in oracle.reconciliations
    assert not set(cached_before).intersection(oracle.calls)
    assert restored.budget.spent >= marker["spent"] and journal.verify()
    terminal_counts = dict(restored.budget.counts)
    again, fresh_oracle, _ = fixture(tmp_path, reopen=True)
    assert again.run()["status"] == "verified" and fresh_oracle.calls == []
    assert again.budget.counts == terminal_counts


def test_restore_rejects_option_bool_int_alias_without_reading_oracle(tmp_path):
    session, oracle, journal = fixture(tmp_path)
    changed = [
        ObservationOption(option.name, {"x": True} if option.name == "cheap" else option.inputs, option.cost)
        for option in session.options
    ]
    # Python dict equality would accept True as the saved integer 1.
    assert changed[1].inputs == session.options[1].inputs
    before = session.budget.counts.copy()
    with pytest.raises(ValueError, match="recovery contract changed"):
        RefinementSession.restore(
            journal=journal, grammar=session.grammar, domain=session.domain, oracle=oracle, options=changed
        )
    assert oracle.calls == [] and session.budget.counts == before


def test_restore_consumes_option_generator_once_and_retains_live_options(tmp_path):
    session, _, _ = fixture(tmp_path)
    options = tuple(session.options)
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    journal = ExperienceJournal(store, "refinement-review", {})
    oracle = CachedOracle(tmp_path / "oracle.json")
    visited = []

    def supplied():
        for option in options:
            visited.append(option.name)
            yield option

    restored = RefinementSession.restore(
        journal=journal, grammar=session.grammar, domain=session.domain, oracle=oracle, options=supplied()
    )
    assert visited == [option.name for option in options]
    assert restored.options == options and len(restored.options) == 3
    assert restored.run()["status"] == "verified"
    assert visited == [option.name for option in options]
    assert "review:observation:cheap" in oracle.calls


@pytest.mark.parametrize("boundary", ["constructor_options", "constructor_initial", "restore_options"])
def test_host_iterables_are_consumed_at_most_257_items(tmp_path, boundary):
    reference, oracle, journal = fixture(tmp_path)
    visited = []

    def unbounded():
        index = 0
        while True:
            visited.append(index)
            yield "initial" if boundary == "constructor_initial" else ObservationOption(str(index), {"x": 0}, 1)
            index += 1

    with pytest.raises(ValueError):
        if boundary == "restore_options":
            RefinementSession.restore(
                journal=journal, grammar=reference.grammar, domain=reference.domain, oracle=oracle, options=unbounded()
            )
        else:
            RefinementSession(
                reference.grammar,
                reference.domain,
                oracle,
                unbounded() if boundary == "constructor_options" else reference.options,
                WorkBudget(1000),
                session_id="bounded-host",
                journal=None,
                initial=unbounded() if boundary == "constructor_initial" else (),
            )
    assert len(visited) == 257 and not oracle.calls
