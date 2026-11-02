"""The document application lifecycle: checked decisions, durable review, replay and version boundaries."""
import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from invoice_workflow import DemoReader, FileInbox, InvoiceWorkflow, demo_model  # noqa: E402
from solvi.experimental.process import ProcessRunError, ProcessSpecError  # noqa: E402
from solvi.core.store import JSONLStorage  # noqa: E402
from solvi.models import llm  # noqa: E402

DOC = "invoice_number: A-1\ntotal: 123.45\ncurrency: EUR\n"
MISSING = "invoice_number: A-2\ncurrency: EUR\n"


def test_automatic_policy_and_no_statistical_promise(tmp_path):
    w = InvoiceWorkflow(tmp_path, demo_model())
    good = w.run(DOC, "123.45", "EUR")
    bad = w.run(DOC, "123.46", "EUR")
    assert (good.status, good.answer, good.automatic) == ("decided", "approve", True)
    assert (bad.status, bad.answer) == ("decided", "reject")
    state = w.operator_state(good)
    assert state["records_verified"] and state["statistical_promise"] is None
    assert state["label_source"] == "none" and not state["drift"]["watched"]
    assert state["result"]["risk"] is None


def test_missing_fact_survives_restart_then_human_answer_is_immutable(tmp_path):
    w = InvoiceWorkflow(tmp_path, demo_model())
    r = w.run(MISSING, "123.45", "EUR")
    assert r.status == "pending" and r.answer is None
    restarted = InvoiceWorkflow(tmp_path, demo_model())
    assert restarted.process.resume(r.run).status == "pending"
    restarted.inbox.answer(r.ticket, "reject", "anna", "No total")
    with pytest.raises(FileExistsError):
        FileInbox(tmp_path / "inbox").answer(r.ticket, "approve", "other")
    result = restarted.process.resume(r.run)
    assert (result.status, result.answer, result.automatic) == ("decided", "reject", False)
    assert restarted.process.replay_all()["ok"]
    with pytest.raises(ProcessRunError):
        restarted.process.resume(r.run)
    assert len(list(JSONLStorage(tmp_path / "process.jsonl").iter("human"))) == 1


def test_cancel_is_terminal_across_restart(tmp_path):
    w = InvoiceWorkflow(tmp_path, demo_model())
    r = w.run(MISSING, "10", "EUR")
    w.process.cancel(r.run, "withdrawn")
    w = InvoiceWorkflow(tmp_path, demo_model())
    assert not w.process.pending() and w.process.replay(r.run)["ok"]
    with pytest.raises(FileExistsError):
        w.inbox.answer(r.ticket, "approve", "anna")


def test_review_from_another_python_process(tmp_path):
    w = InvoiceWorkflow(tmp_path, demo_model())
    result = w.run(MISSING, "10", "EUR")
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from invoice_workflow import InvoiceWorkflow, demo_model
w = InvoiceWorkflow(sys.argv[2], demo_model())
w.inbox.answer(sys.argv[3], 'reject', 'other-process')
r = w.process.resume(sys.argv[4])
assert (r.status, r.answer, r.automatic) == ('decided', 'reject', False)
assert w.process.replay_all()['ok']
"""
    subprocess.run([sys.executable, "-c", script, str(Path(__file__).parents[1] / "examples"),
                    str(tmp_path), result.ticket, result.run], check=True, capture_output=True, timeout=30)
    assert not w.process.pending() and w.process.replay(result.run)["ok"]


def test_changed_model_is_refused_and_old_records_still_replay(tmp_path):
    w = InvoiceWorkflow(tmp_path, demo_model())
    r = w.run(DOC, "123.45", "EUR")
    model = llm("http://offline/v1", "changed-model", opener=DemoReader(), workers=1,
                logprobs=False, ask="confidence", retries=0)
    with pytest.raises(ProcessSpecError):
        InvoiceWorkflow(tmp_path, model)
    assert InvoiceWorkflow(tmp_path, demo_model()).process.replay(r.run)["ok"]


def test_reviewer_and_input_validation(tmp_path):
    w = InvoiceWorkflow(tmp_path, demo_model())
    for amount, currency in [("NaN", "EUR"), ("10", "eur")]:
        with pytest.raises(ValueError):
            w.run(DOC, amount, currency)
    r = w.run(MISSING, "10", "EUR")
    with pytest.raises(ValueError):
        w.inbox.answer(r.ticket, "unknown", "anna")
    with pytest.raises(ValueError):
        w.inbox.answer("../../escape", "approve", "anna")
