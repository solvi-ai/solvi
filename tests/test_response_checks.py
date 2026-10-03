"""res.checks (1.0): every check of a decision as data — name, questions, status, hard, reason, the `then` it applied —
in to_dict() and in stored decisions (derived from the trace: a stored record's hash is over what was stored, so
records stored before keep their hashes and give the same list when loaded)."""
import json
import shutil
import sys
from pathlib import Path

from solvi import Catalog, Question, Response, System
from solvi.core.slow.refine import Fail
from solvi.core.response import CheckResult
from solvi.core.store import open_storage


def refunds():
    cat = Catalog()

    @cat.fn
    def days(purchase_day: int, today: int) -> int:
        return today - purchase_day

    @cat.check
    def recent(days: int) -> bool:
        "bought in the last 30 days"
        return days <= 30

    @cat.check(hard=True, then={"refund": "no"})
    def known_customer(customer: str) -> bool:
        return Fail(f"{customer} is blocked") if customer == "blocked" else True

    @cat.check(hard=True)
    def has_receipt(receipt: str) -> bool:
        return receipt.startswith("R")

    @cat.rule("refund")
    def refund(recent: bool, has_receipt: bool) -> bool:
        return recent

    @cat.rule("note")
    def note(days: int) -> bool:
        return days > 0
    return System(cat, [Question("refund", "Refund?"), Question("note", "Note?")])


OK = {"purchase_day": 1, "today": 5, "customer": "ann", "receipt": "R1"}


def test_every_check_with_its_status():
    res = refunds().ask(OK)
    by = {c.name: c for c in res.checks}
    assert [c.name for c in res.checks] == [st.part.name for st in res.flow.steps if st.part.kind == "check"]
    assert by["recent"] == CheckResult("recent", ["refund", "note"], "passed", False)
    assert by["known_customer"] == CheckResult("known_customer", ["refund"], "passed", True)
    assert all(c.passed for c in res.checks)


def test_a_failed_hard_check_with_its_reason_and_then():
    res = refunds().ask(dict(OK, customer="blocked"))
    c = {c.name: c for c in res.checks}["known_customer"]
    assert (c.status, c.hard, c.reason, c.then) == ("failed", True, "blocked is blocked", {"refund": "no"})


def test_a_failed_soft_check_says_its_docstring_and_skipped_checks_say_why():
    s = refunds()
    res = s.ask(dict(OK, today=90))
    assert {c.name: c for c in res.checks}["recent"].reason == "bought in the last 30 days"
    res = s.ask(dict(OK, receipt="none"), questions="refund")       # has_receipt fails: refund closes, recent skipped
    by = {c.name: c for c in res.checks}
    assert by["has_receipt"].status == "failed" and by["has_receipt"].then is None
    assert by["recent"].status == "skipped" and "has_receipt" in by["recent"].reason


def test_a_check_that_cannot_be_evaluated_is_an_error():
    res = refunds().ask(dict(OK, receipt=None))
    c = {c.name: c for c in res.checks}["has_receipt"]
    assert c.status == "error" and c.reason.startswith("type rejected: receipt = None")


def test_in_to_dict_and_back():
    s = refunds()
    res = s.ask(dict(OK, customer="blocked"))
    d = res.to_dict()
    assert d["checks"] == [c.to_dict() for c in res.checks]
    assert {"name": "known_customer", "questions": ["refund"], "status": "failed", "hard": True,
            "reason": "blocked is blocked", "then": {"refund": "no"}} in d["checks"]
    json.dumps(d, allow_nan=False)
    assert Response.from_json(res.to_json()).checks == res.checks
    assert Response.from_json(res.to_json(), catalog=s).checks == res.checks


def test_stored_decisions_carry_the_checks(tmp_path):
    s = refunds()
    s.storage = open_storage(str(tmp_path / "d.db"))
    s.storage.catalog = s
    res = s.ask(dict(OK, customer="blocked"))
    st = next(iter(s.storage.iter()))
    assert st.data["response"]["checks"] == res.to_dict()["checks"]
    assert s.storage.verify()["ok"] and s.storage.replay_all(s) == []
    assert s.storage.get(st.id, s).checks == res.checks


def test_a_0_9_store_keeps_its_hashes_and_lists_its_checks(tmp_path):
    data = Path(__file__).parent / "fixtures" / "store_0_9_0"
    for f in data.iterdir():
        if f.name.startswith("decisions."):
            shutil.copy(f, tmp_path / f.name)
    sys.path.insert(0, str(data))
    try:
        import task
        system = task.system(calibration=data / "calibration.json")
        store = open_storage(str(tmp_path / "decisions.jsonl"))
        store.catalog = system
        assert store.verify()["ok"]
        for s in store.iter():
            assert "checks" not in s.data["response"]            # written by 0.9: as it was
            loaded = store.get(s.id, system)
            assert isinstance(loaded.checks, list)
            assert all(isinstance(c, CheckResult) for c in loaded.checks)
    finally:
        sys.path.remove(str(data))
        sys.modules.pop("task", None)
