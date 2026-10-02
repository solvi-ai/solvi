"""res.audit(): the audit is data — to_dict() is JSON-ready like every other to_dict(); an unknown question is named."""
import json
from datetime import date

import pytest

from solvi import Answer, Catalog, Question, System


def _leave():
    cat = Catalog()

    @cat.fn
    def days_requested(start, end):
        return (end - start).days + 1

    @cat.check(hard=True, then={"approve": "reject"})
    def enough_balance(balance, days_requested) -> bool:
        return balance >= days_requested

    @cat.rule("approve")
    def approve(days_requested):
        return "approve" if days_requested < 5 else "needs_manager"
    return System(cat, [Question("approve", "Approve?", Answer.choice(["approve", "needs_manager", "reject"]),
                                 requires=["enough_balance"])])


STATE = {"start": date(2026, 10, 19), "end": date(2026, 10, 23), "balance": 14}


def test_an_audit_with_dates_in_its_inputs_is_json_ready():
    """json.dumps(res.audit("approve").to_dict()) raised TypeError: Object of type date is not JSON serializable."""
    res = _leave().ask(STATE)
    one = res.audit("approve").to_dict()
    text = json.dumps(one)
    assert "2026-10-19" in text and one["answer"] == "needs_manager"
    whole = res.audit().to_dict()
    assert json.loads(json.dumps(whole))["answers"]["approve"] == json.loads(text)


def test_auditing_a_question_that_was_not_asked_names_what_was():
    res = _leave().ask(STATE)
    with pytest.raises(KeyError, match="no answer to 'nope' in this response: it answered approve"):
        res.audit("nope")
    with pytest.raises(KeyError, match="no audit of 'nope'"):
        res.audit()["nope"]
