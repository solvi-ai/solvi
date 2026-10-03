"""solvi.core.slow.refine: a check that says why (Fail — falsy, recorded, in the answer's why, the audit and the replay; a plain
False still works), and the loop propose → check → re-ask with the reasons → escalate: stops at the first accepted
proposal, escalates after N rounds with the last reasons, feeds back a rejected reply, ends on a proposer that fails,
lets the System generate with the feedback as a fact, and replays — every round's trace and what the loop did."""
import io
import json

import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, System
from solvi.core.slow.generate import generator
from solvi.core.slow.refine import Fail, Refinement, failed_checks, refine


class Slot(BaseModel):
    day: str
    start: int


def _system(storage=None):
    cat = Catalog()

    @cat.fn
    def slot(proposal: Slot) -> Slot:
        return proposal

    @cat.check(hard=True, then={"ok": "no"})
    def allowed_day(slot: Slot) -> bool:
        return True if slot.day in ("Mon", "Tue") else Fail(f"{slot.day} is not allowed (Mon, Tue)")

    @cat.check(hard=True, then={"ok": "no"})
    def nobody_busy(slot: Slot) -> bool:
        """Nobody has a meeting in the slot."""
        return slot.start not in (13, 14)

    @cat.check
    def morning(slot: Slot) -> bool:
        return slot.start < 12

    @cat.rule("ok")
    def ok(allowed_day, nobody_busy) -> bool:
        return True
    return System(cat, [Question("ok", "Does the slot work?", Answer.yes_no(),
                                 requires=["allowed_day", "nobody_busy"])], storage=storage), cat


def test_fail_is_false_everywhere_and_its_reasons_are_recorded_shown_and_in_the_why():
    assert not Fail("x") and Fail("a", ["b", "c"]).reasons == ["a", "b", "c"] and Fail().reasons == []
    s, _ = _system()
    res = s.ask({"proposal": Slot(day="Sun", start=13)})
    r = res["ok"]
    assert r.answer == "no" and r.status == "forced"
    assert r.why == "hard check allowed_day is false: Sun is not allowed (Mon, Tue)"
    rec = next(x for x in res.trace.records if x.name == "allowed_day")
    assert rec.value is False and rec.extra == {"reasons": ["Sun is not allowed (Mon, Tue)"]}
    assert "allowed_day = False (hard, decides the answer): Sun is not allowed (Mon, Tue)" in str(res.audit("ok"))
    assert res.trace.replay(s)["ok"]
    assert [(f.check, f.hard, f.reasons) for f in failed_checks(res, "ok")] == [
        ("allowed_day", True, ["Sun is not allowed (Mon, Tue)"]), ("nobody_busy", True, ["Nobody has a meeting in the slot."])]
    res = s.ask({"proposal": Slot(day="Mon", start=15)})              # a soft check that is False: its default reason
    assert [(f.check, f.hard, f.reasons) for f in failed_checks(res)] == [("morning", False, ["morning is false"])]


def test_a_check_whose_reasons_change_is_a_replay_mismatch():
    cat = Catalog()
    said = {"why": "first reason"}

    @cat.check(hard=True, then={"ok": "no"})
    def c(x) -> bool:
        return Fail(said["why"])

    @cat.rule("ok")
    def ok(c) -> bool:
        return True
    s = System(cat, [Question("ok", "?", Answer.yes_no(), requires=["c"])])
    res = s.ask({"x": 1})
    said["why"] = "another reason"
    rep = res.trace.replay(s)
    assert not rep["ok"] and "reasons ['first reason'] ≠ recomputed ['another reason']" in rep["mismatches"][0][2]


def proposals(*slots):
    seen = []

    def propose(state, rounds):
        seen.append([r.feedback for r in rounds])
        return slots[min(len(seen) - 1, len(slots) - 1)]
    propose.seen = seen
    return propose


def test_the_loop_stops_at_the_first_accepted_proposal_and_feeds_back_the_reasons_of_the_failed_hard_checks():
    s, _ = _system()
    p = proposals(Slot(day="Sun", start=13), Slot(day="Mon", start=13), Slot(day="Mon", start=15), Slot(day="Tue", start=9))
    run = refine(s, {}, "ok", propose=p, into="proposal", rounds=4, accept="yes")
    assert run.accepted and run.proposal == Slot(day="Mon", start=15) and run.escalation is None and len(run.rounds) == 3
    assert p.seen == [[], [["Sun is not allowed (Mon, Tue)", "Nobody has a meeting in the slot."]],
                      [["Sun is not allowed (Mon, Tue)", "Nobody has a meeting in the slot."],
                       ["Nobody has a meeting in the slot."]]]
    assert run.result.answer == "yes" and run.rounds[-1].feedback is None
    rep = run.replay(s)
    assert rep["ok"] and rep["feedback"] == "checked" and rep["rounds"] == 3


def test_the_loop_escalates_after_the_last_round_with_its_reasons():
    s, _ = _system()
    run = refine(s, {}, "ok", propose=proposals(Slot(day="Sun", start=9)), into="proposal", rounds=2)
    assert not run.accepted and run.proposal is None and len(run.rounds) == 2
    assert run.escalation == "not accepted after 2 round(s): Sun is not allowed (Mon, Tue)"
    assert run.replay(s)["ok"]


def test_accept_checks_takes_a_proposal_that_passes_every_governing_hard_check_whatever_the_soft_ones_say():
    s, cat = _system()
    run = refine(s, {}, "ok", propose=proposals(Slot(day="Mon", start=15)), into="proposal")
    assert run.accepted and run.rounds[0].failed[0].check == "morning" and run.rounds[0].reasons == []
    run = refine(s, {}, "ok", propose=proposals(Slot(day="Mon", start=15), Slot(day="Mon", start=9)), into="proposal",
                 accept=lambda res: res.values["morning"])
    assert run.proposal == Slot(day="Mon", start=9)
    with pytest.raises(ValueError, match="pass it again"):
        Refinement.from_dict(run.to_dict(), catalog=cat).replay(s)
    assert Refinement.from_dict(run.to_dict(), catalog=cat).replay(s, accept=lambda res: res.values["morning"])["ok"]


def test_a_proposal_that_cannot_be_read_feeds_back_the_error_of_the_part_that_failed():
    s, _ = _system()
    run = refine(s, {}, "ok", propose=proposals({"day": "Mon"}, Slot(day="Mon", start=9)), into="proposal")
    assert run.accepted and len(run.rounds) == 2
    assert run.rounds[0].causes[0].startswith("slot: type rejected") and run.rounds[0].feedback == run.rounds[0].causes


class FakeServer:
    def __init__(self, *replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.bodies.append(body)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, int):
            import urllib.error
            raise urllib.error.HTTPError(req.full_url, r, "error", {}, io.BytesIO(b"{}"))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": r}, "finish_reason": "stop"}]}).encode())


def test_a_generator_proposer_re_asks_with_its_reply_and_the_reasons_and_a_rejected_reply_is_a_round():
    srv = FakeServer('{"day": "Sun", "start": 9}', "sorry, no JSON", '{"day": "Mon", "start": 9}')
    g = generator("http://127.0.0.1:9/v1", "m", opener=srv)
    s, _ = _system()
    run = refine(s, {}, "ok", propose=g.proposer("Propose a slot as JSON.", schema=Slot), into="proposal", rounds=3)
    assert run.accepted and [r.accepted for r in run.rounds] == [False, False, True]
    assert run.rounds[1].error == "the reply was rejected: the reply is not JSON" and run.rounds[1].response is None
    assert run.rounds[0].generated["request"] and run.rounds[2].generated["text"] == '{"day": "Mon", "start": 9}'
    last = srv.bodies[2]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "user", "assistant", "user"]
    assert last[1]["content"] == '{"day": "Sun", "start": 9}' and last[3]["content"] == "sorry, no JSON"
    assert last[2]["content"] == ("Your answer was checked by a program, and it does not work:\n"
                                  "- Sun is not allowed (Mon, Tue)\nGive a corrected answer.")
    assert last[4]["content"].endswith("- the reply was rejected: the reply is not JSON\nGive a corrected answer.")
    assert run.replay(s)["ok"]


def test_a_proposer_that_fails_ends_the_loop_with_an_escalation():
    g = generator("http://127.0.0.1:9/v1", "m", opener=FakeServer(503), retries=0)
    s, _ = _system()
    run = refine(s, {}, "ok", propose=g.proposer("q", schema=Slot), into="proposal", rounds=3)
    assert not run.accepted and len(run.rounds) == 1
    assert run.escalation.startswith("the proposer failed: Unanswered: the LLM server did not answer") and run.replay(s)["ok"]


def test_without_a_proposer_the_system_generates_and_reads_the_earlier_feedback_as_a_fact():
    cat = Catalog()

    @cat.fn
    def draft(feedback) -> str:
        return "SELECT 1" if feedback else "DROP TABLE x"

    @cat.check(hard=True, then={"ok": "no"})
    def read_only(draft) -> bool:
        return draft.startswith("SELECT") or Fail(f"`{draft}` changes the database")

    @cat.rule("ok")
    def ok(read_only) -> bool:
        return True
    s = System(cat, [Question("ok", "?", Answer.yes_no(), requires=["read_only"])])
    run = refine(s, {}, "ok", rounds=2)
    assert run.accepted and [r.response.trace.init["feedback"] for r in run.rounds] == [[], ["`DROP TABLE x` changes the database"]]
    assert run.replay(s)["ok"]


def test_replay_names_a_round_whose_record_was_edited_and_a_round_after_an_accepted_one():
    from solvi import JSONLStorage
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as d:
        store = JSONLStorage(os.path.join(d, "s.jsonl"))
        s, cat = _system(storage=store)
        run = refine(s, {}, "ok", propose=proposals(Slot(day="Sun", start=9), Slot(day="Mon", start=9)), into="proposal")
        assert [r.stored_id is not None for r in run.rounds] == [True, True] and store.verify()["ok"]
        d_ = json.loads(run.to_json())
        back = Refinement.from_dict(d_, catalog=cat)
        assert back.replay(s)["ok"] and back.proposal == {"day": "Mon", "start": 9}
        d_["rounds"][0]["accepted"] = True
        rep = Refinement.from_dict(d_, catalog=cat).replay(s)
        assert not rep["ok"] and (0, "accepted", "recorded True, the response gives False") in rep["mismatches"]
        assert (1, "loop", "a round ran after an accepted one") in rep["mismatches"]
        d_ = json.loads(run.to_json())
        d_["rounds"][0]["feedback"] = ["something else"]
        rep = Refinement.from_dict(d_, catalog=cat).replay(s)
        assert (0, "feedback", "the recorded feedback is not what the feedback function gives") in rep["mismatches"]


def test_refine_refuses_what_it_cannot_run():
    s, _ = _system()
    with pytest.raises(ValueError, match="rounds"):
        refine(s, {}, "ok", propose=proposals(1), rounds=0)
    with pytest.raises(ValueError, match="not a question"):
        refine(s, {}, "nope", propose=proposals(1))
    with pytest.raises(TypeError):
        refine(s, {}, "ok", propose="not callable")
