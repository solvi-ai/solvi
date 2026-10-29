"""reads_with= (the state-conditioned reads spec, S8 candidate): the question and other state keys reach the field
reads as a <state> block in the prompt — while the text, the span grounding and the quote offsets stay on the one
text fact (no question composed into the document). Fingerprint: a recipe without reads_with is unchanged; with
reads_with it is its own catalog (a calibration/record made without it does not attach)."""
import io
import json
import re
import warnings

import pytest
from pydantic import BaseModel, Field

import solvi
from solvi import Span
from solvi.core.deciders.llm import llm


class Server:
    """Reads the fields by regex from the text it is shown; REFUSES when the question is not in its prompt —
    so a passing test proves the question reached the read."""

    def __init__(self):
        self.calls, self.bodies = 0, []

    def __call__(self, req, timeout=None):
        self.calls += 1
        body = json.loads(req.data.decode())
        self.bodies.append(body)
        msg = body["messages"][-1]["content"]
        nums = re.search(r"Revenue was (\S+) and cost was (\S+)\.", msg)
        qm = re.search(r"Question: (.+)", msg)
        if qm is None:
            return io.BytesIO(json.dumps({"model": body["model"], "choices": [
                {"index": 0, "message": {"role": "assistant", "content": json.dumps(
                    {"answer": "", "confidence": 0.0, "quote": ""})}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode())
        q = qm.group(1)
        field = "a" if "The first number" in msg.split("<text>")[0] else "b"
        if "more" in q:                      # the question conditions the answer: revenue minus cost
            ans = nums.group(1) if field == "a" else f"-{nums.group(2)}"
        else:
            ans = nums.group(1) if field == "a" else nums.group(2)
        c = json.dumps({"answer": ans, "confidence": 0.9, "quote": ""})
        ch = {"index": 0, "message": {"role": "assistant", "content": c}, "finish_reason": "stop"}
        return io.BytesIO(json.dumps({"model": body["model"], "choices": [ch],
                                      "usage": {"prompt_tokens": 40, "completion_tokens": 9}}).encode())


class Ops(BaseModel):
    a: Span[float] = Field(description="The first number the question in <state> needs, as a digit, exactly as written.")
    b: Span[float] = Field(description="The second number the question in <state> needs, as a digit, exactly as written.")


def doc_of(i, a="$1,250", b="(1,250)"):
    return f"Case {i}. Revenue was {a} and cost was {b}. Other figure: 77."


def model_of(server):
    return llm("http://fake/v1", "fake-model", opener=server, workers=1, logprobs=False, ask="confidence")


def compute_diff(f, s):
    return (f.a - f.b) if s["question"].startswith("How many more") else (f.a + f.b)


def recipe_of(server, **kw):
    kw.setdefault("compute", compute_diff)
    return solvi.document(Ops, model_of(server), task="A short company note.",
                          reads_with=("question",), **kw)


def plain_recipe_of(server, **kw):
    kw.setdefault("compute", compute_diff)
    return solvi.document(Ops, model_of(server), task="A short company note.", **kw)


def test_question_reaches_the_read_grounding_stays_on_doc():
    """К1: the question is in the read's prompt (the fake server refuses without it), the answer uses it, and the
    facts' quotes still point into the ORIGINAL document — no question composed into the text."""
    srv = Server()
    recipe = recipe_of(srv)
    doc = doc_of(1)
    res = recipe.ask({"doc": doc, "question": "How many more dollars was the revenue than the cost?"})
    q = res["answer"]
    assert q.status == "ok" and q.answer == 2500.0
    user_msgs = [b["messages"][-1]["content"] for b in srv.bodies]
    assert all("How many more dollars" in m for m in user_msgs)      # every read saw the question
    assert all("<state>" in m for m in user_msgs)
    facts = {f.name: f for f in res.facts}
    for name, quote in (("a", "$1,250"), ("b", "(1,250)")):
        f = facts[name]
        assert f.quote == doc[f.start:f.end] == quote                # grounding on the original doc
    assert not any("How many more" in doc[f.start:f.end] for f in facts.values())


def test_missing_state_key_is_a_clear_error():
    """A missing state key is a clear, named refusal: the read's error names reads_with and the key (the library's
    philosophy: a part's failure is a fact, not an exception — it surfaces in the abstention's reason)."""
    recipe = recipe_of(Server())
    res = recipe.ask({"doc": doc_of(1)})                             # no question in the input
    q = res["answer"]
    assert q.status == "abstain" and "question" in q.why and "reads_with" in q.why


def test_fingerprint_boundary():
    """К2: without reads_with (or with None) the fingerprint is what it was; with reads_with it is another catalog,
    and a calibration made on one side does not attach to the other."""
    import tempfile, os
    s1 = plain_recipe_of(Server())
    s2 = plain_recipe_of(Server(), reads_with=None)
    s3 = recipe_of(Server())
    assert s1.system.fingerprint() == s2.system.fingerprint()
    assert s1.system.fingerprint() != s3.system.fingerprint()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cal.json")
        rows = [({"doc": doc_of(i), "question": "How many more dollars was the revenue than the cost?"}, 2500.0)
                for i in range(12)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s3.calibrate(rows, max_risk=0.20, correct=lambda r, y: abs((r.answer or 0) - y) <= 1.0)
        s3.save_calibration(p)
        s4 = plain_recipe_of(Server())
        with pytest.raises(ValueError):                              # made for another catalog (reads_with)
            s4.load_calibration(p)
        s5 = recipe_of(Server())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s5.load_calibration(p)                                   # the same reads_with: attaches


def test_slow_hands_over_on_abstain():
    """S7: slow= — when the recipe abstains, the answer comes from the slow path (a callable or a System), the
    response names the hand-over, and the store keeps the recipe's own abstain (verify/replay stay exact)."""
    import os
    import tempfile
    from solvi.core.store import JSONLStorage

    def slow_answer(state):
        return 42.0

    with tempfile.TemporaryDirectory() as d:
        st = JSONLStorage(os.path.join(d, "t.jsonl"), record="full")

        class UnknownServer(Server):
            """Reads "$9,999" — a number not in the text: the read is rejected, the answer abstains."""

            def __call__(self, req, timeout=None):
                out = super().__call__(req, timeout)
                body = json.loads(req.data.decode())
                msg = body["messages"][-1]["content"]
                field = "a" if "The first number" in msg.split("<text>")[0] else "b"
                c = json.loads(json.loads(out.read().decode())["choices"][0]["message"]["content"])
                c["answer"] = "$9,999" if field == "a" else "(1,250)"
                ch = {"index": 0, "message": {"role": "assistant", "content": json.dumps(c)},
                      "finish_reason": "stop"}
                return io.BytesIO(json.dumps({"model": body["model"], "choices": [ch],
                                              "usage": {"prompt_tokens": 40, "completion_tokens": 9}}).encode())

        recipe = recipe_of(UnknownServer(), slow=slow_answer, storage=st)
        # an unknown number makes the reads reject → abstain → hand over
        res = recipe.ask({"doc": doc_of(1), "question": "How many more dollars was the revenue than the cost?"})
        assert res["answer"].answer == 42.0 and res["answer"].status == "ok"
        assert res.hand_over["slow"].startswith("the slow callable")
        assert res.hand_over["recipe_abstained"]       # the abstention's reason (reads rejected the unknown number)
        assert recipe.replay_all() == []               # the store keeps the recipe's own abstain
