"""solvi.document: one model read per field, each fact found in the text, code computes the answer, hard checks, a
calibrated threshold over System.guarantee, a replayable record — against a fake chat-completions server (no network)."""
import datetime
import io
import json
import re
import warnings
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import solvi
from solvi import Answer, Span
from solvi.core.deciders.llm import llm
from solvi.core.guarantees.guarantee import GuaranteeWarning


class Server:
    """A chat-completions server: reply(user_message, body) → (answer, confidence) of a span / choice reply."""

    def __init__(self, reply):
        self.reply, self.calls, self.bodies = reply, 0, []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        self.calls += 1
        self.bodies.append(body)
        ans, conf = self.reply(body["messages"][-1]["content"], body)
        c = json.dumps({"answer": ans, "confidence": conf, "quote": ""})
        ch = {"index": 0, "message": {"role": "assistant", "content": c}, "finish_reason": "stop"}
        return io.BytesIO(json.dumps({"model": body["model"], "choices": [ch],
                                      "usage": {"prompt_tokens": 40, "completion_tokens": 9}}).encode())


class Ops(BaseModel):
    a: Span[float] = Field(description="The revenue, exactly as written.")
    b: Span[float] = Field(description="The cost, exactly as written.")


def doc_of(i, a="$1,250", b="(1,250)", w="77"):
    return f"Case {i}. Revenue was {a} and cost was {b}. Other figure: {w}."


def case_reply(wrong=frozenset(), unknown_a=frozenset()):
    """Reads a and b from the case's own document; in a `wrong` case it reads the other figure for a, unsure."""
    def reply(msg, body):
        i = int(re.search(r"Case (\d+)\.", msg).group(1))
        nums = re.search(r"Revenue was (\S+) and cost was (\S+)\. Other figure: (\S+)\.", msg).groups()
        if "revenue" in msg.split("<text>")[0]:
            if i in unknown_a:
                return "$9,999", 0.95                     # not in the text
            return (nums[2], 0.3) if i in wrong else (nums[0], 0.9)
        return nums[1], 0.95
    return reply


def model_of(server, **kw):
    kw.setdefault("logprobs", False)                     # chosen: the recipe keeps it (no rebuild, no warning)
    kw.setdefault("ask", "confidence")
    return llm("http://fake/v1", "fake-model", opener=server, workers=1, **kw)


def ops_system(server, **kw):
    kw.setdefault("compute", lambda f, s: f.a - f.b if s["op"] == "diff" else f.a + f.b)
    kw.setdefault("checks", {"b_not_zero": lambda f, s: f.b != 0})
    model = kw.pop("model", None) or model_of(server)
    return solvi.document(Ops, model, task="A short company note.", **kw)


def test_solvi_document_is_the_recipe_entry():
    from solvi.solutions.document import DocumentSystem, document
    assert solvi.document is document and "document" in solvi.__all__ and len(solvi.__all__) == 24
    s = ops_system(Server(case_reply()))
    assert isinstance(s, DocumentSystem) and isinstance(s.system, solvi.System)
    assert s.fields == ["a", "b"] and s.checks == ["b_not_zero"]


def test_grounded_facts_are_parsed_and_code_computes():
    srv = Server(case_reply())
    s = ops_system(srv)
    res = s.ask({"doc": doc_of(1), "op": "diff"})
    r = res["answer"]
    assert r.status == "ok" and r.answer == 1250.0 - (-1250.0)      # "$1,250" → 1250; "(1,250)" → −1250
    assert r.confidence == pytest.approx(0.9)                        # the weaker read
    assert srv.calls == 2                                            # one read per field
    assert res.values["a"] == "$1,250" and res.values["b"] == "(1,250)"    # the text's own substrings
    assert s.ask({"doc": doc_of(2), "op": "sum"})["answer"].answer == 0.0
    prompts = [b["messages"][-1]["content"] for b in srv.bodies]
    assert all(p.startswith("Question: A short company note. The ") for p in prompts)
    assert "op" not in prompts[0].split("<text>")[1]                 # the model reads the text alone


NUMBERS = [  # (quote, value or None: does not read → the answer abstains)
    ("$1,250", 1250.0), ("1,250 million", 1250.0), ("1,234 thousand", 1234.0), ("3 mn.", 3.0),
    # one scale rule: never multiplied, word or suffix
    ("1.5 bn", 1.5), ("$1.2 billion", 1.2), ("1.2bn", 1.2), ("1.2B", 1.2), ("$1.2M", 1.2), ("1.2m", 1.2), ("1.2k", 1.2),
    # accounting negatives, with currency and scale around them
    ("(1,234)", -1234.0), ("$(1,234)", -1234.0), ("(1,234) million", -1234.0), ("$(1,234) million", -1234.0),
    ("EUR (1,234)", -1234.0), ("(EUR 1,234)", -1234.0), ("( 1,234 )", -1234.0), ("(3.5%)", -3.5), ("(3)", None),
    ("(2019)", None), ("(1)", None), ("((5))", None), ("(-5)", None), ("-(5)", None),
    # signs in any dash, before or after the currency, trailing
    ("-1,234", -1234.0), ("−1,234", -1234.0), ("–1,234", -1234.0), ("—1,234", -1234.0), ("- 1,234", -1234.0),
    ("-$1,234", -1234.0), ("$-1,234", -1234.0), ("1,234-", -1234.0), ("+5%", 5.0), ("-.5", -0.5),
    # percentages, separators
    ("12.5%", 12.5), ("12.5 %", 12.5), ("12.5 percent", 12.5), ("12,5%", 12.5), ("%5", None), ("1,234.5", 1234.5),
    ("1.234,56", 1234.56), ("€1.234,56", 1234.56), ("EUR 18,851.12", 18851.12), ("2.500", None), ("1 234", None),
    # not one number
    ("10-12", None), ("10–12", None), ("10 to 12", None), ("1e3", None), ("1_000", None), ("nan", None), ("NaN", None),
    ("inf", None), ("Infinity", None), ("-inf", None), ("1/2", None), ("½", None), ("3x", None), ("50 bps", None),
    ("n/a", None), ("", None), ("$", None), ("million", None), ("-", None), ("1,2,3", None), ("0x10", None),
]


@pytest.mark.parametrize("raw,v", NUMBERS)
def test_numbers_as_documents_write_them(raw, v):
    from solvi.solutions.document import _read_number
    if v is None:
        with pytest.raises(ValueError):
            _read_number(float, raw)
    else:
        assert _read_number(float, raw) == pytest.approx(v)


def test_a_sign_the_quote_left_out_abstains():
    def reply(msg, body):
        return ("1,234", 0.9) if "revenue" in msg.split("<text>")[0] else ("5", 0.9)
    for doc in ("Revenue was -1,234 and cost was 5.", "Revenue was (1,234) and cost was 5.",
                "Revenue was -$1,234 and cost was 5.", "Revenue was −1,234 and cost was 5."):
        res = ops_system(Server(reply)).ask({"doc": doc, "op": "diff"})
        assert res["answer"].status == "abstain", doc
        assert [c.status for c in res.checks if c.name == "a_sign"] == ["failed"], doc
    ok = ops_system(Server(reply)).ask({"doc": "Revenue was 1,234 and cost was 5.", "op": "diff"})["answer"]
    assert ok.status == "ok" and ok.answer == 1229.0


def test_a_result_that_is_not_finite_abstains():
    s = ops_system(Server(case_reply()), compute=lambda f, s: float("nan"))
    r = s.ask({"doc": doc_of(1), "op": "diff"})["answer"]
    assert r.status == "abstain" and "not a finite number" in r.why
    s = ops_system(Server(case_reply()), compute=lambda f, s: f.a * float("inf"))
    assert s.ask({"doc": doc_of(1), "op": "diff"})["answer"].status == "abstain"


def test_an_ungrounded_fact_abstains_with_the_reason():
    s = ops_system(Server(case_reply(unknown_a={3})))
    r = s.ask({"doc": doc_of(3), "op": "diff"})["answer"]
    assert r.status == "abstain" and r.answer is None
    assert "a: model escalated" in r.why and "'$9,999' is not literally in the text" in r.why, r.why


def test_a_failed_check_abstains():
    s = ops_system(Server(case_reply()))
    res = s.ask({"doc": doc_of(4, b="0"), "op": "diff"})
    r = res["answer"]
    assert r.status == "abstain" and r.answer is None
    assert [c.status for c in res.checks if c.name == "b_not_zero"] == ["failed"]


def test_a_value_that_does_not_read_abstains_with_the_field_named():
    s = ops_system(Server(case_reply()))
    r = s.ask({"doc": doc_of(5, a="2.500"), "op": "diff"})["answer"]
    assert r.status == "abstain" and "a: '2.500' does not read as float" in r.why, r.why


def labelled(n, wrong):
    out = []
    for i in range(n):
        a, b = 1000 + i, 10 + i % 7
        out.append(({"doc": doc_of(i, a=f"${a:,}", b=str(b), w=str(5 + i)), "op": "diff"}, float(a - b)))
    return out


def test_calibrate_then_ask_store_verify_replay_and_save_load(tmp_path):
    wrong = frozenset(range(0, 150, 5))                              # 30 of 150 read the wrong figure, unsure
    srv = Server(case_reply(wrong=wrong))
    path = tmp_path / "decisions.jsonl"
    s = ops_system(srv, storage=str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("error", GuaranteeWarning)            # 150 examples: no "few examples" warning
        rep = s.calibrate(labelled(150, wrong), max_risk=0.10)
    assert rep["n"] == 150 and 0.3 < rep["threshold"] <= 0.9 and rep["answered"] == pytest.approx(0.8)
    assert s.calibration is rep and "calibrated" in repr(s)
    sure = s.ask({"doc": doc_of(1000, a="$2,000", b="5"), "op": "diff"})
    assert sure["answer"].status == "ok" and sure["answer"].answer == 1995.0 and sure.stored_id
    unsure = s.ask({"doc": doc_of(1005, a="$2,000", b="5"), "op": "diff"})    # 1005 % 5 == 0: but not in `wrong`
    assert unsure["answer"].status == "ok"
    srv.reply = case_reply(wrong={1010})
    low = s.ask({"doc": doc_of(1010, a="$2,000", b="5"), "op": "diff"})["answer"]
    assert low.status == "abstain" and low.answer is None            # below the threshold
    assert s.verify()["ok"] and s.replay_all() == []
    saved = s.save_calibration(tmp_path / "doc.guarantee.json")
    fresh = ops_system(Server(case_reply()), calibration=saved)
    assert fresh.calibration["threshold"] == rep["threshold"]
    assert fresh.storage is None
    from solvi.core.store import JSONLStorage
    assert JSONLStorage(path).replay_all(fresh.system) == []        # replays without calibrating again
    other = ops_system(Server(case_reply()), compute=lambda f, s: f.a * f.b)    # another rule: another catalog
    with pytest.raises(ValueError, match="calibrated on catalog"):
        other.load_calibration(saved)


def test_tampered_record_is_caught(tmp_path):
    path = tmp_path / "d.jsonl"
    s = ops_system(Server(case_reply()), storage=str(path))
    s.ask({"doc": doc_of(1), "op": "diff"})
    lines = path.read_text().splitlines()
    assert "2500.0" in lines[-1]
    lines[-1] = lines[-1].replace("2500.0", "2600.0")
    path.write_text("\n".join(lines) + "\n")
    from solvi.core.store import JSONLStorage
    assert not JSONLStorage(path).verify()["ok"]


def test_fewer_than_100_examples_warn():
    s = ops_system(Server(case_reply()))
    with pytest.warns(GuaranteeWarning, match="calibrating on 40 examples"):
        rep = s.calibrate(labelled(40, ()), max_risk=0.10, weak="warn")
    assert any("calibrating on 40 examples" in w for w in rep["warnings"])


def test_an_llm_with_auto_logprobs_is_asked_for_a_stated_confidence():
    srv = Server(case_reply())
    with pytest.warns(UserWarning, match="usage counters are on recipe.model"):
        s = ops_system(srv, model=model_of(srv, logprobs="auto", ask="probabilities"))
    assert s.model.scorer.logprobs is False and s.model.scorer.ask == "confidence"
    s.ask({"doc": doc_of(1), "op": "diff"})
    assert all(not b.get("logprobs") for b in srv.bodies)
    given = model_of(Server(case_reply()), logprobs=True)
    assert solvi.document(Ops, given, compute=lambda f, s: f.a).model is given     # chosen explicitly: kept


def test_choice_field_asks_the_plain_option_list_with_descriptions_in_the_question():
    class Claim(BaseModel):
        amount: Span[float] = Field(description="The amount claimed.")
        kind: Literal["damage", "theft"] = Field(description="What happened.", json_schema_extra={
            "options": {"damage": "something was broken", "theft": "something was stolen"}})

    def reply(msg, body):
        if "What happened" in msg:
            return "theft", 0.9
        return "$300", 0.9
    srv = Server(reply)
    s = solvi.document(Claim, model_of(srv), compute=lambda f, s: "pay" if f.kind == "theft" and f.amount < s["limit"]
                       else "hold", answer=Answer.choice(["pay", "hold"]))
    r = s.ask({"doc": "A bike worth $300 was stolen.", "limit": 500})["answer"]
    assert r.status == "ok" and r.answer == "pay"
    kind_msg = next(b["messages"][-1]["content"] for b in srv.bodies if "What happened" in b["messages"][-1]["content"])
    assert "The options mean: damage — something was broken; theft — something was stolen." in kind_msg
    assert "\n- damage\n- theft\n" in kind_msg                       # the options themselves are plain


def test_dates_text_and_one_field_without_compute():
    class Invoice(BaseModel):
        due: Span[datetime.date] = Field(description="The due date.")
        vendor: Span[str] = Field(description="The vendor.")

    def reply(msg, body):
        return ("21 July 2026", 0.9) if "due date" in msg else ("Acme Ltd", 0.8)
    s = solvi.document(Invoice, model_of(Server(reply)),
                       compute=lambda f, s: "pay" if f.due >= s["today"] and f.vendor in s["approved"] else "hold",
                       answer=Answer.choice(["pay", "hold"]))
    doc = "Invoice from Acme Ltd, due 21 July 2026."
    r = s.ask({"doc": doc, "today": "2026-07-01", "approved": ["Acme Ltd"]})["answer"]
    assert r.status == "abstain"                                     # "2026-07-01" is a str: comparing raises
    r = s.ask({"doc": doc, "today": datetime.date(2026, 7, 1), "approved": ["Acme Ltd"]})["answer"]
    assert r.status == "ok" and r.answer == "pay" and r.confidence == pytest.approx(0.8)

    class One(BaseModel):
        total: Span[float] = Field(description="The total.")
    one = solvi.document(One, model_of(Server(lambda m, b: ("$41.50", 0.9))))
    assert one.ask({"doc": "Total due: $41.50."})["answer"].answer == 41.5


def test_refusals_at_build_time():
    srv = Server(case_reply())
    with pytest.raises(ValueError, match="give compute="):
        solvi.document(Ops, model_of(srv))
    with pytest.raises(TypeError, match="pydantic model class"):
        solvi.document(dict, model_of(srv), compute=lambda f, s: 1)

    class Bad(BaseModel):
        state: Span[float]
    with pytest.raises(ValueError, match="cannot be named 'state'"):
        solvi.document(Bad, model_of(srv))

    class Elsewhere(BaseModel):
        x: Span[float, "note"]                                       # noqa: F821 (a source name, not a type)
    with pytest.raises(ValueError, match="span of 'note'"):
        solvi.document(Elsewhere, model_of(srv))
    for bad in ("a", "a_sign", "answer", "compute", "state", "doc"):
        with pytest.raises(ValueError, match="check name"):
            ops_system(srv, checks={bad: lambda f, s: True})
    s = ops_system(srv)
    with pytest.raises(KeyError, match="has no 'doc'"):
        s.ask({"text": "x"})
    with pytest.raises(ValueError, match="keeps no record"):
        s.verify()


def test_a_calibration_file_is_refused_for_another_model_or_prompt(tmp_path):
    wrong = frozenset(range(0, 120, 6))
    s = ops_system(Server(case_reply(wrong=wrong)))
    s.calibrate(labelled(120, wrong), max_risk=0.10)
    path = s.save_calibration(tmp_path / "g.json")
    assert ops_system(Server(case_reply()), calibration=path).calibration["threshold"] == s.calibration["threshold"]
    others = {"model": dict(model=llm("http://fake/v1", "other-model", opener=Server(case_reply()), workers=1,
                                      logprobs=False, ask="confidence")),
              "endpoint": dict(model=llm("http://elsewhere/v1", "fake-model", opener=Server(case_reply()), workers=1,
                                         logprobs=False, ask="confidence")),
              "reply settings": dict(model=model_of(Server(case_reply()), logprobs=True))}
    for what, kw in others.items():
        with pytest.raises(ValueError, match="calibrated with other models"):
            ops_system(Server(case_reply()), calibration=path, **kw)
    with pytest.raises(ValueError, match="calibrated (on catalog|with other models)"):      # another task=
        solvi.document(Ops, model_of(Server(case_reply())), task="Another note.", calibration=path,
                       compute=lambda f, s: f.a - f.b if s["op"] == "diff" else f.a + f.b,
                       checks={"b_not_zero": lambda f, s: f.b != 0})

    class Ops2(BaseModel):
        a: Span[float] = Field(description="The total revenue, exactly as written.")
        b: Span[float] = Field(description="The cost, exactly as written.")
    with pytest.raises(ValueError, match="calibrated (on catalog|with other models)"):   # another description
        solvi.document(Ops2, model_of(Server(case_reply())), task="A short company note.", calibration=path,
                       compute=lambda f, s: f.a - f.b if s["op"] == "diff" else f.a + f.b,
                       checks={"b_not_zero": lambda f, s: f.b != 0})


def test_the_default_number_judge_reads_the_label_at_its_precision():
    from types import SimpleNamespace as R

    from solvi.solutions.document import _same_number
    assert _same_number(R(answer=33.3333), "33.33") and _same_number(R(answer=33.3333), 33.33)
    assert not _same_number(R(answer=33.35), "33.33")
    assert _same_number(R(answer=1500.0), "1,500") and _same_number(R(answer=1500.2), 1500)
    assert not _same_number(R(answer=1501.0), "1,500")
    assert _same_number(R(answer=-2.5), "(2.5)") and _same_number(R(answer=0.1 + 0.2), 0.3)
    assert not _same_number(R(answer=None), 1.0) and not _same_number(R(answer=float("nan")), 1.0)
    assert not _same_number(R(answer=1.0), "n/a")


def test_a_failed_calibration_names_correct_for_the_default_judge():
    s = ops_system(Server(case_reply()))
    with pytest.raises(ValueError, match="pass correct="):
        s.calibrate([(st, y if i % 2 else y + 1) for i, (st, y) in enumerate(labelled(60, ()))], max_risk=0.10)


def test_the_input_is_stored_as_plain_data_and_replays(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "d.jsonl"
    mk = lambda **kw: ops_system(Server(case_reply()),                     # noqa: E731
                                 compute=lambda f, s: f.a * s["k"] + len(s["tags"]) + len(s["seen"]), **kw)
    s = mk(storage=str(path))
    r = s.ask({"doc": doc_of(1), "op": "diff", "k": np.float64(2.0), "tags": {"x", "y"}, "seen": ("a",),
               "day": datetime.date(2026, 7, 1)})["answer"]
    assert r.status == "ok" and r.answer == 2503.0
    from solvi.core.store import JSONLStorage
    assert s.replay_all() == [] and JSONLStorage(path).replay_all(mk().system) == []
    with pytest.raises(TypeError, match="not stored as JSON"):
        s.ask({"doc": doc_of(1), "op": "diff", "k": object()})
    with pytest.raises(TypeError, match="not a string"):
        s.ask({"doc": doc_of(1), "op": "diff", "k": {1: 2}})


def test_one_field_without_compute_answers_its_value_or_is_refused():
    class Kind(BaseModel):
        kind: Literal["damage", "theft"] = Field(description="What happened.")
    s = solvi.document(Kind, model_of(Server(lambda m, b: ("theft", 0.9))))
    r = s.ask({"doc": "A bike was stolen."})["answer"]
    assert r.status == "ok" and r.answer == "theft" and r.kind == "choice"

    class Flag(BaseModel):
        late: bool = Field(description="Is the payment late?")
    assert solvi.document(Flag, model_of(Server(lambda m, b: ("yes", 0.9)))).ask({"doc": "Late."})["answer"].status == "ok"
    for t in (str, datetime.date):
        class One(BaseModel):
            x: Span[t] = Field(description="The value.")
        with pytest.raises(TypeError, match="give compute="):
            solvi.document(One, model_of(Server(case_reply())))


def test_a_plain_field_type_names_span():
    for t in (float, str):
        class Plain(BaseModel):
            x: t = Field(description="The value.")
        with pytest.raises(TypeError, match=rf"write Span\[{t.__name__}\]"):
            solvi.document(Plain, model_of(Server(case_reply())), compute=lambda f, s: 1.0)


def test_another_decider_is_used_as_given():
    from solvi.core.deciders.llm import LLMScorer

    class MyScorer(LLMScorer):
        pass
    m = model_of(Server(case_reply()), logprobs="auto")
    m.scorer.__class__ = MyScorer
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert ops_system(Server(case_reply()), model=m).model is m
