"""S309 of the 2026-10-30 external handoff: a number field's fact offsets name the line the model meant. An LLM's
span answer grounds at the first occurrence of its digits — "SUB TOTAL 28,000" above "TOTAL 28,000" grounded the
TOTAL read to the SUB TOTAL line (the value right, the provenance the wrong line). A field whose name is a label
("total", "sub_total"; a one-letter name is no label) grounds on the one occurrence of the quoted digits on a line
that starts with the label, or its immediate numeric continuation. Repeated numbers without a unique supported
location abstain instead of asserting the first occurrence. Fake server, no network; records still verify and replay."""
import json
import os
import sys

from pydantic import BaseModel, Field

import solvi
from solvi import Span

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_document_recipe import Server, model_of  # noqa: E402


class Totals(BaseModel):
    total: Span[float] = Field(description="the TOTAL, exactly as written")


class SubTotal(BaseModel):
    sub_total: Span[float] = Field(description="the SUB TOTAL, exactly as written")


class One(BaseModel):
    x: Span[float] = Field(description="the total, exactly as written")


DOC = "SUB TOTAL 28,000\nTOTAL 28,000\n"       # the digits: [10,16) on the SUB TOTAL line, [23,29) on TOTAL's


def totals(cls=Totals, **kw):
    return solvi.document(cls, model_of(Server(lambda m, b: ("28,000", 0.9))), **kw)


def test_the_totals_read_grounds_to_the_total_line():
    """Their repro: the model answers 28,000 for the TOTAL field; the fact must be the TOTAL line, not the first
    occurrence of the digits (the SUB TOTAL line) — and the value still reads and the answer stands."""
    res = totals().ask({"doc": DOC})
    assert res["answer"].status == "ok" and res["answer"].answer == 28000.0
    (f,) = res.facts
    assert f.name == "total" and (f.start, f.end) == (23, 29)          # doc.index("TOTAL ") + len("TOTAL ")
    assert f.quote == DOC[f.start:f.end] == "28,000" and f.source == "doc"
    assert res.values["total"] == "28,000"                             # the kept value is the text's substring


def test_the_sub_total_field_grounds_to_its_own_line():
    """The rule is the label's, not a preference for the last line: "sub_total" carries the SUB TOTAL line."""
    res = totals(SubTotal).ask({"doc": DOC})
    (f,) = res.facts
    assert (f.start, f.end) == (10, 16) and f.quote == "28,000"


def test_a_one_letter_field_name_is_no_label():
    """A one-letter name supplies no label and cannot select among duplicate amounts."""
    res = totals(One).ask({"doc": DOC})
    assert res["answer"].status == "abstain"
    assert not res.facts


def test_two_labelled_lines_abstain_instead_of_claiming_the_first_occurrence():
    """The same digits under two TOTAL labels: which one the field means, the text does not say."""
    doc = "TOTAL 28,000\nTOTAL 28,000\n"
    res = totals().ask({"doc": doc})
    assert res["answer"].status == "abstain"
    assert not res.facts


def test_missing_label_is_ambiguous_and_does_not_jump_across_another_label():
    """Repeated missing-label digits abstain; a preceding label cannot cross another label."""
    res = totals().ask({"doc": "Amount due 28,000; elsewhere 28,000."})
    assert res["answer"].status == "abstain"
    assert not res.facts
    gap = "TOTAL\nSUB TOTAL 28,000\n"
    res = totals().ask({"doc": gap})
    assert (res.facts[0].start, res.facts[0].end) == (16, 22)          # the SUB TOTAL line: not reached by the label


def test_the_label_matches_case_punctuation_and_more_words_before_it():
    """"Total:" carries the label "total"; words may sit between the label and the number, and the label may be
    several of the field name's words ("net_sales")."""
    doc = "Sub total 28,000\nTOTAL: 28,000\n"
    res = totals().ask({"doc": doc})
    assert (res.facts[0].start, res.facts[0].end) == (24, 30)

    class Net(BaseModel):
        net_sales: Span[float] = Field(description="the net sales, exactly as written")
    res = solvi.document(Net, model_of(Server(lambda m, b: ("1,000", 0.9)))).ask({"doc": "Sales 1,000\nNet sales 1,000\n"})
    assert (res.facts[0].start, res.facts[0].end) == (22, 27)


def test_a_quoted_currency_grounds_on_the_label_line_too():
    """The occurrence searched is the quote's own substring, currency and all."""
    doc = "SUB TOTAL $1,250\nTOTAL $1,250\n"
    res = solvi.document(Totals, model_of(Server(lambda m, b: ("$1,250", 0.9)))).ask({"doc": doc})
    assert res["answer"].status == "ok" and res["answer"].answer == 1250.0
    (f,) = res.facts
    assert (f.start, f.end) == (23, 29) and f.quote == "$1,250"


def test_the_records_quote_note_moves_with_the_fact_and_it_replays(tmp_path):
    """The stored record names one line: the fact's offsets, and the LLM quote note (extra["llm"]["quote"]) that
    the why renders offsets from; verify and replay stay exact."""
    class QuoteServer(Server):
        """A server whose replies also carry a quote (the supporting passage the contract asks for)."""

        def __call__(self, req, timeout=None):
            body = json.loads(req.data.decode())
            self.calls += 1
            self.bodies.append(body)
            ans, conf = self.reply(body["messages"][-1]["content"], body)
            c = json.dumps({"answer": ans, "confidence": conf, "quote": "28,000"})
            import io
            ch = {"index": 0, "message": {"role": "assistant", "content": c}, "finish_reason": "stop"}
            return io.BytesIO(json.dumps({"model": body["model"], "choices": [ch],
                                          "usage": {"prompt_tokens": 40, "completion_tokens": 9}}).encode())

    path = tmp_path / "decisions.jsonl"
    s = solvi.document(Totals, model_of(QuoteServer(lambda m, b: ("28,000", 0.9))), storage=str(path))
    res = s.ask({"doc": DOC})
    rec = next(r for r in res.trace.records if r.name == "total")
    assert rec.extra["llm"]["quote"] == ["28,000", 23, 29]              # was the first occurrence when read
    assert s.verify()["ok"] and s.replay_all() == []
    from solvi.core.store import JSONLStorage
    assert JSONLStorage(path).replay_all(totals().system) == []         # a fresh same recipe replays it clean


def test_all_number_fields_are_wrapped_and_one_letter_names_cannot_guess_offsets():
    """All numeric spans require unambiguous provenance; usable names additionally select their label.
    The wrapper preserves the decision metadata (name, model, question)."""
    s = totals()
    part = s.system.catalog.parts["total"].func
    assert part.__class__.__name__ == "_LabelGrounded" and part.__name__ == "total"
    assert part.model_id and "TOTAL" in part.task

    from test_document_recipe import ops_system                       # a / b: one-letter names, no label
    plain = ops_system(Server(lambda m, b: ("$1,250", 0.9)))
    assert plain.system.catalog.parts["a"].func.__class__.__name__ == "_LabelGrounded"
    res = plain.ask({"doc": "Case 1. Revenue was $1,250 and cost was $1,250. Other figure: 77.", "op": "diff"})
    assert res["answer"].status == "abstain"
    assert not res.facts


def test_a_reads_with_recipe_grounds_the_labelled_field_too():
    """The wrapper's other call path: a read with a <state> block (reads_with=) still grounds its span on the
    label's line — the state conditions the question, not the offsets (S8's own contract)."""
    srv = Server(lambda m, b: ("28,000", 0.9))
    s = solvi.document(Totals, model_of(srv), reads_with=("note",))
    res = s.ask({"doc": DOC, "note": "the invoice total"})
    assert res["answer"].status == "ok"
    (f,) = res.facts
    assert (f.start, f.end) == (23, 29)
    assert "<state>" in srv.bodies[-1]["messages"][-1]["content"]


def test_label_words_and_grounding_places():
    """The unit rules: what a field's label is, and which places it grounds."""
    from solvi.solutions.document import _label_ground, _label_words
    assert _label_words("total") == ("total",) and _label_words("sub_total") == ("sub", "total")
    assert _label_words("net_sales_2026") == ("net", "sales", "2026")
    assert _label_words("a") is None and _label_words("x_1") is None and _label_words("") is None
    assert _label_ground("28,000", DOC, ("total",)) == 23
    assert _label_ground("28,000", DOC, ("sub", "total")) == 10
    assert _label_ground("28,000", "TOTAL 28,000\nTOTAL 28,000\n", ("total",)) is None      # two: not said
    assert _label_ground("28,000", "Amount 28,000; elsewhere 28,000.", ("total",)) is None  # no label line
    assert _label_ground("28,000", "TOTAL\n28,000\n", ("total",)) == 6                  # numeric continuation
    assert _label_ground("28,000", "TOTALS 28,000\n", ("total",)) == 7                  # another label ("totals")
    assert _label_ground("28,000", "TOTAL 128,000\n", ("total",)) is None                  # cuts a longer number
    assert _label_ground("28,000", "  Total (2025) 28,000\n", ("total",)) == 15            # indent + words between


def test_multiline_label_selects_the_numeric_line_without_crossing_other_labels():
    doc = "SUB TOTAL 28,000\nTOTAL:\n28,000\n"
    res = totals().ask({"doc": doc})
    assert res["answer"].status == "ok" and res["answer"].answer == 28000.0
    (fact,) = res.facts
    assert fact.start == doc.rfind("28,000")
    assert fact.quote == doc[fact.start:fact.end] == "28,000"


def test_unlabelled_unique_number_remains_grounded():
    res = totals().ask({"doc": "Amount due 28,000."})
    assert res["answer"].status == "ok"
    assert res.facts[0].start == 11


def test_ambiguous_decision_keeps_the_reading_but_removes_unsupported_offsets():
    from types import SimpleNamespace

    from solvi.core.catalog import Decision, Quote
    from solvi.solutions.document import _LabelGrounded

    doc = "TOTAL 28,000\nTOTAL 28,000\n"
    quote = Quote("28,000", 6, 12, "doc")
    decision = Decision(quote, confidence=0.9, evidence=[quote],
                        extra={"llm": {"quote": ["28,000", 6, 12]}})
    result = _LabelGrounded(SimpleNamespace(facts=["doc"]), ("total",))._grounded(decision, doc)
    assert result.value == "28,000"
    assert result.escalate.startswith("ambiguous number grounding")
    assert result.evidence == []
    assert result.extra["grounding"]["status"] == "ambiguous"
    assert "quote" not in result.extra["llm"]


def test_selected_percent_line_cannot_borrow_units_from_another_amount():
    res = totals().ask({"doc": "SUB TOTAL 28,000\nTOTAL 28,000%\n"})
    assert res["answer"].status == "abstain"
    assert not res.facts


def test_one_letter_unique_amount_is_distinct_from_a_percent():
    res = totals(One).ask({"doc": "Amount 28,000. Margin 28,000%."})
    assert res["answer"].status == "ok"
    (fact,) = res.facts
    assert (fact.start, fact.end) == (7, 13)
