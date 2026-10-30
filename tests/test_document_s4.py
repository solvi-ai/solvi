"""S4 of the solvi.document plan: the number parser and Span[date] readings of the core (gap 18 — table tests), and
the EPS note of the ROADMAP (gap 10): a stated 0.9 is 0.9 — rounded to 12 decimals at the source, the same number in
calibration and in every decision. The parser itself and the guarded readings shipped with S1; this locks their table."""
import datetime
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_llm import TEAMS, FakeLLM, model  # noqa: E402
from test_document_recipe import Server, case_reply, doc_of, ops_system  # noqa: E402

from solvi.core.textin import ParseError, parse_number
from solvi.core.types import span_value


@pytest.mark.parametrize("s, want", [
    ("$1,250", "1250"),
    ("(1,250)", "1250"),                 # the core parser reads the magnitude; the sign is the reader's (below)
    ("$1,250 million", "1250000000"),
    ("1.5 million", "1500000"),
    ("-1,250", "-1250"),
    ("EUR 18,851.12", "18851.12"),
    ("2k", "2000"),
])
def test_parse_number_table(s, want):
    assert parse_number(s) == want


@pytest.mark.parametrize("s", ["1.000", "1 ,234", "5 m", "3 100", "5%", "03/04/2026"])
def test_parse_number_refusals(s):
    with pytest.raises(ParseError):
        parse_number(s, {"percent": False} if s.endswith("%") else None)


def test_span_float_accounting_negative_through_the_recipe():
    """The recipe's reader applies the accounting sign the core magnitude leaves out: "(1,250)" is -1250 as a value."""
    s = ops_system(Server(case_reply()))
    res = s.ask({"doc": doc_of(1), "op": "diff"})
    assert res.values["b_value"] == -1250.0


@pytest.mark.parametrize("s, want", [
    ("21 July 2026", datetime.date(2026, 7, 21)),
    ("July 21, 2026", datetime.date(2026, 7, 21)),
    ("21.07.2026", datetime.date(2026, 7, 21)),
    ("2026-09-12", datetime.date(2026, 9, 12)),
])
def test_span_date_table(s, want):
    assert span_value(datetime.date, s) == want     # the recipe passes the bare type (pydantic unwraps Span)


def test_span_date_refuses_what_would_be_a_guess():
    for bad in ("03/04/2026", "12.09.2026"):       # day or month first?
        with pytest.raises(ValueError):
            span_value(datetime.date, bad)


def test_stated_0_9_is_0_9():
    """The EPS note, main claim: a stated 0.9 comes back exactly 0.9 (12-decimal rounding at the source) — the same
    number a threshold is calibrated on and every later decision is judged by."""
    fake = FakeLLM(reply='{"answer": "billing", "probabilities": {"billing": 0.9, "shipping": 0.1}, "quote": ""}')
    d = model(fake).decision("team", "Which team should handle this?", "email", TEAMS) \
        .decide("I was charged twice for order 7.")
    assert d.probs["billing"] == 0.9 and d.probs["shipping"] == 0.1
    assert repr(d.probs["billing"]) == "0.9"       # rounded to 12 decimals, not 0.8999999999999999
