"""Seen invoice failures are regression cases, not a fresh accuracy benchmark. No network."""
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

import solvi
from solvi import Span
from solvi.core.catalog import Quote, check_evidence, cuts_number, find_whole
from solvi.core.deciders.llm import locate
from solvi.core.provenance import find_normalized
from solvi.solutions.document import Unit, _ground, _read_quote

from test_document_recipe import Server, case_reply, model_of, ops_system


class Total(BaseModel):
    total: Span[Decimal] = Field(description="The total amount exactly as written.")


def recipe(quote, **kwargs):
    kwargs.setdefault("compute", lambda facts, state: "read")
    kwargs.setdefault("answer", solvi.Answer.choice(["read", "matched", "wrong"]))
    return solvi.document(Total, model_of(Server(lambda message, body: (quote, .99))), **kwargs)


@pytest.mark.parametrize("raw,value,currency", [
    ("188'813.24 EUR", "188813.24", "EUR"),
    ("23’388.48 CHF", "23388.48", "CHF"),
    ("-12'345.67 EUR", "-12345.67", "EUR"),
    ("(1'234.50) USD", "-1234.50", "USD"),
    ("(1'234)", "-1234", None),
    ("1'234,50 CHF", "1234.50", "CHF"),
    ("1’234’567.89", "1234567.89", None),
    ("1'234.50 EUR-", "-1234.50", "EUR"),
])
def test_complete_apostrophe_groups_are_exact(raw, value, currency):
    assert _read_quote(Decimal, raw) == (Decimal(value), Unit(currency=currency))


def test_apostrophe_reader_preserves_precision_and_declared_decimal():
    assert _read_quote(Decimal, "123'456'789'012'345'678'901'234'567'890.123456789")[0] == Decimal(
        "123456789012345678901234567890.123456789")
    assert _read_quote(Decimal, "-123'456'789'012'345'678'901'234'567'890.123456789")[0] == Decimal(
        "-123456789012345678901234567890.123456789")
    assert _read_quote(int, "1'234.00")[0] == 1234
    with pytest.raises(ValueError):
        _read_quote(int, "1'234.000000000000000000000000000001")
    for text, decimal in [("1'234.56", ","), ("1'234,56", ".")]:
        with pytest.raises(ValueError):
            _read_quote(Decimal, text, decimal=decimal)
    assert _read_quote(Decimal, "1'234,56", decimal=",")[0] == Decimal("1234.56")
    assert _read_quote(Decimal, "-0'000.00")[0].is_signed()


def test_no_readable_fragment_can_change_a_grouped_amount():
    full = "-12'345'678.90"
    expected = Decimal("-12345678.90")
    for a in range(len(full)):
        for b in range(a + 1, len(full) + 1):
            fragment = full[a:b]
            try:
                value, _ = _read_quote(Decimal, fragment)
            except ValueError:
                continue
            if value != expected:
                assert _ground(fragment, full)[0] is not None, fragment


@pytest.mark.parametrize("raw", [
    "12'34", "1234'567", "12'3456", "1'234'56", "1'234 567", "1'234,567.89", "1'234.567,89",
    "1 '234", "1' 234", "5'", "5' 3\"", "1'234 CR", "[1'234]", "1'234-2'345", "1_234", "2.500",
    "1'234 -", "1'234 - EUR",
])
def test_malformed_or_ambiguous_numeric_quotes_still_refuse(raw):
    with pytest.raises(ValueError):
        _read_quote(Decimal, raw)


@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_number_boundaries_include_apostrophe_groups(apostrophe):
    doc = f"Balance -12{apostrophe}345.67 EUR; separate 345.67."
    first = doc.index("345.67")
    assert cuts_number(doc, first, first + 6)
    assert check_evidence([Quote("345.67", first, first + 6, "doc")], {"doc": doc}) is not None
    assert find_whole("345.67", doc) == doc.rindex("345.67")
    assert locate("345.67", doc) == (doc.rindex("345.67"), doc.rindex("345.67") + 6)
    assert find_normalized(doc[:doc.index(";")], "345.67") is None
    for raw in ["345.67", "12", f"12{apostrophe}"]:
        assert _ground(raw, doc[:doc.index(";")])[0] is not None
    # Feet/minutes punctuation followed by a space is not a thousands group.
    assert find_whole("5", "5' 3\"") == 0
    assert not cuts_number("5'", 0, 1)


@pytest.mark.parametrize("raw,doc", [
    ("3456", "Balance -12'3456 EUR"),
    ("12", "Balance -12'34 EUR"),
    ("1'234", "Balance 1'234e5 EUR"),
    ("1'234", "Balance 1'234_567 EUR"),
    ("1'234.50", "Balance 1'234.50E-3 EUR"),
    ("1'234.50", "Balance -1'234.50 EUR"),
    ("(1'234.50)", "Now 2'345.50 (1'234.50) EUR"),
    ("1'234.50", "Amount 1'234.50 EUR; another 1'234.50 CHF"),
])
def test_partial_sign_unit_and_comparative_guards_still_refuse(raw, doc):
    assert _ground(raw, doc)[0] is not None


def test_own_label_is_read_without_rewriting_the_quote(tmp_path):
    raw = "TOTAL: -70.671,98 EUR"
    doc = "Invoice\n" + raw + "\n"
    s = recipe(raw, storage=tmp_path / "decisions.jsonl",
               compute=lambda facts, state: "matched" if facts.total == Decimal("-70671.98")
               and facts.units["total"].currency == "EUR" else "wrong")
    result = s.ask({"doc": doc})
    assert (result["answer"].status, result["answer"].answer) == ("ok", "matched")
    fact, = result.facts
    assert fact.quote == raw == doc[fact.start:fact.end]
    assert fact.value == raw and fact.start == len("Invoice\n")
    assert result.values["total_value"] == Decimal("-70671.98")
    assert s.verify()["ok"] and s.replay_all() == []


@pytest.mark.parametrize("raw", [
    "SUB TOTAL: 1'234.50 EUR", "Amount: 1'234.50 EUR", "TOTAL estimated: 1'234.50 EUR",
    "TOTAL: 1'234.50 EUR and 2'345.50 EUR", "TOTAL: (2019)", "TOTAL: 2.500",
    "TOTAL: 1'234.50 EUR USD", "TOTAL: 1'234.50 CR", "TOTAL: [1'234.50]",
    "TOTAL:\n1'234.50", "TOTAL: EUR 1\t234", "TOTAL: 1_234",
    "TOTAL: 1'234 -",
])
def test_a_label_does_not_waive_numeric_guards(raw):
    assert recipe(raw).ask({"doc": raw})["answer"].status == "abstain"


def test_label_words_cannot_hide_units_or_digits():
    for name, raw in [("usd", "USD: 1'234"), ("million", "MILLION: 1'234"),
                      ("cr", "CR: 1'234"), ("total2", "TOTAL2: 1'234"),
                      ("per_cent", "PER CENT: 1'234"), ("bps", "BPS: 1'234"),
                      ("basis_points", "BASIS POINTS: 1'234"), ("rs", "Rs: 1'234"),
                      ("rs_total", "Rs Total: 1'234"), ("total_rs", "Total Rs: 1'234")]:
        with pytest.raises(ValueError):
            _read_quote(Decimal, raw, label=name)
    assert _read_quote(Decimal, "TOTAL: 1'234 CHF", label="total") == (Decimal(1234), Unit(currency="CHF"))
    assert _read_quote(Decimal, "Rs 1'234", label="rs") == (Decimal(1234), Unit(currency="INR"))


def test_currency_name_is_not_discarded_as_a_field_label():
    class Rupees(BaseModel):
        rs: Span[Decimal]

    model = model_of(Server(lambda message, body: ("Rs: 1'234", .99)))
    s = solvi.document(Rupees, model, compute=lambda facts, state: "read", answer=solvi.Answer.choice(["read"]))
    assert s.ask({"doc": "Rs: 1'234"})["answer"].status == "abstain"


def test_numeric_reading_change_does_not_silently_reuse_old_calibration():
    # Produced by the unchanged test_document_recipe.Ops before this patch; it attached successfully then.
    fixture = Path(__file__).parent / "fixtures/document_numbers_before.json"
    with pytest.raises(ValueError, match="catalog|fingerprint|different"):
        ops_system(Server(case_reply()), calibration=fixture)
