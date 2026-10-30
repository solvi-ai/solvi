"""The 2026-10-30 external handoff, documents part (S308/S310–S313): solvi.document's and numguard's number
readers aligned and completed, every refusal still the author's never-guess rule. A number in parentheses right
after another number on its line is a comparative — numguard leaves the parentheses and reads it unsigned (what
document's grounding abstains on); space-grouped digits read when the quote writes their currency; a dot that
groups thousands is a per-document declaration (`document(..., decimal="dot")`), not a guess; a table's units
header ("SEK million", "(in millions)") applies to its cells in `check`; a scale word binds to the number on its
own line. Parser-level: no model, no network."""
import inspect
import warnings
from decimal import Decimal

import pytest
from pydantic import BaseModel

import solvi
from solvi import Span

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.numguard import ProgramError, check, fill, numerals, parse_tables, table_calc


# ------------------------------------------------------------------------------------------------ S313
def test_space_grouped_digits_read_when_the_quote_writes_their_currency():
    """The currency is the escape hatch textin documents; _read_quote used to strip it before the value reader."""
    from solvi.solutions.document import Unit, _ground, _read_quote
    assert _read_quote(float, "€1 234") == (1234.0, Unit(currency="EUR"))
    assert _read_quote(float, "$1 234") == (1234.0, Unit(currency="$"))
    assert _read_quote(float, "EUR 1 234")[0] == 1234.0
    assert _ground("€1 234", "Total €1 234 units") == (None, {})          # grounds — and now it reads
    with pytest.raises(ValueError, match="digits grouped by spaces"):     # bare: may be two numbers, protected
        _read_quote(float, "1 234")
    with pytest.raises(ValueError, match="one number or two"):            # "1 ,234": protected since 1.4
        _read_quote(float, "1 ,234")


# ------------------------------------------------------------------------------------------------ S310
def test_parentheses_after_a_number_are_a_comparative_in_fill_and_check():
    """numguard used to read "(18.8%)" after "16.6%" as -18.8; document abstains on the same text."""
    from solvi.solutions.document import _ground
    src = "a margin of 16.6% (18.8%). Earnings per share SEK 2.84 (5.45)."
    assert {n.surface: n.number for n in numerals(src)} == {
        "16.6%": Decimal("16.6"), "(18.8%)": Decimal("18.8"), "2.84": Decimal("2.84"), "(5.45)": Decimal("5.45")}
    assert fill("margin [1] against [2]", src).text == "margin 16.6% against (18.8%)"    # the parentheses stay
    v = check("margin 16.6% against (18.8%)", src)
    assert [(c.surface, c.source, c.how) for c in v.numbers] == [
        ("16.6%", "input", "exact"), ("(18.8%)", "input", "exact")]       # both sides read it unsigned now
    assert v.ok
    assert [c.how for c in check("margin 16.6%; a year ago -18.8%", src).numbers] == ["exact", "sign"]
    assert "a comparative or a negative" in _ground("(18.8%)", src)[0]    # document abstains on the same shape


# ------------------------------------------------------------------------------------------------ S312
def test_a_scale_word_binds_to_the_number_on_its_own_line():
    """The scale word used to cross a new line: '£1,000\\nmillion' was one numeral with the newline inside."""
    ns = numerals("The £1,000\nmillion revolving credit facility")
    assert [n.surface for n in ns] == ["£1,000"]
    assert ns[0].number == Decimal(1000) and ns[0].scale == 1 and "\n" not in ns[0].surface
    one = numerals("The £1,000 million revolving credit facility")        # same text on one line: still one
    assert one[0].surface == "£1,000 million" and one[0].value == Decimal(1000000000)


# ------------------------------------------------------------------------------------------------ S308
def test_a_dot_that_groups_thousands_is_a_declaration_not_a_guess():
    from solvi.core.types import span_value
    from solvi.solutions.document import _read_quote
    with pytest.raises(ValueError) as e:                                  # the default abstains and names the knob
        _read_quote(float, "30.000")
    assert 'document(..., decimal="dot")' in str(e.value) and "TextIn" not in str(e.value)
    assert _read_quote(float, "1.500.000")[0] == 1500000.0                # multi-group dots always read
    dot = ","                                            # the decimal mark a dot-thousands document writes
    assert _read_quote(float, "30.000", decimal=dot)[0] == 30000.0        # the declaration, end to end
    assert _read_quote(float, "1.000", decimal=dot)[0] == 1000.0
    assert _read_quote(float, "12,5", decimal=dot)[0] == 12.5             # the comma reads decimals
    assert _read_quote(float, "30,000", decimal=".")[0] == 30000.0        # the comma-thousands way
    assert _read_quote(float, "12.5", decimal=".")[0] == 12.5
    assert span_value(float, "30.000", spec={"decimal": ","}) == 30000.0  # the low-level passthrough


def test_document_exposes_the_decimal_knob():
    class Total(BaseModel):
        total: Span[float]

    assert "decimal" in inspect.signature(solvi.document).parameters
    with pytest.raises(ValueError, match='decimal must be "dot"'):
        solvi.document(Total, object(), decimal="point")
    with pytest.raises(ValueError, match='decimal must be "dot"'):
        solvi.document(Total, object(), decimal=".")


# ------------------------------------------------------------------------------------------------ S311
def test_a_units_header_scales_the_tables_cells_in_check():
    src = "| SEK million | Q3 2025 |\n|---|---|\n| Net sales | 18,742 |\n"
    (t,) = parse_tables(src)
    assert t.scale == 10 ** 6 and t.currency == "SEK"
    assert "values in SEK millions" in t.describe()
    (c,) = check("Net sales were SEK 18,742 million.", src).numbers       # used to be unsourced (or rescale-only)
    assert (c.surface, c.source, c.how) == ("18,742 million", "input", "table cell")
    apart = check("Net sales were SEK 18,742 million.", "No table here.", tables=src)
    assert [(c.surface, c.source, c.how) for c in apart.numbers] == [("18,742 million", "input", "table cell")]
    assert check("Net sales were 18,742.", "No table here.", tables=src).ok is False   # six orders off
    from solvi.experimental.numguard import NumberGuard
    out = NumberGuard(on_fail="report").run("Net sales were 18,742 million.", src)     # the guard's own path
    assert out.ok, out.verdict.feedback()


def test_units_headers_in_millions_and_currency_symbols():
    (t,) = parse_tables("| (in millions) | Q3 2025 |\n|---|---|\n| Net sales | 18,742 |\n")
    assert t.scale == 10 ** 6 and t.currency == ""
    (t2,) = parse_tables("| £ million | Q3 2025 |\n|---|---|\n| x | 5 |\n")
    assert t2.scale == 10 ** 6 and t2.currency == "£"
    (c,) = check("Net sales were 18,742 million.",
                 "| (in millions) | Q3 2025 |\n|---|---|\n| Net sales | 18,742 |\n").numbers
    assert (c.source, c.how) == ("input", "table cell")


def test_a_currency_header_is_the_unit_of_its_cells():
    src = "| £ million | 2023 | 2022 |\n|---|---|---|\n| EU | 1,250 | $1,100 |\n"
    (t,) = parse_tables(src)
    assert t.currency == "£" and t.scale == 10 ** 6
    assert table_calc('sum("2023", norm)', src) == Decimal(1250000000)
    with pytest.raises(ProgramError, match="mixed units"):                # a $ cell in a £ table: refused
        table_calc("sum(row 1)", src)
