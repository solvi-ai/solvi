"""solvi.experimental.numguard, the typed tools: <table>, <cmp>, <scale>, <count> computed exactly by code, sharing
the @k numbering with <calc> / <date>, refusing what cannot be computed, and their results as sources in check
(docs/experimental.md#numguard)."""
import datetime as dt
import warnings
from decimal import Decimal

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.numguard import (CallResult, Cell, NumberGuard, ProgramError, Table, annotate, check,
                                             cmp_calc, count_calc, fill, parse_tables, scale_calc, table_calc)

SRC = """Results by region (in thousands):

| Region | 2023 | 2022 | Margin |
|---|---|---|---|
| EU | $1,250 | $1,100 | 12.5% |
| US | (300) | 2,000 | 8% |
| Asia | 1,250 | — | 10% |
| LatAm | -40 | 75 | 4% |

Threshold 1,000; the target region is EU. Invoice dated 2024-03-01, due 2024-04-15."""


def num(text, k):
    """The k-th numeral of the annotated source."""
    return annotate(text)[1][k - 1]


def ref(text, surface, nth=1):
    """The [k] of the nth numeral written `surface`."""
    hits = [k for k, n in enumerate(annotate(text)[1], 1) if n.surface == surface]
    return hits[nth - 1]


# ------------------------------------------------------------------------------------------------ tables
def test_pipe_tables_are_parsed_with_header_labels_and_caption_scale():
    (t,) = parse_tables(SRC)
    assert t.columns == ["2023", "2022", "Margin"] and t.labels == ["EU", "US", "Asia", "LatAm"]
    assert t.shape == (4, 3) and t.scale == 1000
    assert "values in thousands" in t.describe()
    t2 = parse_tables("Costs, $ in millions\n| a | b |\n| x | 1 |\n\nno table here\n| c | d |\n| y | 2 |")
    assert [x.scale for x in t2] == [10 ** 6, 1]


def test_explicit_tables_and_short_rows():
    t = Table([["", "Q1", "Q2"], ["A", "1"], ["B", "3", "4"]])
    assert t.rows[1] == ["A", "1", ""]
    with pytest.raises(ValueError):
        Table([["only a header"]])
    assert table_calc('sum("Q1")', [[["", "Q1"], ["A", "2"], ["B", "5"]]]) == 7


def test_cell_by_labels_and_by_numbers_with_parenthesised_and_negative_numbers():
    assert table_calc('cell("US", "2023")', SRC) == Decimal(-300)
    assert table_calc("cell(4, 1)", SRC) == Decimal(-40)
    assert table_calc('cell("eu", "margin")', SRC) == Decimal("12.5")     # labels: case and spaces ignored
    v, cells = table_calc('cell("EU", 2)', SRC, cells=True)
    assert v == 1100 and cells == (Cell(1, 1, 2, "EU", "2022", "$1,100"),)
    assert table_calc('cell("EU", "2023", norm)', SRC) == 1250000                  # the caption's scale
    assert table_calc('cell("EU", "Region")', SRC) == "EU"                         # the label column by its header


def test_aggregates_over_a_column_range_and_filter():
    assert table_calc('sum("2023")', SRC) == 2160
    assert table_calc('sum("2023", norm)', SRC) == 2160000
    assert table_calc('sum("2023", 2..4)', SRC) == 910
    assert table_calc('sum("2023", "US".."Asia")', SRC) == 950
    assert table_calc('avg("Margin")', SRC) == Decimal("8.625")
    assert table_calc('avg("Margin", norm)', SRC) == Decimal("8.625")              # percents are not scaled
    assert table_calc('min("2023")', SRC) == -300
    assert table_calc('count("2022")', SRC) == 3                                   # the blank cell is not counted
    assert table_calc('sum("2023", where "Margin" >= 10%)', SRC) == 2500
    # a blank cell matches no condition, so "not" of one is true: Asia joins
    assert table_calc('sum("2023", where "Region" in ("EU", "US") or not "2022" > 0)', SRC) == 2200
    assert table_calc('sum("2022", where "2023" between -300 and 0)', SRC) == 2075
    assert table_calc('argmin("Margin")', SRC) == "LatAm"


def test_ties_max_gives_every_tied_cell_and_argmax_refuses():
    v, cells = table_calc('max("2023")', SRC, cells=True)
    assert v == 1250 and [c.row_label for c in cells] == ["EU", "Asia"]
    with pytest.raises(ProgramError, match="a tie, rows 'EU', 'Asia'"):
        table_calc('argmax("2023")', SRC)
    assert table_calc('argmax("2023", 1..2)', SRC) == "EU"


def test_a_second_table_and_no_row_labels():
    text = SRC + "\n\n| x | y |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    assert table_calc('table2: sum("y")', text) == 6
    t = Table([["x", "y"], ["1", "2"], ["3", "4"]], row_labels=False)
    assert table_calc("argmax(2)", t) == 2 and table_calc("cell(1, 1)", t) == 1
    with pytest.raises(ProgramError, match="no row labels"):
        table_calc('cell("a", 1)', t)


@pytest.mark.parametrize("expr, why", [
    ('avg("2022")', "blank"),                                    # a blank cell in an aggregate
    ('cell("Asia", "2022")', "blank"),
    ('sum("2023", 3..1)', "is empty"),
    ('avg("2023", where "Margin" > 50%)', "division by zero"),
    ('sum("2023", where "Margin" > 50%)', "no rows selected"),
    ('sum("Region")', "not a number"),
    ('cell(9, 1)', "no row 9"),
    ('cell(1, 7)', "no column 7"),
    ('cell("Mars", 1)', "no row labelled 'Mars'"),
    ('table3: sum(1)', "table 3 does not exist"),
    ('median(1)', "table allows"),
    ('sum("2023", where "Margin" > 5)', None),                   # unitless literal against a percent is allowed
    ('sum("2023", where "Margin" > $5)', "cannot read"),
    ('sum("2023", where "Region" > "A")', "only with == and !="),
    ('sum("2023", where "2023" == "x")', "a number with a str"),
])
def test_table_validators(expr, why):
    if why is None:
        table_calc(expr, SRC)
        return
    with pytest.raises(ProgramError, match=why):
        table_calc(expr, SRC)


def test_mixed_units_and_scale_words_in_an_aggregate_are_refused_unless_normalised():
    t = Table([["", "v"], ["a", "4.2 million"], ["b", "300 thousand"]])
    with pytest.raises(ProgramError, match="mixed scales"):
        table_calc('sum("v")', t)
    assert table_calc('sum("v", norm)', t) == 4500000
    t = Table([["", "v"], ["a", "$5"], ["b", "7%"]])
    with pytest.raises(ProgramError, match="mixed units"):
        table_calc('sum("v", norm)', t)
    with pytest.raises(ProgramError, match="needs a table"):
        table_calc('sum("v")', [])


# ------------------------------------------------------------------------------------------------ cmp
def test_cmp_numbers_dates_strings_and_logic():
    assert cmp_calc("3 < 5 and 5 <= 5 and 7 > 2 and 2 >= 2 and 4 == 4.0 and 4 != 5") is True
    assert cmp_calc('2024-03-01 between 2024-01-01 and 2024-03-01') is True
    assert cmp_calc('"EU" == "eu" and not ("a" != "a")') is True
    assert cmp_calc("1 > 2 or (3 in (1, 2, 3) and 4 not in (1, 2))") is True
    assert cmp_calc("-5 < -4.5") is True


@pytest.mark.parametrize("expr, why", [
    ("3 < 2024-01-01", "a number with a date"),
    ('"a" < "b"', "only with == and !="),
    ("12% > 5% and 3 > 1%", None),
    ("3 between 1", "between needs 'and'"),
    ("3", "not a yes/no"),
    ("[1] > 2", "use the call inside fill"),
    ("@1 > 2", "use the call inside fill"),
    ("3 < ", "ends too early"),
    ("3 < 4 4", "unexpected '4'"),
])
def test_cmp_validators(expr, why):
    if why is None:
        assert cmp_calc(expr) is True
        return
    with pytest.raises(ProgramError, match=why):
        cmp_calc(expr)


def test_cmp_refuses_mixed_units_and_scale_words_from_the_input():
    src = "Sales $4.2 million, last year 300 thousand; margin 12% against a €15 fee."
    f = fill("<cmp>[1] > [2]</cmp> <cmp>[3] > [4]</cmp> <cmp>[3] > 10</cmp>", src, literals="any")
    assert "mixed scales" in f.errors[0] and "mixed units" in f.errors[1]
    assert f.calls[2].value is True


# ------------------------------------------------------------------------------------------------ scale
@pytest.mark.parametrize("expr, want", [
    ("1250 thousand to million", Decimal("1.25")),
    ("1.25 million to units", Decimal(1250000)),
    ("3 bn to k", Decimal(3000000)),
    ("2 trillion -> billion", Decimal(2000)),
    ("12.5% to fraction", Decimal("0.125")),
    ("12.5 percent to bp", Decimal(1250)),
    ("0.035 fraction to percent", Decimal("3.5")),
    ("14 days to weeks", Decimal(2)),
    ("6 months to years", Decimal("0.5")),
    ("2 quarters to months", Decimal(6)),
    ("2 years to days rule 360", Decimal(720)),
    ("1 year to days rule 365", Decimal(365)),
    ("90 days in months rule 30/360", Decimal(3)),
])
def test_scale_conversions_are_exact(expr, want):
    assert scale_calc(expr) == want


def test_a_fraction_that_does_not_end_has_fifty_digits():
    assert str(scale_calc("10 days to weeks")).startswith("1.428571428571428571428571428571428571428571428571")


@pytest.mark.parametrize("expr, why", [
    ("3 months to days", "no exact ratio"),
    ("2 weeks to years", "no exact ratio"),
    ("5 million to days", "different kinds of unit"),
    ("5 to thousand", "give the unit"),
    ("5 thousand to parsecs", "unknown target unit"),
    ("5 thousand million", "write VALUE"),
    ("5 days to weeks rule 360", "a rule applies only"),
    ("5 months to days rule 400", "is not one of"),
    ('"x" thousand to million', "not a number"),
])
def test_scale_validators(expr, why):
    with pytest.raises(ProgramError, match=why):
        scale_calc(expr)


def test_scale_takes_the_inputs_own_scale_word_or_percent_and_refuses_a_conflict():
    src = "Revenue $4.2 million, margin 12.5%, staff 45."
    f = fill("<scale>[1] to thousand</scale>; <scale>[2] to fraction</scale>; <scale>[1] thousand to units</scale>; "
             "<scale>[2] million to units</scale>; <scale>[3] to million</scale>", src)
    assert [c.shown for c in f.calls[:2]] == ["4200", "0.125"]
    assert "written in millions, not thousand" in f.calls[2].error
    assert "is a percent, not million" in f.calls[3].error
    assert "give the unit" in f.calls[4].error


# ------------------------------------------------------------------------------------------------ count
def test_count_lists_columns_and_membership():
    assert count_calc("(1, 5, 7, 12) if > 4") == 3
    assert count_calc("(1, 5, 7, 12)") == 4
    assert count_calc('column("2023") if between 0 and 1300', SRC) == 2
    assert count_calc('column("2022", where "Margin" < 10%)', SRC) == 2
    assert count_calc('column("Region") if in ("EU", "Mars")', SRC) == 1
    assert count_calc('"asia" in column("Region")', SRC) is True
    assert count_calc('"Mars" not in column("Region")', SRC) is True
    assert count_calc("3 in (1, 2)") is False


@pytest.mark.parametrize("expr, why", [
    ("()", "an empty list"),
    ("3", "a list"),
    ('(1, "a") if > 0', "a str with a number"),
    ('column("2023", 4..2)', "is empty"),
    ('3 in column("Region")', "a number with a str"),
])
def test_count_validators(expr, why):
    with pytest.raises(ProgramError, match=why):
        count_calc(expr, SRC)


# ------------------------------------------------------------------------------------------------ fill and check
def test_fill_shares_the_call_numbering_and_nests_refs_in_table_filters():
    thr = ref(SRC, "1,000")
    text = (f"Top region <table>argmax(\"2023\", 1..2)</table>, with <table>cell(@1, \"2023\")</table> thousand; "
            f"<count>column(\"2023\") if >= [{thr}]</count> regions reach the threshold; above it: "
            f"<table>sum(\"2023\", where \"2023\" >= [{thr}] and \"Region\" != @1)</table>; that is "
            f"<scale>@4 thousand to million</scale> million; <calc>@2 - @4</calc> more; EU leads: "
            f"<cmp>@2 >= @4 and @1 == \"EU\"</cmp>.")
    f = fill(text, SRC)
    assert f.errors == []
    assert f.text == ("Top region EU, with 1,250 thousand; 2 regions reach the threshold; above it: 1,250; that is "
                      "1.25 million; 0 more; EU leads: yes.")
    assert [c.kind for c in f.calls] == ["table", "table", "count", "table", "scale", "calc", "cmp"]


def test_fill_nested_ref_in_a_filter_against_a_later_threshold():
    text = ('<table>cell("EU", "Margin")</table> <table>count("2023", where "Margin" < @1)</table> '
            '<table>avg("Margin", where "Margin" < @1)</table>')
    f = fill(text, SRC)
    assert f.errors == [] and [c.shown for c in f.calls] == ["12.5", "3", "7.333333333"]


def test_booleans_render_yes_no_or_as_configured_and_are_not_numbers():
    src = "Budget 1,000 and spend 900 on 2024-03-01."
    assert fill("Within budget: <cmp>[2] <= [1]</cmp>.", src).text == "Within budget: yes."
    assert fill("<cmp>[2] > [1]</cmp>", src, booleans=("true", "false")).text == "false"
    f = fill("<cmp>[2] > [1]</cmp> <calc>@1 + 1</calc> <date>[3] + @1D</date>", src)
    assert "a yes/no, not a number" in f.errors[0] and "a yes/no, not a number" in f.errors[1]
    assert NumberGuard().run("<cmp>[2] <= [1]</cmp>, @1", src).text == "yes, yes"


def test_literals_in_typed_calls_follow_the_policy_but_pointers_do_not():
    f = fill('<table>sum("2023", where "2023" > 999)</table> <table>cell(4, 3)</table>', SRC)
    assert "the literal 999 is not allowed" in f.errors[0] and len(f.errors) == 1
    f = fill("<cmp>[{}] > 2024-01-01</cmp>".format(ref(SRC, "2024-03-01")), SRC)
    assert "literal date" in f.errors[0]
    assert fill("<cmp>[{}] > 2024-01-01</cmp>".format(ref(SRC, "2024-03-01")), SRC, literals="any").text == "yes"
    assert fill("<cmp>[{}] < [{}]</cmp>".format(ref(SRC, "2024-03-01"), ref(SRC, "2024-04-15")), SRC).text == "yes"


def test_typed_call_errors_name_the_call_and_bad_refs():
    f = fill('<table>cell(@5, 1)</table> <cmp>[99] > 1</cmp> <count>@1 in (1, 2)</count>', SRC)
    errs = " | ".join(f.errors)
    assert "call @1 <table>" in errs and "@5 is not an earlier call" in errs
    assert "[99] does not exist" in errs and "@1 failed" in errs


def test_fill_with_tables_given_apart():
    t = Table([["", "Q1", "Q2"], ["A", "10", "20"], ["B", "30", "(5)"]])
    f = fill('<table>sum("Q2")</table>', "A plain source.", tables=[t])
    assert f.text == "15" and f.tables == [t]
    assert "needs a table" in fill("<table>sum(1)</table>", "A plain source.").errors[0]


def test_check_reports_the_mode_and_the_cells():
    g = NumberGuard()
    out = g.run('Total <table>sum("2023", 1..2)</table>; <count>column("2023") if > 0</count> regions; '
                '<scale>@1 thousand to million</scale>.', SRC)
    assert out.ok, out.verdict.feedback()
    nums = out.verdict.numbers
    assert [(c.surface, c.source, c.index, c.how) for c in nums] == [
        ("950", "tool", 1, "table"), ("2", "tool", 2, "count"), ("0.95", "tool", 3, "scale")]
    assert [c.row_label for c in nums[0].cells] == ["EU", "US"] and nums[2].cells == nums[0].cells
    d = out.verdict.to_dict()["numbers"][0]
    assert d["cells"][0] == {"table": 1, "row": 1, "col": 1, "row_label": "EU", "col_label": "2023",
                             "surface": "$1,250"}
    # a model that typed the result: matched to the call's value, how = the mode
    calls = out.calls
    v = check("It was 950 and 2.", SRC, calls)
    assert [(c.source, c.how) for c in v.numbers] == [("tool", "table"), ("tool", "count")]
    assert v.numbers[0].cells == calls[0].cells


def test_check_accepts_cells_of_tables_given_apart():
    t = Table([["", "Q1"], ["A", "1,234"], ["B", "(56)"]])
    v = check("Q1 of A was 1,234 and B lost 56; 999 is invented.", "No table here.", tables=[t])
    assert [(c.surface, c.source, c.how) for c in v.numbers] == [
        ("1,234", "input", "table cell"), ("56", "input", "sign"), ("999", "unsourced", "")]
    g = NumberGuard(on_fail="report")
    out = g.run("A: <table>cell(\"A\", 1)</table>, also 1,234.", "No table here.", tables=[t])
    assert out.ok and [c.how for c in out.verdict.numbers] == ["table", "table cell"]


def test_calc_and_date_keep_pointer_and_exact():
    out = NumberGuard().run("<calc>[1] - [2]</calc>", "Was 10 then 4.")
    assert [c.how for c in out.verdict.numbers] == ["pointer"]
    assert [c.how for c in check("6", "Was 10 then 4.", out.calls).numbers] == ["exact"]
    assert out.calls[0].cells == () and isinstance(out.calls[0], CallResult)


def test_a_date_cell_compares_as_a_date():
    t = Table([["", "due"], ["a", "2024-03-01"], ["b", "2024-05-01"]])
    assert table_calc('cell("a", "due")', t) == dt.date(2024, 3, 1)
    assert count_calc('column("due") if between 2024-01-01 and 2024-04-01', t) == 1


def test_a_label_twice_is_ambiguous_and_wants_the_number():
    t = Table([["", "v"], ["Total", "1"], ["Total", "2"]])
    with pytest.raises(ProgramError, match="rows 1, 2 are all labelled 'Total'"):
        table_calc('cell("Total", "v")', t)
    assert table_calc('cell(2, "v")', t) == 2
