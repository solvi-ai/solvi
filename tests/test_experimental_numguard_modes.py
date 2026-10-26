"""solvi.experimental.numguard: aggregates over a table row (argmax / argmin give the column label), time and speed
units in <scale>, the opt-in constant pi, and the public mode registry with extra modes run by fill
(docs/experimental.md#numguard)."""
import datetime as dt
import warnings
from decimal import Decimal

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.numguard import (MODES, PI, SMALL, CallContext, NumberGuard, ProgramError, Table, calc,
                                             check, cmp_calc, count_calc, fill, register_mode, scale_calc,
                                             table_calc, unregister_mode)

SALES = """Units sold by product:

| Product | 2021 | 2022 | 2023 | Status |
|---|---|---|---|---|
| Alpha | 120 | 150 | 140 | open |
| Beta | 80 | — | 95 | open |
| Type | actual | actual | forecast | |
"""


# ------------------------------------------------------------------------------------------------ row aggregates
def test_row_aggregates_over_all_columns_and_a_range():
    assert table_calc('sum(row "Alpha", "2021".."2023")', SALES) == Decimal(410)
    assert table_calc('avg(row "Alpha", 1..3)', SALES) == Decimal("136.66666666666666666666666666666666666666666666667")
    assert table_calc('max(row "Alpha", 1..3)', SALES) == Decimal(150)
    assert table_calc('min(row 1, 1..3)', SALES) == Decimal(120)
    assert table_calc('count(row "Beta")', SALES) == 3                # the dash is blank; "open" counts
    assert table_calc('count(row "Beta", 1..3)', SALES) == 2


def test_row_argmax_returns_the_column_label():
    v, cells = table_calc('argmax(row "Alpha", 1..3)', SALES, cells=True)
    assert v == "2022" and [(c.row_label, c.col_label) for c in cells] == [("Alpha", "2022")]
    assert table_calc('argmin(row "Alpha", "2021".."2023")', SALES) == "2021"
    out = fill('Best year: <table>argmax(row "Alpha", 1..3)</table>.', SALES)
    assert out.text == "Best year: 2022." and not out.errors


def test_row_filter_compares_another_row_in_each_column():
    assert table_calc('sum(row "Alpha", where "Type" == "actual")', SALES) == Decimal(270)
    assert table_calc('argmax(row "Alpha", where "Type" == "actual")', SALES) == "2022"
    assert table_calc('sum(row "Alpha", 1..3, where not "Type" == "actual")', SALES) == Decimal(140)


def test_row_aggregates_refuse_blanks_text_ties_and_empty_selections():
    with pytest.raises(ProgramError, match="row 2 column 2 is blank"):
        table_calc('sum(row "Beta", 1..3)', SALES)
    with pytest.raises(ProgramError, match="not a number"):
        table_calc('sum(row "Alpha")', SALES)                         # "open" in Status
    t = Table([["x", "a", "b"], ["r", "5", "5"]])
    with pytest.raises(ProgramError, match="a tie, columns 'a', 'b'"):
        table_calc('argmax(row "r")', t)
    with pytest.raises(ProgramError, match="no rows selected"):
        table_calc('max(row "Alpha", where "Type" == "none")', SALES)


def test_column_syntax_unchanged_and_column_keyword_optional():
    assert table_calc('sum("2021", 1..2)', SALES) == table_calc('sum(column "2021", 1..2)', SALES) == Decimal(200)
    assert table_calc('argmax("2023", 1..2)', SALES) == "Alpha"


def test_count_and_membership_over_a_row():
    assert count_calc('row("Alpha", 1..3) if > 130', SALES) == 2
    assert count_calc('"forecast" in row("Type")', SALES) is True


# ------------------------------------------------------------------------------------------------ scale: time, speed
@pytest.mark.parametrize("expr, want", [
    ("36 km/h to m/s", "10"), ("1 m/s to km/h", "3.6"), ("10 m/s to km/h", "36"), ("1 km/h to m/s", None),
    ("90 minutes to hours", "1.5"), ("2 hours to seconds", "7200"), ("1 day to hours", "24"),
    ("1 week to minutes", "10080"), ("45 s to min", "0.75"), ("1 year to hours rule 365", "8760"),
    ("1 mph to km/h", "1.609344"), ("1 hour to days", None),
])
def test_scale_time_and_speed_are_exact(expr, want):
    v = scale_calc(expr)
    if want is not None:
        assert v == Decimal(want)
    elif expr.startswith("1 km/h"):
        assert v == Decimal("0.27777777777777777777777777777777777777777777777778")
    else:
        assert v * 24 == Decimal(1)


def test_scale_refuses_mixed_kinds_and_rule_less_months():
    with pytest.raises(ProgramError, match="different kinds"):
        scale_calc("5 km/h to hours")
    with pytest.raises(ProgramError, match="no exact ratio"):
        scale_calc("3600 seconds to months")
    assert scale_calc("1 month to hours rule 360") == Decimal(720)


def test_scale_speed_from_an_input_numeral():
    out = fill("<scale>[1] km/h to m/s</scale> m/s", "The car drove at 72 km/h.")
    assert out.text == "20 m/s" and not out.errors


# ------------------------------------------------------------------------------------------------ pi
def test_pi_value_and_calc():
    assert str(PI) == "3.141592653589793238462643383" and len(PI.as_tuple().digits) == 28
    assert calc("2 * pi") == 2 * PI and calc("PI") == PI
    assert PI not in SMALL


def test_pi_is_opt_in_inside_fill():
    src = "The radius is 3 m."
    out = fill("Area <calc>pi * [1]^2</calc> m2", src)
    assert out.errors and "pi is not an allowed constant" in out.errors[0]
    out = fill("Area <calc>pi * [1]^2</calc> m2", src, constants=SMALL | {PI})
    assert not out.errors and out.calls[0].value == Decimal("28.274333882308139146163790447")
    assert not fill("<calc>pi</calc>", src, literals="any").errors
    assert cmp_calc("pi > 3")
    bad = fill("<cmp>[1] < pi</cmp>", src)
    assert bad.errors and "pi is not an allowed constant" in bad.errors[0]


# ------------------------------------------------------------------------------------------------ the registry
def test_registry_lists_the_builtin_modes():
    assert list(MODES) == ["calc", "date", "table", "cmp", "scale", "count"]
    for m in MODES.values():
        assert m.builtin and m.slots and m.result and m.description and callable(m.executor)
    assert MODES["cmp"].validator(True) and not MODES["cmp"].validator(Decimal(1))
    assert MODES["calc"].validator(Decimal(1)) and not MODES["calc"].validator(True)
    assert MODES["date"].validator(dt.date(2024, 1, 1)) and MODES["date"].validator(3)
    assert MODES["count"].validator(2) and MODES["table"].validator("2022")
    assert MODES["scale"].executor("1 m/s to km/h") == Decimal("3.6")
    with pytest.raises(TypeError):
        MODES["x"] = None                                         # read-only view


def test_builtin_modes_cannot_be_replaced_or_removed():
    with pytest.raises(ValueError):
        register_mode("calc", lambda e, c: Decimal(0))
    with pytest.raises(ValueError):
        unregister_mode("table")
    with pytest.raises(ValueError):
        register_mode("Bad-Name", None)


@pytest.fixture
def extra_modes():
    names = []

    def add(name, *a, **kw):
        names.append(name)
        return register_mode(name, *a, **kw)
    yield add
    for n in names:
        if n in MODES:
            unregister_mode(n)


def test_an_extra_mode_runs_in_fill_shares_numbering_and_is_checked(extra_modes):
    seen = {}

    def lookup(expr, ctx: CallContext):
        seen["ctx"] = ctx
        return {"boiling point": Decimal(100)}[expr]
    m = extra_modes("lookup", lookup, slots="a fact name", result="Decimal",
                    validator=lambda v: isinstance(v, Decimal), description="a fact the input does not hold")
    assert MODES["lookup"] is m and not m.builtin
    src = "Water at 20 degrees."
    out = fill("<lookup>boiling point</lookup> °C, <calc>@1 - [1]</calc> more", src)
    assert out.text == "100 °C, 80 more" and not out.errors
    assert seen["ctx"].k == 1 and len(seen["ctx"].inputs) == 1
    v = check(out.text, src, out.calls, inserted=out.inserted)
    assert v.ok and v.numbers[0].how == "lookup"


def test_extra_mode_failures_fail_closed(extra_modes):
    extra_modes("logic", None, description="inference by an external engine")
    out = fill("<logic>a and b</logic>", "x")
    assert out.errors and "has no executor" in out.errors[0]
    extra_modes("boom", lambda e, c: 1 / 0)
    out = fill("<boom>x</boom>", "x")
    assert out.errors and "ZeroDivisionError" in out.errors[0]
    extra_modes("typed", lambda e, c: "text", result="Decimal", validator=lambda v: isinstance(v, Decimal))
    out = fill("<typed>x</typed>", "x")
    assert out.errors and "returned a str, not Decimal" in out.errors[0]
    extra_modes("typed", lambda e, c: Decimal(2), replace=True)
    assert not fill("<typed>x</typed>", "x").errors
    with pytest.raises(ValueError, match="replace=True"):
        register_mode("typed", None)


def test_an_unregistered_mode_tag_is_plain_text(extra_modes):
    extra_modes("logic", None)
    unregister_mode("logic")
    out = NumberGuard(on_fail="report").run("<logic>a</logic> fine", "x")
    assert out.filled.calls == [] and "<logic>a</logic>" in out.text
