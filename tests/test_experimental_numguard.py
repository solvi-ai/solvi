"""solvi.experimental.numguard: pointers filled with exact values, every number of the final text with a source, a
fail-closed guard and one retry (docs/experimental.md#numguard)."""
import datetime as dt
import warnings
from decimal import Decimal

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.numguard import (MEASURED_CONSTANTS, SMALL, CallResult, NumberGuard, ProgramError,
                                             UnsourcedNumbers, annotate, calc, check, date_calc, fill, numerals, years)

SOURCE = ("Revenue was $1,250 million in 2023 and $1,100 million in 2022. Invoice dated 2024-03-01, net 45 days. "
          "Margin fell to 12.5% from (3.2)% a year ago; headcount 8,220.")


def kinds(text):
    return [(n.kind, n.surface) for n in numerals(text)]


# ------------------------------------------------------------------------------------------------ numerals
def test_negative_numbers_minus_unicode_minus_and_parentheses():
    ns = [n for n in numerals("a loss of -5, then −7.5, and a reported (12.3) in the table") if n.kind == "number"]
    assert [n.number for n in ns] == [Decimal(-5), Decimal("-7.5"), Decimal("-12.3")]
    assert ns[2].plain() == "-12.3"


def test_parentheses_after_another_number_are_a_comparative_not_a_negative():
    # "16.6% (18.8%)": the same shape solvi.solutions.document abstains on — the sign is not read (S310)
    ns = numerals("a margin of 16.6% (18.8%). Earnings per share SEK 2.84 (5.45).")
    got = {n.surface: n.number for n in ns if n.kind == "number"}
    assert got == {"16.6%": Decimal("16.6"), "(18.8%)": Decimal("18.8"), "2.84": Decimal("2.84"),
                   "(5.45)": Decimal("5.45")}
    assert ns[1].plain() == "(18.8%)"                       # the parentheses stay: as the input writes it
    assert numerals("12.3 percent (11.8)")[1].number == Decimal("11.8")
    assert numerals("SEK 1,234 m (1,100)")[1].number == Decimal("1100")
    assert numerals("2.1x\n(2.5)")[1].number == Decimal("2.5")     # one new line: still its logical line
    assert [n.number for n in numerals("(5.45) then (300) | (2)")] == \
        [Decimal("-5.45"), Decimal(-300), Decimal(-2)]                     # alone: negative


def test_a_hyphen_between_numbers_is_not_a_sign():
    assert [n.number for n in numerals("in 2019-2020 and pages 3-5")] == [2019, 2020, 3, 5]


def test_percents_with_sign_word_and_space():
    ns = numerals("up 12.5%, then 3 percent, then 4 per cent and 7 %")
    assert [(n.number, n.unit) for n in ns] == [(Decimal("12.5"), "%"), (3, "%"), (4, "%"), (7, "%")]


def test_comma_thousands_with_decimals_and_currency():
    ns = numerals("paid $1,234,567.89 and €2,000")
    assert [n.number for n in ns] == [Decimal("1234567.89"), 2000]
    assert ns[0].unit == "$" and ns[0].surface == "$1,234,567.89"


def test_dates_in_several_formats():
    ns = numerals("on 2024-03-15, 15.03.2024, 15/03/2024, March 15, 2024, 15 March 2024, 15th of March and Mar 15")
    assert all(n.kind == "date" for n in ns)
    assert [n.date for n in ns] == [dt.date(2024, 3, 15)] * 5 + [(3, 15), (3, 15)]


def test_an_impossible_date_is_not_a_date():
    assert all(n.kind != "date" for n in numerals("on 2024-02-30"))


def test_numbers_inside_identifiers_are_identifiers():
    got = kinds("order A-10457, COVID-19, v2.1, Q3, 1Z999AA10 and FY2023")
    assert got == [("id", "A-10457"), ("id", "COVID-19"), ("id", "v2.1"), ("id", "Q3"), ("id", "1Z999AA10"),
                   ("id", "FY2023")]


def test_a_number_with_a_unit_or_an_ordinal_is_a_number():
    ns = numerals("ran 5km on the 3rd day")
    assert [(n.kind, n.number, n.unit) for n in ns] == [("number", 5, ""), ("number", 3, "rd")]


def test_scale_words_and_suffixes():
    ns = numerals("$4.2 million, 3 billion, 12k users, $1.2bn and 7 thousand")
    assert [(n.number, n.scale, n.value) for n in ns] == [
        (Decimal("4.2"), 10 ** 6, Decimal("4200000.0")), (3, 10 ** 9, 3 * 10 ** 9), (12, 1000, 12000),
        (Decimal("1.2"), 10 ** 9, Decimal("1200000000.0")), (7, 1000, 7000)]


def test_annotate_numbers_numbers_and_dates_but_not_identifiers():
    text, nums = annotate("Order A-7 of 2024-03-01: 3 items at $1,250.")
    assert text == "Order A-7 of [1]2024-03-01: [2]3 items at [3]$1,250."
    assert [n.kind for n in nums] == ["date", "number", "number"]


# ------------------------------------------------------------------------------------------------ executors
@pytest.mark.parametrize("expr,want", [("1250 - 1100", "150"), ("0.1 + 0.2", "0.3"), ("-(3 - 5)", "2"),
                                       ("2^10", "1024"), ("2**3**2", "512"), ("50%", "0.5"), ("1,000 * 3", "3000"),
                                       ("2^-1", "0.5"), ("-2^2", "-4"), ("(2 + 3) * 4", "20"), ("4^0.5", "2")])
def test_calc_is_exact_decimal(expr, want):
    assert calc(expr) == Decimal(want)


def test_calc_keeps_fifty_digits():
    assert str(calc("150 / 1100 * 100")).startswith("13.6363636363636363636363636363636363636363636363")
    assert fill("<calc>1 / 3 * 3</calc>", "").text == "1"                          # 50 digits, shown to 10


@pytest.mark.parametrize("expr", ["", "1 +", "(1 + 2", "1 / 0", "2 $ 3", "1 2", "(-8)^0.5", "0^-1", "abc"])
def test_malformed_calc_raises_a_readable_program_error(expr):
    with pytest.raises(ProgramError):
        calc(expr)


def test_date_arithmetic():
    assert date_calc("2024-01-31 + 1M") == dt.date(2024, 2, 29)
    assert date_calc("31.01.2023 + 1M") == dt.date(2023, 2, 28)
    assert date_calc("2024-03-01 + 45D") == dt.date(2024, 4, 15)
    assert date_calc("2024-03-01 + 2W - 1Y") == dt.date(2023, 3, 15)
    assert date_calc("2024-03-01 - 2024-01-01") == 60
    assert date_calc("DAYS 2024-03-01 - 2024-01-01") == 60
    assert date_calc("WEEKDAY 2024-04-15") == "Monday"


@pytest.mark.parametrize("expr", ["45D", "2024-03-01 + 45", "2024-03-01 + 2024-01-01", "2024-03-01 - 2024-01-01 + 1D",
                                  "2024-13-01 + 1D", "WEEKDAY 2024-03-01 - 2024-01-01"])
def test_malformed_date_raises(expr):
    with pytest.raises(ProgramError):
        date_calc(expr)


# ------------------------------------------------------------------------------------------------ fill
def test_fill_writes_input_numerals_as_the_input_writes_them_and_calls_exactly():
    f = fill("Revenue grew by <calc>[1] - [3]</calc> million, to [1]; margin [7] (was [8]).", SOURCE)
    assert f.errors == []
    assert f.text == "Revenue grew by 150 million, to $1,250 million; margin 12.5% (was (3.2)%)."
    assert [c.shown for c in f.calls] == ["150"]


def test_fill_does_not_double_a_currency_or_a_percent_around_the_pointer():
    f = fill("It was $[1] and [7]%.", SOURCE)
    assert f.text == "It was $1,250 million and 12.5%."


def test_fill_results_refer_by_at_k_and_braces_and_dates_use_iso():
    f = fill("<calc>[1]-[3]</calc> then {@1} and @1; due <date>[5] + [6]D</date>, a <date>WEEKDAY @2</date>.", SOURCE,
             show_calls=False)
    assert f.errors == []
    assert f.text == " then 150 and 150; due , a ."
    f = fill("Due <date>[5] + [6]D</date> (<date>WEEKDAY @1</date>), <date>@1 - [5]</date> days.", SOURCE)
    assert f.text == "Due 2024-04-15 (Monday), 45 days."
    assert fill("Due <date>[5] + [6]D</date>.", SOURCE, date_format="%d %B %Y").text == "Due 15 April 2024."


def test_fill_formats_results_digits_places_and_grouping():
    assert fill("<calc>[1] * [9]</calc>", SOURCE).text == "10,275,000"                 # the source groups thousands
    assert fill("<calc>[1] * [9]</calc>", SOURCE, grouping=False).text == "10275000"
    assert fill("<calc>[1] / [3]</calc>", SOURCE).text == "1.136363636"
    assert fill("<calc>[1] / [3]</calc>", SOURCE, places=2).text == "1.14"
    assert fill("<calc>[1] / [3]</calc>", SOURCE, digits=4).text == "1.136"


def test_a_negative_input_goes_into_calc_in_parentheses():
    assert fill("<calc>[1] - [2]</calc>", "fell to 12.5% from -3.2% a year ago").text == "15.7"


def test_pointer_out_of_range_is_an_error_and_stays_as_written():
    f = fill("It is [42] and <calc>[0] + 1</calc>.", SOURCE)
    assert any("[42]" in e for e in f.errors) and any("[0]" in e for e in f.errors)
    assert "[42]" in f.text


def test_unknown_later_or_failed_call_is_an_error():
    f = fill("<calc>@2 + 1</calc> <calc>(1</calc> <calc>@2 * 2</calc> and @7", SOURCE)
    errs = " | ".join(f.errors)
    assert "@2 is not an earlier call" in errs and "'(' is not closed" in errs and "@2 failed" in errs
    assert "@7, which does not exist" in errs


def test_a_date_in_calc_or_a_number_as_a_date_is_an_error():
    f = fill("<calc>[5] + 1</calc> <date>[7] + 1D</date>", SOURCE)
    assert "use <date> for dates" in f.errors[0] and "not a full date or a whole number" in f.errors[1]


def test_literals_constants_input_any():
    assert fill("<calc>[1] * 1.07</calc>", SOURCE).errors[0].endswith("point at the input ([k]) or an earlier call (@k)")
    assert fill("<calc>[1] * 100</calc>", SOURCE).errors == []                         # 100 is a constant
    assert fill("<calc>[1] / 365</calc>", SOURCE).errors != []                         # 365 only in the measured set
    assert fill("<calc>[1] / 365</calc>", SOURCE, constants=MEASURED_CONSTANTS).errors == []
    assert fill("<calc>1250 - 1100</calc>", SOURCE, literals="input").errors == []
    assert fill("<calc>[1] * 1.07</calc>", SOURCE, literals="any").errors == []
    assert fill("<date>[5] + 400D</date>", SOURCE).errors != []                         # a date literal amount too


def test_a_call_may_name_itself():
    assert fill("<calc>@1 = [1] - [3]</calc>", SOURCE).text == "150"


def test_answer_marker_keeps_only_the_answer_and_its_spans():
    f = fill("<calc>[1]-[3]</calc>\nANSWER: {@1} million", SOURCE, answer_marker="ANSWER:")
    assert f.text == "150 million" and f.inserted[0].start == 0
    assert fill("no marker", SOURCE, answer_marker="ANSWER:").errors == ["no 'ANSWER:' line: the answer goes after it"]


# ------------------------------------------------------------------------------------------------ check
def by_surface(v):
    return {c.surface: (c.source, c.index, c.how) for c in v.numbers}


def test_check_sources_a_typed_text():
    v = check("Revenue $1,250 million, down to -1,100 million? 45 days, 2023, 77 and 9,999.", SOURCE)
    s = by_surface(v)
    assert s["$1,250 million"] == ("input", 1, "exact")
    assert s["-1,100 million"] == ("input", 3, "sign")
    assert s["45"] == ("input", 6, "exact") and s["2023"] == ("input", 2, "exact")
    assert s["77"] == ("unsourced", 0, "")                                            # not in the strict default
    assert s["9,999"] == ("unsourced", 0, "")
    assert not v.ok and [c.surface for c in v.unsourced] == ["77", "9,999"]
    assert by_surface(check("77", SOURCE, constants=MEASURED_CONSTANTS))["77"] == ("constant", 0, "exact")


def test_signed_check_refuses_a_flipped_sign():
    src = "Margin fell to 12.5% from -3.2% a year ago."
    assert check("loss 3.2%", src, signed=True).ok is False
    assert check("loss -3.2%", src, signed=True).ok


def test_scale_words_normalised_and_rescale_is_an_option_and_reported():
    v = check("revenue 1,250,000,000 and 1.25 billion", SOURCE)
    assert [c.how for c in v.numbers] == ["scale_word", "scale_word"]
    assert check("headcount 8,220,000", SOURCE).ok is False
    v = check("headcount 8,220,000 and 8.22", SOURCE, rescale=True)
    assert [c.how for c in v.numbers] == ["rescaled x1000", "rescaled /1000"]


def test_rounding_is_an_option():
    src = "ratio 13.636363 and 0.5"
    assert check("about 13.64", src).ok is False
    assert by_surface(check("about 13.64", src, rounding=True))["13.64"] == ("input", 1, "rounded")


def test_dates_and_years_of_dates():
    v = check("Dated March 1, 2024 (in 2024), due 2024-04-15.", SOURCE, [CallResult(1, "date", "", dt.date(2024, 4, 15))])
    s = by_surface(v)
    assert s["March 1, 2024"] == ("input", 5, "exact") and s["2024"] == ("input", 5, "date_year")
    assert s["2024-04-15"] == ("tool", 1, "exact")
    assert check("due 2024-04-16", SOURCE).ok is False


def test_results_from_outside_count_as_tool_sources():
    v = check("the API said 4,567.8", SOURCE, ["total: 4567.8"])
    assert by_surface(v)["4,567.8"] == ("tool", 1, "exact")
    assert check("got 31", SOURCE, [Decimal(31)], constants=SMALL).ok


def test_constants_are_configurable():
    assert check("year 2031", SOURCE).ok is False
    assert check("year 2031", SOURCE, constants=SMALL | years(2000, 2040)).ok
    assert check("77 items", SOURCE).ok is False and check("12 items", SOURCE).ok        # SMALL is the default
    assert check("77 items", SOURCE, constants=MEASURED_CONSTANTS).ok
    assert check("77 items", SOURCE, constants=lambda d: d < 1000).ok


def test_identifiers_are_skipped_or_checked_verbatim():
    assert check("ticket INV-99", SOURCE).numbers == []
    v = check("ticket INV-99 and A-7", "Ticket A-7 is open", ids="check")
    assert by_surface(v) == {"INV-99": ("unsourced", 0, ""), "A-7": ("input", 0, "verbatim")}


def test_verdict_serialises():
    d = check("45 and 9,999", SOURCE).to_dict()
    assert d["ok"] is False and d["counts"] == {"input": 1, "tool": 0, "constant": 0, "unsourced": 1}
    assert d["numbers"][1] == {"surface": "9,999", "start": 7, "end": 12, "kind": "number", "source": "unsourced",
                               "index": 0, "how": ""}


# ------------------------------------------------------------------------------------------------ the guard
def test_guard_run_fills_and_every_number_written_by_code_is_sourced():
    out = NumberGuard().run("Grew <calc>[1] - [3]</calc> million (<calc>@1 / [3] * 100</calc>%), due "
                            "<date>[5] + [6]D</date>.", SOURCE)
    assert out.ok and not out.refused
    assert out.text == "Grew 150 million (13.63636364%), due 2024-04-15."
    assert [c.how for c in out.verdict.numbers] == ["pointer"] * 3
    assert [(c.source, c.index) for c in out.verdict.numbers] == [("tool", 1), ("tool", 2), ("tool", 3)]


def test_guard_modes_refuse_raise_report():
    text = "Revenue was 1,300 million."
    out = NumberGuard().run(text, SOURCE)
    assert out.refused and not out.ok and out.text == NumberGuard().refusal
    assert [c.surface for c in out.verdict.unsourced] == ["1,300 million"]
    with pytest.raises(UnsourcedNumbers) as e:
        NumberGuard(on_fail="raise").run(text, SOURCE)
    assert "1,300 million" in str(e.value) and e.value.outcome.verdict.unsourced
    rep = NumberGuard(on_fail="report").run(text, SOURCE)
    assert rep.text == text and not rep.ok and not rep.refused


def test_a_program_error_fails_the_guard_even_without_numbers():
    out = NumberGuard(on_fail="report").run("It is [12].", SOURCE)
    assert not out.ok and out.verdict.unsourced == [] and out.verdict.errors


def test_retry_hook_asks_again_with_the_unsourced_numbers():
    seen = []

    def model(feedback):
        seen.append(feedback)
        return "Revenue was 1,300 million." if feedback is None else "Revenue was [1]."
    out = NumberGuard().run_with_retry(model, SOURCE)
    assert out.ok and out.text == "Revenue was $1,250 million." and len(out.attempts) == 2
    assert seen[0] is None and "'1,300 million'" in seen[1] and "[k]" in seen[1]


def test_retry_hook_refuses_after_the_last_retry():
    calls = []
    out = NumberGuard().run_with_retry(lambda fb: calls.append(fb) or "It is 9,999.", SOURCE, retries=2)
    assert out.refused and len(calls) == 3 and len(out.attempts) == 3
    with pytest.raises(UnsourcedNumbers):
        NumberGuard(on_fail="raise").run_with_retry(lambda fb: "It is 9,999.", SOURCE, retries=0)


def test_guard_ids_check_and_extra_results():
    g = NumberGuard(on_fail="report", ids="check")
    out = g.run("Order A-7 total [1].", "Order A-7: $1,250.")
    assert out.ok and [c.how for c in out.verdict.numbers] == ["verbatim", "pointer"]
    out = NumberGuard().run("<calc>[6] * 2</calc>; API total 4,567.8", SOURCE, results=["4567.8"])
    assert out.ok and [(c.source, c.index) for c in out.verdict.numbers] == [("tool", 1), ("tool", 1)]


def test_outcome_to_dict():
    d = NumberGuard().run("Grew <calc>[1] - [3]</calc>.", SOURCE).to_dict()
    assert d["ok"] and d["calls"] == [{"k": 1, "kind": "calc", "expr": "[1] - [3]", "shown": "150", "error": ""}]


def test_fill_earlier_result_in_exponent_form_is_read_back():
    # 1 246 300 / 100 % is 1.2463E+6 as str(Decimal); the next call must still read it
    f = fill("<calc>[1] / [2]%</calc> then <calc>@1 - [3]</calc>", "1,246.3 and 0.1% and 300")
    assert not f.errors, f.errors
    assert [c.value for c in f.calls] == [Decimal("1246300"), Decimal("1246000")]


@pytest.mark.parametrize("sign", ["-", "−", "–"])
@pytest.mark.parametrize("amount", ["369.7", "$369.7 billion", "$1,234.5", "369.7%"])
def test_unicode_minus_preserved_through_numguard(sign, amount):
    source = "Net transactions were " + sign + amount + "."
    n, = numerals(source)
    assert n.number == (-Decimal("1234.5") if "1,234" in amount else -Decimal("369.7"))
    assert n.surface == sign + amount
    assert annotate(source)[0] == "Net transactions were [1]" + sign + amount + "."
    assert fill("Result [1].", source).text == "Result " + sign + amount + "."
    assert not check("Result " + amount + ".", source, signed=True).ok
    assert check("Result " + sign + amount + ".", source, signed=True).ok


@pytest.mark.parametrize("measurement", ["4.0 percent", "7.0", "3.0 percent", "4%", "$7", "0 million"])
def test_default_small_constants_do_not_source_measurements(measurement):
    assert not check(measurement, "Reported 48.9 percent.").ok
    assert check(measurement, measurement).ok
    assert check(measurement, "", results=[Decimal(measurement.split()[0].strip("%$"))]).ok


def test_explicit_constants_can_allow_decimal_measurements():
    assert check("7.0", "", constants={Decimal("7.0")}).ok
    assert check("4%", "", constants=lambda value: value == 4).ok
    assert check("7 items and 100 total", "").ok


@pytest.mark.parametrize("text", ["$21.87 quadrillion", "3 quintillions", "2 gazillion", "4 тыс.",
                                  "4 млн", "4 млрд", "4 трлн", "2026,000", "105,6 млрд",
                                  "12,34", "1,2345", "1,234,56", "$2026,000", "(2026,000)"])
def test_unsupported_numeric_formats_refuse_instead_of_partial_values(text):
    for operation in (numerals, annotate, lambda value: fill("Value [1]", value),
                      lambda value: check(value, "21.87 and 2026"),
                      lambda value: NumberGuard().run(value, "21.87 and 2026")):
        with pytest.raises(ValueError, match="unsupported|ambiguous"):
            operation(text)


def test_supported_numeric_formats_and_ordinary_following_words_keep_spans():
    text = "$1,234.56 trillion; 105 people; 3 млнx; 42%; 12k; 4.2 million; 1,234,567"
    nums = numerals(text)
    assert [n.value for n in nums] == [Decimal("1234560000000000"), Decimal(105), Decimal(3),
                                      Decimal(42), Decimal(12000), Decimal(4200000), Decimal(1234567)]
    assert all(text[n.start:n.end] == n.surface for n in nums)
    assert check("$1,234.56 trillion; 42%", text).ok


def test_explicit_decimal_comma_contract():
    text = "Рост −₽136,60; 2,7%; (3,4)% и 1 234,50."
    ns = numerals(text, decimal=",")
    assert [n.number for n in ns] == [Decimal('-136.60'), Decimal('2.7'), Decimal('-3.4'), Decimal('1234.50')]
    assert all(text[n.start:n.end] == n.surface for n in ns)
    marked, annotated = annotate(text, decimal=",")
    assert annotated == ns and '[1]' in marked
    assert not check('Цены 2,7%.', 'Цены 4,9%.', decimal=',').ok
    assert check('Цены 4,9%.', 'Цены 4,9%.', decimal=',').ok
    guard = NumberGuard(decimal=',')
    out = guard.run('Рост [1].', 'Рост 4,9%.')
    assert out.ok and '4,9%' in out.text
    assert guard.run('<calc>[1] + [1]</calc>', 'Рост 4,9.').ok


@pytest.mark.parametrize('text', ['2,7', '2026,000', '1,23,456'])
def test_decimal_comma_keeps_protected_default(text):
    with pytest.raises(ValueError):
        numerals(text)


@pytest.mark.parametrize('text', ['1.5', '.5', '1,234.5', '1,2,3', '1234 000,5', '12 34,5'])
def test_decimal_comma_rejects_mixed_or_foreign_punctuation(text):
    with pytest.raises(ValueError):
        numerals(text, decimal=',')


def test_decimal_comma_mode_validation_and_default_grouping():
    assert numerals('1,234.50')[0].number == Decimal('1234.50')
    for decimal in ['auto', '', None]:
        with pytest.raises(ValueError):
            numerals('3', decimal=decimal)


def test_comma_typed_integer_is_measurement():
    assert not check("4,0", decimal=",").ok
    assert check("4", decimal=",").ok


@pytest.mark.parametrize("operation", ["numerals", "check", "run"])
def test_text_number_work_scales_without_pairwise_scans(operation):
    """Count executed Python lines, not wall time; 3x input must not need 9x work."""
    import sys
    from solvi.experimental import numguard as ng

    def work(size):
        source = " ".join(f"Account ${10000 + i}.25 today." for i in range(size))
        typed = " ".join(f"Value ${10000 + i}.25." for i in reversed(range(size)))
        reply = " ".join(f"[{i}]" for i in range(1, size + 1))
        count = 0
        def trace(frame, event, arg):
            nonlocal count
            if event == "line" and frame.f_code.co_filename == ng.__file__:
                count += 1
            return trace
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            if operation == "numerals":
                result = ng.numerals(source)
            elif operation == "check":
                result = ng.check(typed, source)
            else:
                result = ng.NumberGuard(on_fail="report").run(reply, source)
        finally:
            sys.settrace(previous)
        assert len(result) == size if operation == "numerals" else result.ok
        return count

    small, large = work(80), work(240)
    assert large < 5 * small, (operation, small, large)


def test_indexed_number_matching_preserves_full_scan_ranking():
    from solvi.experimental import numguard as ng
    # Exact outranks earlier sign/scale matches; ties retain original source order.
    entries = [("input", 1, Decimal("-4.2"), Decimal("-4200000")),
               ("input", 2, Decimal("4200000"), Decimal("4200000")),
               ("tool", 3, Decimal("4.2"), Decimal("4200000")),
               ("tool", 4, Decimal("4.2"), Decimal("4200000")),
               ("input", 5, Decimal("13.63636"), Decimal("13.63636"))]
    for pool in (entries, entries[::-1]):
        indexed = ng._NumberPool(pool)
        for n in ng.numerals("4.2 million; −4.2 million; 4200000; 4.2; 13.64; 999"):
            for signed in (False, True):
                for rescale, rounding in ((False, False), (True, False), (False, True), (True, True)):
                    assert ng._match_number(n, indexed, signed, rescale, rounding) == ng._match_number(
                        n, pool, signed, rescale, rounding)


def test_inserted_sweep_preserves_first_overlap_in_caller_order():
    from solvi.experimental.numguard import Inserted
    # Deliberately unsorted, overlapping spans: the first iterable entry wins,
    # not the leftmost or the shortest. Once expired, the next overlap wins.
    spans = [Inserted(4, 10, "tool", 9), Inserted(0, 15, "input", 7), Inserted(0, 3, "tool", 1)]
    verdict = check("101 202 303 404", inserted=spans, constants=frozenset())
    assert [(n.source, n.index) for n in verdict.numbers] == [
        ("input", 7), ("tool", 9), ("tool", 9), ("input", 7)]


def test_span_index_preserves_dates_ids_and_locale():
    source = "2024-03-15 A-10457 March 16, 2024 1,234.50; −5.25; (7.5)%"
    assert [(n.kind, n.surface) for n in numerals(source)] == [
        ("date", "2024-03-15"), ("id", "A-10457"), ("date", "March 16, 2024"),
        ("number", "1,234.50"), ("number", "−5.25"), ("number", "(7.5)%")]
    assert [n.surface for n in numerals("2024-03-15 A-10457 1 234,50", decimal=",")] == [
        "2024-03-15", "A-10457", "1 234,50"]


def test_ru_output_locale_roundtrips_without_inserted_trust():
    src = 'The goods and services deficit was $105.6 billion; prices rose 3.4 percent. The update begins September 30, 2026.'
    template = 'Дефицит составил [1], цены выросли на [2], пересмотр начнётся [3].'
    filled = fill(template, src, locale='ru')
    assert filled.text == 'Дефицит составил 105,6 млрд долл., цены выросли на 3,4%, пересмотр начнётся 30 сентября 2026 г.'
    assert check(filled.text, src, locale='ru', constants=frozenset()).ok
    assert not check(filled.text.replace('105,6', '105,7'), src, locale='ru', inserted=filled.inserted).ok
    assert NumberGuard(locale='ru').run(template, src).ok
    assert '$105.6 billion' in fill(template, src).text


def test_ru_reader_spans_grouping_aliases_and_locale_validation():
    text = '1\u00a0400\u00a0000; 2 500,5 млн руб.; 30 сентября 2026 г.'
    nums = numerals(text, locale='ru')
    assert [n.value for n in nums[:2]] == [Decimal('1400000'), Decimal('2500500000')]
    assert nums[1].unit == '₽'
    assert nums[2].date == dt.date(2026, 9, 30)
    assert all(text[n.start:n.end] == n.surface for n in nums)
    for function, args in ((fill, ('[1]', '12.5')), (check, ('12', '12')), (numerals, ('12',))):
        with pytest.raises(ValueError, match='locale'):
            function(*args, locale='unknown')


def test_ru_source_and_output_punctuation_are_independent_and_calls_checked():
    assert NumberGuard(locale='ru', decimal='.').run('[1] плюс <calc>[1] + 1</calc>', '12.5').ok
    assert fill('[1]', '12,5', decimal=',', locale='ru').text == '12,5'
    with pytest.raises(ValueError):
        fill('[1]', '12,5', decimal='.', locale='ru')
    verdict = check('1 400 000; 2,5 млн долл.; 3,4%', '1,400,000; $2.5 million; 3.4 percent', locale='ru')
    assert verdict.ok and len(verdict.numbers) == 3
    assert [n.index for n in verdict.numbers] == [1, 2, 3]
    assert check('30 сентября 2026 г.', 'September 30, 2026', locale='ru').ok
    assert not check('30 сентября 2027 г.', 'September 30, 2026', locale='ru').ok


def test_ru_pointer_preserves_full_precision_and_provenance():
    source = '$123456789.123456789 million'
    result = fill('[1]', source, locale='ru', grouping=True)
    assert result.text == '123\u00a0456\u00a0789,123456789 млн долл.'
    assert check(result.text, source, locale='ru', constants=frozenset()).ok
    parsed = numerals(result.text, locale='ru')[0]
    assert parsed.value == Decimal('123456789123456.789')
    assert result.inserted[0].source == 'input' and result.inserted[0].index == 1
    assert result.text[result.inserted[0].start:result.inserted[0].end] == result.text


def test_ru_inserted_pointer_keeps_designated_duplicate_provenance():
    f = fill('[2]', '12.5; 12.5', locale='ru')
    v = check(f.text, '12.5; 12.5', inserted=f.inserted, locale='ru')
    assert v.ok
    assert (v.numbers[0].source, v.numbers[0].index, v.numbers[0].how) == ('input', 2, 'pointer')
    f = fill('[2]', '11.5; 12.5', locale='ru')
    # The replacement matches source1, but is no longer what pointer2 meant.
    v = check('11,5', '11.5; 12.5', inserted=f.inserted, locale='ru')
    assert not v.ok and v.numbers[0].source == 'unsourced'
    tool = fill('<calc>[1] + 1</calc>', '12.5', locale='ru')
    v = check(tool.text, '12.5', results=tool.calls, inserted=tool.inserted, locale='ru')
    assert v.ok and (v.numbers[0].source, v.numbers[0].index) == ('tool', 1)


@pytest.mark.parametrize('text', ['12,5 руб. долларов', '12,5 долл. евро',
                                  '12,5 млн млрд', '12,5 млн руб. млн', '12,5 тыс. тыс.'])
def test_ru_rejects_adjacent_currency_and_scale_suffix_chains(text):
    with pytest.raises(ValueError, match='suffix chain'):
        numerals(text, locale='ru')
    with pytest.raises(ValueError, match='suffix chain'):
        check(text, '$12.5', locale='ru', constants=frozenset())


def test_ru_designated_date_and_typed_tool_cells_survive_verification():
    source = 'September 30, 2026; September 30, 2026'
    f = fill('[2].', source, locale='ru')
    v = check(f.text, source, inserted=f.inserted, locale='ru')
    assert v.ok and v.numbers[0].index == 2 and v.numbers[0].how == 'pointer'
    table = '| Region | Amount |\n| --- | --- |\n| EU | 12.5 |'
    f = fill('<table>cell("EU", "Amount")</table>', table, locale='ru')
    assert not f.errors
    v = check(f.text, table, f.calls, inserted=f.inserted, locale='ru')
    assert v.ok and v.numbers[0].source == 'tool' and v.numbers[0].index == 1
    assert v.numbers[0].how == 'table' and v.numbers[0].cells == f.calls[0].cells
