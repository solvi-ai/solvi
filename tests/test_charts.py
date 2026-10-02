"""solvi.charts / solvi.specialist: every number of a chart is checked against the source; traps are caught; the SVG is
deterministic, accessible and replays."""
import io
import json
import xml.etree.ElementTree as ET
from itertools import combinations

import pytest

from solvi.charts import (PALETTE, ChartSpecialist, FixedProposer, LLMProposer, chart, contrast,
                          render_svg)
from solvi.charts.render import BG, INK, MIN_FONT, MUTED
from solvi.specialist import BLOCKED, CHANGED, DROPPED, WARNING, Trace

PRESS = """ACME Corp. reports third-quarter 2025 results

BERLIN, 14 October 2025 — ACME Corp. today reported revenue of $4.2 billion for the third quarter, up 12% from a year
earlier. By region, Europe accounted for 42% of revenue, North America for 35% and Asia-Pacific for 23%. Operating
margin improved by 3 percentage points to 18%. Headcount grew 15% to 480 employees. The company opened 12 new stores."""

QUARTERS = """Quarterly revenue, fiscal 2025 (unaudited)
Q1 2025: $1.2 million
Q2 2025: $1.5 million
Q3 2025: $1.9 million
Q4 2025: $2.3 million
Total revenue for the year was $6.9 million."""

WASTE = """In 2025 the city's waste was handled as follows: 1,200 tonnes were recycled, 800 tonnes composted and 2,500
tonnes sent to landfill, a total of 4,500 tonnes."""

SURVEY = """Respondents could pick several answers. 48% said they commute by bus, 39% by bicycle and 21% by car."""

RU = """Выручка сети за сентябрь: Москва — 1 500 000 руб., Казань — 900 000 руб., Самара — 650 000 руб."""


def spec(points, kind="bar", unit="%", scale="", title="Chart", total=None, series_unit=None, name=""):
    s = {"kind": kind, "title": title, "unit": unit, "scale": scale,
         "series": [{"name": name, "unit": series_unit, "points": [
             {"label": lab, "value": v, "quote": ({"text": q} if isinstance(q, str) else q)} for lab, v, q in points]}]}
    if total:
        s["total"] = {"label": "Total", "value": total[0], "quote": {"text": total[1]}}
    return s


def run(text, s, **kw):
    return ChartSpecialist(FixedProposer(s), **kw).run(text)


def codes(r, severity=None):
    return [i.code for i in r.issues if severity is None or i.severity == severity]


# ------------------------------------------------------------------------------------------------ the happy paths
def test_rule_proposer_press_release_pie():
    r = chart(PRESS, "revenue by region")
    assert r.ok and r.checked.verified.kind == "pie"
    assert [p.label for p in r.checked.verified.series[0].points] == ["Europe", "North America", "Asia-Pacific"]
    assert not codes(r, DROPPED)
    for label in ("Europe: 42%", "North America: 35%", "Asia-Pacific: 23%"):
        assert label in r.output


def test_rule_proposer_quarters_line_with_total():
    r = chart(QUARTERS, "quarterly revenue")
    v = r.checked.verified
    assert r.ok and v.kind == "line" and v.unit == "USD" and v.scale == "million"
    assert v.total is not None and str(v.total.value) == "6.9"
    assert "total_mismatch" not in codes(r)
    assert "$1.2m" in r.output and "$2.3m" in r.output


def test_rule_proposer_labels_after_the_number_and_total():
    r = chart(WASTE)
    v = r.checked.verified
    assert [c for c in v.categories] == ["recycled", "composted", "landfill"]
    assert v.unit == "tonne" and v.unit_label == "tonnes" and v.total.value == 4500
    assert "Tonnes" in r.output and "total_mismatch" not in codes(r)


@pytest.mark.parametrize("text", ["In 2023 and 2024 things happened.", "Since 2019 offices opened in Lyon.",
                                  "From 2020 to 2024 stores closed."])
def test_rule_proposer_does_not_chart_a_year_followed_by_a_word(text):
    r = chart(text)
    assert not r.ok and r.checked.verified is None and r.proposal.series[0].points == []


def test_rule_proposer_still_charts_a_four_digit_count_that_is_not_a_year():
    r = chart("In 2023 the plant had 2000 employees and the office 1950 employees.")
    v = r.checked.verified
    assert r.ok and v.unit == "employee" and sorted(p.as_written for p in v.series[0].points) == ["1950", "2000"]


def test_russian_currency_space_grouped():
    r = chart(RU)
    v = r.checked.verified
    assert r.ok and v.unit == "RUB" and v.categories == ["Москва", "Казань", "Самара"]
    assert v.scale == "thousand" and "1,500k ₽" in r.output and "650k ₽" in r.output


# ------------------------------------------------------------------------------------------------------------ traps
def test_trap_wrong_unit_percent_vs_points_and_employees():
    s = spec([("Operating margin change", 3, "3 percentage points"),     # pp charted as %
              ("Headcount", 15, "15%")], unit="employees")               # % charted as employees
    r = run(PRESS, s)
    assert not r.ok and codes(r, DROPPED) == ["unit_mismatch", "unit_mismatch"]
    s2 = spec([("Headcount", 480, "480 employees"), ("Stores", 12, "12 new stores")], unit="employees")
    r2 = run(PRESS, s2)
    assert r2.ok and [p.label for p in r2.checked.verified.series[0].points if p] == ["Headcount"]
    d = r2.dropped[0]
    assert d.code == "unit_mismatch" and "'employee' is not next to" in d.message


def test_trap_made_up_value_by_stand_in_proposer():
    s = spec([("Europe", 42, "42%"), ("North America", 35, "35%"),
              ("Asia-Pacific", 32, "23%"),          # a digit swap: the quote says 23
              ("Latin America", 7, "7%"),           # not in the text at all
              ("Africa", 3, None)], kind="pie")     # no quote
    r = run(PRESS, s)
    assert codes(r, DROPPED) == ["value_mismatch", "quote_outside", "no_quote"]
    assert "pie_refused" in codes(r, CHANGED)       # a slice did not verify → no whole to show
    v = r.checked.verified
    assert v.kind == "bar" and v.categories == ["Europe", "North America", "Asia-Pacific", "Latin America", "Africa"]
    assert [p is None for p in v.series[0].points] == [False, False, True, True, True]
    assert ">32<" not in r.output and "7%" not in r.output          # made-up numbers are never drawn
    assert "not verified in the source" in r.output
    assert "2 values quoted from the source; 3 proposed, not verified" in r.output
    rep = r.report()
    assert "dropped: series[0].points[2] — 'Asia-Pacific' = 32%: the quote states '23%', not 32" in rep


def test_trap_percentages_that_do_not_sum():
    r = chart(SURVEY, "how people commute")
    assert r.proposal.kind == "bar"                 # the rule proposer does not propose a pie: 108%
    s = spec([("bus", 48, "48%"), ("bicycle", 39, "39%"), ("car", 21, "21%")], kind="pie")
    r = run(SURVEY, s)
    assert r.ok and r.checked.verified.kind == "bar"
    msg = [i.message for i in r.issues if i.code == "pie_refused"][0]
    assert "add up to 108%, not 100%" in msg
    ok = run(PRESS, spec([("Europe", 42, "42%"), ("North America", 35, "35%"), ("Asia-Pacific", 23, "23%")], kind="pie"))
    assert ok.checked.verified.kind == "pie" and not ok.issues


def test_trap_scale_and_currency():
    r = run(PRESS, spec([("Revenue Q3", 4.2, "$4.2 billion")], unit="USD"))           # 4.2 is 4.2 billion
    assert codes(r) == ["scale_mismatch", "nothing_verified"] and r.output is None
    r = run(PRESS, spec([("Revenue Q3", 4.2, "$4.2 billion")], unit="USD", scale="million"))
    assert codes(r, DROPPED) == ["scale_mismatch"]
    r = run(PRESS, spec([("Revenue Q3", 4200, "$4.2 billion")], unit="USD", scale="million"))
    assert r.ok and not r.issues
    r = run(PRESS, spec([("Revenue Q3", 4.2, "$4.2 billion")], unit="EUR", scale="billion"))
    assert codes(r, DROPPED) == ["unit_mismatch"] and "in USD, the chart shows it in EUR" in r.dropped[0].message
    r = run(PRESS, spec([("Growth", 12, "12%")], unit=""))                          # a percentage as a plain number
    assert codes(r, DROPPED) == ["unit_mismatch"]


def test_trap_quote_cuts_a_number_and_quote_at_wrong_offset():
    t = "Churn was 14.2% in March and 4.2% in April."
    r = run(t, spec([("March", 4.2, {"text": "4.2%", "start": t.index("14.2%") + 1})]))   # inside "14.2%"
    assert codes(r, DROPPED) == ["no_number"]
    r = run(t, spec([("April", 4.2, {"text": "4.2%", "start": 3})]))
    assert codes(r, DROPPED) == ["quote_outside"]
    r = run(t, spec([("March", 14.2, "14.2%"), ("April", 4.2, "4.2%")], kind="line"))  # "4.2%" also occurs in "14.2%"
    assert r.ok and not codes(r, DROPPED)
    assert [p.start for p in r.checked.verified.series[0].points] == [t.index("14.2%"), t.index(" 4.2%") + 1]


def test_trap_one_number_drawn_twice():
    r = run(PRESS, spec([("Europe", 42, "42%"), ("Europe again", 42, {"text": "42%", "start": PRESS.index("42%")})]))
    assert codes(r, DROPPED) == ["quote_reused"]


def test_trap_invented_numbers_in_labels_and_title():
    r = run(QUARTERS, spec([("Q1 2025", 1.2, "$1.2 million"), ("Q1 2027", 1.5, "$1.5 million")], unit="USD",
                           scale="million", title="Revenue 2025 vs 2027"))
    assert codes(r, DROPPED) == ["label_number"]
    assert r.checked.verified.title == "Revenue 2025 vs [?]" and "text_number" in codes(r, CHANGED)


def test_trap_ambiguous_numbers():
    t = "Units shipped: north 1.000, south 2.500, and 3 100 in the west."
    r = run(t, spec([("north", 1000, "1.000"), ("south", 2500, "2.500"), ("west", 3100, "3 100")], unit=""))
    assert codes(r, DROPPED) == ["ambiguous_number"] * 3 and not r.ok
    r = run(t, spec([("north", 1000, "1.000"), ("south", 2500, "2.500")], unit=""), decimal=",")
    assert r.ok and not codes(r, DROPPED)


def test_totals_stated_and_inconsistent():
    s = spec([("recycled", 1200, "1,200"), ("composted", 800, "800")], unit="tonnes", total=(4500, "4,500"))
    r = run(WASTE, s)
    assert r.ok and codes(r, WARNING) == ["total_mismatch"]
    s["kind"] = "pie"
    r = run(WASTE, s)
    assert r.checked.verified.kind == "bar" and "the slices add up to 2000, the source's total is 4500" in r.report()
    s = spec([("recycled", 1200, "1,200"), ("composted", 800, "800"), ("landfill", 2500, "2,500")], unit="tonnes",
             kind="pie", total=(4500, "4,500"))
    r = run(WASTE, s)
    assert r.checked.verified.kind == "pie" and not r.issues


def test_series_with_another_unit_is_dropped_and_label_far_from_value_warns():
    s = {"kind": "bar", "title": "t", "unit": "%", "series": [
        {"name": "share", "points": [{"label": "Europe", "value": 42, "quote": {"text": "42%"}}]},
        {"name": "staff", "unit": "employees", "points": [{"label": "Europe", "value": 480, "quote": {"text": "480"}}]}]}
    r = run(PRESS, s)
    assert "unit_mismatch_series" in codes(r, DROPPED) and len(r.checked.verified.series) == 1
    r = run(PRESS, spec([("Japan", 42, "42%")]))
    assert codes(r, WARNING) == ["label_not_near"]


def test_line_needs_two_points_and_gaps_break_the_line():
    r = run(QUARTERS, spec([("Q1 2025", 1.2, "$1.2 million")], kind="line", unit="USD", scale="million"))
    assert r.checked.verified.kind == "bar" and "line_refused" in codes(r, CHANGED)
    r = run(QUARTERS, spec([("Q1 2025", 1.2, "$1.2 million"), ("Q2 2025", 1.6, "$1.5 million"),
                            ("Q3 2025", 1.9, "$1.9 million"), ("Q4 2025", 2.3, "$2.3 million")], kind="line",
                           unit="USD", scale="million"))
    v = r.checked.verified
    assert v.kind == "line" and v.series[0].points[1] is None
    assert r.output.count("<polyline") == 1 and "Q2 2025" in r.output    # only Q3–Q4 are joined; Q1 stands alone


# -------------------------------------------------------------------------------------------- blocked and invalid
def test_nothing_verified_invalid_proposal_and_failing_proposer():
    r = run(PRESS, spec([("Mars", 99, "99%")]))
    assert not r.ok and codes(r) == ["quote_outside", "nothing_verified"] and "NOT rendered" in r.report()
    r = run(PRESS, {"kind": "donut", "series": []})
    assert not r.ok and codes(r, BLOCKED) == ["invalid_proposal"]

    def boom(source, question):
        raise RuntimeError("model offline")
    r = ChartSpecialist(boom).run(PRESS)
    assert not r.ok and codes(r) == ["proposer_failed"] and "model offline" in r.report()
    assert ChartSpecialist().replay(r.to_dict(), PRESS).ok


# --------------------------------------------------------------------------------------- the trace and the replay
def test_trace_replay_identical_bytes_and_tamper_detection():
    r = chart(PRESS, "revenue by region")
    sp = ChartSpecialist()
    rec = json.loads(json.dumps(r.to_dict()))                         # stored as JSON and read back
    assert not Trace(rec["trace"]).verify()
    assert sp.replay(rec, PRESS).ok and sp.replay(r).ok
    again = sp.run(PRESS, proposal=rec["proposal"])
    assert again.output == r.output                                    # the same bytes
    assert sp.replay(rec, PRESS + " ").problems == ["the source is not the one the run read (its sha256 differs)"]
    bad = json.loads(json.dumps(rec))
    bad["trace"][1]["data"]["spec"]["series"][0]["points"][0]["value"] = "52"
    probs = sp.replay(bad, PRESS).problems
    assert any("its hash does not match" in p for p in probs)
    bad = json.loads(json.dumps(rec))
    bad["output_sha256"] = "0" * 64
    assert "the re-rendered output differs from the recorded one" in sp.replay(bad, PRESS).problems
    assert not sp.replay(rec).ok                                       # no source
    assert sp.replay(r.to_dict(with_source=True)).ok


def test_rendering_is_deterministic():
    a = chart(QUARTERS, "quarterly revenue").output
    b = chart(QUARTERS, "quarterly revenue").output
    assert a == b


# ----------------------------------------------------------------------------------------------- the layout rules
def _assert_layout(drawing):
    ET.fromstring(drawing.svg)                                         # well-formed
    root = ET.fromstring(drawing.svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert root.get("role") == "img" and root.find(ns + "title") is not None and root.find(ns + "desc") is not None
    assert min(drawing.fonts) >= MIN_FONT
    for b in drawing.texts:
        assert 0 <= b.x0 and b.x1 <= drawing.width and 0 <= b.y0 and b.y1 <= drawing.height, b
    for a, b in combinations(drawing.texts, 2):
        assert not a.hits(b, pad=0), (a, b)


def test_colours_meet_contrast():
    assert all(contrast(c, BG) >= 3 for c in PALETTE)
    assert contrast(INK, BG) >= 4.5 and contrast(MUTED, BG) >= 4.5


@pytest.mark.parametrize("text,question", [(PRESS, "revenue by region"), (QUARTERS, "quarterly revenue"), (WASTE, None),
                                           (RU, None), (SURVEY, None)])
def test_layout_no_overlaps(text, question):
    r = chart(text, question)
    _assert_layout(ChartSpecialist().draw(r.checked))


def test_layout_long_labels_go_horizontal_and_wrap():
    labels = ["Municipal solid waste collected from households", "Construction and demolition debris",
              "Commercial and industrial waste streams", "Hazardous waste (batteries, paints, solvents)"]
    vals = ["1,200", "800", "2,500", "40"]
    t = " ".join(f"{lab}: {v} tonnes." for lab, v in zip(labels, vals))
    r = run(t, spec([(lab, v.replace(",", ""), v) for lab, v in zip(labels, vals)], unit="tonnes",
                    title="Waste by stream in 2025 — a long title that has to wrap onto a second line of the chart"))
    assert r.ok and not codes(r, DROPPED)
    d = ChartSpecialist().draw(r.checked)
    _assert_layout(d)
    assert 'text-anchor="end"' in d.svg                                # the horizontal layout (labels right-aligned)


def test_layout_crowded_line_and_many_slices():
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    vals = ["1,204", "1,210", "1,199", "1,207", "1,212", "1,208", "1,215", "1,209", "1,221", "1,218", "1,230", "1,226"]
    t = "Active users by month: " + ", ".join(f"{m} {v} users" for m, v in zip(months, vals)) + "."
    r = run(t, spec([(m, v.replace(",", ""), v + " users") for m, v in zip(months, vals)], kind="line", unit="users"))
    assert r.ok and r.checked.verified.kind == "line" and not codes(r, DROPPED)
    _assert_layout(ChartSpecialist().draw(r.checked))
    shares = [("Alpha", 31), ("Beta", 22), ("Gamma", 14), ("Delta", 11), ("Epsilon", 9), ("Zeta", 6), ("Eta", 4),
              ("Theta", 3)]
    t = "Market share: " + ", ".join(f"{n} {v}%" for n, v in shares) + "."
    r = run(t, spec([(n, v, f"{v}%") for n, v in shares], kind="pie"))
    assert r.checked.verified.kind == "pie"
    _assert_layout(ChartSpecialist().draw(r.checked))


def test_negative_values_bars():
    t = "Net income: 2023 $120 million, 2024 -$45 million, 2025 $80 million."
    t = t.replace("-$45", "−45")    # a minus sign before the number, no currency
    r = run(t, spec([("2023", 120, "$120 million"), ("2024", -45, "−45 million"), ("2025", 80, "$80 million")],
                    unit="USD", scale="million"))
    # "−45 million" has no currency in the text: the unit does not verify
    assert codes(r, DROPPED) == ["unit_mismatch"]
    t2 = "Net income: 2023 $120 million, 2024 $−45 million, 2025 $80 million."
    r = run(t2, spec([("2023", 120, "$120 million"), ("2024", -45, "$−45 million"), ("2025", 80, "$80 million")],
                     unit="USD", scale="million"))
    assert r.ok and not codes(r, DROPPED)
    _assert_layout(ChartSpecialist().draw(r.checked))


def test_render_svg_directly():
    r = chart(PRESS, "revenue by region")
    d = render_svg(r.checked.verified)
    assert d.svg == r.output and "3 values quoted from the source" in d.svg


# ------------------------------------------------------------------------------------------------ the LLM proposer
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_llm_proposer_through_a_fake_server():
    seen = {}
    reply = {"kind": "pie", "title": "Revenue by region", "unit": "%", "series": [{"points": [
        {"label": "Europe", "value": 42, "quote": {"text": "Europe accounted for 42%"}},
        {"label": "North America", "value": 35, "quote": {"text": "North America for 35%"}},
        {"label": "Asia-Pacific", "value": 25, "quote": {"text": "Asia-Pacific for 23%"}}]}]}   # a made-up 25

    def opener(req, timeout=None):
        seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _Resp(json.dumps({"choices": [{"message": {"content": "```json\n" + json.dumps(reply) + "\n```"}}]}).encode())

    p = LLMProposer("http://127.0.0.1:9/v1", "tiny", api_key="sk-secret", opener=opener)
    r = ChartSpecialist(p).run(PRESS, "revenue by region")
    assert seen["url"] == "http://127.0.0.1:9/v1/chat/completions" and seen["auth"] == "Bearer sk-secret"
    assert "revenue by region" in seen["body"]["messages"][1]["content"]
    assert codes(r, DROPPED) == ["value_mismatch"] and r.checked.verified.kind == "bar"
    assert "sk-secret" not in json.dumps(r.to_dict()) and r.trace.step("propose")["proposer"].startswith("llm:tiny@")
    with pytest.raises(ValueError):
        LLMProposer("file:///etc/passwd", "x")


def test_example_21(tmp_path, capsys):
    from examples_loader import load
    r1, r2, r3 = load("21_verified_chart").main(tmp_path)
    out = capsys.readouterr().out
    assert r1.checked.verified.kind == "pie" and r2.checked.verified.kind == "line"
    assert codes(r3, DROPPED) == ["value_mismatch", "quote_outside", "unit_mismatch", "no_quote"]
    assert "replay: True" in out and "replay of an edited record: False" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["21_careless_model.svg", "21_quarters_line.svg",
                                                          "21_region_pie.svg"]
    assert (tmp_path / "21_careless_model.svg").read_text() == r3.output


# ------------------------------------------------------------------------------------------------ years next to amounts
YEARS = "Net income: 2023 $120 million, 2024 -$45 million, 2025 $80 million."


def test_a_currency_sign_before_the_next_number_is_not_the_unit_of_the_year_before_it():
    from solvi.charts.check import SourceIndex
    got = [(r.as_written, str(r.value), r.unit) for r in SourceIndex(YEARS).readings]
    assert got == [("2023", "2023", ""), ("$120 million", "120000000", "USD"), ("2024", "2024", ""),
                   ("-$45 million", "-45000000", "USD"), ("2025", "2025", ""), ("$80 million.", "80000000", "USD")]
    assert [r.words for r in SourceIndex(YEARS).readings if r.unit == ""] == [(), (), ()]
    assert SourceIndex("Paid 1 500 000 $ in 2024.").readings[0].unit == "USD"        # a sign after the number: still read


def test_a_year_is_not_verified_as_a_currency_amount():
    r = run(YEARS, spec([("Net income", 2023, "2023 $"), ("2025", 2025, "2025 $")], unit="USD"))
    assert not r.ok and codes(r, DROPPED) == ["unit_mismatch", "unit_mismatch"]


def test_the_rule_proposer_charts_the_amounts_by_year_and_a_negative_amount_with_its_sign():
    r = chart(YEARS)
    v = r.checked.verified
    assert r.ok and not r.issues and v.unit == "USD" and v.scale == "million"
    assert [(p.label, str(p.value), p.as_written) for p in v.series[0].points] == [
        ("2023", "120", "$120 million"), ("2024", "-45", "-$45 million"), ("2025", "80", "$80 million.")]
    r = run(YEARS, spec([("2024", -45, "-$45 million")], unit="USD", scale="million"))
    assert r.ok and not codes(r, DROPPED)
    r = run(YEARS, spec([("2024", 45, "$45 million")], unit="USD", scale="million"))    # the sign is part of the number
    assert codes(r, DROPPED) == ["value_mismatch"]


def test_a_number_that_shares_characters_with_a_drawn_one_is_not_drawn_again():
    from decimal import Decimal
    from solvi.charts.check import ChartChecker, Reading, SourceIndex
    from solvi.charts.spec import ChartSpec
    idx = SourceIndex("a 10 $20 b")
    idx.readings[0] = Reading(2, 6, "10 $", Decimal(10), "USD", ())                     # as the checker once read it
    pt = ChartSpec.model_validate(spec([("a", 10, "10 $")], unit="USD")).series[0].points[0]
    vp, why = ChartChecker()._verify_at(pt, idx, 2, 6, "USD", Decimal(1), {(5, 8): "series[0].points[0] ('b')"})
    assert vp is None and why[0] == "quote_reused"


def test_replay_compares_the_records_own_issues_and_output_hash_with_the_trace():
    """A stored record whose `issues` were emptied replayed ok; for a blocked run nothing but the chain was checked."""
    t = "Revenue by region: Europe 42%, North America 35%, Asia 30%."
    sp = ChartSpecialist(FixedProposer(spec([("Europe", 42, "42%"), ("Asia", 23, "30%")])))
    r = sp.run(t)
    rec = r.to_dict()
    assert rec["issues"] and sp.replay(rec, t).ok
    for edit in ({"issues": []}, {"issues": rec["issues"][:-1] + [dict(rec["issues"][-1], message="fine")]},
                 {"output_sha256": "0" * 64}):
        got = sp.replay({**rec, **edit}, t)
        assert not got.ok and any("the record's" in p for p in got.problems), edit
    blocked = ChartSpecialist(FixedProposer("not json")).run(t)
    rec = blocked.to_dict()
    assert not blocked.ok and ChartSpecialist().replay(rec, t).ok
    assert not ChartSpecialist().replay({**rec, "issues": []}, t).ok


def test_a_quote_without_a_start_must_verify_at_every_place_it_occurs():
    """"Europe grew 42 percent. Asia shipped 42 units.": the point labelled Asia, in %, verified at '42 percent'."""
    t = "Europe grew 42 percent. Asia shipped 42 units."
    r = run(t, spec([("Asia", 42, "42")]))
    assert not r.ok and codes(r, DROPPED) == ["unit_mismatch"] and "occurs 2 times: give its start" in r.issues[0].message
    r = run(t, spec([("Europe", 42, {"text": "42", "start": t.index("42")})]))          # with its start: that place
    assert r.ok and not codes(r, DROPPED)
    t2 = "Europe 42%, Asia 42%, Africa 16%."
    r = run(t2, spec([("Europe", 42, "42%"), ("Asia", 42, "42%"), ("Africa", 16, "16%")], kind="pie"))
    assert r.ok and not codes(r, DROPPED)                                              # two places, both verify: one each
    assert [p.start for p in r.checked.verified.series[0].points] == [7, 17, 29]
