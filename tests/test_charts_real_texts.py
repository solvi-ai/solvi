"""benchmarks/charts_real_texts.py (the pre-registered charts-on-real-texts stand): the script's own logic on the gallery texts of tests/test_charts.py —
the re-verification of rendered values against a text (X3-1), the matching of drawn points against a hand reading
(X3-2), the refusal-reason tally (X3-3), the proposal reduced to its verified points (X3-4) and the logged loader.
The measured set (benchmarks/datasets/charts_texts) is not read here."""
import importlib.util
import json
import sys
import warnings
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from test_charts import PRESS, QUARTERS, WASTE  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.charts import ChartSpecialist, FixedProposer, RuleProposer

_spec = importlib.util.spec_from_file_location("charts_real_texts", ROOT / "benchmarks" / "charts_real_texts.py")
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)


def rule(text):
    return ChartSpecialist(RuleProposer()).run(text)


def fixed(text, spec):
    return ChartSpecialist(FixedProposer(spec)).run(text)


def bar(points, unit="%", scale="", kind="bar", title="Chart"):
    return {"kind": kind, "title": title, "unit": unit, "scale": scale,
            "series": [{"points": [{"label": lab, "value": v, "quote": {"text": q}} for lab, v, q in points]}]}


# ------------------------------------------------------------------------------------------------ the gallery
def test_the_dry_set_is_the_three_gallery_texts_with_labels_for_each():
    assert list(X.GALLERY) == ["press", "quarters", "waste"]
    assert X.GALLERY["press"] == PRESS and X.GALLERY["quarters"] == QUARTERS and X.GALLERY["waste"] == WASTE
    for k, text in X.GALLERY.items():
        lab = X.GALLERY_LABELS[k]
        for n in lab["numbers"]:                                   # every labelled number is where it says
            assert text[n["start"]:n["end"]] == n["as_written"]
        assert lab["chart"]["series"]


# ------------------------------------------------------------------------------------------------ X3-1
@pytest.mark.parametrize("written, value", [
    ("$4.2 billion", "4200000000"), ("1,200 tonnes", "1200"), ("42%", "42"), ("12 percent", "12"),
    ("$1.9 million", "1900000"), ("eight", "8"), ("Seven", "7"), ("–$27.61 trillion", "27610000000000"),
    ("$5.86 trillion", "5860000000000"), ("9.9-percent", "9.9"), ("561,000 b/d", "561000"), ("$7,672", "7672"),
    ("3.1 billion cubic feet per day", "3100000000"), ("0.7 BkWh", "0.7")])
def test_parse_quantity_reads_a_number_as_written_on_its_own(written, value):
    assert X.parse_quantity(written) == Decimal(value)


def test_parse_quantity_refuses_what_is_not_a_number():
    assert X.parse_quantity("Europe") is None and X.parse_quantity("") is None


def test_every_rendered_value_of_the_gallery_reverifies():
    for text in (PRESS, QUARTERS, WASTE):
        run = rule(text)
        r = X.reverify(run, text)
        assert run.ok and r["values"] >= 3 and r["values_ok"] == r["values"] and r["problems"] == []
        assert r["svg_numbers"] >= r["values"] and r["svg_numbers_ok"] == r["svg_numbers"]


def test_rendered_numbers_include_the_scaled_display_and_the_total_in_the_description():
    run = rule(QUARTERS)                                  # "$1.2m" on the chart for "$1.2 million" in the text
    toks = X.svg_numbers(run.output)
    assert "$1.2m" in toks and "$2.3m" in toks
    assert not any("quoted from the source" in t for t in toks)        # the footer count is not a value
    assert X.reverify(run, QUARTERS)["svg_numbers_ok"] == len(toks)


def test_a_number_added_to_the_svg_that_is_not_in_the_text_is_caught():
    run = rule(PRESS)
    run.output = run.output.replace("</svg>", "<text x='1' y='1'>Japan: 17%</text></svg>")
    r = X.reverify(run, PRESS)
    assert r["svg_numbers_ok"] == r["svg_numbers"] - 1
    assert any("17%" in p for p in r["problems"])


def test_a_verified_value_that_is_not_the_number_at_its_place_is_caught():
    run = rule(PRESS)
    v = run.checked.verified
    p0 = v.series[0].points[0]
    v.series[0].points[0] = p0.model_copy(update={"value": Decimal("24")})      # 42% at its place, 24 claimed
    r = X.reverify(run, PRESS)
    assert r["values_ok"] == r["values"] - 1 and any("24" in p for p in r["problems"])
    run2 = rule(PRESS)
    q0 = run2.checked.verified.series[0].points[0]
    run2.checked.verified.series[0].points[0] = q0.model_copy(update={"start": q0.start + 1})
    assert X.reverify(run2, PRESS)["values_ok"] == r["values"] - 1             # the place no longer holds it


def test_a_run_without_output_has_nothing_to_reverify():
    run = rule("No numbers here at all.")
    assert not run.ok
    assert X.reverify(run, "No numbers here at all.") == {"values": 0, "values_ok": 0, "svg_numbers": 0,
                                                          "svg_numbers_ok": 0, "problems": []}


# ------------------------------------------------------------------------------------------------ X3-2
@pytest.mark.parametrize("kw, no, hay, ok", [
    ("Europe", "", "Europe", True), ("Europe", "", "europe accounted", True), ("Europe", "", "Asia-Pacific", False),
    ("urban ; 2022|from", "", "urban rate from", True), ("urban ; 2022|from", "", "urban rate", False),
    ("Arctic", "Antarctic", "Antarctic", False), ("Arctic", "Antarctic", "the Arctic", True), ("", "", "anything", True)])
def test_label_keywords_groups_alternatives_and_exclusions(kw, no, hay, ok):
    assert X.label_agrees(kw, no, hay) is ok


def test_the_gallery_points_of_the_rule_proposer_agree_with_the_hand_reading():
    for k in ("press", "quarters", "waste"):
        text, lab = X.GALLERY[k], X.GALLERY_LABELS[k]
        rows = X.match_points(rule(text), lab)
        assert rows and all(r["label_ok"] and r["unit_ok"] and r["sign_ok"] for r in rows), (k, rows)


def test_a_wrong_label_is_a_disagreement_and_the_unit_is_judged_apart():
    lab = X.GALLERY_LABELS["press"]
    run = fixed(PRESS, bar([("Asia-Pacific", 42, "42%"), ("North America", 35, "35%"), ("Europe", 23, "23%")]))
    rows = {r["label"]: r for r in X.match_points(run, lab)}
    assert not rows["Asia-Pacific"]["label_ok"] and rows["Asia-Pacific"]["unit_ok"]
    assert rows["North America"]["label_ok"] and not rows["Europe"]["label_ok"]


def test_a_unit_the_hand_reading_does_not_accept_is_a_disagreement():
    lab = X.GALLERY_LABELS["waste"]          # tonnes; drawn as plain numbers, which the checker lets through
    run = fixed(WASTE, bar([("recycled", 1200, "1,200 tonnes"), ("composted", 800, "800 tonnes")], unit=""))
    rows = X.match_points(run, lab)
    assert run.ok and len(rows) == 2 and all(r["label_ok"] and not r["unit_ok"] for r in rows)
    run = fixed(WASTE, bar([("recycled", "1.2", "1,200 tonnes"), ("composted", "0.8", "800 tonnes")],
                           unit="tonnes", scale="thousand"))
    assert all(r["unit_ok"] for r in X.match_points(run, lab))           # 1.2 thousand tonnes is 1,200 tonnes


def test_a_point_on_a_number_that_is_not_a_value_disagrees_with_its_reason():
    lab = X.GALLERY_LABELS["quarters"]
    run = fixed(QUARTERS, bar([("fiscal", 2025, "2025"), ("Q1", "1.2", "$1.2 million")], unit="", scale=""))
    rows = {r["label"]: r for r in X.match_points(run, lab)}
    assert "fiscal" in rows and not rows["fiscal"]["label_ok"] and rows["fiscal"]["reason"].startswith("not a value")


def test_the_sign_is_reported_apart():
    text = "Sales rose 5% in Europe, fell 3% in Asia and rose 2% in Africa."
    lab = {"numbers": X.locate(text, [
        {"as_written": "5%", "occurrence": 0, "value": "5", "unit": "%", "role": "value", "of": "", "label_keywords": "Europe", "label_excludes": ""},
        {"as_written": "3%", "occurrence": 0, "value": "-3", "unit": "%", "role": "value", "of": "", "label_keywords": "Asia", "label_excludes": ""},
        {"as_written": "2%", "occurrence": 0, "value": "2", "unit": "%", "role": "value", "of": "", "label_keywords": "Africa", "label_excludes": ""}]),
        "chart": {}}
    run = fixed(text, bar([("Europe", 5, "5%"), ("Asia", 3, "3%"), ("Africa", 2, "2%")]))
    rows = {r["label"]: r for r in X.match_points(run, lab)}
    assert rows["Asia"]["label_ok"] and rows["Asia"]["unit_ok"] and not rows["Asia"]["sign_ok"]
    assert rows["Europe"]["sign_ok"] and rows["Africa"]["sign_ok"]


def test_locate_finds_whole_numbers_at_their_occurrence():
    text = "In 2025 it was 3%, then 23% and 3% again; $6.2 and $6.2."
    nums = X.locate(text, [{"as_written": "3%", "occurrence": 1}, {"as_written": "$6.2", "occurrence": 1}])
    assert text[nums[0]["start"]:].startswith("3% again") and nums[1]["start"] == text.rindex("$6.2")
    with pytest.raises(ValueError):
        X.locate(text, [{"as_written": "3%", "occurrence": 2}])


# ------------------------------------------------------------------------------------------------ X3-3
def test_refusal_reasons():
    assert X.refusal_reason(rule(PRESS)) is None
    assert X.refusal_reason(rule("No numbers here at all.")) == "no_values_proposed"
    wrong = fixed(PRESS, bar([("Europe", 41, "42%"), ("Asia", 22, "23%")]))
    assert not wrong.ok and X.refusal_reason(wrong) == "value_mismatch"
    units = fixed(PRESS, bar([("Europe", 42, "42%"), ("Asia", 23, "23%")], unit="USD"))
    assert X.refusal_reason(units) == "unit_mismatch"
    assert X.refusal_reason(fixed(PRESS, {"kind": "donut"})) == "invalid_proposal"


def test_the_tally_counts_drawn_runs_and_each_refusal_reason():
    runs = [rule(PRESS), rule(QUARTERS), rule("No numbers."), rule("Nothing to chart.")]
    t = X.tally(runs)
    assert t == {"drawn": 2, "refused": 2, "reasons": {"no_values_proposed": 2}}


# ------------------------------------------------------------------------------------------------ X3-4
def test_verified_only_keeps_the_points_that_verified_and_the_total_if_it_did():
    run = fixed(QUARTERS, {"kind": "line", "title": "q", "unit": "USD", "scale": "million",
                           "series": [{"points": [{"label": "Q1 2025", "value": "1.2", "quote": {"text": "$1.2 million"}},
                                                  {"label": "Q2 2025", "value": "1.6", "quote": {"text": "$1.5 million"}},
                                                  {"label": "Q3 2025", "value": "1.9", "quote": {"text": "$1.9 million"}}]}],
                           "total": {"label": "Total", "value": "6.9", "quote": {"text": "$6.9 million"}}})
    p = X.verified_only(run)
    assert [pt["label"] for pt in p["series"][0]["points"]] == ["Q1 2025", "Q3 2025"]
    assert p["total"]["value"] == "6.9"
    again = ChartSpecialist().run(QUARTERS, proposal=p)
    assert again.ok and not again.dropped


def test_corruption_arms_run_on_the_gallery_records():
    runs = {k: rule(t) for k, t in X.GALLERY.items()}
    rep = X.corruption(runs, X.GALLERY, n=20, seed=0)
    assert set(rep) == {"recorded", "verified_only"}
    for arm in rep.values():
        assert set(arm) == {"change_digit", "swap", "drop"}
        assert arm["change_digit"]["n"] == 20 and arm["change_digit"]["caught"] == 20
        assert arm["swap"]["n"] == 20
    assert X.corruption(runs, X.GALLERY, n=20, seed=0) == rep                 # deterministic


# ------------------------------------------------------------------------------------------------ the loader
def test_the_loader_logs_every_read(tmp_path):
    p = tmp_path / "texts.jsonl"
    p.write_text(json.dumps({"id": "t01", "text": "A 5% rise."}) + "\n")
    log = tmp_path / "reads.log"
    assert X.load_texts(p, mode="test", log=log) == [{"id": "t01", "text": "A 5% rise."}]
    X.load_texts(p, mode="measured", log=log)
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and "mode=test" in lines[0] and "mode=measured" in lines[1]
    assert all("pid=" in ln and "sha256=" in ln for ln in lines)


def test_the_measured_set_needs_an_explicit_flag(tmp_path):
    with pytest.raises(SystemExit):
        X.main(["--out", str(tmp_path)])                 # neither --dry nor --run: refuses, reads nothing
