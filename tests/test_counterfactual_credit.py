"""The logic of benchmarks/counterfactual_credit.py (the counterfactual credit measurement): the brute-force crossing over an integer
domain, the validity re-check of an explanation, the minimality cases, the population and its hash order, the
sensitive-field assertion and the no-explanation diagnosis — on hand-made applications, no data needed."""
import hashlib
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
warnings.simplefilter("ignore")
import counterfactual_credit as cfc  # noqa: E402

NEUTRAL = {"id": "t1", "checking_account": "200 DM or more, or salary assigned for a year", "duration_months": 12,
           "credit_history": "existing credits paid back duly till now", "purpose": "radio/television",
           "credit_amount_dm": 2000, "savings": "unknown or no savings account", "employed_since": "7 years or more",
           "other_debtors": "none", "property": "real estate", "other_installment_plans": "none"}
DOMAINS = {"credit_amount_dm": (250, 18424), "duration_months": (4, 72),
           "checking_account": ["below 0 DM", "0 to 200 DM", "200 DM or more, or salary assigned for a year",
                                "no checking account"],
           "savings": ["below 100 DM", "unknown or no savings account"],
           "credit_history": ["existing credits paid back duly till now", "all credits at this bank paid back duly"],
           "purpose": ["radio/television", "car (used)", "education"],
           "employed_since": ["7 years or more", "unemployed"], "other_debtors": ["none", "guarantor"],
           "property": ["real estate", "unknown or none"], "other_installment_plans": ["none", "bank"]}


@pytest.fixture(scope="module")
def v2():
    return cfc.build_v2(sensitive=set(cfc.SENSITIVE_DEFAULT))


# ---------------------------------------------------------------- brute force over an integer domain
def test_nearest_crossing_scans_every_integer_outward():
    ok = lambda x: x <= 7                                          # noqa: E731
    assert cfc.nearest_crossing(ok, 10, 0, 100, -1) == 7
    assert cfc.nearest_crossing(ok, 10, 0, 100, +1) is None
    assert cfc.nearest_crossing(ok, 10, 8, 100, -1) is None        # the domain's bound stops it
    band = lambda x: 15 <= x <= 16                                 # noqa: E731  (a band: no monotonicity assumed)
    assert cfc.nearest_crossing(band, 10, 0, 100, +1) == 15
    assert cfc.nearest_crossing(band, 20, 0, 100, -1) == 16
    assert cfc.nearest_crossing(ok, 10, 0, 100, -1, evals=(n := [0])) == 7 and n[0] == 3   # 9, 8, 7


def test_brute_crossings_on_the_credit_catalog(v2):
    app = {**NEUTRAL, "checking_account": "0 to 200 DM", "credit_amount_dm": 10000}     # 2 + 2 = 4 → refuse
    assert cfc.decide(v2, app) == ("refuse", "ok")
    got = cfc.brute_crossings(v2, app, "credit_amount_dm", (250, 18424))
    assert got == {-1: 8000, 1: None}
    assert cfc.brute_crossings(v2, app, "duration_months", (4, 72)) == {-1: None, 1: None}


# ---------------------------------------------------------------- validity
def test_apply_changes_leaves_the_input_alone():
    app = dict(NEUTRAL)
    cf = {"changes": [{"fact": "credit_amount_dm", "to": 8000, "op": "≤"}, {"fact": "savings", "to": "x", "op": "="}]}
    new = cfc.apply_changes(app, cf)
    assert new["credit_amount_dm"] == 8000 and new["savings"] == "x" and app == NEUTRAL
    with pytest.raises(ValueError):
        cfc.apply_changes(app, {"changes": [{"fact": "credit_amount_dm", "to": 8000, "op": "<"}]})   # strict: no point


def test_validity_recheck_catches_a_wrong_explanation(v2):
    app = {**NEUTRAL, "checking_account": "0 to 200 DM", "credit_amount_dm": 10000}
    good = {"changes": [{"fact": "credit_amount_dm", "to": 8000, "op": "≤"}]}
    bad = {"changes": [{"fact": "credit_amount_dm", "to": 8001, "op": "≤"}]}
    assert cfc.validity(v2, app, good) == (True, "approve", "ok")
    assert cfc.validity(v2, app, bad) == (False, "refuse", "ok")
    strict = {"changes": [{"fact": "credit_amount_dm", "to": 8000.5, "op": "<"}]}
    assert cfc.validity(v2, app, strict)[0] is False


# ---------------------------------------------------------------- minimality cases
def test_minimality_cases():
    found = [{"changes": [{"fact": "credit_amount_dm", "to": 8000, "op": "≤", "now": 10000}]},
             {"changes": [{"fact": "duration_months", "to": 25, "op": "≥", "now": 12}]},
             {"changes": [{"fact": "credit_amount_dm", "to": 1, "op": "≤", "now": 10000},
                          {"fact": "savings", "to": "x", "op": "="}]}]              # two fields: not a case
    brute = {"credit_amount_dm": {-1: 8000, 1: None}, "duration_months": {-1: 4, 1: 27}}
    cases = cfc.minimality_cases(found, brute)
    by = {(c["field"], c["dir"]): c for c in cases}
    assert set(by) == {("credit_amount_dm", -1), ("duration_months", -1), ("duration_months", 1)}
    assert by["credit_amount_dm", -1]["ok"] and by["credit_amount_dm", -1]["found"] == 8000
    assert not by["duration_months", -1]["ok"] and by["duration_months", -1]["found"] is None     # missed
    assert not by["duration_months", 1]["ok"]                                                     # off by 2
    assert cfc.minimality_cases([{"changes": [{"fact": "duration_months", "to": 26, "op": "≥"}]}],
                                {"duration_months": {-1: None, 1: 27}})[0]["ok"]                  # within 1 unit


# ---------------------------------------------------------------- population, hash order, sensitive fields
def test_population_is_refused_and_review_in_hash_order():
    dec = {"a": "approve", "b": "refuse", "c": "review", "d": "refuse"}
    pop = cfc.population(dec)
    assert set(pop) == {"b", "c", "d"}
    assert pop == sorted(pop, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    assert cfc.by_hash(["b", "c", "d"]) == pop


def test_catalog_given_inputs_and_the_sensitive_assertion(v2):
    given = cfc.given_inputs(v2)
    assert "credit_amount_dm" in given and "points" not in given and "checking_points" not in given
    cfc.assert_no_sensitive(v2, ["credit_amount_dm"], {"age"})
    young = cfc.cc.build(2, young=True)
    with pytest.raises(AssertionError, match="age"):
        cfc.assert_no_sensitive(young, ["credit_amount_dm"], {"age"})
    with pytest.raises(AssertionError, match="age"):
        cfc.assert_no_sensitive(v2, ["credit_amount_dm", "age"], {"age"})


# ---------------------------------------------------------------- the search and the diagnosis end to end
def test_explain_one_finds_the_brute_force_crossing(v2):
    app = {**NEUTRAL, "checking_account": "0 to 200 DM", "credit_amount_dm": 10000}
    res = v2.ask(app, store=False)
    rec = cfc.explain_one(v2, res, app, DOMAINS, brute=True)
    texts = [c["text"] for c in rec["mc1"]["found"]]
    assert "approve if credit_amount_dm ≤ 8000 (now 10000)" in texts
    assert all(v["ok"] for v in rec["valid"])
    assert any(c["field"] == "credit_amount_dm" and c["ok"] for c in rec["minimality"])
    assert rec["mc1"]["exhausted"] is False and rec["mc2"]["found"]


def test_diagnosis_when_no_two_field_change_exists(v2):
    app = {**NEUTRAL, "checking_account": "below 0 DM", "credit_amount_dm": 10000, "duration_months": 40,
           "savings": "below 100 DM", "credit_history": "all credits at this bank paid back duly"}   # 3+2+2+1+2 = 10
    res = v2.ask(app, store=False)
    rec = cfc.explain_one(v2, res, app, DOMAINS, brute=True)
    assert not rec["mc2"]["found"]
    d = rec["diagnosis"]
    assert d["points"] == 10 and d["approve_at_most"] == 2 and d["needed"] == 8
    assert d["reach"]["checking_account"] == 4 and d["reach"]["credit_amount_dm"] == 2
    assert d["best_two"] == 6 and "no pair" in d["why"]
