import copy
import json
import math
import pytest
from solvi.stat import agreement_diagnostics


def fixture():
    return [
        dict(gold="T", primary=p, checkers={"a": a, "b": b})
        for p, a, b in [
            ("T", "T", "T"),
            ("T", "T", "X"),
            ("T", "X", "T"),
            ("T", "X", "X"),
            ("X", "X", "X"),
            ("X", "X", "X"),
            ("X", "T", "T"),
            ("X", "Y", "Z"),
        ]
    ]


def test_hand_counted_eight_rows():
    rows = fixture()
    before = copy.deepcopy(rows)
    r = agreement_diagnostics(rows, checkers=("a", "b"))
    assert rows == before
    assert r["classes"] == {"primary_correct": 4, "primary_wrong": 4}
    for c in ("a", "b"):
        for cls in ("primary_correct", "primary_wrong"):
            assert r["checkers"][c][cls] == dict(
                n_class=4, n_observed=4, n_missing=0, n_agree=2, n_disagree=2, p_agree=0.5
            )
        assert r["checkers"][c]["same_wrong_with_primary"] == dict(
            n_observed=8, n_both_wrong=3, n_same_wrong=2, p_same_wrong=2 / 3
        )
    p = r["pairs"][0]
    assert (p["a"], p["b"]) == ("a", "b")
    assert p["primary_correct"]["cells"] == dict(n00=1, n01=1, n10=1, n11=1)
    assert p["primary_correct"]["phi"] == 0
    assert p["primary_wrong"]["cells"] == dict(n00=2, n01=0, n10=0, n11=2)
    assert p["primary_wrong"]["phi"] == 1
    assert p["same_wrong"] == dict(n_observed=8, n_both_wrong=4, n_same_wrong=3, p_same_wrong=0.75)
    json.dumps(r, allow_nan=False)


def test_missing_uses_checker_specific_and_pairwise_denominators():
    rows = fixture() + [
        dict(gold="T", primary="T", checkers={"b": "T"}),
        dict(gold=None, primary="X", checkers={"a": "X", "b": "X"}),
        dict(gold="T", primary=None, checkers={"a": "X", "b": "X"}),
    ]
    r = agreement_diagnostics(rows, checkers=("a", "b"))
    assert r["input"] == dict(n_rows=11, n_eligible=9, missing_gold=1, missing_primary=1, missing_both=0)
    assert r["checkers"]["a"]["primary_correct"]["p_agree"] == 0.5
    assert r["checkers"]["a"]["primary_correct"]["n_missing"] == 1
    assert r["checkers"]["b"]["primary_correct"]["p_agree"] == 3 / 5
    p = r["pairs"][0]["primary_correct"]
    assert p["n_observed"] == 4 and p["missing_a"] == 1 and p["phi"] == 0
    assert r["pairs"][0]["same_wrong"]["p_same_wrong"] == 0.75


def test_negative_phi_and_pair_reversal():
    rows = [
        dict(gold="T", primary="X", checkers={"a": "X", "b": "Y"}),
        dict(gold="T", primary="X", checkers={"a": "Y", "b": "X"}),
    ]
    r = agreement_diagnostics(rows, checkers=("a", "b"))["pairs"][0]["primary_wrong"]
    assert r["cells"] == dict(n00=0, n01=1, n10=1, n11=0) and r["phi"] == -1
    rev = agreement_diagnostics(rows, checkers=("b", "a"))["pairs"][0]["primary_wrong"]
    assert rev["phi"] == r["phi"]


def test_constant_marginal_empty_class_and_no_joint_errors():
    rows = [
        dict(gold="T", primary="T", checkers={"a": "T", "b": "T"}),
        dict(gold="T", primary="T", checkers={"a": "T", "b": "X"}),
    ]
    r = agreement_diagnostics(rows, checkers=("a", "b"))
    p = r["pairs"][0]
    assert p["primary_correct"]["phi"] is None and p["primary_correct"]["phi_reason"] == "constant_agreement"
    assert p["primary_wrong"]["phi"] is None and p["primary_wrong"]["phi_reason"] == "no_pairs"
    assert p["same_wrong"]["p_same_wrong"] is None
    assert r["checkers"]["a"]["primary_wrong"]["p_agree"] is None


def test_all_missing_empty_and_overlap_counts():
    r = agreement_diagnostics(
        [dict(gold=None, primary=None, checkers={}), dict(gold="T", primary="T", checkers={})], checkers=("a", "b")
    )
    assert r["input"] == dict(n_rows=2, n_eligible=1, missing_gold=1, missing_primary=1, missing_both=1)
    p = r["pairs"][0]["primary_correct"]
    assert p["missing_a"] == p["missing_b"] == p["missing_both"] == 1 and p["n_observed"] == 0
    assert r["checkers"]["a"]["primary_correct"]["p_agree"] is None
    empty = agreement_diagnostics([], checkers=("a", "b"))
    assert empty["input"]["n_rows"] == 0
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize("checkers", [("a", "a"), ("a", 1), ("a", ""), "a", None, {"a", "b"}])
def test_bad_checker_schema(checkers):
    with pytest.raises(ValueError):
        agreement_diagnostics([], checkers=checkers)


@pytest.mark.parametrize(
    "row",
    [
        {},
        dict(gold="T", primary="T", checkers={}, extra=1),
        dict(gold="T", primary="T", checkers={"unexpected": "T"}),
        dict(gold="T", primary="T", checkers=[]),
        dict(gold=1, primary="T", checkers={}),
        dict(gold="T", primary=True, checkers={}),
        dict(gold="T", primary="T", checkers={"a": math.nan}),
    ],
)
def test_bad_row_schema_is_not_hidden_by_missingness(row):
    with pytest.raises(ValueError):
        agreement_diagnostics([row], checkers=("a",))


def test_exact_caller_tokens_unicode_names_and_deterministic_generator():
    rows = [dict(gold="1", primary="1.0", checkers={"a-b": "1", "проверка": ""})]
    r = agreement_diagnostics(iter(rows), checkers=("a-b", "проверка"))
    assert r == agreement_diagnostics(rows, checkers=("a-b", "проверка"))
    assert r["classes"]["primary_wrong"] == 1
    assert r["checkers"]["a-b"]["primary_wrong"]["n_disagree"] == 1
    assert r["checkers"]["проверка"]["primary_wrong"]["n_observed"] == 1


def test_asymmetric_pair_cells_reverse_but_phi_does_not():
    rows = [dict(gold="T", primary="T", checkers={"a": a, "b": b}) for a, b in [("T", "T"), ("T", "X"), ("X", "X")]]
    forward = agreement_diagnostics(rows, checkers=("a", "b"))["pairs"][0]["primary_correct"]
    reverse = agreement_diagnostics(rows, checkers=("b", "a"))["pairs"][0]["primary_correct"]
    assert forward["cells"] == dict(n00=1, n01=0, n10=1, n11=1)
    assert reverse["cells"] == dict(n00=1, n01=1, n10=0, n11=1)
    assert forward["phi"] == reverse["phi"] == 0.5
