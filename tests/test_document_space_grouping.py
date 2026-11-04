from decimal import Decimal

import pytest

from solvi.solutions.document import _ground, _read_quote
from test_document_numeric_spans import recipe


@pytest.mark.parametrize("raw,value", [
    ("1 234", "1234"), ("12 345 678", "12345678"),
    ("1\u202f234,50", "1234.50"), ("-1 234.50", "-1234.50"),
])
def test_explicit_space_groups_are_read_exactly(raw, value):
    assert _read_quote(Decimal, raw, space_grouping=True)[0] == Decimal(value)


@pytest.mark.parametrize("raw", ["1 23", "1234 567", "1 234 56", "1\n234", "1\t234"])
def test_declaration_does_not_admit_incomplete_or_cross_line_groups(raw):
    with pytest.raises(ValueError):
        _read_quote(Decimal, raw, space_grouping=True)


def test_original_s313_witness_has_an_explicit_end_to_end_path(tmp_path):
    raw, doc = "1 234", "Total 1 234 units"
    assert _ground(raw, doc) == (None, {})
    assert _ground("234", doc)[0] is not None
    assert recipe(raw).ask({"doc": doc})["answer"].status == "abstain"
    s = recipe(raw, space_grouping=True, storage=tmp_path / "decisions.jsonl",
               compute=lambda facts, state: "matched" if facts.total == Decimal(1234) else "wrong")
    result = s.ask({"doc": doc})
    assert result["answer"].answer == "matched"
    assert result.values["total_value"] == Decimal(1234)
    assert result.facts[0].quote == raw
    assert s.replay_all() == []


def test_space_grouping_changes_catalog_identity_and_rejects_non_bool():
    assert recipe("1 234").system.fingerprint() != recipe("1 234", space_grouping=True).system.fingerprint()
    with pytest.raises(TypeError, match="space_grouping must be a bool"):
        recipe("1 234", space_grouping="yes")
