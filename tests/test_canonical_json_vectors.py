"""The canonical JSON a record hash is taken over (docs/trace_spec.md §3, "Canonical form"): every vector in
tests/fixtures/canonical_json/vectors.json recomputes with the library's own functions, and the edge cases the spec
states (float spelling, key order, escapes, non-finite floats, non-string keys) hold as stated."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from solvi.core.store import _cj, _digest, record_body, record_hash
from solvi.core.store import signature

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "canonical_json" / "vectors.json").read_text(encoding="utf-8"))
CASES = VECTORS["vectors"]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_vectors_cover_values_records_and_a_chain():
    kinds = [v["kind"] for v in CASES]
    assert kinds.count("value") >= 20 and kinds.count("record") == 2 and kinds.count("chain") == 1
    assert len({v["name"] for v in CASES}) == len(CASES)


@pytest.mark.parametrize("v", CASES, ids=[v["name"] for v in CASES])
def test_a_vector_recomputes(v):
    assert sha(v["canonical"]) == v["sha256"]
    if v["kind"] == "value":
        obj = json.loads(v["input"])
        assert _cj(obj) == v["canonical"]
        assert _digest(obj) == v["sha256"]
        assert signature._cj(obj) == v["canonical"]          # default=repr is not reached by JSON data
    elif v["kind"] == "record":
        rec = json.loads(v["input"])
        assert _cj(record_body(rec)) == v["canonical"]
        assert record_hash(rec) == v["sha256"] == rec["hash"]
    else:
        a, b = (json.loads(x) for x in v["input"])
        assert b["prev"] == record_hash(a) == a["hash"]
        assert _cj(record_body(b)) == v["canonical"] and record_hash(b) == v["sha256"] == b["hash"]


@pytest.mark.parametrize("x, text", [
    (0.1, "0.1"), (1e-4, "0.0001"), (1e-5, "1e-05"), (1e-6, "1e-06"), (2.5e-5, "2.5e-05"), (1e15, "1000000000000000.0"),
    (9999999999999998.0, "9999999999999998.0"), (1e16, "1e+16"), (1e21, "1e+21"), (1e22, "1e+22"),
    (123456789012345678.0, "1.2345678901234568e+17"), (5e-324, "5e-324"), (-1e-320, "-1e-320"),
    (1.7976931348623157e308, "1.7976931348623157e+308"), (-0.0, "-0.0"), (100.0, "100.0"), (0.1 + 0.2, "0.30000000000000004"),
    (1, "1"), (1.0, "1.0"), (10 ** 16, "10000000000000000"), (2 ** 64, "18446744073709551616"),
])
def test_number_spelling(x, text):
    assert _cj(x) == text


def test_non_finite_floats_are_tokens_in_the_canonical_form_and_refused_by_the_writer():
    from solvi.core.store import _body
    assert _cj([float("nan"), float("inf"), -float("inf")]) == "[NaN,Infinity,-Infinity]"
    with pytest.raises(ValueError):
        _body({"x": float("nan")})                       # a record holding one is never written


def test_strings_key_order_and_escapes():
    assert _cj({"￿": 0, "\U0001F600": 0}) == '{"￿":0,"\U0001F600":0}'     # code points; UTF-16 differs
    assert _cj("  ") == '"  "'
    assert _cj("\"\\/\x00\x08\x09\x0a\x0b\x0c\x0d\x1f\x7f") == '"\\"\\\\/\\u0000\\b\\t\\n\\u000b\\f\\r\\u001f\x7f"'
    with pytest.raises(UnicodeEncodeError):
        _digest("\ud800")                                # a lone surrogate has no UTF-8 bytes: refused


def test_non_string_keys_sort_before_they_become_strings():
    """Known (docs/trace_spec.md §3, "Canonical form"): a dict with int keys sorts them as numbers, the same dict read
    back from JSON as strings — a record appended with such a body does not verify. Pinned so a fix is a decision."""
    d = {2: "a", 10: "b"}
    assert _cj(d) == '{"2":"a","10":"b"}'
    assert _cj(json.loads(json.dumps(d))) == '{"10":"b","2":"a"}'
