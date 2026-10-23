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


# --- non-string keys: stringified before sorting (the fix of 1.4.0)
def _old_cj(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


FIXTURES = Path(__file__).parent / "fixtures"
STORES = sorted(p for p in FIXTURES.rglob("*") if p.suffix in (".jsonl", ".db") and p.parent.name != "canonical_json")


def test_non_string_keys_are_written_as_their_json_strings_before_sorting():
    assert _cj({2: "a", 10: "b"}) == '{"10":"b","2":"a"}' == _cj(json.loads(json.dumps({2: "a", 10: "b"})))
    assert _cj({True: 1, None: 2, 1.5: 3, -1: 4, "s": [{3: 0}]}) == '{"-1":4,"1.5":3,"null":2,"s":[{"3":0}],"true":1}'
    assert signature._cj({2: "a", 10: "b"}) == '{"10":"b","2":"a"}'
    d = {"k": [1, {"x": 2}], "e": {}}                    # string keys only: the same object back, no copy
    from solvi.core.store import _jkeys
    assert _jkeys(d) is d


def test_list_tuple_and_dict_subclasses_are_walked_as_json_writes_them():
    """json.dumps writes a list / tuple subclass as an array and a dict subclass as an object: their dicts' keys are
    turned into JSON strings before sorting too, and string-keyed ones come back as they were."""
    import collections

    from solvi.core.store import _jkeys

    class L(list):
        pass

    class T(tuple):
        pass

    class D(dict):
        pass
    plain = '{"a":[{"10":"b","2":"a"}]}'
    for wrap in (L, T, list, tuple):
        assert _cj({"a": wrap([{2: "a", 10: "b"}])}) == plain
        assert _cj({"a": wrap([D({2: "a", 10: "b"})])}) == plain
        assert _cj({"a": wrap([collections.OrderedDict([(2, "a"), (10, "b")])])}) == plain
        same = wrap([{"x": 1}])
        assert _jkeys(same) is same and json.dumps(same) == '[{"x": 1}]'


@pytest.mark.parametrize("d", [{1: "a", "1": "b"}, {1.0: "a", "1.0": "b"}, {None: 0, "null": 1}, {"x": {True: 0, "true": 1}}])
def test_two_keys_that_become_one_json_key_are_refused(d):
    with pytest.raises(ValueError, match="two dict keys"):
        _cj(d)


def test_string_keyed_data_gives_the_same_bytes_as_before_on_every_stored_fixture():
    """Every record in every store under tests/fixtures canonicalizes to the bytes the old _cj gave: data read from JSON
    has string keys only, so no stored hash moves."""
    import sqlite3
    n = 0
    for p in STORES:
        if p.suffix == ".jsonl":
            rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            with sqlite3.connect(p) as c:
                rows = [r[0] for r in c.execute("SELECT body FROM records")]
        for ln in rows:
            rec = json.loads(ln)
            assert _cj(rec) == _old_cj(rec)
            assert _cj(record_body(rec)) == _old_cj(record_body(rec))
            n += 1
    assert n > 50 and len(STORES) >= 9


@pytest.mark.parametrize("path", STORES, ids=[str(p.relative_to(FIXTURES)) for p in STORES])
def test_every_fixture_store_verifies_as_it_did_with_the_old_canonical_json(path, tmp_path, monkeypatch):
    import shutil

    from solvi.core import store as S
    for f in path.parent.iterdir():
        if f.name.startswith(path.name):
            shutil.copy(f, tmp_path / f.name)
    new = S.open_storage(str(tmp_path / path.name)).verify()
    monkeypatch.setattr(S, "_cj", _old_cj)
    monkeypatch.setattr(signature, "_cj", lambda o: json.dumps(o, ensure_ascii=False, sort_keys=True,
                                                               separators=(",", ":"), default=repr))
    old = S.open_storage(str(tmp_path / path.name)).verify()
    assert new == old


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_a_body_with_int_keys_verifies_after_it_is_stored(backend, tmp_path):
    from solvi.core.store import FORMAT, JSONLStorage, SQLiteStorage
    st = JSONLStorage(str(tmp_path / "a.jsonl")) if backend == "jsonl" else SQLiteStorage(str(tmp_path / "a.db"))
    rec = st._append({"v": FORMAT, "kind": "x", "m": {2: "a", 10: "b"}})
    st._append({"v": FORMAT, "kind": "x", "m": {"z": 1}})
    v = st.verify()
    assert v["ok"] and v["problems"] == [] and v["count"] == 2
    assert record_hash(rec) == rec["hash"]


def test_a_body_with_two_keys_for_one_json_key_is_not_stored(tmp_path):
    from solvi.core.store import FORMAT, JSONLStorage
    st = JSONLStorage(str(tmp_path / "a.jsonl"))
    with pytest.raises(ValueError):
        st._append({"v": FORMAT, "kind": "x", "m": {1: "a", "1": "b"}})
    st._append({"v": FORMAT, "kind": "x"})
    assert st.verify()["ok"] and st.verify()["count"] == 1
