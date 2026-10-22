"""Writes vectors.json: test vectors for the canonical JSON a record hash is taken over (docs/trace_spec.md §3,
"Canonical form"). Run from the repository root: PYTHONPATH=src python tests/fixtures/canonical_json/make.py

Each vector holds its input as JSON text (`input`; read it with a parser that keeps a number written with ".", "e"
or "E" a float and every other number an integer, as Python's json does), the canonical text solvi gives it
(`canonical`) and the SHA-256 of that text's UTF-8 bytes (`sha256`). Kind "value": the canonical JSON of the input.
Kind "record": the input is a stored record (a line of a JSONL store); `canonical` is the canonical JSON of its hash
body (record_body) and `sha256` its record hash — the `hash` it carries. Kind "chain": `input` is two consecutive
stored records; the second one's `prev` is the first one's hash. Floats are written in `input` in another spelling
than the canonical one where they can be (1E16, 0.000001, 100.00) so that a port has to reformat them; `source`
says which float each was."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from solvi.core.store import _cj, record_body, record_hash

HERE = Path(__file__).parent
FIX = HERE.parent

VALUES = [
    ("float 0.1", '{"x":0.1}', "0.1"),
    ("float 1e-06 (Python switches to an exponent below 1e-4)", '{"x":0.000001}', "1e-06"),
    ("float 0.0001 (still fixed)", '{"x":1.0E-4}', "0.0001"),
    ("float 1e+16 (Python switches to an exponent from 1e16)", '{"x":1E16}', "1e+16"),
    ("float 1e15 (still fixed, with .0)", '{"x":1e15}', "1e15"),
    ("integer 10**16 (an integer stays an integer)", '{"x":10000000000000000}', None),
    ("float 5e-324 (smallest subnormal)", '{"x":4.9406564584124654e-324}', "5e-324"),
    ("float max", '{"x":1.7976931348623157E+308}', "1.7976931348623157e308"),
    ("float -0.0", '{"x":-0.0}', "-0.0"),
    ("float 100.0 (integral float keeps .0)", '{"x":100.00}', "100.0"),
    ("float 0.30000000000000004", '{"x":0.30000000000000004}', "0.1 + 0.2"),
    ("float 1.2345678901234568e+17 (shortest round-trip digits)", '{"x":123456789012345678.0}', "123456789012345678.0"),
    ("integer 1 and float 1.0 differ", '{"f":1.0,"i":1}', None),
    ("large integers (no 2**53 limit)",
     '{"a":18446744073709551616,"b":-9223372036854775808,"c":9007199254740993,"d":1000000000000000000000000000000}', None),
    ("booleans, null, empty containers", '{"e":{},"l":[],"n":null,"s":"","t":true,"u":false}', None),
    ("nested objects and arrays, keys sorted at every level",
     '{"z":[{"b":2,"a":1},[3,[]]],"a":{"y":{"d":null,"c":[true]},"x":0}}', None),
    ("non-ASCII written as is (BMP)", '{"k":"ünïcode ✓ — Привет 中文"}', None),
    ("key order: code points, not UTF-16 (astral after U+FFFF)",
     '{"\\ud83d\\ude00":1,"\\uffff":2,"\\ue000":3,"a":4,"B":5,"\\u00e9":6}', None),
    ("U+2028 and U+2029 are not escaped", '{"\\u2028":"a\\u2028b\\u2029c"}', None),
    ("escapes: quote, backslash, controls; / and DEL are not escaped",
     '{"s":"\\"\\\\/\\u0000\\u0001\\b\\t\\n\\u000b\\f\\r\\u001f\\u007f"}', None),
    ("keys that look like numbers sort as strings (a Python int key is turned into its string first)",
     '{"2":"a","10":"b","-1":"c","1.5":"d"}', None),
    ("legacy (records before 0.7): NaN and Infinity tokens", '{"a":NaN,"b":Infinity,"c":-Infinity}', None),
]


def vec(name, text, source, kind="value"):
    obj = json.loads(text)
    canon = _cj(obj)
    v = {"name": name, "kind": kind, "input": text}
    if source is not None:
        v["source"] = source
    v["canonical"] = canon
    v["sha256"] = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return v


def record_vec(name, line):
    rec = json.loads(line)
    canon = _cj(record_body(rec))
    h = record_hash(rec)
    assert h == rec["hash"] == hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return {"name": name, "kind": "record", "input": line, "canonical": canon, "sha256": h}


def main():
    out = [vec(*v) for v in VALUES]
    l1 = (FIX / "store_0_7_1" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    l2 = (FIX / "store_1_0_0" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    out.append(record_vec("a format 1 record (solvi 0.7.1): the record without id and hash", l1[0]))
    out.append(record_vec("a format 2 record (solvi 1.0.0, compact): lasting fields, #answers, #content", l2[0]))
    a, b = json.loads(l2[0]), json.loads(l2[1])
    assert b["prev"] == a["hash"]
    out.append({"name": "chain link: the second record's prev is the first one's hash", "kind": "chain",
                "input": [l2[0], l2[1]], "canonical": _cj(record_body(b)), "sha256": b["hash"]})
    doc = {"about": __doc__.split("\n\n", 1)[1].replace("\n", " "), "spec": "docs/trace_spec.md", "vectors": out}
    (HERE / "vectors.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
