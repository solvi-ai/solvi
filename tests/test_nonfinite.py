"""Non-finite floats (an escalation threshold no calibration could meet is inf) in traces, stored records and JSON outputs:
written as {"$float": "inf"} (strict JSON, no Infinity / NaN), read back as the float, with the trace's hashes unchanged."""
import json
import math

import pytest

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.runtime import vhash
from solvi.core.schema import dumps, tag_floats, untag_floats
from solvi.core.store import JSONLStorage, SQLiteStorage, record_hash
from solvi.core.system import Response


class M:
    model_id, version = "demo/m", "1"


def system():
    cat = Catalog()

    @cat.rule("team", model=M())
    def team(doc):
        return Decision("billing", {"billing": 0.9, "shipping": 0.1},
                        extra={"threshold": math.inf, "floor": -math.inf, "odd": math.nan, "ok": 0.5})
    return System(cat, [Question("team", "Team?", Answer.choice(["billing", "shipping"]))])


def test_tag_and_untag_round_trip():
    v = {"a": [1.0, math.inf, {"b": -math.inf}], "c": math.nan, "d": "x", "e": (1, 2)}
    t = tag_floats(v)
    assert t["a"][1] == {"$float": "inf"} and t["a"][2]["b"] == {"$float": "-inf"} and t["c"] == {"$float": "nan"}
    json.dumps(t, allow_nan=False)
    back = untag_floats(json.loads(json.dumps(t)))
    assert back["a"][1] == math.inf and back["a"][2]["b"] == -math.inf and math.isnan(back["c"])
    same = {"a": [1, 2.5, "x"], "b": {"c": None}}
    assert tag_floats(same) is same and untag_floats(same) is same          # nothing to tag: no copy
    with pytest.raises(ValueError):
        json.dumps({"x": math.inf}, allow_nan=False)
    assert dumps({"x": math.inf}) == '{"x": {"$float": "inf"}}'


def test_a_trace_with_an_infinite_threshold_is_strict_json_and_round_trips():
    s = system()
    res = s.ask({"doc": "charged twice"})
    rec = [r for r in res.trace.records if r.name == "answer:team"][0]
    assert rec.extra["threshold"] == math.inf
    text = res.to_json()                                                   # allow_nan=False: raises on a bare inf
    assert "Infinity" not in text and "NaN" not in text and '{"$float": "inf"}' in text
    back = Response.from_json(text, catalog=s)
    r2 = [r for r in back.trace.records if r.name == "answer:team"][0]
    assert r2.extra["threshold"] == math.inf and r2.extra["floor"] == -math.inf and math.isnan(r2.extra["odd"])
    assert vhash(r2.body()) == rec.hash                                    # the trace's hashes do not change
    assert back.trace.replay(s.catalog, back.flow, trust_models=True)["ok"]


@pytest.mark.parametrize("kind", ["jsonl", "sqlite"])
def test_stored_records_hold_no_infinity_and_verify(tmp_path, kind):
    s = system()
    st = JSONLStorage(tmp_path / "d.jsonl") if kind == "jsonl" else SQLiteStorage(tmp_path / "d.db")
    sid = st.save(s.ask({"doc": "charged twice"}))
    if kind == "jsonl":
        raw = (tmp_path / "d.jsonl").read_text()
        assert "Infinity" not in raw and "NaN" not in raw
    assert st.verify()["ok"]
    got = st.get(sid, s)
    ex = [r for r in got.trace.records if r.name == "answer:team"][0].extra
    assert ex["threshold"] == math.inf


def test_a_record_written_before_0_7_with_infinity_still_verifies_and_loads(tmp_path):
    s = system()
    new = tmp_path / "new.jsonl"
    JSONLStorage(new).save(s.ask({"doc": "charged twice"}))
    rec = untag_floats(json.loads(new.read_text()))                        # the record as 0.6 held it: bare inf / nan
    rec = {k: v for k, v in rec.items() if k not in ("id", "hash")}
    rec["hash"] = record_hash(rec)
    rec["id"] = rec["hash"][:16]
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")   # Infinity / NaN, as 0.6 wrote them
    assert "Infinity" in old.read_text()
    (tmp_path / "old.jsonl.head").write_text(json.dumps({"count": 1, "hash": rec["hash"]}))
    st = JSONLStorage(old)
    assert st.verify()["ok"]
    got = st.get(rec["id"], s)
    assert [r for r in got.trace.records if r.name == "answer:team"][0].extra["threshold"] == math.inf


def test_numpy_non_finite_values_are_tagged_too():
    import numpy as np

    from solvi.core.schema import jsonable
    d = jsonable({"a": np.float32("inf"), "b": np.float64("-inf"), "c": np.array([1.0, np.inf]), "s": {2.0, math.inf}})
    json.dumps(d, allow_nan=False)
    assert d["a"] == {"$float": "inf"} and d["b"] == {"$float": "-inf"} and d["c"] == [1.0, {"$float": "inf"}]
