"""Real records validate against the versioned record schema (src/solvi/core/store/record.schema.json), and the schema
rejects what the format does not allow. docs/trace_spec.md §2."""
import copy

import pytest
from semantics_helpers import MemoryStorage, fill, load_schema, read_lines, rng_for, system, validate

from solvi.core.store import FORMAT, KEPT, JSONLStorage

SCHEMA = load_schema()


def _records(tmp_path, seed):
    s = system()
    rng = rng_for(seed)
    out = []
    for mode in ("full", "compact", "sample:2"):
        st = JSONLStorage(tmp_path / f"{mode.replace(':', '_')}_{seed}.jsonl", system=s, record=mode)
        fill(st, s, rng, n=3)
        out += read_lines(st.path)
    rot = JSONLStorage(tmp_path / f"rot_{seed}.jsonl", system=s, rotate_records=2)
    for _ in range(3):
        rot.save(s.ask({"amount": rng.randint(1, 500), "limit": 100}))
    out += read_lines(rot.path)
    tid = rot.save_correction("size", {"amount": 3, "limit": 100}, "small", label_source="outcome", note="seen")
    rot.redact(tid, by="dpo")
    out += read_lines(rot.path)
    return out


def test_the_schema_is_the_format_the_code_writes():
    assert SCHEMA["x-solvi-format"] == FORMAT == 2
    assert SCHEMA["properties"]["v"] == {"const": FORMAT}
    assert SCHEMA["x-schema-version"].startswith(f"{FORMAT}.")
    # a redacted decision holds only lasting fields, its answers and its mark
    allowed = set(SCHEMA["$defs"]["ask_redacted"]["properties"])
    assert allowed <= set(KEPT) | {"answers", "id", "hash", "redacted"}


@pytest.mark.parametrize("seed", range(3))
def test_every_record_a_small_system_writes_validates(tmp_path, seed):
    recs = _records(tmp_path, seed)
    kinds = {(r["kind"], r.get("record"), bool(r.get("redacted"))) for r in recs}
    assert {("ask", None, False), ("ask", "compact", False), ("teach", None, False), ("redaction", None, False),
            ("segment", None, False), ("teach", None, True)} <= kinds
    for r in recs:
        assert validate(r, SCHEMA) == [], (r["kind"], r["seq"], validate(r, SCHEMA)[:5])


def test_jsonschema_agrees_when_it_is_installed(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    v = jsonschema.Draft202012Validator(SCHEMA)
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    for r in _records(tmp_path, 7):
        assert not list(v.iter_errors(r)), (r["kind"], [e.message for e in v.iter_errors(r)][:3])


def _one(kind_pred):
    st = MemoryStorage(system())
    fill(st, system(), rng_for(0), n=2)
    return copy.deepcopy(next(r for r in st.rows if kind_pred(r)))


@pytest.mark.parametrize("edit", [
    lambda r: r.update(extra_field=1),                       # a new field is a new schema version
    lambda r: r.update(v=1),
    lambda r: r["answers"].update(pay=["yes", 1.0, "maybe"]),
    lambda r: r.update(hash="x" * 64),
    lambda r: r.update(id=r["id"][:8]),
    lambda r: r.update(seq=-1),
    lambda r: r.pop("response"),
    lambda r: r["records"].append([1, "cap", "not-a-hash"]),
    lambda r: r["response"]["trace"]["records"][0].update(provenance="guessed"),
])
def test_the_schema_rejects_what_the_format_does_not_allow(edit):
    r = _one(lambda r: r["kind"] == "ask" and "response" in r)
    assert validate(r, SCHEMA) == []
    edit(r)
    assert validate(r, SCHEMA) != []


def test_a_redacted_record_with_content_left_is_rejected():
    st = MemoryStorage(system())
    fill(st, system(), rng_for(1), n=2)
    r = copy.deepcopy(next(x for x in st.rows if x.get("redacted")))
    assert validate(r, SCHEMA) == []
    r["response"] = {}
    assert validate(r, SCHEMA) != []


def test_a_process_record_is_checked_for_the_envelope(tmp_path):
    """solvi.experimental.process writes one record of kind "process" per run (the index of its path): the schema lists
    the kind among those checked for the envelope only."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental.process import Process
    s = system()
    p = Process("one", storage=JSONLStorage(tmp_path / "process.jsonl"), on_abstain="abstain", on_human="abstain")
    p.node("a", s, reads=["amount", "limit"], writes={"size": "answer:size"}, end=True)
    r = p.run({"amount": 3, "limit": 100})
    rec = next(x for x in read_lines(tmp_path / "process.jsonl") if x["kind"] == "process")
    assert rec["id"] == r.stored_id and validate(rec, SCHEMA) == []
    assert "process" in SCHEMA["description"]
    bad = dict(rec, seq=-1)
    assert validate(bad, SCHEMA) != []
