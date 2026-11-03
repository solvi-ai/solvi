import copy
import pytest
from solvi import Knowledge, Catalog, Question, Answer, System


def setup(tmp_path, dates=("2021-01-01", "2022-01-01", "2023-01-01")):
    k = Knowledge()
    c = Catalog()

    @c.rule("dose")
    def dose(knowledge: dict, drug: str) -> str:
        return Knowledge.value(knowledge, drug, "dose", "unknown")

    s = System(
        c,
        [Question("dose", "Dose?", Answer.choice(["old", "new", "other", "unknown"]))],
        storage=str(tmp_path / "decisions.jsonl"),
    )
    old = k.tell({"s": "a", "r": "dose", "o": "old"}, source="spec", by="label", valid=("2020-01-01", None))
    other = k.tell({"s": "b", "r": "dose", "o": "other"}, source="spec", by="label", valid=("2020-01-01", None))
    for drug in ["a", "b"]:
        for date in dates:
            s.ask({"knowledge": k.snapshot(about=drug, valid_at=date), "drug": drug, "date": date})
    k.tell({"s": "a", "r": "dose", "o": "new"}, source="spec", by="correction", valid=("2022-01-01", None))
    return k, s, old, other


def test_correction_boundary_queries_no_retraction_or_storage_write(tmp_path):
    k, s, old, other = setup(tmp_path)
    before = copy.deepcopy(k.store.journal)
    n = len(list(s.storage.iter()))
    out = k.redecide(s.storage, s, affected=[old, other], valid_at_key="date")
    changes = out["answer_changes"]
    assert len(changes) == 2
    assert all(x["old"]["dose"][0] == "old" and x["new"]["dose"][0] == "new" for x in changes)
    assert len(list(s.storage.iter())) == n and k.store.journal == before
    assert k.value(k.snapshot(about="a", valid_at="2021-01-01"), "a", "dose") == "old"


@pytest.mark.parametrize("bad", [None, "", "2022-99-01", True, 3])
def test_invalid_later_date_preflight_no_callbacks(tmp_path, bad):
    k, s, old, _ = setup(tmp_path)
    rows = list(s.storage.iter())
    rows[1].input["date"] = bad

    class Rows:
        def iter(self):
            return iter(rows)

    calls = []
    with pytest.raises(ValueError):
        k.redecide(Rows(), lambda st: calls.append(st), affected=[old], valid_at_key="date")
    assert calls == []


@pytest.mark.parametrize("affected,key", [([], "date"), (["bad"], "date"), (None, "date"), (["placeholder"], "")])
def test_bad_config_no_callbacks(tmp_path, affected, key):
    k, s, old, _ = setup(tmp_path)
    calls = []
    with pytest.raises((ValueError, TypeError)):
        k.redecide(s.storage, lambda st: calls.append(st), affected=affected, valid_at_key=key)
    assert not calls


def test_historical_default_still_removal_unknown(tmp_path):
    k, s, old, _ = setup(tmp_path)
    changes, _ = k.store.redecide(s.storage, [old], s)
    assert changes and all(x["new"]["dose"][0] == "unknown" for x in changes)


def test_conflict_uses_resolver_instead_of_guessing_new_value(tmp_path):
    k, s, old, _ = setup(tmp_path)
    k.tell({"s": "a", "r": "dose", "o": "other"}, source="spec", by="another", valid=("2022-01-01", None))
    expected = Knowledge.value(k.snapshot(about="a", valid_at="2023-01-01"), "a", "dose", "unknown")
    out = k.redecide(s.storage, s, affected=[old], valid_at_key="date")
    assert expected == "unknown"
    assert any(x["new"]["dose"][0] == expected for x in out["answer_changes"])


def test_callback_mutation_is_private_to_each_input(tmp_path):
    k, s, old, _ = setup(tmp_path)
    seen = []

    def decide(st):
        seen.append(st["drug"])
        value = Knowledge.value(st["knowledge"], st["drug"], "dose", "unknown")
        st["knowledge"]["facts"].clear()
        st["drug"] = "mutated"
        return value

    k.redecide(s.storage, decide, affected=[old], valid_at_key="date")
    assert seen and set(seen) == {"a"}
    assert all(row.input["drug"] in ("a", "b") for row in s.storage.iter())


@pytest.mark.parametrize("mode", ["bool_number", "reorder", "structured"])
def test_correction_json_comparison_preserves_types_and_objects(tmp_path, mode):
    k, s, old, _ = setup(tmp_path)

    def decide(st):
        changed = Knowledge.value(st["knowledge"], "a", "dose", "unknown") == "new"
        if mode == "bool_number":
            return 1 if changed else True
        if mode == "reorder":
            return {"b": 2, "a": 1} if changed else {"a": 1, "b": 2}
        return {"answer": [2 if changed else 1]}

    out = k.redecide(s.storage, decide, affected=[old], valid_at_key="date")
    assert len(out["answer_changes"]) == (0 if mode == "reorder" else 2)


def test_structured_system_answers_stay_json(tmp_path):
    k, s, old, _ = setup(tmp_path)
    c = Catalog()

    @c.rule("value")
    def value(knowledge: dict, drug: str):
        return {"dose": [Knowledge.value(knowledge, drug, "dose", "unknown")]}

    system = System(c, [Question("value", "Value?", Answer.value())])
    out = k.redecide(s.storage, system, affected=[old], valid_at_key="date")
    assert len(out["answer_changes"]) == 2
    assert out["answer_changes"][0]["new"]["value"][0] == {"dose": ["new"]}


@pytest.mark.parametrize("kind", ["duplicate", "missing_at", "bad_query"])
def test_malformed_selected_records_fail_before_callbacks(tmp_path, kind):
    k, s, old, _ = setup(tmp_path)
    rows = list(s.storage.iter())
    calls = []
    if kind == "duplicate":
        rows.append(rows[0])
    elif kind == "missing_at":
        del rows[1].input["knowledge"]["at"]
    else:
        rows[1].input["knowledge"]["query"] = []

    class Rows:
        def iter(self):
            return iter(rows)

    with pytest.raises(ValueError):
        k.redecide(Rows(), lambda st: calls.append(st), affected=[old], valid_at_key="date")
    assert not calls


@pytest.mark.parametrize("bad", [float("nan"), {1: "bad"}, (1,), object()])
def test_non_json_callback_output_refused(tmp_path, bad):
    k, s, old, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="JSON"):
        k.redecide(s.storage, lambda st: bad, affected=[old], valid_at_key="date")


def test_unknown_comparison_tag_distinct_from_literal_marker():
    from solvi import Unknown
    from solvi.core.knowledge.store import _correction_json

    assert _correction_json(Unknown, unknown=True)[1] != _correction_json({"$solvi": "unknown"}, unknown=True)[1]
    assert _correction_json(Unknown, unknown=True)[1] != _correction_json("not stated", unknown=True)[1]


def test_system_unknown_to_literal_marker_has_explicit_sidecar(tmp_path):
    from solvi import Unknown

    k, s, old, _ = setup(tmp_path)
    c = Catalog()

    @c.rule("value")
    def value(knowledge: dict, drug: str):
        return {"$solvi": "unknown"} if Knowledge.value(knowledge, drug, "dose", "unknown") == "new" else Unknown

    system = System(c, [Question("value", "Value?", Answer.maybe(Answer.value()))])
    out = k.redecide(s.storage, system, affected=[old], valid_at_key="date")
    assert len(out["answer_changes"]) == 2
    change = out["answer_changes"][0]
    assert change["old"]["value"][0] == change["new"]["value"][0] == {"$solvi": "unknown"}
    assert change["old_unknown_paths"] == ["/value/0"]
    assert change["new_unknown_paths"] == []
    assert change["old"]["value"][1] == change["new"]["value"][1] == "ok"
