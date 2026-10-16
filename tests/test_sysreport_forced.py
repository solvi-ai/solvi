"""The system report on answers forced by a hard check: whether the check changed the answer (the answer the rule
gave without it is in the trace) or not, and how many forced answers cannot say because early_exit skipped the rule."""
from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage


def _system(storage, early_exit):
    cat = Catalog()

    @cat.check(hard=True, then={"size": "half"})
    def ok(y: bool) -> bool:
        return y

    @cat.rule("size")
    def size(z: bool) -> str:
        return "half" if z else "full"
    return System(cat, [Question("size", "s", Answer.choice(["full", "half", "zero"]), requires=["ok"])],
                  storage=storage, early_exit=early_exit)


def _fill(s):
    for z in (True, True, False):                       # the check fails three times; the rule says half, half, full
        s.ask({"y": False, "z": z})
    s.ask({"y": True, "z": False})                      # passes: the rule answers alone


def test_forced_answers_split_into_changed_unchanged_and_unknown(tmp_path):
    s = _system(JSONLStorage(tmp_path / "a.jsonl"), early_exit=False)
    _fill(s)
    p = s.report().to_dict()["system1"]["size"]
    assert (p["forced"], p["alone"]) == (3, 1)
    assert p["forced_split"] == {"changed": 1, "unchanged": 2, "unknown": 0}
    assert p["answered_by"]["hard check: ok"] == 3
    text = str(s.report())
    assert "changed the answer 1" in text and "would have been the same 2" in text


def test_early_exit_records_are_flagged_as_unknown(tmp_path):
    s = _system(JSONLStorage(tmp_path / "b.jsonl"), early_exit=True)
    _fill(s)
    p = s.report().to_dict()["system1"]["size"]
    assert p["forced_split"] == {"changed": 0, "unchanged": 0, "unknown": 3}
    assert "early_exit" in str(s.report())
