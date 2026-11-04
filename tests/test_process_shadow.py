"""A complete multi-System candidate runs on an isolated input and journal namespace."""
import copy
import os
import warnings

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import Process


def pipeline(tmp_path, prefix, offset=0):
    first, second = Catalog(), Catalog()

    @first.rule("score")
    def score(x):
        return x + offset

    @second.rule("result")
    def result(value):
        return value * 2

    a = System(first, [Question("score", "score", Answer.value())], storage=tmp_path / f"{prefix}-a.jsonl")
    b = System(second, [Question("result", "result", Answer.value())], storage=tmp_path / f"{prefix}-b.jsonl")
    p = Process(prefix, storage=tmp_path / f"{prefix}-process.jsonl", on_abstain="abstain", on_human="abstain")
    p.node("first", a, question="score", reads=["x"], writes={"value": "answer"})
    p.node("second", b, question="result", reads=["value"], writes={"answer": "answer"}, end=True)
    p.edge("first", "second")
    return p


def test_whole_graph_difference_returns_current_and_each_side_replays(tmp_path):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate", 1)
    journal = JSONLStorage(tmp_path / "comparisons.jsonl")
    shadow = current.shadow(candidate, storage=journal)
    facts = {"x": 3, "unused": {"rows": [1]}}
    expected = copy.deepcopy(facts)
    served = shadow.run(facts, run="request-1")
    assert served is shadow.last["current"]
    assert served.answer == 6 and shadow.last["candidate"].answer == 8
    assert facts == expected
    assert shadow.stats == {"runs": 1, "agree": 0, "differ": 1, "errors": 0}
    assert current.replay(served.run)["ok"]
    assert candidate.replay(shadow.last["candidate"].run)["ok"]
    assert journal.verify()["ok"]
    comparison = next(journal.iter(kind="process_shadow")).data["shadow"]
    assert comparison["current"]["stored_id"] == served.stored_id
    assert comparison["candidate"]["stored_id"] == shadow.last["candidate"].stored_id
    assert "input" not in comparison and comparison["different"]


def test_candidate_failure_does_not_replace_served_result_and_history_is_bounded(tmp_path, monkeypatch):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate")
    shadow = current.shadow(candidate, keep=1)

    def fail(*args, **kwargs):
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(candidate, "run", fail)
    assert shadow.run({"x": 2}).answer == 4
    assert shadow.run({"x": 3}).answer == 6
    assert shadow.stats["errors"] == 2 and len(shadow.errors) == 1
    assert shadow.last["candidate"] is None
    assert shadow.last["comparison"]["error"] == "RuntimeError"
    assert shadow.last["comparison"]["error_hash"]


def test_candidate_receives_original_input_when_current_mutates_its_private_copy(tmp_path, monkeypatch):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate")
    native = current.run

    def mutating(facts, run=None):
        result = native(facts, run=run)
        facts["x"] = 1000
        facts["nested"].append("external mutation")
        return result

    monkeypatch.setattr(current, "run", mutating)
    shadow = current.shadow(candidate)
    facts = {"x": 2, "nested": []}
    assert shadow.run(facts).answer == 4
    assert shadow.last["candidate"].answer == 4
    assert facts == {"x": 2, "nested": []}
    assert shadow.stats["agree"] == 1


@pytest.mark.parametrize("alias", ["same_process", "system", "journal", "hardlink", "comparison"])
def test_aliases_fail_before_serving(tmp_path, alias):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate")
    storage = None
    if alias == "same_process":
        candidate = current
    elif alias == "system":
        candidate.systems["first"] = current.systems["first"]
    elif alias == "journal":
        candidate.storage = current.storage
    elif alias == "hardlink":
        path = tmp_path / "hardlink.jsonl"
        current.systems["first"].storage._append({"kind": "fixture"})
        os.link(current.systems["first"].storage.path, path)
        candidate.storage = path
    else:
        storage = current.storage
    before = current.systems["first"].storage.head()
    with pytest.raises(ValueError, match="distinct|disjoint|separate"):
        current.shadow(candidate, storage=storage)
    assert current.systems["first"].storage.head() == before


def test_alias_added_later_and_human_handoff_are_refused_before_live_calls(tmp_path):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate")
    shadow = current.shadow(candidate)
    before = current.systems["first"].storage.head()
    candidate.on_human = "handoff"
    with pytest.raises(ValueError, match="disable human"):
        shadow.run({"x": 3})
    candidate.on_human = "abstain"
    candidate.systems["second"].storage = current.systems["first"].storage
    with pytest.raises(ValueError, match="disjoint"):
        shadow.run({"x": 3})
    assert current.systems["first"].storage.head() == before


def test_failed_comparison_write_is_visible_but_does_not_change_answer(tmp_path, monkeypatch):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate")
    journal = JSONLStorage(tmp_path / "comparisons.jsonl")
    shadow = current.shadow(candidate, storage=journal)

    def fail(record):
        raise OSError("disk error")

    monkeypatch.setattr(journal, "_append", fail)
    assert shadow.run({"x": 5}).answer == 10
    assert shadow.stats["errors"] == 1
    assert "comparison journal" in shadow.errors[-1]["error"]


def test_exception_message_is_hashed_not_written_to_comparison_journal(tmp_path, monkeypatch):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate")
    journal = JSONLStorage(tmp_path / "comparisons.jsonl")
    shadow = current.shadow(candidate, storage=journal)

    def fail(facts, **kwargs):
        raise RuntimeError(facts["secret"])

    monkeypatch.setattr(candidate, "run", fail)
    assert shadow.run({"x": 3, "secret": "PRIVATE-ONLY-IN-NODE-TRACE"}).answer == 6
    record = next(journal.iter(kind="process_shadow")).data
    assert record["shadow"]["error"] == "RuntimeError"
    assert "PRIVATE-ONLY-IN-NODE-TRACE" not in str(record)


def test_sidecar_alias_and_unknown_store_backend_are_refused_before_runs(tmp_path):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate")
    candidate.storage = str(current.systems["first"].storage.path) + ".head"
    with pytest.raises(ValueError, match="disjoint"):
        current.shadow(candidate)
    candidate = pipeline(tmp_path, "other")
    candidate.storage = object()
    with pytest.raises(ValueError, match="native JSONL"):
        current.shadow(candidate)


def test_exception_with_broken_renderer_cannot_replace_served_answer(tmp_path, monkeypatch):
    current, candidate = pipeline(tmp_path, "live"), pipeline(tmp_path, "candidate")

    class BrokenError(Exception):
        def __str__(self):
            raise RuntimeError("cannot render")

    def fail(*args, **kwargs):
        raise BrokenError()

    monkeypatch.setattr(candidate, "run", fail)
    shadow = current.shadow(candidate)
    assert shadow.run({"x": 4}).answer == 8
    assert shadow.last["comparison"]["error"] == "BrokenError"
    assert shadow.last["comparison"]["error_hash"] is None
