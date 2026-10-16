"""solvi.testing.history: verdict classes across runs, quarantine, the known-bug registry and the run-level status."""
import json

from solvi.testing import CaseResult, FileResult
from solvi.testing.history import KnownBugs, VerdictHistory, run_status


def test_classes_over_runs():
    h = VerdictHistory(window=5)
    runs = [{"a": "ok", "b": "ok", "c": "regression", "d": "ok", "e": "ok"},
            {"a": "ok", "b": "regression", "c": "regression", "d": "regression", "e": "ok"},
            {"a": "ok", "b": "regression", "c": "ok", "d": "ok", "e": {"status": "ok", "attempt": 2}}]
    for r in runs:
        h.add(r)
    assert {k: h.classify(k) for k in "abcde"} == {"a": "stable", "b": "regression", "c": "fixed", "d": "flaky",
                                                    "e": "flaky"}
    assert h.classify("zzz") == "new"
    assert h.quarantined() == ["d", "e"]
    assert h.stable_share() == 0.2
    assert h.report()["e"]["later_attempt"] == 1
    g = h.gate({"a": "ok", "b": "regression", "d": "regression"})
    assert g == {"ok": False, "failed": ["b"], "quarantined": ["d"]}
    assert h.gate({"d": "error"})["ok"] is True


def test_window_forgets_old_flips():
    h = VerdictHistory(window=3)
    for s in ["ok", "regression", "ok", "ok", "ok"]:
        h.add({"x": s})
    assert h.classify("x") == "stable"


def test_history_kept_in_a_store(tmp_path):
    p = tmp_path / "v.jsonl"
    VerdictHistory(p).add({"a": "ok"}, run="r1")
    h = VerdictHistory(p)
    h.add({"a": "regression"}, run="r2")
    again = VerdictHistory(p)
    assert [v["run"] for v in again.verdicts("a")] == ["r1", "r2"] and again.classify("a") == "regression"
    assert all(json.loads(line).get("hash") for line in p.read_text().splitlines())    # chained like any record


def test_run_status_and_file_results(tmp_path):
    f = FileResult(tmp_path / "x.cases.json",
                   [CaseResult("a"), CaseResult("b", ["crash: ConnectionError"], status="error"),
                    CaseResult("c", ["timeout: 5 s"], status="timeout"), CaseResult("d", ["crash: ConnectionError"],
                                                                                    status="error")])
    st = run_status(f)
    assert st["status"] == "degraded" and st["blocked"] == 3 and st["passed"] == 1
    assert run_status(f, min_blocked=4)["status"] == "failed"
    assert run_status({"a": "ok", "b": "ok"})["status"] == "passed"
    assert run_status({"a": "regression", "b": "regression", "c": "regression"},
                      pattern="unreachable")["status"] == "failed"
    h = VerdictHistory()
    assert h.add(f, skip_degraded=True)["status"] == "degraded" and h.runs == []
    h.add(f)
    assert str(tmp_path / "x.cases.json") + "::b" in h.keys()


def test_known_bugs(tmp_path):
    p = tmp_path / "bugs.jsonl"
    kb = KnownBugs(p, quiet=2)
    kb.register("BUG-1", r"expected 'high', got 'low'")
    kb.register("BUG-2", r"ConnectionError", key=r"^net::")
    run1 = {"t::a": CaseResult("a", ["p: expected 'high', got 'low' [ok]"]),
            "net::b": CaseResult("b", ["crash: ConnectionError"], status="error"),
            "t::c": CaseResult("c", ["crash: ConnectionError"], status="error"),
            "t::d": "ok"}
    out = kb.triage(run1)
    assert out["known"] == {"t::a": "BUG-1", "net::b": "BUG-2"} and out["new"] == ["t::c"]
    kb.fixed("BUG-1")
    assert kb.triage({"t::a": CaseResult("a", ["p: expected 'high', got 'low'"])})["reappeared"] == ["BUG-1"]
    kb.triage({"t::d": "ok"})
    again = KnownBugs(p, quiet=2)                     # replayed from the store
    assert again.run == 3 and again.bugs["BUG-1"]["status"] == "fixed"
    assert again.not_seen() == ["BUG-2"]              # open, last seen in run 1: check that it is fixed
