"""run_file(isolate=True): each case in a fresh process — the same statuses as in-process, no module state shared
between cases, a dead interpreter or a case stuck in a C call is that case's failure; max_mem= caps a case's memory
(status "memory"), checked after the case where the platform does not enforce it."""
import json
import sys

import pytest

from solvi import cli, testing
from solvi.testing import _isolate

from test_cases_runner_status import CASES, _write

HOG_TASK = '''
import os
import time
from solvi import Answer, Catalog, Question

cat = Catalog()
_first = []


@cat.fn
def words(text):
    if not _first:                                    # module state: a later case in the same process is not first
        _first.append(text)
    if "hog" in text:
        blob = bytearray(3 * 2 ** 30)                 # 3 GB
        return [str(len(blob))]
    if "die" in text:
        os._exit(3)                                   # the interpreter ends: no exception to catch
    if "stuck" in text:
        time.sleep(30)                                # one C call: SIGALRM would wait for it to return
    print("noise on stdout")                          # must not break the protocol
    return text.lower().split() + ["first" if _first[0] == text else "later"]


@cat.rule("route")
def route(words):
    return "billing" if "first" in words else "support"


QUESTIONS = [Question("route", "Route?", Answer.choice(["billing", "support"]))]
'''


def _hog(tmp_path, names):
    cases = [{"name": n, "state": {"text": n}, "expected": {"route": "billing"}} for n in names]
    (tmp_path / "task.py").write_text(HOG_TASK)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    return tmp_path / "cases.json"


def test_isolated_cases_have_the_in_process_statuses(tmp_path):
    res = testing.run_file(_write(tmp_path), timeout=2, reruns=3, isolate=True)
    st = {c.name: c.status for c in res.cases}
    assert st == {"fine": "ok", "wrong": "regression", "flip": "nondeterministic", "drift": "nondeterministic",
                  "slow": "timeout", "boom": "error", "after": "ok"}, st
    assert res.counts() == {"ok": 2, "regression": 1, "nondeterministic": 2, "timeout": 1, "error": 1}
    assert "the process was killed" in next(c for c in res.cases if c.name == "slow").problems[0]


def test_each_case_has_its_own_process(tmp_path):
    f = _hog(tmp_path, ["one", "two", "three"])
    assert [c.status for c in testing.run_file(f).cases] == ["ok", "regression", "regression"]     # shared state
    assert [c.status for c in testing.run_file(f, isolate=True).cases] == ["ok", "ok", "ok"]


def test_a_dead_interpreter_and_a_stuck_c_call_fail_only_their_case(tmp_path):
    res = testing.run_file(_hog(tmp_path, ["die", "stuck", "after"]), timeout=1.5, isolate=True)
    die, stuck, after = res.cases
    assert die.status == "error" and "exited with code 3" in die.problems[0]
    assert stuck.status == "timeout" and stuck.ms < 10_000
    assert after.ok


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is enforced on Linux")
def test_max_mem_stops_a_case_that_needs_more(tmp_path):
    res = testing.run_file(_hog(tmp_path, ["hog", "after"]), isolate=True, max_mem="600M")
    hog, after = res.cases
    assert hog.status == "memory" and hog.problems[0].startswith("memory: ") and "MemoryError" in " ".join(hog.problems)
    assert after.ok
    assert res.counts() == {"ok": 1, "memory": 1}


def test_max_mem_needs_isolate_and_a_size(tmp_path):
    f = _hog(tmp_path, ["one"])
    with pytest.raises(ValueError, match="isolate=True"):
        testing.run_file(f, max_mem="1G")
    with pytest.raises(ValueError):
        testing.run_file(f, isolate=True, max_mem="lots")
    assert _isolate.parse_mem("512M") == 512 * 2 ** 20 and _isolate.parse_mem("1.5GB") == 3 * 2 ** 29
    assert _isolate.parse_mem("2GiB") == 2 * 2 ** 30 and _isolate.parse_mem(1000) == 1000


def test_where_the_cap_is_not_enforced_the_peak_is_checked_after_the_case(monkeypatch):
    monkeypatch.setattr(_isolate.sys, "platform", "darwin")
    note = _isolate.cap(2 ** 30)
    assert note and "macOS" in note
    r = _isolate._over(testing.CaseResult("c"), 2 ** 20, 5 * 2 ** 20, note)
    assert r.status == "memory" and "checked after the case" in r.problems[0]
    assert _isolate._over(testing.CaseResult("c"), 2 ** 20, 5 * 2 ** 20, None).status == "ok"   # enforced: no check


def test_command_line_isolate(tmp_path, capsys):
    f = _hog(tmp_path, ["one", "two"])
    assert cli.main(["test", str(f), "--isolate", "-q"]) == 0
    assert "2/2 cases passed" in capsys.readouterr().out
    assert cli.main(["test", str(f), "--max-mem", "1G"]) == 2
    assert "isolate" in capsys.readouterr().err


def test_a_task_that_does_not_load_is_the_files_error(tmp_path):
    (tmp_path / "task.py").write_text("raise RuntimeError('broken task')\n")
    (tmp_path / "cases.json").write_text(json.dumps(CASES[:1]))
    res = testing.run_file(tmp_path / "cases.json", isolate=True)
    assert res.error and "broken task" in res.error and not res.cases
