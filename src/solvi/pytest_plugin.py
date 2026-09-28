"""pytest plugin (registered as `solvi` under the pytest11 entry point): collects cases.json / *.cases.json files that have
a task module (a task.py next to them, or {"task": ...} in the file) as test items — one item per case, checked by
solvi.testing (expected answers, statuses, safeguards, trace replay).

    pytest gallery/                      # every gallery entry's cases.json
    pytest gallery/ --solvi-fuzz 30      # and 30 mutated inputs per case (solvi.testing.fuzz)

A JSON file without a task module is left alone. Turn the plugin off with `-p no:solvi`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import testing


def pytest_addoption(parser):
    g = parser.getgroup("solvi")
    g.addoption("--solvi-fuzz", type=int, default=0, metavar="N",
                help="solvi cases: also ask N mutated inputs per case; an exception escaping solvi, or an answer with a "
                     "NaN / out-of-range confidence, fails the case")
    g.addoption("--solvi-seed", type=int, default=0, help="solvi cases: fuzzing seed (default 0)")


def _has_task(path):
    if (path.parent / "task.py").exists():
        return True
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and bool(data.get("task")) and isinstance(data.get("cases"), list)


def pytest_collect_file(file_path, parent):
    path = Path(file_path)
    if testing.is_cases_file(path) and _has_task(path):
        return CasesFile.from_parent(parent, path=path)
    return None


class CasesFile(pytest.File):
    def collect(self):
        suite = testing.load(self.path)
        system = suite.system()
        names = {}
        for i, case in enumerate(suite.cases):
            name = str(case.get("name") or f"case {i + 1}")
            names[name] = names.get(name, 0) + 1
            if names[name] > 1:
                name = f"{name} ({names[name]})"
            yield CaseItem.from_parent(self, name=name, suite=suite, system=system, case=case, index=i)


class CaseFailed(Exception):
    pass


class CaseItem(pytest.Item):
    def __init__(self, *, suite, system, case, index, **kw):
        super().__init__(**kw)
        self.suite, self.system, self.case, self.index = suite, system, case, index

    def runtest(self):
        state = self.suite.state(self.case)
        r = testing.check(self.system, self.case, state)
        n = self.config.getoption("--solvi-fuzz")
        if n:
            seed = self.config.getoption("--solvi-seed") + self.index
            r.problems += [f"fuzz {c}" for c in testing.fuzz(self.system, state, n, seed)]
        if r.problems:
            raise CaseFailed(r.problems)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, CaseFailed):
            return f"case {self.name!r} in {self.path}:\n  " + "\n  ".join(p.rstrip() for p in excinfo.value.args[0])
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, None, f"solvi case: {self.name}"
