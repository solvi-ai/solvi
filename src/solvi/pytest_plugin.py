"""pytest plugin (registered as `solvi` under the pytest11 entry point): collects cases.json / *.cases.json files that have
a task module (a task.py next to them, or {"task": ...} in the file) as test items — one item per case, checked by
solvi.testing (expected answers, statuses, safeguards, trace replay).

    pytest gallery/                      # every gallery entry's cases.json
    pytest gallery/ --solvi-fuzz 30      # and 30 mutated inputs per case (solvi.testing.fuzz)

The plugin loads in every pytest session of an environment where solvi is installed, so it decides what is its own
without running anything: a file is collected only when its cases are solvi cases (a list of objects, each with a
"state" object) and its task module's source imports solvi. Any other JSON file is left alone, and so is the module
next to it — another project's task.py is not executed. Cases that look like solvi's next to a task module that does
not import solvi are not collected, with a warning that says so. Turn the plugin off with `-p no:solvi`."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pytest

from . import testing


def pytest_addoption(parser):
    g = parser.getgroup("solvi")
    g.addoption("--solvi-fuzz", type=int, default=0, metavar="N",
                help="solvi cases: also ask N mutated inputs per case; an exception escaping solvi, or an answer with a "
                     "NaN / out-of-range confidence, fails the case")
    g.addoption("--solvi-seed", type=int, default=0, help="solvi cases: fuzzing seed (default 0)")


_IMPORTS_SOLVI = re.compile(r"^[ \t]*(?:from|import)[ \t]+solvi\b", re.M)


def _solvi_cases(path):
    """Is this cases file solvi's? Decided by reading, never by running: → (yes, why not — when the cases look like
    solvi's but the task module does not; else None)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False, None
    cases, task = (data, None) if isinstance(data, list) else (data.get("cases"), data.get("task")) \
        if isinstance(data, dict) else (None, None)
    if not isinstance(cases, list) or not cases or not all(isinstance(c, dict) and isinstance(c.get("state"), dict)
                                                          for c in cases):
        return False, None
    if task is not None and not isinstance(task, str):
        return False, None
    module = path.parent / (task or "task.py")
    try:
        src = module.read_text()
    except (OSError, ValueError):
        return False, (f"{path}: the cases look like solvi cases, but their task module {module.name} cannot be read"
                       if task else None)
    if _IMPORTS_SOLVI.search(src):
        return True, None
    return False, (f"{path}: the cases look like solvi cases, but {module.name} does not import solvi, so they are not "
                   "collected (the plugin does not run a module to find out) — import solvi in the task module, or run "
                   "`solvi test`")


def pytest_collect_file(file_path, parent):
    path = Path(file_path)
    if not testing.is_cases_file(path):
        return None
    ok, why = _solvi_cases(path)
    if why:
        warnings.warn(pytest.PytestCollectionWarning(why), stacklevel=1)
    return CasesFile.from_parent(parent, path=path) if ok else None


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
        r = testing.run_case(self.system, self.case, state)
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


__all__ = ["pytest_addoption", "pytest_collect_file"]
