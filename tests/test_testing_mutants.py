"""solvi.testing.mutants: small faults made in a catalog (a comparison flipped or moved by its edge, a threshold
constant nudged, a rule branch swapped, a check dropped) and run against its cases — a mutation score for the cases."""
from __future__ import annotations

import json

import pytest

from solvi.testing import mutants

TASK = '''
from solvi import Answer, Catalog, Question

cat = Catalog()
LIMIT = 100


@cat.fn
def big(amount):
    if amount is None:
        raise ValueError("no amount")
    return amount >= LIMIT


@cat.check(hard=True, then={"action": "block"})
def not_sanctioned(country):
    """a sanctioned country is blocked"""
    return country != "XX"


@cat.rule("action")
def action(big, vip):
    if big:
        return "review" if not vip else "pay"
    else:
        return "pay"


QUESTIONS = [Question("action", "What to do?", Answer.choice(["pay", "review", "block"]))]
'''


def case(name, amount, vip, country, want):
    return {"name": name, "state": {"amount": amount, "vip": vip, "country": country}, "expected": {"action": want}}


WEAK = [case("small", 5, False, "DE", "pay"), case("huge", 1000, False, "DE", "review")]
STRONG = WEAK + [case("at the limit", 100, False, "DE", "review"), case("just under", 99, False, "DE", "pay"),
                 case("vip", 1000, True, "DE", "pay"), case("sanctioned", 5, False, "XX", "block")]


def suite(tmp_path, cases, name="t"):
    d = tmp_path / name
    d.mkdir()
    (d / "task.py").write_text(TASK)
    (d / "cases.json").write_text(json.dumps(cases))
    return d / "cases.json"


def test_the_four_kinds_are_generated_with_line_and_description():
    ms = mutants.generate(TASK)
    kinds = {m.kind for m in ms}
    assert kinds == set(mutants.KINDS)
    d = {(m.kind, m.description) for m in ms}
    assert ("flip", ">= → <") in d and ("boundary", ">= → >") in d and ("flip", "!= → ==") in d
    assert ("constant", "100 → 101") in d and ("constant", "100 → 99") in d
    assert ("drop", "check not_sanctioned() always passes") in d
    assert ("drop", "guard 'if amount is None: raise' never raises") in d
    assert ("branch", "if / else bodies swapped") in d and ("branch", "a if c else b → b if c else a") in d
    assert all(m.line > 0 and m.source != TASK for m in ms)
    assert {m.kind for m in mutants.generate(TASK, kinds=("flip",))} == {"flip"}


def test_weak_cases_let_faults_through_and_strong_cases_kill_them(tmp_path):
    weak = mutants.run(suite(tmp_path, WEAK, "w"))
    strong = mutants.run(suite(tmp_path, STRONG, "s"))
    assert weak.total == strong.total > 0 and weak.score < strong.score
    surv = {(m.kind, m.description) for m in weak.survivors()}
    assert ("boundary", ">= → >") in surv                         # nobody tested the edge
    assert ("drop", "check not_sanctioned() always passes") in surv
    kept = {(m.kind, m.description) for m in strong.survivors()}
    assert ("boundary", ">= → >") not in kept and ("drop", "check not_sanctioned() always passes") not in kept
    killed = next(m for m in strong.mutants if m.description == "check not_sanctioned() always passes")
    assert killed.status == "killed" and killed.killed_by == "sanctioned"
    d = strong.to_dict()
    assert d["score"] == strong.score and set(d["by_kind"]) <= set(mutants.KINDS) and d["seconds"] > 0
    # the original task file is never written
    assert (tmp_path / "s" / "task.py").read_text() == TASK


def test_a_failing_suite_is_refused_and_limit_spreads_the_mutants(tmp_path):
    bad = suite(tmp_path, [case("wrong", 5, False, "DE", "review")], "b")
    with pytest.raises(ValueError, match="fail on the catalog as it is"):
        mutants.run(bad)
    rep = mutants.run(suite(tmp_path, STRONG, "l"), limit=4)
    assert len(rep.mutants) == 4 and len({m.line for m in rep.mutants}) > 1


def test_a_mutant_that_cannot_be_built_is_stillborn_not_scored(tmp_path):
    p = suite(tmp_path, STRONG, "x")
    (p.parent / "task.py").write_text(TASK + "\n_PINNED = {100: 1}[LIMIT]\n")   # a KeyError at load
    rep = mutants.run(p, kinds=("constant",))
    still = [m for m in rep.mutants if m.status == "stillborn"]
    assert {m.description for m in still} == {"100 → 101", "100 → 99"} and all("KeyError" in m.killed_by for m in still)
    assert rep.stillborn == 2 and rep.total == len(rep.mutants) - 2


def test_the_command_line(tmp_path, capsys):
    p = suite(tmp_path, WEAK, "c")
    assert mutants.main([str(p), "--kinds", "boundary,drop"]) == 0
    out = capsys.readouterr().out
    assert "score" in out and "survived" in out
    assert mutants.main([str(p), "--kinds", "nope"]) == 2
    assert mutants.main([str(p), "--json", "--kinds", "flip"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["score"] is not None
    from solvi.cli import main as solvi_main
    assert solvi_main(["mutants", str(p), "--kinds", "flip"]) == 0
    assert "score" in capsys.readouterr().out


def test_factory_named_import_reaches_mutants_and_restores_modules(tmp_path):
    import sys
    import types

    p = suite(tmp_path, STRONG)
    (p.parent / 'run.py').write_text(
        'from task import cat, QUESTIONS\nfrom solvi import System\n'
        'def system():\n    return System(cat, QUESTIONS)\n')
    p.write_text(json.dumps({'system': 'run.py:system', 'cases': STRONG}))
    cached = types.ModuleType('task')
    cached.__file__ = str(p.parent / 'task.py')
    old = sys.modules.get('task')
    sys.modules['task'] = cached
    try:
        report = mutants.run(p, kinds=('boundary',))
        assert report.killed == 1 and report.survived == report.stillborn == 0
        assert sys.modules['task'] is cached
        assert not any('solvi-mutants-' in str(getattr(m, '__file__', '')) for m in sys.modules.values())
    finally:
        if old is None:
            sys.modules.pop('task', None)
        else:
            sys.modules['task'] = old


@pytest.mark.parametrize('isolate', [False, True])
def test_other_directory_task_keeps_neighbour(tmp_path, isolate):
    p = suite(tmp_path, STRONG)
    (p.parent / 'limits.py').write_text('LIMIT = 100\n')
    (p.parent / 'task.py').write_text(TASK.replace('LIMIT = 100', 'from limits import LIMIT'))
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps({'task': str(p.parent / 'task.py'), 'cases': STRONG}))
    report = mutants.run(cases, kinds=('boundary',), isolate=isolate)
    assert report.killed == 1 and report.survived == report.stillborn == 0


def test_import_cleanup_resolves_symlink_roots(tmp_path):
    import sys
    import types

    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    unrelated = types.ModuleType('_mutation_unrelated')
    sys.modules[unrelated.__name__] = unrelated
    try:
        with mutants._imports((alias,)):
            local = types.ModuleType('_mutation_local')
            local.__file__ = str(real / 'task.py')
            sys.modules[local.__name__] = local
        assert '_mutation_local' not in sys.modules
        assert sys.modules[unrelated.__name__] is unrelated
    finally:
        sys.modules.pop(unrelated.__name__, None)


@pytest.mark.parametrize('stage', ['import', 'factory', 'prepare'])
@pytest.mark.parametrize('isolate', [False, True])
def test_lifecycle_timeout_kills_import_loop_and_collects_worker(tmp_path, isolate, monkeypatch, stage):
    import os
    import subprocess
    import time

    p = suite(tmp_path, STRONG)
    loop = 'while LIMIT != int("100"):\n    pass\n'
    if stage == 'factory':
        addition = '\nfrom solvi import System\ndef system():\n' + ''.join('    ' + line + '\n' for line in loop.splitlines()) + '    return System(cat, QUESTIONS)\n'
    elif stage == 'prepare':
        addition = '\ndef prepare(state):\n' + ''.join('    ' + line + '\n' for line in loop.splitlines()) + '    return state\n'
    else:
        addition = '\n' + loop
    (p.parent / 'task.py').write_text(TASK + addition)
    children = []
    original = subprocess.Popen

    def track(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, 'Popen', track)
    started = time.monotonic()
    report = mutants.run(p, kinds=('constant',), timeout=1, isolate=isolate)
    assert time.monotonic() - started < 10
    hung = [m for m in report.mutants if m.description in ('100 → 101', '100 → 99')]
    assert len(hung) == 2
    assert all(m.status == 'killed' and 'timeout: lifecycle' in m.killed_by for m in hung)
    assert all(child.poll() is not None for child in children)
    for child in children:
        with pytest.raises(ProcessLookupError):
            os.kill(child.pid, 0)


@pytest.mark.parametrize('timeout', [0, -1, None, float('nan'), float('inf')])
def test_lifecycle_requires_a_bounded_budget(tmp_path, timeout):
    with pytest.raises(ValueError, match='finite positive'):
        mutants.run(tmp_path / 'missing.json', timeout=timeout)


def test_prepare_fault_counts_as_killed_and_stops_before_next_case(tmp_path):
    p = suite(tmp_path, STRONG)
    # Only constant mutants change LIMIT: baseline prepares every case successfully.
    # A mutant must stop before the second prepare, which would crash the worker.
    addition = '''
_PREPARES = 0
def prepare(state):
    global _PREPARES
    _PREPARES += 1
    if LIMIT != int("100"):
        if _PREPARES > 1:
            import os
            os._exit(19)
        raise ValueError("mutated guard")
    return state
'''
    (p.parent / "task.py").write_text(TASK + addition)
    report = mutants.run(p, kinds=("constant",))
    guarded = [m for m in report.mutants if m.description in ("100 → 101", "100 → 99")]
    assert len(guarded) == 2
    assert all(m.status == "killed" and m.killed_by == "small" for m in guarded)


def test_first_failed_answer_stops_remaining_cases(tmp_path):
    p = suite(tmp_path, WEAK)
    addition = '''
_PREPARES = 0
def prepare(state):
    global _PREPARES
    _PREPARES += 1
    if LIMIT != int("100") and _PREPARES > 1:
        import os
        os._exit(19)
    return state
'''
    # LIMIT 101 makes the first answer wrong; a second prepare must never run.
    p.write_text(json.dumps([case("edge", 100, False, "DE", "review"), *WEAK]))
    (p.parent / "task.py").write_text(TASK + addition)
    report = mutants.run(p, kinds=("constant",))
    changed = next(m for m in report.mutants if m.description == "100 → 101")
    assert changed.status == "killed" and changed.killed_by == "edge"


def test_invalid_mutant_syntax_is_stillborn(tmp_path, monkeypatch):
    p = suite(tmp_path, WEAK)
    monkeypatch.setattr(mutants, "generate", lambda *args, **kwargs: [mutants.Mutant("flip", 1, "invalid", "def !")])
    report = mutants.run(p)
    assert report.stillborn == 1 and report.total == 0
    assert "SyntaxError" in report.mutants[0].killed_by


def test_cli_rejects_missing_cases_and_no_selected_mutants(tmp_path, capsys):
    assert mutants.main([str(tmp_path)]) == 2
    assert "no cases files found" in capsys.readouterr().err
    p = suite(tmp_path, WEAK)
    (p.parent / "task.py").write_text("from solvi import Catalog\ncat = Catalog()\nQUESTIONS = []\n")
    p.write_text(json.dumps([]))
    assert mutants.main([str(p)]) == 2
    assert "no mutation sites selected" in capsys.readouterr().err


@pytest.mark.parametrize("crash", ["raise SystemExit(23)", "__import__('os')._exit(24)",
                                  "__import__('os').kill(__import__('os').getpid(), 11)"])
def test_runtime_crash_after_build_is_killed_in_bounded_driver(tmp_path, crash):
    import subprocess
    import sys

    p = suite(tmp_path, WEAK)
    (p.parent / "task.py").write_text(TASK + "\ndef prepare(state):\n    if LIMIT != int('100'):\n        " +
                                     crash + "\n    return state\n")
    driver = """
import sys
from solvi.testing import mutants
report = mutants.run(sys.argv[1], kinds=('constant',), timeout=1)
changed = [m for m in report.mutants if m.description in ('100 → 101', '100 → 99')]
assert len(changed) == 2
assert all(m.status == 'killed' for m in changed), changed
"""
    subprocess.run([sys.executable, "-c", driver, str(p)], check=True, timeout=15)


@pytest.mark.skipif(__import__('os').name != 'posix', reason='POSIX process group containment')
def test_factory_timeout_collects_its_descendant_in_bounded_driver(tmp_path):
    import subprocess
    import sys

    p = suite(tmp_path, WEAK)
    pid_file = tmp_path / 'descendant.pid'
    (p.parent / "task.py").write_text(TASK + f"""
from solvi import System
def system():
    if LIMIT != int('100'):
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        with open({str(pid_file)!r}, 'a') as output:
            output.write(str(child.pid) + '\\n')
            output.flush()
        while True:
            time.sleep(1)
    return System(cat, QUESTIONS)
""")
    driver = """
import os, signal, sys, time
from pathlib import Path
from solvi.testing import mutants
pids = Path(sys.argv[2])
try:
    report = mutants.run(sys.argv[1], kinds=('constant',), timeout=1)
    changed = [m for m in report.mutants if m.description in ('100 → 101', '100 → 99')]
    assert len(changed) == 2 and all(m.status == 'killed' for m in changed)
    ids = [int(p) for p in pids.read_text().splitlines()]
    assert len(ids) == 2
    for pid in ids:
        for attempt in range(100):
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                break
            time.sleep(.01)
        else:
            raise AssertionError(f'live descendant: {pid}')
finally:
    if pids.exists():
        for pid in map(int, pids.read_text().splitlines()):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
"""
    subprocess.run([sys.executable, "-c", driver, str(p), str(pid_file)], check=True, timeout=15)


# S348/S349/S350: explicit bounded source scope, not a reachability analysis.
def test_main_guards_are_never_sites_or_executed_by_mutants():
    source = """
def rule(x):
    return x >= 18
if __name__ == "__main__":
    if 1 < 2:
        effects.append("script")
if "__main__" == __name__ and enabled:
    effects.append("reversed")
"""
    generated = mutants.generate(source)
    assert generated and all(m.line <= 3 for m in generated)
    for m in generated:
        namespace = {"__name__": "imported_task", "effects": []}
        exec(compile(m.source, "<mutant-test>", "exec"), namespace)
        assert namespace["effects"] == []


def test_function_scope_excludes_helpers_constants_and_checks_names():
    source = """
LIMIT = 18
def rule(age):
    return age >= LIMIT
def demo(age):
    return age < 5
class Rules:
    def decide(self, amount):
        return amount > 8
"""
    generated = mutants.generate(source, functions=["rule"])
    assert generated and {m.line for m in generated} == {4}
    assert all(m.line == 9 for m in mutants.generate(source, functions=["Rules.decide"]))
    with pytest.raises(ValueError, match="unknown.*function"):
        mutants.generate(source, functions=["missing"])
    with pytest.raises(ValueError, match="function"):
        mutants.generate(source, functions=[])


def test_exact_duplicate_asts_and_unchanged_mutants_are_removed():
    import ast
    source = "def rule(x):\n    return 7 if x else 7\n"
    assert mutants.generate(source, kinds=("branch",)) == []
    generated = mutants.generate("def rule(x):\n    return 1 if x < 8 else 2\n")
    trees = [ast.dump(ast.parse(m.source), include_attributes=False) for m in generated]
    assert len(trees) == len(set(trees))
    # Different ASTs may still be semantically equivalent: no claim otherwise.
    generated = mutants.generate("def rule(age):\n    return age >= 18\n")
    assert {m.description for m in generated} >= {">= → >", "18 → 19"}


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_mutant_limit_rejected_before_loading(tmp_path, limit):
    with pytest.raises(ValueError, match="limit.*positive integer"):
        mutants.run(tmp_path / "does-not-exist.json", limit=limit)


def test_simple_boolean_signed_annotated_and_assert_mutations_change_behavior():
    source = """
LIMIT: float = -0.5
def bare(flag):
    if flag:
        return 1
    return 2
def both(a, b):
    return a and b
def below(amount):
    return amount < -5
def checked(flag):
    assert flag
    return 8
"""
    generated = mutants.generate(source)
    descriptions = {m.description for m in generated}
    assert {"-0.5 → -0.55", "-5 → -4", "-5 → -6", "and → or", "condition negated",
            "assert always passes"} <= descriptions
    by_description = {m.description: m for m in generated}
    def load(description):
        namespace = {}
        exec(by_description[description].source, namespace)
        return namespace
    assert load("condition negated")["bare"](True) == 2
    assert load("and → or")["both"](True, False) is True
    assert load("-5 → -4")["below"](-5) is True
    assert load("-5 → -6")["below"](-6) is False
    assert load("assert always passes")["checked"](False) == 8
    assert load("-0.5 → -0.55")["LIMIT"] == -0.55


def test_run_function_scope_leaves_script_and_demo_untouched(tmp_path):
    p = suite(tmp_path, STRONG)
    task = p.parent / "task.py"
    task.write_text(TASK + """
def demo(amount):
    return amount < 42
if __name__ == "__main__":
    raise RuntimeError("must not execute during import")
""")
    report = mutants.run(p, kinds=("boundary",), functions=(name for name in ["big"]))
    assert report.killed == 1 and len(report.mutants) == 1
    assert report.mutants[0].description == ">= → >"
    assert 'raise RuntimeError' in report.mutants[0].source


def test_identical_ast_from_overlapping_site_discovery_counts_once(monkeypatch):
    # Site finders may discover the same fault via more than one operator/path.
    # Public generation must not give those identical resulting programs weight twice.
    original = mutants._sites
    def duplicate_sites(tree, functions=None):
        sites = original(tree, functions)
        return sites + sites
    monkeypatch.setattr(mutants, "_sites", duplicate_sites)
    generated = mutants.generate("def rule(x):\n    return x >= 18\n", kinds=("boundary",))
    assert len(generated) == 1 and generated[0].description == ">= → >"
