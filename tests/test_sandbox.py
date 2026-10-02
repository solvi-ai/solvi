"""solvi.sandbox: the ast allowlist refuses imports outside the pure standard library, reflection names and dunders;
`run` executes a module in a subprocess under a trusted driver with limits; `load` gives the namespace in this process
with the source registered, so a part's fingerprint is taken from its syntax tree."""
import inspect

import pytest

from solvi import Catalog
from solvi.provenance import code_fingerprint
from solvi.sandbox import Refused, check, load, run

GOOD = "import math\n\ndef area(r):\n    return math.pi * r * r\n"


def test_a_pure_module_with_standard_library_imports_is_allowed():
    assert check(GOOD) == []
    assert check("from collections import Counter\n\ndef f(xs):\n    return Counter(xs)\n") == []


@pytest.mark.parametrize("src, word", [
    ("import os\n", "import os"),
    ("from subprocess import run\n", "from subprocess"),
    ("from . import x\n", "from ."),
    ("def f():\n    return open('x').read()\n", "'open'"),
    ("def f(x):\n    return getattr(x, 'y')\n", "'getattr'"),
    ("def f(x):\n    return x.__class__\n", "__class__"),
    ("def f():\n    return __builtins__\n", "__builtins__"),
    ("if __name__ == '__main__':\n    pass\n", "__name__"),
    ("n = 0\ndef f():\n    global n\n    n += 1\n", "global"),
    ("def f(\n", "syntax error"),
])
def test_imports_reflection_dunders_global_state_and_syntax_errors_are_refused_with_the_line(src, word):
    why = check(src)
    assert why and any(word in w for w in why)


def test_load_refuses_what_check_refuses_and_names_the_reasons():
    with pytest.raises(Refused) as e:
        load("import os\n")
    assert "import os" in e.value.reasons[0]


def test_a_loaded_function_has_its_source_so_its_fingerprint_is_the_same_for_the_same_text():
    ns = load(GOOD)
    assert "math.pi" in inspect.getsource(ns["area"])
    assert code_fingerprint(ns["area"]) == code_fingerprint(load(GOOD)["area"])
    other = load(GOOD.replace("r * r", "r ** 2"))
    assert code_fingerprint(ns["area"]) != code_fingerprint(other["area"])
    cat = Catalog()
    cat.fn(ns["area"])
    assert cat.parts["area"].func(r=1.0) == pytest.approx(3.14159, abs=1e-4)


def test_the_builtins_of_a_loaded_module_admit_only_the_allowed_imports():
    from solvi.sandbox import safe_builtins
    b = safe_builtins()
    assert "open" not in b and "getattr" not in b
    assert b["__import__"]("json").dumps(1) == "1"
    with pytest.raises(ImportError):
        b["__import__"]("os")
    assert load("def g():\n    import json\n    return json.dumps(1)\n")["g"]() == "1"


def test_run_executes_the_module_in_a_subprocess_under_the_trusted_driver():
    out = run(GOOD, "sandbox_driver:areas", {"rs": [1, 2]}, path=[_root()])
    assert out == {"areas": [pytest.approx(3.14159, abs=1e-4), pytest.approx(12.566, abs=1e-3)]}


def test_run_reports_a_module_that_does_not_load_and_one_that_runs_over_its_limits():
    assert "import os" in run("import os\n", "sandbox_driver:areas", {"rs": []}, path=[_root()])["load_error"]
    boom = "def area(r):\n    raise ValueError('no')\nx = area(1)\n"
    assert "ValueError" in run(boom, "sandbox_driver:areas", {"rs": []}, path=[_root()])["load_error"]
    slow = "def area(r):\n    while True:\n        r += 1\n"
    out = run(slow, "sandbox_driver:areas", {"rs": [1]}, path=[_root()], cpu_s=2, wall_s=20)
    assert "crash" in out or "TimeoutError" in str(out)


def _root():
    from pathlib import Path
    return Path(__file__).resolve().parent
