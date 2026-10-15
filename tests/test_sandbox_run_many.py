"""sandbox.run_many: many modules in one subprocess, each with its own result; a module that kills the process is
reported and the rest run in a fresh one."""
import time
from pathlib import Path

import pytest

from solvi.experimental.compile.sandbox import run, run_many

ROOT = [Path(__file__).resolve().parent]
GOOD = "import math\n\ndef area(r):\n    return math.pi * r * r\n"
DOUBLE = "def area(r):\n    return 2 * r\n"


def test_each_module_gets_its_own_result_in_order():
    out = run_many([GOOD, "import os\n", DOUBLE, "def area(r):\n    raise ValueError('no')\n"],
                   "sandbox_driver:areas", {"rs": [1, 2]}, path=ROOT)
    assert len(out) == 4
    assert out[0] == {"areas": [pytest.approx(3.14159, abs=1e-4), pytest.approx(12.566, abs=1e-3)]}
    assert "import os" in out[1]["load_error"]
    assert out[2] == {"areas": [2, 4]}
    assert "ValueError" in out[3]["crash"]
    assert run_many([], "sandbox_driver:areas", {"rs": []}, path=ROOT) == []


def test_a_module_that_kills_the_process_does_not_take_the_others_with_it():
    hog = "def area(r):\n    while True:\n        r += 1\n"
    spin = "x = 0\nwhile True:\n    x += 1\n"           # loops at load time: the CPU limit ends the process
    out = run_many([DOUBLE, spin, DOUBLE, hog, DOUBLE], "sandbox_driver:areas", {"rs": [3]}, path=ROOT, cpu_s=3,
                   wall_s=30)
    assert out[0] == out[2] == out[4] == {"areas": [6]}
    assert "crash" in out[1]
    assert "TimeoutError" in str(out[3])               # the driver's per-item alarm


def test_one_process_is_faster_than_one_per_module():
    t0 = time.perf_counter()
    for _ in range(4):
        run(DOUBLE, "sandbox_driver:areas", {"rs": [1]}, path=ROOT)
    single = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_many([DOUBLE] * 4, "sandbox_driver:areas", {"rs": [1]}, path=ROOT)
    assert time.perf_counter() - t0 < single
