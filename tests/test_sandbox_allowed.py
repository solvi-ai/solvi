"""The sandbox admits cmath, and more modules by allowed= (never a forbidden one); the subprocess runner passes them on."""
import os

import pytest

from solvi.experimental.compile.sandbox import Refused, check, load, run

CMATH = "import cmath\ndef f():\n    return cmath.sqrt(-1)\n"
ARRAY = "import array\ndef f():\n    return array.array('i', [1, 2]).tolist()\n"


def test_cmath_is_allowed():
    assert check(CMATH) == []
    assert load(CMATH)["f"]() == 1j


def test_allowed_adds_modules_for_check_and_load():
    assert check(ARRAY) and "import array is not allowed" in check(ARRAY)[0]
    with pytest.raises(Refused):
        load(ARRAY)
    assert check(ARRAY, allowed=("array",)) == []
    assert load(ARRAY, allowed=["array"])["f"]() == [1, 2]
    assert check("import array.x\n", allowed="array") == []


def test_allowed_never_admits_a_forbidden_name():
    for bad in ("os", ("sys",), ["subprocess"]):
        with pytest.raises(ValueError, match="forbidden"):
            check("x = 1\n", allowed=bad)
    with pytest.raises(ValueError):
        check("x = 1\n", allowed=("not a module",))


def test_run_passes_allowed_to_the_child():
    here = (os.path.dirname(__file__),)
    src = "import array\ndef area(r):\n    return array.array('i', [r]).tolist()[0] * 2\n"
    assert "load_error" in run(src, "sandbox_driver:areas", {"rs": [1]}, path=here)
    assert run(src, "sandbox_driver:areas", {"rs": [1, 3]}, path=here, allowed=("array",)) == {"areas": [2, 6]}
