"""The experimental warning never names a release that is already out as the one that removes the piece."""
import re
import warnings

import solvi
from solvi import experimental


def _release(v):
    return tuple(int(x) for x in re.match(r"(\d+)\.(\d+)", v).groups())


def test_the_deadline_is_not_a_release_already_out():
    m = re.fullmatch(r"\d+\.\d+(\.\d+)?", experimental.DEADLINE)
    assert m is None or _release(experimental.DEADLINE) > _release(solvi.__version__), experimental.DEADLINE
    assert all(st["deadline"] == experimental.DEADLINE for st in experimental.STATUS.values())


def test_the_import_warning_says_the_deadline():
    experimental._WARNED.discard("solvi.experimental.laws_probe")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        experimental.warn_on_import("solvi.experimental.laws_probe")
    experimental._WARNED.discard("solvi.experimental.laws_probe")
    text = " ".join(str(x.message) for x in w)
    assert f"removed in {experimental.DEADLINE} unless it graduates" in text
    assert "removed in 1.2" not in text
