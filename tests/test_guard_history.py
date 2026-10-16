"""System.guard_history: every calibration attempt of System.guarantee is logged — installed (and whether it replaced
one), failed (with the error), removed — so a guarantee replaced or lost along the way can be traced."""
import pytest

from solvi import Answer, Catalog, Question, System


def _system():
    cat = Catalog()

    @cat.fn
    def score(x) -> float:
        return x

    @cat.rule("ok")
    def ok(x) -> str:
        return "yes" if x > 0.5 else "no"
    return System(cat, [Question("ok", "ok?", Answer.yes_no())])


def _ex(n):
    return [({"x": i / n}, "yes" if i / n > 0.5 or i % 7 == 0 else "no") for i in range(n)]


def test_every_attempt_is_logged():
    s = _system()
    assert s.guard_history == []
    s.guarantee("ok", _ex(400), max_risk=0.1, signal="score")
    s.guarantee("ok", _ex(300), max_error=0.1, signal="score", method="ltt")
    with pytest.raises(ValueError):
        s.guarantee("ok", _ex(300), max_risk=0.1, signal="nowhere")
    s.guarantee("ok", False)
    h = s.guard_history
    assert [e["outcome"] for e in h] == ["installed", "installed", "failed", "removed"]
    assert [e["replaced"] for e in h] == [False, True, True, False]   # the failed attempt took the old one with it
    assert h[0]["question"] == "ok" and h[0]["n"] == 400 and h[0]["threshold"] is not None and h[0]["promise"]
    assert h[1]["n"] == 300 and "ValueError" in h[2]["error"]
    assert all(isinstance(e["time"], float) for e in h)
    assert "ok" not in s.guards
