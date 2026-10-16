"""System.guard_history: every calibration attempt of System.guarantee is logged — installed (and whether it replaced
one), failed (with the error), removed — so a guarantee replaced or lost along the way can be traced. A failed attempt
keeps the guarantee in force before it, with a warning."""
import warnings

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.guarantees.guarantee import GuaranteeWarning


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
    with pytest.raises(ValueError), pytest.warns(GuaranteeWarning, match="stays in force"):
        s.guarantee("ok", _ex(300), max_risk=0.1, signal="nowhere")
    s.guarantee("ok", False)
    h = s.guard_history
    assert [e["outcome"] for e in h] == ["installed", "installed", "failed", "removed"]
    assert [e["replaced"] for e in h] == [False, True, True, True]    # the failed attempt kept the one in force
    assert h[2]["kept"] is True
    assert h[0]["question"] == "ok" and h[0]["n"] == 400 and h[0]["threshold"] is not None and h[0]["promise"]
    assert h[1]["n"] == 300 and "ValueError" in h[2]["error"]
    assert all(isinstance(e["time"], float) for e in h)
    assert "ok" not in s.guards


def test_a_failed_attempt_keeps_the_guarantee_in_force():
    s = _system()
    s.guarantee("ok", _ex(400), max_risk=0.1, signal="score")
    guard, question = s.guards["ok"], s.questions["ok"]
    states = [{"x": i / 50} for i in range(50)]
    before = [(r["ok"].answer, r["ok"].status) for r in (s.ask(st, store=False) for st in states)]
    for bad in ({"signal": "nowhere"}, {"max_risk": 0.1, "sources": ("human",)}):     # one fails calibrating, one before
        with pytest.raises(ValueError), pytest.warns(GuaranteeWarning, match="stays in force"):
            s.guarantee("ok", _ex(300), **({"max_risk": 0.1} | bad))
        assert s.guards["ok"] is guard and s.questions["ok"] == question and guard.added == ["score"]
        assert [(r["ok"].answer, r["ok"].status) for r in (s.ask(st, store=False) for st in states)] == before
    assert [e["outcome"] for e in s.guard_history] == ["installed", "failed", "failed"]
    assert all(e["kept"] for e in s.guard_history[1:])


def test_a_failed_first_attempt_leaves_no_guarantee_and_does_not_warn():
    s = _system()
    question = s.questions["ok"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            s.guarantee("ok", None, max_risk=0.1, signal="score")      # no examples: fails after adding the checkpoint
    assert "ok" not in s.guards and s.questions["ok"] == question
    assert s.guard_history[-1]["outcome"] == "failed" and s.guard_history[-1]["kept"] is False
