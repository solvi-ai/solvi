"""build(novel=[...]): the open-set gate calibrated on real out-of-scope inputs instead of the leave-options-out
simulation — also where the simulation cannot run (a yes/no answer); too few or in-scope ones are warned about."""
import warnings

import pytest
from test_auto import FILLER, WORDS, Q, _catalog, _learner, _pairs, _texts

from solvi.core.guarantees.guarantee import GuaranteeWarning
from solvi.solutions.decisions import build


def _outside(n, seed=3):
    import random
    rng = random.Random(seed)
    return [{"text": " ".join(rng.choice(FILLER) for _ in range(rng.randint(1, 4)))} for _ in range(n)]


def test_real_outside_examples_calibrate_the_gate():
    ex = _texts(1600, WORDS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GuaranteeWarning)
        s = build("topic", ex, learner=_learner, max_error=0.10, novel=_outside(120))
    assert s.gate is not None and "openset" in s.calibration
    assert "120 real out-of-scope" in s.explain() and "refitted 3 times" not in s.explain()


def test_a_yes_no_answer_takes_real_outside_examples_too():
    import random
    rng = random.Random(5)
    far = [{"x": round(0.5 + rng.uniform(-0.02, 0.02), 4)} for _ in range(120)]   # inputs the head cannot tell
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = build(Q, _pairs(800), catalog=_catalog(), max_risk=0.05, novel=far)
    assert s.gate is not None


def test_few_or_in_scope_outside_examples_are_warned_about_and_too_few_refused():
    ex = _texts(1600, WORDS)
    with pytest.warns(GuaranteeWarning, match="about 100"):
        build("topic", ex, learner=_learner, max_error=0.10, novel=_outside(60))
    with pytest.warns(GuaranteeWarning, match="also labelled examples"):
        build("topic", ex, learner=_learner, max_error=0.10, novel=_outside(110) + [st for st, _ in ex[:10]])
    with pytest.raises(ValueError, match="at least 30"):
        build("topic", ex, learner=_learner, max_error=0.10, novel=_outside(10))
    with pytest.raises(ValueError, match="novel"):
        build("topic", ex, learner=_learner, max_error=0.10, novel=["just text"])
