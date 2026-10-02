"""System.calibrate: Platt scaling of a question's confidence on held-out examples."""
import json
import random

import pytest

from solvi import Answer, Catalog, Question, System


def _rule_system(**kw):
    cat = Catalog()

    @cat.rule("q")
    def q(x):
        return x > 5
    return System(cat, [Question("q", "?", Answer.yes_no())], **kw)


def _held_out(p_right, seed, n=300):
    rnd = random.Random(seed)
    held = [{"x": rnd.uniform(0, 10)} for _ in range(n)]
    truth = ["yes" if (st["x"] > 5) ^ (rnd.random() > p_right) else "no" for st in held]
    acc = sum((st["x"] > 5) == (t == "yes") for st, t in zip(held, truth)) / n
    return held, truth, acc


@pytest.mark.parametrize("p_right", [0.6, 0.75, 0.9])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_calibrating_a_rule_answer_gives_the_share_of_right_answers_as_its_confidence(p_right, seed):
    s = _rule_system()
    held, truth, acc = _held_out(p_right, seed)
    s.calibrate("q", list(zip(held, truth)))
    assert s.ask({"x": 9})["q"].confidence == pytest.approx(acc, abs=1e-3)


def test_calibrating_twice_on_the_same_examples_gives_the_same_parameters():
    s = _rule_system()
    held, truth, _ = _held_out(0.7, 4)
    first = s.calibrate("q", list(zip(held, truth)))
    assert s.calibrate("q", list(zip(held, truth))) == pytest.approx(first)


def test_calibrate_does_not_store_or_count_its_held_out_examples(tmp_path):
    s = _rule_system(storage=str(tmp_path / "j.jsonl"))
    held, truth, _ = _held_out(0.7, 5, n=40)
    s.calibrate("q", list(zip(held, truth)))
    assert s.stats["asks"] == 0
    assert not (tmp_path / "j.jsonl").exists() or not (tmp_path / "j.jsonl").read_text().strip()
    s.ask({"x": 1})
    assert s.stats["asks"] == 1
    assert len([json.loads(x) for x in (tmp_path / "j.jsonl").read_text().splitlines()]) == 1


def test_calibrate_takes_true_and_false_as_the_correct_answers_of_a_yes_no_question():
    held, truth, acc = _held_out(0.8, 6)
    a, b = _rule_system().calibrate("q", list(zip(held, truth)))
    assert _rule_system().calibrate("q", [(s, t == "yes") for s, t in zip(held, truth)]) == pytest.approx((a, b))
    with pytest.raises(ValueError):
        _rule_system().calibrate("q", [(s, "maybe") for s in held])
    with pytest.raises(ValueError, match=r"examples are \[\(init_state, correct answer\)"):
        _rule_system().calibrate("q", held)


def test_calibrate_takes_examples_as_fit_does_and_the_0_7_form_with_a_warning():
    """calibrate(question, states, truth) took its examples apart while fit, learn_rule and guarantee take
    [(state, answer)]; the 0.7 form still works for one release."""
    held, truth, _ = _held_out(0.8, 7)
    new = _rule_system().calibrate("q", list(zip(held, truth)))
    with pytest.warns(DeprecationWarning, match=r"calibrate\(question, \[\(state, answer\), \.\.\.\]\)"):
        assert _rule_system().calibrate("q", held, truth) == pytest.approx(new)
    with pytest.raises(ValueError, match="300 examples and 2 correct answers"):
        _rule_system().calibrate("q", held, truth[:2])                  # warned once already


def test_calibrating_spread_confidences_reaches_the_maximum_likelihood_fit():
    np = pytest.importorskip("numpy")
    from solvi.system import _platt_fit
    rnd = np.random.default_rng(0)
    xs = rnd.uniform(0.1, 5.5, 400)
    ys = (rnd.uniform(size=400) < 1 / (1 + np.exp(-(0.4 * xs + 0.3)))).astype(float)
    a, b = _platt_fit(xs, ys)
    p = 1 / (1 + np.exp(-(a * xs + b)))
    assert abs(float(((p - ys) * xs).mean())) < 1e-6 and abs(float((p - ys).mean())) < 1e-6   # the gradient is zero
    assert 0.1 < a < 0.8


def test_calibrating_perfectly_separated_confidences_stays_finite():
    np = pytest.importorskip("numpy")
    from solvi.system import _platt_fit
    xs = np.array([0.5] * 10 + [3.0] * 10)
    ys = np.array([0.0] * 10 + [1.0] * 10)
    a, b = _platt_fit(xs, ys)
    assert abs(a) <= 1e3 and abs(b) <= 1e3
