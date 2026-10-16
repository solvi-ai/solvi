"""solvi.experimental.ood: an out-of-domain watch next to the default Platt calibration ("out of domain" in why for an
answer unlike the calibration answers, shift(batch) for a batch)."""
import random
import warnings

import pytest

np = pytest.importorskip("numpy")

from solvi import Answer, Catalog, Question, System  # noqa: E402
from solvi.core.catalog import Decision  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.ood import ShiftWatch, features, ks_2samp  # noqa: E402

OPTS = [f"o{i}" for i in range(10)]


def _answers(n, rng, sharp):
    """Answers over 10 options: a softmax of random logits scaled by `sharp` (larger: more confident)."""
    out = []
    for _ in range(n):
        z = rng.normal(size=10) * sharp
        p = np.exp(z - z.max())
        p /= p.sum()
        out.append({"confidence": float(p.max()), "probs": {o: float(v) for o, v in zip(OPTS, p)}})
    return out


def test_features_of_a_known_answer():
    lc, m, e = features(0.5, {"a": 0.5, "b": 0.25, "c": 0.25})
    assert lc == pytest.approx(0.0) and m == pytest.approx(np.log(2))
    assert e == pytest.approx(-(0.5 * np.log(0.5) + 0.5 * np.log(0.25)) / np.log(3))
    lc, m, e = features(0.8)                          # no options: the two-option reading
    assert m == pytest.approx(lc) and 0 < e < 1


def test_ks_matches_a_known_value():
    d, p = ks_2samp([1, 2, 3, 4], [5, 6, 7, 8])
    assert d == 1.0 and p < 0.05
    d, p = ks_2samp(list(range(100)), list(range(100)))
    assert d == 0.0 and p == 1.0


def test_shift_is_quiet_on_batches_like_the_reference_and_fires_on_a_shifted_one():
    rng = np.random.default_rng(0)
    w = ShiftWatch().fit(_answers(800, rng, 3.0))
    quiet = sum(w.shift(_answers(200, rng, 3.0))["shifted"] for _ in range(40))
    assert quiet <= 3                                  # alpha = 0.01, Bonferroni: rare false alarms
    s = w.shift(_answers(200, rng, 1.0))
    assert s["shifted"] and "distributions differ" in s["why"] and s["n"] == 200


def test_the_single_answer_note_flags_about_alpha_in_distribution_and_far_answers():
    rng = np.random.default_rng(1)
    w = ShiftWatch(alpha=0.05).fit(_answers(2000, rng, 3.0))
    rate_in = np.mean([bool(n) for n in w.notes_many(_answers(2000, rng, 3.0))])
    assert 0.02 <= rate_in <= 0.09
    far = {"confidence": 0.1001, "probs": {o: 0.1 for o in OPTS} | {"o0": 0.1001}}
    assert "features outside the calibration data" in w.note(far["confidence"], far["probs"])[0]


def test_fit_is_deterministic_and_the_fingerprint_follows_the_reference():
    rng = np.random.default_rng(2)
    ref = _answers(300, rng, 2.0)
    assert ShiftWatch().fit(ref).fingerprint() == ShiftWatch().fit(ref).fingerprint()
    assert ShiftWatch().fit(ref).fingerprint() != ShiftWatch().fit(ref[:-5]).fingerprint()
    assert ShiftWatch().fingerprint() == "unfitted"


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        ShiftWatch(alpha=0)
    with pytest.raises(ValueError):
        ShiftWatch().fit([{"confidence": 0.5}])
    with pytest.raises(ValueError):
        ShiftWatch().fit([{"confidence": 1.5}, {"confidence": 0.5}])
    with pytest.raises(RuntimeError):
        ShiftWatch().shift([{"confidence": 0.5}])
    w = ShiftWatch().fit([{"confidence": 0.5}, {"confidence": 0.7}, {"confidence": 0.9}])
    with pytest.raises(ValueError):
        w.shift([])


# ------------------------------------------------------------------------------------------- System.calibrate(watch=)
class _M:
    name = "fake"
    version = "1"


def _system():
    cat = Catalog()

    def q(sharp, seed):
        r = np.random.default_rng(int(seed))
        z = r.normal(size=10) * sharp
        p = np.exp(z - z.max())
        p /= p.sum()
        return Decision(OPTS[int(p.argmax())], {o: float(v) for o, v in zip(OPTS, p)})
    cat.rule("q", model=_M())(q)
    return System(cat, [Question("q", "?", Answer.choice(OPTS))])


def _examples(n, sharp, seed):
    rnd = random.Random(seed)
    return [({"sharp": sharp, "seed": rnd.randrange(10**9)}, rnd.choice(OPTS)) for _ in range(n)]


def test_calibrate_with_a_watch_keeps_the_platt_map_and_notes_far_answers():
    plain, watched = _system(), _system()
    ex = _examples(300, 3.0, 0)
    ab = plain.calibrate("q", ex)
    assert watched.calibrate("q", ex, watch=ShiftWatch(alpha=0.05)) == pytest.approx(ab)
    st = {"sharp": 3.0, "seed": 12345}
    a, b = plain.ask(st)["q"], watched.ask(st)["q"]
    assert a.confidence == pytest.approx(b.confidence) and a.answer == b.answer
    raw = b.extra["out_of_domain"]["raw"]
    assert 0 < raw <= 1 and raw != b.confidence
    far = [watched.ask({"sharp": 0.05, "seed": s})["q"] for s in range(20)]
    assert all("out of domain: features outside the calibration data" in r.why for r in far)
    batch = [watched.ask({"sharp": 0.5, "seed": s})["q"] for s in range(200)]
    assert watched.shift_watch["q"].shift(batch)["shifted"]                # raw confidences from extra
    near = [watched.ask({"sharp": 3.0, "seed": 10**6 + s})["q"] for s in range(200)]
    assert not watched.shift_watch["q"].shift(near)["shifted"]


def test_a_watch_marks_the_system_experimental_enters_the_fingerprint_and_can_be_removed():
    from solvi.core.system import experimental_of
    s = _system()
    ex = _examples(100, 2.0, 1)
    s.calibrate("q", ex)
    fp0 = s._questions_fp()
    assert experimental_of(s) == [] and "out_of_domain" not in (s.ask({"sharp": 2.0, "seed": 1})["q"].extra or {})
    s.calibrate("q", ex, watch=ShiftWatch())
    fp1 = s._questions_fp()
    assert experimental_of(s) == ["ood"] and fp1 != fp0
    s.calibrate("q", ex)                                                  # no watch=: the watch is kept
    assert s._questions_fp() == fp1
    s.calibrate("q", ex, watch=False)
    assert s._questions_fp() == fp0 and experimental_of(s) == [] and not s.shift_watch
