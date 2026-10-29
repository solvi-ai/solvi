"""solvi.experimental.laws.residual_check: an opt-in `confirm=` closure that keeps a fitted numeric law only while the
χ² of its residuals after a refit on all rows is below the level quantile at n − k. Synthetic worlds: the law is
known."""
from __future__ import annotations

import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition, residual_check

SIGMA = 1.0
KW = dict(min_support=5, min_precision=0.8, sized=True, delta=0.01, null=0.5, space=4)


def world(f, design_x, held_x, seed=0):
    """Rows y = f(x) + N(0, σ²): a design group and a held-out group → (grammar, buffer, rows)."""
    rng = random.Random(seed)
    g = Grammar([Atom("lab", lambda s: True, feature="lab")], k=1)
    buf = Buffer(g)
    rows = []
    for group, xs in (("design", design_x), ("held", held_x)):
        for x in xs:
            y = f(x) + rng.gauss(0.0, SIGMA)
            rows.append((x, y))
            buf.add(Transition(x, "y", y, group=group))
    return g, buf, rows


def slope_refit(rows):
    """The law y = c·x refitted on every row (least squares through 0) → (residuals in σ, k = 1)."""
    def refit(candidate):
        c = sum(x * y for x, y in rows) / sum(x * x for x, _ in rows)
        return [(y - c * x) / SIGMA for x, y in rows], 1
    return refit


def biased():
    """Truth y = 2x + 2 (σ = 1) on x in [0.1, 4.1], 30 design and 30 held-out rows; the list has only y = c·x."""
    return world(lambda x: 2.0 * x + 2.0, [0.1 + i * 4 / 30 for i in range(30)],
                 [0.1 + 2 / 30 + i * 4 / 30 for i in range(30)])


def design_slope(rows):
    d = rows[:30]
    return sum(x * y for x, y in d) / sum(x * x for x, _ in d)


def lawset(g, tol, confirm=None):
    return LawSet(g, apply=lambda e, s: e * s, tol=tol, confirm=confirm, **KW)


def test_default_behaviour_unchanged():
    # nothing is checked unless the closure is passed: the row test alone, as before (it accepts a biased law)
    g, buf, rows = biased()
    c = Candidate("y", ("lab",), design_slope(rows))
    assert lawset(g, tol=4.0).confirm is None
    assert len(lawset(g, tol=4.0).test([c], buf, exclude="design")) == 1
    check = residual_check(slope_refit(rows))
    assert check.level == 0.995 and residual_check(slope_refit(rows), level=0.99).level == 0.99


def test_biased_wrong_law_inside_4_sigma_is_refused():
    # the list has only y = c·x: after the best slope the bias is up to ~1.9σ, every held-out row inside a 4σ band
    g, buf, rows = biased()
    c = Candidate("y", ("lab",), design_slope(rows))
    sup, hits, _ = lawset(g, tol=4.0).score(c, buf, exclude="design")
    assert sup == 30 and hits == 30                                        # every held-out row inside 4σ
    assert lawset(g, tol=4.0).test([c], buf, exclude="design")             # the row rule accepts the wrong law
    check = residual_check(slope_refit(rows))
    assert not check(c)
    assert lawset(g, tol=math.inf, confirm=check).test([c], buf, exclude="design") == []
    assert lawset(g, tol=4.0, confirm=check).test([c], buf, exclude="design") == []


def test_refit_true_law_refused_per_row_is_accepted():
    # truth y = 2x; the constant fitted on a short design window near 0 is off by 0.1, so at x ≈ 100 every held-out
    # prediction is ~10σ out and the row rule refuses the true law; refitted on all rows it fits within the noise
    g, buf, rows = world(lambda x: 2.0 * x, [0.1 + i / 100 for i in range(10)], [100.0 + i for i in range(14)], seed=1)
    c = Candidate("y", ("lab",), 2.1)
    sup, hits, _ = lawset(g, tol=4.0).score(c, buf, exclude="design")
    assert sup == 14 and hits == 0
    assert lawset(g, tol=4.0).test([c], buf, exclude="design") == []
    check = residual_check(slope_refit(rows))
    assert check(c)
    laws = lawset(g, tol=math.inf, confirm=check).test([c], buf, exclude="design")
    assert len(laws) == 1 and laws[0].effect == 2.1


def test_refusals_on_degenerate_refits():
    check = residual_check(lambda c: None)
    assert not check("any")                                                # cannot be fitted
    assert not residual_check(lambda c: ([0.1, 0.2], 2))("any")            # n − k = 0
    assert not residual_check(lambda c: ([0.1, float("nan"), 0.3], 1))("any")
    assert residual_check(lambda c: ([0.1, -0.2, 0.3], 1))("any")


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, float("nan")])
def test_invalid_level_raises(level):
    with pytest.raises(ValueError):
        residual_check(lambda c: ([0.0], 0), level=level)
