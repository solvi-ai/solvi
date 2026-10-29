"""Known-answer cross-checks of the guarantee arithmetic: learn-then-test (`ltt_certified`, `ltt_threshold`), conformal
risk control (`crc_threshold`) and the split-conformal quantile (`conformal_quantile`), each against a value computed
independently of solvi's code — exact fractions, a closed form from the paper that defines the procedure, or scipy.

Sources of the expected values (each test names its own):
- [LTT] Angelopoulos, Bates, Candès, Jordan, Lei (2021), "Learn then Test: Calibrating Predictive Algorithms to Achieve
  Risk Control", arXiv 2110.01052: a threshold is certified when the null "risk > alpha" is rejected; for a 0/1 loss the
  exact binomial-tail p-value is P(Binomial(n, alpha) <= n * empirical risk); family-wise error by Bonferroni or by a
  fixed sequence tested in a pre-set order.
- [CP] Clopper & Pearson (1934): the one-sided (1 - a) upper bound of a binomial share with m successes of k is the
  (1 - a) quantile of Beta(m + 1, k - m); "P(Bin(k, e) <= m) <= a" holds exactly when e >= that bound.
- [CRC] Angelopoulos, Bates, Fisch, Lei, Schuster (2022), "Conformal Risk Control", arXiv 2208.02814, Thm. 1:
  lambda_hat = inf{lambda : n / (n + 1) * R_n(lambda) + B / (n + 1) <= alpha}, B = 1 for a 0/1 loss.
- [SC] Angelopoulos & Bates (2021), "A Gentle Introduction to Conformal Prediction", arXiv 2107.07511, eq. (1): q_hat is
  the ceil((n + 1)(1 - alpha)) / n empirical quantile of the calibration scores; for exchangeable, almost surely
  distinct scores the coverage is exactly ceil((n + 1)(1 - alpha)) / (n + 1) (Vovk, Gammerman, Shafer 2005; the
  Gentle Introduction, Thm. D.1 and D.2). LAC scores 1 - p_y: Sadinle, Lei, Wasserman (2019).
"""
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import beta, binom

from solvi.core.calibration import (_binom_cdf, conformal_quantile, crc_threshold, ltt_certified, ltt_grid,
                                    ltt_threshold, set_scores)


def _exact_tail(m, k, e):
    """P(Binomial(k, e) <= m) as an exact fraction (e given as a Fraction)."""
    return sum(Fraction(math.comb(k, i)) * e ** i * (1 - e) ** (k - i) for i in range(int(m) + 1))


# --- the binomial tail every LTT test uses
@pytest.mark.parametrize("m,k,e", [(2, 10, "1/2"), (0, 22, "1/10"), (0, 21, "1/10"), (3, 50, "1/10"), (10, 200, "1/20"),
                                   (7, 7, "3/10"), (0, 1, "1/100")])
def test_binomial_tail_matches_exact_fractions(m, k, e):
    """[LTT] binomial-tail p-value: solvi's log-space sum against the exact rational sum and scipy's binom.cdf."""
    want = float(_exact_tail(m, k, Fraction(e)))
    assert _binom_cdf(m, k, float(Fraction(e))) == pytest.approx(want, rel=1e-12, abs=1e-15)
    assert binom.cdf(m, k, float(Fraction(e))) == pytest.approx(want, rel=1e-9, abs=1e-15)


def test_binomial_tail_hand_values():
    # P(Bin(10, 1/2) <= 2) = (1 + 10 + 45) / 1024 = 56 / 1024; P(Bin(22, 0.1) = 0) = 0.9^22
    assert _binom_cdf(2, 10, 0.5) == pytest.approx(56 / 1024, rel=1e-12)
    assert _binom_cdf(0, 22, 0.1) == pytest.approx(0.9 ** 22, rel=1e-12)


# --- learn-then-test
def test_fewest_answers_that_can_be_certified():
    """[LTT] with no wrong answer the p-value is (1 - error)^k: certified at delta iff k >= ln(delta) / ln(1 - error).
    error 0.1, delta 0.1: ln 0.1 / ln 0.9 = 21.85, so 22 answers pass (0.9^22 = 0.0985) and 21 do not (0.1094)."""
    assert math.log(0.1) / math.log(0.9) == pytest.approx(21.854, abs=1e-3)
    for proc in ("bonferroni", "fixed"):        # "fallback" starts at (1 - fallback) * delta: a later start, by design
        assert ltt_certified([22], [0], 0.1, 0.1, proc).tolist() == [True], proc
        assert ltt_certified([21], [0], 0.1, 0.1, proc).tolist() == [False], proc
    # error 0.05, delta 0.05: ln 0.05 / ln 0.95 = 58.40 → 59 answers
    assert ltt_certified([59], [0], 0.05, 0.05).tolist() == [True]
    assert ltt_certified([58], [0], 0.05, 0.05).tolist() == [False]


def test_one_wrong_answer_hand_value():
    """[LTT] k = 38 answered, 1 wrong, error 0.1: p = 0.9^38 + 38 * 0.1 * 0.9^37 = 0.0953 ≤ 0.1; k = 37 gives 0.1036."""
    p38 = 0.9 ** 38 + 38 * 0.1 * 0.9 ** 37
    p37 = 0.9 ** 37 + 37 * 0.1 * 0.9 ** 36
    assert p38 == pytest.approx(0.09530, abs=1e-5) and p37 == pytest.approx(0.10363, abs=1e-5)
    assert ltt_certified([38], [1], 0.1, 0.1).tolist() == [True]
    assert ltt_certified([37], [1], 0.1, 0.1).tolist() == [False]


def _counts(seed, n=400):
    rng = np.random.default_rng(seed)
    s = rng.uniform(size=n)
    w = rng.uniform(size=n) < 0.25 * (1 - s) ** 2
    grid = np.sort(ltt_grid(s, 24))
    k = np.array([int((s >= t).sum()) for t in grid])
    m = np.array([int(w[s >= t].sum()) for t in grid])
    return s, w, grid, k, m


@pytest.mark.parametrize("seed", range(6))
def test_bonferroni_equals_clopper_pearson_rule(seed):
    """[CP] + [LTT] Bonferroni over G thresholds certifies threshold i iff the one-sided Clopper–Pearson upper bound at
    level delta / G of its error is at most `error` — the same test written through the Beta quantile instead of the
    binomial sum (cases within 1e-9 of the boundary are skipped)."""
    _, _, grid, k, m = _counts(seed)
    err, delta = 0.05, 0.1
    a = delta / len(grid)
    got = ltt_certified(k, m, err, delta, "bonferroni")
    for i in range(len(grid)):
        if abs(binom.cdf(m[i], k[i], err) - a) < 1e-9:
            continue
        upper = 1.0 if m[i] >= k[i] else beta.ppf(1 - a, m[i] + 1, k[i] - m[i])
        assert bool(got[i]) == (k[i] >= 1 and upper <= err), (i, k[i], m[i], upper)


def _ltt_reference(s, w, grid, err, delta, procedure):
    """[LTT] the procedure written from the paper with exact rational p-values: Bonferroni — every threshold at
    delta / G; fixed sequence — from the highest threshold down at the full delta, starting at the first threshold with
    at least k0 = ceil(ln delta / ln(1 - err)) answered (a choice from the counts only, solvi's documented start), stopping
    at the first failure. → the lowest certified threshold, inf when none."""
    e = Fraction(err).limit_denominator(10 ** 6)
    ks = [int((s >= t).sum()) for t in grid]
    ms = [int(w[s >= t].sum()) for t in grid]
    ok = [False] * len(grid)
    if procedure == "bonferroni":
        lvl = Fraction(delta).limit_denominator(10 ** 6) / len(grid)
        ok = [k >= 1 and _exact_tail(m, k, e) <= lvl for k, m in zip(ks, ms)]
    else:
        k0 = max(1, math.ceil(math.log(delta) / math.log1p(-err) - 1e-9))
        lvl, started = Fraction(delta).limit_denominator(10 ** 6), False
        for i in range(len(grid) - 1, -1, -1):
            if not started and ks[i] < k0:
                continue
            started = True
            if _exact_tail(ms[i], ks[i], e) <= lvl:
                ok[i] = True
            else:
                break
    hit = [g for g, o in zip(grid, ok) if o]
    return float(hit[0]) if hit else math.inf


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("procedure", ["bonferroni", "fixed"])
def test_ltt_threshold_equals_the_paper_procedure(seed, procedure):
    s, w, grid, _, _ = _counts(seed, 300)
    for err, delta in ((0.05, 0.1), (0.1, 0.05), (0.02, 0.2)):
        assert ltt_threshold(s, w, err, delta, grid=grid, procedure=procedure) == \
            _ltt_reference(s, w, grid, err, delta, procedure), (err, delta)


# --- conformal risk control
def _crc_reference(s, w, alpha):
    """[CRC] Thm. 1 with B = 1: the smallest candidate t (the scores, then +inf) with (Σ L_i(t) + 1) / (n + 1) <= alpha,
    L_i(t) = 1[s_i >= t and wrong_i], in exact fractions (the loss falls as t rises, so the first that holds is the inf)."""
    n = len(s)
    for t in list(np.unique(s)) + [math.inf]:
        loss = sum(1 for x, y in zip(s, w) if x >= t and y)
        if Fraction(loss + 1, n + 1) <= Fraction(alpha).limit_denominator(10 ** 6):
            return float(t)
    return math.inf


def test_crc_hand_values():
    """[CRC] n = 9, alpha = 0.1: (L + 1) / 10 <= 0.1 needs L = 0 — the threshold sits just above the highest wrong
    score; n = 8: 1 / 9 > 0.1 even with no loss → inf (too few examples)."""
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    w = [1, 0, 1, 0, 0, 1, 0, 0, 0]                  # the highest wrong score is 0.6
    assert crc_threshold(s, w, 0.1) == 0.7
    assert crc_threshold(s[:8], w[:8], 0.1) == math.inf
    # alpha 0.3: (L + 1) / 10 <= 0.3 allows L = 2 → the lowest t keeping at most two wrong answered is 0.2 (wrong: 0.3, 0.6)
    assert crc_threshold(s, w, 0.3) == 0.2


@pytest.mark.parametrize("seed", range(6))
def test_crc_equals_the_theorem(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(20, 200))
    s = np.round(rng.uniform(size=n), 2)          # ties on purpose
    w = rng.uniform(size=n) < 0.4 * (1 - s)
    for alpha in (0.05, 0.1, 0.2):
        assert crc_threshold(s, w, alpha) == _crc_reference(s, w, alpha), alpha


# --- the split-conformal quantile
@pytest.mark.parametrize("n,alpha,rank", [(9, 0.1, 9), (19, 0.1, 18), (99, 0.05, 95), (1000, 0.1, 901), (4, 0.2, 4),
                                          (8, 0.1, None), (18, 0.05, None)])
def test_conformal_quantile_textbook_ranks(n, alpha, rank):
    """[SC] the ceil((n + 1)(1 - alpha))-th smallest of n scores; inf when that rank exceeds n (n < 1 / alpha - 1)."""
    scores = np.random.default_rng(n).permutation(np.arange(1, n + 1)).astype(float)   # the k-th smallest is k
    q = conformal_quantile(scores, alpha)
    assert q == (math.inf if rank is None else float(rank))


@pytest.mark.parametrize("n,alpha", [(19, 0.1), (49, 0.2), (9, 0.1)])
def test_conformal_coverage_is_the_exact_rank_share(n, alpha):
    """[SC] continuous i.i.d. scores: P(S_test <= q_hat) = k / (n + 1), k = ceil((n + 1)(1 - alpha)) — checked by 40,000
    seeded draws, within 4.5 standard errors."""
    rng = np.random.default_rng(7)
    trials = 40_000
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    cal = rng.uniform(size=(trials, n))
    test = rng.uniform(size=trials)
    q = np.array([conformal_quantile(row, alpha) for row in cal])
    cover = float((test <= q).mean())
    want = k / (n + 1)
    assert want >= 1 - alpha
    assert abs(cover - want) < 4.5 * math.sqrt(want * (1 - want) / trials)


def test_lac_scores_are_one_minus_p():
    """[SC] LAC: the non-conformity of an answer is 1 - its probability."""
    p = np.array([0.7, 0.2, 0.1])
    assert np.allclose(set_scores(p), [0.3, 0.8, 0.9])
