"""The arithmetic of a numeric series, which is not solvi (numpy only): a robust scale, the series laid on a regular
time grid, and — vectorised over a whole series — the score of every point, which the solution gives each decision as
the series' memory of its earlier scores. Every value at point i reads points before i only.

  z_level   (x − trailing median) / trailing robust scale
  z_day     (x − median of the same time of day over the previous ≤ 7 days) / robust scale of that residual
  seasonal  1 − scale(day residual) / scale(level residual): how much a daily profile explains (≤ 0: nothing)
  score     |z_day| where the daily profile explains the series (seasonal > 0.3), else |z_level|
"""
import calendar
import warnings
from datetime import datetime

import numpy as np

W = 576            # trailing window, points
K_DAYS = 7         # daily profile: the same time of day over up to this many previous days
MIN_HIST = 48      # facts need this many trailing points
EXPLAINS = 0.3     # the daily profile is used where it removes more than this share of the trailing scale


def robust_scale(v, axis=-1):
    """A scale that survives old anomalies and zero-heavy series: the largest of 1.4826·MAD, the 5–95% range / 3.29 and
    sd / 3 (the first is 0 on count series that are mostly 0, the last alone is inflated by every past spike)."""
    med = np.nanmedian(v, axis=axis, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(v - med), axis=axis)
    q = np.nanquantile(v, [0.05, 0.95], axis=axis)
    return np.maximum(np.maximum(mad, (q[1] - q[0]) / 3.29), np.nanstd(v, axis=axis) / 3.0)


def floor_scale(scale, med):
    """A constant series has scale 0: any change is then 'infinitely' unusual. Floor the scale at 1e-3 of the level."""
    return np.maximum(scale, 1e-3 * (np.abs(med) + 1.0))


def grid(ts, xs):
    """(position of each point on the regular grid, points per day, the series laid on the grid with NaN for gaps)"""
    x = np.asarray(xs, float)
    t = np.array([calendar.timegm(datetime.fromisoformat(s[:19]).timetuple()) for s in ts], dtype=np.int64)
    dt = int(np.median(np.diff(t))) or 1
    g = np.round((t - t[0]) / dt).astype(int)
    dense = np.full(g[-1] + 1, np.nan)
    dense[g] = x
    return g, max(1, int(round(86400 / dt))), dense


def _trailing(v, w, fn, chunk=2000):
    """fn over the w values before each index (fewer near the start, NaN with under MIN_HIST non-NaN values)."""
    n = len(v)
    out = np.full(n, np.nan)
    view = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(w, np.nan), v]), w)[:n]   # view[i] = v[i-w:i]
    for a in range(0, n, chunk):
        blk = view[a:a + chunk]
        ok = (~np.isnan(blk)).sum(1) >= MIN_HIST
        if ok.any():
            out[a:a + chunk][ok] = fn(blk[ok])
    return out


def series_scores(ts, xs):
    """→ (the expected value of each point from the same time on earlier days, the score of each point)."""
    x = np.asarray(xs, float)
    g, day, dense = grid(ts, xs)
    lags = np.stack([np.where(g - day * k >= 0, dense[np.clip(g - day * k, 0, None)], np.nan) for k in range(1, K_DAYS + 1)], 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = _trailing(x, W, lambda b: np.nanmedian(b, 1))
        sc_level = floor_scale(_trailing(x, W, robust_scale), med)
        expected = np.where((~np.isnan(lags)).sum(1) >= 2, np.nanmedian(lags, 1), np.nan)
        sc_day = _trailing(x - expected, W, robust_scale)
        z_level, z_day = (x - med) / sc_level, (x - expected) / floor_scale(sc_day, med)
        seasonal = 1.0 - sc_day / sc_level
    nz = lambda a: np.nan_to_num(np.abs(a), nan=0.0, posinf=1e9)
    use_day = (np.nan_to_num(seasonal, nan=-1.0) > EXPLAINS) & ~np.isnan(z_day)
    return expected, np.where(use_day, nz(z_day), nz(z_level))
