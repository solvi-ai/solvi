"""Milliseconds per `ask` on small and large inputs: what the README's Speed table is made from.

    uv run --with numpy python benchmarks/ask_speed.py              # ~1 minute
    uv run --with numpy python benchmarks/ask_speed.py --quick      # fewer passes
    uv run --with numpy python benchmarks/ask_speed.py --json out.json

The cases, none with a model (what is measured is solvi's own work: planning, running the parts, hashing the trace):
- **quickstart**: the README's leave request (5 steps, a few small values);
- **large input**: a stream alert on one point of a numeric series, the catalog of the NAB anomaly task (level, daily
  profile, score, conformal threshold, a hard check, a rule; 7 steps) over an input of 3,712 floats — the 576 points
  before it, their 576 residuals against a daily profile, the series' last 2,553 scores and 7 earlier days' values;
- **catalog N parts**: `strategist_scale.py`'s random layered catalog (arithmetic parts, 5 questions), 20 integer inputs;
- **gallery**: every gallery entry asked with its own cases (rules, checks, learned heads; one entry checks its texts
  for instruction-like sentences, which dominates its time).

Each case is asked once to warm up, then `--reps` passes round robin; the table gives the median and p90 per ask.
Numbers vary with the machine; compare rows within one run, or runs of two versions on the same machine
(PYTHONPATH=<other src> runs this file against another solvi)."""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def quickstart():
    from solvi import Answer, Catalog, Question, System
    cat = Catalog()

    @cat.fn
    def days_requested(start, end):
        return (end - start).days + 1

    @cat.fn
    def remaining_after(balance, days_requested):
        return balance - days_requested

    @cat.check(hard=True, then={"approve": "reject"})
    def enough_balance(remaining_after):
        return remaining_after >= 0

    @cat.check
    def enough_notice(start, today, days_requested):
        return days_requested < 5 or (start - today).days >= 14

    @cat.rule("approve")
    def approve(enough_notice):
        return "approve" if enough_notice else "needs_manager"

    system = System(cat, [Question("approve", "Approve the leave?",
                                   Answer.choice(["approve", "needs_manager", "reject"]),
                                   requires=["enough_balance"])])
    states = [{"start": date(2026, 10, 19), "end": date(2026, 10, 23), "today": date(2026, 9, 25), "balance": b}
              for b in (14, 3, 30, 5)]
    return system, states


def _robust_scale(v):
    import numpy as np
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    q = np.nanquantile(v, [0.05, 0.95])
    return max(mad, (q[1] - q[0]) / 3.29, np.nanstd(v) / 3.0)


def large_input(n_states=8, seed=0):
    """The NAB task's catalog over a synthetic series with a daily cycle (288 points a day)."""
    import numpy as np

    from solvi import Answer, Catalog, Question, System
    from solvi.core.calibration import conformal_quantile
    W, DAY, K_DAYS, HORIZON, MIN_HIST = 576, 288, 7, 2553, 48
    cat = Catalog()

    @cat.check(hard=True, then={"alert": "no"})
    def enough_history(recent):
        return len(recent) >= MIN_HIST

    @cat.fn
    def level(value, recent):
        r = np.asarray(recent, float)
        med = float(np.median(r))
        scale = max(float(_robust_scale(r)), 1e-9)
        return {"median": med, "scale": scale, "z": (value - med) / scale}

    @cat.fn
    def day_profile(value, same_time_days, day_residuals, level):
        lags = [v for v in same_time_days if v is not None]
        res = np.array([np.nan if v is None else v for v in day_residuals], float)
        if len(lags) < 2 or int((~np.isnan(res)).sum()) < MIN_HIST:
            return {"available": False}
        expected = float(np.median(lags))
        scale = max(float(_robust_scale(res)), 1e-9)
        return {"available": True, "expected": expected, "scale": scale, "z": (value - expected) / scale,
                "explains": 1.0 - scale / level["scale"]}

    @cat.fn
    def score(level, day_profile):
        z_level = min(abs(level["z"]), 1e9)
        z_day = min(abs(day_profile["z"]), 1e9) if day_profile["available"] else None
        return z_day if z_day is not None and day_profile["explains"] > 0.3 else z_level

    @cat.fn
    def threshold(past_scores):
        return min(float(conformal_quantile(past_scores, 0.001)), 1e308)

    @cat.check
    def above_floor(score):
        return score > 3.0

    @cat.rule("alert")
    def alert(score, threshold, above_floor):
        return bool(above_floor and score > threshold)

    system = System(cat, [Question("alert", "Raise an alert at this point?", Answer.yes_no(),
                                   requires=["enough_history"])])
    rng = np.random.default_rng(seed)
    n = HORIZON + W + DAY * K_DAYS + n_states
    t = np.arange(n)
    x = 10 + 3 * np.sin(2 * np.pi * t / DAY) + rng.normal(0, 0.5, n)
    resid = x - (10 + 3 * np.sin(2 * np.pi * t / DAY))
    resid[rng.random(n) < 0.02] = np.nan                   # a few points without a daily profile
    scores = np.abs(rng.normal(0, 1, n))
    states = []
    for i in range(n - n_states, n):
        states.append({"series": "synthetic", "time": int(i), "value": float(x[i]),
                       "recent": [float(v) for v in x[i - W:i]],
                       "same_time_days": [float(x[i - DAY * k]) for k in range(1, K_DAYS + 1)],
                       "day_residuals": [None if np.isnan(v) else float(v) for v in resid[i - W:i]],
                       "past_scores": [float(v) for v in scores[i - HORIZON:i]]})
    return system, states


def catalog(n_parts):
    from strategist_scale import make_catalog

    from solvi import System
    cat, qs, state = make_catalog(n_parts)
    rng = random.Random(1)
    states = [state] + [{k: rng.randint(0, 100) for k in state} for _ in range(3)]
    return System(cat, qs), states


def gallery(root):
    from ask_overhead import gallery_jobs
    return [(f"gallery {name}", build, states) for name, build, states in gallery_jobs(root)]


def cases(root, only=None):
    out = [("quickstart", quickstart), ("large input (3.7k floats)", large_input),
           ("catalog 50 parts", lambda: catalog(50)), ("catalog 1000 parts", lambda: catalog(1000))]
    out = [(n, f()) for n, f in out]
    if only != "small":
        out += [(n, (b(), s)) for n, b, s in gallery(root)]
    return out


def measure(jobs, reps):
    """jobs: [(name, (system, states))] → {name: [ms per ask]}: one warm-up pass, then `reps` passes round robin."""
    out = {name: [] for name, _ in jobs}
    for rep in range(reps + 1):
        for name, (system, states) in jobs:
            for st in states:
                s = copy.copy(st)
                t0 = time.perf_counter()
                system.ask(s, store=False)
                if rep:
                    out[name].append((time.perf_counter() - t0) * 1000)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="fewer passes")
    p.add_argument("--reps", type=int, help="passes over each case's states (default 30; --quick: 8)")
    p.add_argument("--only", choices=["small"], help="small: without the gallery")
    p.add_argument("--json", help="write the table to this file")
    p.add_argument("--gallery", default=str(HERE.parent / "gallery"))
    a = p.parse_args(argv)
    reps = a.reps or (8 if a.quick else 30)
    import solvi
    jobs = cases(a.gallery, a.only)
    rows = []
    print(f"solvi {solvi.__version__} ({Path(solvi.__file__).parent}), Python {sys.version.split()[0]}, {reps} passes")
    print(f"  {'case':44s} {'steps':>5s} {'asks':>6s} {'median ms':>10s} {'p90 ms':>9s}")
    for name, xs in measure(jobs, reps).items():
        system, states = dict(jobs)[name]
        steps = statistics.median(len(system.ask(copy.copy(s), store=False).trace.records) for s in states)
        xs = sorted(xs)
        r = {"case": name, "steps": steps, "asks": len(xs), "median_ms": round(statistics.median(xs), 3),
             "p90_ms": round(xs[int(0.9 * (len(xs) - 1))], 3)}
        rows.append(r)
        print(f"  {name:44s} {steps:5g} {r['asks']:6d} {r['median_ms']:10.3f} {r['p90_ms']:9.3f}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"solvi": solvi.__version__, "reps": reps, "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
