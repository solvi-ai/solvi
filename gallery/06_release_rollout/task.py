"""Release rollout: canary metrics against the baseline -> promote / hold / roll_back, and page the on-call?

The numbers are computed from raw counts: error rates, the relative error increase (%), a two-proportion z-test that says
whether the increase is more than noise, p95 latency from the latency histograms (Prometheus-style cumulative buckets, linear
interpolation), the latency increase (%) and the SLO burn rate (how many times faster than allowed the canary spends the
error budget). Hard checks run first: an absolute error rate above the ceiling rolls back and pages at once (the histogram
and z-test work is skipped: early exit); too little canary traffic holds. Missing histograms make `decision` abstain.
Try: raise the canary errors to 300, or cut canary requests to 800."""
import math

from solvi import Answer, Catalog, Question

cat = Catalog()


def _p95(buckets):
    """95th percentile from cumulative {"upper bound ms": count} buckets, as histogram_quantile does"""
    edges = sorted(((math.inf if k == "+Inf" else float(k)), n) for k, n in buckets.items())
    total, rank, lo, below = edges[-1][1], 0.95 * edges[-1][1], 0.0, 0
    for hi, n in edges:
        if n >= rank:
            return round(lo if hi == math.inf else lo + (hi - lo) * (rank - below) / max(n - below, 1), 1)
        lo, below = hi, n
    raise ValueError(f"empty histogram ({total} requests)")


# ---------- computed facts
@cat.fn
def baseline_error_pct(baseline):
    return round(100 * baseline["errors"] / baseline["requests"], 3)


@cat.fn
def canary_error_pct(canary):
    return round(100 * canary["errors"] / canary["requests"], 3)


@cat.fn
def error_increase_pct(baseline_error_pct, canary_error_pct):
    return round((canary_error_pct - baseline_error_pct) / max(baseline_error_pct, 0.01) * 100, 1)


@cat.fn
def error_z(baseline, canary):
    """two-proportion z-score of the canary error rate against the baseline"""
    n1, n2, x1, x2 = baseline["requests"], canary["requests"], baseline["errors"], canary["errors"]
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-9
    return round((x2 / n2 - x1 / n1) / se, 2)


@cat.fn
def baseline_p95_ms(baseline):
    return _p95(baseline["latency_buckets"])


@cat.fn
def canary_p95_ms(canary):
    return _p95(canary["latency_buckets"])


@cat.fn
def latency_increase_pct(baseline_p95_ms, canary_p95_ms):
    return round((canary_p95_ms - baseline_p95_ms) / baseline_p95_ms * 100, 1)


@cat.fn
def slo_burn_rate(canary_error_pct, slo):
    """1.0 = the canary spends the error budget exactly as fast as the SLO allows"""
    return round(canary_error_pct / (100 - slo["availability_pct"]), 2)


# ---------- hard checks (the first failing one by name decides: the ceiling before traffic)
@cat.check(hard=True, then={"decision": "roll_back", "page_oncall": "yes"})
def canary_errors_under_ceiling(canary_error_pct, max_error_pct):
    return canary_error_pct <= max_error_pct


@cat.check(hard=True, then={"decision": "hold"})
def enough_canary_traffic(canary, min_canary_requests):
    return canary["requests"] >= min_canary_requests


# ---------- answers
@cat.rule("decision")
def decision(error_increase_pct, error_z, canary_p95_ms, latency_increase_pct, slo_burn_rate, slo):
    if (error_increase_pct > 50 and error_z > 3) or canary_p95_ms > slo["p95_ms"] or slo_burn_rate >= 10:
        return "roll_back"
    if error_increase_pct > 20 or latency_increase_pct > 15 or slo_burn_rate >= 1:
        return "hold"
    return "promote"


@cat.rule("page_oncall")
def page_oncall(slo_burn_rate):
    return slo_burn_rate >= 3


QUESTIONS = [
    Question("decision", "Promote the canary, hold it where it is, or roll back?", Answer.choice(["promote", "hold", "roll_back"]),
             checkpoints=["canary_errors_under_ceiling", "enough_canary_traffic"]),
    Question("page_oncall", "Page the on-call engineer now?", Answer.yes_no(), checkpoints=["canary_errors_under_ceiling"]),
]
