"""The strategist on a complex scenario: an insurance claim desk with a large catalog — policy, medical, fraud, payout and legal
parts, several of them slow (a fraud model, registry lookups, a medical-code service). For each set of questions the strategist
generates a different plan: only the parts those questions need, the checkpoints, and checks on what gets computed. Hard checks
run first — when one fails, the expensive rest is skipped. Independent slow parts run in parallel (System(..., workers=8)).
Every plan is printed and timed against a plain script that computes everything one by one.

Run:  uv run python examples/09_strategy_at_scale.py"""
from __future__ import annotations

import time
from datetime import date

from solvi import Answer, Catalog, Question, System
from solvi.strategist import plan

cat = Catalog()
SLOW = {}                                                  # part -> simulated latency, seconds


def slow(seconds):
    def wrap(f):
        SLOW[f.__name__] = seconds
        return f
    return wrap


def _wait(name):
    time.sleep(SLOW.get(name, 0))


# ---------- policy
@cat.fn
def policy_active(policy_start, policy_end, incident_date):
    return policy_start <= incident_date <= policy_end


@cat.fn
def days_to_report(incident_date, reported_date):
    return (reported_date - incident_date).days


@cat.fn
def covered_peril(peril, policy_perils):
    return peril in policy_perils


@cat.fn
@slow(0.12)
def policy_exclusions(policy_id, claim_type):
    """lookup in the policy administration system (slow)"""
    _wait("policy_exclusions")
    return ["flood"] if policy_id.startswith("H") and claim_type == "property" else []


@cat.fn
def excluded(peril, policy_exclusions):
    return peril in policy_exclusions


# ---------- medical
@cat.fn
@slow(0.25)
def medical_codes_valid(diagnosis_codes):
    """external coding service (slow)"""
    _wait("medical_codes_valid")
    return all(c[:1].isalpha() and c[1:3].isdigit() for c in diagnosis_codes)


@cat.fn
def treatment_cost(invoice_lines):
    return round(sum(invoice_lines), 2)


@cat.fn
def cost_per_day(treatment_cost, hospital_days):
    return treatment_cost / max(1, hospital_days)


# ---------- fraud
@cat.fn
@slow(0.30)
def fraud_score(claimant_id, amount_claimed, days_to_report):
    """ML model behind an API (slow)"""
    _wait("fraud_score")
    return min(1.0, 0.1 + 0.02 * max(0, days_to_report - 14) + (0.3 if amount_claimed > 50_000 else 0.0))


@cat.fn
@slow(0.20)
def prior_claims(claimant_id):
    """claims registry lookup (slow)"""
    _wait("prior_claims")
    return {"C-17": 4, "C-02": 0}.get(claimant_id, 1)


@cat.fn
@slow(0.15)
def sanctions_hit(claimant_name):
    """sanctions list screening (slow)"""
    _wait("sanctions_hit")
    return claimant_name.lower() in {"ivan shady"}


@cat.fn
def fraud_level(fraud_score, prior_claims):
    s = fraud_score + 0.1 * prior_claims
    return "high" if s >= 0.6 else ("medium" if s >= 0.3 else "low")


# ---------- payout
@cat.fn
def deductible_applied(amount_claimed, deductible):
    return max(0.0, amount_claimed - deductible)


@cat.fn
def payable(deductible_applied, coverage_limit):
    return min(deductible_applied, coverage_limit)


@cat.fn
def payable_medical(treatment_cost, deductible, coverage_limit):
    return min(max(0.0, treatment_cost - deductible), coverage_limit)


# ---------- legal / reporting
@cat.fn
@slow(0.10)
def litigation_open(claimant_id):
    """legal case system (slow)"""
    _wait("litigation_open")
    return claimant_id == "C-17"


@cat.fn
def regulator_report_needed(payable, sanctions_hit):
    return payable > 100_000 or sanctions_hit


# ---------- checks
@cat.check(hard=True, then={"decision": "deny", "fast_track": "no"})
def policy_in_force(policy_active):
    """hard: no active policy on the incident date -> deny, nothing else matters"""
    return policy_active


@cat.check(hard=True, then={"decision": "deny"})
def not_sanctioned(sanctions_hit):
    return not sanctions_hit


@cat.check
def reported_in_time(days_to_report):
    return days_to_report <= 30


@cat.check
def cost_plausible(cost_per_day):
    return cost_per_day <= 5_000


# ---------- answers
@cat.rule("decision")
def decision(covered_peril, excluded, reported_in_time, fraud_level):
    if not covered_peril or excluded:
        return "deny"
    if fraud_level == "high" or not reported_in_time:
        return "investigate"
    return "pay"


@cat.rule("payout_band")
def payout_band(payable):
    return "none" if payable == 0 else ("under 10k" if payable < 10_000 else ("10k-100k" if payable <= 100_000 else "over 100k"))


@cat.rule("fast_track")
def fast_track(payable, fraud_level, reported_in_time):
    return payable < 5_000 and fraud_level == "low" and reported_in_time


@cat.rule("medical_ok")
def medical_ok(medical_codes_valid, cost_plausible):
    return medical_codes_valid and cost_plausible


@cat.rule("report_to_regulator")
def report_to_regulator(regulator_report_needed):
    return regulator_report_needed


QUESTIONS = [
    Question("decision", "Pay, deny or investigate?", Answer.choice(["pay", "deny", "investigate"]),
             checkpoints=["policy_in_force", "not_sanctioned"]),
    Question("payout_band", "How much?", Answer.choice(["none", "under 10k", "10k-100k", "over 100k"])),
    Question("fast_track", "Fast-track the payment?", Answer.yes_no(), checkpoints=["policy_in_force"]),
    Question("medical_ok", "Medical part consistent?", Answer.yes_no()),
    Question("report_to_regulator", "Report to the regulator?", Answer.yes_no()),
]

CLAIM = {
    "policy_id": "H-2291", "claim_type": "property", "claimant_id": "C-02", "claimant_name": "Anna Berg",
    "policy_start": date(2025, 1, 1), "policy_end": date(2026, 12, 31), "policy_perils": ["fire", "theft", "flood"],
    "incident_date": date(2026, 8, 3), "reported_date": date(2026, 8, 6), "peril": "theft",
    "amount_claimed": 4_200.0, "deductible": 500.0, "coverage_limit": 250_000.0,
    "diagnosis_codes": ["S52", "T14"], "invoice_lines": [1_200.0, 340.0], "hospital_days": 2,
}


def run_everything(state):
    """What a script without a strategist does: compute every fact in the catalog."""
    vals = dict(state)
    for p in cat.parts.values():
        vals[p.name] = p.func(**{x: vals[x] for x in p.inputs})
    return vals


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, (time.perf_counter() - t0) * 1000


if __name__ == "__main__":
    system = System(cat, QUESTIONS)
    print(f"catalog: {len(cat.parts)} parts + {len(cat.rules)} rules; slow parts: "
          + ", ".join(f"{k} {int(v * 1000)} ms" for k, v in SLOW.items()) + "\n")
    scenarios = [
        ("Only 'fast track?'", CLAIM, ["fast_track"]),
        ("Only 'medical part consistent?'", CLAIM, ["medical_ok"]),
        ("The full decision", CLAIM, ["decision", "payout_band"]),
        ("Everything", CLAIM, None),
        ("Policy expired -> the hard check decides first, the slow parts are skipped", {**CLAIM, "incident_date": date(2027, 2, 1),
                                                                          "reported_date": date(2027, 2, 2)}, ["decision"]),
    ]
    for title, state, names in scenarios:
        qs = [q for q in QUESTIONS if names is None or q.name in names]
        flow, t_plan = timed(lambda: plan(cat, qs, state.keys()))
        _, t_seq = timed(lambda: system.ask(state, names))
        resp, t_ask = timed(lambda: system.ask(state, names, workers=8))
        slow_run = [r.name for r in resp.trace.records if r.name in SLOW]
        print(f"=== {title} ===")
        print(f"strategy ({len(flow.steps)} of {len(cat.parts) + len(cat.rules)} parts, planned in {t_plan:.2f} ms):")
        print("\n".join("   " + line for line in str(flow).splitlines()[:len(flow.steps)]))
        answers = ", ".join(f"{q} = {r.answer}" + (" [forced]" if r.status == "forced" else "") for q, r in resp.results.items())
        print(f"answers: {answers}")
        if resp.trace.skipped:
            print("skipped at run time: " + ", ".join(n for n, _ in resp.trace.skipped))
        print(f"time: {t_seq:.0f} ms one by one, {t_ask:.0f} ms in parallel (slow parts run: {', '.join(slow_run) or 'none'})\n")
    _, t_all = timed(lambda: run_everything(CLAIM))
    print(f"for comparison, computing every fact without a strategist: {t_all:.0f} ms")
