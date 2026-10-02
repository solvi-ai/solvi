"""The policy of POLICY.md as plain functions: the reference the scorer checks every arm against, and the baseline's core."""
RISKY_HISTORY = ("all credits at this bank paid back duly", "no credits taken, or all paid back duly")


def points(a, version):
    p = {}
    p["checking"] = {"below 0 DM": 3, "0 to 200 DM": 2, "no checking account": -1}.get(a["checking_account"], 0)
    p["duration"] = 2 if a["duration_months"] > 36 else 1 if a["duration_months"] > 24 else 0
    p["amount"] = 2 if a["credit_amount_dm"] > 8000 else 0
    p["savings"] = 1 if a["savings"] == "below 100 DM" else 0
    p["employment"] = 1 if a["employed_since"] in ("unemployed", "less than 1 year") else 0
    p["history"] = 2 if a["credit_history"] in RISKY_HISTORY else 0
    p["other_plans"] = 1 if a["other_installment_plans"] in ("bank", "stores") else 0
    if version == 1:
        p["property"] = 1 if a["property"] == "unknown or none" else 0
    else:
        p["guarantor"] = -1 if a["other_debtors"] == "guarantor" else 0
        p["purpose"] = -1 if a["purpose"] == "car (used)" else 1 if a["purpose"] in ("education", "repairs") else 0
    return p


def decide(a, version):
    """→ (decision, total points, points per rule, the hard rule that fired or None)."""
    p = points(a, version)
    total = sum(p.values())
    if a["credit_amount_dm"] > 15000 and a["duration_months"] > 48:
        return "refuse", total, p, "too_large"
    refuse_at = 5 if version == 1 else 4
    return ("refuse" if total >= refuse_at else "review" if total == refuse_at - 1 else "approve"), total, p, None
