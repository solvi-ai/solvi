"""Consumer credit: a loan application -> approve / decline / refer, the principal adverse-action reason, and "refer to a senior
underwriter?".

A points scorecard (eight factors, 100 points) and three knock-outs give the decision. The adverse-action reasons that
ECOA / Regulation B and FCRA require on a decline are computed facts, not a story told afterwards: the factors that lost
the most points, knock-outs first, worded like the Regulation B sample notice, at most four. Eligibility (age to contract,
residency) is a hard check. "Refer to underwriter?" has no written rule: run.py learns it in milliseconds from 200 past files
(fit) and absorbs each underwriter correction instantly; here, without that history, it abstains.
Try: age 17, monthly_debt_payments 2400, credit_history_months 8, or delinquencies_24m 3. Synthetic policy, not lending advice."""
from __future__ import annotations

from solvi import Answer, Catalog, Question

cat = Catalog()

ELIGIBLE_RESIDENCY = ("citizen", "permanent_resident")
REASONS = {                          # factor -> adverse-action wording (Regulation B sample form style)
    "dti": "Excessive obligations in relation to income",
    "credit_score": "Credit score insufficient",
    "history": "Limited credit experience",
    "delinquencies": "Delinquent past or present credit obligations with others",
    "utilization": "Proportion of balances to credit limits is too high",
    "inquiries": "Number of recent inquiries on credit bureau report",
    "employment": "Length of employment",
    "loan_to_income": "Income insufficient for amount of credit requested",
}
MAX_POINTS = {"dti": 20, "credit_score": 20, "delinquencies": 16, "history": 12, "utilization": 12, "inquiries": 8,
              "employment": 7, "loan_to_income": 5}
APPROVE_AT, REFER_AT = 72, 58
UNDERWRITER_FACTS = ["loan_to_income", "employment_months", "self_employed", "credit_history_months", "requested_amount",
                     "dti", "credit_score"]


def _steps(x, bands):
    """bands: [(upper_bound, points), ...] in increasing order of the bound; x above every bound gets the last points."""
    for bound, pts in bands[:-1]:
        if x <= bound:
            return pts
    return bands[-1][1]


# ---------- computations
@cat.fn
def monthly_payment(requested_amount, annual_rate, term_months):
    """annuity payment of the requested loan"""
    r = annual_rate / 12
    return round(requested_amount * r / (1 - (1 + r) ** -term_months), 2) if r else round(requested_amount / term_months, 2)


@cat.fn
def dti(monthly_debt_payments, monthly_payment, annual_income):
    """debt-to-income after the new loan: all monthly debt payments / gross monthly income"""
    return round((monthly_debt_payments + monthly_payment) / (annual_income / 12), 4)


@cat.fn
def loan_to_income(requested_amount, annual_income):
    return round(requested_amount / annual_income, 3)


@cat.fn
def scorecard(dti, credit_score, credit_history_months, delinquencies_24m, utilization, inquiries_6m, employment_months,
              loan_to_income):
    """points per factor (out of MAX_POINTS)"""
    return {
        "dti": _steps(dti, [(0.30, 20), (0.36, 15), (0.43, 9), (0.50, 3), (None, 0)]),
        "credit_score": _steps(-credit_score, [(-740, 20), (-700, 16), (-660, 11), (-620, 6), (-580, 2), (None, 0)]),
        "delinquencies": _steps(delinquencies_24m, [(0, 16), (1, 8), (2, 3), (None, 0)]),
        "history": _steps(-credit_history_months, [(-84, 12), (-48, 9), (-24, 6), (-12, 3), (None, 0)]),
        "utilization": _steps(utilization, [(0.30, 12), (0.50, 8), (0.75, 4), (None, 0)]),
        "inquiries": _steps(inquiries_6m, [(1, 8), (3, 5), (5, 2), (None, 0)]),
        "employment": _steps(-employment_months, [(-24, 7), (-12, 4), (-6, 2), (None, 0)]),
        "loan_to_income": _steps(loan_to_income, [(0.20, 5), (0.35, 3), (0.50, 1), (None, 0)]),
    }


@cat.fn
def points(scorecard):
    return sum(scorecard.values())


@cat.fn
def knockouts(credit_score, dti, delinquencies_24m):
    """policy knock-outs: decline whatever the points"""
    k = []
    if dti > 0.50:
        k.append("dti")
    if credit_score < 580:
        k.append("credit_score")
    if delinquencies_24m >= 3:
        k.append("delinquencies")
    return k


@cat.fn
def adverse_action_reasons(scorecard, knockouts):
    """up to four principal reasons: knock-outs first, then the factors that lost the most points (at least a quarter of
    their maximum), ties in MAX_POINTS order"""
    lost = sorted((f for f in MAX_POINTS if f not in knockouts and MAX_POINTS[f] - scorecard[f] >= MAX_POINTS[f] / 4),
                  key=lambda f: -(MAX_POINTS[f] - scorecard[f]))
    return [REASONS[f] for f in (knockouts + lost)[:4]]


# ---------- eligibility (hard)
@cat.check(hard=True, then={"decision": "decline", "principal_reason": "Applicant under the legal age to contract",
                            "refer_to_underwriter": "no"})
def adult(age):
    """hard: 18 or older (capacity to contract is the one lawful use of age here)"""
    return age >= 18


@cat.check(hard=True, then={"decision": "decline", "principal_reason": "Temporary residence", "refer_to_underwriter": "no"})
def resident(residency_status):
    """hard: citizens and permanent residents only"""
    return residency_status in ELIGIBLE_RESIDENCY


# ---------- answers
def _outcome(points, knockouts):
    if knockouts:
        return "decline"
    return "approve" if points >= APPROVE_AT else ("refer" if points >= REFER_AT else "decline")


@cat.rule("decision")
def decision(points, knockouts):
    """knock-out -> decline; points >= 72 approve, 58-71 refer, below 58 decline"""
    return _outcome(points, knockouts)


@cat.rule("principal_reason")
def principal_reason(points, knockouts, adverse_action_reasons):
    """the first adverse-action reason when the application is not approved"""
    if _outcome(points, knockouts) == "approve" or not adverse_action_reasons:
        return "none"
    return adverse_action_reasons[0]


ELIGIBILITY = ["adult", "resident"]
QUESTIONS = [
    Question("decision", "Approve, decline or refer?", Answer.choice(["approve", "decline", "refer"]), checkpoints=ELIGIBILITY),
    Question("principal_reason", "Principal adverse-action reason",
             Answer.choice(["none", *REASONS.values(), "Applicant under the legal age to contract", "Temporary residence"]),
             checkpoints=ELIGIBILITY),
    Question("refer_to_underwriter", "Refer the file to a senior underwriter?", Answer.yes_no(), checkpoints=ELIGIBILITY,
             uses=UNDERWRITER_FACTS),
]
