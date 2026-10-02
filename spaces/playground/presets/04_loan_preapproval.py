"""Lending: pre-approve a personal loan from an application.
Ratios (debt-to-income, payment-to-income) are ordinary functions; "applicant is an adult" is a hard check
that no score can override. Try: age 17, or monthly_debts 2500."""
from solvi import Answer, Catalog, Question

cat = Catalog()


@cat.fn
def monthly_payment(loan_amount, annual_rate, term_months):
    """annuity payment"""
    r = annual_rate / 12
    return round(loan_amount * r / (1 - (1 + r) ** -term_months), 2) if r else round(loan_amount / term_months, 2)


@cat.fn
def debt_to_income(monthly_debts, monthly_payment, monthly_income):
    """all debts including the new loan, as a share of income"""
    return round((monthly_debts + monthly_payment) / monthly_income, 3)


@cat.fn
def payment_to_income(monthly_payment, monthly_income):
    return round(monthly_payment / monthly_income, 3)


@cat.fn
def years_employed(months_employed):
    return months_employed / 12


@cat.check(hard=True, then={"preapprove": "decline", "needs_cosigner": "no"})
def adult(age):
    """hard: applicants must be 18 or older"""
    return age >= 18


@cat.check(hard=True, then={"preapprove": "decline"})
def income_positive(monthly_income):
    return monthly_income > 0


@cat.check
def dti_ok(debt_to_income):
    """soft: at most 43% of income goes to debt"""
    return debt_to_income <= 0.43


@cat.check
def stable_job(years_employed):
    return years_employed >= 1


@cat.fn
def marketing_segment(zip_code, segments_db):      # needs data we don't have: not run
    return segments_db.get(zip_code, "general")


@cat.rule("preapprove")
def preapprove(debt_to_income, payment_to_income, credit_score, stable_job):
    if credit_score < 580 or debt_to_income > 0.5:
        return "decline"
    if debt_to_income <= 0.36 and payment_to_income <= 0.15 and credit_score >= 680 and stable_job:
        return "approve"
    return "refer"


@cat.rule("needs_cosigner")
def needs_cosigner(credit_score, years_employed):
    return credit_score < 650 or years_employed < 1


QUESTIONS = [
    Question("preapprove", "Pre-approve the loan?", Answer.choice(["approve", "refer", "decline"]),
             requires=["adult", "income_positive"]),
    Question("needs_cosigner", "Ask for a co-signer?", Answer.yes_no(), requires=["adult"]),
]
