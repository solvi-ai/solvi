# Credit policy (the task's specification — both arms implement exactly this)

An application is scored in points; the points decide. Fields are those of `$STAND_DATA/credit/prepared/applications_*.jsonl`.

## Version 1

| rule | condition | points |
|---|---|---|
| checking | checking_account is "below 0 DM" | +3 |
| checking | checking_account is "0 to 200 DM" | +2 |
| checking | checking_account is "no checking account" | −1 |
| duration | duration_months > 36 | +2 |
| duration | 24 < duration_months ≤ 36 | +1 |
| amount | credit_amount_dm > 8000 | +2 |
| savings | savings is "below 100 DM" | +1 |
| employment | employed_since is "unemployed" or "less than 1 year" | +1 |
| history | credit_history is "all credits at this bank paid back duly" or "no credits taken, or all paid back duly" | +2 |
| other_plans | other_installment_plans is "bank" or "stores" | +1 |
| property | property is "unknown or none" | +1 |

Hard rule `too_large`: credit_amount_dm > 15000 and duration_months > 48 → refuse, whatever the points.

Decision: points ≥ 5 → **refuse**; points = 4 → **review** (a person decides); otherwise **approve**.

## Version 2 (the change under audit)

- the thresholds move: points ≥ 4 → refuse; points = 3 → review;
- new rule `guarantor`: other_debtors is "guarantor" → −1;
- new rule `purpose`: purpose is "car (used)" → −1; purpose is "education" or "repairs" → +1;
- rule `property` is removed;
- everything else as in version 1.

## A change that must be rejected

Proposed rule `young`: age ≤ 25 → +1. It reads a sensitive field (`about.json`: personal_status_and_sex, foreign_worker,
age). The system must notice this before the rule is used, not after.

## What is checked

1. **Correctness** — decisions under v1 and v2 for all 1,000 applications equal the reference (`score.py`).
2. **Record** — every decision is stored with the version, the input and the points each rule gave.
3. **Replay** — the 600 history decisions made under v1 reproduce from their records.
4. **Diff** — which history decisions v2 would change, and which rule or threshold is responsible for each.
5. **Tamper** — after one stored input is edited (an amount), the edit is detected.
6. **Moved rules** — once v2 is deployed, replaying a v1 record says "the rules changed", not "the data is corrupted".
7. **Sensitive field** — the `young` rule is refused with the reason.
8. **Cost** — on `applications_new`: 5 per approved bad loan, 1 per refused good one; reviews counted separately.

For 2–7 a solution reports pass / fail with the evidence, and how many lines of its own code each took.
