"""HR: approve a leave request. Rules and hard checks over a plain dict — no text, no model.

Run:  uv run python examples/01_leave_request.py
Try:  change the dates in REQUEST, the balance or the colleagues' leaves and watch the answers and the flow change."""
from __future__ import annotations

from datetime import date, timedelta

from solvi import Answer, Catalog, Question, System
from solvi.show import show

cat = Catalog()


# ---------- computations
@cat.fn
def days_requested(start, end):
    """working days in the request (Mon-Fri)"""
    d, n = start, 0
    while d <= end:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


@cat.fn
def remaining_after(balance, days_requested):
    return balance - days_requested


@cat.fn
def team_overlap(start, end, team_leaves, employee):
    """how many colleagues are on leave on the same dates"""
    return sum(1 for who, s, e in team_leaves if who != employee and s <= end and start <= e)


@cat.fn
def notice_days(start, today):
    return (start - today).days


@cat.fn
def in_blackout(start, end, blackout):
    return any(s <= end and start <= e for s, e in blackout)


# ---------- checks
@cat.check(hard=True, then={"approve": "reject"})
def enough_balance(remaining_after):
    """hard: the balance may not go negative — reject no matter what"""
    return remaining_after >= 0


@cat.check(hard=True, then={"approve": "reject"})
def dates_valid(start, end):
    return start <= end


@cat.check
def enough_notice(notice_days, days_requested):
    """a long leave (5+ days) needs at least 14 days notice"""
    return days_requested < 5 or notice_days >= 14


@cat.check
def team_covered(team_overlap, team_size):
    return team_overlap < team_size // 2


@cat.check
def not_blackout(in_blackout):
    return not in_blackout


# ---------- not needed for these questions (in the catalog, the strategist skips them)
@cat.fn
def payroll_code(employee, payroll_db):
    return payroll_db.get(employee, "?")


@cat.fn
def travel_budget(destination, budget_table):
    return budget_table.get(destination, 0)


# ---------- answer rules
@cat.rule("approve")
def approve(enough_notice, team_covered, not_blackout):
    return "approve" if (enough_notice and team_covered and not_blackout) else "needs_manager"


@cat.rule("notify_hr")
def notify_hr(days_requested, remaining_after):
    return days_requested >= 10 or remaining_after <= 2


QUESTIONS = [
    Question("approve", "Approve the leave?", Answer.choice(["approve", "needs_manager", "reject"]),
             checkpoints=["enough_balance", "dates_valid"]),
    Question("notify_hr", "Notify HR?", Answer.yes_no()),
]

TODAY = date(2026, 9, 25)
REQUEST = {
    "employee": "anna",
    "start": date(2026, 10, 19),
    "end": date(2026, 10, 30),
    "today": TODAY,
    "balance": 14,
    "team_size": 6,
    "team_leaves": [("boris", date(2026, 10, 26), date(2026, 11, 6)), ("vera", date(2026, 12, 1), date(2026, 12, 10))],
    "blackout": [(date(2026, 12, 20), date(2026, 12, 31))],          # year-end close
}

if __name__ == "__main__":
    system = System(cat, QUESTIONS)
    print("\n=== the request as is ===")
    show(system.ask(REQUEST), cat)
    print("\n=== same request, balance only 5 days ===")
    show(system.ask({**REQUEST, "balance": 5}), cat, flow=False)
    print("\n=== short leave during the year-end close ===")
    show(system.ask({**REQUEST, "start": date(2026, 12, 21), "end": date(2026, 12, 23)}), cat, flow=False)
