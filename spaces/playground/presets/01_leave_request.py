"""HR: approve a leave request (from examples/01_leave_request.py).
Rules and hard checks over plain data. Try: balance 5, or dates inside the year-end blackout."""
from datetime import date, timedelta

from solvi import Answer, Catalog, Question

cat = Catalog()


def prepare(state):
    """JSON has no dates: turn ISO strings into datetime.date."""
    d = date.fromisoformat
    state["start"], state["end"], state["today"] = d(state["start"]), d(state["end"]), d(state["today"])
    state["team_leaves"] = [(who, d(s), d(e)) for who, s, e in state["team_leaves"]]
    state["blackout"] = [(d(s), d(e)) for s, e in state["blackout"]]
    return state


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
    """hard: the balance may not go negative"""
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


# ---------- in the catalog but not needed here: the strategist skips them
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
