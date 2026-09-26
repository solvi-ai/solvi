"""Blank template: a tiny working system to start from. Replace it with your own task.

1. Put the request data in init_state (right-hand JSON box). Its keys are the starting facts.
2. Write small functions. ARGUMENT NAMES = facts they read; FUNCTION NAME = the fact they produce.
3. Add checks (return True/False). hard=True + then={question: answer} forces that answer when the check fails.
4. Write one rule per question with @cat.rule("question_name"); it must return one of the answer options.
5. List the questions in QUESTIONS. Optionally define prepare(state) to convert JSON values (e.g. ISO dates).
"""
from solvi import Answer, Catalog, Question, Quote  # noqa: F401  (Quote: values extracted from text, with offsets)

cat = Catalog()


def prepare(state):
    """optional: convert JSON values into Python types before asking"""
    return state


@cat.fn
def sleep_debt(hours_slept, hours_needed):
    return hours_needed - hours_slept


@cat.fn
def cups_left_today(cups_today, daily_limit):
    return daily_limit - cups_today


@cat.check(hard=True, then={"coffee": "no"})
def not_too_late(hour):
    """hard: no coffee after 17:00, no matter how tired"""
    return hour < 17


@cat.rule("coffee")
def coffee(sleep_debt, cups_left_today):
    return sleep_debt >= 1 and cups_left_today > 0


@cat.rule("which")
def which(sleep_debt):
    return "espresso" if sleep_debt >= 3 else ("flat white" if sleep_debt >= 1 else "decaf")


QUESTIONS = [
    Question("coffee", "Have a coffee now?", Answer.yes_no(), checkpoints=["not_too_late"]),
    Question("which", "Which one?", Answer.choice(["espresso", "flat white", "decaf"])),
]
