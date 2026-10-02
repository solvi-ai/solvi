"""Strategist: takes only what is needed from the catalog, inserts checkpoints, adds checks on computed facts."""
import random

from solvi import Answer, Catalog, Question, System
from examples_loader import load
from solvi.strategist import plan

I = load("03_invoices")

DISTRACT = {"contract_parties", "penalty_amount", "shipping_eta", "vendor_upper", "vendor_name_short"}


def test_distractors_not_executed_and_checkpoint_present():
    init, _ = I.make(random.Random(1), 1)
    flow = plan(I.cat, [q for q in I.QUESTIONS if q.name != "risk"], init.keys())
    executed = {s.part.name for s in flow.steps}
    assert not executed & DISTRACT
    assert "not_duplicate" in flow.per_question["approve"]
    assert "contract_parties" in flow.skipped and flow.skipped["contract_parties"] == "no inputs"


def test_rule_question_takes_only_rule_closure():
    init, _ = I.make(random.Random(2), 2)
    flow = plan(I.cat, [q for q in I.QUESTIONS if q.name == "duplicate"], init.keys())
    assert [s.part.name for s in flow.steps] == ["not_duplicate", "answer:duplicate"]


def _tiny():
    cat = Catalog()

    @cat.fn
    def a(x):
        return x * 2

    @cat.fn
    def b(a):
        return a + 1

    @cat.check
    def a_positive(a):              # not needed by the rule, but touches the computed fact a
        return a > 0

    @cat.check
    def x_small(x):                 # touches only an input, so not a check on computed facts
        return x < 100

    @cat.rule("q")
    def q_rule(b):
        return b > 3

    return cat, [Question("q", "b > 3?", Answer.yes_no())]


def test_verification_checks_added():
    cat, qs = _tiny()
    flow = plan(cat, qs, {"x"})
    names = [s.part.name for s in flow.steps]
    assert "a_positive" in names
    step = next(s for s in flow.steps if s.part.name == "a_positive")
    assert any("check on computed" in r for r in step.reasons)
    assert "x_small" not in names


def test_each_part_executed_once_for_many_questions():
    init, _ = I.make(random.Random(3), 3)
    r = System(I.cat, [q for q in I.QUESTIONS if q.name != "risk"]).ask(init)
    names = [x.name for x in r.trace.records]
    assert len(names) == len(set(names))


def test_unfitted_question_without_rule_is_not_narrowed():
    """A question with no rule, fit or uses: the strategist takes everything computable and says so in the reasons."""
    init, _ = I.make(random.Random(4), 4)
    flow = plan(I.cat, [q for q in I.QUESTIONS if q.name == "risk"], init.keys())
    assert any("not narrowed" in r for s in flow.steps for r in s.reasons)
    assert "contract_parties" not in {s.part.name for s in flow.steps}      # but non-computable parts are not taken


def test_fitted_head_narrows_flow():
    rng = random.Random(0)
    s = System(I.cat, I.QUESTIONS)
    s.fit("risk", [(a, t["risk"]) for a, t in (I.make(rng, i) for i in range(120))])
    init, _ = I.make(random.Random(5), 5)
    flow = plan(I.cat, I.QUESTIONS, init.keys(), s.heads)
    ran = {st.part.name for st in flow.steps}
    # the head reads the facts it selected; vendor_name_short (read from vendor_upper, a function of the vendor) may be
    # among them since 0.8: it tells the risk apart a little on these examples
    unfitted = {st.part.name for st in plan(I.cat, I.QUESTIONS, init.keys()).steps}
    assert not ran & (DISTRACT - {"vendor_name_short", "vendor_upper"}) and ran < unfitted
    assert set(s.heads["risk"].features) <= ran | set(init)


def test_answers_do_not_depend_on_which_other_questions_are_asked():
    cat = Catalog()

    @cat.fn
    def level(x):
        return x * 2

    @cat.check(hard=True, then={"state": "stop"})
    def below_trip(level):
        return level < 10

    @cat.rule("state")
    def state(level):
        return "run"

    @cat.rule("cause")
    def cause(level):
        return "high" if level > 6 else "normal"

    qs = [Question("state", "", Answer.choice(["run", "stop"]), checkpoints=["below_trip"]),
          Question("cause", "", Answer.choice(["high", "normal"]))]
    s = System(cat, qs)
    together = s.ask({"x": 7})
    alone = s.ask({"x": 7}, ["cause"])
    assert together["state"].answer == "stop" and together["state"].status == "forced"
    assert together["cause"].answer == alone["cause"].answer == "high"       # the hard check governs only "state"
    assert together["cause"].status == alone["cause"].status == "ok" and together["cause"].why == alone["cause"].why
