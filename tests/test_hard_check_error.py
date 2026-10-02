from solvi import Answer, Catalog, Question, System


def _system(checkpoints):
    cat = Catalog()

    @cat.check(hard=True, then={"approve": "no"})
    def amount_ok(amount):
        return float(amount) < 1000

    @cat.rule("approve")
    def approve(amount):
        return "yes"

    return System(cat, [Question("approve", "Approve?", Answer.yes_no(), requires=checkpoints)])


def test_hard_check_that_raises_makes_the_question_abstain():
    s = _system(["amount_ok"])
    assert s.ask({"amount": 50})["approve"].answer == "yes"
    assert s.ask({"amount": 5000})["approve"].status == "forced"
    r = s.ask({"amount": "abc"})["approve"]
    assert r.answer is None and r.status == "abstain" and r.guard == "hard_check" and "could not be evaluated" in r.why
    assert s.ask({"amount": "abc"}).trace.replay(s.catalog)["ok"]


def _plan_catalog(with_check=True):
    cat = Catalog()

    @cat.fn
    def spec(plan):
        if not plan:
            raise ValueError("no slot in the plan")
        return plan

    @cat.fn
    def violations(spec):
        return [x for x in spec if x < 0]

    @cat.fn
    def count(violations):
        return len(violations)

    if with_check:
        @cat.check(hard=True, then={"accept": "no"})
        def day_allowed(violations):
            return not violations

    @cat.rule("accept")
    def accept(count):
        return "yes" if count == 0 else "no"
    return System(cat, [Question("accept", "Accept?", Answer.yes_no(), requires=["day_allowed"] if with_check else [])])


def test_an_abstention_names_the_part_that_failed_not_only_the_check_that_could_not_run():
    """The reason used to stop at the last link — "hard check day_allowed could not be evaluated: missing inputs:
    violations" — while the cause was two records up: spec raised."""
    r = _plan_catalog().ask({"plan": []})["accept"]
    assert (r.status, r.guard) == ("abstain", "hard_check")
    assert r.why == ("hard check day_allowed could not be evaluated: missing inputs: violations; "
                     "caused by spec: ValueError: no slot in the plan")
    r = _plan_catalog(with_check=False).ask({"plan": []})["accept"]
    assert r.status == "abstain" and r.why.startswith("rule not computed: missing inputs: count; missing ")
    assert r.why.endswith("; caused by spec: ValueError: no slot in the plan")
    assert _plan_catalog().ask({"plan": [1]})["accept"].answer == "yes"


def test_an_abstention_whose_step_failed_by_itself_has_no_other_cause_to_name():
    r = _system(["amount_ok"]).ask({"amount": "abc"})["approve"]
    assert "caused by" not in r.why and "could not be evaluated: ValueError" in r.why
