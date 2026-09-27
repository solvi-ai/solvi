from solvi import Answer, Catalog, Question, System


def _system(checkpoints):
    cat = Catalog()

    @cat.check(hard=True, then={"approve": "no"})
    def amount_ok(amount):
        return float(amount) < 1000

    @cat.rule("approve")
    def approve(amount):
        return "yes"

    return System(cat, [Question("approve", "Approve?", Answer.yes_no(), checkpoints=checkpoints)])


def test_hard_check_that_raises_makes_the_question_abstain():
    s = _system(["amount_ok"])
    assert s.ask({"amount": 50})["approve"].answer == "yes"
    assert s.ask({"amount": 5000})["approve"].status == "forced"
    r = s.ask({"amount": "abc"})["approve"]
    assert r.answer is None and r.status == "abstain" and r.guard == "hard_check" and "could not be evaluated" in r.why
    assert s.ask({"amount": "abc"}).trace.replay(s.catalog)["ok"]
