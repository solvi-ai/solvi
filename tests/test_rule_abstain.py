from solvi import Answer, Catalog, Question, System


def test_rule_returning_none_is_an_abstention_not_outside_options():
    cat = Catalog()

    @cat.rule("team")
    def team(votes):
        return None if votes == "split" else votes

    s = System(cat, [Question("team", "Which team?", Answer.choice(["billing", "tech"]))])
    r = s.ask({"votes": "split"})["team"]
    assert r.answer is None and r.status == "abstain" and r.guard == "rule_abstained"
    audit = s.ask({"votes": "split"}).audit()
    assert "rule abstained" in str(audit) and "outside the options" not in str(audit)
    bad = s.ask({"votes": "sales"})["team"]
    assert bad.guard == "outside_options"
