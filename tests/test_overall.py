import math

from solvi import Answer, Catalog, Question, System


def _system():
    cat = Catalog()

    @cat.rule("team")
    def team(votes):
        return None if votes == "split" else votes

    @cat.rule("urgent")
    def urgent(level):
        return level > 2

    return System(cat, [Question("team", "Which team?", Answer.choice(["billing", "tech"])),
                        Question("urgent", "Urgent?", Answer.yes_no())])


def test_overall_confidence_is_product_of_answered():
    s = _system()
    r = s.ask({"votes": "tech", "level": 3})
    assert r.complete and r.confidence == math.prod(x.confidence for x in r.results.values())
    assert r.overall["answered"] == 2 and r.overall["abstained"] == [] and r.weakest[0] in r.results


def test_overall_confidence_below_one_is_the_product_and_the_weakest_answer_is_named():
    """The test above cannot fail (every rule answers at 1.0): here two answers carry their own confidence."""
    from solvi import Decision
    cat = Catalog()

    @cat.rule("team")
    def team(votes):
        return Decision(votes, {"billing": 0.2, "tech": 0.8})

    @cat.rule("urgent")
    def urgent(level):
        return Decision("yes" if level > 2 else "no", {"yes": 0.6, "no": 0.4} if level > 2 else {"yes": 0.1, "no": 0.9})
    s = System(cat, [Question("team", "Which team?", Answer.choice(["billing", "tech"])),
                     Question("urgent", "Urgent?", Answer.yes_no())])
    r = s.ask({"votes": "tech", "level": 3})
    assert [round(x.confidence, 2) for x in r.results.values()] == [0.8, 0.6]
    assert math.isclose(r.confidence, 0.48) and r.weakest[0] == "urgent"


def test_overall_with_abstention_and_json():
    s = _system()
    r = s.ask({"votes": "split", "level": 1})
    assert not r.complete and r.overall["abstained"] == ["team"] and r.overall["answered"] == 1
    assert "overall: confidence" in str(r.audit()) and "abstained: team" in str(r.audit())
    assert r.to_dict()["overall"]["abstained"] == ["team"]
