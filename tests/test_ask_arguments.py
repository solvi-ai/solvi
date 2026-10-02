"""System.ask / aask arguments: question names, order and workers are checked, not silently misread."""
import asyncio

import pytest

from solvi import Answer, Catalog, Question, System


def _shop():
    cat = Catalog()

    @cat.check(hard=True, then={"ship": "no"})
    def paid(status) -> bool:
        return status == "paid"

    @cat.rule("ship")
    def ship(paid):
        return "yes"

    @cat.rule("gift")
    def gift(total):
        return "yes" if total > 100 else "no"
    return System(cat, [Question("ship", "Ship?", Answer.yes_no(), checkpoints=["paid"]),
                        Question("gift", "Gift?", Answer.yes_no())])


STATE = {"status": "unpaid", "total": 120}


def test_a_single_question_name_is_one_question_not_its_letters():
    """ask(state, "ship") used to raise KeyError: 's' (the string was iterated)."""
    s = _shop()
    assert list(s.ask(STATE, "ship").results) == ["ship"]
    assert list(asyncio.run(s.aask(STATE, "gift")).results) == ["gift"]
    with pytest.raises(KeyError, match="no question 'shp'"):
        s.ask(STATE, ["shp"])


def test_order_and_workers_are_checked():
    s = _shop()
    with pytest.raises(ValueError, match="order must be"):
        s.ask(STATE, order="bogus")
    with pytest.raises(ValueError, match="workers must be a positive int, not 'learned'"):
        s.ask(STATE, None, "learned")                              # aask's third positional argument is `order`
    with pytest.raises(ValueError, match="workers must be"):
        s.ask(STATE, workers=-3)
    assert s.ask(STATE, order="learned")["ship"].answer == "no"


def test_two_questions_with_one_name_are_refused():
    cat = Catalog()
    with pytest.raises(ValueError, match="two questions named 'ship'"):
        System(cat, [Question("ship", "first", Answer.yes_no()), Question("ship", "second", Answer.choice(["a"]))])
