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
    return System(cat, [Question("ship", "Ship?", Answer.yes_no(), requires=["paid"]),
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
        s.ask(STATE, workers="learned")
    with pytest.raises(TypeError):
        s.ask(STATE, None, "learned")                              # keyword-only: aask's third used to be `order`
    with pytest.raises(ValueError, match="workers must be"):
        s.ask(STATE, workers=-3)
    assert s.ask(STATE, order="learned")["ship"].answer == "no"


def test_two_questions_with_one_name_are_refused():
    cat = Catalog()
    with pytest.raises(ValueError, match="two questions named 'ship'"):
        System(cat, [Question("ship", "first", Answer.yes_no()), Question("ship", "second", Answer.choice(["a"]))])


def test_ask_options_after_the_questions_are_keyword_only_and_names_is_a_deprecated_spelling():
    """ask(state, names, workers, order, store) and aask(state, names, order, store, ...) took a different third
    positional argument; everything after the questions is keyword-only now, and the questions are `questions=`."""
    import asyncio

    import pytest
    from solvi import Answer, Catalog, Question, System
    cat = Catalog()

    @cat.rule("ship")
    def ship(x) -> bool:
        return x > 1

    @cat.rule("hold")
    def hold(x) -> bool:
        return x > 5
    s = System(cat, [Question("ship", "Ship?", Answer.yes_no()), Question("hold", "Hold?", Answer.yes_no())])
    with pytest.raises(TypeError):
        s.ask({"x": 2}, ["ship"], 2)
    with pytest.raises(TypeError):
        System(cat, [], None)
    assert list(s.ask({"x": 2}, questions="ship").results) == ["ship"]
    with pytest.warns(DeprecationWarning, match=r"System.ask\(names=\) is deprecated: use questions="):
        assert list(s.ask({"x": 2}, names=["hold"]).results) == ["hold"]
    with pytest.warns(DeprecationWarning, match=r"System.aask\(names=\)"):
        assert list(asyncio.run(s.aask({"x": 2}, names=["hold"])).results) == ["hold"]
    with pytest.raises(TypeError, match="got both names= and questions="):
        s.ask({"x": 2}, names=["hold"], questions=["ship"])
