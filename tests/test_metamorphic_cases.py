"""testing.metamorphic: a state transform and the relation the answers must keep (the same answer after a unit
conversion, "heavy" after a much larger weight) → cases in the cases.json format whose expected answers come from
the source case's answers; a System that breaks the relation fails them."""
import warnings

import pytest

from solvi import Answer, Catalog, Question, System, testing


def _dose_system(honours_unit=True):
    cat = Catalog()

    @cat.fn
    def weight_kg(weight, unit):
        return weight / 1000 if (honours_unit and unit == "g") else weight

    @cat.rule("dose")
    def dose(weight_kg):
        return "low" if weight_kg * 2 < 150 else "high"

    @cat.rule("heavy")
    def heavy(weight_kg) -> bool:
        return weight_kg >= 80

    return System(cat, [Question("dose", "Dose?", Answer.choice(["low", "high"])), Question("heavy", "Heavy?", Answer.yes_no())])


SEEDS = [{"name": "adult", "state": {"weight": 70, "unit": "kg"}},
         {"name": "big", "state": {"weight": 95.5, "unit": "kg"}}]
TO_GRAMS = testing.Relation("kg to g", lambda s: {**s, "weight": s["weight"] * 1000, "unit": "g"} if s["unit"] == "kg"
                            else None)


def test_a_unit_conversion_keeps_the_answers():
    system = _dose_system()
    out = testing.metamorphic(SEEDS, [TO_GRAMS], system=system)
    assert [c["state"] for c in out] == [{"weight": 70000, "unit": "g"}, {"weight": 95500.0, "unit": "g"}]
    assert out[0]["expected"] == {"dose": "low", "heavy": "no"} and out[1]["expected"]["heavy"] == "yes"
    assert out[0]["note"] == "metamorphic: kg to g of adult"
    assert all(testing.run_case(system, c, c["state"]).ok for c in out)


def test_a_system_that_breaks_the_relation_fails_the_cases():
    out = testing.metamorphic(SEEDS, [TO_GRAMS], system=_dose_system())
    broken = _dose_system(honours_unit=False)           # reads grams as kilograms
    res = [testing.run_case(broken, c, c["state"]) for c in out]
    assert res[0].status == "regression"               # 70 000 read as kg: a high dose, heavy
    assert res[1].ok                                    # 95 500 "kg" gives what 95.5 kg gives: the relation holds
    assert any("dose: expected 'low'" in p for p in res[0].problems)


def test_an_answer_relation_and_the_cases_expected_answers():
    double = testing.Relation("double weight", lambda s: {**s, "weight": s["weight"] * 2},
                              expect=lambda a: {"dose": "high" if a["dose"] == "low" else a["dose"]})
    seeds = [{"name": "adult", "state": {"weight": 70, "unit": "kg"}, "expected": {"dose": "low", "heavy": "no"}}]
    out = testing.metamorphic(seeds, [double])          # no system: from the case's expected answers
    assert out == [{"name": "double weight 1", "state": {"weight": 140, "unit": "kg"}, "expected": {"dose": "high"},
                    "ask": ["dose"], "note": "metamorphic: double weight of adult"}]
    assert testing.run_case(_dose_system(), out[0], out[0]["state"]).ok


def test_not_applicable_raising_and_nothing_to_compare():
    seeds = SEEDS + [{"name": "grams", "state": {"weight": 500, "unit": "g"}},
                     {"name": "no answers", "state": {"weight": 1, "unit": "kg"}}]
    bad = testing.Relation("bad", lambda s: s["missing"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = testing.metamorphic(seeds[:3], [TO_GRAMS, bad], system=_dose_system())
        none = testing.metamorphic(seeds[3:], [TO_GRAMS])
    assert len(out) == 2                                # grams: not applicable (None); "bad" raised on every case
    assert none == []
    msgs = " ".join(str(x.message) for x in w)
    assert "3 transform(s) raised" in msgs and "no answers to compare" in msgs
    with pytest.raises(TypeError):
        testing.Relation("x", "not callable")
