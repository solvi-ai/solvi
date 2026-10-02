"""Typed facts (solvi 0.5): type hints on catalog functions are checked between producers and consumers at registration,
validated / coerced with pydantic at run time; a failure is the `type_rejected` safeguard (the fact is missing)."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from enum import Enum
from typing import Any, Literal, Optional

import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, Quote, Response, System
from solvi.runtime import Trace
from solvi.typed import FactTypeError


class Band(str, Enum):
    low = "low"
    high = "high"


class Color(Enum):                  # a plain Enum (members are not equal to their values)
    red = 1
    green = 2


def typed_catalog():
    cat = Catalog()

    @cat.fn
    def risk_points(flags: list[str]) -> dict[str, float]:
        return {f: 1.5 for f in flags}

    @cat.fn
    def risk_score(risk_points: dict[str, float]) -> float:
        return sum(risk_points.values())

    @cat.fn
    def due(start: date, days: int) -> date:
        return start + timedelta(days=days)

    @cat.rule("band")
    def band(risk_score: float, due: date) -> Band:
        return Band.high if risk_score > 2 else "low"          # a plain value is coerced to the Enum

    return cat


# --- registration: producer / consumer types
def test_mismatch_at_registration_names_both_functions():
    cat = typed_catalog()
    with pytest.raises(FactTypeError, match=r"risk_score returns float, but label reads risk_score: str"):
        @cat.fn
        def label(risk_score: str) -> str:
            return risk_score
    assert "label" not in cat.parts and "label" not in cat.readers.get("risk_score", {})   # nothing was registered


def test_mismatch_when_the_producer_comes_second():
    cat = Catalog()

    @cat.fn
    def total_text(total: str) -> str:
        return total
    with pytest.raises(FactTypeError, match=r"total returns int, but total_text reads total: str"):
        @cat.fn
        def total(items: list) -> int:
            return len(items)
    assert "total" not in cat.parts


def test_compatible_types_pass():
    cat = Catalog()

    @cat.fn
    def n(xs: list) -> int:
        return len(xs)

    @cat.fn
    def ratio(n: float) -> float:                   # int → float (numeric tower)
        return n / 2

    @cat.fn
    def maybe(xs: list) -> Optional[str]:
        return None

    @cat.fn
    def shout(maybe: str) -> str:                   # str | None → str may pass: checked at run time
        return maybe.upper()

    @cat.fn
    def when(stamp: str) -> str:
        return stamp

    @cat.fn
    def day(when: date) -> int:                     # str → date: pydantic parses it
        return when.day

    @cat.fn
    def anything(n: Any, ratio) -> int:             # Any / no annotation: never checked
        return 1
    assert cat.types["n"] is int and cat.readers["n"] == {"ratio": float}
    assert cat.parts["anything"].tin is None


def test_alternative_producer_types_checked():
    cat = Catalog()

    @cat.fn
    def fee(total: float) -> float:
        return total * 0.01

    @cat.fn(provides="total")
    def total_table(doc: str) -> float:
        return 1.0
    with pytest.raises(FactTypeError, match="total_bad returns str, but fee reads total: float"):
        @cat.fn(provides="total")
        def total_bad(doc: str) -> str:
            return "1"


def test_rule_type_must_fit_question():
    cat = typed_catalog()
    with pytest.raises(FactTypeError, match=r"rule band .*'medium'|rule band returns"):
        System(cat, [Question("band", "Band", Answer.choice(["low", "medium"]))])
    System(cat, [Question("band", "Band", Answer.choice(["low", "high", "medium"]))])    # a superset is fine
    with pytest.raises(ValueError, match="no answer type"):
        System(Catalog(), [Question("q", "no rule, no type")])


# --- run time: validation, coercion, rejection
def test_coercion_and_trace():
    cat = typed_catalog()
    s = System(cat, [Question("band", "Band")])                 # the answer type comes from the rule: choice low/high
    assert s.questions["band"].answer.options == ["low", "high"]
    res = s.ask({"flags": ["a", "b"], "start": "2026-01-01", "days": "3"})
    assert res["band"].answer == "high" and res.values["due"] == date(2026, 1, 4)
    assert res.flow.types["risk_score"] is float and res.flow.types["start"] is date
    assert res.trace.replay(cat)["ok"]
    assert not res.safeguards and s.stats["type_rejected"] == 0


def test_rejection_abstains_and_is_counted():
    cat = typed_catalog()
    s = System(cat, [Question("band", "Band")])
    res = s.ask({"flags": "nope", "start": date(2026, 1, 1), "days": 3})
    r = res["band"]
    assert r.status == "abstain" and r.answer is None
    ev = [e for e in res.safeguards if e["kind"] == "type_rejected"]
    assert ev and ev[0]["fact"] == "risk_points" and "not list[str]" in ev[0]["detail"] and ev[0]["questions"] == ["band"]
    assert s.stats["type_rejected"] == 1
    assert "type rejected" in str(res.audit("band"))
    assert res.trace.replay(cat)["ok"]                          # the rejection replays as recorded


def test_output_rejection_falls_back_to_next_producer():
    cat = Catalog()

    @cat.fn(provides="rate")
    def rate_table(currency: str) -> float:
        return {"EUR": "n/a"}.get(currency)                    # not a number: rejected, not passed on

    @cat.fn(provides="rate")
    def rate_feed(currency: str) -> float:
        return "1.08"                                           # a numeric string is coerced

    @cat.rule("expensive")
    def expensive(rate: float) -> bool:
        return rate > 1

    s = System(cat, [Question("expensive", "Expensive?")])
    assert s.questions["expensive"].answer.kind == "yes_no"
    res = s.ask({"currency": "EUR"})
    assert res["expensive"].answer == "yes"
    rec = next(r for r in res.trace.records if r.name == "rate")
    assert rec.producer == "rate_feed" and rec.value == 1.08 and rec.tried[0][1].startswith("type rejected")
    kinds = [e["kind"] for e in res.safeguards]
    assert "type_rejected" in kinds and "fallback" in kinds
    assert s.stats["type_rejected"] == 1 and s.stats["fallbacks"] == 1
    assert res.trace.replay(cat)["ok"]


def test_typed_extract_value():
    cat = Catalog()

    @cat.extract
    def amount(doc: str) -> float:                              # the fact's type is the quote's value type
        i = doc.index("12")
        return Quote(doc[i:i + 5], i, i + 5)

    @cat.rule("big")
    def big(amount: float) -> bool:
        return amount > 10

    s = System(cat, [Question("big", "Big?")])
    res = s.ask({"doc": "total 12.50 EUR"})
    rec = res.trace.records[0]
    assert rec.value == 12.5 and rec.quote == (6, 11, "doc") and res["big"].answer == "yes"
    assert res.trace.replay(cat)["ok"]


# --- closed sets: Literal / Enum
def test_literal_rule_outside_options_abstains():
    cat = Catalog()

    @cat.rule("route")
    def route(team: str) -> Literal["billing", "tech"]:
        return team

    s = System(cat, [Question("route", "Route")])
    assert s.ask({"team": "tech"})["route"].answer == "tech"
    res = s.ask({"team": "sales"})
    assert res["route"].status == "abstain"
    assert [e["kind"] for e in res.safeguards] == ["outside_options"]


def test_enum_and_multi_answers():
    cat = Catalog()

    @cat.rule("color")
    def color(n: int) -> Color:
        return Color.red if n == 1 else 2                       # a member, or a value coerced to one

    @cat.rule("tags")
    def tags(words: list[str]) -> list[Literal["pii", "abuse", "spam"]]:
        return [w for w in words if w in ("spam", "pii")]

    s = System(cat, [Question("color", "Color"), Question("tags", "Tags")])
    assert s.questions["color"].answer.options == [1, 2] and s.questions["tags"].answer.kind == "multi"
    res = s.ask({"n": "2", "words": ["spam", "pii", "x"]})
    assert res["color"].answer == 2 and res["tags"].answer == ("pii", "spam")
    assert res.trace.replay(cat)["ok"]


def test_answer_from_type():
    assert Answer.from_type(bool).kind == "yes_no"
    assert Answer.from_type(Literal["yes", "no"]).kind == "yes_no"
    a = Answer.from_type(Literal["low", "mid", "high"], ordinal=True)
    assert a.kind == "ordinal" and a.options == ["low", "mid", "high"]
    assert Answer.from_type(Band).options == ["low", "high"]
    assert Answer.from_type(Optional[Band]).kind == "choice"
    m = Answer.from_type(set[Literal["a", "b"]])
    assert m.kind == "multi" and m.normalize({"b", "a"}) == ("a", "b")
    assert Answer.from_type(Band).normalize(Band.high) == "high"
    with pytest.raises(TypeError):
        Answer.from_type(int)


# --- typed input state
class Request(BaseModel):
    flags: list[str]
    start: date
    days: int = 5


def test_basemodel_input():
    cat = typed_catalog()
    s = System(cat, [Question("band", "Band")])
    res = s.ask(Request(flags=["x", "y", "z"], start=date(2026, 2, 1)))     # fields (and defaults) are the given facts
    assert res["band"].answer == "high" and res.values["due"] == date(2026, 2, 6)
    assert res.trace.init["days"] == 5


def test_inputs_model_validates_dicts():
    cat = typed_catalog()
    s = System(cat, [Question("band", "Band")], inputs=Request)
    ok = s.ask({"flags": ["x"], "start": "2026-03-01"})
    assert ok["band"].answer == "low" and ok.trace.init["start"] == date(2026, 3, 1) and ok.trace.init["days"] == 5
    bad = s.ask({"flags": ["x"], "start": "not a date", "days": 2})
    assert bad["band"].status == "abstain" and "start" not in bad.trace.init
    ev = bad.safeguards
    assert ev[0]["kind"] == "type_rejected" and ev[0]["fact"] == "start" and ev[0]["questions"] == ["band"]
    assert s.stats["type_rejected"] == 1


# --- serialization and replay
def test_response_json_round_trip_and_schema():
    cat = typed_catalog()
    s = System(cat, [Question("band", "Band")])
    res = s.ask({"flags": ["a", "b"], "start": date(2026, 1, 1), "days": 3})
    j = res.to_json()
    d = json.loads(j)
    assert d["results"]["band"]["answer"] == "high" and d["values"]["due"] == "2026-01-04"
    back = Response.from_json(j, catalog=cat)
    assert back.to_json() == j
    assert back["band"].answer == "high" and back.values["due"] == date(2026, 1, 4)
    assert back.trace.replay(cat)["ok"]
    assert res.model_dump()["values"]["due"] == date(2026, 1, 4)           # python mode keeps the objects
    assert "results" in Response.model_json_schema()["properties"]
    sch = s.response_schema()
    band = sch["$defs"]["Result_band"]["properties"]["answer"]
    assert {"enum": ["low", "high"], "type": "string"} in band["anyOf"]
    q = Question("band", "Band", Answer.choice(["low", "high"]))
    assert Question.from_json(q.to_json()) == q


def test_replay_with_typed_facts():
    cat = typed_catalog()
    s = System(cat, [Question("band", "Band")])
    res = s.ask({"flags": ["a", "b"], "start": date(2026, 1, 1), "days": "3"})
    assert res.trace.replay(cat)["ok"]
    d = res.trace.model_dump("json")
    assert Trace.model_validate(d, catalog=cat).replay(cat)["ok"]           # dates restored from the facts' types
    assert Trace.model_validate(d, catalog=s).replay(s)["ok"]
    assert Trace.model_validate(d).replay(cat)["ok"]                        # also without them: the dump has each type
    old = {**d, "records": [{k: v for k, v in r.items() if k != "type"} for r in d["records"]]}
    old.pop("init_types", None)                                             # as solvi ≤ 0.7.1 wrote it: a date stays a
    rep = Trace.model_validate(old).replay(cat)                             # string, which is no verdict on the data
    assert not rep["ok"] and set(rep["kinds"]) == {"not_restored"}
    rec = next(r for r in res.trace.records if r.name == "risk_score")
    rec.value = 4.0                                                         # tamper with a coerced value
    rep = res.trace.replay(cat)
    assert not rep["ok"] and any(m[1] == "risk_score" for m in rep["mismatches"])


# --- untyped stays untouched
def test_untyped_catalog_skips_everything():
    cat = Catalog()

    @cat.fn
    def total(items):
        return sum(items)

    @cat.rule("big")
    def big(total: Any):
        return total > 10

    p, r = cat.parts["total"], cat.rules["big"]
    assert p.types is None and p.returns is None and p.tin is None and p.tout is None and r.tin is None
    assert not cat.types and not cat.readers
    res = System(cat, [Question("big", "Big?", Answer.yes_no())]).ask({"items": [5, 6]})
    assert res["big"].answer == "yes" and res.flow.types == {}


def test_import_is_light():
    out = subprocess.run([sys.executable, "-c", "import sys, solvi; print('pydantic' in sys.modules)"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_example_14():
    from examples_loader import load
    E = load("14_typed_catalog")
    s = System(E.cat, E.QUESTIONS, inputs=E.Parcel)
    r = s.ask(E.PARCEL)
    assert (r["duty"].answer, r["flags"].answer, r["release"].answer) == ("standard", (), "yes")
    r = s.ask(E.PARCEL.model_copy(update={"label": "Weight: 12 kg", "declared_weight_kg": 41.0}))
    assert r["flags"].answer == ("heavy",) and s.stats["type_rejected"] == 1 and s.stats["fallbacks"] == 1


def test_a_typed_input_comes_in_the_declared_field_order():
    """A decider reads a state's keys in their order, and answers depend on it. With System(inputs=Model) the order is
    the model's, whoever built the dict; without a model the dict is taken as it comes."""
    from pydantic import BaseModel

    from solvi.decide import state_text

    class Order(BaseModel):
        customer: str
        amount: float
        note: str = ""

    cat = Catalog()

    @cat.rule("ok")
    def ok(amount, extra=None) -> bool:
        return amount < 100

    s = System(cat, [Question("ok", "Ok?")], inputs=Order)
    a = s.ask({"note": "x", "zeta": 1, "amount": 5, "customer": "ann", "alpha": 2})
    b = s.ask({"customer": "ann", "alpha": 2, "amount": 5, "zeta": 1, "note": "x"})
    assert list(a.trace.init) == ["customer", "amount", "note", "zeta", "alpha"]       # fields, then the rest as given
    assert list(b.trace.init)[:3] == ["customer", "amount", "note"]
    assert state_text({k: a.trace.init[k] for k in Order.model_fields}) == state_text(Order(customer="ann", amount=5, note="x"))
    assert a.trace.init_hash == b.trace.init_hash and a.trace.replay(s)["ok"]
    assert list(s.ask({"amount": 5, "customer": "ann"}).trace.init) == ["customer", "amount", "note"]    # the default too
    bad = s.ask({"note": "x", "amount": "many", "customer": "ann"})                    # a rejected field: the order holds
    assert list(bad.trace.init) == ["customer", "note"] and bad["ok"].status == "abstain"
    plain = System(cat, [Question("ok", "Ok?")])
    assert list(plain.ask({"note": "x", "amount": 5, "customer": "ann"}).trace.init) == ["note", "amount", "customer"]
