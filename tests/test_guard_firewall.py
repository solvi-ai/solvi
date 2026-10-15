"""Guard.firewall: two declarative, deterministic checks of a tool call — the arguments validate against the tool's
declared schema, and every slot value the call gives is in the request text or the facts the app gives (numbers as
digits; enumerations and booleans exempt by their declaration) — with a per-slot report; a call failing either is
refused or escalated per the declaration."""
import json
from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel

from solvi.solutions.guard import Guard

REQ = [{"role": "system", "content": "You book trains."},
       {"role": "user", "content": "Book 2 seats from Paris to Lyon on 2026-11-03, second class."}]


class Seat(str, Enum):
    window = "window"
    aisle = "aisle"


def make(**fw):
    g = Guard()
    made = []

    @g.tool
    def book_train(origin: str, destination: str, date: str, seats: int = 1,
                   travel_class: Literal["first", "second"] = "second", seat: Seat = Seat.window,
                   refundable: bool = False) -> str:
        """Book a train."""
        made.append((origin, destination, date, seats))
        return "booked"

    @g.tool
    def weather(city: str) -> str:
        """The weather in a city."""
        return "sunny"
    g.firewall(**fw)
    return g, made


def call(name="book_train", **args):
    base = {"origin": "Paris", "destination": "Lyon", "date": "2026-11-03", "seats": 2} if name == "book_train" else {}
    return {"name": name, "arguments": {**base, **args}}


def test_a_grounded_call_is_allowed_and_every_slot_is_reported():
    g, made = make()
    d = g.call(call(travel_class="second", seat="aisle", refundable=True), REQ)
    assert d.outcome == "allow" and made == [("Paris", "Lyon", "2026-11-03", 2)]
    rep = d.slots
    assert rep["origin"]["status"] == "found" and rep["origin"]["where"] == "request"
    assert rep["seats"]["status"] == "found"                                   # 2 is written as a digit
    assert rep["travel_class"] == {"value": "second", "status": "exempt", "why": "enumeration"}
    assert rep["seat"]["status"] == "exempt" and rep["refundable"]["why"] == "boolean"
    assert "slots_grounded" not in d.failed


def test_a_slot_value_not_in_the_request_escalates_by_default_with_the_slot_named():
    g, made = make()
    d = g.call(call("weather", city="Berlin"), REQ)
    assert d.outcome == "escalate" and not made and d.failed == ["slots_grounded"]
    assert d.slots["city"]["status"] == "missing"
    assert any("city='Berlin'" in r and "not in the request" in r for r in d.reasons), d.reasons


def test_on_ungrounded_deny_refuses():
    g, _ = make(on_ungrounded="deny")
    d = g.call(call("weather", city="Berlin"), REQ)
    assert d.outcome == "deny" and d.failed == ["slots_grounded"]


def test_numbers_must_be_written_as_digits():
    g, _ = make()
    assert g.check(call(seats=3), REQ).outcome == "escalate"                   # 3 is nowhere
    words = [{"role": "user", "content": "Book two seats from Paris to Lyon on 2026-11-03."}]
    d = g.check(call(seats=2), words)
    assert d.outcome == "escalate" and d.slots["seats"]["status"] == "missing"  # "two" is not a digit


def test_strings_are_matched_after_normalisation_but_not_inside_a_longer_word():
    g, _ = make()
    ctx = [{"role": "user", "content": "book 2 seats from PARIS to lyon on 2026-11-03"}]
    assert g.check(call(), ctx).outcome == "allow"                             # case and spacing differ
    ctx2 = [{"role": "user", "content": "book 2 seats from Parisville to Lyon on 2026-11-03"}]
    d = g.check(call(), ctx2)
    assert d.outcome == "escalate" and d.slots["origin"]["status"] == "missing"


def test_the_assistants_own_words_do_not_ground_a_slot():
    g, _ = make()
    ctx = REQ + [{"role": "assistant", "content": "Shall I check the weather in Berlin?"}]
    assert g.check(call("weather", city="Berlin"), ctx).outcome == "escalate"
    g2, _ = make(sources=("user", "assistant"))
    assert g2.check(call("weather", city="Berlin"), ctx).outcome == "allow"


def test_a_value_the_app_gives_as_a_fact_grounds_a_slot_unless_facts_false():
    g, _ = make()
    d = g.check(call("weather", city="Berlin"), REQ, facts={"home_city": "Berlin"})
    assert d.outcome == "allow" and d.slots["city"]["where"] == "fact home_city"
    g2, _ = make(facts=False)
    assert g2.check(call("weather", city="Berlin"), REQ, facts={"home_city": "Berlin"}).outcome == "escalate"


def test_exempt_arguments_by_declaration():
    g, _ = make(exempt=["city"])
    d = g.check(call("weather", city="Berlin"), REQ)
    assert d.outcome == "allow" and d.slots["city"] == {"value": "Berlin", "status": "exempt", "why": "declared"}


def test_defaults_the_call_did_not_give_are_not_slots():
    g, _ = make()
    d = g.check({"name": "book_train", "arguments": {"origin": "Paris", "destination": "Lyon", "date": "2026-11-03"}},
                REQ)
    assert d.outcome == "allow" and set(d.slots) == {"origin", "destination", "date"}


def test_nested_and_list_slots_are_checked_item_by_item():
    g = Guard()

    class Leg(BaseModel):
        city: str
        nights: int

    @g.tool
    def plan_trip(legs: list[Leg], tags: list[str]) -> str:
        """Plan a trip."""
        return "ok"
    g.firewall("plan_trip")
    ctx = [{"role": "user", "content": "Plan Rome for 3 nights, then Milan for 2. Tag it summer."}]
    ok = g.check({"name": "plan_trip", "arguments": {"legs": [{"city": "Rome", "nights": 3},
                                                              {"city": "Milan", "nights": 2}], "tags": ["summer"]}}, ctx)
    assert ok.outcome == "allow" and ok.slots["legs[1].city"]["status"] == "found"
    bad = g.check({"name": "plan_trip", "arguments": {"legs": [{"city": "Rome", "nights": 4}], "tags": ["summer"]}}, ctx)
    assert bad.outcome == "escalate" and bad.slots["legs[0].nights"]["status"] == "missing"


def test_the_schema_check_refuses_wrong_types_missing_required_and_values_outside_an_enum():
    g, made = make()
    assert g.check(call(seats="many"), REQ).outcome == "deny"
    assert g.check({"name": "book_train", "arguments": {"origin": "Paris"}}, REQ).outcome == "deny"
    d = g.check(call(travel_class="sleeper"), REQ)
    assert d.outcome == "deny" and d.failed[0] == "arguments_valid" and not made


def test_a_tool_whose_schema_cannot_be_read_is_refused_under_the_firewall():
    g = Guard()
    g.declare("query")
    g.adopt("query", {"type": "object", "properties": {"q": {"type": "string", "pattern": "("}}})
    assert g.check({"name": "query", "arguments": {"q": "x"}}, "x").outcome == "escalate"     # without the firewall
    g.firewall("query", grounded=False)
    d = g.check({"name": "query", "arguments": {"q": "x"}}, "x")
    assert d.outcome == "deny" and "schema_readable" in d.failed


def test_firewall_applies_to_tools_declared_later_and_to_named_tools_only():
    g = Guard()
    g.firewall("later")

    @g.tool
    def later(x: str) -> str:
        """."""
        return x

    @g.tool
    def other(x: str) -> str:
        """."""
        return x
    assert g.check({"name": "later", "arguments": {"x": "zzz"}}, "hello").outcome == "escalate"
    assert g.check({"name": "other", "arguments": {"x": "zzz"}}, "hello").outcome == "allow"


def test_bad_declarations_raise():
    g, _ = make()
    with pytest.raises(ValueError):
        g.firewall(on_ungrounded="ask")
    with pytest.raises(ValueError):
        g.firewall(sources=("nobody",))
    with pytest.raises(ValueError):
        g.firewall(match="regex")
    g.firewall("no_such_tool")                         # a typo in a tool's name: raised when a call is checked
    with pytest.raises(ValueError, match="no_such_tool"):
        g.check(call("weather", city="Paris"), REQ)


def test_the_decision_replays_and_the_report_is_in_the_trace():
    g, _ = make()
    d = g.check(call("weather", city="Berlin"), REQ, facts={"home_city": "Rome"})
    assert d.replay()["ok"]
    assert d.response.values["slot_grounding"]["missing"] == ["city='Berlin'"]
    json.dumps(d.response.values["slot_grounding"])                            # plain data


def test_the_message_tells_the_agent_what_to_do():
    g, _ = make()
    d = g.check(call("weather", city="Berlin"), REQ)
    assert "request" in d.advice()


def test_the_benchmark_reports_three_arms():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "guard_firewall.py"
    spec = importlib.util.spec_from_file_location("guard_firewall", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rep = mod.run()
    assert set(rep["arms"]) == {"none", "schema", "schema+grounded"}
    a = rep["arms"]
    assert a["none"]["false_call_rate"] >= a["schema"]["false_call_rate"] >= a["schema+grounded"]["false_call_rate"]
    assert a["schema+grounded"]["false_call_rate"] < a["none"]["false_call_rate"]
    assert rep["n"]["irrelevant"] > 0 and rep["n"]["relevant"] > 0 and rep["n"]["near_miss"] > 0
    assert 0.0 <= a["schema+grounded"]["relevant_allowed"] <= 1.0
    assert rep == mod.run()                                                    # deterministic


# ───────────────────────────────────────────── the firewall reads numbers under the tool's locale; "/" between words
def _locale_guard(**fw):
    g = Guard()

    @g.tool(locale="en")
    def transfer(amount: float) -> str:
        return "ok"

    @g.tool
    def pay(amount: float) -> str:
        return "ok"

    @g.tool
    def autofill(info_type: str) -> str:
        return "ok"
    g.firewall(**fw)
    return g


def _out(g, name, args, text):
    return g.check({"name": name, "arguments": args}, [{"role": "user", "content": text}], store=False).outcome


def test_the_firewall_reads_a_grouped_number_under_the_tools_locale():
    g = _locale_guard()
    assert _out(g, "transfer", {"amount": 3000}, "transfer $3,000 now") == "allow"
    assert _out(g, "transfer", {"amount": 3000}, "transfer $3000 now") == "allow"
    assert _out(g, "transfer", {"amount": 3}, "transfer $3,000 now") == "escalate"
    assert _out(g, "pay", {"amount": 3000}, "pay $3,000 now") == "escalate"       # no locale: "3,000" grounds nothing
    g = _locale_guard(locale="de")                                              # the firewall's own locale wins
    assert _out(g, "pay", {"amount": 3000}, "zahle 3.000 EUR") == "allow"
    assert _out(g, "transfer", {"amount": 3000}, "transfer $3,000 now") == "escalate"
    assert _out(g, "transfer", {"amount": 3}, "überweise 3,000 EUR") == "allow"
    g = _locale_guard(locale=None)                                              # None: no locale, also not the tool's
    assert _out(g, "transfer", {"amount": 3000}, "transfer $3,000 now") == "escalate"
    with pytest.raises(ValueError, match="locale"):
        _locale_guard(locale="xx")


def test_the_firewall_splits_a_slash_between_two_words_but_not_a_path_or_url():
    g = _locale_guard()
    assert _out(g, "autofill", {"info_type": "address"}, "fill my personal/address info") == "allow"
    assert _out(g, "autofill", {"info_type": "personal"}, "fill my personal/address info") == "allow"
    assert _out(g, "autofill", {"info_type": "address"}, "fill my personal address info") == "allow"
    for text in ("open /home/address now", "see https://x.org/address", "read docs/v2/address", "see x.org/address",
                 "file a/address.txt", "the a/b/address form"):
        assert _out(g, "autofill", {"info_type": "address"}, text) == "escalate", text
