"""solvi.textin: a free text → the entry point (a question) and its typed input state, read with quotes; deterministic
parsers; missing fields for a clarifying question; provenance in the trace and the audit (read by a model, never given);
an unsure entry point escalates; a dialogue turn changes fields. A keyword scorer stands in for the decider."""
import datetime as dt
from typing import Literal

import numpy as np
import pytest

from solvi import Catalog, Response, System
from solvi.decide import DecideModel
from solvi.textin import (CueExtractor, DeciderExtractor, ParseError, TextIn, parse_bool, parse_date, parse_enum,
                          parse_number, parse_text, replay_record)

TODAY = dt.date(2026, 9, 28)
KEYS = {"request_refund": ["refund", "money back", "charged"], "change_delivery_address": ["address", "deliver"],
        "cancel_order": ["cancel"]}


class RouteScorer:
    """Keyword logits over the entry points (3 per keyword found in the text)."""
    model_id = "test/route-scorer"

    def fingerprint(self):
        return "route-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            z = np.array([3.0 * sum(low.count(k) for k in KEYS.get(o, [])) for o in it.options])
            out.append(np.stack([z, z - 1.0], 1))
        return out


def decider():
    return DecideModel(RouteScorer(), meta={"format": "test", "temperature": 1.0})


def shop():
    cat = Catalog()

    @cat.check(hard=True, then={"request_refund": "reject", "cancel_order": "keep", "change_delivery_address": "refused"})
    def order_exists(order_id: str) -> bool:
        """Orders are A-<digits>."""
        return order_id.startswith("A-")

    @cat.fn
    def days_since_purchase(purchase_date: dt.date) -> int:
        return (TODAY - purchase_date).days

    @cat.fn
    def amount_eur(amount: float, currency: Literal["EUR", "USD", "RUB"]) -> float:
        return amount * {"EUR": 1.0, "USD": 0.9, "RUB": 0.01}[currency]

    @cat.rule("request_refund")
    def refund(order_exists: bool, days_since_purchase: int, amount_eur: float) -> Literal["approve", "review", "reject"]:
        if days_since_purchase > 30:
            return "reject"
        return "review" if amount_eur > 1000 else "approve"

    @cat.rule("change_delivery_address")
    def change(order_exists: bool, new_address: str) -> Literal["changed", "refused"]:
        return "changed" if len(new_address) > 5 else "refused"

    @cat.rule("cancel_order")
    def cancel(order_exists: bool, urgent: bool) -> Literal["cancelled", "keep"]:
        return "cancelled" if urgent else "keep"

    from solvi import Question
    qs = [Question("request_refund", "Refund an order"), Question("change_delivery_address", "Change where an order is delivered"),
          Question("cancel_order", "Cancel an order")]
    return cat, System(cat, qs)


def textin(system, **kw):
    return TextIn(system, decider(), today=TODAY, patterns={"order_id": r"[A-Z]-\d+"},
                  synonyms={"currency": {"EUR": ["euro", "euros", "€"], "RUB": ["rubles", "руб", "₽"], "USD": ["dollars", "$"]}},
                  **kw)


REFUND = "Hi, please refund order A-10457: I paid 1.5 million rubles on 12 September and it arrived broken."


# --------------------------------------------------------------------------------------------------- entry points
def test_entry_points_list_questions_with_typed_inputs():
    _, s = shop()
    eps = {e.name: e for e in s.entry_points()}
    assert set(eps) == {"request_refund", "change_delivery_address", "cancel_order"}
    r = eps["request_refund"]
    assert set(r.fields) == {"order_id", "amount", "currency", "purchase_date"}
    assert r.fields["amount"].type is float and r.fields["purchase_date"].type is dt.date
    assert set(r.required) == {"order_id", "amount", "currency", "purchase_date"}
    t = r.tool()
    assert t["function"]["name"] == "request_refund" and "amount" in t["function"]["parameters"]["properties"]
    assert [e.name for e in s.entry_points(["cancel_order"])] == ["cancel_order"]
    with pytest.raises(KeyError):
        s.entry_points(["nope"])


# --------------------------------------------------------------------------------------------------- parsers
@pytest.mark.parametrize("s,want", [("1.5 million", "1500000"), ("1,500.50", "1500.5"), ("1 500 000 руб", "1500000"),
                                    ("2k", "2000"), ("полтора миллиона", "1500000"), ("12,5", "12.5"), ("€ 99", "99"),
                                    ("a million", "1000000"), ("1.500.000", "1500000"), ("3 млн", "3000000"), ("-4", "-4")])
def test_parse_number(s, want):
    assert parse_number(s) == want


def test_parse_number_rejects():
    for s in ("no digits", "1 and 2", ""):
        with pytest.raises(ParseError):
            parse_number(s)
    with pytest.raises(ParseError, match="whole"):
        parse_number("1.5", {"integer": True})


def test_parse_date():
    t = {"today": "2026-09-28"}
    assert parse_date("2026-09-12") == "2026-09-12"
    assert parse_date("12.09.2026") == "2026-09-12" and parse_date("09/12/26", {"dayfirst": False, **t}) == "2026-09-12"
    assert parse_date("12 September 2026") == parse_date("September 12, 2026") == parse_date("12th of Sep 2026") == "2026-09-12"
    assert parse_date("12 сентября", t) == "2026-09-12" and parse_date("yesterday", t) == "2026-09-27"
    with pytest.raises(ParseError, match="no year"):
        parse_date("12 September")                         # never a guessed year
    with pytest.raises(ParseError):
        parse_date("31.02.2026")
    with pytest.raises(ParseError, match="more than one"):
        parse_date("2026-09-12 or 2026-09-13")


def test_parse_bool_enum_text():
    assert parse_bool("yes") is True and parse_bool("нет") is False
    assert parse_bool("urgent", {"cues": ["urgent"]}) is True
    assert parse_bool("not urgent", {"cues": ["urgent"], "negatives": ["not urgent"]}) is False
    lab = {"EUR": ["euro", "€"], "RUB": ["rubles"]}
    assert parse_enum("Euro", {"labels": lab}) == "EUR" and parse_enum("5 rubles", {"labels": lab}) == "RUB"
    with pytest.raises(ParseError):
        parse_enum("euro or rubles", {"labels": lab})
    assert parse_text(' "Main St 5". ') == "Main St 5"
    with pytest.raises(ParseError):
        parse_text("abc", {"pattern": r"[A-Z]-\d+"})


# --------------------------------------------------------------------------------------------------- reading
def test_routing_and_fields_with_quotes():
    _, s = shop()
    read = textin(s).read(REFUND)
    assert read.question == "request_refund" and read.route["by"] == "decider" and read.ok
    assert read.state == {"order_id": "A-10457", "amount": 1500000.0, "currency": "RUB",
                          "purchase_date": dt.date(2026, 9, 12)}
    for f, r in read.fields.items():
        assert REFUND[r.quote.start:r.quote.end] == r.quote.value, f
    assert read.fields["amount"].quote.value == "1.5 million" and read.fields["currency"].quote.value == "rubles"
    assert read.clarify() is None


def test_missing_required_fields_are_listed_not_guessed():
    _, s = shop()
    tin = textin(s)
    read = tin.read("Please refund order A-10457, I paid 20 euros.")
    assert read.question == "request_refund" and read.missing == ["purchase_date"]
    assert read.fields["purchase_date"].status == "not_stated" and "purchase date" in read.clarify()
    res = s.ask_text(read)
    r = res["request_refund"]
    assert r.status == "abstain" and "not stated in the text: purchase_date" in r.why
    assert res.textin is read
    # a year-less date without today= is not guessed either: unparsed, and the clarifying question says what was read
    read2 = TextIn(s, decider(), patterns={"order_id": r"[A-Z]-\d+"}).read("refund A-1: 20 EUR on 12 September")
    assert read2.fields["purchase_date"].status == "unparsed" and "no year" in read2.fields["purchase_date"].why
    assert "'12 September'" in read2.clarify()


def test_ask_text_one_trace_with_provenance_and_audit():
    _, s = shop()
    res = s.ask_text(REFUND, textin=textin(s))
    assert res["request_refund"].answer == "review"          # 16 days ago, 1.5 million RUB = 15 000 EUR
    tr = res.trace
    kinds = [(r.kind, r.name) for r in tr.records]
    assert ("textin", "textin") in kinds and ("textin", "textin:amount") in kinds
    assert tr.init["request_text"] == REFUND and tr.init["amount"] == 1500000.0
    rec = next(r for r in tr.records if r.name == "textin:amount")
    assert rec.origin == "quoted" and rec.model["type"] == "CueExtractor" and rec.extra["parser"] == "number"
    route = next(r for r in tr.records if r.name == "textin")
    assert route.origin == "decided" and route.model["id"] == "test/route-scorer" and route.value == "request_refund"
    au = res.audit("request_refund")
    names = {q["name"] for q in au.quoted}
    assert {"amount", "currency", "purchase_date", "order_id"} <= names
    assert not any(g["name"] in names for g in au.given)     # read from the text: never "given"
    assert au.counts["quoted_by_model"] >= 4 and au.counts["decided"] == 1 and au.share_deterministic < 1
    assert "quoted by model" in au.support_line() and "entry point" in str(au)
    assert res.model_outputs >= 5 and s.stats["model_outputs"] >= 5
    rep = tr.replay(s)
    assert rep["ok"], rep["mismatches"]
    assert ("textin:amount" in {n for _, n, _ in rep["models"]})
    # the round trip keeps the typed values of the read fields (a date) and still replays
    r2 = Response.from_json(res.to_json(), catalog=s)
    assert r2.trace.replay(s)["ok"]


def test_answer_confidence_is_capped_by_the_reading():
    _, s = shop()
    res = s.ask_text("refund A-7: 20 EUR paid 2026-09-20", textin=textin(s))
    r = res["request_refund"]
    assert r.answer == "approve" and r.confidence <= res.textin.route["confidence"] + 1e-12


def test_replay_catches_a_value_that_is_not_what_the_text_says():
    _, s = shop()
    res = s.ask_text(REFUND, textin=textin(s))
    rec = next(r for r in res.trace.records if r.name == "textin:amount")
    assert replay_record(rec, res.trace.init) == []
    bad = dict(res.trace.init, amount=15.0)
    assert any("not the value read" in why for *_, why in replay_record(rec, bad))
    moved = dict(res.trace.init, request_text="x" + REFUND)
    assert any("not at" in why for *_, why in replay_record(rec, moved))


def test_unsure_entry_point_escalates_and_nothing_runs():
    _, s = shop()
    tin = textin(s)
    res = s.ask_text("Cancel it, or maybe a refund instead?", textin=tin)
    assert res.textin.question is None and "entry point unsure" in res.textin.escalated
    assert set(res.results) == {"cancel_order", "request_refund"}
    assert all(r.status == "abstain" and r.guard == "escalated" for r in res.results.values())
    assert [r.kind for r in res.trace.records] == ["textin"]  # only the routing record: no part ran
    assert [e["kind"] for e in res.safeguards] == ["escalated"] and s.stats["model_escalated"] == 1
    assert res.textin.clarify().startswith("Which of these")
    assert res.trace.replay(s)["ok"]


def test_wrong_entry_point_escalates_instead_of_guessing():
    _, s = shop()
    res = s.ask_text("What will the weather be like tomorrow?", textin=textin(s))
    assert res.textin.question is None and not res.complete
    assert all(r.guard == "escalated" for r in res.results.values())


def test_question_given_skips_routing():
    _, s = shop()
    res = s.ask_text("A-12 cancel, it is urgent", textin=textin(s), question="cancel_order")
    assert res["cancel_order"].answer == "cancelled" and res.textin.route["by"] == "given"
    res = s.ask_text("A-12 cancel, not urgent", textin=textin(s, negatives={"urgent": ["not urgent"]}),
                     question="cancel_order")
    assert res["cancel_order"].answer == "keep" and res.textin.fields["urgent"].value is False
    with pytest.raises(KeyError):
        textin(s).read("x", question="nope")


def test_dialogue_update_changes_fields_with_quotes():
    _, s = shop()
    tin = textin(s)
    r1 = tin.read("Please change the delivery address of order A-10457.")
    assert r1.question == "change_delivery_address" and r1.missing == ["new_address"]
    r2 = tin.update(r1, "The new address is 5 Main Street, Springfield. Sorry, the order is not A-10457 but A-10475.")
    ch = {c.field: c for c in r2.changes}
    assert ch["order_id"].old == "A-10457" and ch["order_id"].new == "A-10475"
    assert ch["new_address"].old is None and ch["new_address"].new == "5 Main Street, Springfield"
    for f, r in r2.fields.items():                          # quotes point into the whole dialogue
        assert r2.text[r.quote.start:r.quote.end] == r.quote.value, f
    assert r2.ok and r2.fields["order_id"].was == "A-10457"
    res = s.ask_text(r2)
    assert res["change_delivery_address"].answer == "changed" and res.trace.replay(s)["ok"]
    rec = next(r for r in res.trace.records if r.name == "textin:order_id")
    assert rec.extra["was"] == "'A-10457'"
    # a turn that does not restate a field keeps it
    r3 = tin.update(r2, "Thanks!")
    assert r3.state == r2.state and r3.changes == []
    # from a plain state dict: the given fields stay given
    r4 = tin.update({"order_id": "A-1"}, "new address: 7 Oak Road", question="change_delivery_address")
    assert r4.fields["order_id"].status == "given" and r4.state["new_address"] == "7 Oak Road"
    res4 = s.ask_text(r4)
    au = res4.audit("change_delivery_address")
    assert "order_id" in {g["name"] for g in au.given} and "new_address" in {q["name"] for q in au.quoted}
    assert res4.trace.replay(s)["ok"]


def test_the_call_is_data():
    """A text cannot name anything but an entry point, and values are only what the parsers produce."""
    _, s = shop()
    read = textin(s).read("refund __import__('os').system('rm -rf /') A-1 20 EUR 2026-09-20")
    assert read.question in {e.name for e in s.entry_points()}
    assert all(isinstance(v, (str, float, dt.date, bool)) for v in read.state.values())


def test_decider_pointer_as_the_extractor():
    from test_primitives import L14G, L14gStub
    m = DecideModel(L14gStub(), L14G)
    ex = DeciderExtractor(m)
    cat = Catalog()

    @cat.rule("pay")
    def pay(amount: float) -> Literal["ok", "big"]:
        return "big" if amount > 1000 else "ok"
    from solvi import Question
    s = System(cat, [Question("pay", "Pay an invoice")])
    tin = TextIn(s, m, extractor=ex)
    assert isinstance(tin.extractors[0], DeciderExtractor)
    res = s.ask_text("Invoice. Amount 1250.50 EUR due.", textin=tin)
    assert res.textin.route["by"] == "single" and res["pay"].answer == "big"
    rec = next(r for r in res.trace.records if r.name == "textin:amount")
    assert rec.model["type"] == "DeciderExtractor" and rec.extra["quoted"] == "1250.50"
    assert res.audit("pay").counts["quoted_by_model"] == 1 and res.trace.replay(s)["ok"]
    res2 = s.ask_text("unclear note", textin=tin)
    assert res2.textin.fields["amount"].status == "not_stated" and res2["pay"].status == "abstain"
    with pytest.raises(ValueError, match="pointer"):
        DeciderExtractor(decider())


def test_cue_extractor_picks_the_number_after_its_cue():
    _, s = shop()
    tin = textin(s)
    fs = tin.spec("request_refund", "amount")
    qs = CueExtractor().find("order A-5, 3 items, amount 250 EUR", fs)
    assert qs[0].value == "250" and qs[0].confidence == 1.0


# --------------------------------------------------------------------------------------------------- fixes before 0.7
def test_bool_description_words_never_decide_true():
    """A bool field's description words rank candidates but never make it True; only its name and cues= do."""
    from solvi.textin import EntryField, field_spec
    fs = field_spec(EntryField("urgent", bool, "Whether the customer asked for express handling", True))
    assert fs.cues == ["urgent"] and "customer" in fs.hints
    with pytest.raises(ParseError):
        parse_bool("customer", fs.parser_spec())
    assert CueExtractor().find("The customer asked to cancel order A-1.", fs) == []
    qs = CueExtractor().find("The customer says it is urgent.", fs)
    assert [q.value for q in qs] == ["urgent"] and parse_bool(qs[0].value, fs.parser_spec()) is True
    fs2 = field_spec(EntryField("is_urgent", bool, None, True), cues=["asap"])
    assert fs2.cues == ["is urgent", "urgent", "asap"] and parse_bool("asap", fs2.parser_spec()) is True


@pytest.mark.parametrize("text", ["A-12 cancel, it isn't urgent", "A-12 cancel, not at all urgent",
                                  "A-12 cancel, it is not really urgent", "A-12 cancel, hardly urgent",
                                  "A-12 cancel, never urgent", "A-12 cancel, I don't think it's urgent"])
def test_bool_negated_cue_is_unparsed_not_true(text):
    _, s = shop()
    read = textin(s).read(text, question="cancel_order")
    f = read.fields["urgent"]
    assert f.status == "unparsed" and f.value is None and "urgent" in read.missing
    res = s.ask_text(read)
    assert res["cancel_order"].status == "abstain"


def test_bool_negation_russian_and_declared_negative():
    fs_spec = {"cues": ["срочно"]}
    for q in ("не срочно", "ни разу не срочно"):
        with pytest.raises(ParseError, match="negates"):
            parse_bool(q, fs_spec)
    assert parse_bool("срочно", fs_spec) is True
    assert parse_bool("не срочно", {**fs_spec, "negatives": ["не срочно"]}) is False
    _, s = shop()
    tin = textin(s, cues={"urgent": ["срочно"]}, negatives={"urgent": ["не срочно"]})
    r = tin.read("A-12 отмените, это не срочно", question="cancel_order")
    assert r.fields["urgent"].value is False and r.fields["urgent"].quote.value == "не срочно"
    r = textin(s, cues={"urgent": ["срочно"]}).read("A-12 отмените, это не срочно", question="cancel_order")
    assert r.fields["urgent"].status == "unparsed"
    # a negation in another clause does not touch the cue
    r = textin(s).read("I do not want a refund, cancel A-12, it is urgent", question="cancel_order")
    assert r.fields["urgent"].value is True


def test_ask_text_rederives_values_from_quotes():
    """A caller-built TextRead cannot carry a value its quote does not state: ask_text re-derives each field."""
    import dataclasses
    _, s = shop()
    read = textin(s).read("refund A-7: 500 EUR paid 2026-09-20")
    assert read.fields["amount"].quote.value == "500"
    forged = dataclasses.replace(read, fields=dict(read.fields, amount=dataclasses.replace(read.fields["amount"],
                                                                                          value=5_000_000.0)))
    res = s.ask_text(forged)
    f = res.textin.fields["amount"]
    assert f.status == "unparsed" and "re-derive" in f.why and "amount" in res.textin.missing
    assert "amount" not in res.trace.init and res["request_refund"].status == "abstain"
    assert forged.fields["amount"].value == 5_000_000.0          # the caller's object is not changed
    assert res.trace.replay(s)["ok"]
    # a quote that is not in the text, or a canonical form the quote does not parse to
    moved = dataclasses.replace(read, fields=dict(read.fields, amount=dataclasses.replace(
        read.fields["amount"], quote=dataclasses.replace(read.fields["amount"].quote, value="900"))))
    assert s.ask_text(moved).textin.fields["amount"].status == "unparsed"
    canon = dataclasses.replace(read, fields=dict(read.fields, amount=dataclasses.replace(
        read.fields["amount"], canonical="5000000", value=5_000_000.0)))
    assert s.ask_text(canon).textin.fields["amount"].status == "unparsed"
    # an honest read goes through as it is
    res = s.ask_text(read)
    assert res.textin is read and res.trace.init["amount"] == 500.0 and res["request_refund"].answer == "approve"


def test_replay_rebuilds_the_typed_value_from_the_quote():
    import dataclasses
    _, s = shop()
    res = s.ask_text("refund A-7: 500 EUR paid 2026-09-20", textin=textin(s))
    rec = next(r for r in res.trace.records if r.name == "textin:amount")
    assert rec.extra["vtype"] == "float" and replay_record(rec, res.trace.init) == []
    forged = dataclasses.replace(rec, value=5_000_000.0)
    bad = replay_record(forged, dict(res.trace.init, amount=5_000_000.0))
    assert any("not what the quote parses to" in why for *_, why in bad)
    d = next(r for r in res.trace.records if r.name == "textin:purchase_date")
    assert replay_record(dataclasses.replace(d, value=dt.date(2026, 9, 21)),
                         dict(res.trace.init, purchase_date=dt.date(2026, 9, 21)))


def test_dialogue_correction_that_does_not_parse_is_a_conflict():
    _, s = shop()
    tin = TextIn(s, decider(), patterns={"order_id": r"[A-Z]-\d+"})          # no today=: "12 October" does not read
    r1 = tin.read("refund A-7: 500 EUR paid 2026-09-20")
    assert r1.ok
    r2 = tin.update(r1, "sorry, it was paid on 12 October")
    f = r2.fields["purchase_date"]
    assert f.status == "conflict" and f.quote.value == "12 October" and f.was == dt.date(2026, 9, 20)
    assert "purchase_date" not in r2.state and "purchase_date" in r2.missing and not r2.ok
    assert "12 October" in r2.clarify() and "purchase date" in r2.clarify()
    res = s.ask_text(r2)
    assert res["request_refund"].status == "abstain" and "purchase_date" not in res.trace.init
    assert res.trace.replay(s)["ok"]
    assert [c.field for c in r2.changes] == []               # nothing changed to a value: it is asked instead


@pytest.mark.parametrize("s", ["5 m long", "2 b", "1.000", "12.345", "3 100", "5%", "12 percent", "a b"])
def test_parse_number_refuses_ambiguous_forms(s):
    with pytest.raises(ParseError):
        parse_number(s)


def test_parse_number_ambiguity_resolved_by_context():
    assert parse_number("5m") == "5000000" and parse_number("$5 m") == "5000000" and parse_number("2bn") == "2000000000"
    assert parse_number("1,000") == "1000" and parse_number("0.125") == "0.125" and parse_number("1.5") == "1.5"
    assert parse_number("1.000", {"decimal": ","}) == "1000" and parse_number("1.000", {"decimal": "."}) == "1"
    assert parse_number("1,000", {"decimal": ","}) == "1" and parse_number("1.234,5", {"decimal": ","}) == "1234.5"
    assert parse_number("1 500 000 руб") == "1500000" and parse_number("$3 100") == "3100"
    assert parse_number("1\u00a0500\u00a0000") == "1500000"      # a no-break space groups digits on purpose
    assert parse_number("5%", {"percent": True}) == "5" and parse_number("12.5 percent", {"percent": True}) == "12.5"


def test_number_fields_in_a_text_are_not_misread():
    _, s = shop()
    tin = textin(s)
    r = tin.read("refund A-7: I paid 1 500 000 руб on 2026-09-20")
    assert r.fields["amount"].value == 1500000.0 and r.fields["amount"].quote.value == "1 500 000 руб"
    assert s.ask_text(r).trace.replay(s)["ok"]
    r = tin.read("refund A-7: the box is 5 m long, 20 EUR paid 2026-09-20")
    assert r.fields["amount"].value != 5_000_000.0
    r = tin.read("refund A-7: amount 1.000 EUR paid 2026-09-20")
    assert r.fields["amount"].status == "unparsed" and "ambiguous" in r.fields["amount"].why
    r = textin(s, decimal=",").read("refund A-7: amount 1.000 EUR paid 2026-09-20")
    assert r.fields["amount"].value == 1000.0 and r.fields["amount"].spec == {"decimal": ","}
    res = s.ask_text(r)
    assert res.trace.replay(s)["ok"]
    r = tin.read("refund A-7: amount 5% EUR paid 2026-09-20")
    assert r.fields["amount"].status == "unparsed" and "percent" in r.fields["amount"].why


def test_two_digit_year_needs_today_and_a_window():
    with pytest.raises(ParseError, match="two-digit year"):
        parse_date("01.02.85")
    t = {"today": "2026-09-28"}
    assert parse_date("01.02.85", t) == "1985-02-01" and parse_date("01.02.30", t) == "2030-02-01"
    assert parse_date("01.02.46", t) == "2046-02-01" and parse_date("01.02.47", t) == "1947-02-01"
    assert parse_date("01.02.26", t) == "2026-02-01"


def test_a_spelled_out_number_that_goes_on_is_refused_not_cut():
    """Only a number with one scale word is read. "две тысячи триста" was read as 2000 and "one hundred fifty" as 100:
    the words after the scale were dropped."""
    from solvi.textin import ParseError, parse_number
    for text in ("две тысячи триста", "one hundred fifty", "one hundred and fifty", "триста две тысячи",
                 "ten thousand and one nights"):
        with pytest.raises(ParseError, match="a number in several words"):
            parse_number(text)
    for text, want in (("две тысячи", "2000"), ("two thousand euros", "2000"), ("полтора миллиона", "1500000"),
                       ("пять тысяч рублей", "5000"), ("twelve hundred", "1200"), ("1.5 million", "1500000"),
                       ("2 thousand items for five people", "2000"), ("about three hundred people in two groups", "300")):
        assert parse_number(text) == want, text
