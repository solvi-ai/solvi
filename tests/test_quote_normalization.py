"""Quotes matched on a normalized view (1.0): a model's quote that differs from its source only in how characters are
typed — no-break spaces and hyphens, "…" for "...", dashes, whitespace runs, curly quotes, NFKC forms — is accepted;
the stored quote is the source's own substring at offsets into the original text, the record says the view was needed,
and replay re-checks it. Several labelled sources: a quote records which one it came from. Catalog(quotes="literal")
keeps the 0.9 rule."""
import pytest

from solvi import Answer, Catalog, Claim, Decision, Question, Quote, System
from solvi.core import find_quote
from solvi.core.provenance import NORM_FORM, find_normalized, matches_normalized, norm_view, normalized, part_fingerprint
from solvi.core.store import open_storage
from solvi.core.types import Span

NOTES = "Link said: the key is under the‑stone…  “really”."     # nbsp, nb hyphen, …, curly


class FakeModel:
    """A model behind an extract part (its quotes are strict)."""
    deterministic = True


# --- the normalized view
@pytest.mark.parametrize("written, source", [
    ("the key is under the-stone...", "the key is under the‑stone…"),       # the field report's cases
    ("10 000 EUR", "10 000 EUR"),                                                # no-break / narrow spaces
    ("pages 3-5", "pages 3–5"),                                                       # en dash
    ("a - b", "a — b"),                                                               # em dash
    ("x = -1", "x = −1"),                                                             # minus sign
    ('"yes," he said', "“yes,” he said"),                                        # curly double quotes
    ("it's", "it’s"),                                                                 # curly apostrophe
    ('"oui"', "«oui»"),                                                           # angle quotes
    ("one two", "one \n\t two"),                                                           # whitespace runs
    ("office", "oﬃce"),                                                               # ligature (NFKC)
    ("ABC 123", "ＡＢＣ １２３"),                                  # full-width (NFKC)
    ("café", "café"),                                                           # combining accent (NFKC)
    ("zero width", "zero​ width"),                                                    # zero-width space
])
def test_the_normalized_view_equates_how_characters_are_typed(written, source):
    assert matches_normalized(written, source)
    at = find_normalized("before " + source + " after", written)
    assert at is not None and ("before " + source + " after")[at[0]:at[1]] == source


def test_the_view_keeps_what_the_text_says():
    assert not matches_normalized("the key", "the Key")                  # case is not normalized
    assert not matches_normalized("3", "30")
    assert find_normalized("it was 30 EUR", "3") is None              # whole numbers: "3" is not "30"
    assert find_normalized("category", "cat") is None
    view, starts, ends = norm_view("a  b…")
    assert view == "a b..." and [starts[i] for i in range(len(view))] == [0, 1, 3, 4, 4, 4]
    assert ends[1] == 3                                                    # one space for the whole run
    assert normalized("  x　y ") == "x y"


# --- find_quote: several labelled sources
def test_find_quote_names_the_source_and_keeps_its_own_text():
    q = find_quote('the key is under the-stone... "really"', {"dialogues": "Zelda: go west", "notes": NOTES})
    assert q.source == "notes" and q.value == NOTES[q.start:q.end] == "the key is under the‑stone…  “really”"
    assert find_quote("go west", {"dialogues": "Zelda: go west", "notes": NOTES}) == Quote("go west", 7, 14, "dialogues")
    assert find_quote("under the-stone", NOTES).source == "doc"
    assert find_quote("under the-stone", {"notes": NOTES}, quotes="literal") is None
    assert find_quote("not there", {"notes": NOTES}) is None
    with pytest.raises(ValueError, match="normalized"):
        find_quote("x", "x", quotes="loose")


def test_a_literal_occurrence_wins_over_an_earlier_normalized_one():
    text = "a b, then a b"
    assert find_quote("a b", text) == Quote("a b", 10, 13, "doc")


# --- evidence of a check over several sources (the field report: "this quote is not from the notes")
def _quote_check(quotes="normalized", evidence="under the-stone..."):
    cat = Catalog(quotes=quotes)

    @cat.check
    def quoted(plan: str, notes: str, dialogues: str) -> bool:
        return Claim(True, evidence=[evidence])

    @cat.rule("ok")
    def ok(quoted: bool) -> bool:
        return quoted
    return System(cat, [Question("ok", "A quoted plan?", Answer.yes_no())])


STATE = {"plan": "lift the stone", "notes": NOTES, "dialogues": "Zelda: go west"}


def test_evidence_typed_otherwise_is_accepted_and_kept_as_the_source_text():
    s = _quote_check()
    res = s.ask(STATE)
    assert res["ok"].answer == "yes"
    rec = res.trace.records[0]
    assert rec.extra["evidence"] == [[22, 38, "notes", "under the‑stone…"]]
    assert NOTES[22:38] == "under the‑stone…"
    assert rec.extra["quote_match"] == {"form": NORM_FORM, "written": {"evidence 0": "under the-stone..."}}
    assert res.trace.replay(s, res.flow)["ok"]


def test_evidence_in_another_text_input_names_that_source():
    s = _quote_check(evidence="go west")
    res = s.ask(STATE)
    assert res["ok"].answer == "yes"
    assert res.trace.records[0].extra == {"evidence": [[7, 14, "dialogues", "go west"]]}     # literal: no note


def test_literal_mode_keeps_the_0_9_rule():
    s = _quote_check("literal")
    res = s.ask(STATE)
    assert res["ok"].status == "abstain" and "not grounded" in res["ok"].why
    assert _quote_check("literal", "go west").ask(STATE)["ok"].answer == "yes"   # other sources: searched in both modes
    with pytest.raises(ValueError, match="literal"):
        Catalog(quotes="exact")


def test_a_fabricated_quote_is_still_rejected():
    res = _quote_check(evidence="under the rock").ask(STATE)
    assert res["ok"].status == "abstain" and "not grounded" in res["ok"].why


def test_evidence_with_offsets_typed_otherwise_keeps_the_source_text():
    cat = Catalog()

    @cat.check
    def quoted(notes: str) -> bool:
        return Claim(True, evidence=[Quote("the key is under the-stone...", 11, 38, "notes")])

    @cat.rule("ok")
    def ok(quoted: bool) -> bool:
        return quoted
    s = System(cat, [Question("ok", "?", Answer.yes_no())])
    res = s.ask({"notes": NOTES})
    assert res["ok"].answer == "yes"
    assert res.trace.records[0].extra["evidence"] == [[11, 38, "notes", NOTES[11:38]]]
    assert res.trace.replay(s, res.flow)["ok"]


# --- a model's own quote, and spans
def _extract_system(written):
    doc = "Total: 120 EUR — paid in full…"
    cat = Catalog()

    @cat.extract(model=FakeModel())
    def status(doc):
        return Quote(written, 7, len(doc))

    @cat.rule("paid")
    def paid(status) -> bool:
        return "paid" in status

    @cat.rule("where")
    def where(doc) -> Span[str]:
        return "paid in full..."
    return System(cat, [Question("paid", "?"), Question("where", "?")]), doc


def test_a_models_quote_typed_otherwise_is_the_source_text():
    s, doc = _extract_system("120 EUR - paid in full...")
    res = s.ask({"doc": doc})
    rec = res.trace.records[0]
    assert res.values["status"] == doc[7:] and rec.quote == (7, len(doc), "doc")
    assert rec.extra == {"quote_match": {"form": NORM_FORM, "written": {"value": "120 EUR - paid in full..."}}}
    assert res["where"].answer == "paid in full…" and res["where"].evidence[0] == Quote(doc[17:], 17, len(doc), "doc")
    for trust in (False, True):
        assert res.trace.replay(s, res.flow, trust_models=trust)["ok"]


def test_a_models_wrong_quote_is_still_rejected():
    s, doc = _extract_system("120 EUR - unpaid")
    res = s.ask({"doc": doc})
    assert res["paid"].status == "abstain" and "not grounded" in res.trace.records[0].error


def test_an_exact_quote_records_as_in_0_9():
    s, doc = _extract_system("120 EUR — paid in full…")
    rec = s.ask({"doc": doc}).trace.records[0]
    assert rec.extra is None and rec.value == doc[7:]


def test_a_span_with_offsets_typed_otherwise_is_accepted():
    doc = "Due on 3 May"
    cat = Catalog()

    @cat.rule("due")
    def due(doc) -> Span[str]:
        return Quote("3 May", 7, 12)
    res = System(cat, [Question("due", "?")]).ask({"doc": doc})
    assert res["due"].answer == "3 May"


# --- stored, replayed, tampered
def test_a_stored_normalized_quote_verifies_and_replays(tmp_path):
    s = _quote_check()
    s.storage = open_storage(str(tmp_path / "d.jsonl"))
    s.storage.catalog = s
    s.ask(STATE)
    assert s.storage.verify()["ok"]
    assert s.storage.replay_all(s) == []
    st = next(iter(s.storage.iter()))
    res = s.storage.get(st.id, s)
    assert res.trace.replay(s, res.flow)["ok"]


def test_replay_rechecks_the_note():
    s, doc = _extract_system("120 EUR - paid in full...")
    res = s.ask({"doc": doc})
    rec = res.trace.records[0]
    rec.extra["quote_match"]["form"] = "nfkc-ws-dash-quote/99"
    out = res.trace.replay(s, res.flow, trust_models=True)
    assert not out["ok"] and any("unknown normalization" in m[2] for m in out["mismatches"])
    rec.extra["quote_match"] = {"form": NORM_FORM, "written": {"value": "120 EUR - unpaid"}}
    out = res.trace.replay(s, res.flow, trust_models=True)
    assert any("is not" in m[2] and "normalized view" in m[2] for m in out["mismatches"])


# --- fingerprints
def test_the_default_mode_keeps_fingerprints_and_literal_changes_them():
    def part(quotes):
        cat = Catalog(quotes=quotes) if quotes else Catalog()

        @cat.extract
        def name(doc):
            return Quote(doc[:3], 0, 3)
        return cat.parts["name"]
    assert part(None).quotes == "normalized"
    assert part(None).quotes == part("normalized").quotes
    assert part_fingerprint(part(None)) == part_fingerprint(part("normalized"))
    assert part_fingerprint(part(None)) != part_fingerprint(part("literal"))


def test_a_decision_with_evidence_typed_otherwise():
    cat = Catalog()

    @cat.rule("tone", model=FakeModel())
    def tone(doc) -> bool:
        return Decision("yes", {"yes": 0.9, "no": 0.1}, evidence=["\"thank you\""])
    s = System(cat, [Question("tone", "Polite?")])
    res = s.ask({"doc": "She wrote “thank you” twice."})
    assert res["tone"].answer == "yes"
    assert res["tone"].evidence[0].value == "“thank you”"
