import pytest
from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


def setup_book(mode="matched"):
    atoms = [
        Atom(f"{f}={v}", lambda s, f=f, v=v: s.get(f) == v, feature=f)
        for f, v in [("in", "leap"), ("beat", "on"), ("beat", "off"), ("beat", "mid")]
    ]
    grammar = Grammar(atoms, k=2)
    buffer = Buffer(grammar)
    for i in range(60):
        buffer.add(Transition({"in": "leap", "beat": "on" if i % 2 else "off"}, "ct", "yes", group=i, id=str(i)))
    book = LawSet(grammar, base=None, support_on=mode)
    narrow = [book.add(Candidate("ct", (f"beat={beat}", "in=leap"), "yes")) for beat in ["on", "off"]]
    return book, buffer, Candidate("ct", ("in=leap",), "yes"), narrow


@pytest.mark.parametrize("mode", ["matched", "applied"])
def test_zero_net_explains_observed_narrow_coverage_without_admitting(mode):
    book, buffer, candidate, narrow = setup_book(mode)
    cost, tested = dict(book.cost), buffer.tested
    result = book.why_not(candidate, buffer)
    assert result["failed"] == "net" and result["net"] == 0
    assert result["covered_by"] == [law.id for law in narrow]
    assert "held-out rows" in result["hint"] and "unseen" in result["hint"]
    assert dict(book.cost) == cost and buffer.tested == tested
    assert book.test([candidate], buffer) == []
    assert book.predict({"in": "leap", "beat": "mid"}, "ct") is None


def test_coverage_hint_excludes_other_action_effect_and_retired_laws():
    book, buffer, candidate, narrow = setup_book()
    retired = book.add(Candidate("ct", ("beat=on", "in=leap"), "yes"))
    book.retire(retired)
    book.add(Candidate("other", ("beat=on", "in=leap"), "yes"))
    book.add(Candidate("ct", ("beat=on", "in=leap"), "no"))
    result = book.why_not(candidate, buffer)
    assert result["failed"] == "net"
    assert result["covered_by"] == [law.id for law in narrow]
    assert retired.id not in result["covered_by"]


def test_empty_or_excluded_heldout_overlap_does_not_claim_coverage():
    book, buffer, candidate, narrow = setup_book()
    empty = Buffer(book.grammar)
    result = book.why_not(candidate, empty)
    assert result["failed"] == "net"
    assert "covered_by" not in result and "hint" not in result
    held = Buffer(book.grammar)
    held.add(Transition({"in": "leap", "beat": "on"}, "ct", "yes", group="excluded", id="held"))
    result = book.why_not(candidate, held, exclude="excluded")
    assert result["failed"] == "net"
    assert "covered_by" not in result and "hint" not in result
