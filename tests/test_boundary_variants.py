"""testing.boundary: None where a field has shown None, a value of another type the field has shown, Unicode
normal-form variants (decomposed accents, a compatibility ligature), and the cases taken one per input shape first."""
import unicodedata

from solvi import testing


def _notes(cases):
    return [c["note"] for c in cases]


def _states_for(cases, field, what):
    return [c["state"][field] for c in cases if c["note"] == f"boundary: {field}: {what}"]


def test_none_where_the_field_has_shown_none():
    seeds = [{"state": {"x": 3, "who": "ann"}}, {"state": {"x": None, "who": "bob"}}]
    out = testing.boundary(seeds, n=500)
    assert [{"x": None, "who": "ann"}] == [c["state"] for c in out if c["note"] == "boundary: x: none"]
    # a field that never showed None gets no None
    assert not any(c["note"] == "boundary: who: none" for c in out)


def test_another_type_the_field_has_shown():
    seeds = [{"state": {"id": 7}}, {"state": {"id": "7a"}}]
    out = testing.boundary(seeds, n=500)
    swapped = [c["state"] for c in out if c["note"] == "boundary: id: another type seen"]
    assert {"id": "7"} in swapped                    # the value itself in the other type (str / int confusion)
    seeds = [{"state": {"id": 7, "k": [1]}}, {"state": {"id": "x", "k": [2, 3]}}]
    swapped = [c["state"] for c in testing.boundary(seeds, n=500) if c["note"] == "boundary: id: another type seen"]
    assert {"id": "7", "k": [1]} in swapped and {"id": 7, "k": [2, 3]} in swapped     # no clean conversion: an example
    # one field of one type only: no type swap
    out1 = testing.boundary([{"state": {"id": 7}}, {"state": {"id": 9}}], n=500)
    assert not any("another type seen" in n for n in _notes(out1))


def test_unicode_normal_forms():
    out = testing.boundary([{"state": {"name": "cafe"}}], n=500)
    nfd = _states_for(out, "name", "decomposed accents (NFD)")
    assert len(nfd) == 1
    assert unicodedata.normalize("NFC", nfd[0]) != nfd[0]            # really decomposed
    assert unicodedata.normalize("NFD", nfd[0]) == nfd[0]
    lig = _states_for(out, "name", "compatibility ligature")
    assert lig and "ﬁ" in lig[0] and unicodedata.normalize("NFKC", lig[0]) != lig[0]


def test_one_case_per_input_shape_first():
    # twenty cases of one shape, then one where "tag" is a list: within a kind of change the list-shaped case comes
    # second, not after the twenty (before, the second round of every kind took case 2, 3, ... of the first shape)
    seeds = [{"state": {"x": i, "tag": f"t{i}"}} for i in range(20)] + [{"state": {"x": 50, "tag": ["a"]}}]
    out = testing.boundary(seeds, n=60)
    assert any(isinstance(c["state"]["tag"], list) and c["note"] == "boundary: x: minus one" for c in out)


def test_deterministic_and_seed_states_left_out():
    seeds = [{"state": {"x": None, "s": "é"}}, {"state": {"x": 1, "s": "x"}}]
    a = testing.boundary(seeds, n=100)
    assert a == testing.boundary(seeds, n=100)
    assert not any(c["state"] in [s["state"] for s in seeds] for c in a)
