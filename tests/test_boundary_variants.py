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


# --- thresholds from the rules' constants; huge negative numbers; long distinct lists, sets and dict keys
LIMIT = 250


def _threshold_system():
    from solvi import Answer, Catalog, Question, System
    cat = Catalog()

    @cat.rule("big")
    def big(amount, score) -> bool:
        return amount >= 100 and 0.5 < score <= LIMIT

    return System(cat, [Question("big", "Big?", Answer.yes_no())])


def test_thresholds_from_the_rules_constants():
    system = _threshold_system()
    seeds = [{"state": {"amount": 3, "score": 0.9}}, {"state": {"amount": 7, "score": 0.7}}]
    out = testing.boundary(seeds, n=500, system=system)
    amounts = {c["state"]["amount"] for c in out if "rule threshold" in c["note"] and "amount" in c["note"]}
    assert {99, 100, 101} <= amounts                  # amount >= 100: one below, at and one above, as ints
    scores = {c["state"]["score"] for c in out if "rule threshold" in c["note"] and "score" in c["note"]}
    assert 0.5 in scores and any(0.49 < s < 0.5 for s in scores) and any(0.5 < s < 0.51 for s in scores)
    assert 250 in scores or 250.0 in scores           # a module constant the rule compares with
    # the answers flip at the threshold, and the pinned answers say so
    at = {c["state"]["amount"]: c["expected"]["big"] for c in out
          if c["note"].startswith("boundary: amount: ") and "rule threshold" in c["note"] and c["state"]["score"] == 0.9}
    assert at[99] == "no" and at[100] == "yes"
    # off, and given by hand
    off = testing.boundary(seeds, n=500, system=system, thresholds=False)
    assert not any("rule threshold" in c["note"] for c in off)
    hand = testing.boundary(seeds, n=500, thresholds={"amount": [42]})
    assert {41, 42, 43} <= {c["state"]["amount"] for c in hand if "rule threshold" in c["note"]}
    # without a system and without thresholds=: none (there are no rules to read)
    assert not any("rule threshold" in c["note"] for c in testing.boundary(seeds, n=500))


def test_huge_negative_numbers():
    out = testing.boundary([{"state": {"x": 5, "y": 1.5}}], n=500)
    assert {"x": -2 ** 63, "y": 1.5} in [c["state"] for c in out]
    assert {"x": 5, "y": -1e300} in [c["state"] for c in out]


def test_long_distinct_lists_sets_and_dict_keys():
    seeds = [{"state": {"tags": ["a", "b"], "ids": [1, 2], "seen": {"x"}, "counts": {"a": 1}}}]
    out = testing.boundary(seeds, n=1000)
    by = {c["note"]: c["state"] for c in out}
    tags = by["boundary: tags: long list of distinct values"]["tags"]
    assert len(tags) == len(set(tags)) == testing._traces.LONG and all(isinstance(t, str) for t in tags)
    ids = by["boundary: ids: long list of distinct values"]["ids"]
    assert len(set(ids)) == testing._traces.LONG and all(isinstance(t, int) for t in ids)
    seen = by["boundary: seen: long set of distinct values"]["seen"]
    assert isinstance(seen, set) and len(seen) == testing._traces.LONG
    assert by["boundary: seen: empty set"]["seen"] == set()
    counts = by["boundary: counts: many keys"]["counts"]
    assert len(counts) == testing._traces.LONG and counts["a"] == 1
