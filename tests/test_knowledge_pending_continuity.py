from solvi.core.knowledge import KnowledgeStore


def fixture(writer="product_team", until=None):
    st = KnowledgeStore()

    def add(value, by, since, end=None):
        return st.add(
            "fact",
            {"s": "Orion X", "r": "warranty_months", "o": value, "eff": since},
            source="outcome",
            by=by,
            valid=(since, end),
        )

    old24 = add(24, writer, 0)
    st.retract(old24, only_from=writer)
    old36 = add(36, "product_team", 0)
    new24 = add(24, "vendor_feed", 120, until)
    st.retest(new24, ok=True, by="human")
    add(24, writer, 120, until)
    st.retract_source("vendor_feed", by="auditor")
    return st, new24, old36


def values(st, at):
    return [r["body"]["o"] for r in st.snapshot(about="Orion X", relation="warranty_months", valid_at=at)["facts"]]


def test_original_wrong36_becomes_gap_until_fresh_independent_retest():
    st, new24, old36 = fixture()
    assert st.pending_retest(new24) and values(st, 126) == []
    assert not st.usable(old36, valid_at=126) and values(st, 100) == [36]
    assert st.periods(old36) == [[0, 120]]
    assert st.verify() and st.rebuild().fingerprint() == st.fingerprint()
    st.retest(new24, ok=True, by="independent-human")
    assert values(st, 126) == [24] and st.verify()


def test_other_writer_pending_does_not_block_old_value():
    st, new24, old36 = fixture(writer="other_writer")
    assert st.pending_retest(new24) and values(st, 126) == [36]


def test_bounded_gap_and_actual_withdrawal_restore_old_version():
    st, new24, old36 = fixture(until=200)
    assert values(st, 126) == [] and values(st, 200) == [36]
    prefix = len(st.journal)
    st.retract(new24, by="product_team")
    assert values(st, 126) == [36]
    assert values(st.rebuild(upto=prefix), 126) == []
    assert st.verify()


def test_flags_and_branch_keep_same_continuity_gap():
    st, new24, old36 = fixture()
    child = st.branch()
    assert values(child, 126) == [] and child.verify()
    st.flag(ids=[old36], by="audit")
    assert values(st, 100) == [] and values(st, 126) == []


def test_independent_old_support_survives_same_version_unit_aggregation():
    st, new24, old36 = fixture()
    assert values(st, 126) == []
    other = st.add(
        "fact",
        {"s": "Orion X", "r": "warranty_months", "o": 36, "eff": 0},
        source="outcome",
        by="independent-old-writer",
        valid=(0, None),
    )
    assert other == old36 and values(st, 126) == [36]
    assert st.verify()


def pending_with_old_support(*, undated=False, mode="direct"):
    st = KnowledgeStore()
    old_false = st.add(
        "fact",
        {"s": "Orion X", "r": "warranty_months", "o": 24, "eff": 0},
        source="outcome",
        by="other_writer",
        valid=(0, None),
    )
    st.retract(old_false)
    premise = st.add("fact", {"s": "support", "r": "v", "o": 1}, source="outcome", by="proof")
    kwargs = {}
    if mode in ("derived", "inferred"):
        kwargs["derived_from"] = [premise]
    if mode == "inferred":
        kwargs["inferred"] = True
    if mode == "consulted":
        kwargs["consulted"] = [premise]
    writer = "product_team" if mode != "direct" else "independent_old"
    old = st.add(
        "fact",
        {"s": "Orion X", "r": "warranty_months", "o": 36, "eff": 0},
        source="outcome",
        by=writer,
        valid=None if undated else (0, None),
        **kwargs,
    )
    new = st.add(
        "fact",
        {"s": "Orion X", "r": "warranty_months", "o": 24, "eff": 120},
        source="outcome",
        by="vendor",
        valid=(120, None),
    )
    st.retest(new, ok=True, by="human")
    st.add(
        "fact",
        {"s": "Orion X", "r": "warranty_months", "o": 24, "eff": 120},
        source="outcome",
        by="product_team",
        valid=(120, None),
    )
    st.retract_source("vendor")
    return st, old, new


def test_undated_old_foreign_pending_keeps_its_full_interval():
    st, old, new = pending_with_old_support(undated=True)
    st.add(
        "fact",
        {"s": "Orion X", "r": "warranty_months", "o": 48, "eff": 300},
        source="outcome",
        by="future",
        valid=(300, None),
    )
    assert st.pending_retest(new) and values(st, 126) == [36]
    assert values(st, 10) == [36] and values(st, 350) == [48]
    assert st.usable(old, valid_at=126) and st.periods(old) == [[None, 300]]
    assert st.verify()


def test_old_nondirect_contributions_do_not_get_continuity_restrictions():
    for mode in ("derived", "inferred", "consulted"):
        st, old, new = pending_with_old_support(mode=mode)
        assert st.pending_retest(new)
        assert values(st, 126) == [36], mode
        assert st.usable(old, valid_at=126) and st.verify()


def test_gap_fragment_does_not_become_a_newer_version_than_independent150():
    st, new24, old36 = fixture(until=200)
    newer = st.add(
        "fact",
        {"s": "Orion X", "r": "warranty_months", "o": 48, "eff": 150},
        source="outcome",
        by="independent",
        valid=(150, None),
    )
    assert values(st, 126) == []
    assert values(st, 175) == [48] and values(st, 210) == [48]
    assert st.periods(old36) == [[0, 120]]
    assert st.periods(newer) == [[150, None]]
    assert st.verify() and st.rebuild().fingerprint() == st.fingerprint()
