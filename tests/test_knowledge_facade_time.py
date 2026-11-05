import datetime

import pytest

from solvi import Knowledge


def versions(**kwargs):
    knowledge = Knowledge(**kwargs)
    old = knowledge.tell(("pump", "rate", 1), source="outcome", by="meter", valid=(0, None))
    new = knowledge.tell(("pump", "rate", 2), source="outcome", by="meter", valid=(10, None))
    return knowledge, old, new


def test_explicit_time_matches_store_without_changing_status_only_default():
    knowledge, old, new = versions()
    assert knowledge.usable(old)  # existing status-only contract
    for time in (5, 10, 15):
        for iid in (old, new):
            assert knowledge.usable(iid, valid_at=time) == knowledge.store.usable(iid, valid_at=time)
    assert knowledge.usable(old, valid_at=5) and not knowledge.usable(old, valid_at=10)
    assert knowledge.usable(new, valid_at=10)
    assert knowledge.usable(None, valid_at=5) is False


def test_explicit_time_applies_to_derived_support():
    knowledge, old, _ = versions()
    child = knowledge.tell(("estimate", "rate", 1), source="outcome", by="reader", derived_from=[old])
    assert knowledge.usable(child, valid_at=5)
    assert not knowledge.usable(child, valid_at=15)


def test_explicit_time_overrides_declared_clock_and_none_retains_clock():
    clock = [5]
    knowledge, old, _ = versions(now=lambda: clock[0])
    assert knowledge.usable(old)
    clock[0] = 15
    assert not knowledge.usable(old) and not knowledge.usable(old, valid_at=None)
    assert knowledge.usable(old, valid_at=5)


def test_date_and_iso_time_use_existing_store_normalization():
    knowledge = Knowledge()
    old = knowledge.tell(
        ("person", "role", "analyst"),
        source="outcome",
        by="hr",
        valid=(datetime.date(2026, 1, 1), datetime.date(2026, 2, 1)),
    )
    assert knowledge.usable(old, valid_at=datetime.date(2026, 1, 20))
    assert not knowledge.usable(old, valid_at="2026-02-01")


def test_incompatible_time_units_keep_native_clear_error_and_leave_state_unchanged():
    knowledge, old, _ = versions()
    before = knowledge.store.fingerprint(), len(knowledge.store.journal)
    with pytest.raises(ValueError, match="valid_at=.*does not compare"):
        knowledge.usable(old, valid_at="2026-01-01")
    assert (knowledge.store.fingerprint(), len(knowledge.store.journal)) == before
