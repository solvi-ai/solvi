"""Original S326 scoped-key proposal and S239 explicit temporal contract, without a parallel resolver."""
import pytest

from solvi import Knowledge

WRITERS = {"wikidata": "outcome", "a": "outcome", "b": "outcome"}


def tell(km, subject, value, valid=None, *, writer="wikidata", holder=False):
    return km.tell({"s": subject, "r": "P35", "o": value}, source="outcome", by=writer,
                   valid=valid, scope={"holder": value} if holder else None)


def values(st, subject, at):
    return set(Knowledge.values(st.snapshot(about=subject, relation="P35", valid_at=at), subject, "P35"))


def test_original_coruler_witness_with_declared_holder_keys_and_singular_read_refusal():
    km = Knowledge(writers=WRITERS)
    for value in ("captain A", "captain B"):
        tell(km, "SM", value, ("2020-04-01", "2020-10-01"), holder=True)
    tell(km, "AD", "bishop", ("2003-05-12", None), holder=True)
    tell(km, "AD", "France", ("2017-05-14", None), holder=True)
    snap = km.snapshot(valid_at="2020-06-01")
    assert set(Knowledge.values(snap, "SM", "P35")) == {"captain A", "captain B"}
    assert set(Knowledge.values(snap, "AD", "P35")) == {"bishop", "France"}
    assert not km.store.disputes()
    with pytest.raises(ValueError):
        Knowledge.value(snap, "SM", "P35")
    assert Knowledge.value(snap, "SM", "P35", scope={"holder": "captain A"}) == "captain A"
    assert not Knowledge.values(km.snapshot(valid_at="2020-10-01"), "SM", "P35")
    # An undeclared relation key keeps succession: explicit scope is opt-in, never a global rewrite.
    tell(km, "single", "old", (0, None))
    tell(km, "single", "new", (10, None))
    assert values(km.store, "single", 15) == {"new"}


def test_scoped_claims_persist_branch_rebuild_and_writer_retract(tmp_path):
    km = Knowledge(tmp_path / "parent.jsonl", writers=WRITERS)
    a = tell(km, "SM", "A", (0, 20), writer="a", holder=True)
    b = tell(km, "SM", "B", (10, 30), writer="b", holder=True)
    assert values(km.store, "SM", 5) == {"A"}
    assert values(km.store, "SM", 15) == {"A", "B"}
    assert values(km.store, "SM", 25) == {"B"}
    head = km.store.head()
    child = km.branch(tmp_path / "child.jsonl")
    child.store.retract(a, only_from="a")
    assert values(child.store, "SM", 15) == {"B"}
    assert values(km.store, "SM", 15) == {"A", "B"} and km.store.head() == head
    for st in (km.store, km.store.rebuild(), Knowledge(tmp_path / "parent.jsonl", writers=WRITERS).store):
        assert values(st, "SM", 15) == {"A", "B"} and st.verify()
    reopened = Knowledge(tmp_path / "child.jsonl", writers=WRITERS).store
    assert values(reopened, "SM", 15) == {"B"} and reopened.verify()
    reopened.retract_source("b")
    assert not values(reopened, "SM", 15) and reopened.verify()
    # Rebuilding an earlier journal prefix restores the original support, not the child takedown.
    assert values(child.store.rebuild(upto=head["count"]), "SM", 15) == {"A", "B"}
    assert b != a


@pytest.mark.parametrize("left,right,disputed", [(None, None, False),
    ((None, None), (None, None), True), (None, (None, None), True), ((None, None), None, True)])
def test_explicit_unbounded_is_a_temporal_contract_not_a_legacy_migration(left, right, disputed):
    km = Knowledge(writers=WRITERS)
    a = tell(km, "x", "A", left, writer="a")
    b = tell(km, "x", "B", right, writer="b")
    assert bool(km.store.disputes()) == disputed
    if disputed:
        assert km.store.status(a) == km.store.status(b) == "disputed"
        assert not values(km.store, "x", 5)
    else:
        assert km.store.status(a) == "refuted" and km.store.status(b) == "active"
        assert values(km.store, "x", 5) == {"B"}
    assert km.store.verify() and km.store.rebuild().fingerprint() == km.store.fingerprint()


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_undated_and_bounded_claims_follow_world_time_and_restore_base(reverse, tmp_path):
    path = tmp_path / "mixed.jsonl"
    km = Knowledge(path, writers=WRITERS)
    claims = [("base", None, "a"), ("temporary", (10, 20), "b")]
    for name, valid, writer in reversed(claims) if reverse else claims:
        tell(km, "x", name, valid, writer=writer)
    for st in (km.store, km.store.rebuild(), km.store.branch(), Knowledge(path, writers=WRITERS).store):
        assert values(st, "x", 0) == {"base"}
        assert values(st, "x", 10) == {"temporary"}
        assert values(st, "x", 19) == {"temporary"}
        assert values(st, "x", 20) == {"base"}
        assert not st.disputes() and st.verify()
