"""Per-check source names are actual typed dependencies, not an independence certificate."""
import asyncio

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.check import lint
from solvi.core.provenance import catalog_fingerprint
from solvi.core.types import FactTypeError


def make(mapping=None, *, computed=False, cache=None, then="no"):
    c = Catalog()
    if computed:
        @c.fn
        def fresh_tax(database: int) -> int:
            return database
    original = {"tax": "fresh_tax"} if mapping is None else mapping

    @c.check(hard=True, then={"q": then}, reads_from=original)
    def valid(total: int, tax: int) -> bool:
        return total == tax

    @c.rule("q")
    def rule(stale_tax: int) -> str:
        return "yes"

    return System(c, [Question("q", "?", Answer.choice(["yes", "no"]))], cache=cache)


DATA = {"total": 5, "stale_tax": 5, "fresh_tax": 7}


def test_original_stale_source_witness_sync_async_replay_and_typed_dependencies():
    s = make()
    p = s.catalog.parts["valid"]
    assert p.inputs == ["total", "fresh_tax"] and set(p.types) == {"total", "fresh_tax"}
    fields = {name for entry in s.entry_points() for name in entry.fields}
    assert "fresh_tax" in fields and "tax" not in fields
    for r in (s.ask(DATA, store=False), asyncio.run(s.aask(DATA, store=False))):
        assert r["q"].answer == "no" and r["q"].status == "forced"
        rec = next(x for x in r.trace.records if x.name == "valid")
        assert set(rec.inputs) == {"total", "fresh_tax"}
        assert r.trace.replay(s, r.flow)["ok"]
    assert s.ask({**DATA, "fresh_tax": "wrong type"}, store=False)["q"].status == "abstain"


def test_input_mapping_is_detached_and_hash_cache_replay_bind_actual_sources():
    mapping = {"tax": "fresh_tax"}
    s = make(mapping, cache=8)
    before = catalog_fingerprint(s.catalog)["fp"]
    mapping["tax"] = "stale_tax"
    assert catalog_fingerprint(s.catalog)["fp"] == before
    assert s.ask(DATA, store=False)["q"].answer == "no"
    assert s.ask({**DATA, "fresh_tax": 5}, store=False)["q"].answer == "yes"
    r = s.ask(DATA, store=False)
    s.catalog.parts["valid"].reads_from = (("tax", "total"),)
    assert catalog_fingerprint(s.catalog)["fp"] != before
    assert not r.trace.replay(s, r.flow)["ok"]
    assert s.ask(DATA, store=False)["q"].status == "abstain"


def test_mapping_uses_computed_fact_and_cannot_be_overridden_by_given():
    s = make(computed=True)
    facts = {k: v for k, v in {**DATA, "database": 7}.items() if k != "fresh_tax"}
    assert s.ask(facts, store=False)["q"].answer == "no"
    with pytest.raises(ValueError, match="cannot stand in"):
        s.ask(DATA, store=False)


def test_mapping_types_checked_against_actual_producer_before_registration():
    c = Catalog()

    @c.fn
    def text(source: str) -> str:
        return source

    with pytest.raises(FactTypeError):
        @c.check(reads_from={"tax": "text"})
        def check(tax: int) -> bool:
            return tax > 0
    assert "check" not in c.parts


@pytest.mark.parametrize("mapping", [{"unknown": "source"}, {"tax": "valid"},
    {"tax": "total"}, {"tax": "not valid"}, {"tax": 1}, [], [("tax", "source")]])
def test_bad_mapping_fails_before_any_check_call(mapping):
    with pytest.raises(ValueError):
        make(mapping)


def test_then_arguments_are_not_remapped_and_boundary_aliases_work():
    def forced(tax: int) -> str:
        return "no" if tax == 9 else "yes"
    s = make(then=forced).rename(inputs={"fresh_tax": "db_tax"})
    facts = {k: v for k, v in {**DATA, "db_tax": 7, "tax": 9}.items() if k != "fresh_tax"}
    r = s.ask(facts, store=False)
    assert r["q"].answer == "no"
    assert r.trace.replay(s, r.flow)["ok"]


def test_shared_pass_guard_refuses_registration_without_invoking_the_model():
    calls = []

    def model(tax):
        calls.append(tax)
        return True
    model.in_pass = lambda *args: calls.append(args)
    with pytest.raises(ValueError, match="shared-pass"):
        Catalog().check(model, reads_from={"tax": "other"})
    assert not calls


def test_lint_shared_given_roots_is_opt_in_and_names_actual_sources():
    shared = make({"tax": "stale_tax"})
    messages = [x for x in lint(shared).findings if x.code == "check_shared_source"]
    assert len(messages) == 1 and "stale_tax" in messages[0].message
    assert "not an independence certificate" in messages[0].message
    assert not [x for x in lint(make()).findings if x.code == "check_shared_source"]


def test_mapping_rejects_transitive_self_evidence_before_registration():
    c = Catalog()

    @c.fn
    def indirect(check: bool) -> int:
        return int(check)

    with pytest.raises(ValueError, match="dependency cycle"):
        @c.check(reads_from={"tax": "indirect"})
        def check(tax: int) -> bool:
            return tax > 0
    assert "check" not in c.parts


def test_same_root_lint_traverses_computed_sources_and_aliases():
    s = make(computed=True)
    s.catalog.rules["q"].inputs = ["database"]
    findings = [x for x in lint(s).findings if x.code == "check_shared_source"]
    assert findings and "database" in findings[0].message
    # Existing catalogs without an explicit binding retain their lint behavior.
    s.catalog.parts["valid"].reads_from = None
    assert not [x for x in lint(s).findings if x.code == "check_shared_source"]


def test_empty_mapping_detaches_caller_alias_without_changing_cache_or_replay():
    mapping = {}
    s = make(mapping, cache=8)
    facts = {**DATA, "tax": 7}
    fp = catalog_fingerprint(s.catalog)["fp"]
    original = s.ask(facts, store=False)
    assert original["q"].answer == "no"
    mapping["tax"] = "stale_tax"
    assert s.catalog.parts["valid"].reads_from is None
    assert catalog_fingerprint(s.catalog)["fp"] == fp
    assert s.ask(facts, store=False)["q"].answer == "no"
    assert original.trace.replay(s, original.flow)["ok"]
