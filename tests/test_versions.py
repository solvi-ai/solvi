"""solvi.core.versions (stable since 1.1, moved out of solvi.experimental.compile): a specification as numbered clauses
(Spec, revise), a catalog version (CatalogVersion: the module of pure functions, its parts with the clauses each
implements), a folder of versions with parents, the version in force and rollback (Versions, replay_all), and the
decisions a new version moves with the clauses why (decision_diff, DecisionDiff: only=, extraneous, exact, cites)."""
import json
import subprocess
import sys
import warnings

import pytest

from solvi import Answer, Question
from solvi.core.store import JSONLStorage
from solvi.core.versions import (CatalogVersion, Clause, DecisionDiff, Rejected, Spec, Versions, build_catalog,
                                 decision_diff, split_clauses)

POLICY = """# Shipping

| zone | fee |
|---|---|
| domestic | 5 |
| world | 20 |

- An order of 50 or more ships free.
- Orders to the world zone heavier than 30 kg are refused.
"""
V2 = POLICY.replace("50 or more", "80 or more")
QS = [Question("ship", "Shipped, and at what fee?", Answer.choice(["free", "paid", "refused"]))]

GOOD = '''
def zone_fee(zone):
    return {"domestic": 5, "world": 20}[zone]

def free_shipping(total):
    return total >= 50

def not_too_heavy(zone, weight):
    return not (zone == "world" and weight > 30) or Fail(f"{weight} kg to the world zone")

def ship(zone_fee, free_shipping):
    return "free" if free_shipping else "paid"
'''
GOOD2 = GOOD.replace("total >= 50", "total >= 80")
PARTS = {
    "zone_fee": {"kind": "fn", "clauses": ["c1", "c2"]},
    "free_shipping": {"kind": "fn", "clauses": ["c3"]},
    "not_too_heavy": {"kind": "check", "hard": True, "then": {"ship": "refused"}, "clauses": ["c4"]},
    "ship": {"kind": "rule", "question": "ship", "clauses": ["c3"]},
}


def v1():
    return CatalogVersion(Spec(POLICY), QS, GOOD, PARTS)


def v2():
    return CatalogVersion(Spec(POLICY).revise(V2), QS, GOOD2, PARTS)


# ----------------------------------------------------------------------------------------------- the specification
def test_paragraphs_list_items_and_table_rows_are_clauses_and_headings_name_their_section():
    cl = split_clauses(POLICY)
    assert cl == [("Shipping", "zone: domestic; fee: 5"), ("Shipping", "zone: world; fee: 20"),
                  ("Shipping", "An order of 50 or more ships free."),
                  ("Shipping", "Orders to the world zone heavier than 30 kg are refused.")]
    s = Spec(POLICY)
    assert list(s.clauses) == ["c1", "c2", "c3", "c4"] and "[c3] An order of 50" in s.render()
    assert s.clauses["c4"] == Clause("c4", "Orders to the world zone heavier than 30 kg are refused.", "Shipping")
    assert s.hash == Spec(POLICY).hash != Spec(POLICY.replace("50", "60")).hash
    assert "[changed] An order" in s.render({"c3": "changed"})


def test_a_revised_specification_keeps_the_ids_of_unchanged_clauses_and_names_the_changes():
    s = Spec(POLICY)
    r = s.revise(POLICY.replace("50 or more", "60 or more").replace("are refused.", "are refused.\n- Gift cards ship free."))
    assert r.changes == {"unchanged": ["c1", "c2", "c4"], "changed": ["c3"], "added": ["c5"], "removed": []}
    assert r.clauses["c5"].text == "Gift cards ship free." and r.parent == s.hash
    gone = s.revise(POLICY.replace("- Orders to the world zone heavier than 30 kg are refused.\n", ""))
    assert gone.changes["removed"] == ["c4"] and "c4" not in gone.clauses


def test_a_spec_from_given_clauses_and_back_from_its_record():
    s = Spec(clauses={"c1": "Refunds up to 100.", "c7": "Larger ones go to a person."}, name="refunds")
    assert list(s.clauses) == ["c1", "c7"] and s.text == "Refunds up to 100.\n\nLarger ones go to a person."
    assert list(Spec(clauses=["a", "b"]).clauses) == ["c1", "c2"]
    back = Spec.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.hash == s.hash and back.name == "refunds" and back.to_dict()["hash"] == s.hash
    with pytest.raises(ValueError, match="a text or clauses"):
        Spec()
    with pytest.raises(ValueError, match="no clauses"):
        Spec("")


# ----------------------------------------------------------------------------------------------- a catalog version
def test_a_catalog_version_is_an_ordinary_system_whose_parts_name_their_clauses():
    v = v1()
    assert v.accepted and v.clauses_of("not_too_heavy") == ["c4"] and v.clauses_of("answer:ship") == ["c3"]
    assert v.part_of("answer:ship") == "ship" and v.part_of("zone_fee") == "zone_fee" and v.part_of("nope") is None
    s = v.system()
    assert s.ask({"zone": "world", "total": 60, "weight": 40})["ship"].answer == "refused"
    assert s.ask({"zone": "domestic", "total": 60, "weight": 40})["ship"].answer == "free"
    assert getattr(s, "experimental", None) in (None, [])      # a stable version marks no experimental piece
    assert v.fingerprint == v1().fingerprint != v2().fingerprint
    cat, qs = build_catalog(v._ns, PARTS, QS)
    assert qs[0].requires == ["not_too_heavy"] and set(cat.parts) >= {"zone_fee", "free_shipping"}


def test_a_catalog_version_is_saved_loaded_and_refused_when_edited_or_not_accepted(tmp_path):
    v = v1()
    path = v.save(tmp_path / "v")
    assert (path / "module.py").read_text() == GOOD
    back = CatalogVersion.load(path)
    assert back.fingerprint == v.fingerprint and back.parts == PARTS and back.spec.hash == v.spec.hash
    assert CatalogVersion.from_dict(json.loads(json.dumps(v.to_dict()))).source == GOOD
    (path / "module.py").write_text(GOOD.replace("50", "5"))
    with pytest.raises(ValueError, match="differs from the source"):
        CatalogVersion.load(path)
    draft = CatalogVersion(Spec(POLICY), QS, GOOD, PARTS, accepted=False, reason="the drafts disagree")
    with pytest.raises(Rejected, match="the drafts disagree"):
        draft.catalog()
    assert draft.system(allow_unaccepted=True).ask({"zone": "domestic", "total": 1, "weight": 1})["ship"].answer == "paid"


def test_a_module_outside_pure_functions_is_refused_before_it_runs():
    bad = CatalogVersion(Spec(POLICY), QS, "import os\n\ndef zone_fee(zone):\n    return os.getpid()\n",
                         {"zone_fee": {"kind": "fn", "clauses": ["c1"]}})
    with pytest.raises(ValueError, match="import os is not allowed"):
        bad.catalog()


# ----------------------------------------------------------------------------------------------- the decisions a change moves
def test_the_decision_diff_lists_the_decisions_a_change_moves_with_the_clauses_of_their_causes():
    xs = [{"zone": "domestic", "total": t, "weight": 1} for t in (10, 50, 79, 80, 120)]
    dd = decision_diff(v1(), v2(), inputs=xs)
    assert isinstance(dd, DecisionDiff)
    assert [(d["seq"], d["old"], d["new"]) for d in dd.changed] == [(1, "free", "paid"), (2, "free", "paid")]
    assert dd.changed[0]["causes"][0]["part"] == "free_shipping" and dd.changed[0]["causes"][0]["clauses"] == ["c3"]
    assert str(dd).startswith("2 of 5 decisions change their answer") and len(dd.moved) == 2
    assert "because of clauses c3: 2" in str(dd)


def test_only_names_the_decisions_a_change_may_move(tmp_path):
    xs = [{"id": f"o{t}", "zone": "domestic", "total": t, "weight": 1} for t in (10, 50, 79, 80, 120)]
    exact = decision_diff(v1(), v2(), inputs=xs, only={"o50", "o79"})
    assert exact.exact is True and exact.extraneous == [] and [d["input"] for d in exact.changed] == ["o50", "o79"]
    broad = decision_diff(v1(), v2(), inputs=xs, only={"o50"})
    assert broad.exact is False and broad.extraneous == ["o79"]
    store = JSONLStorage(str(tmp_path / "d.jsonl"))
    s1 = v1().system(storage=store)
    ids = [s1.ask(x).stored_id for x in xs]
    by_store = decision_diff(v1(), v2(), store=store, only={ids[1]})
    assert by_store.extraneous == [ids[2]] and by_store.total == 5
    with pytest.raises(ValueError, match="a store of decisions or inputs"):
        decision_diff(v1(), v2())


# ----------------------------------------------------------------------------------------------- versions
def test_versions_replay_each_stored_decision_with_the_catalog_that_made_it(tmp_path):
    v = Versions(tmp_path / "versions")
    a, b = v1(), v2()
    assert (v.add(a, "v1"), v.add(b, "v2")) == (1, 2) and len(v) == 2 and v.numbers() == [1, 2]
    assert v.find(a.fingerprint) == 1 and v.get(2).source == b.source and v.find("nope") is None
    store = JSONLStorage(str(tmp_path / "d.jsonl"))
    s1 = v.system(1, storage=store)
    for t in (60, 70):
        s1.ask({"zone": "domestic", "total": t, "weight": 1})
    v.system(2, storage=store).ask({"zone": "domestic", "total": 90, "weight": 1})
    assert v.replay_all(store) == []
    assert len(store.replay_all(v.system(2))) == 2              # the v1 decisions do not recompute under v2
    with pytest.raises(Rejected):
        v.add(CatalogVersion(Spec(POLICY), QS, "x = 1\n", {}, accepted=False, reason="not accepted"))


def test_a_version_has_a_parent_and_the_version_in_force_can_be_rolled_back(tmp_path):
    v = Versions(tmp_path / "versions")
    assert v.current is None
    v.add(v1(), "v1")
    v.add(v2(), "v2")
    assert v.current == 2 and v.info(1)["parent"] is None and v.info(2)["parent"] == 1
    assert v.info(2)["label"] == "v2" and v.info(2)["fingerprint"] == v2().fingerprint
    assert v.rollback(1, by="ann", why="v2 refuses too much") == 1
    assert v.current == 1 and v.get().source == GOOD                  # get() / system(): the version in force
    assert v.system().ask({"zone": "domestic", "total": 60, "weight": 1})["ship"].answer == "free"
    assert v.add(v2(), "v2 again") == 3 and v.info(3)["parent"] == 1 and v.current == 3
    assert Versions(tmp_path / "versions").current == 3                # read back from the folder
    assert [h["to"] for h in v.history()] == [1] and v.history()[0]["by"] == "ann"
    with pytest.raises(KeyError, match="no version 9"):
        v.rollback(9)


def test_a_change_pipeline_promotes_into_and_rolls_back_the_stable_versions(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental.changes import Change, Pipeline, VersionsTarget
    vs = Versions(tmp_path / "versions")
    vs.add(v1(), "v1")
    target = VersionsTarget(vs)
    pipe = Pipeline([], str(tmp_path / "changes.jsonl"), targets=[target], approval={"compiled": "gate"})
    res = pipe.propose(Change("document", artifact="compiled", payload=v2(), reason="v2 of the policy"))
    assert res.promoted and res.ref == {"version": 2, "before": 1} and vs.current == 2
    pipe.rollback(0)
    assert target.in_force == 1 and vs.current == 1


# ----------------------------------------------------------------------------------------------- the move
def test_the_compiler_re_exports_the_moved_names_as_the_same_objects():
    import solvi.check
    from solvi.core import versions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental import compile as comp
        from solvi.experimental.compile import sandbox
    for name in ("Spec", "Clause", "Versions", "DecisionDiff", "decision_diff", "Rejected", "split_clauses",
                 "build_catalog", "CatalogVersion"):
        assert getattr(comp, name) is getattr(versions, name), name
    assert issubclass(comp.Compiled, versions.CatalogVersion) and solvi.check.split_clauses is versions.split_clauses
    from solvi.core.versions import _sandbox                  # the in-process check and load of a module moved too
    assert sandbox.check is _sandbox.check and sandbox.load is _sandbox.load


def test_moved_names_keep_the_module_their_fingerprints_record():
    from solvi.core.provenance import fp_module
    for name in ("Spec", "Clause", "Versions", "DecisionDiff", "decision_diff", "Rejected", "split_clauses"):
        assert fp_module("solvi.core.versions", name) == "solvi.compile", name
    assert fp_module("solvi.core.versions", "CatalogVersion") == "solvi.core.versions"   # new in 1.1


def test_the_stable_module_imports_nothing_experimental_and_does_not_warn():
    code = ("import sys, warnings\nwarnings.simplefilter('error')\nimport solvi.core.versions\n"
            "assert not [m for m in sys.modules if m.startswith('solvi.experimental.')], sorted(sys.modules)\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]


def test_the_change_pipeline_and_the_credit_benchmark_import_the_stable_module():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bench = (root / "benchmarks" / "changes_credit.py").read_text()
    assert "from solvi.core.versions import" in bench and "solvi.experimental.compile" not in bench
    for f in (root / "src" / "solvi" / "experimental" / "changes").glob("*.py"):
        text = f.read_text()
        assert "from ..compile import decision_diff" not in text and "from ..compile import Versions" not in text, f
