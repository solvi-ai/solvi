"""Citations on hand-written parts: `cite=` (a verbatim quote of the policy) and `clause=` on the decorators, checked
against the text by solvi.check.citations (and `solvi check --text`), carried into the causes of a diff."""
import textwrap
import warnings

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.check import citations
from solvi.cli import main
from solvi.core.store import JSONLStorage
from solvi.core.store.diff import diff

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import Change, Pipeline, Static
    from solvi.core.versions import Spec, decision_diff

C1 = "An application is scored in points; the points decide."
C7 = 'credit_amount_dm > 8000 → +2'
C14 = "Decision: points ≥ 2 → refuse; otherwise approve."
C16 = 'new rule `guarantor`: other_debtors is “guarantor” → −1;'
SPEC = Spec(clauses={"c1": C1, "c7": C7, "c14": C14, "c16": C16}, name="credit policy")
POLICY = SPEC.text
GUARANTOR = 'new rule `guarantor`: other_debtors is "guarantor" → −1'     # straight quotes: found in the normalized view
DECISION = Question("decision", "Approve the loan?", Answer.choice(["approve", "refuse"]))


def credit(version=1, *, guarantor_cite=GUARANTOR, guarantor_clause="c16"):
    cat = Catalog()

    @cat.fn(clause="c7", cite="credit_amount_dm > 8000 → +2")
    def amount_points(credit_amount_dm):
        return 2 if credit_amount_dm > 8000 else 0

    if version == 2:
        @cat.fn(clause=guarantor_clause, cite=guarantor_cite)
        def guarantor_points(other_debtors):
            return -1 if other_debtors == "guarantor" else 0

        @cat.fn(clause="c1", cite="the points decide")
        def points(amount_points, guarantor_points):
            return amount_points + guarantor_points
    else:
        @cat.fn(clause="c1", cite="the points decide")
        def points(amount_points):
            return amount_points

    @cat.rule("decision", clause="c14", cite="points ≥ 2 → refuse")
    def decision(points):
        return "refuse" if points >= 2 else "approve"
    return System(cat, [DECISION])


APPS = [{"credit_amount_dm": 9000, "other_debtors": "guarantor"},
        {"credit_amount_dm": 9000, "other_debtors": "none"},
        {"credit_amount_dm": 1000, "other_debtors": "guarantor"}]


def test_cite_and_clause_on_the_decorators_become_the_parts_cites():
    s = credit(2)
    assert s.catalog.parts["guarantor_points"].cites == [("c16", GUARANTOR)]
    assert s.catalog.rules["decision"].cites == [("c14", "points ≥ 2 → refuse")]
    cat = Catalog()

    @cat.check(cite=["a", "b"], clause="c3")
    def two(x) -> bool:
        return True

    @cat.extract(cite="a")
    def one(doc):
        return None
    assert cat.parts["two"].cites == [("c3", "a"), ("c3", "b")] and cat.parts["one"].cites == [(None, "a")]
    assert Catalog().parts == {} and credit(1).catalog.parts["amount_points"].cites == [("c7", C7)]
    with pytest.raises(ValueError, match="same length"):
        @cat.fn(cite=["a", "b"], clause=["c1"])
        def three(x):
            return x


def test_a_quote_found_in_the_text_is_ok_and_one_not_found_is_an_error():
    assert citations(credit(2), POLICY).ok and citations(credit(2), POLICY).findings == []
    rep = citations(credit(2, guarantor_cite="other_debtors is a guarantor → −1"), POLICY)
    assert rep.codes("error") == ["cite_not_found"] and rep.errors[0].where == "guarantor_points" and not rep.ok


def test_with_the_spec_a_clause_must_exist_and_hold_the_quote():
    assert citations(credit(2), POLICY, spec=SPEC).ok
    outside = citations(credit(2, guarantor_clause="c14"), POLICY, spec=SPEC)
    assert outside.codes("error") == ["cite_outside_clause"] and outside.errors[0].where == "guarantor_points"
    unknown = citations(credit(2, guarantor_clause="c99"), POLICY, spec=SPEC)
    assert unknown.codes("error") == ["clause_unknown"]
    person = citations(credit(2, guarantor_clause="person:k07"), POLICY, spec=SPEC)   # a correction: the quote only
    assert person.ok


def test_require_flags_a_part_without_a_citation():
    s = credit(2)
    cat = s.catalog

    @cat.fn
    def uncited(points):
        return points
    rep = citations(s, POLICY, require=True)
    assert [(f.level, f.code, f.where) for f in rep.findings] == [("warning", "no_cite", "uncited")] and rep.ok
    assert citations(s, POLICY).findings == []


def test_the_diff_causes_carry_the_cited_clauses(tmp_path):
    """The T1 chain: clause → rule → decision. Decisions stored under v1, re-run with v2 whose new rule cites c16."""
    store = JSONLStorage(tmp_path / "d.jsonl")
    v1 = credit(1)
    v1.storage, store.catalog = store, v1
    ids = [v1.ask(a).stored_id for a in APPS]
    rep = diff(store, credit(2))
    assert [c["id"] for c in rep.changed] == [ids[0]]
    causes = rep.changed[0]["questions"]["decision"]["causes"]
    by = {c["name"]: c for c in causes}
    assert by["guarantor_points"]["cites"] == [{"clause": "c16", "quote": GUARANTOR}]
    assert by["points"]["cites"] == [{"clause": "c1", "quote": "the points decide"}]
    dd = decision_diff(credit(1), credit(2), inputs=APPS)                 # plain catalogs: clauses from the citations
    assert len(dd.moved) == 1
    cause = {c["step"]: c for c in dd.moved[0]["causes"]}["guarantor_points"]
    assert cause["clauses"] == ["c16"] and cause["cites"] == [{"clause": "c16", "quote": GUARANTOR}]


class StateTarget:
    artifact = "update"

    def __init__(self):
        self.state = None

    def promote(self, change, measured):
        self.state = change.payload["fingerprint"]
        return {"state": self.state}

    def rollback(self, ref, by=None, why=None):
        self.state = None


def test_static_refuses_a_candidate_whose_quote_is_not_in_the_text(tmp_path):
    t = StateTarget()
    pipe = Pipeline([Static(text=POLICY)], str(tmp_path / "c.jsonl"), targets=[t], approval={("update", "person"): "gate"})
    bad = credit(2, guarantor_cite="other_debtors is a guarantor → −1")
    res = pipe.propose(Change("person", artifact="update", payload={"fingerprint": "bad", "system": bad}))
    assert res.status == "rejected" and res.stage == "static" and "cite_not_found" in res.reason
    ok = pipe.propose(Change("person", artifact="update", payload={"fingerprint": "v2", "system": credit(2)}))
    assert ok.promoted and t.state == "v2"
    spec = Pipeline([Static(text=POLICY, spec=SPEC)], str(tmp_path / "c2.jsonl"), targets=[t],
                    approval={("update", "person"): "gate"})
    res = spec.propose(Change("person", artifact="update",
                              payload={"fingerprint": "out", "system": credit(2, guarantor_clause="c14")}))
    assert res.status == "rejected" and "cite_outside_clause" in res.reason


def test_solvi_check_text(tmp_path, capsys):
    (tmp_path / "POLICY.md").write_text(POLICY)
    f = tmp_path / "credit.py"
    f.write_text(textwrap.dedent('''
        from solvi import Answer, Catalog, Question, System

        cat = Catalog()

        @cat.rule("decision", clause="c14", cite=CITE)
        def decision(points):
            return "refuse" if points >= 2 else "approve"

        system = System(cat, [Question("decision", "Approve?", Answer.choice(["approve", "refuse"]))])
    '''))
    for i, (cite, code) in enumerate((("points ≥ 2 → refuse", 0), ("points ≥ 3 → refuse", 1))):
        g = tmp_path / f"credit{i}.py"                  # a module of its own: the loader caches a module by its file
        g.write_text(f.read_text().replace("CITE", repr(cite)))
        capsys.readouterr()
        assert main(["check", f"{g}:system", "--text", str(tmp_path / "POLICY.md")]) == code
        out = capsys.readouterr().out
        assert ("cite_not_found" in out) == (code == 1)
