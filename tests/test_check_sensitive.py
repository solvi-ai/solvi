"""Sensitive fields: declared on the catalog (Catalog(sensitive=)) or given to lint / `solvi check --sensitive`; a part
that reads one, directly or through a fact computed from one, is an error."""
import textwrap

from solvi import Answer, Catalog, Question, System
from solvi.check import lint
from solvi.cli import main

DECISION = Question("decision", "Approve the loan?", Answer.choice(["approve", "refuse"]))


def system(cat):
    return System(cat, [DECISION])


def findings(rep, code):
    return [f for f in rep.findings if f.code == code]


def test_a_part_that_reads_a_sensitive_field_is_one_error_naming_the_part_and_the_field():
    cat = Catalog()

    @cat.fn
    def young_points(age):
        return 1 if age <= 25 else 0

    @cat.rule("decision")
    def decision(amount, young_points):
        return "refuse" if amount > 5000 else "approve"

    rep = lint(system(cat), sensitive={"age"})
    direct = findings(rep, "reads_sensitive")
    assert len(direct) == 1 and direct[0].level == "error" and direct[0].where == "young_points"
    assert "'young_points' reads 'age': a sensitive field" in direct[0].message
    assert not rep.ok
    assert lint(system(cat)).ok                                   # nothing declared: nothing to refuse


def test_a_fact_computed_from_a_sensitive_field_carries_it_with_the_chain():
    cat = Catalog()

    @cat.fn
    def risk_band(age):
        return "high" if age <= 25 else "low"

    @cat.fn
    def risk_points(risk_band):
        return 2 if risk_band == "high" else 0

    @cat.rule("decision")
    def decision(risk_points):
        return "refuse" if risk_points >= 2 else "approve"

    rep = lint(system(cat), sensitive=["age"])
    assert [f.where for f in findings(rep, "reads_sensitive")] == ["risk_band"]
    via = {f.where: f for f in findings(rep, "reads_sensitive_via")}
    assert set(via) == {"risk_points", "decision"} and all(f.level == "error" for f in via.values())
    assert "reads 'risk_band', computed from 'age'" in via["risk_points"].message
    assert "reads 'risk_points', computed from 'risk_band', computed from 'age'" in via["decision"].message


def test_a_then_function_and_an_alternative_producer_that_read_it_are_errors():
    cat = Catalog()

    def refuse_young(age):
        return "refuse" if age < 21 else "approve"

    @cat.check(hard=True, then={"decision": refuse_young})
    def small_enough(amount) -> bool:
        return amount < 20000

    @cat.fn(provides="score")
    def score_amount(amount):
        return amount / 1000

    @cat.fn(provides="score")
    def score_age(age):
        return age / 10

    @cat.rule("decision")
    def decision(score):
        return "refuse" if score > 5 else "approve"

    rep = lint(system(cat), sensitive={"age"})
    direct = {f.where for f in findings(rep, "reads_sensitive")}
    assert direct == {"small_enough.then[decision]", "score (score_age)"}
    assert [f.where for f in findings(rep, "reads_sensitive_via")] == ["decision"]


def test_a_declared_field_nothing_reads_is_a_note_and_the_report_stays_ok():
    cat = Catalog(sensitive={"sex"})

    @cat.rule("decision")
    def decision(amount):
        return "refuse" if amount > 5000 else "approve"

    rep = lint(system(cat))
    assert [(f.level, f.code, f.where) for f in findings(rep, "sensitive_unread")] == [("note", "sensitive_unread", "sex")]
    assert rep.ok


def test_the_catalogs_own_set_is_found_without_the_keyword_and_the_keyword_adds_to_it():
    cat = Catalog(sensitive=("age",))
    assert cat.sensitive == frozenset({"age"})
    assert Catalog().sensitive == frozenset()

    @cat.fn
    def young_points(age):
        return 1 if age <= 25 else 0

    @cat.fn
    def male_points(sex):
        return 1 if sex == "male" else 0

    @cat.rule("decision")
    def decision(young_points, male_points):
        return "refuse" if young_points + male_points >= 2 else "approve"

    alone = lint(system(cat))
    assert [f.where for f in findings(alone, "reads_sensitive")] == ["young_points"]
    both = lint(system(cat), sensitive={"sex"})
    assert sorted(f.where for f in findings(both, "reads_sensitive")) == ["male_points", "young_points"]
    assert len(findings(both, "reads_sensitive_via")) == 2                  # the rule: once for each tainted fact it reads
    assert lint(cat, sensitive={"sex"}).codes("error").count("reads_sensitive") == 2    # a bare Catalog too


def _run(capsys, *argv):
    capsys.readouterr()
    try:
        code = main([str(a) for a in argv])
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr().out


def test_solvi_check_sensitive(tmp_path, capsys):
    f = tmp_path / "credit.py"
    f.write_text(textwrap.dedent('''
        from solvi import Answer, Catalog, Question, System

        cat = Catalog()

        @cat.fn
        def young_points(age):
            return 1 if age <= 25 else 0

        @cat.rule("decision")
        def decision(amount, young_points):
            return "refuse" if amount > 5000 or young_points else "approve"

        system = System(cat, [Question("decision", "Approve?", Answer.choice(["approve", "refuse"]))])
    '''))
    code, out = _run(capsys, "check", f"{f}:system")
    assert code == 0 and "reads_sensitive" not in out
    code, out = _run(capsys, "check", f"{f}:system", "--sensitive", "age,sex")
    assert code == 1 and "reads_sensitive" in out and "young_points" in out and "sensitive_unread" in out
