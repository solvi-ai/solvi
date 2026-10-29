"""`solvi check` reads a lambda rule's return value, and says when it cannot read a rule (no source) instead of
reporting "no problems found"."""
from solvi import Answer, Catalog, Question, System
from solvi.check import lint

Q = Question("q", "q", Answer.choice(["approve", "deny"]))


def _codes(rep):
    return [(f.level, f.code) for f in rep.findings]


def test_a_lambda_rule_returning_a_non_option_is_an_error():
    cat = Catalog()
    cat.rule("q")(lambda x: "aprove")
    rep = lint(System(cat, [Q]))
    assert ("error", "rule_returns_non_option") in _codes(rep)
    assert not rep.ok


def test_a_lambda_rule_with_branches_of_options_passes():
    cat = Catalog()
    cat.rule("q")(lambda x: "approve" if x else "deny")
    assert not [c for c in _codes(lint(System(cat, [Q]))) if c[1] in ("rule_returns_non_option", "rule_unchecked")]


def test_a_rule_without_source_is_reported_unchecked():
    ns = {}
    exec("def r(x):\n    return 'aprove'\n", ns)
    cat = Catalog()
    cat.rule("q")(ns["r"])
    rep = lint(System(cat, [Q]))
    assert ("warning", "rule_unchecked") in _codes(rep)
    assert "no problems found" not in str(rep)


def test_two_lambdas_on_one_line_with_the_same_arguments_are_unchecked_not_guessed():
    cat = Catalog()
    f, g = (lambda x: "aprove"), (lambda x: "deny")
    cat.rule("q")(f)
    assert ("warning", "rule_unchecked") in _codes(lint(System(cat, [Q])))
    assert g("") == "deny"
