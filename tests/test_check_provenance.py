"""Where a check's evidence comes from: Catalog.check(family=, shares_call_with=) records it (metadata only: the
fingerprints do not change), and lint warns about a check that re-reads another part's model call and about a question
whose checks are all of one family."""
from solvi import Answer, Catalog, Question, System
from solvi.check import lint


def _system(fam_a=None, fam_b=None, shares=None):
    cat = Catalog()

    @cat.fn
    def draft(text: str) -> str:
        return text.upper()

    @cat.check(family=fam_a, shares_call_with=shares)
    def a_ok(draft: str) -> bool:
        return bool(draft)

    @cat.check(family=fam_b)
    def b_ok(text: str) -> bool:
        return len(text) < 100

    @cat.rule("ok")
    def ok(draft: str, a_ok: bool, b_ok: bool) -> str:
        return "yes" if a_ok and b_ok else "no"
    return System(cat, [Question("ok", "o", Answer.yes_no())])


def codes(system):
    return {f.code: f for f in lint(system).findings}


def test_the_metadata_is_recorded_and_leaves_fingerprints_alone():
    plain, tagged = _system(), _system("gpt", "gpt", "draft")
    p = tagged.catalog.parts["a_ok"]
    assert (p.family, p.shares_call_with) == ("gpt", "draft")
    assert plain.fingerprint()["catalog"] == tagged.fingerprint()["catalog"]


def test_lint_warns_on_a_shared_call_and_on_one_family():
    c = codes(_system("gpt", "gpt", "draft"))
    assert c["check_shares_call"].level == "warning" and "draft" in c["check_shares_call"].message
    assert c["checks_one_family"].level == "warning" and "'gpt'" in c["checks_one_family"].message


def test_lint_is_silent_for_independent_checks():
    c = codes(_system("gpt", "code"))
    assert "check_shares_call" not in c and "checks_one_family" not in c
    c = codes(_system())
    assert "check_shares_call" not in c and "checks_one_family" not in c


def test_a_shared_call_with_an_unknown_part_is_named():
    c = codes(_system(shares="nowhere"))
    assert "no part or question" in c["check_shares_call"].message
