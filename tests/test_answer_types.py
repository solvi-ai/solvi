"""Answer types: ordinal (median of a learned distribution), multi (subset; rules and learned per-option heads), descriptions."""
import random

import pytest

from solvi import Answer, Catalog, Question, System


def test_multi_rule_normalizes_to_option_order():
    cat = Catalog()

    @cat.rule("tags")
    def tags(text):
        return {t for t in ("urgent", "billing", "legal") if t in text}

    s = System(cat, [Question("tags", "", Answer.multi(["legal", "billing", "urgent"]))])
    assert s.ask({"text": "urgent billing issue"})["tags"].answer == ("billing", "urgent")
    assert s.ask({"text": "hello"})["tags"].answer == ()
    with pytest.raises(ValueError):
        Answer.multi(["a"]).normalize(["b"])


def test_multi_learned_with_fit_fast_and_teach():
    cat = Catalog()

    @cat.fn
    def amount(x):
        return x

    @cat.fn
    def foreign(country):
        return country != "DE"

    q = Question("flags", "", Answer.multi(["big", "foreign"]))
    rng = random.Random(0)

    def ex():
        x, c = rng.uniform(0, 100), rng.choice(["DE", "FR", "US"])
        return {"x": x, "country": c}, [f for f, on in (("big", x > 60), ("foreign", c != "DE")) if on]
    s = System(cat, [q])
    s.fit("flags", [ex() for _ in range(200)], select=False)
    test = [ex() for _ in range(200)]
    acc = sum(s.ask(st)["flags"].answer == Answer.multi(["big", "foreign"]).normalize(y) for st, y in test) / len(test)
    assert acc > 0.8
    assert s.teach("flags", *ex()) is not None


def test_ordinal_answers_with_the_median():
    from solvi.core.system import System as S
    at = Answer.ordinal({"low": "no action", "medium": "watch", "high": "act now"})
    assert at.options == ["low", "medium", "high"] and at.descriptions["high"] == "act now"
    with pytest.raises(AttributeError, match=r"removed in 0.9: use options.index\(v\)"):
        at.rank("medium")

    class FakeHead:                          # bimodal distribution: argmax would say "high", the median is "medium"
        features = []

        def predict(self, row):
            return {"low": 0.40, "medium": 0.15, "high": 0.45}

        def contributions(self, row):
            return {}
    cat = Catalog()
    s = S(cat, [Question("risk", "", at)])
    s.heads["risk"] = FakeHead()
    assert s.ask({"x": 1})["risk"].answer == "medium"


class _Head:
    features = []

    def __init__(self, p):
        self.p = p

    def predict(self, row):
        return self.p

    def contributions(self, row):
        return {}


def _guard(p_verdict, p_harm, fixed_verdict=None):
    cat = Catalog()

    @cat.constraint
    def unsafe_if_harm(verdict, harm):
        return harm == "none" or verdict == "unsafe"

    if fixed_verdict:
        @cat.rule("verdict")
        def verdict_rule(x):
            return fixed_verdict
    s = System(cat, [Question("verdict", "", Answer.choice(["safe", "unsafe"])),
                     Question("harm", "", Answer.choice(["none", "prompt_injection", "pii"]))])
    if not fixed_verdict:
        s.heads["verdict"] = _Head(p_verdict)
    s.heads["harm"] = _Head(p_harm)
    return s.ask({"x": 1})


def test_joint_decoding_fixes_contradictory_learned_answers():
    # independently: harm = prompt_injection (0.82), verdict = safe (0.52) — a contradiction
    r = _guard({"safe": 0.52, "unsafe": 0.48}, {"none": 0.10, "prompt_injection": 0.82, "pii": 0.08})
    assert (r["verdict"].answer, r["harm"].answer) == ("unsafe", "prompt_injection")
    assert r.feasible and "unsafe_if_harm" in r["verdict"].why


def test_rule_answers_stay_fixed_and_learned_ones_adapt_or_report():
    r = _guard(None, {"none": 0.30, "prompt_injection": 0.60, "pii": 0.10}, fixed_verdict="safe")
    assert r["verdict"].answer == "safe" and r["harm"].answer == "none"      # the rule wins, the learned answer adapts
    assert r.feasible


def test_consistent_answers_untouched():
    r = _guard({"safe": 0.9, "unsafe": 0.1}, {"none": 0.9, "prompt_injection": 0.05, "pii": 0.05})
    assert (r["verdict"].answer, r["harm"].answer) == ("safe", "none") and r.feasible and not r.violations


# ---------------------------------------------------------------- constraints: what used to pass without a word
def _yes_no_system(n, constraint=True):
    """n yes/no questions, each answered by a model with P(yes) 0.6 + 0.01·i, under "at most one yes"."""
    import inspect
    from solvi import Decision
    cat = Catalog()
    names = [f"m{i}" for i in range(n)]

    class M:
        version = "1"
    for i, name in enumerate(names):
        def rule(x, _p=0.6 + 0.01 * i):
            return Decision("yes", {"yes": _p, "no": 1 - _p})
        rule.__name__ = name
        rule.__signature__ = inspect.Signature([inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        cat.rule(name, model=M())(rule)
    if constraint:
        src = f"def at_most_one({', '.join(names)}):\n    return [{', '.join(names)}].count('yes') <= 1\n"
        ns = {}
        exec(src, ns)  # noqa: S102 — a constraint over n questions, written out
        cat.constraint(ns["at_most_one"])
    return System(cat, [Question(q, "?", Answer.yes_no()) for q in names]), names


def test_joint_decoding_says_when_it_could_not_try_every_combination():
    """15 yes/no answers under one constraint were repaired; with 16 nothing was, and nothing said why."""
    s, names = _yes_no_system(15)
    res = s.ask({"x": 1})
    assert res.feasible and [res[q].answer for q in names].count("yes") == 1 and res["m14"].answer == "yes"
    s, names = _yes_no_system(16)
    res = s.ask({"x": 1})
    assert not res.feasible and res.violations == ["at_most_one"] and all(res[q].answer == "yes" for q in names)
    assert all("not repaired: at_most_one broken, and joint decoding tried only the 1 most probable answer(s) of each "
               "question (65,536 combinations of 16 answers exceed its limit of 50,000)" in res[q].why for q in names)


def test_a_constraint_with_a_wrong_name_a_duplicate_or_an_exception_is_not_silent():
    cat = Catalog()

    @cat.rule("verdict")
    def verdict(x):
        return "safe"

    @cat.rule("harm")
    def harm(x):
        return "none"

    @cat.constraint
    def fit(verdict, harn):                              # a typo: it would never apply
        return True
    qs = [Question("verdict", "?", Answer.choice(["safe", "unsafe"])), Question("harm", "?", Answer.choice(["none", "high"]))]
    with pytest.raises(ValueError, match="constraint fit reads harn, which is not a question of this system"):
        System(cat, qs)
    del cat.constraints["fit"]

    @cat.constraint
    def together(verdict, harm):
        return verdict.no_such_attribute
    with pytest.raises(ValueError, match="constraint together is already in the catalog"):
        @cat.constraint
        def together(verdict):                           # noqa: F811 — the duplicate is the point
            return True
    res = System(cat, qs).ask({"x": 1})
    assert not res.feasible and res.violations == ["together"]
    assert all("constraint together raised AttributeError: 'str' object has no attribute 'no_such_attribute'" in r.why
               for r in res.results.values())


def test_answer_factories_refuse_what_they_would_ignore_or_crash_on():
    """estimate(lo=0, hi=10) returned an estimate with no bins; estimate([0, 5], lo=1, hi=2, step=1) ignored three
    keywords; from_type(Literal[1, "a"]) raised TypeError: '<' not supported; choice([]) and duplicates were accepted."""
    import enum
    from typing import Literal
    with pytest.raises(ValueError, match="needs step="):
        Answer.estimate(lo=0, hi=10)
    with pytest.raises(ValueError, match="not both"):
        Answer.estimate([0, 5], lo=1, hi=2, step=1)
    with pytest.raises(ValueError, match="lo < hi"):
        Answer.estimate(lo=10, hi=0, step=1)
    assert Answer.estimate(lo=0, hi=10, step=5).bins == [0, 5, 10]
    with pytest.raises(ValueError, match="at least one option"):
        Answer.choice([])
    with pytest.raises(ValueError, match="'a' is given twice"):
        Answer.multi(["a", "b", "a"])
    with pytest.raises(ValueError, match="not the string"):
        Answer.choice("ab")

    class Mixed(enum.Enum):
        ONE = 1
        BEE = "b"
    assert Answer.from_type(Literal[1, "a"]).options == [1, "a"] and Answer.from_type(Mixed).options == [1, "b"]
    assert Answer.from_type(Literal["no", "yes"]).kind == "yes_no"
