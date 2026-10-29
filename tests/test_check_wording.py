"""`solvi check`: the wording of questions — no catch-all option, bare-number levels, two questions in one, given facts
nobody reads. Warnings or notes (no_other_option: a closed label set is often right), never errors."""
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, System
from solvi.check import lint


def system(questions, rules=(), input_model=None, extra=None):
    """A catalog with one fact (`text_len`) and a rule per name in `rules`; the other questions are answered by a model
    (no rule)."""
    cat = Catalog()

    @cat.fn
    def text_len(text: str) -> int:
        return len(text)
    for q in rules:
        cat.rule(q)(lambda text_len: "a")
    if extra:
        extra(cat)
    return System(cat, questions, input_model=input_model)


def warned(sys, code, level="warning"):
    return {f.where: f.message for f in lint(sys).findings if f.code == code and f.level == level}


def noted(sys, code):
    return warned(sys, code, "note")


# --- (a) no_other_option
def test_a_model_choice_without_a_catch_all_is_a_note_once_with_the_fix():
    s = system([Question("team", "Which team handles it?", Answer.choice(["billing", "tech"]), uses=["text_len"])])
    got = noted(s, "no_other_option")
    assert list(got) == ["team"] and '"other"' in got["team"] and "Answer.maybe" in got["team"]
    assert warned(s, "no_other_option") == {}
    assert lint(s).ok and lint(s, strict=True).ok                       # a note: closed label sets pass --strict


def test_a_catch_all_option_or_not_stated_or_a_rule_silences_it():
    ok = [Question("a", "Which team?", Answer.choice(["billing", "other"]), uses=["text_len"]),
          Question("b", "Which team?", Answer.choice(["billing", "None of these"]), uses=["text_len"]),
          Question("c", "Which team?", Answer.choice(["billing", "unknown"]), uses=["text_len"]),
          Question("d", "Which team?", Answer.maybe(Answer.choice(["billing", "tech"])), uses=["text_len"]),
          Question("e", "Which team?", Answer.choice({"billing": "charges", "general": "anything else"}),
                   uses=["text_len"]),
          Question("f", "Which team?", Answer.choice(["a", "b"]))]           # answered by a rule: code decides
    assert noted(system(ok, rules=["f"]), "no_other_option") == {}


# --- (b) numeric_levels
def test_bare_number_levels_warn_and_described_levels_do_not():
    bare = Question("risk", "How risky is it?", Answer.ordinal(["1", "2", "3"]), uses=["text_len"])
    told = Question("risk2", "How risky is it?", Answer.ordinal({1: "none", 2: "some", 3: "high"}), uses=["text_len"])
    words = Question("risk3", "How risky is it?", Answer.ordinal(["low", "medium", "high"]), uses=["text_len"])
    got = warned(system([bare, told, words]), "numeric_levels")
    assert list(got) == ["risk"] and "say what each means" in got["risk"]


# --- (c) compound_question
def test_a_question_that_asks_two_things_warns():
    qs = [Question("q1", "Is the invoice paid? Is it late?", Answer.yes_no()),
          Question("q2", "Is the invoice paid and whether it is late", Answer.yes_no()),
          Question("q3", "Say if the refund is due or whether a person decides", Answer.yes_no())]
    got = warned(system(qs, rules=["q1", "q2", "q3"]), "compound_question")
    assert set(got) == {"q1", "q2", "q3"} and "split it into two questions" in got["q1"]


def test_ordinary_questions_do_not_warn_as_compound():
    texts = ["Is the tool in the guard's catalog?", "Refund, send to a person, or reject?",
             "Which team should handle this ticket?", "Is the amount above the limit and the customer new?",
             "Approve the expense?", "Does the contract say whether it renews?", "Should we approve or reject?",
             "Is the guard running (not paused)?", "What is the total (in EUR)?"]
    qs = [Question(f"q{i}", t, Answer.yes_no()) for i, t in enumerate(texts)]
    assert warned(system(qs, rules=[q.name for q in qs]), "compound_question") == {}


# --- (d) unused_given
class Order(BaseModel):
    text: str
    total: float
    note: str = ""


def test_an_input_field_nothing_reads_warns():
    def more(cat):
        @cat.check(hard=False)
        def big(total: float) -> bool:
            return total > 10
    s = system([Question("q", "Ok?", Answer.yes_no())], rules=["q"], input_model=Order, extra=more)
    got = warned(s, "unused_given")
    assert list(got) == ["note"] and "nothing reads" in got["note"]


def test_unused_given_is_skipped_when_a_learned_head_may_read_any_fact():
    s = system([Question("q", "Ok?", Answer.yes_no())], input_model=Order)   # no rule, no uses: a head reads all
    assert warned(s, "unused_given") == {}
    s = system([Question("q", "Ok?", Answer.yes_no(), uses=["note", "total"])], input_model=Order)
    assert warned(s, "unused_given") == {}


def test_a_field_read_only_by_a_guarantee_is_not_unused():
    """A guarantee's groups= (a field name, a hierarchy, a function's parameters) and a signal= naming a fact read the
    field: it is not flagged."""
    from solvi.core.guarantees.guarantee import QuestionGuard

    def more(cat):
        @cat.check(hard=False)
        def big(total: float) -> bool:
            return total > 10
    for groups in ("note", ["note"], lambda note: note[:1]):
        s = system([Question("q", "Ok?", Answer.yes_no())], rules=["q"], input_model=Order, extra=more)
        assert list(warned(s, "unused_given")) == ["note"]
        s.guards["q"] = QuestionGuard("q", None, groups=groups)
        assert warned(s, "unused_given") == {}
    s = system([Question("q", "Ok?", Answer.yes_no())], rules=["q"], input_model=Order)
    s.guards["q"] = QuestionGuard("q", None, signal="note")
    assert list(warned(s, "unused_given")) == ["total"]                 # `note` is the guarantee's signal
