"""solvi.core.contract: a declared contract on an answer — its type matches what the question asks, it is not the
input it was read from (an echo), and an applicability precondition is checked before the answerer is called. A
certificate on a program certifies that it runs as it should, not that it read the question right: these checks are
hard (they override confidence), each with a named reason, in a System, in the dispatcher's slow path and in a
specialist."""
import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, System
from solvi.core.contract import Contract, Violation
from solvi.core.dispatch import AskPath, Dispatcher, SlowPath, Thought
from solvi.core.specialist import BLOCKED, Checked, Specialist
from solvi.testing.conformance import ConformanceError, check_specialist


# ------------------------------------------------------------------------------------------------ the contract itself
def test_violations_name_the_wrong_type_and_the_echo():
    c = Contract(returns=int)
    assert c.violations(12, {"text": "how many?"}) == []
    [v] = c.violations("12", {"text": "how many?"})
    assert isinstance(v, Violation) and v.code == "wrong_type" and "int" in v.message
    assert c.violations(True, {})[0].code == "wrong_type"                     # a bool is not a number here
    assert Contract(returns=float).violations(3, {}) == []                    # an int is a float
    [e] = Contract().violations("  Hello  world ", {"text": "hello world"})
    assert e.code == "echo" and "text" in e.message
    assert Contract().violations(12, {"text": "12"})[0].code == "echo"        # the same number as text
    assert Contract(echo=False).violations("hello world", {"text": "hello world"}) == []
    assert Contract().violations(None, {"text": None}) == []                  # an abstention is not checked


def test_the_answer_type_of_the_question_is_the_default_type():
    c = Contract()
    assert c.violations("x", {}, Answer.estimate()) [0].code == "wrong_type"
    assert c.violations(3.5, {}, Answer.estimate()) == []
    assert c.violations("c", {}, Answer.choice(["a", "b"]))[0].code == "wrong_type"


def test_applies_runs_the_precondition_and_names_why_not():
    def asks_a_count(state) -> bool:
        """The question asks for a count."""
        return "how many" in state["text"].lower()
    c = Contract(applicable=asks_a_count)
    assert c.applies({"text": "How many apples?"}) == (True, None)
    ok, why = c.applies({"text": "Which apple?"})
    assert not ok and "asks_a_count" in why
    boom = Contract(applicable=lambda s: s["missing"])
    ok, why = boom.applies({})
    assert not ok and "could not be evaluated" in why                        # a precondition that fails is not met
    assert Contract(applicable=asks_a_count).fingerprint() != Contract().fingerprint()


# ------------------------------------------------------------------------------------------------ in a System
def word_system(contract=None, calls=None):
    cat = Catalog()

    @cat.fn
    def cleaned(text):
        return text.strip()

    @cat.rule("first_word")
    def first_word(cleaned) -> str:
        if calls is not None:
            calls.append(cleaned)
        return cleaned if " " not in cleaned else cleaned.split()[0]
    if contract is not None:
        cat.contract("first_word", **contract)
    return System(cat, [Question("first_word", "The first word?", Answer.span("text"))])


def test_an_echo_of_the_input_abstains_with_the_contract_as_its_reason():
    s = word_system({})
    r = s.ask({"text": "hello world"})["first_word"]
    assert (r.status, r.answer) == ("ok", "hello")
    r = s.ask({"text": "hello"})["first_word"]                               # the answer is the whole input
    assert (r.status, r.answer, r.guard) == ("abstain", None, "contract")
    assert "echo" in r.why and "'hello'" in r.why
    assert r.extra["contract"][0]["code"] == "echo"
    assert word_system().ask({"text": "hello"})["first_word"].answer == "hello"     # no contract: answered


def test_a_wrong_type_abstains_even_with_full_confidence():
    s = word_system({"returns": int, "echo": False})
    res = s.ask({"text": "hello world"})
    r = res["first_word"]
    assert r.status == "abstain" and r.guard == "contract" and r.confidence == 0.0 and "wrong_type" in r.why
    assert res.trace.replay(s)["ok"]


def test_a_question_that_does_not_apply_is_not_computed():
    calls = []

    def one_line(state) -> bool:
        """The text is one line."""
        return "\n" not in state["text"]
    s = word_system({"applicable": one_line}, calls)
    r = s.ask({"text": "a\nb"})["first_word"]
    assert (r.status, r.guard) == ("abstain", "not_applicable") and calls == []
    assert "not applicable" in r.why and "one_line" in r.why
    assert s.ask({"text": "hello world"})["first_word"].answer == "hello" and calls == ["hello world"]


def test_echo_names_the_inputs_to_compare_with():
    cat = Catalog()

    @cat.rule("label")
    def label(text, hint) -> str:
        return hint
    cat.contract("label", echo=["text"])
    s = System(cat, [Question("label", "The label?", Answer.choice(["a", "b"]))])
    assert s.ask({"text": "x", "hint": "a"})["label"].answer == "a"          # equal to hint: not compared
    assert s.ask({"text": "a", "hint": "a"})["label"].guard == "contract"


def test_contract_declarations_are_checked_and_change_the_fingerprint():
    cat = Catalog()

    @cat.rule("q")
    def q(text) -> str:
        return text
    with pytest.raises(ValueError):
        cat.contract("nope")
    with pytest.raises(TypeError):
        cat.contract("q", applicable="yes")
    a = System(cat, [Question("q", "?", Answer.span("text"))]).fingerprint()
    cat.contract("q", echo=False)
    b = System(cat, [Question("q", "?", Answer.span("text"))]).fingerprint()
    assert a != b


def test_the_audit_counts_contract_refusals():
    s = word_system({})
    s.ask({"text": "hello"})
    assert s.stats["contract_refused"] == 1
    assert "contract" in s.safeguard_summary()


# ------------------------------------------------------------------------------------------------ in the dispatcher
def unsure():
    """System 1: always abstains (nothing it can answer alone)."""
    cat = Catalog()

    @cat.rule("count")
    def count(text):
        return None
    return System(cat, [Question("count", "How many items?", Answer.estimate())])


def slow_system(answer_of, calls):
    cat = Catalog()

    @cat.rule("count")
    def count(text):
        calls.append(text)
        return answer_of(text)
    return System(cat, [Question("count", "How many items?", Answer.estimate())])


def applies_to_counts(state) -> bool:
    """The text lists items to count."""
    return "items:" in state["text"]


def test_a_non_applicable_slow_path_is_not_called_and_a_person_gets_it():
    calls = []
    slow = AskPath(slow_system(lambda t: float(t.count(",") + 1), calls), contract=Contract(applicable=applies_to_counts))
    d = Dispatcher(unsure(), slow)
    res = d.ask({"text": "what a lovely day, isn't it"})
    assert calls == [] and (res.by, res.answer) == ("human", None)
    assert res.s2 is not None and res.s2.contract["applicable"] is False
    assert any("not applicable" in r and "applies_to_counts" in r for r in res.reasons)
    assert d.replay(res)["ok"]
    res = d.ask({"text": "items: a, b, c"})
    assert (res.by, res.answer) == ("s2", 3.0) and calls == ["items: a, b, c"]
    assert d.replay(res)["ok"]


def test_the_slow_path_falls_back_on_its_systems_catalog_contract():
    calls = []
    s2 = slow_system(lambda t: 1.0, calls)
    s2.catalog.contract("count", applicable=applies_to_counts)
    d = Dispatcher(unsure(), AskPath(s2))
    assert d.ask({"text": "nothing to count"}).by == "human" and calls == []


class Echoing(SlowPath):
    """A slow path that answers with what it read (a program certified on its examples, called on the wrong input)."""
    mode = "echo_test"

    def __init__(self, system, answer, **kw):
        super().__init__(system, **kw)
        self.answer = answer

    def think(self, state, question, **kw):
        return Thought(self.mode, self.answer(state), True)

    def fingerprint(self):
        return "echo-test"


def test_a_slow_path_answer_that_echoes_or_has_the_wrong_type_is_not_accepted():
    s2 = slow_system(lambda t: 0.0, [])
    d = Dispatcher(unsure(), Echoing(s2, lambda st: st["text"], contract=Contract(echo=True)))
    res = d.ask({"text": "items: a"})
    assert res.by == "human" and res.s2.accepted is False
    assert res.s2.contract["violations"][0]["code"] in ("echo", "wrong_type")
    assert any("contract" in r for r in res.reasons)
    assert d.replay(res)["ok"]
    d2 = Dispatcher(unsure(), Echoing(s2, lambda st: "7", contract=Contract()))
    res = d2.ask({"text": "items: a"})
    assert res.by == "human" and res.s2.contract["violations"][0]["code"] == "wrong_type"
    assert d2.replay(res)["ok"]
    d3 = Dispatcher(unsure(), Echoing(s2, lambda st: 7.0, contract=Contract()))
    assert d3.ask({"text": "items: a"}).answer == 7.0


def test_a_recorded_contract_refusal_that_no_longer_follows_is_a_replay_mismatch():
    s2 = slow_system(lambda t: 0.0, [])
    d = Dispatcher(unsure(), Echoing(s2, lambda st: "7", contract=Contract()))
    res = d.ask({"text": "items: a"})
    res.s2.contract["violations"] = []
    assert not d.replay(res)["ok"]


def test_thought_contract_round_trips():
    th = Thought("ask", None, False, "not applicable: x", None, contract={"applicable": False, "why": "x"})
    back = Thought.from_dict(th.to_dict())
    assert back.contract == th.contract
    assert "contract" not in Thought("ask").to_dict()                         # the record is unchanged without one


# ------------------------------------------------------------------------------------------------ in a specialist
class Spec(BaseModel):
    text: str


class Copier(Specialist):
    """A toy specialist that renders the spec's text: it applies to sources with a colon only."""
    name = "copier"
    version = "1"
    spec_type = Spec

    def applicable(self, source, question):
        return ":" in source

    def check(self, spec, source):
        return Checked(spec, [], [spec.text])

    def render(self, checked):
        return checked.verified.text


def test_a_specialist_that_does_not_apply_is_blocked_without_calling_its_proposer():
    calls = []

    def proposer(source, question):
        calls.append(source)
        return {"text": source.split(":")[1].strip().upper()}
    sp = Copier(proposer)
    run = sp.run("no colon here")
    assert run.output is None and calls == []
    assert [i.code for i in run.issues] == ["not_applicable"] and run.issues[0].severity == BLOCKED
    assert sp.replay(run.to_dict(), "no colon here").ok
    assert sp.run("name: ada").output == "ADA"


def test_a_specialist_output_that_echoes_its_source_is_blocked():
    sp = Copier(lambda s, q: {"text": s})
    run = sp.run("key: value")
    assert run.output is None and any(i.code == "echo" for i in run.issues)
    assert sp.replay(run.to_dict(), "key: value").ok


def test_check_specialist_checks_the_contract():
    def proposer(source, question):
        return {"text": source.split(":")[1].strip().upper()}
    rep = check_specialist(Copier(proposer), ["name: ada", "city: oslo"], inapplicable=["plain text"])
    assert rep["not_applicable"] == 1

    class Careless(Copier):
        name = "careless"

        def applicable(self, source, question):
            return True
    with pytest.raises(ConformanceError, match="not applicable"):
        check_specialist(Careless(proposer), ["name: ada"], inapplicable=["plain: text"])
