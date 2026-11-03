"""Joint repair must preserve the confidence of the evidence path."""
import pytest

from solvi import Answer as QuestionAnswer, Catalog, Claim, Decision, Question, System


@pytest.mark.parametrize("answer_type", [QuestionAnswer.yes_no(), QuestionAnswer.choice(["yes", "no"])])
def test_joint_rule_repair_keeps_low_confidence_path(answer_type):
    cat = Catalog()

    @cat.fn
    def evidence(doc):
        return Claim(doc, confidence=0.1)

    @cat.rule("q")
    def q(evidence):
        return Decision("no", probs={"yes": 0.9, "no": 0.1})

    system = System(cat, [Question("q", "Question?", answer_type)])
    initial = system.ask({"doc": "evidence"})["q"]
    assert initial.answer == "no"
    assert initial.confidence <= 0.1

    @cat.constraint
    def force_yes(q):
        return q == "yes"

    repaired = system.ask({"doc": "evidence"})["q"]
    assert repaired.answer == "yes"
    assert repaired.repaired[0] == "no"
    assert repaired.confidence <= 0.1


class FakeHead:
    features = ["evidence"]

    def predict(self, values):
        return {"no": 0.8, "yes": 0.2}

    def contributions(self, values):
        return {"evidence": 1.0}


def test_joint_learned_head_repair_multiplies_selected_probability_by_path():
    cat = Catalog()

    @cat.fn
    def evidence(doc):
        return Claim(doc, confidence=0.1)

    system = System(cat, [Question("q", "Question?", QuestionAnswer.yes_no(), uses=["evidence"])])
    system.heads["q"] = FakeHead()
    initial = system.ask({"doc": "evidence"})["q"]
    assert initial.answer == "no"
    assert initial.confidence == pytest.approx(0.8 * 0.1)

    @cat.constraint
    def force_yes(q):
        return q == "yes"

    repaired = system.ask({"doc": "evidence"})["q"]
    assert repaired.answer == "yes"
    assert repaired.repaired[0] == "no"
    assert repaired.confidence == pytest.approx(0.2 * 0.1)


def test_joint_repair_cannot_cross_min_confidence_by_dropping_path_bound():
    cat = Catalog()

    @cat.fn
    def evidence(doc):
        return Claim(doc, confidence=0.1)

    @cat.rule('q')
    def q(evidence):
        return Decision('no', probs={'yes': 0.9, 'no': 0.1})

    @cat.constraint
    def force_yes(q):
        return q == 'yes'

    system = System(cat, [Question('q', 'Question?', QuestionAnswer.yes_no(), min_confidence=.5)])
    result = system.ask({'doc': 'evidence'})['q']
    assert result.status == 'abstain' and result.guard == 'low_confidence'
    assert result.confidence <= .1


def test_joint_repair_keeps_text_entry_point_confidence_bound():
    from types import SimpleNamespace
    cat = Catalog()

    @cat.rule('q')
    def q(doc):
        return Decision('no', probs={'yes': .9, 'no': .1})

    @cat.constraint
    def force_yes(q):
        return q == 'yes'

    system = System(cat, [Question('q', 'Question?', QuestionAnswer.yes_no())])
    response = system.ask({'doc': 'evidence'})
    read = SimpleNamespace(question='q', route={'confidence': .05}, fields={}, missing=[])
    results, _, _ = system._results(list(system.questions.values()), response.flow, response.trace, response.values, read)
    assert results['q'].answer == 'yes' and results['q'].confidence <= .05


@pytest.mark.parametrize('kind', ['rank', 'multi'])
def test_joint_primitive_repair_keeps_evidence_path_bound(kind):
    cat = Catalog()

    @cat.fn
    def evidence(doc):
        return Claim(doc, confidence=.1)

    @cat.rule('q')
    def q(evidence):
        return Decision(['b', 'a'] if kind == 'rank' else ['b'], probs={'a': .9, 'b': .1})

    answer = QuestionAnswer.rank(['a', 'b']) if kind == 'rank' else QuestionAnswer.multi(['a', 'b'])
    system = System(cat, [Question('q', 'Question?', answer)])
    initial = system.ask({'doc': 'evidence'})['q']
    assert initial.confidence <= .1

    @cat.constraint
    def include_a_first(q):
        return q[0] == 'a'

    repaired = system.ask({'doc': 'evidence'})['q']
    assert repaired.status == 'ok' and repaired.answer[0] == 'a'
    assert repaired.repaired and repaired.confidence <= .1
