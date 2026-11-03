import pytest

from solvi import Answer, Catalog, Claim, Decision, Question, System


def make(*, constrained, path=1., threshold=None):
    cat = Catalog()

    @cat.fn
    def evidence(doc):
        return Claim(doc, confidence=path)

    @cat.rule('q')
    def q(evidence):
        return Decision('no', probs={'yes': .9, 'no': .1})

    if constrained:
        @cat.constraint
        def force_yes(q):
            return q == 'yes'
    system = System(cat, [Question('q', 'Question?', Answer.yes_no(), min_confidence=threshold)])
    system.calib['q'] = (0., 0.)  # a legitimate constant Platt map: every raw confidence -> .5
    return system


@pytest.mark.parametrize('constrained', [False, True])
def test_calibration_applies_to_the_final_selected_answer(constrained):
    result = make(constrained=constrained).ask({'doc': 'evidence'})['q']
    assert result.answer == ('yes' if constrained else 'no')
    assert result.confidence == pytest.approx(.5)


@pytest.mark.parametrize('constrained', [False, True])
def test_calibration_retains_evidence_path_bound(constrained):
    result = make(constrained=constrained, path=.1).ask({'doc': 'evidence'})['q']
    assert result.status == 'ok' and result.confidence <= .1


def test_repair_cannot_skip_min_confidence_after_calibration():
    result = make(constrained=True, threshold=.6).ask({'doc': 'evidence'})['q']
    assert result.status == 'abstain' and result.guard == 'low_confidence'
    assert result.confidence == pytest.approx(.5)


class Head:
    features = ['doc']

    def predict(self, values):
        return {'no': .8, 'yes': .2}

    def contributions(self, values):
        return {'doc': 1.}


def test_head_repair_calibrates_selected_probability_once():
    cat = Catalog()

    @cat.constraint
    def force_yes(q):
        return q == 'yes'

    system = System(cat, [Question('q', 'Question?', Answer.yes_no(), uses=['doc'])])
    system.heads['q'] = Head()
    system.calib['q'] = (2., 0.)
    result = system.ask({'doc': 1.})['q']
    assert result.answer == 'yes'
    # sigmoid(2 * logit(.2)); recalibrating that again would give a different value.
    assert result.confidence == pytest.approx(.2 ** 2 / (.2 ** 2 + .8 ** 2))


def test_selected_score_is_calibrated_before_text_entry_point_cap():
    import math
    from types import SimpleNamespace
    system = make(constrained=True)
    system.calib['q'] = (1., -2.)
    response = system.ask({'doc': 'evidence'})
    read = SimpleNamespace(question='q', route={'confidence': .7}, fields={}, missing=[])
    results, _, _ = system._results(list(system.questions.values()), response.flow, response.trace, response.values, read)
    expected = 1 / (1 + math.exp(-(math.log(.9 / .1) - 2)))
    assert results['q'].answer == 'yes'
    assert results['q'].confidence == pytest.approx(min(expected, .7))


@pytest.mark.parametrize('kind', ['multi', 'rank'])
def test_calibrated_primitive_repair_uses_path_score(kind):
    import math
    cat = Catalog()

    @cat.fn
    def evidence(doc):
        return Claim(doc, confidence=.5)

    @cat.rule('q')
    def q(evidence):
        return Decision(['b', 'a'] if kind == 'rank' else ['b'], probs={'a': .9, 'b': .1})

    @cat.constraint
    def first_a(q):
        return q[0] == 'a'

    answer = Answer.rank(['a', 'b']) if kind == 'rank' else Answer.multi(['a', 'b'])
    system = System(cat, [Question('q', 'Question?', answer)])
    system.calib['q'] = (1., -2.)
    response = system.ask({'doc': 'evidence'})
    assert response['q'].answer[0] == 'a'
    # Selected .9 (rank) / .81 (multi) is capped at the .5 evidence path before Platt.
    assert response['q'].confidence == pytest.approx(1 / (1 + math.exp(2)))
    assert response.trace.replay(system, response.flow)['ok']


def test_nonconstant_head_calibration_uses_probability_times_path():
    import math
    cat = Catalog()

    @cat.fn
    def evidence(doc):
        return Claim(doc, confidence=.5)

    @cat.constraint
    def force_yes(q):
        return q == 'yes'

    class PathHead(Head):
        features = ['evidence']

        def contributions(self, values):
            return {'evidence': 1.}

    system = System(cat, [Question('q', 'Question?', Answer.yes_no(), uses=['evidence'])])
    system.heads['q'] = PathHead()
    system.calib['q'] = (1., -2.)
    result = system.ask({'doc': 'evidence'})['q']
    assert result.answer == 'yes'
    raw = .2 * .5
    assert result.confidence == pytest.approx(1 / (1 + math.exp(-(math.log(raw / (1 - raw)) - 2))))
