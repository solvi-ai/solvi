"""Answer derivation consumes captured outputs, not another invocation of user code."""
import pytest

from solvi import Answer, Catalog, Claim, Question, System


class CountedHead:
    features = ['x']

    def __init__(self):
        self.predictions = self.explanations = 0

    def predict(self, values):
        self.predictions += 1
        return {'yes': .9, 'no': .1}

    def contributions(self, values):
        self.explanations += 1
        return {'x': .2}

    def fingerprint(self):
        return 'counted-head-v1'


def test_answers_of_reuses_head_prediction_and_explanation():
    sys = System(Catalog(), [Question('q', 'Question?', Answer.yes_no(), uses=['x'])])
    head = sys.heads['q'] = CountedHead()
    response = sys.ask({'x': 3})
    expected = response['q']
    assert (head.predictions, head.explanations) == (1, 1)
    derived = sys.answers_of(response.trace, ['q'], response.flow)['q']
    assert derived.answer == expected.answer and derived.why == expected.why
    assert derived.confidence == expected.confidence
    assert (head.predictions, head.explanations) == (1, 1)
    assert response.trace.replay(sys, response.flow)['ok']
    assert (head.predictions, head.explanations) == (2, 1)  # one independent model verification


def test_answers_of_reuses_callable_then_output():
    cat, calls = Catalog(), []

    def forced(x):
        calls.append(x)
        return True

    @cat.check(hard=True, then={'q': forced})
    def allowed(x):
        return False

    sys = System(cat, [Question('q', 'Question?', Answer.yes_no(), requires=['allowed'])])
    response = sys.ask({'x': 3})
    assert calls == [3]
    result = sys.answers_of(response.trace, ['q'], response.flow)['q']
    assert result.answer == 'yes' and result.status == 'forced'
    assert result.why == response['q'].why and calls == [3]
    assert response.trace.replay(sys, response.flow)['ok']
    assert calls == [3, 3]


def test_answers_of_reuses_failed_callable_then():
    cat, calls = Catalog(), []

    def forced(x):
        calls.append(x)
        raise ValueError('failed source')

    @cat.check(hard=True, then={'q': forced})
    def allowed(x):
        return False

    sys = System(cat, [Question('q', 'Question?', Answer.yes_no(), requires=['allowed'])])
    response = sys.ask({'x': 3})
    result = sys.answers_of(response.trace, ['q'], response.flow)['q']
    assert result.status == 'abstain' and result.why == response['q'].why
    assert calls == [3]


def test_missing_then_record_does_not_execute_callback_during_derivation():
    cat, calls = Catalog(), []

    def forced(x):
        calls.append(x)
        return True

    @cat.check(hard=True, then={'q': forced})
    def allowed(x):
        return False

    sys = System(cat, [Question('q', 'Question?', Answer.yes_no())])
    response = sys.ask({'x': 3})
    response.trace.records = [r for r in response.trace.records if r.kind != 'then']
    result = sys.answers_of(response.trace, ['q'], response.flow)['q']
    assert result.status == 'abstain' and result.guard == 'missing_record'
    assert calls == [3]


def test_missing_head_record_does_not_execute_prediction_during_derivation():
    sys = System(Catalog(), [Question('q', 'Question?', Answer.yes_no(), uses=['x'])])
    head = sys.heads['q'] = CountedHead()
    response = sys.ask({'x': 3})
    response.trace.records = [r for r in response.trace.records if r.kind != 'head']
    result = sys.answers_of(response.trace, ['q'], response.flow)['q']
    assert result.status == 'abstain' and result.guard == 'missing_record'
    assert (head.predictions, head.explanations) == (1, 1)


@pytest.mark.parametrize('change', ['remove', 'replace'])
def test_recorded_head_repair_keeps_recorded_features_after_head_changes(change):
    cat = Catalog()

    @cat.fn
    def evidence(doc):
        return Claim(doc, confidence=.2)

    @cat.constraint
    def force_no(q):
        return q == 'no'

    class Head(CountedHead):
        features = ['evidence']

        def predict(self, values):
            self.predictions += 1
            return {'yes': .6, 'no': .4}

    sys = System(cat, [Question('q', 'Question?', Answer.yes_no(), uses=['evidence'])])
    head = sys.heads['q'] = Head()
    response = sys.ask({'doc': 'source'})
    assert response['q'].confidence == pytest.approx(.08)
    if change == 'remove':
        del sys.heads['q']
    else:
        sys.heads['q'] = CountedHead()
    result = sys.answers_of(response.trace, ['q'], response.flow)['q']
    assert result.answer == 'no' and result.confidence == pytest.approx(.08)
    assert head.predictions == 1
