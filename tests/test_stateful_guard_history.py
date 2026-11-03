import copy
import pytest
from solvi import Catalog, Question, System, Response
from solvi.core.guarantees.guarantee import QuestionGuard
from solvi.experimental.stream import StreamPromise


def make(staged=False, auto=False):
    cat = Catalog()

    @cat.rule("a")
    def a(x: int) -> bool:
        return x > 0

    questions = [Question("a", "A?")]
    if staged:
        cat.answer_fact("a")

        @cat.rule("b")
        def b(a: str) -> bool:
            return a == "yes"

        questions.append(Question("b", "B?"))
    sys = System(cat, questions)
    promise = StreamPromise(max_risk=0.1, kind="joint", **({"grid_from": 3} if auto else {"grid": [0.5]}))
    sys.guards["a"] = QuestionGuard("a", promise)
    return sys, promise


@pytest.mark.parametrize("staged", [False, True])
def test_guard_replay_remains_historical_after_real_stream_learning(staged):
    sys, p = make(staged)
    out = sys.ask({"x": 1})
    assert out.trace.replay(sys, out.flow)["ok"]
    original = copy.deepcopy(out["a"].extra["guarantee"])
    p.learn(1.0, "yes", "yes", sampling_policy="complete")
    assert p.n == 1
    again = sys.answers_of(out.trace, ["a"], out.flow)["a"]
    assert again.extra["guarantee"] == original
    assert out.trace.replay(sys, out.flow)["ok"]
    restored = Response.from_json(out.to_json(), catalog=sys)
    assert restored.trace.replay(sys, restored.flow)["ok"]
    sys.guards["a"] = QuestionGuard("a", StreamPromise(max_risk=0.2, kind="joint", grid=[0.5]))
    assert not restored.trace.replay(sys, restored.flow)["ok"]


@pytest.mark.parametrize("staged", [False, True])
def test_private_decide_does_not_observe_or_freeze_auto_grid(staged):
    sys, p = make(staged, auto=True)
    before = copy.deepcopy(p.dump())
    for _ in range(4):
        out = sys._decide({"x": 1})
        assert not any(r.kind == "guard" for r in out.trace.records)
    assert p.dump() == before
    sys.ask({"x": 1})
    assert p.seen == before["seen"] + 1
