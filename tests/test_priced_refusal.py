import pytest

from solvi.solutions.wrap import priced_proposal, wrap


def option(action, p, penalty, price=0, permitted=True):
    return dict(action=action, p_wrong=p, wrong_cost=penalty, price=price, permitted=permitted)


def test_refusal_can_be_more_expensive_than_acting_or_asking():
    default = option("refuse", 0.8, 4)
    act, ask = option("act", 0.1, 10), option("ask", 0.02, 10, 0.2)
    result = priced_proposal(default, [act, ask], evidence=["independent-verification"])
    assert result["action"] == "ask"
    assert result["loss"] == pytest.approx(0.4)
    assert result["default_loss"] == pytest.approx(3.2)
    assert priced_proposal(default, [act], evidence=["v"])["action"] == "act"
    assert priced_proposal(default, [option("act", 0.9, 10)], evidence=["v"]) is None


def test_hard_refusal_missing_permission_and_ties_do_not_override():
    default = option("refuse", 0.8, 4)
    assert priced_proposal({**default, "hard": True}, [option("act", 0, 10)], evidence=["v"]) is None
    assert priced_proposal(default, [option("act", 0, 10, permitted=False)], evidence=["v"]) is None
    assert priced_proposal(default, [option("act", 0.8, 4)], evidence=["v"]) is None
    assert priced_proposal(default, [option("refuse", 0, 4)], evidence=["v"]) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("p_wrong", True),
        ("p_wrong", -1),
        ("p_wrong", 2),
        ("wrong_cost", -1),
        ("wrong_cost", float("nan")),
        ("price", float("inf")),
        ("permitted", 1),
    ],
)
def test_invalid_loss_declarations_refuse(key, value):
    with pytest.raises(ValueError):
        priced_proposal(option("refuse", 0.5, 1), [{**option("act", 0.1, 1), key: value}], evidence=["v"])


def test_overflow_and_absent_evidence_refuse():
    with pytest.raises(ValueError):
        priced_proposal(option("refuse", 1, 1e308, 1e308), [], evidence=["v"])
    with pytest.raises(ValueError):
        priced_proposal(option("refuse", 0.5, 1), [], evidence=[])


def test_native_adapter_can_review_refusal_without_executing_action(tmp_path):
    class Agent:
        calls = 0

        def decide(self, state):
            self.calls += 1
            return "refuse"

    def review(view):
        return priced_proposal(option(view["default"], 0.8, 4), [option("act", 0.1, 10)], evidence=["safe-proposal"])

    agent = Agent()
    wrapped = wrap(agent, decision_points={"decide": review}, storage=tmp_path / "overrides.jsonl")
    assert wrapped.decide({"task": "allowed"}) == "act"
    assert agent.calls == 1
    from solvi.core.store import JSONLStorage

    row = next(iter(JSONLStorage(tmp_path / "overrides.jsonl").iter(kind="external_override"))).data
    assert row["proposal"]["pricing"]["default"]["action"] == "refuse"
    assert row["proposal"]["pricing"]["alternatives"][0]["expected_loss"] == 1
