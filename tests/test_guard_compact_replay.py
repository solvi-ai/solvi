"""Guard.replay / replay_all on a store that keeps decisions compact (record="compact")."""
import pytest

from solvi.core.store import JSONLStorage
from solvi.solutions.guard import Guard


def _guard(path, cap=100, record="compact"):
    g = Guard(storage=JSONLStorage(str(path), record=record))

    @g.tool
    def refund(amount: float) -> str:
        """Refund."""
        return "ok"

    @g.policy("refund")
    def limit(amount: float) -> bool:
        return amount <= cap

    return g


@pytest.mark.parametrize("record", ["full", "compact"])
def test_replay_all_on_a_store_replays(tmp_path, record):
    g = _guard(tmp_path / "g.jsonl", record=record)
    d = g.check({"name": "refund", "arguments": {"amount": 50}}, [("user", "refund 50")])
    assert g.replay_all() == []
    assert g.replay(d.stored_id)["ok"]


def test_a_changed_policy_does_not_replay_a_compact_record(tmp_path):
    g = _guard(tmp_path / "g.jsonl")
    g.check({"name": "refund", "arguments": {"amount": 50}}, [("user", "refund 50")])
    g2 = _guard(tmp_path / "g.jsonl", cap=10)
    bad = g2.replay_all()
    assert len(bad) == 1 and bad[0]["tool"] == "refund" and bad[0]["mismatches"]
