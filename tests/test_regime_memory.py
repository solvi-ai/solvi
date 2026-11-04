import copy
import json
import pytest
from solvi.experimental.regime_memory import RegimeMemory


def model(**kw):
    a = dict(
        epoch="unique-epoch",
        model_id="noise-v1",
        regimes={"A": "laws-a", "B": "laws-b"},
        prior=[0.5, 0.5],
        transition=[[0.9, 0.1], [0.1, 0.9]],
        probes={"p": {"yes": [0.8, 0.2], "no": [0.2, 0.8]}},
        max_delay=3,
    )
    a.update(kw)
    return RegimeMemory(**a)


def test_delayed_evidence_is_at_issue_time_not_current():
    m = model()
    t = m.issue("one", "p")
    m.advance()
    assert m.observe(t, "yes") == pytest.approx((0.74, 0.26))  # .8*.9 + .2*.1
    assert m.lawset_ids == ("laws-a", "laws-b")


def test_out_of_order_same_as_chronological_full_filter():
    a = model()
    a0 = a.issue("t0", "p")
    a.advance()
    a1 = a.issue("t1", "p")
    a.observe(a1, "no")
    actual = a.observe(a0, "yes")
    b = model()
    b0 = b.issue("t0", "p")
    b.observe(b0, "yes")
    b.advance()
    b1 = b.issue("t1", "p")
    assert actual == pytest.approx(b.observe(b1, "no"))
    assert actual == pytest.approx((0.74 * 0.2 / (0.74 * 0.2 + 0.26 * 0.8), 0.26 * 0.8 / (0.74 * 0.2 + 0.26 * 0.8)))


def test_snapshot_restore_detached_and_sorted_json():
    a = model()
    t = a.issue("late", "p")
    a.advance()
    snap = a.snapshot()
    b = RegimeMemory.restore(json.loads(json.dumps(snap, sort_keys=True)))
    snap["tickets"]["late"]["probe"] = "broken"
    assert b.observe(t, "yes") == pytest.approx(a.observe(t, "yes"))
    assert RegimeMemory.restore(b.snapshot()).weights == b.weights


@pytest.mark.parametrize("failure", ["duplicate", "unknown", "outcome", "epoch", "model", "stale"])
def test_bad_responses_atomic(failure):
    m = model(max_delay=0)
    t = m.issue("t", "p")
    if failure == "duplicate":
        m.observe(t, "yes")
    if failure == "stale":
        m.advance()
    bad = dict(t)
    value = "yes"
    if failure in ("epoch", "model"):
        bad[failure] = "wrong"
    if failure == "unknown":
        bad["ticket_id"] = "unknown"
    if failure == "outcome":
        value = "unknown"
    before = m.snapshot()
    weights = m.weights
    with pytest.raises(ValueError):
        m.observe(bad, value)
    assert m.snapshot() == before and m.weights == weights


def test_impossible_observation_and_limits_atomic():
    m = model(probes={"p": {"yes": [0, 0], "no": [1, 1]}}, max_pending=1, max_steps=2)
    t = m.issue("one", "p")
    before = m.snapshot()
    with pytest.raises(ValueError):
        m.observe(t, "yes")
    assert m.snapshot() == before
    with pytest.raises(ValueError):
        m.issue("two", "p")
    m.advance()
    before = m.snapshot()
    with pytest.raises(ValueError):
        m.advance()
    assert m.snapshot() == before


def test_config_snapshot_mutations_no_alias_and_validation():
    transition = [[1, 0], [0, 1]]
    m = model(transition=transition)
    transition[0][0] = 0
    assert m.advance() == (0.5, 0.5)
    bad = m.snapshot()
    bad["tickets"] = {"x": {"probe": "p", "step": 9, "outcome": None, "resolved": None}}
    with pytest.raises(ValueError):
        RegimeMemory.restore(bad)
    with pytest.raises(ValueError):
        model(prior=[float("nan"), 0])
