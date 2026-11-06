"""Packed frame and feedback sampler known answers."""
import json
from dataclasses import FrozenInstanceError

import pytest
from solvi.experimental.experience import PackedObservationFrame, PackedStratifiedSampler, WorkBudget, BudgetStop


def fixture():
    records = [(f"p{i}", {"f0": 0 if i < 4 else 7}) for i in range(9)]
    records += [(f"c{i}", {"f0": 0 if i < 10 else 7}) for i in range(16)]
    labels = tuple((f"p{i}", i == 4) for i in range(9))
    return records, labels


def make(strategy="adaptive", **kwargs):
    records, pilot = fixture()
    budget = WorkBudget(100000)
    frame = PackedObservationFrame(records, budget, source="host")
    return frame, PackedStratifiedSampler(frame, pilot, budget, strategy=strategy, **kwargs), budget


def test_feedback_changes_exact_priority():
    _, positive, budget = make()
    assert positive.features == ("f0",)
    assert positive.choose(budget) == "c0"
    positive.observe("c0", True, budget)
    assert positive.choose(budget) == "c1"
    _, negative, budget = make()
    assert negative.choose(budget) == "c0"
    negative.observe("c0", False, budget)
    assert negative.choose(budget) == "c10"


def test_round_robin_pending_and_strict_feedback():
    _, sampler, budget = make("round_robin")
    assert sampler.choose(budget) == "c0"
    with pytest.raises(ValueError):
        sampler.choose(budget)
    with pytest.raises(ValueError):
        sampler.observe("c0", 1, budget)
    with pytest.raises(ValueError):
        sampler.observe("c1", True, budget)
    sampler.observe("c0", False, budget)
    assert sampler.choose(budget) == "c10"


def test_snapshot_mid_pending_native_and_json_no_refunds():
    frame, sampler, budget = make()
    sampler.observe(sampler.choose(budget), True, budget)
    sampler.choose(budget)
    for saved in (sampler.to_dict(), json.loads(json.dumps(sampler.to_dict()))):
        before = budget.spent
        restored = PackedStratifiedSampler.restore(frame, saved, budget)
        assert budget.spent > before
        assert restored.to_dict() == sampler.to_dict()
        restored.observe(restored.pending, False, budget)
        sampler_copy = PackedStratifiedSampler.restore(frame, sampler.to_dict(), budget)
        sampler_copy.observe(sampler_copy.pending, False, budget)
        assert restored.choose(budget) == sampler_copy.choose(budget)
    changed = PackedObservationFrame([(name, {"f0": 1}) for name in frame.names], budget, source="host")
    with pytest.raises(ValueError):
        PackedStratifiedSampler.restore(changed, sampler.to_dict(), budget)
    changed = PackedObservationFrame(fixture()[0], budget, source="other")
    with pytest.raises(ValueError):
        PackedStratifiedSampler.restore(changed, sampler.to_dict(), budget)


def test_immutable_snapshot_and_strict_bins_schema():
    records, _ = fixture()
    budget = WorkBudget(100000)
    frame = PackedObservationFrame(records, budget, source="host")
    records[0][1]["f0"] = 7
    assert frame.values[0] == (0,)
    with pytest.raises(FrozenInstanceError):
        frame.source = "new"
    with pytest.raises(TypeError):
        frame.index["new"] = 1
    for row in ({"f0": True}, {"f0": 8}, {"f1": 0}, {"f0": 1.0}):
        with pytest.raises(ValueError):
            PackedObservationFrame([("a", row)], budget, source="host")
    for records in (iter([]), [], [("a", {"f0": 0})] * 129, [("a", {"f0": 0})] * 2):
        with pytest.raises(ValueError):
            PackedObservationFrame(records, budget, source="host")


def test_sparse_all_cuts_and_median_fallback():
    budget = WorkBudget(100000)
    records = [(f"p{i}", {"f0": 0 if i < 24 else 7, "f1": 0 if i < 24 else 7}) for i in range(32)]
    records += [(f"c{i}", {"f0": 7, "f1": 7}) for i in range(8)]
    pilot = tuple((f"p{i}", i >= 24) for i in range(32))
    frame = PackedObservationFrame(records, budget, source="host")
    all_cuts = PackedStratifiedSampler(frame, pilot, budget)
    assert all_cuts.features == ("f0", "f1")
    assert all_cuts.thresholds == (1, 1)
    median = PackedStratifiedSampler(frame, pilot, budget, cut_mode="median")
    assert median.features == ()
    for i in range(8):
        assert median.choose(budget) == f"c{i}"
        median.observe(f"c{i}", False, budget)
    assert median.choose(budget) is None


def test_costs_and_sink_failure_propagate():
    class Sink:
        def __init__(self):
            self.calls = []
        def charge(self, kind, units=1):
            self.calls.append((kind, units))
    sink = Sink()
    PackedObservationFrame([("a", {"f0": 1})], sink, source="host")
    assert sink.calls == [("packed_frame_meta", 1), ("packed_frame_meta", 2),
                          ("packed_frame_schema", 1), ("packed_frame_fields", 2)]
    failure = RuntimeError("sink")
    class Broken:
        def charge(self, *args):
            raise failure
    with pytest.raises(RuntimeError) as exc:
        PackedObservationFrame([], Broken(), source="host")
    assert exc.value is failure
    with pytest.raises(BudgetStop):
        PackedObservationFrame([("a", {"f0": 1})], WorkBudget(0), source="host")


def test_bounds_one_class_and_readonly_configuration():
    frame, _, budget = make()
    pilot = fixture()[1]
    for kwargs in ({"max_steps": True}, {"max_steps": 17}, {"max_steps": 0},
                   {"strategy": "unknown"}, {"cut_mode": "unknown"}):
        with pytest.raises(ValueError):
            PackedStratifiedSampler(frame, pilot, budget, **kwargs)
    for bad in (iter(pilot), [], [pilot[0]] * 9, [("absent", True)], [("p0", 1)]):
        with pytest.raises(ValueError):
            PackedStratifiedSampler(frame, bad, budget)
    sampler = PackedStratifiedSampler(frame, tuple((name, False) for name, _ in pilot), budget)
    assert sampler.reason == "random_prefix"
    assert sampler.choose(budget) == "c0"
    with pytest.raises(AttributeError):
        sampler.features = ("f99",)
    with pytest.raises(AttributeError):
        sampler.strategy = "adaptive"


def test_first_selected_feature_is_low_bit_and_exact_sink_on_choose_observe():
    budget = WorkBudget(100000)
    records = [(f"p{i}", {"f0": 0 if i < 4 else 7, "f1": 0 if i < 4 else 7}) for i in range(8)]
    records += [("both", {"f0": 7, "f1": 7}), ("second", {"f0": 0, "f1": 7}),
                ("first", {"f0": 7, "f1": 0}), ("neither", {"f0": 0, "f1": 0})]
    frame = PackedObservationFrame(records, budget, source="host")
    sampler = PackedStratifiedSampler(frame, tuple((f"p{i}", i >= 4) for i in range(8)), budget, max_steps=4)
    assert sampler.queues == (("neither",), ("first",), ("second",), ("both",))
    error = RuntimeError("sink failure")
    class Broken:
        def charge(self, *args):
            raise error
    with pytest.raises(RuntimeError) as exc:
        sampler.choose(Broken())
    assert exc.value is error and sampler.pending is None
    name = sampler.choose(budget)
    with pytest.raises(RuntimeError) as exc:
        sampler.observe(name, True, Broken())
    assert exc.value is error and sampler.pending == name and not sampler.received


def test_public_state_is_readonly_and_snapshots_are_immutable():
    frame, sampler, budget = make()
    sampler.choose(budget)
    before = sampler.to_dict()
    for field, value in (("frame", frame), ("pilot", ()), ("cursor", 99),
                         ("pending", "foreign"), ("received", ()), ("queues", ()),
                         ("counts", ()), ("positives", ()), ("positions", ())):
        with pytest.raises(AttributeError):
            setattr(sampler, field, value)
    for snapshot in (sampler.pilot, sampler.received, sampler.queues,
                     sampler.counts, sampler.positives, sampler.positions):
        assert type(snapshot) is tuple
    with pytest.raises(TypeError):
        sampler.queues[0][0] = "foreign"
    assert sampler.to_dict() == before
    with pytest.raises(ValueError):
        sampler.observe("foreign", True, budget)
    sampler.observe(sampler.pending, True, budget)
    assert len(sampler.received) == 1


def test_observe_verifies_queue_even_if_private_state_is_corrupted():
    _, sampler, budget = make()
    sampler.choose(budget)
    sampler._pending = "foreign"
    with pytest.raises(ValueError, match="queue mismatch"):
        sampler.observe("foreign", True, budget)
    assert sampler.received == ()


def test_restore_rejects_missing_pending_and_unknown_fields():
    frame, sampler, budget = make()
    sampler.choose(budget)
    saved = sampler.to_dict()
    del saved['pending']
    with pytest.raises(ValueError):
        PackedStratifiedSampler.restore(frame, saved, budget)
    saved = sampler.to_dict()
    saved['unrecognized'] = 'stage'
    with pytest.raises(ValueError):
        PackedStratifiedSampler.restore(frame, saved, budget)
