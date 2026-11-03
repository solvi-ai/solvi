import contextlib

from solvi.core.deciders import DecideModel


class Provider:
    generation = 1
    calls = 0
    held = False

    @contextlib.contextmanager
    def lease(self):
        self.held = True
        try:
            yield self.generation
        finally:
            self.held = False

    def logits(self, items):
        assert self.held
        self.calls += 1
        return [[2. if option == "yes" else -2. for option in it.options] for it in items]


def ask(model, text="x"):
    return model.decide(text, "Q", ["yes", "no"])


def test_lease_pins_scoring_and_cached_requests():
    p = Provider()
    m = DecideModel(p, identity_safe=True, identity_lease=p.lease)
    assert ask(m).value == "yes"
    assert ask(m).value == "yes"
    assert p.calls == 1 and not p.held


def test_swap_clears_cache_and_abstains_without_scoring():
    p = Provider()
    m = DecideModel(p, identity_safe=True, identity_lease=p.lease)
    ask(m)
    p.generation = 2
    d = ask(m)
    assert d.value is None and d.escalate and not m._cache
    assert m.identity_guard.stale and p.calls == 1


def test_unavailable_batch_preserves_shape():
    m = DecideModel(Provider(), identity_safe=True)
    ds = ask(m, ["x", "y"])
    assert len(ds) == 2 and all(d.value is None and d.escalate for d in ds)
    assert m.identity_guard.state == "unknown"


def test_exit_exception_replaces_already_computed_answer():
    p = Provider()
    @contextlib.contextmanager
    def broken():
        with p.lease() as token:
            yield token
            raise RuntimeError("lease lost")
    m = DecideModel(p, identity_safe=True, identity_lease=broken)
    d = ask(m)
    assert d.value is None and d.escalate and not m._cache


def test_generation_rollback_never_restores_confirmed_answer():
    p = Provider()
    m = DecideModel(p, identity_safe=True, identity_lease=p.lease)
    ask(m)
    p.generation = 0
    assert ask(m).value is None
    p.generation = 1
    assert ask(m).value is None


def test_legacy_without_optin_unchanged():
    p = Provider()
    p.held = True
    assert ask(DecideModel(p)).value == "yes"


def part_model(p):
    m = DecideModel(p, identity_safe=True, identity_lease=p.lease)
    return m, m.decision("answer", "Q", "text", ["yes", "no"])


def test_direct_part_and_batch_hold_lease():
    m, part = part_model(Provider())
    assert part.decide("x").value == "yes"
    assert part(text="x").value == "yes"
    assert all(d.value == "yes" for d in m.decide_pass("x", [part]))


def test_unbound_thresholds_and_promises_abstain():
    m, part = part_model(Provider())
    part.escalate_below = 0.0
    part.guarantee = {"promise": "old"}
    part.conformal_set = {"coverage": .9, "q": .1}
    # Gate checks bindings even if a shortcut supplied a plausible decision.
    from solvi.core.deciders._identity import part_gate
    from solvi.core.catalog import Decision
    with m.identity_guard.operation():
        d = part_gate(part, Decision("yes", {"yes": 1.}, extra={"guarantee": part.guarantee}))
    assert d.value is None and "guarantee" not in d.extra


def test_successful_calibration_recovers_new_generation():
    p = Provider()
    m, part = part_model(p)
    part.calibrate_for([("x", "yes"), ("y", "yes")], max_error=0.)
    assert part.decide("x").value == "yes"
    p.generation = 2
    assert part.decide("x").value is None
    part.calibrate_for([("x", "yes"), ("y", "yes")], max_error=0.)
    assert part.decide("x").value == "yes" and not m.identity_guard.stale


def test_failed_calibration_remains_stale():
    m, part = part_model(Provider())
    import pytest
    with pytest.raises(ValueError):
        part.calibrate_for([], max_error=-1.)
    assert m.identity_guard.stale and part.decide("x").value is None


def test_load_even_strict_false_requires_fresh_calibration(tmp_path):
    m, part = part_model(Provider())
    part.calibrate_for([("x", "yes"), ("y", "yes")], max_error=0.)
    path = tmp_path / "cal.json"
    part.save_calibration(path)
    part.load_calibration(path, strict=False)
    assert part.decide("x").value is None


def test_calibration_does_not_revalidate_other_parts():
    p = Provider()
    m, first = part_model(p)
    second = m.decision("second", "Q", "text", ["yes", "no"])
    rows = [("x", "yes"), ("y", "yes")]
    first.calibrate_for(rows, max_error=0.)
    second.calibrate_for(rows, max_error=0.)
    p.generation = 2
    assert first.decide("x").value is None
    second.calibrate_for(rows, max_error=0.)
    assert first.decide("x").value is None


def test_inflight_answer_is_held_in_generation_until_finish():
    import threading
    started, release = threading.Event(), threading.Event()
    class Pinned(Provider):
        def __init__(self):
            self.lock = threading.Lock()
        @contextlib.contextmanager
        def lease(self):
            with self.lock:
                with super().lease() as token:
                    yield token
        def logits(self, items):
            started.set()
            assert release.wait(2)
            return super().logits(items)
    p = Pinned()
    m, part = part_model(p)
    results = []
    worker = threading.Thread(target=lambda: results.append(part.decide("x")))
    worker.start()
    assert started.wait(2)
    assert not p.lock.acquire(blocking=False)
    second = threading.Thread(target=lambda: results.append(part.decide("x")))
    second.start()
    release.set()
    worker.join(2)
    second.join(2)
    assert not worker.is_alive() and not second.is_alive()
    assert len(results) == 2 and all(d.value == "yes" for d in results) and p.calls == 1
    with p.lock:
        p.generation = 2
    assert part.decide("x").value is None and not m._cache


def test_calibration_exit_exception_never_commits_binding():
    p = Provider()
    @contextlib.contextmanager
    def broken():
        with p.lease() as token:
            yield token
            raise RuntimeError("lost lease")
    m = DecideModel(p, identity_safe=True, identity_lease=broken)
    part = m.decision("answer", "Q", "text", ["yes", "no"])
    import pytest
    from solvi.core.deciders._identity import IdentityUnavailable
    with pytest.raises(IdentityUnavailable):
        part.calibrate_for([("x", "yes"), ("y", "yes")], max_error=0.)
    assert part._identity_binding is None and m.identity_guard.stale


def test_private_in_pass_and_unavailable_exit():
    p = Provider()
    m, part = part_model(p)
    assert part._in_pass([part], {"text": "x"}).value == "yes"
    p.generation = None
    assert part._in_pass([part], {"text": "x"}).value is None


def test_nested_combinations_fail_closed_before_unleased_leaf_scoring():
    from solvi.core.deciders.combine import Cascade, Vote
    p = Provider()
    m, part = part_model(p)
    other = m.decision("other", "Q", "text", ["yes", "no"])
    combo = Vote([Cascade([part, other]), part])
    combo.guarantee = {"promise": "stale"}
    d = combo.decide("x")
    assert d.value is None and "guarantee" not in d.extra and p.calls == 0


def test_failed_calibration_restores_old_settings_but_not_binding():
    import pytest
    m, part = part_model(Provider())
    part.escalate_below = .75
    part.guarantee = {"promise": "old"}
    with pytest.raises(ValueError):
        part.calibrate_for([], max_error=-1.)
    assert part.escalate_below == .75 and part.guarantee == {"promise": "old"}
    assert part._identity_binding is None and part.decide("x").value is None


def test_preloaded_checkpoint_calibration_never_silently_binds():
    p = Provider()
    m = DecideModel(p, meta={"temperature": 2.}, identity_safe=True, identity_lease=p.lease)
    assert ask(m).value is None and p.calls == 0


def test_preloaded_adaptation_is_discarded_and_requires_new_calibration():
    p = Provider()
    m, part = part_model(p)
    # Simulate state restored by legacy code before opt-in construction/serving.
    from solvi.core.deciders.adapt import Adaptation
    m.adaptations[("old",)] = Adaptation()
    assert part.decide("x").value is None and not m.adaptations
    part.calibrate_for([("x", "yes"), ("y", "yes")], max_error=0.)
    assert part.decide("x").value == "yes"


def test_public_mutations_reject_before_scoring_or_consuming_inputs():
    import pytest
    p = Provider()
    m, part = part_model(p)
    rows = [("x", "yes"), ("y", "yes")]
    part.calibrate_for(rows, max_error=0.)
    binding, threshold, calls = part._identity_binding, part.escalate_below, p.calls
    def unread():
        raise AssertionError("unsupported mutation consumed inputs")
        yield "x"
    operations = [lambda: part.adapt(unread()), lambda: part.fit(unread()), lambda: part.teach("x", "yes"),
                  part.reset, lambda: m.adapt(unread(), "Q", ["yes", "no"]),
                  lambda: m.fit(unread(), "Q", ["yes", "no"]),
                  lambda: m.teach("x", "yes", "Q", ["yes", "no"]), m.reset,
                  lambda: m.load_adaptations("missing.json", strict=False),
                  lambda: m.adaptation("Q", ["yes", "no"], create=True),
                  lambda: part.load_lora("missing.safetensors")]
    for operation in operations:
        with pytest.raises(ValueError, match="unsupported"):
            operation()
        assert p.calls == calls and part._identity_binding == binding and part.escalate_below == threshold
    assert m.adaptation("Q", ["yes", "no"], create=False) is None


def test_combination_lowlevel_and_calibration_reject_before_scoring():
    import pytest
    from solvi.core.deciders.combine import Cascade, Vote
    p = Provider()
    m, part = part_model(p)
    for combo in (Cascade([part]), Vote([part])):
        operations = [lambda: combo.state(None), lambda: combo.final(None, None),
                      lambda: combo.calibrate_for([]), lambda: combo.act_guard([]),
                      lambda: combo.conformal([]), lambda: combo.fit([]), lambda: combo.adapt([]), combo.reset,
                      lambda: combo.load_calibration("missing.json", strict=False)]
        for operation in operations:
            with pytest.raises(ValueError, match="unsupported"):
                operation()
        assert combo.decide("x").value is None and p.calls == 0
