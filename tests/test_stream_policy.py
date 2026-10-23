"""solvi.experimental.stream, the adapter loop's stage 1: the policy (LoopPolicy, LoopState, Budget, DefaultPolicy)
and the gate arithmetic (stream.gates), against hand-built states — no loop, no model, no torch."""
import dataclasses
import math
import pickle
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import (Budget, DefaultPolicy, LabelRow, LoopPolicy, LoopState, Paired,
                                           TrainReport, VersionInfo)
    from solvi.experimental.stream import gates

DAY = 86_400.0


def _state(**kw):
    """A state with version 0 served (n_train 0) and nothing else unless given."""
    base = dict(n=1000, labels={"train": 0, "calib": 0, "holdout": 0}, now=10 * DAY)
    base.update(kw)
    return LoopState(**base)


def _ready(policy=None, **kw):
    """A state in which DefaultPolicy wants a training: 300 train labels over version 0, which answers nothing."""
    kw.setdefault("labels", {"train": 300, "calib": 64, "holdout": 64})
    return _state(**kw)


def _cand(certified=True, samples=500, **kw):
    """A state with version 1 frozen as the candidate."""
    vs = (VersionInfo(0, "served"), VersionInfo(1, "candidate", n_train=300, samples=samples, certified=certified,
                                                threshold=0.9 if certified else math.inf, frozen_at=400))
    return _state(versions=vs, candidate=1, **kw)


# ───────────────────────────────────────────────────────────── the split
def test_the_split_is_stable_has_its_shares_and_never_crosses():
    keys = [f"intent|{i:06d}" for i in range(20_000)]
    first = [gates.split(k) for k in keys]
    assert first == [gates.split(k) for k in reversed(keys)][::-1]              # order does not matter
    assert first == [gates.split(k) for k in keys]                               # nor the call
    share = {s: first.count(s) / len(keys) for s in gates.SPLITS}
    assert abs(share["train"] - 0.70) < 0.015 and abs(share["calib"] - 0.15) < 0.01
    assert abs(share["holdout"] - 0.15) < 0.01
    p = DefaultPolicy()
    st = _state()
    seen: dict = {}                                       # a stream with repeats: a key keeps its first split
    r = random.Random(0)
    for _ in range(30_000):
        k = r.choice(keys[:2000])
        s = p.split(st, k)
        assert seen.setdefault(k, s) == s
    # the policy pickled and loaded splits as before
    q = pickle.loads(pickle.dumps(p))
    assert all(q.split(st, k) == seen[k] for k in seen)


def test_the_split_refuses_shares_that_leave_nothing_to_train():
    with pytest.raises(ValueError):
        gates.split("k", calib=0.5, holdout=0.5)
    with pytest.raises(ValueError):
        DefaultPolicy(calib=-0.1)
    assert gates.split("k", calib=0.0, holdout=0.0) == "train"


# ───────────────────────────────────────────────────────────── accept
def test_accept_takes_trusted_sources_and_slow_path_labels_only_on_opt_in():
    st = _state()
    p, p2 = DefaultPolicy(), DefaultPolicy(s2_labels=True)
    for src in ("human", "outcome", "rule", "audit"):
        assert p.accept(st, "x", "a", src) and p2.accept(st, "x", "a", src)
    for src in ("s2", "verified"):
        assert not p.accept(st, "x", "a", src) and p2.accept(st, "x", "a", src)
    assert not p2.accept(st, "x", "a", "model") and not p.accept(st, "x", "a", "system")


# ───────────────────────────────────────────────────────────── want_train
def test_want_train_waits_for_first_then_for_growth_over_the_served_version():
    p = DefaultPolicy()
    assert not p.want_train(_ready(labels={"train": 299}))
    assert "299 train label(s) < 300" in p.why_not_train(_ready(labels={"train": 299}))
    assert p.want_train(_ready())
    vs = (VersionInfo(0, "retired"), VersionInfo(1, "served", n_train=400))
    weak = tuple((True, True) for _ in range(50))            # served errs alone: short
    assert not p.want_train(_ready(versions=vs, served=1, labels={"train": 599}, holdout=weak))
    assert "growth 1.5 × the served version's 400" in p.why_not_train(
        _ready(versions=vs, served=1, labels={"train": 599}, holdout=weak))
    assert p.want_train(_ready(versions=vs, served=1, labels={"train": 600}, holdout=weak))


def test_want_train_needs_the_improvement_signal():
    p = DefaultPolicy()
    vs = (VersionInfo(0, "retired"), VersionInfo(1, "served", n_train=300))
    labels = {"train": 1000}
    strong = tuple([(True, False)] * 190 + [(False, False)] * 10)            # 95 % alone, 0 errors
    st = _ready(versions=vs, served=1, labels=labels, holdout=strong)
    assert not p.want_train(st) and p.why_not_train(st).startswith("no improvement signal")
    # error alone above α (0.05): 12 wrong of 190 alone
    erring = tuple([(True, True)] * 12 + [(True, False)] * 178 + [(False, False)] * 10)
    assert p.want_train(_ready(versions=vs, served=1, labels=labels, holdout=erring))
    # only the last `window` held-out labels count: old errors scroll out
    assert not p.want_train(_ready(versions=vs, served=1, labels=labels, holdout=erring + strong))
    # a served version answering nothing alone is short; so is one with no held-out label yet
    assert p.want_train(_ready(versions=vs, served=1, labels=labels, holdout=((False, False),) * 200))
    assert p.want_train(_ready(versions=vs, served=1, labels=labels, holdout=()))
    # target_share: a strong but timid incumbent is short of it
    timid = tuple([(True, False)] * 40 + [(False, False)] * 160)
    assert not p.want_train(_ready(versions=vs, served=1, labels=labels, holdout=timid))
    assert DefaultPolicy(target_share=0.5).want_train(_ready(versions=vs, served=1, labels=labels, holdout=timid))
    # the promised level is the state's
    assert not p.want_train(_ready(versions=vs, served=1, labels=labels, holdout=erring, alpha=0.10))


def test_want_train_is_off_while_paused_or_with_a_candidate_in_trial():
    p = DefaultPolicy()
    assert p.why_not_train(_ready(paused=True)) == "paused"
    st = _cand(labels={"train": 5000})
    assert not p.want_train(st) and "in trial" in p.why_not_train(st)


def test_budget_refusals():
    b = Budget(max_trainings=3, per_day=2, min_interval_s=600, max_train_s=1800)
    now = 10 * DAY
    assert b.refuse([], now) is None
    assert "min_interval_s" in b.refuse([now - 100], now)
    assert b.refuse([now - 700], now) is None
    assert "last 24 h" in b.refuse([now - 5000, now - 4000], now)
    assert b.refuse([now - DAY - 1, now - 4000], now) is None                  # one of them is older than a day
    assert "max_trainings 3" in b.refuse([now - 3 * DAY, now - 2 * DAY, now - 1.5 * DAY], now)
    assert "max_train_s" in b.refuse([], now, estimate_s=1801) and b.refuse([], now, estimate_s=1800) is None
    p = DefaultPolicy(budget=Budget(per_day=1))
    st = _ready(trainings=(10 * DAY - 3600,))
    assert not p.want_train(st) and "per_day 1" in p.why_not_train(st)
    assert DefaultPolicy(budget=Budget(max_trainings=0)).why_not_train(_ready()).startswith("budget")


def test_budget_checks_its_fields():
    for bad in ({"per_day": -1}, {"cpu_threads": 0}, {"max_train_s": -1.0}, {"max_train_s": math.inf},
                {"device": "gpu"}, {"device": "cuda:x"}, {"max_examples": 0}, {"max_trainings": 1.5},
                {"min_interval_s": True}):
        with pytest.raises(ValueError):
            Budget(**bad)
    assert Budget(device="cuda:1").device == "cuda:1"
    with pytest.raises(TypeError):
        DefaultPolicy(budget={"per_day": 3})


# ───────────────────────────────────────────────────────────── select / hyper
def _rows(n, labels="abc", start=0):
    return [LabelRow(f"k{i}", f"x{i}", labels[i % len(labels)], "human", start + i) for i in range(n)]


def test_select_keeps_everything_under_the_cap_and_balances_above_it():
    p = DefaultPolicy(budget=Budget(max_examples=60))
    st = _ready()
    rows = _rows(50)
    assert p.select(st, list(reversed(rows))) == rows                    # all, in decision order
    skewed = [LabelRow(f"k{i}", "x", "a" if i < 900 else "b", "human", i) for i in range(1000)]
    got = p.select(st, skewed)
    assert len(got) == 60 and sum(r.y == "b" for r in got) == 30           # 100 "b" of 1000 → half the selection
    assert got == p.select(st, skewed)                                    # deterministic
    assert [r.t for r in got] == sorted(r.t for r in got)


def test_select_takes_half_from_the_recent_rows():
    p = DefaultPolicy(budget=Budget(max_examples=100))
    vs = (VersionInfo(0, "retired"), VersionInfo(1, "served", n_train=50))
    st = _ready(versions=vs, served=1)
    rows = _rows(1000)
    got = p.select(st, rows)
    recent = {r.t for r in rows[-100:]}                                   # the last 2 × n_train
    assert len(got) == 100 and sum(r.t in recent for r in got) == 50
    # a different next version, a different sample of the same rows
    vs2 = vs + (VersionInfo(2, "discarded", n_train=60),)
    assert p.select(_ready(versions=vs2, served=1), rows) != got


def test_hyper_scales_updates_and_seeds_by_the_next_version():
    p = DefaultPolicy()
    h = p.hyper(_ready(labels={"train": 300}))
    assert h == {"r": 8, "lr": 3e-4, "updates": 200, "seed": 1, "warm_start": False}
    assert p.hyper(_ready(labels={"train": 4000}))["updates"] == 1000
    assert p.hyper(_ready(labels={"train": 50_000}))["updates"] == 3000
    assert DefaultPolicy(budget=Budget(max_examples=2000)).hyper(_ready(labels={"train": 50_000}))["updates"] == 500
    assert p.hyper(_cand())["seed"] == 2


# ───────────────────────────────────────────────────────────── freeze
def _report(**kw):
    base = dict(n_holdout=150, right=90, wrong=3, served_right=60, served_wrong=2, better=40, worse=10,
                threshold=0.8, honesty_before={"confident_error_rate": 0.05}, honesty_after={"confident_error_rate":
                                                                                               0.06}, seconds=300.0)
    base.update(kw)
    return TrainReport(**base)


def test_freeze_passes_only_with_every_gate():
    p, st = DefaultPolicy(), _ready()
    assert p.freeze(st, _report())
    assert "paired_gain" not in p.freeze_gates(st, _report())                 # stream mode: no Wilson gate
    g = p.freeze_gates(st, _report(n_holdout=99))
    assert not g["heldout"]["ok"] and "99 held-out" in g["heldout"]["why"]
    # gain: (90 − 3) / 150 − (88 − 2) / 150 = 1 / 150 < 0.01
    assert not p.freeze(st, _report(served_right=88))
    assert p.freeze(st, _report(served_right=88, right=91, wrong=2))         # 89 − 86 = 3 / 150 = 0.02
    assert not p.freeze(st, _report(threshold=None)) and not p.freeze(st, _report(threshold=math.inf))
    assert not p.freeze(st, _report(honesty_after={"confident_error_rate": 0.08}))   # 0.03 worse > 0.02
    assert not p.freeze(st, _report(seconds=1801.0))
    assert p.freeze(st, _report(honesty_before={}, honesty_after={}))


def test_label_mode_freezes_only_with_a_paired_gain_whose_wilson_bound_is_positive():
    p = DefaultPolicy()
    st = _ready(mode="label")
    assert p.freeze(st, _report(better=40, worse=10))
    g = p.freeze_gates(st, _report(better=12, worse=10))
    assert not g["paired_gain"]["ok"] and g["paired_gain"]["low"] < 0
    assert not p.freeze(st, _report(better=0, worse=0))
    assert gates.paired_gain_low(0, 0) is None
    assert gates.paired_gain_low(40, 10) > 0 > gates.paired_gain_low(12, 10)
    with pytest.raises(ValueError):
        LoopState(mode="batch")


# ───────────────────────────────────────────────────────────── promote / discard
GOOD = Paired(n_novel=200, alone_k=60, wrong_k=2, alone_s=20, n_seen=100, alone_seen=40, wrong_seen=1)


def test_promote_needs_all_four_conditions_and_the_probe():
    p = DefaultPolicy()
    st = _cand()
    assert p.promote(st, GOOD)
    assert not p.promote(_cand(certified=False), GOOD)                               # 1 certified
    assert not p.promote(st, Paired(99, 30, 1, 5, 100, 40, 1))                       # 2 n_novel ≥ 100
    assert not p.promote(st, Paired(200, 9, 0, 0, 100, 40, 0))                       # 3 share_k 0.045 < 0.05
    assert p.promote(st, Paired(200, 10, 0, 0, 100, 40, 0))                          #   0.05 passes
    assert not p.promote(st, Paired(200, 60, 2, 59, 100, 40, 1))                     # 4 0.30 < 0.295 + 0.01
    assert p.promote(st, Paired(200, 60, 2, 58, 100, 40, 1))                         #   0.30 ≥ 0.29 + 0.01
    memoriser = Paired(n_novel=200, alone_k=60, wrong_k=30, alone_s=20, n_seen=100, alone_seen=80, wrong_seen=0)
    assert not p.promote(st, memoriser)                                              # 5 the probe
    assert not p.promote_checks(st, memoriser)["probe"]["ok"]
    assert DefaultPolicy(gap=1.0).promote(st, memoriser)                              # probe off: promoted
    assert not p.promote(_state(), GOOD)                                              # no candidate


def test_the_probe_is_measured_only_with_enough_alone_answers_on_both_sides():
    assert gates.probe(80, 0, 60, 30)["ok"] is False
    few = gates.probe(29, 0, 60, 30)
    assert few["ok"] and not few["measured"]
    assert not gates.probe(80, 2, 60, 9)["ok"]                # 0.15 − 0.025 = 0.125 > 0.10
    assert gates.probe(80, 4, 60, 6)["ok"]                    # 0.10 − 0.05 = 0.05


def test_label_mode_promotes_on_the_pre_freeze_gates():
    p = DefaultPolicy()
    assert p.promote(_cand(mode="label"), None)
    assert not p.promote(_cand(), None)                       # stream mode with no paired samples: no
    assert p.discard(_cand(mode="label"), None) is None


def test_discard_reasons():
    p = DefaultPolicy()
    assert p.discard(_cand(certified=False, samples=1499), GOOD) is None
    assert p.discard(_cand(certified=False, samples=1500), GOOD) == "not certified after 1500 samples"
    flat = Paired(200, 20, 0, 20, 100, 40, 0)                 # certified, not better
    assert p.discard(_cand(samples=1499), flat) is None
    assert p.discard(_cand(samples=1500), flat).startswith("certified, not better after 1500")
    assert p.discard(_cand(samples=1500), GOOD) is None        # promotable: the loop promotes, not discards
    memoriser = Paired(200, 60, 30, 20, 100, 80, 0)
    assert p.discard(_cand(samples=10), memoriser).startswith("memorisation probe")


# ───────────────────────────────────────────────────────────── alarm
def test_alarm_recertifies_and_rolls_back_on_a_second_alarm_within_probation():
    p = DefaultPolicy()
    vs = (VersionInfo(0, "retired"), VersionInfo(1, "retired", n_train=300), VersionInfo(2, "served", n_train=450))
    st = _state(n=2000, versions=vs, served=2, served_history=(0, 1, 2))
    assert p.alarm(st, "detector", {}) == "recertify"
    once = ({"n": 1800, "who": "monitor", "version": 2},)
    assert p.alarm(_state(n=2000, versions=vs, served=2, served_history=(0, 1, 2), alarms=once),
                   "detector", {}) == "rollback:1"
    assert p.alarm(_state(n=2101, versions=vs, served=2, served_history=(0, 1, 2), alarms=once),
                   "detector", {}) == "recertify"                                    # outside probation
    other = ({"n": 1900, "who": "detector", "version": 1},)
    assert p.alarm(_state(n=2000, versions=vs, served=2, served_history=(0, 1, 2), alarms=other),
                   "monitor", {}) == "recertify"                                     # an alarm on another version
    assert p.alarm(_state(n=200, alarms=({"n": 100, "who": "detector", "version": 0},)), "detector",
                   {}) == "recertify"                                                # nothing to roll back to


# ───────────────────────────────────────────────────────────── the abstract class, pickling, the state
class Counting(DefaultPolicy):
    """A developer's policy: the defaults with its own counter in self (pickled with it)."""

    def __init__(self):
        super().__init__(first=10)
        self.asked = 0

    def want_train(self, st):
        self.asked += 1
        return super().want_train(st)


def test_loop_policy_is_abstract_and_a_subclass_gives_every_hook():
    with pytest.raises(TypeError):
        LoopPolicy()

    class Half(LoopPolicy):
        def accept(self, st, x, y, source):
            return True

    with pytest.raises(TypeError):
        Half()

    e = Counting()
    assert e.want_train(_ready(labels={"train": 10})) and e.asked == 1
    f = pickle.loads(pickle.dumps(e))
    assert f.asked == 1 and f.first == 10 and isinstance(f.budget, Budget)


def test_policies_and_states_pickle():
    p = DefaultPolicy(budget=Budget(per_day=3, device="cpu"), s2_labels=True, target_share=0.4, gap=0.2)
    q = pickle.loads(pickle.dumps(p))
    assert vars(q) == vars(p) and q.budget == p.budget
    st = _cand(holdout=((True, False),), alarms=({"n": 3, "who": "detector", "version": 0},))
    assert pickle.loads(pickle.dumps(st)) == st
    assert pickle.loads(pickle.dumps(GOOD)) == GOOD and pickle.loads(pickle.dumps(_report())) == _report()
    for hook in ("want_train", "hyper"):
        assert getattr(q, hook)(_ready()) == getattr(p, hook)(_ready())


def test_the_state_is_read_only_and_finds_its_versions():
    st = _cand()
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.n = 5
    assert st.served_info.version == 0 and st.candidate_info.version == 1 and st.next_version == 2
    assert _state().candidate_info is None
    with pytest.raises(KeyError):
        st.version(7)


def test_policy_settings_are_checked():
    for bad in ({"first": 0}, {"growth": 0.5}, {"min_share": 1.5}, {"wilson": 1.0}, {"lr": 0.0},
                {"min_updates": 500, "max_updates": 400}, {"probation": -1}, {"target_share": 2}):
        with pytest.raises(ValueError):
            DefaultPolicy(**bad)
