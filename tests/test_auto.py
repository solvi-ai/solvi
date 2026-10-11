"""solvi.solutions.decisions.build: one entry point that fits System 1 from examples, calibrates its guarantee (or an open-set gate)
and the dispatcher, wires the store, and says what it chose."""
import hashlib
import random

import numpy as np
import pytest

from solvi import Answer, Catalog, Question, System
from solvi.solutions.decisions import build
from solvi.core.dispatch import AskPath, Budget, SlowPath


def _pairs(n, seed=0):
    """A yes/no task whose answer follows a number: x > 0.5 → yes, labels flipped more often near 0.5."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.random()
        y = (x > 0.5) != (rng.random() < max(0.0, 0.3 - abs(x - 0.5)))
        out.append(({"x": round(x, 4)}, "yes" if y else "no"))
    return out


def _catalog():
    cat = Catalog()

    @cat.fn
    def margin(x) -> float:
        return abs(x - 0.5)

    @cat.fn
    def high(x) -> bool:
        return x > 0.5

    return cat


Q = Question("ok", "Is it ok?", Answer.yes_no())


def test_a_head_is_fitted_and_its_promise_calibrated_on_examples_it_did_not_see(tmp_path):
    ex = _pairs(800)
    s = build(Q, ex, catalog=_catalog(), max_risk=0.05, storage=tmp_path / "d.jsonl")
    assert s.choices["mode"] == "head"
    assert s.choices["split"]["to fit System 1"] == 600 and s.choices["split"]["to calibrate System 1's guarantee"] == 200
    assert "high" in s.system.heads["ok"].features
    out = [s.ask(st) for st, _ in _pairs(300, seed=1)]
    assert {r.by for r in out} <= {"s1", "human"}
    alone = [r for r in out if r.by == "s1"]
    assert 0.3 < len(alone) / len(out) < 1.0
    text = s.explain()
    assert "ridge head" in text and "P(answered alone and wrong) ≤ 0.05" in text and "No slow path" in text
    assert not s.replay_all()
    assert s.storage.verify()["ok"]
    rep = s.report()
    assert "ok" in str(rep)


def test_a_rule_with_a_constant_confidence_gets_the_fact_that_separates_right_from_wrong():
    cat = Catalog()

    @cat.fn
    def margin(x) -> float:
        return abs(x - 0.5)

    @cat.fn
    def noise(x) -> float:
        return float(int(hashlib.md5(str(x).encode()).hexdigest()[:6], 16) % 1000)

    @cat.rule("ok")
    def ok(x):
        return "yes" if x > 0.5 else "no"

    # labels flip near the boundary: the rule is wrong where the margin is small
    rng = random.Random(3)
    ex = []
    for _ in range(900):
        x = rng.random()
        flip = rng.random() < max(0.0, 0.4 - abs(x - 0.5) * 4)
        ex.append(({"x": round(x, 4)}, "yes" if (x > 0.5) != flip else "no"))
    s = build(Q, ex, catalog=cat, max_risk=0.03)
    assert s.choices["mode"] == "given"
    assert s.system.guards["ok"].signal == "margin"
    assert s.choices["split"]["to choose the signal"] == 300
    assert "margin" in s.explain() and "AUROC" in s.explain()


def test_a_label_that_judges_system_1_is_read_through_correct():
    cat = Catalog()

    @cat.fn
    def margin(x) -> float:
        return abs(x - 0.5)

    @cat.rule("ok")
    def ok(x):
        return "yes" if x > 0.5 else "no"

    rng = random.Random(4)
    ex = []
    for _ in range(600):
        x = rng.random()
        ex.append(({"x": round(x, 4)}, rng.random() > max(0.0, 0.4 - abs(x - 0.5) * 4)))   # label: is the rule right?
    s = build(Q, ex, catalog=cat, max_risk=0.05, correct=lambda answer, label: label)
    assert s.system.guards["ok"].signal == "margin"


def test_a_slow_path_gets_the_slice_only_where_calibration_says_so():
    ex = _pairs(1200)

    def oracle(state):                   # a slow path that reads the number exactly (right ~92% of the time)
        return "yes" if state["x"] > 0.5 else "no"

    s = build(Q, ex, catalog=_catalog(), max_risk=0.05, slow=oracle, reads="x", total=Budget(calls=10_000), min_slice=5)
    sp = s.choices["split"]
    assert sp["to calibrate System 1's guarantee"] == sp["to calibrate the dispatcher"] == 150
    assert "dispatch" in s.calibration and s.dispatcher.policy is not None
    assert "probe on the dispatcher's 150 examples" in s.explain()
    assert "slice" in s.explain()
    out = [s.ask(st) for st, _ in _pairs(200, seed=5)]
    assert {r.by for r in out} <= {"s1", "s2", "human"}
    assert all(s.replay(r)["ok"] for r in out[:20])


def test_a_system_or_a_slow_path_is_taken_as_it_is():
    cat2 = Catalog()

    @cat2.rule("ok")
    def ok2(x):
        return "yes" if x > 0.5 else "no"

    slow = System(cat2, [Q])
    s = build(Q, _pairs(600), catalog=_catalog(), max_risk=0.05, slow=slow)
    assert isinstance(s.slow, SlowPath) and s.slow.system is slow
    s2 = build(Q, _pairs(600), catalog=_catalog(), max_risk=0.05, slow=AskPath(slow))
    assert s2.slow.system is slow


def test_the_arguments_are_checked():
    with pytest.raises(ValueError, match="max_risk"):
        build(Q, _pairs(100), catalog=_catalog())
    with pytest.raises(ValueError, match="max_risk"):
        build(Q, _pairs(100), catalog=_catalog(), max_risk=0.1, max_error=0.1)
    with pytest.raises(ValueError, match="state dict"):
        build(Q, [("x", "yes")], max_risk=0.1)
    with pytest.raises(ValueError, match="no rule answers"):
        build("ok", _pairs(100), max_risk=0.1)
    with pytest.raises(ValueError, match="novel=True"):
        build(Q, _pairs(200), catalog=_catalog(), max_risk=0.1, novel=True)
    with pytest.raises(TypeError, match="slow="):
        build(Q, _pairs(200), catalog=_catalog(), max_risk=0.1, slow=42)


# ------------------------------------------------------------------------------------------------ a learner, new options
WORDS = [f"w{i}" for i in range(9)]
FILLER = [f"z{i}" for i in range(20)]


class _Scorer:
    """A bag-of-words classifier behind solvi's scorer protocol: the score of an option is how often its word occurs;
    act: the margin between the two best."""

    def __init__(self, options):
        self.options = list(options)
        self.model_id = "toy-bow"

    def fingerprint(self):
        return "toy-bow-" + ",".join(self.options)

    def logits(self, items):
        out = []
        for it in items:
            toks = it.text.split()
            c = np.array([toks.count(o) for o in it.options], float) + 0.1
            lp = np.log(c / c.sum())
            s = np.sort(c)[::-1]
            out.append({"logits": lp, "act": float((s[0] - s[1]) / max(1.0, s.sum()) * 6 - 1)})
        return out


def _learner(ex):
    from solvi.core.deciders import DecideModel
    opts = sorted({y for _, y in ex})
    model = DecideModel(_Scorer(opts), meta={"format": "stand-in", "temperature": 1.0, "act": {}})
    return model.decision("topic", "Which word?", "text", opts, min_act=0.0)


def _texts(n, words, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        w = rng.choice(words)
        k = rng.randint(1, 4)
        toks = [w] * k + [rng.choice(WORDS + FILLER) for _ in range(rng.randint(0, 2))]
        rng.shuffle(toks)
        out.append(({"text": " ".join(toks)}, w))
    return out


def test_a_learner_with_many_options_gets_an_open_set_gate():
    ex = _texts(1600, WORDS)
    s = build("topic", ex, learner=_learner, max_error=0.10)
    assert s.choices["mode"] == "learner" and s.gate is not None
    assert s.system.guards["topic"].signal == "act"
    assert "New kinds of input" in s.explain() and "refitted 3 times" in s.explain()
    s.gate.reset()
    out = [s.ask(st) for st, _ in _texts(100, WORDS, seed=9)]
    assert any(r.by == "s1" for r in out)
    off = build("topic", ex, learner=_learner, max_error=0.10, novel=False)
    assert off.gate is None and "guarantee" in off.calibration


def test_a_slow_path_with_no_slice_open_gives_its_share_back_to_system_1():
    """Every input System 1 holds back is on the open-set slice, which goes to a person: the dispatcher's share of the
    examples calibrates the gate instead, and the slow path only checks after a drift flag."""
    ex = _texts(1600, WORDS)
    s = build("topic", ex, learner=_learner, max_error=0.10, slow=lambda state: "w0", reads="text")
    sp = s.choices["split"]
    assert sp["to calibrate the dispatcher"] == 0 and sp["to calibrate System 1's guarantee"] == 400
    assert s.dispatcher.policy is None and set(s.dispatcher.wake) == {"drift", "supervise"}
    assert "No slice is open to the slow path" in s.explain() and "calibrated System 1 instead" in s.explain()
    s.gate.reset()
    out = [s.ask(st) for st, _ in _texts(50, WORDS, seed=11)]
    assert {r.by for r in out} <= {"s1", "human"}


def test_the_example_runs_and_hands_the_notes_slice_to_the_slow_path(capsys):
    from examples_loader import load
    load("24_one_entry_point")
    out = capsys.readouterr().out
    assert "Is a slice open to it? The probe" in out and "replay failures: 0" in out and "'s2': 0" not in out


# ------------------------------------------------------------------------------------------------ groups: a split by source
class _ScreenScorer:
    """Remembers the label of every screen it was fitted on (right 95% there); on a screen it never saw it goes by a
    hint whose strength k (1–5) is also its act signal. The act signal does not know whether the screen was seen."""

    def __init__(self, memory):
        self.memory, self.model_id = memory, "toy-screens"

    def fingerprint(self):
        return "toy-screens-" + str(sorted(self.memory.items()))

    def logits(self, items):
        out = []
        for it in items:
            screen, hint, k = it.text.split()
            k = int(k[1:])
            pick = self.memory.get(screen, hint[1:])
            z = np.array([2.0, -2.0]) if pick == it.options[0] else np.array([-2.0, 2.0])
            out.append({"logits": z, "act": (k - 3) * 1.5})
        return out


def _screen_learner(ex):
    from collections import Counter

    from solvi.core.deciders import DecideModel
    seen = {}
    for st, y in ex:
        seen.setdefault(st["text"].split()[0], Counter())[y] += 1
    memory = {s: c.most_common(1)[0][0] for s, c in seen.items()}
    model = DecideModel(_ScreenScorer(memory), meta={"format": "stand-in", "temperature": 1.0, "act": {}})
    return model.decision("close", "How to close it?", "text", ["a", "b"], min_act=0.0)


def _screens(n_screens, per, seed, first=0):
    rng = random.Random(seed)
    out, gs = [], []
    for s in range(first, first + n_screens):
        label = rng.choice("ab")
        for _ in range(per):
            y = label if rng.random() < 0.95 else ("b" if label == "a" else "a")
            k = rng.randint(1, 5)
            hint = y if rng.random() < 0.5 + 0.09 * k else ("b" if y == "a" else "a")
            out.append(({"text": f"s{s} h{hint} k{k}"}, y))
            gs.append(f"s{s}")
    return out, gs


def _risk_on_new_screens(s, fresh):
    out = [(s.ask(st), y) for st, y in fresh]
    return sum(r.by == "s1" and r.answer != y for r, y in out) / len(out)


def test_groups_split_by_source_and_the_promise_holds_on_unseen_groups():
    """Examples from one screen on both sides of a random split: System 1 is calibrated on screens it remembers and
    the promise (≤ 10% answered alone and wrong) is optimistic on new screens (a user saw 14% against 10%). groups=
    keeps every screen on one side, so the guarantee is calibrated on screens System 1 never saw — and holds there."""
    ex, gs = _screens(120, 8, seed=1)
    fresh, _ = _screens(80, 8, seed=2, first=1000)
    plain = build("close", ex, learner=_screen_learner, max_risk=0.10, novel=False)
    grouped = build("close", ex, learner=_screen_learner, max_risk=0.10, novel=False, groups=gs)
    by_key = build("close", ex, learner=_screen_learner, max_risk=0.10, novel=False,
                   groups=lambda st: st["text"].split()[0])
    assert by_key.choices["split"] == grouped.choices["split"]
    r_plain, r_grouped = _risk_on_new_screens(plain, fresh), _risk_on_new_screens(grouped, fresh)
    assert r_plain > 0.15                                        # the random split's promise does not hold on new screens
    assert r_grouped <= 0.10 + 0.02                              # the grouped one does (up to sampling)
    g = grouped.choices["groups"]
    assert g["n"] == 120 and g["fit"] + g["calibrate"] == 120 and g["fit"] == 90
    assert "split by group" in grouped.explain() and "split by group" not in plain.explain()
    # no screen on both sides
    n_fit = grouped.choices["split"]["to fit System 1"]
    import random as _r
    kinds = list(dict.fromkeys(gs))
    _r.Random(0).shuffle(kinds)
    rank = {k: i for i, k in enumerate(kinds)}
    order = sorted(range(len(ex)), key=lambda i: rank[gs[i]])
    assert not {gs[i] for i in order[:n_fit]} & {gs[i] for i in order[n_fit:]}
    with pytest.raises(ValueError, match="one group per example"):
        build("close", ex, learner=_screen_learner, max_risk=0.10, novel=False, groups=gs[:-1])
    with pytest.raises(ValueError, match="at least two"):
        build("close", ex, learner=_screen_learner, max_risk=0.10, novel=False, groups=["one"] * len(ex))
    with pytest.raises(ValueError, match="has no key 'screen'"):
        build("close", ex, learner=_screen_learner, max_risk=0.10, novel=False, groups="screen")
