"""solvi.experimental.laws: history atoms (History, Tracker, Context, Same). Synthetic data only."""
from __future__ import annotations

import random
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Context, Grammar, History, LawSet, Past, Same, Tracker

NA = 4
ACTIONS = ("push", "wait")


def grammar(k=2):
    return Grammar([Atom(f"f{i}", (lambda s, i=i: s[i] == 1), feature=f"f{i}") for i in range(NA)], k=k)


def test_atoms_read_the_present_and_the_past():
    h = History(grammar(), depth=2, actions=ACTIONS, observed=("N", "E"))
    names = {a.name for a in h.grammar.atoms}
    assert {"f0", "t-1:f0", "t-2:f3", "t-1:action=push", "t-2:observed=E", "t-1:start", "t-2:start"} <= names
    feats = {a.name: a.feature for a in h.grammar.atoms}
    assert feats["f0"] == "f0" and feats["t-1:f0"] == "t-1:f0" and feats["t-1:action=wait"] == "t-1:action"
    c = Context((1, 0, 0, 0), (Past((0, 1, 0, 0), "push", "E"),))
    true = set(h.grammar.true(c, "wait"))
    assert {"f0", "t-1:f1", "t-1:action=push", "t-1:observed=E", "t-2:start"} <= true
    assert not {"t-1:f0", "t-1:start", "t-1:action=wait", "t-2:f1"} & true
    c0 = Context((0, 0, 0, 0))
    assert {"t-1:start", "t-2:start"} <= set(h.grammar.true(c0, "push"))


def test_tracker_builds_the_context_before_the_step_and_resets_per_episode():
    tr = Tracker(depth=2)
    t1 = tr.transition("s1", "a1", "o1")
    t2 = tr.transition("s2", "a2", "o2")
    t3 = tr.transition("s3", "a3", "o3")
    assert t1.state == Context("s1", ())
    assert t2.state.back(1) == Past("s1", "a1", "o1") and t2.state.back(2) is None
    assert t3.state.past == (Past("s2", "a2", "o2"), Past("s1", "a1", "o1"))
    tr.reset()
    assert tr.context("s4").past == ()


def test_same_effect_predicts_the_past_observation_and_never_matches_without_one():
    h = History(grammar(), same=True)
    ap, ef = h.apply(), h.effects()
    c = Context((0,) * NA, (Past((0,) * NA, "wait", "E"),))
    assert ap(Same(1), c) == "E" and ap("N", c) == "N"
    assert Same(1) in ef(c, "E") and Same(1) not in ef(c, "N")
    c0 = Context((0,) * NA)
    assert ap(Same(1), c0) != "E" and ap(Same(1), c0) != Same.MISSING and ef(c0, "E") == ["E"]


def _world(rng, n, kind):
    """kind "action": E where f0 and the last action was push (0.97), else N/E 0.85/0.15; kind "markov": E where f0
    and f1 (0.97); kind "same": under wait where f0 the observation repeats the last one (0.97), else a value 0..5."""
    out, last = [], None
    for j in range(n):
        if j % 25 == 0:
            last = None
        s = tuple(int(rng.random() < 0.5) for _ in range(NA))
        a = rng.choice(ACTIONS)
        if kind == "action":
            fire = s[0] == 1 and last is not None and last[1] == "push"
            o = "E" if fire and rng.random() < 0.97 else ("N" if rng.random() < 0.85 else "E")
        elif kind == "markov":
            fire = s[0] == 1 and s[1] == 1
            o = "E" if fire and rng.random() < 0.97 else ("N" if rng.random() < 0.85 else "E")
        else:
            fire = a == "wait" and s[0] == 1 and last is not None
            o = last[2] if fire and rng.random() < 0.97 else rng.choice("012345")
        out.append((s, a, o, j % 25 == 0))
        last = (s, a, o)
    return out


def _learn(rows, h, group=20):
    ls = LawSet(h.grammar, **h.kwargs(base=lambda s, a: "N"))
    buf = Buffer(h.grammar)
    tr = h.tracker()
    for j, (s, a, o, new) in enumerate(rows):
        if new:
            tr.reset()
        ls.step(tr.transition(s, a, o, group=j // group, id=str(j)), buf, remember=True)
    return ls


def _accuracy(ls, h, rows):
    tr, ok = h.tracker(), 0
    for s, a, o, new in rows:
        if new:
            tr.reset()
        ok += ls.predict(tr.context(s), a) == o
        tr.push(s, a, o)
    return ok / len(rows)


def test_a_law_over_the_last_action_is_learned_with_history_atoms_only():
    rng = random.Random(7)
    rows, test = _world(rng, 3000, "action"), _world(random.Random(8), 2000, "action")
    with_h = History(grammar(), actions=ACTIONS)
    ls = _learn(rows, with_h)
    assert any(set(law.condition) == {"f0", "t-1:action=push"} and law.effect == "E" for law in ls.laws)
    flat = History(grammar(), past_atoms=False, start=False)          # the present grammar alone
    ls0 = _learn(rows, flat)
    assert not any("t-1" in x for law in ls0.laws for x in law.condition)
    assert _accuracy(ls, with_h, test) > _accuracy(ls0, flat, test) + 0.05


def test_a_persistent_value_is_one_same_law():
    rng = random.Random(11)
    rows, test = _world(rng, 3000, "same"), _world(random.Random(12), 2000, "same")
    h = History(grammar(), actions=ACTIONS, observed="012345", same=True)
    ls = _learn(rows, h)
    same = [law for law in ls.laws if law.effect == Same(1)]
    assert [(law.action, law.condition) for law in same] == [("wait", ("f0",))]
    literal = History(grammar(), actions=ACTIONS, observed="012345")      # per-value laws only
    ls0 = _learn(rows, literal)
    assert _accuracy(ls, h, test) >= _accuracy(ls0, literal, test) - 0.01


def test_no_history_law_where_the_world_has_none():
    rng = random.Random(5)
    rows = _world(rng, 3000, "markov")
    h = History(grammar(), actions=ACTIONS, observed="NE", same=True)
    ls = _learn(rows, h)
    assert any(set(law.condition) == {"f0", "f1"} and law.effect == "E" for law in ls.laws)
    for law in ls.laws:            # any law with a history atom must still be a true law of this world (f0 and f1)
        if any(x.startswith("t-") for x in law.condition) or isinstance(law.effect, Same):
            assert {"f0", "f1"} <= set(law.condition) and law.effect == "E"
