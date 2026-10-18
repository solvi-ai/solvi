"""FactoredModel(action=): the per-channel action slot.

Known answer from the MiniGrid stand (stuff/stands_1019/minigrid, VERDICT_GAP_1.md, seed 1): the front channel learned
`key fits -> door open` with no action atom — right for toggle, wrong for every other action facing a locked door with
its key, which the walks rarely produced. With the action slot every channel law names an action value."""
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Grammar
    from solvi.experimental.laws.factored import Channel, FactoredModel, Row

ACTS = ("toggle", "forward", "pickup")
FRONTS = ("locked", "wall", "empty")


def grammar(with_action=True):
    atoms = [Atom(f"front={f}", (lambda f: lambda s: s[0] == f)(f), feature="front") for f in FRONTS]
    atoms += [Atom("key fits", lambda s: s[0] == "locked" and s[1], feature="fits"),
              Atom("carry=none", lambda s: not s[1], feature="carry")]
    if with_action:
        atoms += [Atom(f"act={a}", (lambda a: lambda s: s[2] == a)(a), feature="act") for a in ACTS]
    return Grammar(atoms, k=2)


def world(s):
    """The front cell after: a locked door opens only on toggle with its key."""
    front, key, act = s
    return "open" if (front == "locked" and key and act == "toggle") else front


def stream(seed, n_groups=30, steps=40, other_with_key=0.0):
    """A walk that toggles whenever it faces its door with the key (as the stand's interaction bias does): other
    actions there are rare (`other_with_key`)."""
    rng = random.Random(seed)
    for g in range(n_groups):
        for _ in range(steps):
            front, key = rng.choice(FRONTS), rng.random() < 0.5
            act = ("toggle" if rng.random() >= other_with_key else rng.choice(ACTS[1:])) if (front == "locked" and key) \
                else rng.choice(ACTS)
            s = (front, key, act)
            yield g, s, world(s)


def learn(action, free=False, seed=0):
    ch = Channel("front", grammar(), base=lambda s, name: s[0], action_free=free)
    m = FactoredModel([ch], delta=0.1, action=action)
    cur = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g, s, o in stream(seed):
            if g != cur:
                m.new_group(g)
                cur = g
            m.log([Row("front", s, o)], group=g)
    return m


def wrong(m, law):
    ls = m.laws["front"]
    return any(ls.applied(s, "front") is law and ls.apply(law.effect, s) != world(s)
               for s in [(f, k, a) for f in FRONTS for k in (False, True) for a in ACTS])


def test_without_the_action_slot_the_law_drops_the_action_and_is_false():
    m = learn(action=None)
    laws = m.laws["front"].laws
    assert any(law.effect == "open" for law in laws)
    assert any(wrong(m, law) for law in laws), [law.text() for law in laws]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_with_the_action_slot_every_law_names_an_action_and_none_is_false(seed):
    m = learn(action="act", seed=seed)
    laws = m.laws["front"].laws
    assert any(law.effect == "open" for law in laws)
    for law in laws:
        assert any(n.startswith("act=") for n in law.condition), law.text()
        assert not wrong(m, law), law.text()
    assert m.report()["front"]["action"] == "act" and m.report()["front"]["action_dropped"] > 0


def test_an_action_free_channel_is_not_bound():
    m = learn(action="act", free=True)
    assert m.report()["front"]["action"] == "free" and m.report()["front"]["action_dropped"] == 0


def test_a_channel_without_action_atoms_is_refused():
    with pytest.raises(ValueError, match="action feature"):
        FactoredModel([Channel("front", grammar(with_action=False), base=lambda s, name: s[0])], action="act")
    FactoredModel([Channel("front", grammar(with_action=False), base=lambda s, name: s[0], action_free=True)],
                  action="act")
