"""solvi.experimental.skills footprints with variables: a verified plan's footprint is generalized on the exact copy —
constants to variables, atoms dropped — and a candidate is accepted only when the plan, instantiated for every
verification state that matches it but not the original, reaches its goal on a fresh copy of that state.

The world used throughout: a grove, a 7 × 7 patch with the player in the middle. Cells are grass (walkable), tree (gives
wood), rock (gives stone, only with a pick) or water. Moving into a cell that is not grass turns the player to it;
"do" acts on the faced cell. Every layout has a tree next to the player in a direction drawn from the seed, so a plan
learned for one direction ("move_left, do") holds — turned — in every direction, and "do" on rock needs the pick.
`blind=` makes "do" refused when facing that direction (a world where the turned plan is false)."""
from __future__ import annotations

import random
import warnings

import pytest

from solvi.core import Outcome
from solvi.core.knowledge import KnowledgeStore
from solvi.core.store import JSONLStorage

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Grammar
    from solvi.experimental.skills import (Certificate, Footprint, Lift, SkillLearner, Var, Vocabulary, generalize,
                                           lattice, replay, unify)

DIRS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
NAME = {v: k for k, v in DIRS.items()}
N = 7
MAT = {"wood": "tree", "stone": "rock"}


def _k(c):
    return f"{c[0]},{c[1]}"


class Grove:
    def __init__(self, blind=None, rock=False):
        self.blind, self.rock = blind, rock
        self.reset(0)

    def reset(self, seed=None):
        rng = random.Random(seed)
        self.pos = (3, 3)
        self.facing = rng.choice(list(DIRS.values()))
        self.cells = {_k((x, y)): "grass" for x in range(N) for y in range(N)}
        d = rng.choice(list(DIRS.values()))
        self.cells[_k((3 + d[0], 3 + d[1]))] = "tree"
        if self.rock:
            others = [v for v in DIRS.values() if v != d]
            r = rng.choice(others)
            self.cells[_k((3 + r[0], 3 + r[1]))] = "rock"
        for _ in range(3):                                     # water away from the player
            c = (rng.choice([0, 6]), rng.randrange(N))
            self.cells[_k(c)] = "water"
        self.inv = {"wood": 0, "stone": 0, "pick": int(rng.random() < 0.5)}
        self.colour = rng.randrange(3)                         # irrelevant
        return self.state()

    def state(self):
        return {"pos": list(self.pos), "facing": list(self.facing), "inv": dict(self.inv), "cells": dict(self.cells),
                "colour": self.colour}

    def actions(self, state):
        return [f"move_{n}" for n in DIRS] + ["do"]

    def step(self, a):
        if a.startswith("move_"):
            d = DIRS[a[5:]]
            t = (self.pos[0] + d[0], self.pos[1] + d[1])
            if _k(t) not in self.cells:
                return Outcome(self.state(), accepted=False)
            self.facing = d
            if self.cells[_k(t)] == "grass":
                self.pos = t
            return Outcome(self.state(), accepted=True)
        if a == "do":
            if self.blind is not None and tuple(self.facing) == self.blind:
                return Outcome(self.state(), accepted=False)
            f = _k((self.pos[0] + self.facing[0], self.pos[1] + self.facing[1]))
            m = self.cells.get(f)
            if m == "tree":
                self.inv["wood"] += 1
                return Outcome(self.state(), accepted=True)
            if m == "rock" and self.inv["pick"] >= 1:
                self.inv["stone"] += 1
                self.cells[f] = "grass"
                return Outcome(self.state(), accepted=True)
        return Outcome(self.state(), accepted=False)


def _rot(v, b0, d):
    """The rotation that takes direction b0 to d, applied to v."""
    r = tuple(b0)
    for k in range(4):
        if r == tuple(d):
            x, y = v
            for _ in range(k):
                x, y = -y, x
            return (x, y)
        r = (-r[1], r[0])
    raise ValueError((b0, d))


class GroveVocab(Vocabulary):
    """cell(offset, material), facing(direction), count_ge(item, k), has_tool_for(R); lifts: R (the resource of the
    goal, with its tool) and D (the plan's direction: offsets, facing and moves turn with it)."""
    D = Var("D", "dir")
    R = Var("R", "res")

    def holds(self, atom, s):
        rel = atom[0]
        if rel == "cell":
            off, mat = atom[1], atom[2]
            return s["cells"].get(_k((s["pos"][0] + off[0], s["pos"][1] + off[1]))) == mat
        if rel == "facing":
            return tuple(s["facing"]) == tuple(atom[1])
        if rel == "count_ge":
            return s["inv"].get(atom[1], 0) >= atom[2]
        if rel == "has_tool_for":
            return atom[1] == "wood" or s["inv"]["pick"] >= 1
        raise ValueError(atom)

    def domain(self, var, s, b):
        return list(DIRS.values()) if var.kind == "dir" else ["wood", "stone"]

    def ground(self, term, b):
        if isinstance(term, tuple) and term and term[0] == "@":
            _, v, b0, off = term
            d = b.get(v, v) if isinstance(v, Var) else v
            return term if isinstance(d, Var) else _rot(off, b0, d)
        if isinstance(term, tuple) and term and term[0] == "mat_of":
            r = b.get(term[1], term[1]) if isinstance(term[1], Var) else term[1]
            return term if isinstance(r, Var) else MAT[r]
        if isinstance(term, tuple):
            return tuple(self.ground(x, b) for x in term)
        return b.get(term, term) if isinstance(term, Var) else term

    def goal(self, name):
        return ("get", name)

    def goal_name(self, term):
        return term[1]

    def action(self, a):
        return ("move", DIRS[a[5:]]) if a.startswith("move_") else a

    def act(self, term):
        return f"move_{NAME[tuple(term[1])]}" if isinstance(term, tuple) else term

    def footprint(self, state, actions, predicted):
        s, x0, y0 = state, state["pos"][0], state["pos"][1]
        out, moved = [], False
        for a, nxt in zip(actions, list(predicted) or [None] * len(actions)):
            if a.startswith("move_"):
                d = DIRS[a[5:]]
                c = (s["pos"][0] + d[0], s["pos"][1] + d[1])
                out.append(("cell", (c[0] - x0, c[1] - y0), s["cells"][_k(c)]))
                moved = True
            elif a == "do":
                if not moved:
                    out.append(("facing", tuple(s["facing"])))
                c = (s["pos"][0] + s["facing"][0], s["pos"][1] + s["facing"][1])
                m = s["cells"].get(_k(c))
                out.append(("cell", (c[0] - x0, c[1] - y0), m))
                if m == "rock":
                    out.append(("count_ge", "pick", 1))
            s = nxt if nxt is not None else s
        return tuple(dict.fromkeys(out))

    def lifts(self, atoms, actions, goal):
        out = []
        r0 = goal[1] if isinstance(goal, tuple) and goal[1] in MAT else None
        if r0 is not None and any(a[0] == "cell" and a[2] == MAT[r0] for a in atoms):
            R = self.R

            def lift_r(at, ac, g, r0=r0):
                at2 = [("cell", a[1], ("mat_of", R)) if a[0] == "cell" and a[2] == MAT[r0] else a for a in at
                       if not (a[0] == "count_ge" and a[1] == "pick")]
                return tuple(at2) + (("has_tool_for", R),), ac, ("get", R)
            out.append(Lift("R", lift_r))
        b0 = next((a[1] for a in actions if isinstance(a, tuple) and a[0] == "move"), None)
        if b0 is None:
            b0 = next((a[1] for a in atoms if a[0] == "facing"), None)
        if b0 is not None:
            D = self.D

            def lift_d(at, ac, g, b0=b0):
                at2 = []
                for a in at:
                    if a[0] == "cell":
                        at2.append(("cell", ("@", D, b0, a[1]), a[2]))
                    elif a[0] == "facing":
                        at2.append(("facing", ("@", D, b0, a[1])))
                    else:
                        at2.append(a)
                ac2 = tuple(("move", ("@", D, b0, a[1])) if isinstance(a, tuple) and a[0] == "move" else a
                            for a in ac)
                return tuple(at2), ac2, g
            out.append(Lift("D", lift_d))
        return out


def world_at(state, blind=None):
    """A Grove at a given state (the clone hook of the exact model)."""
    g = Grove(blind=blind)
    g.pos, g.facing = tuple(state["pos"]), tuple(state["facing"])
    g.cells, g.inv, g.colour = dict(state["cells"]), dict(state["inv"]), state["colour"]
    return g


def check_on(blind=None):
    def check(handle, state, actions, goal):
        w = world_at(state)
        w.blind = blind
        trace = []
        for a in actions:
            o = w.step(a)
            trace.append((a, o.accepted))
            if not o.accepted:
                return False, repr(trace)
            if o.state["inv"].get(goal, 0) > state["inv"].get(goal, 0):
                return True, repr(trace)
        return False, repr(trace)
    return check


def pool_of(seeds, **kw):
    g = Grove(**kw)
    return [(g.reset(s), None) for s in seeds]


def state_with_tree(d, pick=0):
    g = Grove()
    g.reset(0)
    for k in list(g.cells):
        if g.cells[k] in ("tree", "rock"):
            g.cells[k] = "grass"
    g.cells[_k((3 + d[0], 3 + d[1]))] = "tree"
    g.inv["pick"] = pick
    return g.state()


VOC = GroveVocab()


# ------------------------------------------------------------------------------------------------ terms and matching
def test_unify_and_a_footprint_finds_the_binding_its_atoms_hold_under():
    D, R = VOC.D, VOC.R
    assert unify(("get", R), ("get", "wood")) == {R: "wood"}
    assert unify(("get", R), ("get", "wood"), {R: "stone"}) is None
    fp = Footprint((("cell", ("@", D, (-1, 0), (-1, 0)), ("mat_of", R)), ("has_tool_for", R)))
    assert set(fp.variables) == {D, R} and not fp.is_ground
    s = state_with_tree((0, 1))
    b = fp.match(VOC, s)
    assert b == {D: (0, 1), R: "wood"}
    assert fp.match(VOC, s, {R: "stone"}) is None                     # no rock next to the player
    assert "has_tool_for(?R)" in str(fp)


def test_the_lattice_is_bounded_most_general_first_and_excludes_the_original():
    atoms = (("cell", (-1, 0), "tree"),)
    c = lattice(atoms, ("move_left", "do"), "wood", VOC)
    gens = [x.generality for x in c]
    assert gens == sorted(gens, reverse=True)
    assert all(x.lifts or x.dropped for x in c)
    full = next(x for x in c if set(x.lifts) == {"R", "D"} and not x.dropped)
    assert set(full.footprint.variables) == {VOC.D, VOC.R}
    assert full.actions[0] == ("move", ("@", VOC.D, (-1, 0), (-1, 0)))
    assert len(lattice(atoms, ("move_left", "do"), "wood", VOC, max_candidates=2)) == 2


# ------------------------------------------------------------------------------------------------ generalize
def _gen(pool, blind=None, **kw):
    s0 = state_with_tree((-1, 0))
    w = world_at(s0)
    atoms = VOC.footprint(s0, ("move_left", "do"), [w.step(a).state for a in ("move_left", "do")])
    orig = Footprint(atoms)
    return generalize(atoms, ("move_left", "do"), "wood", VOC, pool=pool, check=check_on(blind),
                      original=lambda s, g: g == "wood" and orig.match(VOC, s) is not None, **kw)


def test_a_turned_plan_the_world_agrees_with_is_accepted_with_variables_and_its_count():
    pool = pool_of(range(40), rock=True)
    g = _gen(pool, verify_n=12, verify_min=6)
    assert g.accepted and set(g.lifts) == {"R", "D"}
    assert set(g.variables) == {VOC.D, VOC.R}
    assert "has_tool_for(?R)" in str(g.footprint)                    # the drop of the tool atom failed on a rock
    dropped_tool = [c for c in g.candidates if c["dropped"] == [["has_tool_for", "?R"]]]
    assert dropped_tool and dropped_tool[0]["failed"] is not None
    assert g.verified == 12 and len(g.states) == 12
    assert {s["goal"] for s in g.states} == {"wood", "stone"}        # each use is asked for a goal: both checked
    rep = replay(g, check_on())
    assert rep["states"] == 12 and rep["same"] == 12 and rep["differ"] == []


def test_a_lift_the_world_refutes_is_rejected_and_the_most_general_passing_one_is_kept():
    pool = pool_of(range(200), rock=True, blind=(0, -1))
    g = _gen(pool, blind=(0, -1), verify_n=12, verify_min=6)     # "do" facing up is refused in this world
    assert g.accepted and "D" not in g.lifts                          # the turned plan failed on a tree above
    turned = [c for c in g.candidates if "D" in c["lifts"]]
    assert turned and all(c["failed"] is not None or c["why"] for c in turned)
    assert g.lifts == ("R",)


def test_without_verification_states_nothing_is_accepted():
    g = _gen([], verify_n=8)
    assert not g.accepted and g.verified == 0 and g.footprint == g.original
    assert all("verification states found" in (c["why"] or "") for c in g.candidates)


def test_states_where_the_original_holds_are_never_verification_states():
    s0 = state_with_tree((-1, 0))
    pool = [(dict(s0, colour=c), None) for c in range(3)]             # all match the original (tree on the left)
    g = _gen(pool, verify_n=4, verify_min=1)
    for c in g.candidates:
        assert c["pool"] == 0 or c["goal"] != "wood" or c["lifts"]
    assert not any(s["goal"] == "wood" and s["do"] == ["move_left", "do"] for s in g.states)


def test_perturbations_supply_states_the_pool_lacks():
    def perturb(handle, state, rng, cand):
        out = []
        for d in DIRS.values():
            out.append((state_with_tree(d, pick=rng.randrange(2)), None))
        return out
    g = _gen([], perturb=perturb, verify_n=6, verify_min=6, start=(state_with_tree((-1, 0)), None))
    assert g.accepted and "D" in g.lifts and g.verified == 6
    assert {s["source"] for s in g.states} == {"perturbed"}


# ------------------------------------------------------------------------------------------------ the learner
def grammar():
    atoms = [Atom(f"tree_{n}", (lambda s, d=d: s["cells"].get(_k((s["pos"][0] + d[0], s["pos"][1] + d[1]))) == "tree"),
                  feature=f"tree_{n}") for n, d in DIRS.items()]
    atoms += [Atom(f"facing_{n}", (lambda s, d=d: tuple(s["facing"]) == d), feature="facing") for n, d in DIRS.items()]
    atoms += [Atom(f"colour={c}", (lambda s, c=c: s["colour"] == c), feature="colour") for c in range(3)]
    atoms += [Atom("pick", lambda s: s["inv"]["pick"] >= 1, feature="pick")]
    return Grammar(atoms, k=3)


def learner(env, **kw):
    return SkillLearner(env, grammar(), {"wood": lambda s: s["inv"]["wood"] >= 1}, horizon=4, budget=2000,
                        certificate=Certificate(promise=0.2, delta=0.1, scope="set", delta_split=False),
                        clone=lambda e, s: world_at(s, e.blind),
                        **kw)


def stream(L, n=30):
    return [L.run(seed=s, steps=10) for s in range(n)]


def test_generalize_false_keeps_the_old_behaviour_exactly():
    a, b = learner(Grove()), learner(Grove(), lift=False, vocabulary=VOC, verify_n=4)
    stream(a, 20)
    stream(b, 20)
    strip = lambda ds: [{k: v for k, v in d.items()} for d in ds]  # noqa: E731
    assert strip(a.decisions) == strip(b.decisions)
    assert [s.to_dict() for s in a.skills.skills] == [s.to_dict() for s in b.skills.skills]
    assert all(s.footprint is None and s.generalization is None for s in b.skills.skills)
    assert sum(e["generalized"] for e in b.episodes) == 0


def test_the_learner_generalizes_on_the_copy_and_system_1_grows_faster():
    old = learner(Grove())
    new = learner(Grove(), lift="lattice", vocabulary=VOC, verify_n=8, verify_min=4, pool_size=32)
    eo, en = stream(old), stream(new)
    share = lambda eps: sum(e["s1"] for e in eps) / max(1, sum(e["steps"] for e in eps))  # noqa: E731
    assert share(en[-10:]) > share(eo[-10:])
    var = [s for s in new.skills.skills if s.footprint is not None]
    assert var and all(s.variables and s.verified >= 4 and s.original for s in var)
    assert sum(e["generalize_accepted"] for e in en) >= 1 and sum(e["generalized"] for e in en) >= 1
    assert len(new.skills.skills) < len(old.skills.skills)            # one turned skill covers four directions
    s1 = [d for d in new.decisions if d["by"] == "s1"]
    assert s1 and all(d["run_ok"] is not False for d in s1)
    sk = var[0]
    assert new.replay(sk)["same"] == sk.verified
    d = sk.to_dict()
    assert d["footprint"] and d["variables"] and d["verified"] == sk.verified and d["original"]


def test_a_turned_skill_is_refused_where_the_world_disagrees_and_the_certificate_stays_clean():
    new = learner(Grove(blind=(0, -1)), lift="lattice", vocabulary=VOC, verify_n=8, verify_min=4)
    stream(new, 30)
    for s in new.skills.skills:
        if s.footprint is not None:
            assert "D" not in (s.generalization or {}).get("lifts", [])
    refuted = [c for s in new.skills.skills for c in (s.generalization or {}).get("candidates", [])
               if "D" in c["lifts"] and c["failed"] is not None]
    assert refuted and any("move_up" in c["why"] for c in refuted)    # the copy said no: "do" facing up is refused
    assert sum(e["alone_failures"] for e in new.episodes) == 0


def test_the_generalized_skill_is_written_to_knowledge_with_both_footprints(tmp_path):
    new = learner(Grove(), lift="lattice", vocabulary=VOC, verify_n=8, verify_min=4)
    stream(new, 12)
    store = KnowledgeStore(JSONLStorage(tmp_path / "k.jsonl"))
    ids = new.skills.to_knowledge(store, {"env": "grove"})
    var = [s for s in new.skills.skills if s.footprint is not None]
    assert var and var[0].id in ids
    item = store.item(ids[var[0].id])
    assert item["body"]["footprint"] and "?D" in str(item["body"]["do"])
    ev = item["evidence"][-1]
    assert ev["verified"] == var[0].verified and ev["original"] and ev["generalization"]["accepted"]


def test_a_goal_outside_a_variables_domain_is_not_the_skills():
    from solvi.experimental.skills import Skill, SkillSet
    ks = SkillSet(None, certificate=None, vocabulary=VOC)
    s = ks.add(Skill((), ("move_left", "do"), "wood"))
    s.footprint = Footprint((("has_tool_for", VOC.R),))                # holds for wood anywhere
    s.template, s.goal_template = ("do",), ("get", VOC.R)
    st = state_with_tree((-1, 0))
    assert ks.actions_for(s, st, "wood") == ("do",)
    assert ks.binding(s, st, "gold") is None and ks.candidates(st, "gold") == []   # R ranges over wood and stone


def test_generalize_needs_a_vocabulary_and_a_known_mode():
    with pytest.raises(ValueError):
        learner(Grove(), lift="lattice")
    with pytest.raises(ValueError):
        learner(Grove(), lift="widest", vocabulary=VOC)
