"""The factored transition model (solvi.experimental.laws.factored) on a synthetic push world, and a null simulation.

    python benchmarks/laws_factored_toy.py            # the push world (factored and joint arms) + the null run, as JSON
    pytest tests/test_laws_factored.py                # the same with the bars

1. push_world() — a line of 7 cells holding 4 objects of distinct types (6 types per world); each type gets random
   properties per world: M (moves when told to) with p = 0.5, one of P (pushed) / S (stops a mover) / neither with
   p = 0.3 / 0.3 / 0.4, D (vanishes on contact, with whatever shares its cell) with p = 0.3. An action (object,
   direction) tells one object to step. The written rules: an object without M stays; a mover into an empty cell
   moves; into a cell with an S object it is blocked; into a cell with P objects it pushes them one further when that
   cell is free and on the line (else blocked); otherwise it passes and shares the cell; then every object sharing a
   cell with a D object (or being one, while sharing) vanishes. Properties are observable (the caller's features);
   the laws are not.

   Channels (the caller's extraction, below): self (the told object's own properties `p:` → move / stay; clean when
   its target cell is on the line and empty), enter (the occupant's properties `q:` and the mover's `a:` → push /
   block / pass; clean when the mover is predicted to move by a self law — the row's `depends` — the target holds
   exactly one object and the cell beyond is on the line and free), gone (own `p:` and mate `m:` → gone / stays; the
   two of a pass, both seen sharing or both vanished). Written laws: self p:M → move; enter q:S → block, q:P → push;
   gone p:D → gone, m:D → gone. The composer (the turn) is the caller's and reads model.predict.

   Arms: the factored model (three channels, δ = 0.1 split equally) and a joint single LawSet over the whole local
   observation (the mover's and occupant's properties and the target / beyond cells → the tuple of the four outcomes),
   both learned online from the same stream: 150 training worlds × 60 steps (re-placed every 10), each world a group.
   Reported: laws, false laws (an active law that predicts otherwise than the written rules on any state where it is
   the applied law), next-state accuracy on 20 fresh worlds × 60 steps (all transitions; and the clean ones both arms
   can speak on), rows per accepted law. Measured (seeds 0–4): factored next-state accuracy 1.000 on every seed, 0
   false laws; the written laws literally in 4 of 5 seeds (seed 0 holds m:D as m:D & m:M and m:D & p:M, which predict
   the same on every reachable state: one of the pair is always the mover); the joint arm 0.947–0.952 on the clean
   transitions with 2 laws.

2. null_run() — three channels of 40 sparse atoms each (an atom true with p = 0.06), effects drawn at 0.7
   independently of every atom (no law exists), 2,000 rows of 10 groups per channel; every condition of ≤ 2 atoms
   (821) is tested on each channel's buffer. Reported: trials with any law accepted in any channel — the factored
   model at δ = 0.1 shared equally, and fixed thresholds (support ≥ 5, precision ≥ 0.9) as the control. Measured
   (100 trials, seed 0): 0 of 100 factored, 100 of 100 fixed."""
from __future__ import annotations

import itertools
import json
import random
import sys
import time
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition
    from solvi.experimental.laws.factored import Channel, FactoredModel, Row

PROPS = ("M", "P", "S", "D")
LINE = 7
TYPES = 6
OBJECTS = 4


# ───────────────────────────────────────────── the world (written rules)
def make_world(rng):
    props = {}
    for t in range(TYPES):
        p = set()
        if rng.random() < 0.5:
            p.add("M")
        r = rng.random()
        if r < 0.3:
            p.add("P")
        elif r < 0.6:
            p.add("S")
        if rng.random() < 0.3:
            p.add("D")
        props[t] = frozenset(p)
    return props


def place(rng):
    """{object id: (type, cell)}: OBJECTS objects of distinct types on distinct cells."""
    types = rng.sample(range(TYPES), OBJECTS)
    cells = rng.sample(range(LINE), OBJECTS)
    return {i: (types[i], cells[i]) for i in range(OBJECTS)}


def at(pos, cell, but=None):
    return [o for o, (_, c) in pos.items() if c == cell and o != but]


def true_step(props, pos, action):
    o, d = action
    pos = dict(pos)
    t, c = pos[o]
    dest = c + d
    if "M" in props[t] and 0 <= dest < LINE:
        occ = at(pos, dest)
        if not occ:
            pos[o] = (t, dest)
        elif any("S" in props[pos[x][0]] for x in occ):
            pass
        elif any("P" in props[pos[x][0]] for x in occ):
            beyond = dest + d
            if 0 <= beyond < LINE and not at(pos, beyond):
                for x in occ:
                    if "P" in props[pos[x][0]]:
                        pos[x] = (pos[x][0], beyond)
                pos[o] = (t, dest)
        else:
            pos[o] = (t, dest)
    return vanish(pos, lambda x, y: "D" in props[pos[x][0]] or "D" in props[pos[y][0]])


def vanish(pos, gone_with):
    """Remove every object for which some object sharing its cell makes it go (gone_with(x, mate))."""
    out = {}
    for x, (t, c) in pos.items():
        mates = at(pos, c, but=x)
        if not any(gone_with(x, y) for y in mates):
            out[x] = (t, c)
    return out


# ───────────────────────────────────────────── features (observable properties)
def feats(props, t, prefix):
    return frozenset(prefix + p for p in props[t])


class Has:
    """Atom function: the name is in the state's feature set (a class, so a model pickles)."""
    def __init__(self, name):
        self.name = name

    def __call__(self, s):
        return self.name in s


def has(name):
    return Has(name)


def grammar(prefixes):
    return Grammar([Atom(px + p, has(px + p), feature=px + p, code=f"{(px + p)!r} in s")
                    for px in prefixes for p in PROPS], k=2)


def channels():
    return [Channel("self", grammar(["p:"]), base=base_self),
            Channel("enter", grammar(["q:", "a:"]), base=base_enter),
            Channel("gone", grammar(["p:", "m:"]), base=base_gone)]


def base_self(s, a):
    return "stay"


def base_enter(s, a):
    return "pass"


def base_gone(s, a):
    return "stays"


# ───────────────────────────────────────────── the caller's extraction (credit assignment)
class PushRows:
    """Rows of one transition. pre / post: (props, pos)."""

    def rows(self, model, pre, action, post):
        props, pos = pre
        _, npos = post
        o, d = action
        t, c = pos[o]
        dest = c + d
        out = []
        if not 0 <= dest < LINE:
            return out
        occ = at(pos, dest)
        me = feats(props, t, "p:")
        if not occ:
            moved = o in npos and npos[o][1] == dest
            out.append(Row("self", me, "move" if moved else "stay"))
            return out
        law = model.applied("self", me)
        if law is None or law.effect != "move" or len(occ) != 1:
            return out
        q = occ[0]
        beyond = dest + d
        if not 0 <= beyond < LINE or at(pos, beyond):
            return out
        qt = pos[q][0]
        state = feats(props, qt, "q:") | feats(props, t, "a:")
        if q in npos and npos[q][1] == beyond:
            label = "push"
        elif o in npos and npos[o][1] == c:
            label = "block"
        elif o in npos and q in npos and npos[o][1] == dest and npos[q][1] == dest:
            label = "pass"
        elif o not in npos and q not in npos:
            label = "pass"                      # both vanished: only sharing a cell makes an object vanish
        else:
            return out
        out.append(Row("enter", state, label, depends=(law,)))
        if label == "pass":
            both_gone = o not in npos and q not in npos
            both_here = o in npos and q in npos
            if both_gone or both_here:
                obs = "gone" if both_gone else "stays"
                out.append(Row("gone", feats(props, t, "p:") | feats(props, qt, "m:"), obs, depends=(law,)))
                out.append(Row("gone", feats(props, qt, "p:") | feats(props, t, "m:"), obs, depends=(law,)))
        return out


# ───────────────────────────────────────────── the caller's composer (the turn)
def model_step(model, props, pos, action):
    o, d = action
    pos = dict(pos)
    t, c = pos[o]
    dest = c + d
    if model.predict("self", feats(props, t, "p:")) == "move" and 0 <= dest < LINE:
        occ = at(pos, dest)
        said = {x: model.predict("enter", feats(props, pos[x][0], "q:") | feats(props, t, "a:")) for x in occ}
        if not occ:
            pos[o] = (t, dest)
        elif "block" in said.values():
            pass
        elif "push" in said.values():
            beyond = dest + d
            if 0 <= beyond < LINE and not at(pos, beyond):
                for x in occ:
                    if said[x] == "push":
                        pos[x] = (pos[x][0], beyond)
                pos[o] = (t, dest)
        else:
            pos[o] = (t, dest)
    return vanish(pos, lambda x, y: model.predict("gone", feats(props, pos[x][0], "p:")
                                                  | feats(props, pos[y][0], "m:")) == "gone")


def stream(rng, worlds, steps=60, episode=10):
    """(world id, props, pos, action, next pos) — a random walk of told objects, re-placed every `episode` steps."""
    for w in range(worlds):
        props = make_world(rng)
        pos = None
        for i in range(steps):
            if i % episode == 0 or len(pos) < 2:
                pos = place(rng)
            o = rng.choice(sorted(pos))
            a = (o, rng.choice((-1, 1)))
            nxt = true_step(props, pos, a)
            yield w, props, pos, a, nxt
            pos = nxt


# ───────────────────────────────────────────── written laws (the oracle for false laws)
WRITTEN = {"self": lambda s: "move" if "p:M" in s else "stay",
           "enter": lambda s: "block" if "q:S" in s else "push" if "q:P" in s else "pass",
           "gone": lambda s: "gone" if ("p:D" in s or "m:D" in s) else "stays"}


def all_states(prefixes):
    names = [px + p for px in prefixes for p in PROPS]
    for bits in itertools.product((0, 1), repeat=len(names)):
        s = frozenset(n for n, b in zip(names, bits) if b)
        if any(px + "P" in s and px + "S" in s for px in prefixes):
            continue                                              # P and S are exclusive by construction
        yield s


STATES = {"self": list(all_states(["p:"])), "enter": list(all_states(["q:", "a:"])),
          "gone": list(all_states(["p:", "m:"]))}


def false_laws(model):
    """Active laws that predict otherwise than the written rules on any state where they are the applied law."""
    bad = []
    for ch in model.names:
        ls = model.laws[ch]
        for law in ls.laws:
            for s in STATES[ch]:
                if ls.applied(s, ch) is law and ls.apply(law.effect, s) != WRITTEN[ch](s):
                    bad.append((ch, law.text()))
                    break
    return bad


# ───────────────────────────────────────────── the joint arm: one LawSet over the whole local observation
def joint_grammar():
    atoms = [Atom(px + p, has(px + p), feature=px + p) for px in ("a:", "q:") for p in PROPS]
    for f, vals in (("dest", ("empty", "one", "wall")), ("beyond", ("free", "taken"))):
        atoms += [Atom(f"{f}={v}", has(f"{f}={v}"), feature=f) for v in vals]
    return Grammar(atoms, k=3)


JOINT_BASE = ("stay", "stays", "stays", "stays")


def joint_state(props, pos, action):
    """The local situation of a clean transition (target empty / off the line, or one occupant with the cell beyond
    on the line), else None."""
    o, d = action
    t, c = pos[o]
    dest = c + d
    s = set(feats(props, t, "a:"))
    if not 0 <= dest < LINE:
        return frozenset(s | {"dest=wall"})
    occ = at(pos, dest)
    if not occ:
        return frozenset(s | {"dest=empty"})
    beyond = dest + d
    if len(occ) != 1 or not 0 <= beyond < LINE:
        return None
    s |= feats(props, pos[occ[0]][0], "q:") | {"dest=one", "beyond=" + ("taken" if at(pos, beyond) else "free")}
    return frozenset(s)


def joint_label(pos, action, npos):
    o, d = action
    c = pos[o][1]
    occ = at(pos, c + d)
    q = occ[0] if occ else None
    mv = "move" if o in npos and npos[o][1] == c + d else ("stay" if o in npos else "?")
    qm = "pushed" if q is not None and q in npos and npos[q][1] == c + 2 * d else "stays"
    og = "gone" if o not in npos else "stays"
    qg = "gone" if q is not None and q not in npos else "stays"
    if mv == "?":                                             # it vanished: it moved into the shared cell
        mv = "move"
    return (mv, qm, og, qg)


def joint_apply(label, pos, action):
    o, d = action
    pos = dict(pos)
    c = pos[o][1]
    occ = at(pos, c + d)
    mv, qm, og, qg = label
    if qm == "pushed" and occ:
        pos[occ[0]] = (pos[occ[0]][0], c + 2 * d)
    if mv == "move":
        pos[o] = (pos[o][0], c + d)
    if og == "gone":
        pos.pop(o, None)
    if qg == "gone" and occ:
        pos.pop(occ[0], None)
    return pos


# ───────────────────────────────────────────── runs
def learn_factored(seed=0, worlds=150, steps=60, delta=0.1):
    rng = random.Random(seed)
    model = FactoredModel(channels(), PushRows(), delta=delta)
    for w, props, pos, a, nxt in stream(rng, worlds, steps):
        if model.group != ("train", w):
            model.new_group(("train", w))
        model.step((props, pos), a, (props, nxt))
    return model


def learn_joint(seed=0, worlds=150, steps=60, delta=0.1):
    rng = random.Random(seed)
    ls = LawSet(joint_grammar(), base=lambda s, a: JOINT_BASE, delta=delta)
    buf = Buffer(ls.grammar)
    seen = set()
    for w, props, pos, a, nxt in stream(rng, worlds, steps):
        s = joint_state(props, pos, a)
        if s is None:
            continue
        lab = joint_label(pos, a, nxt)
        k = (w, s, lab)
        if k in seen:                                             # the same dedup as the factored model
            continue
        seen.add(k)
        ls.step(Transition(s, "turn", lab, group=("train", w)), buf, remember=True)
    return ls, buf


def evaluate(model, joint, seed=1000, worlds=20, steps=60):
    rng = random.Random(seed)
    n = ok = n_clean = ok_clean = ok_joint = 0
    for _, props, pos, a, nxt in stream(rng, worlds, steps):
        got = model_step(model, props, pos, a)
        n += 1
        ok += got == nxt
        s = joint_state(props, pos, a)
        if s is not None:
            n_clean += 1
            ok_clean += got == nxt
            if joint is not None:
                ok_joint += joint_apply(joint.predict(s, "turn"), pos, a) == nxt
    return {"transitions": n, "accuracy": ok / n, "clean": n_clean, "accuracy_clean": ok_clean / n_clean,
            "joint_accuracy_clean": (ok_joint / n_clean) if joint is not None else None}


def push_world(seed=0, worlds=150, steps=60):
    t0 = time.time()
    model = learn_factored(seed, worlds, steps)
    ls, buf = learn_joint(seed, worlds, steps)
    ev = evaluate(model, ls, seed=1000 + seed)
    rep = model.report()
    laws = {ch: sorted(law.text() for law in model.laws[ch].laws) for ch in model.names}
    rows_f = sum(rep[ch]["rows"] for ch in model.names)
    nf = sum(len(model.laws[ch].laws) for ch in model.names)
    return {"seed": seed, "laws": laws, "false_laws": false_laws(model), **ev,
            "factored_rows_per_law": rows_f / max(1, nf), "joint_laws": len(ls.laws),
            "joint_rows_per_law": len(buf) / max(1, len(ls.laws)), "bound": rep["bound"],
            "report": {ch: {k: rep[ch][k] for k in ("rows", "tested", "need", "dropped", "voided", "refused",
                                                    "rows_per_transition")} for ch in model.names},
            "seconds": round(time.time() - t0, 2)}


# ───────────────────────────────────────────── the null run
def sparse_grammar(n=40, k=2):
    return Grammar([Atom(f"f{i}", (lambda s, i=i: s[i]), feature=f"f{i}") for i in range(n)], k=k)


def every_condition(g, k=2):
    names = [a.name for a in g.atoms]
    return [c for size in range(k + 1) for c in itertools.combinations(names, size)]


def null_trial(rng, *, fixed=False, rows=2000, n=40, delta=0.1):
    """One model of three null channels (fixed=True: the control, three LawSets at the old fixed thresholds) → the
    number of channels that accepted a law."""
    g = sparse_grammar(n)
    conds = every_condition(g)
    names = ("c0", "c1", "c2")
    if fixed:
        sets = {c: LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9, sized=False) for c in names}
        bufs = {c: Buffer(g) for c in names}
    else:
        model = FactoredModel([Channel(c, g, base=lambda s, a: "N", min_support=5, min_precision=0.9) for c in names],
                              delta=delta)
        sets, bufs = model.laws, model.buffers
    hit = 0
    for c in names:
        for r in range(rows):
            s = tuple(rng.random() < 0.06 for _ in range(n))
            bufs[c].add(Transition(s, c, "E" if rng.random() < 0.7 else "N", group=r % 10, id=f"{c}{r}"))
        hit += bool(sets[c].test([Candidate(c, cond, "E") for cond in conds], bufs[c]))
    return hit


def null_run(trials=100, seed=0, rows=2000):
    rng = random.Random(seed)
    sized = sum(null_trial(rng, rows=rows) > 0 for _ in range(trials))
    fixed = sum(null_trial(rng, fixed=True, rows=rows) > 0 for _ in range(trials))
    return {"trials": trials, "any_false_law_factored": sized, "any_false_law_fixed": fixed, "delta": 0.1}


if __name__ == "__main__":
    out = {"push_world": push_world()}
    if "--null" in sys.argv or len(sys.argv) == 1:
        out["null"] = null_run()
    print(json.dumps(out, indent=1, default=str))
