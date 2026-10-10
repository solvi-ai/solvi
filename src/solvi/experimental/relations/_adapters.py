"""Encoders of a state into relations (an egocentric window of a grid, a path tree), relational atoms for a laws
grammar and the proposer that hands them to the unchanged held-out test (see the package docstring)."""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Iterable, Sequence

from ..laws import Atom, Candidate, Grammar
from ._algebra import Relations, _Batch, _eval, readout, show
from ._compile import RUNTIME, compile_ast
from ._enumerate import Enumerator, Library, rank

MATERIALS = ("water", "grass", "stone", "path", "sand", "tree", "lava", "coal", "iron", "diamond", "table", "furnace")
CREATURES = ("zombie", "skeleton", "cow", "arrow", "plant")
PROPOSER_OPS = ("T", "PLUS", "OR", "AND", "MINUS", "COMP")


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


class _Encoder:
    """An encoder: `names` (fixed), `fname` and `source` (plain Python defining fname(s) → (R, n), R = name → set of
    pairs; the same source is executed here, so the interpreted and the compiled atom read the state the same way),
    `sets(s)`, and `__call__(s)` → Relations (cached per state object)."""
    names: tuple = ()
    fname: str = ""
    source: str = ""
    cache_size = 4096

    def _load(self):
        ns: dict = {}
        exec(self.source, ns)                                    # noqa: S102 — the encoder source written here
        self._fn = ns[self.fname]
        self._cache: OrderedDict = OrderedDict()

    def sets(self, state):
        return self._fn(state)

    def _entry(self, state):
        k = id(state)
        hit = self._cache.get(k)
        if hit is not None and hit[0] is state:
            return hit
        R, n = self._fn(state)
        rel = Relations.from_sets(self.names, R, n)
        hit = (state, rel, _Batch([rel]))
        self._cache[k] = hit                                     # the state is kept, so its id is not reused
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return hit

    def __call__(self, state) -> Relations:
        return self._entry(state)[1]

    def batch(self, state) -> _Batch:
        return self._entry(state)[2]


_EGO = '''def __FN__(s):
    r = __R__
    px, py = s["pos"]
    fx, fy = s.get("facing", (0, 1))
    mats = s.get("mats", {})
    objs = s.get("objs", {})
    R = {nm: set() for nm in __NAMES__}
    idx = {}
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            idx[(px + dx, py + dy)] = len(idx) + 1
    for (x, y), i in idx.items():
        j = idx.get((x + 1, y))
        if j is not None:
            R["EAST"].add((i, j))
        j = idx.get((x, y + 1))
        if j is not None:
            R["SOUTH"].add((i, j))
        m = mats.get((x, y))
        if m in __MATS__:
            R["MAT_" + m].add((i, i))
        o = objs.get((x, y))
        if o in __CRTS__:
            R["CRT_" + o].add((i, i))
    R["AT"].add((0, idx[(px, py)]))
    j = idx.get((px + fx, py + fy))
    if j is not None:
        R["FACE"].add((0, j))
    return R, len(idx) + 1
'''


class Egocentric(_Encoder):
    """A grid state seen from the player: node 0 is the player, nodes 1..w² the cells of a w×w window around it (row
    by row from the top left). Relations: EAST, SOUTH (a cell → its right / lower neighbour; west and north are their
    converses), AT (the player → the cell under it), FACE (the player → the faced cell), MAT_<m> and CRT_<k> (diagonal:
    the cell's material, the creature on it). The inventory is not a relation: keep it as flat atoms. The state:
    {"pos": (x, y), "facing": (dx, dy), "mats": {(x, y): material}, "objs": {(x, y): kind}, ...}."""

    def __init__(self, window: int = 3, materials: Sequence[str] = MATERIALS, creatures: Sequence[str] = CREATURES):
        if window < 1 or window % 2 == 0:
            raise ValueError("the window is an odd number of cells (3 or 5)")
        mats, crts = tuple(materials), tuple(creatures)
        self.window = int(window)
        self.names = ("EAST", "SOUTH", "AT", "FACE") + tuple(f"MAT_{m}" for m in mats) + tuple(f"CRT_{c}" for c in crts)
        body = (_EGO.replace("__R__", str(window // 2)).replace("__NAMES__", repr(self.names))
                .replace("__MATS__", repr(mats)).replace("__CRTS__", repr(crts)))
        self.fname = f"_ego_{_h(body)}"
        self.source = body.replace("__FN__", self.fname)
        self._load()


def encode(state, window: int = 3, materials: Sequence[str] = MATERIALS, creatures: Sequence[str] = CREATURES) -> Relations:
    """An egocentric graph of a grid state (see Egocentric): node 0 the player, a 3×3 (or 5×5) window of cells."""
    return Egocentric(window, materials, creatures)(state)


# The path tree, written once and used by both the interpreted and the compiled atoms.
PATHS = '''def _solvi_path_tree(paths, focus=None, names=(), dirs=(), top=None):
    def clean(p):
        return p.rstrip("/") or "/" if p.startswith("/") else p.rstrip("/")

    def parent(q):
        if q == top or q in ("/", ""):
            return None
        i = q.rfind("/")
        if i < 0:
            return None
        return q[:i] if i > 0 else "/"

    isdir = {clean(d) for d in dirs} | {clean(p) for p in paths if p.endswith("/")}
    qs = [clean(p) for p in paths]
    f = clean(focus) if focus is not None else (qs[0] if qs else "/")
    order, seen, par = [], set(), {}
    for q in [f] + qs:
        while q is not None and q not in seen:
            seen.add(q)
            order.append(q)
            p = parent(q)
            par[q] = p
            if p is not None:
                isdir.add(p)
            q = p
    idx = {q: i for i, q in enumerate(order)}
    R = {"PARENT": set(), "SAME_DIR": set(), "IS_DIR": set()}
    R.update({"NAME=" + x: set() for x in names})
    kids = {}
    for q in order:
        p = par[q]
        if p is not None and p in idx:
            R["PARENT"].add((idx[p], idx[q]))
            kids.setdefault(p, []).append(idx[q])
        if q in isdir:
            R["IS_DIR"].add((idx[q], idx[q]))
        base = q.rsplit("/", 1)[-1]
        if "NAME=" + base in R:
            R["NAME=" + base].add((idx[q], idx[q]))
    for ks in kids.values():
        for a in ks:
            for b in ks:
                if a != b:
                    R["SAME_DIR"].add((a, b))
    return R, len(order), order
'''
_ns: dict = {}
exec(PATHS, _ns)                                                 # noqa: S102 — the source above
_path_tree = _ns["_solvi_path_tree"]


def encode_paths(root, focus: str | None = None, names: Iterable[str] | None = None) -> Relations:
    """A path tree as relations: nodes are the paths and every ancestor (node 0: `focus`, default the first path or
    the root), PARENT (a directory → an entry in it), SAME_DIR (two entries of one directory), IS_DIR (diagonal) and
    NAME=x (diagonal: the last component is x; `names` default every name that occurs). `root`: an iterable of '/'
    separated paths (a trailing '/' marks a directory) or a directory on disk, walked (its ancestors left out).
    `.nodes` gives the path of each node."""
    dirs: list[str] = []
    top = None
    if isinstance(root, (str, os.PathLike)):
        top = os.fspath(root).rstrip("/") or "/"
        paths = [top]
        dirs.append(top)
        for d, sub, files in os.walk(top):
            sub.sort()
            for x in sorted(sub):
                paths.append(os.path.join(d, x))
                dirs.append(os.path.join(d, x))
            for x in sorted(files):
                paths.append(os.path.join(d, x))
    else:
        paths = [str(p) for p in root]
    if names is None:
        names = sorted({(p.rstrip("/") or "/").rsplit("/", 1)[-1] for p in paths} - {""})
    names = tuple(names)
    R, n, order = _path_tree(paths, focus, names, dirs, top)
    return Relations.from_sets(("PARENT", "SAME_DIR", "IS_DIR") + tuple(f"NAME={x}" for x in names), R, n, order)


class PathTree(_Encoder):
    """The path a state acts on, as a path tree: node 0 the path (state[key]), then its ancestors; relations PARENT,
    SAME_DIR, IS_DIR, NAME=x for x in `names` (see encode_paths)."""

    def __init__(self, key: str = "path", names: Iterable[str] = ()):
        self.key, self.path_names = str(key), tuple(names)
        self.names = ("PARENT", "SAME_DIR", "IS_DIR") + tuple(f"NAME={x}" for x in self.path_names)
        body = (f"def __FN__(s):\n    p = s[{self.key!r}]\n"
                f"    R, n, _ = _solvi_path_tree([p], p, {self.path_names!r})\n    return R, n\n")
        self.fname = f"_paths_{_h(body)}"
        self.source = PATHS + "\n\n" + body.replace("__FN__", self.fname)
        self._load()


# ───────────────────────────────────────────── relational atoms
def relational_atom(ast, encoder: _Encoder, library: Library | None = None, *, compiled: bool = True) -> Atom:
    """A program as a laws atom: Atom("rel:<program>", fn, feature="rel", code=..., defs=...). fn: the readout of the
    interpreted program on the encoded state (does node 0 relate to anything); code: the same through compiled set code
    (RUNTIME, the encoder's source, the program) — a state's graph is bounded, so complements compile against its
    nodes. compiled=False: no code (the atom then cannot be compiled into a law module)."""
    program = library.expand(ast) if library is not None else ast
    name = "rel:" + show(ast)

    def fn(s, _p=program, _enc=encoder):
        return bool(readout(_eval(_p, _enc.batch(s))[0]))

    if not compiled:
        return Atom(name, fn, feature="rel")
    pname = f"_relprog_{_h(repr(program))}"
    src = compile_ast(program, pname, dense=True)
    return Atom(name, fn, feature="rel", code=f"_rel_row0({pname}(*{encoder.fname}(s)))",
                defs=(RUNTIME, encoder.source, src))


class RelAlgProposer:
    """A laws proposer that invents relational atoms. For one mismatch: the rows of the mismatched transition's own
    group (from `rows`: a Buffer, or a function (lawset, state, action) → transitions) plus the transition itself are
    encoded, the enumerator runs once over them (programs of at most `size` nodes over the base relations `base`,
    default all of the encoder's, the operators `ops`, the `library`), and every behaviour is scored by the support /
    precision of its readout against "the effect predicts this row" for each effect variant; the top `k` (at most
    `per_readout` per readout pattern, each true in the mismatched state) become atoms "rel:<program>" (added to the
    grammar once) and candidates (rel_atom,) + the flat conditions true in the state, of at most k − 1 atoms.
    include_flat: also the flat candidates (as Enumerate). The held-out test of the law set judges, unchanged."""
    name = "relalg"

    def __init__(self, grammar: Grammar, base: Sequence[str] | None = None, k: int = 20, *, encoder: _Encoder | None = None,
                 rows=None, size: int = 4, ops: Sequence[str] = PROPOSER_OPS, library: Library | None = None,
                 per_readout: int = 8, include_flat: bool = True, max_rows: int = 256, max_behaviours: int = 200_000,
                 min_support: int = 1):
        self.grammar = grammar
        self.encoder = encoder if encoder is not None else Egocentric()
        self.base = tuple(base) if base is not None else None
        self.k, self.size, self.ops, self.library = int(k), int(size), tuple(ops), library
        self.rows, self.per_readout, self.include_flat = rows, int(per_readout), bool(include_flat)
        self.max_rows, self.max_behaviours, self.min_support = int(max_rows), int(max_behaviours), int(min_support)
        self.atoms: dict[str, Atom] = {}
        self.last: dict = {}

    def _rows(self, lawset, state, action, observed):
        from ..laws import Transition
        cur = lawset.transition if lawset.transition is not None and lawset.transition.state is state else None
        if cur is None:
            cur = Transition(state, action, observed)
        if self.rows is None:
            got = []
        elif callable(self.rows) and not hasattr(self.rows, "rows"):
            got = list(self.rows(lawset, state, action))
        else:
            got = [r for r in self.rows.rows(action) if lawset.transition is not None and r.group == cur.group]
        got = [r for r in got if r.id != cur.id][-(self.max_rows - 1):] if self.max_rows > 1 else []
        return got + [cur]

    def atom(self, ast) -> Atom:
        """The relational atom of a program, created once and added to the grammar."""
        name = "rel:" + show(ast)
        if name not in self.grammar.by_name:
            a = self.atoms.get(name) or relational_atom(ast, self.encoder, self.library)
            self.atoms[name] = a
            self.grammar.add(a)
        return self.grammar.by_name[name]

    def propose(self, lawset, state, action, observed):
        g = lawset.grammar
        rows = self._rows(lawset, state, action, observed)
        examples = [self.encoder(r.state) for r in rows]
        effects = list(lawset.effects(state, observed))
        ys = [[lawset.apply(e, r.state) == r.observed for r in rows] for e in effects]
        en = Enumerator(_Batch(examples, self.base), self.size, self.library or (), ops=self.ops,
                        max_behaviours=self.max_behaviours)
        tops = rank(en, ys, k=self.k, per_readout=self.per_readout, must=len(rows) - 1, min_support=self.min_support)
        self.last = {"rows": len(rows), "evals": en.evals, "behaviours": en.behaviours, "stopped": en.stopped,
                     "top": [[(show(s.ast), s.score) for s in top] for top in tops]}
        flat_true = [x for x in g.true(state, action) if g.feature[x] != "rel"]
        flat = [c for c in g.conditions(flat_true) if len(c) <= g.k - 1]
        out, seen = [], set()
        for e, top in zip(effects, tops):
            for s in top:
                a = self.atom(s.ast)
                for c in flat:
                    cond = tuple(sorted((a.name,) + c))
                    key = (cond, repr(e))
                    if key not in seen:
                        seen.add(key)
                        out.append(Candidate(action, cond, e, self.name))
        if self.include_flat:
            for e in effects:
                for c in g.conditions(flat_true):
                    key = (c, repr(e))
                    if key not in seen:
                        seen.add(key)
                        out.append(Candidate(action, c, e, "enumerate"))
        return out


__all__ = ["CREATURES", "Egocentric", "MATERIALS", "PATHS", "PROPOSER_OPS", "PathTree", "RelAlgProposer", "encode",
           "encode_paths", "relational_atom"]
