"""Minesweeper with a solvi solver bot that explains every move.

The bot is a solvi catalog over the visible board (it never sees the mines):
  constraints   every revealed number that still touches hidden cells
  deductions    certain facts: single-number deductions ("this 1 already has its mine, so its other neighbours are safe"),
                then pairs of overlapping numbers (subset / "1-2" patterns), repeated until nothing new is found
  risk_map      the chance of a mine in every hidden cell. "exact": the frontier (hidden cells next to numbers) is split into
                independent groups, each group's mine arrangements are enumerated (capped), and the groups are combined with the
                global mine count. "quick": a naive local ratio (mines a number needs / its hidden neighbours).
  certain_moves the deductions, plus cells that are safe (or mines) in every enumerated arrangement
  planned_move  the lowest-risk cell (a certain mine is flagged before any guess)
  HARD check    no_guess_when_certain: a guess is refused while a certain-safe cell is known (the answer is forced to
                "take_certain" and the bot plays the `certain_move` answer instead)
Questions: `move` ("reveal r,c" / "flag r,c", 1-based), `certainty` (certain / guess), `certain_move` (fallback).

Pure logic, no UI: MSGame, new_game, reveal, toggle_flag, bot_decide, bot_step, run_game, benchmark."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import comb

from solvi import Answer, Catalog, Question, System

SIZES = {"9×9 · 10 mines": (9, 9, 10), "16×16 · 40 mines": (16, 16, 40)}
MAX_R = MAX_C = 16
GROUP_CAP = 24          # a frontier group with more cells than this is estimated, not enumerated
NODE_BUDGET = 40_000    # backtracking nodes per group before giving up on enumeration


def nbrs(r, c, R, C):
    return [(a, b) for a in (r - 1, r, r + 1) for b in (c - 1, c, c + 1)
            if (a, b) != (r, c) and 0 <= a < R and 0 <= b < C]


def cid(r, c):
    """inside the catalog a cell is one int, 100·row + column: cheaper to hash in the trace than a (row, column) pair"""
    return r * 100 + c


def rc(x):
    return divmod(x, 100) if isinstance(x, int) else tuple(x)


def name(cell):
    """1-based (row, column) for people"""
    r, c = rc(cell)
    return f"({r + 1},{c + 1})"


def names(cells):
    cells = [name(x) for x in sorted(cells)]
    if len(cells) > 6:
        return ", ".join(cells[:6]) + f" and {len(cells) - 6} more"
    return cells[0] if len(cells) == 1 else ", ".join(cells[:-1]) + " and " + cells[-1]


def _pl(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


# --------------------------------------------------------------------------------------------------------------------
# Solver pieces (plain functions, reused by the catalog)
# --------------------------------------------------------------------------------------------------------------------

def _constraints(view):
    R, C = len(view), len(view[0])
    out = []
    for r in range(R):
        for c in range(C):
            ch = view[r][c]
            if ch in "12345678":
                hid = [cid(a, b) for a, b in nbrs(r, c, R, C) if view[a][b] in "#F"]
                if hid:
                    out.append({"cell": cid(r, c), "n": int(ch), "hidden": hid})
    return out


def _deduce(constraints, max_rounds=60):
    """Fixpoint of single-number and pair deductions. Flags are ignored (the bot does not trust them): mines are known only
    when deduced. Returns a list of {act, cell, rule, evidence, mines_used, why}."""
    mines, safe, out = set(), set(), []

    def add(act, cells, rule, evidence, used, why):
        new = False
        for x in sorted(cells):
            if x in mines or x in safe:
                continue
            (mines if act == "flag" else safe).add(x)
            out.append({"act": act, "cell": x, "rule": rule, "evidence": sorted(evidence), "mines_used": sorted(used),
                        "why": why})
            new = True
        return new

    for _ in range(max_rounds):
        cons = []
        for k in constraints:
            unk = frozenset(x for x in k["hidden"] if x not in mines and x not in safe)
            used = [x for x in k["hidden"] if x in mines]
            if unk:
                cons.append({"cell": k["cell"], "n": k["n"], "need": k["n"] - len(used), "unk": unk, "used": used})
        found = False
        for k in cons:                                               # single-number deductions
            cell, n, need, unk, used = k["cell"], k["n"], k["need"], k["unk"], k["used"]
            if need == 0:
                why = (f"The {n} at {name(cell)} already has {'its mine' if n == 1 else f'all {n} of its mines'}, "
                       f"at {names(used)}, so its other hidden neighbours are safe.")
                found |= add("reveal", unk, "single", [cell], used, why)
            elif need == len(unk):
                have = f"it already has {_pl(len(used), 'mine')} (at {names(used)}), " if used else ""
                why = (f"The {n} at {name(cell)}: {have}it needs {_pl(need, 'more mine') if used else _pl(need, 'mine')}, "
                       f"and exactly {_pl(len(unk), 'hidden cell')} {'is' if len(unk) == 1 else 'are'} left, "
                       f"so {'it is a mine' if len(unk) == 1 else 'they are all mines'}.")
                found |= add("flag", unk, "single", [cell], used, why)
        if found:
            continue
        for a in cons:                                               # pairs of overlapping numbers
            for b in cons:
                if a is b or not (a["unk"] & b["unk"]):
                    continue
                only_b, only_a = b["unk"] - a["unk"], a["unk"] - b["unk"]
                d = b["need"] - a["need"]
                ev, used = [a["cell"], b["cell"]], a["used"] + b["used"]
                an, bn = f"the {a['n']} at {name(a['cell'])}", f"the {b['n']} at {name(b['cell'])}"
                if not only_a and only_b and d == 0:
                    why = (f"All hidden cells of {an}, namely {names(a['unk'])}, also touch {bn}. {an[0].upper() + an[1:]} "
                           f"needs {_pl(a['need'], 'mine')} there, and {bn} needs exactly as many, so {bn}'s other "
                           f"hidden {'cell' if len(only_b) == 1 else 'cells'}, {names(only_b)}, "
                           f"{'is' if len(only_b) == 1 else 'are'} safe.")
                    found |= add("reveal", only_b, "pair", ev, used, why)
                elif only_b and d == len(only_b):
                    why = (f"{bn[0].upper() + bn[1:]} needs {_pl(b['need'], 'mine')}, but at most {a['need']} can sit in the "
                           f"cells it shares with {an}; so its {_pl(len(only_b), 'other cell')}, {names(only_b)}, "
                           f"must {'be a mine' if len(only_b) == 1 else 'all be mines'}.")
                    found |= add("flag", only_b, "pair", ev, used, why)
                    if only_a:
                        why2 = (f"{bn[0].upper() + bn[1:]} forces {a['need']} of its mines into the cells it shares with {an}, "
                                f"which fills {an}; so {an}'s other cells, {names(only_a)}, are safe.")
                        found |= add("reveal", only_a, "pair", ev, used, why2)
        if not found:
            break
    return out


def _components(cons):
    """Split the frontier into groups of cells linked by shared numbers."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k in cons:
        cells = sorted(k["unk"])
        for x in cells:
            parent.setdefault(x, x)
        for x in cells[1:]:
            ra, rb = find(cells[0]), find(x)
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    out = []
    for cells in groups.values():
        cs = set(cells)
        out.append((sorted(cells), [k for k in cons if k["unk"] & cs]))
    return sorted(out, key=lambda g: g[0][0])


class _Budget(Exception):
    pass


def _enumerate(cells, cons):
    """All mine arrangements of one group consistent with its numbers → {k: [ways, per-cell mine counts]}."""
    # order cells so that each constraint closes early (BFS over shared numbers) → strong pruning
    order, seen = [], set()
    by_cell = {}
    for ci, k in enumerate(cons):
        for x in k["unk"]:
            by_cell.setdefault(x, []).append(ci)
    for start in cells:
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        while queue:
            x = queue.pop(0)
            order.append(x)
            for ci in by_cell[x]:
                for y in sorted(cons[ci]["unk"]):
                    if y not in seen:
                        seen.add(y)
                        queue.append(y)
    n = len(order)
    cell_cons = [by_cell[x] for x in order]
    need = [k["need"] for k in cons]
    have = [0] * len(cons)
    left = [len(k["unk"]) for k in cons]
    assign = [0] * n
    counts = {}
    nodes = [0]

    def rec(i, k):
        nodes[0] += 1
        if nodes[0] > NODE_BUDGET:
            raise _Budget
        if i == n:
            e = counts.get(k)
            if e is None:
                e = counts[k] = [0, [0] * n]
            e[0] += 1
            cc = e[1]
            for j in range(n):
                if assign[j]:
                    cc[j] += 1
            return
        for v in (0, 1):
            ok = True
            for ci in cell_cons[i]:
                m = have[ci] + v
                if m > need[ci] or m + left[ci] - 1 < need[ci]:
                    ok = False
                    break
            if not ok:
                continue
            for ci in cell_cons[i]:
                have[ci] += v
                left[ci] -= 1
            assign[i] = v
            rec(i + 1, k + v)
            assign[i] = 0
            for ci in cell_cons[i]:
                have[ci] -= v
                left[ci] += 1
    rec(0, 0)
    pos = {x: j for j, x in enumerate(order)}
    return {k: [w, [cc[pos[x]] for x in cells]] for k, (w, cc) in counts.items()}


def _conv(polys):
    out = {0: 1}
    for p in polys:
        nxt = {}
        for a, wa in out.items():
            for b, wb in p.items():
                nxt[a + b] = nxt.get(a + b, 0) + wa * wb
        out = nxt
    return out


def _local_risk(cons, cells):
    return {x: max(k["need"] / len(k["unk"]) for k in cons if x in k["unk"]) for x in cells}


def _risk(view, constraints, deductions, mines, planner):
    R, C = len(view), len(view[0])
    hidden = [cid(r, c) for r in range(R) for c in range(C) if view[r][c] in "#F"]
    if planner == "quick":
        # naive: every number spreads its mines evenly over its hidden neighbours; no deductions, no enumeration
        cons = [{"need": k["n"], "unk": frozenset(k["hidden"])} for k in constraints]
        front = {x for k in cons for x in k["unk"]}
        p = _local_risk(cons, front)
        interior = mines / len(hidden) if hidden else 0.0
        return {"p": p, "interior": interior, "exact": False, "groups": [], "arrangements": 0,
                "sure_safe": [], "sure_mines": []}
    km = {d["cell"] for d in deductions if d["act"] == "flag"}
    ks = {d["cell"] for d in deductions if d["act"] == "reveal"}
    cons = []
    for k in constraints:
        unk = frozenset(x for x in k["hidden"] if x not in km and x not in ks)
        if unk:
            cons.append({"need": k["n"] - sum(x in km for x in k["hidden"]), "unk": unk, "cell": k["cell"]})
    front = {x for k in cons for x in k["unk"]}
    other = [x for x in hidden if x not in front and x not in km and x not in ks]
    m_rem = mines - len(km)
    p = {x: 0.0 for x in ks}
    p.update({x: 1.0 for x in km})
    exact_groups, approx, sizes, exact = [], [], [], True
    for cells, gcons in _components(cons):
        sizes.append(len(cells))
        dist = None
        if len(cells) <= GROUP_CAP:
            try:
                dist = _enumerate(cells, gcons)
            except _Budget:
                dist = None
        if dist:
            exact_groups.append((cells, dist))
        else:
            approx.append((cells, gcons))
            exact = False
    for cells, gcons in approx:                       # too big to enumerate: local estimate
        p.update(_local_risk(gcons, cells))
    m_eff = m_rem - round(sum(p[x] for cells, _ in approx for x in cells))
    n_other = len(other)
    polys = [{k: e[0] for k, e in dist.items()} for _, dist in exact_groups]
    allc = _conv(polys)
    total = sum(w * comb(n_other, m_eff - k) for k, w in allc.items() if 0 <= m_eff - k <= n_other)
    sure_safe, sure_mines, interior = [], [], 0.0
    if total == 0:                                    # should not happen on a truthful board; fall back to local ratios
        for cells, dist in exact_groups:
            p.update(_local_risk([k for k in cons if k["unk"] & set(cells)], cells))
        interior = m_eff / n_other if n_other else 0.0
        exact = False
    else:
        for gi, (cells, dist) in enumerate(exact_groups):
            rest = _conv(polys[:gi] + polys[gi + 1:])
            num = [0] * len(cells)
            for k, (w, cc) in dist.items():
                weight = sum(wr * comb(n_other, m_eff - k - kr) for kr, wr in rest.items() if 0 <= m_eff - k - kr <= n_other)
                if weight:
                    for j in range(len(cells)):
                        num[j] += cc[j] * weight
            for j, x in enumerate(cells):
                p[x] = num[j] / total
                if num[j] == 0:
                    sure_safe.append(x)
                elif num[j] == total:
                    sure_mines.append(x)
        if n_other:
            inum = sum(w * comb(n_other - 1, m_eff - k - 1) for k, w in allc.items() if 1 <= m_eff - k <= n_other)
            interior = inum / total
    return {"p": p, "interior": interior, "exact": exact, "groups": sizes, "arrangements": total,
            "sure_safe": sorted(sure_safe), "sure_mines": sorted(sure_mines)}


def _certain(deductions, risk_map, constraints):
    """Deductions first, then cells that enumeration shows safe / mined in every arrangement. Safe reveals before flags."""
    known = {d["cell"] for d in deductions}
    extra = []
    for act, cells in (("reveal", risk_map["sure_safe"]), ("flag", risk_map["sure_mines"])):
        for x in cells:
            if x in known:
                continue
            ev = sorted({k["cell"] for k in constraints if x in k["hidden"]})
            what = "has no mine" if act == "reveal" else "has a mine"
            why = (f"No single number or pair settles {name(x)}, so solvi listed every way to place the mines around "
                   f"the numbers {names(ev)}: {name(x)} {what} in every one of them.")
            extra.append({"act": act, "cell": x, "rule": "all arrangements", "evidence": ev, "mines_used": [], "why": why})
    allm = list(deductions) + extra
    return [m for m in allm if m["act"] == "reveal"] + [m for m in allm if m["act"] == "flag"]


def _plan(view, risk_map, certain_moves):
    R, C = len(view), len(view[0])
    inner = risk_map["interior"]
    p = {cid(r, c): inner for r in range(R) for c in range(C) if view[r][c] in "#F"}   # cells away from numbers
    p.update(risk_map["p"])
    rank = {m["cell"]: i for i, m in enumerate(certain_moves) if m["act"] == "reveal"}
    safe = set(rank)
    mines = {m["cell"] for m in certain_moves if m["act"] == "flag"}
    cand = [x for x in p if x not in mines]
    if not cand:
        return {"move": "none", "p": None, "kind": "certain"}

    def key(x):          # lowest risk; ties: explained certain moves first, then cells with fewer hidden neighbours
        hid = sum(view[a][b] in "#F" for a, b in nbrs(*rc(x), R, C))
        return (round(p[x], 12), rank.get(x, 10**6), hid, x)
    best = min(cand, key=key)
    if p[best] == 0 or best in safe:
        return {"move": _mv("reveal", best), "p": p[best], "kind": "certain"}
    unflagged = sorted(x for x in mines if view[x // 100][x % 100] != "F")
    if unflagged:
        return {"move": _mv("flag", unflagged[0]), "p": 1.0, "kind": "certain"}
    return {"move": _mv("reveal", best), "p": p[best], "kind": "guess"}


def _mv(act, cell):
    """answers are 1-based, like the cell names in the explanations: "reveal 4,5" = row 4, column 5"""
    r, c = rc(cell)
    return f"{act} {r + 1},{c + 1}"


def parse_move(s):
    """→ (act, (row, col)) 0-based"""
    act, pos = s.split(" ")
    r, c = pos.split(",")
    return act, (int(r) - 1, int(c) - 1)


# --------------------------------------------------------------------------------------------------------------------
# The solver bot as a solvi catalog
# --------------------------------------------------------------------------------------------------------------------

MOVE_OPTIONS = [_mv(a, (r, c)) for a in ("reveal", "flag") for r in range(MAX_R) for c in range(MAX_C)]


def build_catalog(check=True):
    cat = Catalog()

    @cat.fn
    def constraints(view):
        """every revealed number that still touches hidden cells: its cell, its number, its hidden neighbours"""
        return _constraints(view)

    @cat.fn
    def deductions(constraints):
        """certain facts from single numbers, then from pairs of overlapping numbers, repeated until nothing new"""
        return _deduce(constraints)

    @cat.fn
    def risk_map(view, constraints, deductions, mines, planner):
        """chance of a mine for cells next to numbers (p) and for any cell away from them (interior): exact (frontier groups
        enumerated, combined with the mine count) or quick"""
        return _risk(view, constraints, deductions, mines, planner)

    @cat.fn
    def certain_moves(deductions, risk_map, constraints):
        """deductions plus cells safe (or mined) in every enumerated arrangement; safe reveals first"""
        return _certain(deductions, risk_map, constraints)

    @cat.fn
    def planned_move(view, risk_map, certain_moves):
        """the lowest-risk cell; a known mine is flagged before any guess"""
        return _plan(view, risk_map, certain_moves)

    if check:
        @cat.check(hard=True, then={"move": "take_certain", "certainty": "certain"})
        def no_guess_when_certain(planned_move, certain_moves):
            """hard: a guess is allowed only when no certain-safe cell is known"""
            return planned_move["kind"] == "certain" or not any(m["act"] == "reveal" for m in certain_moves)

    @cat.rule("move")
    def move(planned_move):
        return planned_move["move"]

    @cat.rule("certainty")
    def certainty(planned_move):
        return planned_move["kind"]

    @cat.rule("certain_move")
    def certain_move(certain_moves):
        safe = [m for m in certain_moves if m["act"] == "reveal"]
        return _mv("reveal", safe[0]["cell"]) if safe else "none"

    return cat


def build_system(check=True):
    cat = build_catalog(check)
    qs = [Question("move", "Which cell to reveal or flag?", Answer.choice(MOVE_OPTIONS + ["take_certain", "none"]),
                   checkpoints=["no_guess_when_certain"] if check else []),
          Question("certainty", "Is the move certain or a guess?", Answer.choice(["certain", "guess"]),
                   checkpoints=["no_guess_when_certain"] if check else []),
          Question("certain_move", "A certain-safe cell, if one is known", Answer.choice(MOVE_OPTIONS + ["none"]))]
    return cat, System(cat, qs)


SYSTEMS = {True: build_system(True), False: build_system(False)}   # check on / off → (catalog, system)
QUESTIONS = ["move", "certainty", "certain_move"]

# --------------------------------------------------------------------------------------------------------------------
# Game engine
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class MSGame:
    rows: int = 9
    cols: int = 9
    n_mines: int = 10
    seed: int = 0
    mines: set = field(default_factory=set)           # placed on the first reveal (the first cell always opens an area)
    revealed: set = field(default_factory=set)
    flagged: set = field(default_factory=set)
    over: bool = False
    won: bool = False
    boom: tuple | None = None
    log: list = field(default_factory=list)
    bot_certain: int = 0
    bot_guesses: int = 0
    bot_forced: int = 0

    @property
    def placed(self):
        return bool(self.mines)

    def number(self, cell):
        return sum(x in self.mines for x in nbrs(cell[0], cell[1], self.rows, self.cols))

    def view(self):
        out = []
        for r in range(self.rows):
            s = []
            for c in range(self.cols):
                x = (r, c)
                s.append(str(self.number(x)) if x in self.revealed else "F" if x in self.flagged else "#")
            out.append("".join(s))
        return out

    @property
    def safe_left(self):
        return self.rows * self.cols - self.n_mines - len(self.revealed)


def new_game(size="9×9 · 10 mines", seed=0):
    R, C, M = SIZES[size] if isinstance(size, str) else size
    return MSGame(rows=R, cols=C, n_mines=M, seed=int(seed))


def _place(game, first):
    rng = random.Random(game.seed)
    banned = {first, *nbrs(first[0], first[1], game.rows, game.cols)}
    cells = [(r, c) for r in range(game.rows) for c in range(game.cols) if (r, c) not in banned]
    game.mines = set(rng.sample(cells, game.n_mines))


def reveal(game, cell):
    """→ 'boom' | 'open' | 'won' | 'noop'"""
    if game.over or cell in game.revealed:
        return "noop"
    if not game.placed:
        _place(game, cell)
    game.flagged.discard(cell)
    if cell in game.mines:
        game.over, game.boom = True, cell
        return "boom"
    stack = [cell]
    while stack:
        x = stack.pop()
        if x in game.revealed:
            continue
        game.revealed.add(x)
        game.flagged.discard(x)
        if game.number(x) == 0:
            stack.extend(y for y in nbrs(x[0], x[1], game.rows, game.cols) if y not in game.revealed)
    if game.safe_left == 0:
        game.over, game.won = True, True
        game.flagged = set(game.mines)
        return "won"
    return "open"


def toggle_flag(game, cell):
    if game.over or cell in game.revealed:
        return
    game.flagged ^= {cell}


def solver_state(game, planner="exact"):
    return {"view": game.view(), "mines": game.n_mines, "planner": planner}


def bot_decide(game, planner="exact", check=True):
    """→ (resp, action, cell, info). info: kind (certain/guess), forced (hard check fired), the certain-move record used (or
    None), p (mine chance of the chosen cell), evidence and mines_used cells."""
    cat, system = SYSTEMS[check]
    resp = system.ask(solver_state(game, planner), QUESTIONS)
    ans = resp["move"].answer
    forced = resp["move"].status == "forced"
    if forced:
        ans = resp["certain_move"].answer
    if ans in (None, "none"):
        return resp, None, None, {"kind": "none", "forced": forced}
    act, cell = parse_move(ans)
    x = cid(*cell)
    certain = resp.values.get("certain_moves") or []
    rec = next((m for m in certain if m["cell"] == x and m["act"] == act), None)
    rm = resp.values["risk_map"]
    kind = "certain" if (forced or rec is not None or resp["certainty"].answer == "certain") else "guess"
    ev = rec["evidence"] if rec else sorted({k["cell"] for k in resp.values["constraints"] if x in k["hidden"]})
    info = {"kind": kind, "forced": forced, "rec": rec, "p": rm["p"].get(x, rm["interior"]), "exact": rm["exact"],
            "interior": rm["interior"], "arrangements": rm["arrangements"], "evidence": [rc(y) for y in ev],
            "mines_used": [rc(y) for y in rec["mines_used"]] if rec else [], "planned": resp.values.get("planned_move"),
            "first": not game.placed}
    return resp, act, cell, info


def explain(act, cell, info, planner="exact"):
    """Plain-words reason for a move."""
    if act is None:
        return "Nothing to do: the game is over."
    rec = info.get("rec")
    head = f"{'Reveal' if act == 'reveal' else 'Flag'} {name(cell)}"
    if info.get("forced"):
        pl = info["planned"]
        guess = (f"The {planner} risk map wanted to guess {name(parse_move(pl['move'])[1])} ({pl['p'] * 100:.0f}% mine "
                 f"chance), but the "
                 f"hard check no_guess_when_certain refused it: a certain-safe cell is known. ")
        return head + " — certain. " + guess + (rec["why"] if rec else "")
    if rec is not None:
        return head + " — certain. " + rec["why"]
    if info["kind"] == "certain":
        return head + " — certain: its mine chance is 0%."
    p = info["p"] or 0.0
    if info.get("first"):
        return (f"{head} — the first click. Every cell is equally likely to hide a mine, but the first click is always "
                f"safe here and opens an area. solvi takes a corner (fewest hidden neighbours, most likely to open wide).")
    how = ("exact: counted over every mine arrangement that fits the numbers" if info.get("exact") and planner == "exact"
           else "estimated: a frontier group was too big to enumerate" if planner == "exact"
           else "the quick local estimate: a number's missing mines spread evenly over its hidden neighbours")
    where = (f"it touches the numbers {names(info['evidence'])}" if info["evidence"] else
             "it is away from every number")
    return (f"{head} — a guess. No cell is certain, so solvi picks the lowest mine chance: {p * 100:.1f}% ({how}; {where}). "
            f"A hidden cell away from the numbers: {info['interior'] * 100:.1f}%.")


def assess(game, cell, planner="exact"):
    """solvi's view of a human reveal, computed BEFORE it is played → (text, highlight dict) or (None, None)"""
    if game.over or not game.placed or cell in game.revealed:
        return None, None
    resp = SYSTEMS[True][1].ask(solver_state(game, planner), ["move", "certain_move"])
    x = cid(*cell)
    certain = resp.values["certain_moves"]
    rec = next((m for m in certain if m["cell"] == x), None)
    rm = resp.values["risk_map"]
    p = rm["p"].get(x, rm["interior"])
    safe = [m for m in certain if m["act"] == "reveal"]
    hl = {"target": cell, "evidence": [rc(y) for y in (rec["evidence"] if rec else [])],
          "mines": [rc(y) for y in (rec["mines_used"] if rec else [])], "kind": "certain"}
    if rec is not None and rec["act"] == "reveal":
        return f"Your reveal of {name(cell)} was certain-safe. " + rec["why"], hl
    if rec is not None:
        hl["kind"] = "flag"
        return f"Your reveal of {name(cell)} was a certain mine. " + rec["why"], hl
    hl["kind"] = "guess"
    hl["evidence"] = [rc(k["cell"]) for k in resp.values["constraints"] if x in k["hidden"]]
    if safe:
        alt = safe[0]
        hl["alt"] = rc(alt["cell"])
        return (f"Your reveal of {name(cell)} was a guess ({p * 100:.0f}% mine chance) while "
                f"{_pl(len(safe), 'certain-safe cell')} {'was' if len(safe) == 1 else 'were'} known, for example "
                f"{name(alt['cell'])}: {alt['why']}"), hl
    pl = resp.values["planned_move"]
    return (f"Your reveal of {name(cell)} was a guess ({p * 100:.0f}% mine chance). Nothing was certain; solvi would have "
            f"picked {name(parse_move(pl['move'])[1])} at {pl['p'] * 100:.0f}%."), hl


def bot_step(game, planner="exact", check=True):
    """Decide and play one move. → (resp, act, cell, info, event)"""
    resp, act, cell, info = bot_decide(game, planner, check)
    if act is None:
        return resp, act, cell, info, "noop"
    if act == "flag":
        game.flagged.add(cell)
        ev = "flag"
    else:
        ev = reveal(game, cell)
        if info.get("first"):
            pass
        elif info["kind"] == "certain":
            game.bot_certain += 1
        else:
            game.bot_guesses += 1
    game.bot_forced += bool(info.get("forced"))
    return resp, act, cell, info, ev


def run_game(seed, size="9×9 · 10 mines", planner="exact", check=True, max_moves=2000):
    g = new_game(size, seed)
    wrong = 0
    for _ in range(max_moves):
        if g.over:
            break
        _, act, cell, info, ev = bot_step(g, planner, check)
        if act is None:
            break
        if info["kind"] == "certain" and ((act == "reveal" and cell in g.mines) or (act == "flag" and cell not in g.mines)):
            wrong += 1
    return {"won": g.won, "certain": g.bot_certain, "guesses": g.bot_guesses, "forced": g.bot_forced, "wrong_certain": wrong,
            "first_guess_lost": (not g.won) and g.bot_guesses == 1}


def summarize(games):
    n = len(games)
    reveals = sum(x["certain"] + x["guesses"] for x in games)
    return {"n": n, "win_rate": sum(x["won"] for x in games) / n,
            "certain_pct": sum(x["certain"] for x in games) / max(1, reveals),
            "guesses_per_game": sum(x["guesses"] for x in games) / n,
            "forced_per_game": sum(x["forced"] for x in games) / n,
            "wrong_certain": sum(x["wrong_certain"] for x in games)}


def benchmark(n=100, size="9×9 · 10 mines", configs=(("exact", True), ("quick", True), ("quick", False))):
    return {(pl, ck): summarize([run_game(s, size, pl, ck) for s in range(n)]) for pl, ck in configs}
