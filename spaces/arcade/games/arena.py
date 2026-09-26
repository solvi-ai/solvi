"""Bot arena: a tournament of ghost-maze bots on the same seeded games. Pure logic, no UI.

Every bot is a solvi catalog: the maze catalog from games/maze.py (legal_moves, BFS dot_dist / ghost_dist, ghost_reach,
danger, safe_moves ...) with the bot's own scoring function `greedy_move` plugged in. A bot either keeps the hard check
`greedy_is_safe` or plays without it; with `enforce=True` every bot gets the check. The check lives in the catalog, outside
the bot's own logic, so it lifts weak bots without touching their code.

Built-in bots are ordinary Python functions. Visitor bots are code strings run through games/sandbox.py's guard; a bot
that loops forever is stopped and marked as crashed, and the tournament goes on.

Use: `Tournament(entries, seeds, enforce).steps()` yields progress after every game; then `.results` holds a BotResult per
bot. `head_to_head`, `death_report`, `state_at` and `hall_of_fame_update` work on those results."""
from __future__ import annotations

import hashlib
import inspect
import random
import sys
import time
import zlib
from dataclasses import dataclass, field

from . import maze, sandbox

N_GAMES = 10
GAME_DEADLINE_S = 20.0          # wall-clock limit for ONE game of a visitor bot (the guard's clock restarts per game)
MAX_USER_BOTS = 3
BACK = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


# --------------------------------------------------------------------------------------------------------------------
# Built-in bots: each is one scoring function named greedy_move (argument names = facts from the maze catalog)
# --------------------------------------------------------------------------------------------------------------------

def _greedy():
    def greedy_move(legal_moves, dot_dist, last_move):
        """nearest dot, nothing else (ties: do not turn back)"""
        back = BACK.get(last_move)
        return min(legal_moves, key=lambda m: (dot_dist[m], m == back))
    return greedy_move


def _coward():
    def greedy_move(legal_moves, ghost_dist, dot_dist, last_move):
        """as far from the ghosts as possible (up to 8 cells); only then the nearest dot"""
        back = BACK.get(last_move)
        return max(legal_moves, key=lambda m: (min(ghost_dist[m], 8), -dot_dist[m], m != back))
    return greedy_move


def _center_dist(cell):
    return abs(cell[0] - maze.GHOST_HOME[0]) + abs(cell[1] - maze.GHOST_HOME[1])


def _cornerer():
    def greedy_move(legal_moves, next_cells, dots, ghost_dist, last_move):
        """clears the maze outside-in: heads for the dots farthest from the ghost house; never steps ONTO a ghost, nothing more"""
        back = BACK.get(last_move)
        far = max(_center_dist(d) for d in dots)
        targets = [d for d in dots if _center_dist(d) >= far - 1]
        return min(legal_moves, key=lambda m: (ghost_dist[m] == 0, maze.bfs_dist(next_cells[m], targets), m == back))
    return greedy_move


def _random_safe():
    def greedy_move(legal_moves, safe_moves, pac, ghosts, dots):
        """a random move among the safe ones (seeded by the position, so a replay is identical)"""
        rng = random.Random(zlib.crc32(repr((pac, ghosts, len(dots))).encode()))
        return rng.choice(safe_moves or legal_moves)
    return greedy_move


def _speedrunner():
    def greedy_move(legal_moves, next_cells, dot_dist, ghost_dist, dots, last_move):
        """nearest dot, drawn to dense clusters, no turning back; takes a ghost 1 cell away as a small cost, so it gambles"""
        back = BACK.get(last_move)
        dset = set(dots)

        def cluster(c):
            return sum((c[0] + dr, c[1] + dc) in dset for dr in range(-2, 3) for dc in range(-2, 3))

        def score(m):
            return -dot_dist[m] + 0.15 * cluster(next_cells[m]) - 1.5 * (ghost_dist[m] <= 1) - 0.8 * (m == back)
        return max(legal_moves, key=score)
    return greedy_move


def _solvi_default():
    return maze.load_bot(maze.DEFAULT_BOT_CODE)


@dataclass
class Entry:
    """A bot in the tournament."""
    key: str
    name: str
    icon: str
    blurb: str
    own_check: bool                 # plays with greedy_is_safe on its own
    make: object = None             # built-in: () -> greedy_move
    code: str | None = None         # visitor bot: source code
    kind: str = "builtin"           # builtin | user

    def ident(self):
        src = self.code if self.code is not None else self.key
        return f"{self.key}:{self.name}:" + hashlib.sha1(src.encode()).hexdigest()[:10] + f":{int(self.own_check)}"


ROSTER = [
    Entry("default", "solvi default", "🟣", "the maze tab's scoring rule (dots, ghosts, no dithering) + the hard check "
          "greedy_is_safe", True, _solvi_default),
    Entry("greedy", "Greedy", "🍒", "nearest dot, nothing else", False, _greedy),
    Entry("coward", "Coward", "🐔", "keeps as far from the ghosts as it can; eats only when nothing is near", False, _coward),
    Entry("cornerer", "Cornerer", "📐", "clears the maze outside-in, corners first; only avoids stepping right onto a ghost", False,
          _cornerer),
    Entry("random", "Random-safe", "🎲", "a random move among the safe ones; no plan at all", False, _random_safe),
    Entry("speed", "Speedrunner", "⚡", "nearest dot, drawn to dense clusters, hates turning back; gambles near ghosts", False,
          _speedrunner),
]


class ArenaGuard(sandbox.Guard):
    """sandbox.Guard, plus opcode tracing on Python < 3.12, where a one-line loop (`while True: pass`) emits no line events
    and would slip past a line counter. Pyodide (Python 3.12+) needs only line events, like the sandbox tab."""
    OPCODES = sys.version_info < (3, 12)

    def _global(self, frame, event, arg):
        if frame.f_code.co_filename != "<bot>":
            return None
        if self.OPCODES:
            frame.f_trace_opcodes = True
        return self._local

    def _local(self, frame, event, arg):
        if event == "opcode":
            event = "line"
        return super()._local(frame, event, arg)


def user_entry(slot: int, name: str, code: str, own_check: bool) -> Entry:
    name = (name or "").strip()[:24] or f"Your bot {slot}"
    return Entry(f"user{slot}", name, "🧑‍💻", "your code, run in your browser under the guard", bool(own_check),
                 code=code or "", kind="user")


# templates for the visitor's editors (same signature as the "Build your own bot" tab)
USER_TEMPLATES = {
    "the default rule, braver": maze.DEFAULT_BOT_CODE.replace("W_GHOST = 0.6", "W_GHOST = 0.3").replace(
        "GHOST_CAP = 5", "GHOST_CAP = 3"),
    "danger-aware": '''\
# Uses the catalog's own facts: safe_moves (moves no ghost can reach this tick) and danger.
def greedy_move(legal_moves, safe_moves, dot_dist, ghost_dist, last_move):
    back = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}.get(last_move)
    pool = safe_moves or legal_moves
    return min(pool, key=lambda m: (dot_dist[m] - 0.3 * min(ghost_dist[m], 4), m == back))
''',
    "ghost-blind": '''\
# Ignores the ghosts completely. Weak alone; try it with the hard check on.
def greedy_move(legal_moves, dot_dist):
    return min(legal_moves, key=lambda m: dot_dist[m])
''',
    "infinite loop (see the guard)": '''\
# This bot never returns: the guard stops it and the arena marks it as crashed.
def greedy_move(legal_moves, dot_dist):
    n = 0
    while True:
        n += 1
''',
    "the default rule": maze.DEFAULT_BOT_CODE,
}


# --------------------------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class GameRecord:
    seed: int
    score: int = 0
    won: bool = False
    deaths: int = 0
    ticks: int = 0
    dots: int = 0
    vetoes: int = 0                 # the hard check refused the bot's move
    unsafe: int = 0                 # moves into ghost reach while a safe move existed
    abstains: int = 0               # the bot's function failed or returned an invalid move
    ms: float = 0.0                 # total decision time
    moves: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    lines: list = field(default_factory=list)          # the "why" line of every tick
    death_info: list = field(default_factory=list)     # one dict per life lost


@dataclass
class BotResult:
    entry: Entry
    check_on: bool
    status: str = "ok"              # ok | crashed | error
    error: str = ""
    games: list = field(default_factory=list)
    crashed_seed: int | None = None

    @property
    def name(self):
        return self.entry.name

    def summary(self):
        gs = self.games
        n = len(gs) or 1
        decisions = sum(g.ticks for g in gs) or 1
        return {"games": len(gs), "avg_score": sum(g.score for g in gs) / n, "total_score": sum(g.score for g in gs),
                "cleared": sum(g.won for g in gs), "lives_lost": sum(g.deaths for g in gs),
                "avg_dots_pct": sum(g.dots for g in gs) / n / maze.TOTAL_DOTS,
                "ms_per_decision": sum(g.ms for g in gs) / decisions, "vetoes": sum(g.vetoes for g in gs),
                "unsafe": sum(g.unsafe for g in gs), "abstains": sum(g.abstains for g in gs)}


def _reach(ghosts):
    cells = set(ghosts)
    for g in ghosts:
        cells.update(n for _, n in maze.open_neighbors(g))
    return cells


def _inputs_line(fn_inputs, values, move):
    """the bot's own inputs for the move it wanted: 'dot_dist=2, ghost_dist=4'"""
    out = []
    for name in fn_inputs:
        v = values.get(name)
        if isinstance(v, dict) and move in v:
            out.append(f"{name}={v[move]}")
        elif name == "safe_moves" and isinstance(v, list):
            out.append(f"safe_moves={'/'.join(v) or 'none'}")
    return ", ".join(out[:3]) or "(no per-move facts)"


def play_game(system, fn_inputs, check_on: bool, seed: int, guard=None) -> GameRecord:
    """One seeded game with full records. A BotStopped from the guard propagates (the caller marks the bot crashed)."""
    g = maze.new_game(seed)
    rec = GameRecord(seed=seed)
    if guard is not None:
        guard.t_end = time.perf_counter() + GAME_DEADLINE_S
    while not g.over:
        ghosts = [gh.pos for gh in g.ghosts]
        reach = _reach(ghosts)
        legal = [m for m, _ in maze.open_neighbors(g.pac)]
        safe = [m for m in legal if maze.step_cell(g.pac, m) not in reach]
        gd_now = maze.bfs_dist(g.pac, ghosts)
        t0 = time.perf_counter()
        mv, resp, note = maze.agent_decide(system, g, check_on)
        rec.ms += (time.perf_counter() - t0) * 1000
        wanted = resp.values.get("greedy_move")
        if mv not in maze.DIRS:                 # e.g. "EVADE" returned by the bot itself: not a move
            mv, note = legal[0], "abstain"
        if note == "abstain":
            rec.abstains += 1
        if note == "forced":
            rec.vetoes += 1
        nxt = maze.step_cell(g.pac, mv)
        risky = nxt in reach
        if risky and safe:
            rec.unsafe += 1
        gd_next = maze.bfs_dist(nxt, ghosts) if nxt not in maze.WALLS else gd_now
        tail = f"ghost {gd_next} away ({'DANGER' if risky else 'safe'})"
        if note == "forced":
            line = f"{mv} — hard check greedy_is_safe refused {wanted} (a ghost could reach it), evade_move picked {mv}; {tail}"
        elif note == "abstain":
            err = [r.error for r in resp.trace.records if r.name == "greedy_move" and r.error]
            why = err[0] if err else (f"returned {wanted!r}" if wanted is not None else resp["move"].why)
            line = f"{mv} — the bot's rule failed ({why[:70]}), took the first legal move; {tail}"
        else:
            line = f"{mv} — rule inputs {_inputs_line(fn_inputs, resp.values, mv)}; {tail}"
        deaths_before = g.deaths
        tick = g.tick + 1
        ev = maze.step(g, mv, note)
        rec.moves.append(mv)
        rec.notes.append(note)
        if g.deaths > deaths_before:
            line += "  ✖ caught"
            rec.death_info.append({"seed": seed, "tick": tick, "move": mv, "wanted": wanted, "note": note,
                                   "ghost_dist_before": gd_now, "ghost_dist_after": gd_next, "risky": risky,
                                   "safe_moves": safe, "preventable": risky and bool(safe)})
        elif ev == "won":
            line += "  ★ all dots eaten"
        rec.lines.append(line)
    rec.score, rec.won, rec.deaths, rec.ticks = g.score, g.won, g.deaths, g.tick
    rec.dots = maze.TOTAL_DOTS - len(g.dots)
    return rec


class Tournament:
    """Runs every entry on the same seeds. `steps()` is a generator: one progress dict per finished game.
    `cache` (optional dict) reuses results of an identical bot+check+seeds run, so toggling `enforce` back is instant."""

    def __init__(self, entries, seeds, enforce=False, cache=None):
        self.entries, self.seeds, self.enforce = list(entries), list(seeds), bool(enforce)
        self.cache = cache if cache is not None else {}
        self.results: list[BotResult] = []
        self.seconds = 0.0

    def cache_key(self, entry, check_on):
        return (entry.ident(), bool(check_on), tuple(self.seeds))

    def total(self):
        return len(self.entries) * len(self.seeds)

    def steps(self):
        t0 = time.perf_counter()
        done, total = 0, self.total()
        for entry in self.entries:
            check_on = entry.own_check or self.enforce
            key = self.cache_key(entry, check_on)
            if key in self.cache:
                self.results.append(self.cache[key])
                done += len(self.seeds)
                yield {"done": done, "total": total, "bot": entry.name, "cached": True}
                continue
            res = BotResult(entry, check_on)
            guard = None
            try:
                if entry.kind == "user":
                    if len(entry.code) > sandbox.MAX_CODE:
                        raise ValueError(f"code is {len(entry.code)} characters; the limit is {sandbox.MAX_CODE}")
                    guard = ArenaGuard(deadline_s=GAME_DEADLINE_S)
                    fn = sandbox.load_guarded(entry.code, guard)
                else:
                    fn = entry.make()
                fn_inputs = list(inspect.signature(fn).parameters)
                _, system = maze.build_system(check_on, fn)
            except sandbox.BotStopped as e:
                res.status, res.error = "crashed", str(e)
            except (KeyboardInterrupt, SystemExit) as e:
                res.status, res.error = "error", f"{type(e).__name__}: {e}"
            except Exception as e:  # noqa: BLE001 - syntax errors, no greedy_move, unknown facts ...
                res.status, res.error = "error", sandbox._error_tail(e)
            if res.status != "ok":
                self.results.append(res)
                done += len(self.seeds)
                yield {"done": done, "total": total, "bot": entry.name, "status": res.status}
                continue
            for seed in self.seeds:
                try:
                    res.games.append(play_game(system, fn_inputs, check_on, seed, guard))
                except sandbox.BotStopped as e:
                    res.status, res.error, res.crashed_seed = "crashed", str(e), seed
                except (KeyboardInterrupt, SystemExit) as e:
                    res.status, res.error, res.crashed_seed = "crashed", f"{type(e).__name__}: {e}", seed
                done += 1
                if res.status != "ok":
                    done += len(self.seeds) - len(res.games) - 1
                    yield {"done": done, "total": total, "bot": entry.name, "seed": seed, "status": res.status}
                    break
                yield {"done": done, "total": total, "bot": entry.name, "seed": seed, "status": "ok"}
            self.results.append(res)
            if res.status == "ok":
                self.cache[key] = res
        self.seconds = time.perf_counter() - t0

    def run(self):
        for _ in self.steps():
            pass
        return self.results


def ranked(results, key="avg_score"):
    """Leaderboard order: bots that finished, by `key` (lives lost and decision time: lower is better); then crashed bots."""
    low_better = key in ("lives_lost", "ms_per_decision")

    def k(r):
        s = r.summary()
        v = s[key]
        return (r.status != "ok", v if low_better else -v, -s["avg_score"])
    return sorted(results, key=k)


def head_to_head(results):
    """{(a, b): (wins of a, wins of b, ties)} over the seeds both bots finished; a, b are bot names."""
    out = {}
    by = [(r.name, {g.seed: g.score for g in r.games}) for r in results]
    for a, sa in by:
        for b, sb in by:
            if a == b:
                continue
            common = sorted(set(sa) & set(sb))
            w = sum(sa[s] > sb[s] for s in common)
            lz = sum(sa[s] < sb[s] for s in common)
            out[(a, b)] = (w, lz, len(common) - w - lz)
    return out


def death_report(res: BotResult) -> dict:
    """Why did the bot lose lives: every death with the ghost distance and whether greedy_is_safe would have refused the move."""
    deaths = [d for g in res.games for d in g.death_info]
    prevent = [d for d in deaths if d["preventable"]]
    trapped = [d for d in deaths if not d["safe_moves"]]
    other = [d for d in deaths if d not in prevent and d not in trapped]
    lost = [g for g in res.games if not g.won]
    timeouts = [g for g in lost if g.deaths < maze.LIVES]
    return {"deaths": deaths, "preventable": len(prevent), "trapped": len(trapped), "other": len(other),
            "games_lost": len(lost), "timeouts": len(timeouts),
            "avg_ghost_dist": (sum(d["ghost_dist_before"] for d in deaths) / len(deaths)) if deaths else None,
            "abstains": sum(g.abstains for g in res.games), "unsafe": sum(g.unsafe for g in res.games),
            "vetoes": sum(g.vetoes for g in res.games)}


def state_at(seed, moves, notes, t):
    """The game after the first t ticks of a recorded match (the engine is deterministic for a seed and a move list)."""
    g = maze.new_game(seed)
    for mv, note in zip(moves[:t], notes[:t]):
        maze.step(g, mv, note)
    return g


def hall_of_fame_update(hof, results, seeds, enforce, keep=5):
    """hof: {"entries": [...]} (browser storage). Adds this run's visitor bots that finished; keeps the best `keep`."""
    hof = dict(hof or {})
    entries = list(hof.get("entries") or [])
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for r in results:
        if r.entry.kind != "user" or r.status != "ok" or not r.games:
            continue
        s = r.summary()
        entries.append({"name": r.name, "avg_score": round(s["avg_score"], 1), "cleared": s["cleared"],
                        "games": s["games"], "check": "on" if r.check_on else "off", "enforced": bool(enforce),
                        "seeds": f"{seeds[0]}–{seeds[-1]}", "when": stamp})
    uniq = {}                                   # the same bot re-run on the same seeds counts once (keep the newest)
    for e in entries:
        uniq[(e["name"], e["avg_score"], e["cleared"], e["check"], e["seeds"])] = e
    entries = sorted(uniq.values(), key=lambda e: -e["avg_score"])
    hof["entries"] = entries[:keep]
    return hof
