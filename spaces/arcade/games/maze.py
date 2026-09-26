"""Ghost maze: a small Pac-Man-like game. The Pac agent is a solvi catalog of small functions over the board
(legal moves, BFS distance to the nearest dot and to the nearest ghost per move, danger per move ...), one scoring rule,
and a HARD check: never step onto a cell a ghost can reach this tick when a safer move exists.

Pure logic, no UI: `new_game`, `agent_decide`, `step`, `run_game`, `benchmark`."""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass, field

from solvi import Answer, Catalog, Question, System

LAYOUT = [
    "#############",
    "#.....#.....#",
    "#.###.#.###.#",
    "#...........#",
    "#.#.##.##.#.#",
    "#.#...G...#.#",
    "#.#.##.##.#.#",
    "#...........#",
    "#.###.#.###.#",
    "#.....P.....#",
    "#############",
]
H, W = len(LAYOUT), len(LAYOUT[0])
WALLS = frozenset((r, c) for r in range(H) for c in range(W) if LAYOUT[r][c] == "#")
PAC_START = next((r, c) for r in range(H) for c in range(W) if LAYOUT[r][c] == "P")
GHOST_HOME = next((r, c) for r in range(H) for c in range(W) if LAYOUT[r][c] == "G")
ALL_DOTS = tuple(sorted((r, c) for r in range(H) for c in range(W) if LAYOUT[r][c] in ".G"))
TOTAL_DOTS = len(ALL_DOTS)      # Pac's start cell has no dot

DIRS = {"UP": (-1, 0), "LEFT": (0, -1), "DOWN": (1, 0), "RIGHT": (0, 1)}
REVERSE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT", None: None}
MOVES = list(DIRS)

# ghost tuning (chosen so the game is fair but not trivial; see benchmark numbers in the README)
GHOSTS = [  # name, color, scatter corner, release tick
    ("Blinky", "#ff4d4d", (1, 11), 0),
    ("Pinky", "#ff8fd8", (1, 1), 8),
    ("Clyde", "#ffb347", (9, 1), 16),
]
SCATTER, CYCLE = 7, 34          # first SCATTER ticks of every CYCLE ticks: ghosts go to their corners
GHOST_NOISE = 0.12              # chance a ghost takes a random turn at a junction
GHOST_SPEED = 0.9               # chance a ghost moves on a given tick
MAX_TICKS = 400
LIVES = 3
DOT_POINTS, CLEAR_BONUS = 10, 200


def step_cell(cell, move):
    dr, dc = DIRS[move]
    return (cell[0] + dr, cell[1] + dc)


def open_neighbors(cell):
    return [(m, step_cell(cell, m)) for m in MOVES if step_cell(cell, m) not in WALLS]


def bfs_dist(start, targets):
    """Shortest path length (through open cells) from start to the nearest cell in targets; 99 if none."""
    targets = set(targets)
    if not targets:
        return 99
    if start in targets:
        return 0
    seen, q = {start}, deque([(start, 0)])
    while q:
        cell, d = q.popleft()
        for _, n in open_neighbors(cell):
            if n not in seen:
                if n in targets:
                    return d + 1
                seen.add(n)
                q.append((n, d + 1))
    return 99


# --------------------------------------------------------------------------------------------------------------------
# The Pac agent as a solvi catalog
# --------------------------------------------------------------------------------------------------------------------

DEFAULT_BOT_CODE = '''\
# The Pac agent's scoring rule. It is an ordinary solvi function:
# its argument names are facts the catalog computes each tick, its name is the fact it sets.
# Facts you can use: pac, ghosts, dots, last_move, legal_moves, next_cells, dot_dist,
#                    ghost_dist, ghost_reach, danger, safe_moves, dots_left
# dot_dist / ghost_dist / danger are dicts {move: value}, moves are "UP", "DOWN", "LEFT", "RIGHT".

W_DOT = 1.0       # pull toward the nearest dot
W_GHOST = 0.6     # push away from the nearest ghost ...
GHOST_CAP = 5     # ... but only while it is closer than this
W_REVERSE = 0.5   # penalty for turning back (stops dithering)

def greedy_move(legal_moves, dot_dist, ghost_dist, last_move):
    back = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}.get(last_move)

    def score(m):
        return (-W_DOT * dot_dist[m]
                + W_GHOST * min(ghost_dist[m], GHOST_CAP)
                - W_REVERSE * (m == back))

    return max(legal_moves, key=score)
'''


def load_bot(code: str = DEFAULT_BOT_CODE):
    """exec bot code and return its greedy_move function (raises on errors)."""
    ns: dict = {}
    exec(compile(code, "<bot>", "exec"), ns)  # noqa: S102 - user code runs in the visitor's own browser, under games/sandbox.py's guard
    f = ns.get("greedy_move")
    if not callable(f):
        raise ValueError("the code must define a function named greedy_move")
    return f


def build_catalog(safety: bool = True, greedy=None):
    cat = Catalog()

    @cat.fn
    def legal_moves(pac):
        """moves that do not bump into a wall"""
        return [m for m, _ in open_neighbors(pac)]

    @cat.fn
    def next_cells(pac, legal_moves):
        return {m: step_cell(pac, m) for m in legal_moves}

    @cat.fn
    def ghost_reach(ghosts):
        """cells a ghost occupies now or can step into this tick"""
        cells = set(ghosts)
        for g in ghosts:
            cells.update(n for _, n in open_neighbors(g))
        return sorted(cells)

    @cat.fn
    def dot_dist(dots, next_cells):
        """per move: BFS distance from the next cell to the nearest dot (0 = eats a dot)"""
        return {m: bfs_dist(c, dots) for m, c in next_cells.items()}

    @cat.fn
    def ghost_dist(ghosts, next_cells):
        """per move: BFS distance from the next cell to the nearest ghost"""
        return {m: bfs_dist(c, ghosts) for m, c in next_cells.items()}

    @cat.fn
    def danger(next_cells, ghost_reach):
        """per move: True if a ghost can be on that cell this tick"""
        reach = set(ghost_reach)
        return {m: c in reach for m, c in next_cells.items()}

    @cat.fn
    def safe_moves(legal_moves, danger):
        return [m for m in legal_moves if not danger[m]]

    @cat.fn
    def dots_left(dots):
        return len(dots)

    cat.fn(greedy or load_bot())

    @cat.fn
    def evade_move(safe_moves, legal_moves, dot_dist, ghost_dist):
        """the best safe move: closest dot among safe moves, ties -> farther from ghosts; none safe -> farthest from ghosts"""
        if safe_moves:
            return min(safe_moves, key=lambda m: (dot_dist[m], -ghost_dist[m]))
        return max(legal_moves, key=lambda m: ghost_dist[m])

    if safety:
        @cat.check(hard=True, then={"move": "EVADE"})
        def greedy_is_safe(greedy_move, danger, safe_moves):
            """hard: the scored move may step into ghost reach only if every move does"""
            return not danger.get(greedy_move, False) or not safe_moves

    @cat.rule("move")
    def move(greedy_move):
        return greedy_move

    @cat.rule("fallback")
    def fallback(evade_move):
        return evade_move

    return cat


def build_system(safety: bool = True, greedy=None):
    cat = build_catalog(safety, greedy)
    qs = [Question("move", "Which way should Pac go?", Answer.choice(MOVES + ["EVADE"]),
                   checkpoints=["greedy_is_safe"] if safety else []),
          Question("fallback", "Safest good move if the scored move is refused", Answer.choice(MOVES))]
    return cat, System(cat, qs)


# --------------------------------------------------------------------------------------------------------------------
# Game engine
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class Ghost:
    name: str
    color: str
    corner: tuple
    release: int
    pos: tuple = GHOST_HOME
    dir: str | None = None


@dataclass
class MazeGame:
    seed: int = 0
    pac: tuple = PAC_START
    last_move: str | None = None
    ghosts: list = field(default_factory=list)
    dots: set = field(default_factory=lambda: set(ALL_DOTS))
    score: int = 0
    lives: int = LIVES
    tick: int = 0
    life_tick: int = 0
    over: bool = False
    won: bool = False
    deaths: int = 0
    forced: int = 0
    log: list = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    def state(self):
        """init_state for the agent"""
        return {"pac": self.pac, "ghosts": [g.pos for g in self.ghosts], "dots": sorted(self.dots),
                "last_move": self.last_move}


def new_game(seed: int = 0) -> MazeGame:
    g = MazeGame(seed=seed, rng=random.Random(seed))
    g.ghosts = [Ghost(n, col, corner, rel) for n, col, corner, rel in GHOSTS]
    return g


def _ghost_target(game, gh):
    if game.life_tick % CYCLE < SCATTER:
        return gh.corner
    p = game.pac
    if gh.name == "Pinky" and game.last_move:
        dr, dc = DIRS[game.last_move]
        return (p[0] + 2 * dr, p[1] + 2 * dc)
    if gh.name == "Clyde" and abs(gh.pos[0] - p[0]) + abs(gh.pos[1] - p[1]) <= 4:
        return gh.corner
    return p


def _move_ghosts(game):
    for gh in game.ghosts:
        if game.life_tick < gh.release or game.rng.random() > GHOST_SPEED:
            continue
        opts = [(m, c) for m, c in open_neighbors(gh.pos) if m != REVERSE[gh.dir]] or open_neighbors(gh.pos)
        if game.rng.random() < GHOST_NOISE:
            m, c = game.rng.choice(opts)
        else:
            t = _ghost_target(game, gh)
            m, c = min(opts, key=lambda mc: (mc[1][0] - t[0]) ** 2 + (mc[1][1] - t[1]) ** 2)
        gh.pos, gh.dir = c, m


def _caught(game):
    return any(gh.pos == game.pac for gh in game.ghosts)


def _lose_life(game):
    game.lives -= 1
    game.deaths += 1
    game.pac, game.last_move, game.life_tick = PAC_START, None, 0
    for gh in game.ghosts:
        gh.pos, gh.dir = GHOST_HOME, None
    if game.lives <= 0:
        game.over = True


def agent_decide(system, game, safety=True):
    """Ask the solvi agent. Returns (move, response, note) where note says whether the hard check fired."""
    names = ["move", "fallback"] if safety else ["move"]
    resp = system.ask(game.state(), names)
    r = resp["move"]
    if r.status == "forced":
        return resp["fallback"].answer, resp, "forced"
    if r.status == "abstain" or r.answer is None:
        legal = [m for m, _ in open_neighbors(game.pac)]
        return legal[0], resp, "abstain"
    return r.answer, resp, "ok"


def describe(move, resp, note):
    """one log line: 'move LEFT — nearest dot 3 away, ghost 5 away (safe)'"""
    v = resp.values
    dd, gd, dg = v.get("dot_dist", {}), v.get("ghost_dist", {}), v.get("danger", {})
    risky = dg[move] if move in dg else gd.get(move, 99) <= 1      # without the check, danger is not computed: same test
    tail = f"nearest dot {dd.get(move, '?')} away, ghost {gd.get(move, '?')} away ({'DANGER' if risky else 'safe'})"
    if note == "forced":
        want = v.get("greedy_move")
        return f"move {move} — hard check refused {want} (ghost could reach it); {tail}"
    if note == "abstain":
        return f"move {move} — rule abstained ({resp['move'].why[:60]}); {tail}"
    return f"move {move} — {tail}"


def step(game: MazeGame, move: str, note: str = "") -> str:
    """Advance one tick with Pac playing `move`. Returns an event string ('' / 'dot' / 'caught' / 'won' / 'over')."""
    if game.over:
        return "over"
    game.tick += 1
    game.life_tick += 1
    if note == "forced":
        game.forced += 1
    nxt = step_cell(game.pac, move)
    if nxt not in WALLS:
        game.pac = nxt
        game.last_move = move
    event = ""
    if game.pac in game.dots:
        game.dots.discard(game.pac)
        game.score += DOT_POINTS
        event = "dot"
    if not game.dots:
        game.score += CLEAR_BONUS
        game.over = game.won = True
        return "won"
    if _caught(game):
        _lose_life(game)
        return "over" if game.over else "caught"
    _move_ghosts(game)
    if _caught(game):
        _lose_life(game)
        return "over" if game.over else "caught"
    if game.tick >= MAX_TICKS:
        game.over = True
        return "over"
    return event


def run_game(system, seed, safety=True, max_ticks=MAX_TICKS):
    g = new_game(seed)
    while not g.over and g.tick < max_ticks:
        mv, _, note = agent_decide(system, g, safety)
        step(g, mv, note)
    return {"won": g.won, "score": g.score, "dots": TOTAL_DOTS - len(g.dots), "deaths": g.deaths,
            "ticks": g.tick, "forced": g.forced}


def summarize(results):
    n = len(results)
    return {"games": n,
            "win_rate": sum(r["won"] for r in results) / n,
            "avg_score": sum(r["score"] for r in results) / n,
            "avg_dots_pct": sum(r["dots"] for r in results) / n / TOTAL_DOTS,
            "avg_deaths": sum(r["deaths"] for r in results) / n,
            "avg_forced": sum(r["forced"] for r in results) / n}


def benchmark(n=100, greedy=None, seeds_from=0):
    """n seeded games with and without the hard safety check (same seeds, same scoring rule)."""
    out = {}
    for safety in (True, False):
        _, sysm = build_system(safety, greedy)
        out["with_check" if safety else "without_check"] = summarize(
            [run_game(sysm, seeds_from + s, safety) for s in range(n)])
    return out


def snapshot(game):
    return copy.deepcopy(game)
