"""Tic-tac-toe vs solvi. The catalog is a vendored copy of solvi/examples/05_tic_tac_toe.py (games/_ttt_catalog.py).
We add one more rule to it, "reason", which names why the move was chosen (win, block, fork ...), so every agent move
comes with a readable label on top of the rule inputs."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from solvi import Answer, Question, System

from . import _ttt_catalog as _mod

SOURCE = "solvi/examples/05_tic_tac_toe.py"
cat = _mod.cat
LINES = _mod.LINES
winner_of = _mod._winner

REASONS = {
    "win": "completes a line and wins",
    "block": "blocks your line",
    "fork": "creates two threats at once (a fork)",
    "block_fork": "takes your only fork cell",
    "forcing": "makes a threat you must answer, so you cannot build your fork",
    "center": "takes the center (nothing urgent on the board)",
    "corner": "takes a corner (nothing urgent on the board)",
    "edge": "takes an edge (nothing urgent on the board)",
    "none": "no move",
}


@cat.rule("reason")
def move_reason(winning_moves, blocking_moves, fork_moves, opponent_forks, forcing_moves, free_cells):
    """mirrors pick_move's order and names the branch that decided"""
    for label, group in (("win", winning_moves), ("block", blocking_moves), ("fork", fork_moves)):
        if group:
            return label
    if len(opponent_forks) == 1:
        return "block_fork"
    if opponent_forks and forcing_moves:
        return "forcing"
    for c in (4, 0, 2, 6, 8, 1, 3, 5, 7):
        if c in free_cells:
            return "center" if c == 4 else "corner" if c in (0, 2, 6, 8) else "edge"
    return "none"


QUESTIONS = list(_mod.QUESTIONS) + [
    Question("reason", "Why this move?", Answer.choice(list(REASONS)), requires=["board_valid"])]
system = System(cat, QUESTIONS)


@dataclass
class TTTGame:
    human: str = "X"
    board: str = "." * 9
    over: bool = False
    result: str = ""                 # "agent" | "human" | "draw"
    last_agent_cell: int | None = None
    counted: bool = False            # already added to the win/loss counters
    history: list = field(default_factory=list)

    @property
    def agent(self):
        return "O" if self.human == "X" else "X"

    @property
    def turn(self):
        return "X" if self.board.count("X") == self.board.count("O") else "O"


def _finish(game):
    w = winner_of(game.board)
    if w or "." not in game.board:
        game.over = True
        game.result = "draw" if not w else ("agent" if w == game.agent else "human")


def agent_move(game):
    """The agent plays one move. Returns the solvi Response (or None if it was not the agent's turn)."""
    if game.over or game.turn != game.agent:
        return None
    resp = system.ask({"board": game.board, "me": game.agent}, ["move", "reason"])
    ans = resp["move"].answer
    if ans is None or ans == "none":
        return resp
    c = int(ans)
    game.board = game.board[:c] + game.agent + game.board[c + 1:]
    game.last_agent_cell = c
    game.history.append((game.agent, c, resp["reason"].answer))
    _finish(game)
    return resp


def human_move(game, cell):
    """Returns an error text, or '' when the move was played."""
    if game.over:
        return "The game is over. Press New game."
    if game.turn != game.human:
        return "Wait for the agent."
    if game.board[cell] != ".":
        return "That cell is taken."
    game.board = game.board[:cell] + game.human + game.board[cell + 1:]
    game.history.append((game.human, cell, None))
    _finish(game)
    return ""


def new_game(human="X"):
    g = TTTGame(human=human)
    resp = agent_move(g) if g.agent == "X" else None
    return g, resp


def explain_position(board, me):
    """All facts for `me` on this board (the 'explain the position' view)."""
    return system.ask({"board": board, "me": me})


def vs_random(n=500, seed=0):
    """n games against a uniformly random opponent, agent side alternating at random. Returns counts."""
    rng = random.Random(seed)
    out = {"agent": 0, "draw": 0, "human": 0}
    for _ in range(n):
        g, _ = new_game(rng.choice("XO"))
        while not g.over:
            if g.turn == g.human:
                human_move(g, rng.choice([i for i in range(9) if g.board[i] == "."]))
            else:
                agent_move(g)
        out[g.result] += 1
    return out
