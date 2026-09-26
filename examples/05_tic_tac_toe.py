"""Games: a tic-tac-toe agent built from small functions over the board. Each move is an answer with the reason it was chosen
(win now, block, fork, center, ...), and a hard check refuses to play on an invalid board.

Run:  uv run python examples/05_tic_tac_toe.py
The agent plays 500 games against a random opponent, then every possible opponent line as X and as O — and never loses."""
from __future__ import annotations

import random

from solvi import Answer, Catalog, Question, System
from solvi.show import show

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
CELLS = [str(i) for i in range(9)]

cat = Catalog()


def _wins(board, who):
    return [c for c in range(9) if board[c] == "." and any(c in ln and sum(board[i] == who for i in ln) == 2 and
                                                           all(board[i] in (who, ".") for i in ln) for ln in LINES)]


def _winner(board):
    for a, b, c in LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None


@cat.fn
def opponent(me):
    return "O" if me == "X" else "X"


@cat.fn
def free_cells(board):
    return [c for c in range(9) if board[c] == "."]


@cat.fn
def winning_moves(board, me):
    """cells where `me` completes a line right now"""
    return _wins(board, me)


@cat.fn
def blocking_moves(board, opponent):
    """cells where the opponent would complete a line next move"""
    return _wins(board, opponent)


@cat.fn
def fork_moves(board, me, free_cells):
    """cells that create two winning threats at once"""
    return [c for c in free_cells if len(_wins(board[:c] + me + board[c + 1:], me)) >= 2]


@cat.fn
def opponent_forks(board, opponent, free_cells):
    return [c for c in free_cells if len(_wins(board[:c] + opponent + board[c + 1:], opponent)) >= 2]


@cat.fn
def forcing_moves(board, me, opponent, free_cells):
    """cells that threaten a win so the opponent must block — at a cell that does not give the opponent a fork"""
    out = []
    for c in free_cells:
        b = board[:c] + me + board[c + 1:]
        threats = _wins(b, me)
        if len(threats) == 1:
            r = threats[0]
            b2 = b[:r] + opponent + b[r + 1:]
            if len(_wins(b2, opponent)) < 2:
                out.append(c)
    return out


@cat.fn
def winner(board):
    return _winner(board) or "-"


@cat.check(hard=True, then={"move": "none"})
def board_valid(board, me):
    """hard: 9 cells of X/O/., fair turn order, game not over"""
    x, o = board.count("X"), board.count("O")
    turn_ok = (x == o and me == "X") or (x == o + 1 and me == "O")
    return len(board) == 9 and set(board) <= {"X", "O", "."} and turn_ok and _winner(board) is None and "." in board


@cat.rule("move")
def pick_move(winning_moves, blocking_moves, fork_moves, opponent_forks, forcing_moves, free_cells):
    for group in (winning_moves, blocking_moves, fork_moves):
        if group:
            return str(group[0])
    if len(opponent_forks) == 1:                          # a single fork cell: take it
        return str(opponent_forks[0])
    if opponent_forks and forcing_moves:                  # several: force the opponent to answer a threat instead
        return str(forcing_moves[0])
    for c in (4, 0, 2, 6, 8, 1, 3, 5, 7):                 # center, corners, edges
        if c in free_cells:
            return str(c)
    return "none"


@cat.rule("outcome")
def outcome(winner, free_cells, me):
    if winner == "-":
        return "draw" if not free_cells else "ongoing"
    return "won" if winner == me else "lost"


QUESTIONS = [Question("move", "Which cell to play?", Answer.choice(CELLS + ["none"]), checkpoints=["board_valid"]),
             Question("outcome", "State of the game for me?", Answer.choice(["ongoing", "won", "lost", "draw"]))]


def play(system, rng, agent="X"):
    board, turn = "." * 9, "X"
    while _winner(board) is None and "." in board:
        if turn == agent:
            c = int(system.ask({"board": board, "me": turn}, ["move"])["move"].answer)
        else:
            c = rng.choice([i for i in range(9) if board[i] == "."])
        board = board[:c] + turn + board[c + 1:]
        turn = "O" if turn == "X" else "X"
    return system.ask({"board": board, "me": agent}, ["outcome"])["outcome"].answer


if __name__ == "__main__":
    system = System(cat, QUESTIONS)
    rng = random.Random(0)
    results = [play(system, rng, agent=rng.choice("XO")) for _ in range(500)]
    print({k: results.count(k) for k in ("won", "draw", "lost")}, "in 500 games against a random opponent")
    lost = 0

    def explore(board, turn, agent):                      # every opponent reply, exhaustively
        global lost
        if _winner(board) or "." not in board:
            lost += _winner(board) not in (None, agent)
            return
        if turn == agent:
            c = int(system.ask({"board": board, "me": turn}, ["move"])["move"].answer)
            explore(board[:c] + turn + board[c + 1:], "O" if turn == "X" else "X", agent)
        else:
            for c in [i for i in range(9) if board[i] == "."]:
                explore(board[:c] + turn + board[c + 1:], "O" if turn == "X" else "X", agent)
    explore("." * 9, "X", "X")
    explore("." * 9, "X", "O")
    print(f"exhaustive check over all opponent replies: {lost} lost games")
    print("\n=== O to move: X threatens the top row ===")
    show(system.ask({"board": "XX.O.....", "me": "O"}), cat)
    print("\n=== an invalid board (O moved twice) ===")
    show(system.ask({"board": "OO.......", "me": "X"}), cat, flow=False)
