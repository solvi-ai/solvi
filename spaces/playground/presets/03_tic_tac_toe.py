"""Games: pick a tic-tac-toe move (from examples/05_tic_tac_toe.py).
The board is a 9-character string, cells 0..8 row by row: X, O or '.'.
Each move comes with its reason (win, block, fork, center...). A hard check refuses invalid boards.
Try: "OO......." with me = "X" (O moved twice)."""
from solvi import Answer, Catalog, Question

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
cat = Catalog()


def _wins(board, who):
    return [c for c in range(9) if board[c] == "." and any(
        c in ln and sum(board[i] == who for i in ln) == 2 and all(board[i] in (who, ".") for i in ln) for ln in LINES)]


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
    """cells where I complete a line right now"""
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
    """a threat the opponent must block, where the block does not give them a fork"""
    out = []
    for c in free_cells:
        b = board[:c] + me + board[c + 1:]
        threats = _wins(b, me)
        if len(threats) == 1:
            r = threats[0]
            if len(_wins(b[:r] + opponent + b[r + 1:], opponent)) < 2:
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
    if len(opponent_forks) == 1:
        return str(opponent_forks[0])
    if opponent_forks and forcing_moves:
        return str(forcing_moves[0])
    for c in (4, 0, 2, 6, 8, 1, 3, 5, 7):          # center, corners, edges
        if c in free_cells:
            return str(c)
    return "none"


@cat.rule("outcome")
def outcome(winner, free_cells, me):
    if winner == "-":
        return "draw" if not free_cells else "ongoing"
    return "won" if winner == me else "lost"


QUESTIONS = [
    Question("move", "Which cell to play?", Answer.choice([str(i) for i in range(9)] + ["none"]),
             requires=["board_valid"]),
    Question("outcome", "State of the game for me?", Answer.choice(["ongoing", "won", "lost", "draw"])),
]
