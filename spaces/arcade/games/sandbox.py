"""Run user bot code for the maze in-process, in the visitor's own browser (Pyodide has no subprocesses).

The code is exec'd in a fresh namespace. A sys.settrace guard counts the lines executed in the user's code (the file
"<bot>" only, so solvi and the game engine run at full speed) and stops the run when one call executes too many lines
(an infinite loop) or the whole run passes a wall-clock deadline. The guard raises BotStopped, a BaseException, so solvi
(which turns ordinary exceptions of a step into an abstain) cannot swallow it and the run ends at once.

This protects against runaway Python loops in the bot. It is not a security sandbox, and it cannot interrupt a loop that
runs inside C code (for example sum(itertools.count())); there the tab has to be reloaded."""
from __future__ import annotations

import builtins
import functools
import sys
import time

from . import maze

MAX_CODE = 20_000
N_GAMES = 20
MAX_LINES_PER_CALL = 200_000     # lines of user code in one call of the bot (or in the module body)
DEADLINE_S = 30.0                # the whole run of N_GAMES games (about 6 s on a laptop)


class BotStopped(BaseException):
    """Raised by the guard. BaseException on purpose: solvi catches Exception from a step, not this."""


class Guard:
    def __init__(self, max_lines=MAX_LINES_PER_CALL, deadline_s=DEADLINE_S):
        self.max_lines, self.deadline_s = max_lines, deadline_s
        self.t_end = time.perf_counter() + deadline_s
        self.lines = 0

    def _local(self, frame, event, arg):
        if event == "line":
            self.lines += 1
            if self.lines > self.max_lines:
                raise BotStopped(f"stopped: one call of your code ran more than {self.max_lines:,} lines "
                                 f"(an infinite loop?), at line {frame.f_lineno}")
            if not self.lines & 1023 and time.perf_counter() > self.t_end:
                raise BotStopped(f"stopped: the run took longer than {self.deadline_s:.0f} s")
        return self._local

    def _global(self, frame, event, arg):
        return self._local if frame.f_code.co_filename == "<bot>" else None

    def call(self, f, *args, **kwargs):
        if time.perf_counter() > self.t_end:
            raise BotStopped(f"stopped: the run took longer than {self.deadline_s:.0f} s")
        self.lines = 0
        old = sys.gettrace()
        sys.settrace(self._global)
        try:
            return f(*args, **kwargs)
        finally:
            sys.settrace(old)


def load_guarded(code: str, guard: Guard):
    """exec the code in a fresh namespace under the guard; return greedy_move wrapped so every call is guarded too."""
    ns = {"__name__": "user_bot", "__builtins__": builtins}
    guard.call(exec, compile(code, "<bot>", "exec"), ns)  # noqa: S102 - the visitor's own code, in their own browser
    f = ns.get("greedy_move")
    if not callable(f):
        raise ValueError("the code must define a function named greedy_move")

    @functools.wraps(f)          # keeps the name and the signature: solvi reads the argument names
    def greedy_move(*args, **kwargs):
        return guard.call(f, *args, **kwargs)
    return greedy_move


def _error_tail(e: BaseException) -> str:
    """the traceback lines that point into the user's code, plus the error itself"""
    import traceback
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    at = next((i for i, ln in enumerate(lines) if 'File "<bot>"' in ln), None)
    return "".join(lines[at:] if at is not None else lines[-1:]).strip()


def run_bot(code: str, n: int = N_GAMES, safety: bool = True) -> dict:
    """Returns {"ok": True, "summary": ..., "games": [...], "abstains": k, ...} or {"ok": False, "error": text, "kind": ...}."""
    if len(code) > MAX_CODE:
        return {"ok": False, "kind": "too long", "error": f"code is {len(code)} characters; the limit is {MAX_CODE}"}
    guard = Guard()
    try:
        greedy = load_guarded(code, guard)
        _, system = maze.build_system(safety, greedy)
        games, abstains, first_error = [], 0, None
        for seed in range(n):
            g = maze.new_game(seed)
            while not g.over:
                mv, resp, note = maze.agent_decide(system, g, safety)
                if note == "abstain":
                    abstains += 1
                    if abstains == 1:
                        errs = [r.error for r in resp.trace.records if r.name == "greedy_move" and r.error]
                        first_error = errs[0] if errs else resp["move"].why
                maze.step(g, mv, note)
            games.append({"won": g.won, "score": g.score, "dots": maze.TOTAL_DOTS - len(g.dots), "deaths": g.deaths,
                          "ticks": g.tick, "forced": g.forced})
    except BotStopped as e:
        return {"ok": False, "kind": "stopped by the guard", "error": str(e)}
    except (KeyboardInterrupt, SystemExit) as e:
        return {"ok": False, "kind": "error", "error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001 - syntax errors, a missing greedy_move, errors in the module body
        return {"ok": False, "kind": "error", "error": _error_tail(e)}
    out = {"ok": True, "summary": maze.summarize(games), "games": games, "abstains": abstains}
    if abstains:
        out["first_error"] = first_error
    return out
