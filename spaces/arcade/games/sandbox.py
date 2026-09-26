"""Run user bot code for the maze in a separate process: 5 s wall timeout; the child sets CPU / memory / file-size /
process limits (resource.setrlimit) before it reads the code; a fresh temp dir as cwd and an almost empty environment. This limits runaway code; it is not a hardened security sandbox."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ARCADE_DIR = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "_bot_runner.py"
MAX_CODE = 20_000
N_GAMES = 20
TIMEOUT_S = 5.0


def _paths():
    paths = [str(ARCADE_DIR)] + [p for p in sys.path if p and os.path.isdir(p)]
    return list(dict.fromkeys(paths))


def run_bot(code: str, n: int = N_GAMES, safety: bool = True, timeout: float = TIMEOUT_S) -> dict:
    """Returns {"ok": True, "summary": ..., "games": [...], ...} or {"ok": False, "error": text, "kind": ...}."""
    if len(code) > MAX_CODE:
        return {"ok": False, "kind": "too long", "error": f"code is {len(code)} characters; the limit is {MAX_CODE}"}
    with tempfile.TemporaryDirectory(prefix="solvi-bot-") as tmp:
        code_file = Path(tmp) / "bot.py"
        code_file.write_text(code)
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "PYTHONHASHSEED": "0", "HOME": tmp, "TMPDIR": tmp}
        try:
            p = subprocess.run([sys.executable, "-I", str(RUNNER), str(code_file), json.dumps(_paths()), str(n),
                                "1" if safety else "0"],
                               cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout, start_new_session=True, check=False)
        except subprocess.TimeoutExpired:
            return {"ok": False, "kind": "timeout", "error": f"stopped after {timeout:.0f} s (infinite loop or a bot that is too slow)"}
    if p.returncode != 0:
        err = (p.stderr or "").strip().splitlines()
        if p.returncode in (-9, -24) or (p.returncode < 0 and not err):
            return {"ok": False, "kind": "killed", "error": f"killed by a resource limit (signal {-p.returncode})"}
        at = next((i for i, ln in enumerate(err) if '"<bot>"' in ln), None)
        tail = "\n".join(err[at:] if at is not None else err[-1:]) or f"exit code {p.returncode}"
        return {"ok": False, "kind": "error", "error": tail}
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "kind": "error", "error": "no result from the bot process:\n" + (p.stdout or "")[-500:]}
    out["ok"] = True
    return out
