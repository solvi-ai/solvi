"""Bytes a stored decision takes, full and compact (TraceStorage(record=...)), and that both verify and replay.

    uv run python benchmarks/journal_size.py                 # every gallery entry, a game-like loop, the credit task
    uv run python benchmarks/journal_size.py --json out.json

Three workloads, no model and no network:
- **gallery**: every gallery entry's System asked with its own cases, three times over;
- **game loop**: a frequent-decision loop like an agent playing a game — one decision per step, the input a 40×16
  screen, 1.1 KB of notes, the inventory and the last moves; facts computed from the screen, a proposal, two hard checks
  that give their reasons (one remembers the moves that failed), a rule; 300 steps;
- **credit**: the task stand's German Credit solution (benchmarks/tasks/credit), version 1 deciding 300 history
  applications with every rule's points recorded (early_exit=False); its data is unpacked from tasks/packed/ when
  $STAND_DATA does not hold it.

Each in a fresh JSONLStorage per mode: "full", "compact" and "sample:10". Reported per workload and mode: the bytes a
decision takes in the file (records only: the head file and nothing else), whether verify() passes, and how many
decisions replay_all(system) does not replay (0 expected), with the milliseconds a replay takes per decision."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import sys
import tarfile
import tempfile
import time
import types
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
MODES = ("full", "compact", "sample:10")

from solvi import Answer, Catalog, Question, System  # noqa: E402
from solvi.refine import Fail  # noqa: E402
from solvi.storage import JSONLStorage  # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- workloads: each takes a store and asks its decisions, returning the System
def gallery(folder):
    def run(store):
        raw = json.loads((folder / "cases.json").read_text())
        if isinstance(raw, dict):                          # {"system": "run.py:system", "cases": [...]}
            mod_path, fn = raw["system"].split(":")
            sys_ = getattr(_module(f"gallery_{folder.name}", folder / mod_path), fn)()
            sys_.storage = store
            store.catalog = sys_
            cases, task = raw["cases"], None
        else:
            task = types.ModuleType("task")
            exec(compile((folder / "task.py").read_text(), str(folder / "task.py"), "exec", dont_inherit=True), task.__dict__)
            sys_ = System(task.cat, task.QUESTIONS, storage=store)
            cases = raw
        for _ in range(3):
            for case in cases:
                s = copy.deepcopy(case["state"])
                if task is not None and callable(getattr(task, "prepare", None)):
                    s = task.prepare(s)
                sys_.ask(s)
        return sys_
    return run


def game(store, steps=300):
    rng = random.Random(7)
    W, H = 40, 16
    notes = ("The old man said: the key is in the cave north of the lake. Bombs open cracked walls. " * 12).strip()
    cat = Catalog()

    @cat.fn
    def exits(screen: str, pos: list) -> list:
        rows, (x, y) = screen.split("\n"), pos
        return [d for d, (dx, dy) in {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}.items()
                if 0 <= y + dy < len(rows) and 0 <= x + dx < len(rows[y + dy]) and rows[y + dy][x + dx] != "#"]

    @cat.fn
    def enemies(screen: str) -> int:
        return screen.count("E")

    @cat.fn
    def danger(enemies: int, hearts: int) -> float:
        return round(min(1.0, enemies / max(hearts, 1) / 3), 3)

    @cat.fn
    def goal(notes: str, inventory: list) -> str:
        return "cave" if "key" not in inventory else "dungeon"

    @cat.fn
    def proposal(exits: list, goal: str, last_moves: list) -> str:
        return exits[0] if exits else "wait"

    @cat.check(hard=True, then={"move": "wait"})
    def not_into_wall(proposal: str, exits: list) -> bool:
        return True if proposal == "wait" or proposal in exits else Fail(f"{proposal} is a wall")

    @cat.check(hard=True, then={"move": "wait"})
    def not_repeat_failed(proposal: str, failed_moves: list) -> bool:
        return Fail(f"{proposal} failed in the last steps") if proposal in failed_moves else True

    @cat.rule("move")
    def move(proposal: str, danger: float) -> Literal["north", "south", "west", "east", "wait"]:
        return "wait" if danger > 0.9 else proposal

    q = Question("move", "Which way?", Answer.choice(["north", "south", "west", "east", "wait"]),
                 requires=["not_into_wall", "not_repeat_failed"])
    sys_ = System(cat, [q], storage=store)
    pos, inv, last, failed = [5, 5], ["sword", "shield"], [], []
    for _ in range(steps):
        grid = [["#" if (x in (0, W - 1) or y in (0, H - 1) or rng.random() < 0.08) else "." for x in range(W)]
                for y in range(H)]
        for _ in range(rng.randrange(0, 4)):
            grid[rng.randrange(1, H - 1)][rng.randrange(1, W - 1)] = "E"
        grid[pos[1]][pos[0]] = "@"
        res = sys_.ask({"screen": "\n".join("".join(r) for r in grid), "pos": list(pos), "notes": notes,
                        "inventory": list(inv), "hearts": 3, "last_moves": last[-20:], "failed_moves": failed[-5:]})
        m = res["move"].answer
        last.append(m)
        if rng.random() < 0.2:
            failed.append(m)
    return sys_


def credit(store):
    data = Path(os.environ.get("STAND_DATA") or ROOT / "benchmarks/tasks/data")
    if not (data / "credit/prepared/applications_history.jsonl").exists():
        data = Path(tempfile.mkdtemp())
        with tarfile.open(ROOT / "benchmarks/tasks/packed/prepared.tar.xz") as tf:
            tf.extractall(data, members=[m for m in tf.getmembers() if m.name.startswith("credit/")],
                          filter="data")
        os.environ["STAND_DATA"] = str(data)
    tasks = ROOT / "benchmarks/tasks"
    sys.path[:0] = [str(tasks), str(tasks / "credit")]
    try:
        sol = _module("stand_credit_solution", tasks / "credit/solution.py")
        from common.llm import read_jsonl
        v1 = sol.deploy(1, storage=store)
        for a in read_jsonl(sol.D / "applications_history.jsonl")[:300]:
            v1.ask(sol.Application.model_validate(a))
        return v1
    finally:
        for k in ("score", "reference", "common", "common.llm", "stand_credit_solution"):
            sys.modules.pop(k, None)


def measure(name, run, mode):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "d.jsonl"
        store = JSONLStorage(p, record=mode)
        system = run(store)
        n = len(list(store.iter()))
        size = p.stat().st_size
        ok = store.verify()["ok"]
        t0 = time.perf_counter()
        bad = store.replay_all(system)
        ms = (time.perf_counter() - t0) * 1000 / max(n, 1)
    return {"workload": name, "mode": mode, "decisions": n, "bytes_per_decision": round(size / max(n, 1)),
            "verify_ok": ok, "replay_failed": len(bad), "replay_ms_per_decision": round(ms, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the rows to this file")
    ap.add_argument("--only", choices=("gallery", "game", "credit"))
    args = ap.parse_args(argv)
    work = []
    if args.only in (None, "gallery"):
        work += [(f"gallery/{f.name}", gallery(f)) for f in sorted((ROOT / "gallery").iterdir())
                 if (f / "cases.json").exists()]
    if args.only in (None, "game"):
        work.append(("game loop", game))
    if args.only in (None, "credit"):
        work.append(("credit", credit))
    rows = []
    for name, run in work:
        got = {m: measure(name, run, m) for m in MODES}
        rows += got.values()
        f, c, s = (got[m]["bytes_per_decision"] for m in MODES)
        checks = "ok" if all(r["verify_ok"] and not r["replay_failed"] for r in got.values()) else "FAILED"
        print(f"{name:38s} {got['full']['decisions']:4d} decisions  full {f:6d}  compact {c:6d} ({f / c:.1f}x)  "
              f"sample:10 {s:6d}  bytes a decision; verify + replay {checks}", flush=True)
    g = [(r["bytes_per_decision"]) for r in rows if r["workload"].startswith("gallery/")]
    if g:
        import statistics
        full = [r["bytes_per_decision"] for r in rows if r["workload"].startswith("gallery/") and r["mode"] == "full"]
        comp = [r["bytes_per_decision"] for r in rows if r["workload"].startswith("gallery/") and r["mode"] == "compact"]
        print(f"gallery, {len(full)} entries: full {min(full)}–{max(full)} (median {statistics.median(full):.0f}), "
              f"compact {min(comp)}–{max(comp)} (median {statistics.median(comp):.0f}); "
              f"{min(a / b for a, b in zip(full, comp)):.1f}–{max(a / b for a, b in zip(full, comp)):.1f}x smaller")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0 if all(r["verify_ok"] and not r["replay_failed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
