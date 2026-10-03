"""Record the two playthroughs, replay them, and compute the numbers — from the bundled files alone.

    summary = record(out_dir)          # play run 1 (empty memory) and run 2 (the memory run 1 left), write the files
    check = replay(out_dir)            # every stored decision replays; the maps' journals verify; the logs agree
    nums = numbers(out_dir)            # decisions by System 1 / System 2 per run and per objective, time, moves

The files of a recording (data/runs/):
    run1.decisions.jsonl.gz, run2.decisions...   every decision: one hash-chained solvi.core.dispatch record (System 1's
                                                 response with its trace; System 2's search with the winner's trace),
                                                 a solvi JSONL store, gzipped (open_store() unpacks it)
    run1.moves.jsonl, run2.moves.jsonl           the same decisions as the viewer shows them (where, which exit, who,
                                                 why, where it led)
    run1.map.json, run2.map.json                 the world map after each run (solvi.core.knowledge.worldmap: journal with hashes)
    memory.json                                  System 1's routes after run 2, every consolidation, the episode notes
    summary.json                                 the numbers"""
from __future__ import annotations

import atexit
import gzip
import json
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path

from solvi.core.store import JSONLStorage
from solvi.core.knowledge.worldmap import WorldMap

from .agent import Memory, dispatcher, play
from .world import DATA, World, place_map

RUNS = DATA / "runs"
NAMES = ("run1", "run2")
_TMP, _N = None, [0]                 # one temporary folder per process for unpacked stores, removed at exit


def oracle(world):
    """The fewest moves for each objective of a player who knows the whole map (shortest paths in the recorded world,
    with the story's gates) — a reference, not a player: it does not explore and is never surprised."""
    here, fired, out = world.start, set(), []
    for ob in world.story:
        stage, n = ob["stage"], 0
        while place_map(here) not in ob["targets"]:
            dist, first = {here: 0}, {here: None}
            q = deque([here])
            while q:
                s = q.popleft()
                for k in world.on_offer(s, stage):
                    e = world.places[s]["exits"][k]
                    if stage < e["pass"] or e["to"] in dist:
                        continue
                    dist[e["to"]], first[e["to"]] = dist[s] + 1, first[s] or k
                    q.append(e["to"])
            goal = [t for t in dist if place_map(t) in ob["targets"]]
            ev = [e for i, e in enumerate(world.events) if e["stage"] == stage and i not in fired and e.get("done") == ob["id"]]
            if ev and ev[0]["place"] in dist and (not goal or dist[ev[0]["place"]] < min(dist[t] for t in goal)):
                goal = [ev[0]["place"]]
            t = min(goal, key=lambda x: (dist[x], x))
            if t == here:                    # the event's place: its exit finishes the objective
                st = world.take(here, ev[0]["exit"], stage, fired)
                n += 1
                here = st.to
                break
            st = world.take(here, first[t], stage, fired)
            n += 1
            here = st.to
            if st.event is not None and st.event.get("done") == ob["id"]:
                break
        out.append({"objective": ob["id"], "moves": n})
    return out


def _count(moves):
    by = {p: sum(m["by"] == p for m in moves) for p in ("s1", "s2", "human")}
    ms = {p: round(sum(m["ms"] for m in moves if m["by"] == p), 1) for p in ("s1", "s2")}
    return {"decisions": len(moves), **by, "ms": ms,
            "ms_per_decision": {p: round(ms[p] / by[p], 3) if by[p] else None for p in ("s1", "s2")},
            "blocked": sum(1 for m in moves if m["to"] == m["here"]), "surprises": sum(1 for m in moves if m["surprise"]),
            "events": sum(1 for m in moves if m["event"])}


def numbers_of(moves_by_run, story, oracle_moves=None):
    """The numbers from the move logs: per run and per objective."""
    out = {"runs": {}, "objectives": []}
    for run, moves in moves_by_run.items():
        out["runs"][run] = _count(moves)
    for i, ob in enumerate(story):
        row = {"objective": ob["id"]}
        for run, moves in moves_by_run.items():
            mine = [m for m in moves if m["objective"] == ob["id"]]
            row[run] = {"moves": len(mine), "s1": sum(m["by"] == "s1" for m in mine),
                        "s2": sum(m["by"] == "s2" for m in mine)}
        if oracle_moves is not None:
            row["shortest"] = oracle_moves[i]["moves"]
        out["objectives"].append(row)
    return out


def open_store(out, name):
    """The stored decisions of a run as a solvi JSONLStorage: the bundled .jsonl.gz unpacked into a temporary folder
    (or the plain .jsonl a recording left)."""
    global _TMP
    out = Path(out)
    plain = out / f"{name}.decisions.jsonl"
    if not plain.exists():
        if _TMP is None:
            _TMP = tempfile.mkdtemp(prefix="pokeworld-")
            atexit.register(shutil.rmtree, _TMP, True)
        _N[0] += 1
        tmp = Path(_TMP) / f"{_N[0]}-{plain.name}"
        with gzip.open(out / f"{name}.decisions.jsonl.gz", "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        plain = tmp
    return JSONLStorage(plain)


def _moves(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def record(out=RUNS, world=None, llm=None, log=None):
    """Play run 1 with an empty memory and run 2 with what run 1 left; write the recording to `out` → the summary."""
    world = world or World.load()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        for suffix in (".decisions.jsonl", ".decisions.jsonl.head", ".decisions.jsonl.gz", ".moves.jsonl", ".map.json"):
            (out / f"{name}{suffix}").unlink(missing_ok=True)
    mem = Memory()
    wall, by_run, done = {}, {}, {}
    for name in NAMES:
        t0 = time.perf_counter()
        moves, results, _ = play(world, mem, name, store=JSONLStorage(out / f"{name}.decisions.jsonl"), llm=llm,
                                 log=log)
        wall[name] = round((time.perf_counter() - t0) * 1000, 1)
        rows = [m.to_dict() for m in moves]
        (out / f"{name}.moves.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                 encoding="utf-8")
        mem.map.save(out / f"{name}.map.json")
        plain = out / f"{name}.decisions.jsonl"
        with open(plain, "rb") as src, gzip.GzipFile(out / f"{name}.decisions.jsonl.gz", "wb", mtime=0) as dst:
            shutil.copyfileobj(src, dst)
        plain.unlink()
        Path(f"{plain}.head").unlink(missing_ok=True)
        by_run[name] = rows
        done[name] = [r["objective"] for r in results if r["done"]]
        if log is not None:
            log({"run": name, "wall_ms": wall[name]})
    (out / "memory.json").write_text(json.dumps({"routes": mem.routes, "consolidations": mem.consolidations,
                                                 "notes": mem.notes, "goals": sorted(mem.goals)}, ensure_ascii=False),
                                     encoding="utf-8")
    summary = {"world": world.fingerprint(), "llm": None if llm is None else str(getattr(llm, "model", llm)),
               "wall_ms": wall, "objectives_done": {k: len(v) for k, v in done.items()},
               "objectives": len(world.story), **numbers_of(by_run, world.story, oracle(world))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def numbers(out=RUNS, world=None):
    """The numbers recomputed from the bundled move logs (and the shortest paths in the world file)."""
    world = world or World.load()
    out = Path(out)
    return numbers_of({n: _moves(out / f"{n}.moves.jsonl") for n in NAMES}, world.story, oracle(world))


def replay(out=RUNS, llm=None):
    """Check a recording without the game: every stored decision replays under today's catalogs (System 1's trace,
    the dispatch, System 2's search and its winner's trace); the maps' journals verify (hash chain, claims rebuilt from
    the journal); the move logs say what the stored decisions say; the numbers in summary.json are what the logs
    give. → {"decisions", "replayed", "mismatches", "maps", "logs", "numbers"}."""
    out = Path(out)
    rep = {"decisions": 0, "replayed": 0, "mismatches": [], "maps": {}, "logs": True, "numbers": None}
    for name in NAMES:
        d = dispatcher(storage=open_store(out, name), llm=llm)
        stored = d.stored()
        rep["decisions"] += len(stored)
        bad = d.replay_all(trust_models=True)
        rep["replayed"] += len(stored) - len(bad)
        rep["mismatches"] += [(name, n, m) for n, _, m in bad]
        moves = _moves(out / f"{name}.moves.jsonl")
        if len(moves) != len(stored) or any(
                (getattr(s.answer, "value", s.answer), s.by) != (m["exit"], m["by"]) for s, m in zip(stored, moves)):
            rep["logs"] = False
        m = WorldMap(out / f"{name}.map.json")         # loading rebuilds the claims from the journal (chain checked)
        rep["maps"][name] = {"verify": m.verify(), **m.stats()}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    now = numbers(out)
    rep["numbers"] = all(summary[k] == now[k] for k in ("runs", "objectives"))
    rep["ok"] = not rep["mismatches"] and rep["logs"] and rep["numbers"] and all(v["verify"] for v in rep["maps"].values())
    return rep


def report(out=RUNS, run="run2"):
    """The system report (System.report, solvi.core.store.sysreport) over one run's stored decisions — read from the store
    alone."""
    from .agent import system1
    return system1().report(store=open_store(out, run))


def same_as_recorded(moves, out=RUNS, run="run1"):
    """Did a live run take the same decisions as the recording? → the first difference, or None."""
    rec = _moves(Path(out) / f"{run}.moves.jsonl")
    keys = ("n", "objective", "here", "exit", "by", "to", "event", "surprise")
    for a, b in zip(rec, moves):
        da, db = {k: a[k] for k in keys}, {k: b[k] for k in keys}
        if da != db:
            return {"recorded": da, "now": db}
    if len(rec) != len(moves):
        return {"recorded": f"{len(rec)} decisions", "now": f"{len(moves)} decisions"}
    return None


if __name__ == "__main__":                              # re-record the bundled runs: python -m pokeworld.record
    s = record(log=lambda r: print(r) if "run" in r else None)
    print({k: s["runs"][k]["decisions"] for k in NAMES}, "replay ok:", replay()["ok"])
