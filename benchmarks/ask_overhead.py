"""The overhead solvi adds to one `ask`, with the 0.7 defaults against 0.5.0-style settings.

    uv run python benchmarks/ask_overhead.py              # every gallery entry + a decider project; ~1 minute
    uv run python benchmarks/ask_overhead.py --quick      # fewer repetitions
    uv run python benchmarks/ask_overhead.py --json out.json

Against a release (the gallery as it was then, the same runner):
    git archive v0.5.0 gallery | tar -x -C /tmp/g050
    uvx --from solvi==0.5.0 python benchmarks/ask_overhead.py --only gallery --gallery /tmp/g050/gallery
    uv run python benchmarks/ask_overhead.py --only gallery --gallery /tmp/g050/gallery

Two workloads, both without a real model:
- **gallery**: every gallery entry's System asked with its own cases (catalog code only: rules, checks, learned heads);
- **decider**: the project `solvi init --template support --with-model` writes — a rule, a hard check and a decision
  answered by the keyword stand-in decider (a DecideModel over keyword counts: the decider machinery runs, the network
  is a dictionary lookup, so what is measured is solvi's own work around a model call).

Settings, each measured on the same states (median and p90 of the per-ask wall time, `ask` called directly):
- `0.5.0-style`: no catalog fingerprint in the trace, options in the caller's order (`option_order="given"`), no
  calibrated guarantee, no store — what 0.5.0 did per ask;
- `0.7 default`: `trace.fingerprint` recorded, canonical option order (the 0.7 default);
- `+ guarantee`: the decision calibrated with `act_guard` (every decision records its guarantee) — decider only;
- `+ store jsonl` / `+ store sqlite`: every response saved with its trace (hash-chained TraceStorage);
- `- fingerprint`, `given order`: one 0.7 feature off at a time, to attribute a difference.

Numbers vary with the machine; compare settings within one run."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def settings(fingerprint=True, order="canonical"):
    """Patch the per-ask features 0.5.0 did not have: the trace fingerprint and the canonical option order."""
    from solvi.core.deciders import DecideModel
    from solvi.core.system import System
    if fingerprint and order == "canonical":          # nothing to patch (and an old release may lack what is patched)
        yield
        return
    saved_fp, saved_dec = System._fingerprint, DecideModel.decision
    if not fingerprint:
        System._fingerprint = lambda self, flow: {}
    if order != "canonical":
        def decision(self, *a, **kw):
            kw.setdefault("option_order", order)
            return saved_dec(self, *a, **kw)
        DecideModel.decision = decision
    try:
        yield
    finally:
        System._fingerprint, DecideModel.decision = saved_fp, saved_dec


def measure(workloads, reps):
    """workloads: {setting: (settings kwargs, [(system, states)])}. One warm-up pass, then `reps` passes; each pass runs
    every setting in turn (round robin, so a slower moment of the machine hits them all) → {setting: [ms per ask]}."""
    out = {name: [] for name in workloads}
    for rep in range(reps + 1):
        for name, (kw, jobs) in workloads.items():
            with settings(**kw):
                for system, states in jobs:
                    for st in states:
                        s = dict(st)
                        t0 = time.perf_counter()
                        system.ask(s)
                        if rep:
                            out[name].append((time.perf_counter() - t0) * 1000)
    return out


def _module(path):
    """Execute a task / runner file in a fresh module (as solvi.honesty.load_task does; that did not exist in 0.5.0)."""
    import hashlib
    import types
    path = Path(path).resolve()
    name = f"_bench_{hashlib.sha1(str(path).encode(), usedforsecurity=False).hexdigest()[:10]}_{path.stem}"
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


def gallery_jobs(root):
    """Every gallery entry under `root` → [(name, build, states)]: build() makes a fresh System (the cases file's
    "system" factory, else task.system(), else System(task.cat, task.QUESTIONS)); the states went through prepare()."""
    import copy

    from solvi import System
    out = []
    for f in sorted(Path(root).glob("*/cases.json")):
        data = json.loads(f.read_text())
        cases = data if isinstance(data, list) else data.get("cases", [])
        factory = None if isinstance(data, list) else data.get("system")
        task = _module(f.parent / ((None if isinstance(data, list) else data.get("task")) or "task.py"))

        def build(task=task, factory=factory, where=f.parent):
            if factory:
                file, _, fn = factory.partition(":")
                return getattr(_module(where / file), fn or "system")()
            if callable(getattr(task, "system", None)):
                return task.system()
            return System(task.cat, task.QUESTIONS)
        prep = getattr(task, "prepare", None)
        states = [prep(copy.deepcopy(c["state"])) if callable(prep) else copy.deepcopy(c["state"]) for c in cases]
        out.append((f.parent.name, build, states))
    return out


def gallery_workloads(tmp, root):
    from solvi.core.system import System
    if not hasattr(System, "_fingerprint"):           # a release before 0.5.1 (--gallery with an old solvi): as it is
        return {"installed": ({}, [(build(), states) for _, build, states in gallery_jobs(root)])}
    from solvi.core.store import open_storage
    configs = [("0.5.0-style", dict(fingerprint=False), None), ("0.7 default", {}, None),
               ("- fingerprint", dict(fingerprint=False), None),
               ("+ store jsonl", {}, "jsonl"), ("+ store sqlite", {}, "db")]
    entries = gallery_jobs(root)
    out = {}
    for name, kw, store in configs:
        jobs = []
        for entry, build, states in entries:
            with settings(**kw):
                system = build()
            if store:
                system.storage = open_storage(os.path.join(tmp, f"g_{entry}.{store}"), system)
            jobs.append((system, states))
        out[name] = (kw, jobs)
    return out


def decider_workloads(tmp):
    """`solvi init --template support --with-model` (a keyword stand-in decider), asked with its cases and with every
    labelled message of its labels.csv at three tiers / ages."""
    import csv

    from solvi.cli import main
    from solvi.core.store import open_storage
    from solvi.testing import load
    proj = Path(tmp) / "support"
    quiet = open(os.devnull, "w")
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        main(["init", str(proj), "--template", "support", "--with-model"])
    texts = [r["text"] for r in csv.DictReader(open(proj / "labels.csv", encoding="utf-8"))]
    extra = [{"message": t, "tier": tier, "hours_open": h} for t in texts
             for tier, h in (("standard", 2), ("vip", 6), ("standard", 30))]
    calib = proj / "route.calib.json"
    configs = [("0.5.0-style", dict(fingerprint=False, order="given"), None, False),
               ("0.7 default", {}, None, False),
               ("- fingerprint", dict(fingerprint=False), None, False),
               ("given order", dict(order="given"), None, False),
               ("+ guarantee", {}, None, True),
               ("+ store jsonl", {}, "jsonl", False), ("+ store sqlite", {}, "db", False)]
    out = {}
    for name, kw, store, guarantee in configs:
        if guarantee and not calib.exists():          # solvi calibrate: act_guard on labels.csv, route.calib.json
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                main(["calibrate", f"{proj / 'catalog.py'}:system", "route", str(proj / "labels.csv"), "--risk", "0.3",
                      "--out", str(calib)])
        with settings(**kw):
            suite = load(proj / "cases.json")          # the catalog module runs afresh: its decision part is rebuilt
            system = suite.system()                    # (and loads route.calib.json when it is there)
            states = [suite.state(c) for c in suite.cases] + extra
        if calib.exists():
            calib.unlink()
        if guarantee:
            recs = [r for r in system.ask(dict(states[0]), store=False).trace.records if r.model is not None]
            assert any("guarantee" in (r.extra or {}) for r in recs), "the calibration did not load"
        if store:
            system.storage = open_storage(os.path.join(tmp, f"d_{name[-6:].strip()}.{store}"), system)
        out[name] = (kw, [(system, states)])
    return out


def summary(rows, base="0.7 default"):
    out = []
    ref = statistics.median(rows[base if base in rows else next(iter(rows))])
    for name, xs in rows.items():
        xs = sorted(xs)
        med = statistics.median(xs)
        p90 = xs[int(0.9 * (len(xs) - 1))]
        out.append({"setting": name, "asks": len(xs), "median_ms": round(med, 3), "p90_ms": round(p90, 3),
                    "vs_0.7": round(med / ref, 3)})
    return out


def show(title, table):
    print(f"\n{title}", flush=True)
    print(f"  {'setting':16s} {'asks':>6s} {'median ms':>10s} {'p90 ms':>9s} {'× 0.7 default':>14s}")
    for r in table:
        print(f"  {r['setting']:16s} {r['asks']:6d} {r['median_ms']:10.3f} {r['p90_ms']:9.3f} {r['vs_0.7']:14.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="fewer repetitions")
    p.add_argument("--reps", type=int, help="passes over each workload's states (default 20; --quick: 5)")
    p.add_argument("--json", help="write the tables to this file")
    p.add_argument("--only", choices=["gallery", "decider"], help="one workload")
    p.add_argument("--gallery", default=str(ROOT / "gallery"),
                   help="the gallery folder (e.g. one exported from an older release, to compare releases)")
    a = p.parse_args(argv)
    reps = a.reps or (5 if a.quick else 20)
    import solvi
    print(f"solvi {solvi.__version__}, Python {sys.version.split()[0]}, {reps} passes")
    g = d = None
    with tempfile.TemporaryDirectory() as tmp:
        if a.only != "decider":
            g = summary(measure(gallery_workloads(tmp, a.gallery), reps))
            show(f"gallery ({a.gallery}; catalog code only)", g)
        if a.only != "gallery":
            d = summary(measure(decider_workloads(tmp), reps))
            show("decider (solvi init --with-model: keyword stand-in decider)", d)
    if a.json:
        Path(a.json).write_text(json.dumps({"solvi": solvi.__version__, "reps": reps, "gallery": g, "decider": d},
                                           indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
