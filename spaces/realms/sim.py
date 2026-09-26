"""Headless endless-game simulator: run solvi realms for a long time and check that it stays healthy.

    python sim.py --seed 1 --turns 100000            # one long run → results/endless_1.json
    python sim.py --charts                            # charts from every results/endless_*.json (PNG/SVG need matplotlib)

Plain CPython, no Gradio. From the solvi repo:  uv run --with matplotlib python spaces/realms/sim.py --seed 1 --turns 20000

Every --every turns (default 500) it records: per-decision time (median, p99) and turn time, state size (serialized JSON,
bytes), traced memory (tracemalloc, current and peak), factions alive (the minimum over the window, checked every turn) and
spawned, total population, resource totals, hard-check vetoes per 100 decisions, abstentions, early exits, exceptions,
trace replay on a random 2% sample of decisions, and the adaptive faction's rolling score and decision reward vs the others.
Every --saveload turns it also saves the game to JSON, reloads it, plays 20 turns on both copies and compares state hashes.

PASS/FAIL invariants, fixed before the runs (realms/health.py):

  I1  no exceptions            engine exceptions == 0 and no solvi part raised an error
  I2  decision time flat       p99 decision time, mean over the last 10% of checkpoints <= 2 x the first 10%
  I3  turn time flat           p99 turn time, same test
  I4  state bounded            serialized state: max over the last half <= 1.5 x max over the first half, and < 1 MB
  I5  memory bounded           tracemalloc current (or process RSS if off): max over the last half <= 1.5 x max over the first half
  I6  >= 2 factions alive      at every turn
  I7  economy not degenerate   population, gold, wood, stone, deposit richness: over the last half the minimum stays
                               above 5% of the run's mean (no collapse) and the linear trend moves the value by less than
                               50% of its mean (no runaway)
  I8  treasury never negative  no faction ends a turn with gold < 0 after upkeep
  I9  trace replay OK          every sampled decision's trace replays with no mismatch
  I10 no abstentions           every question got an answer (ok or forced by a hard check)
  I11 save/load exact          a reloaded game continues bit-for-bit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from realms import health  # noqa: E402
from realms.engine import Game, Metrics  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def rss():
    """(current, peak) resident memory in bytes, from /proc (used when tracemalloc is off)."""
    try:
        with open("/proc/self/status") as fh:
            st = dict(line.split(":", 1) for line in fh if ":" in line)
        return int(st["VmRSS"].split()[0]) * 1024, int(st["VmHWM"].split()[0]) * 1024
    except OSError:
        return None


def saveload_check(game, turns=20):
    """Save → load → play `turns` on both → identical state hash?"""
    import copy as _copy
    last, game.last = game.last, {}
    twin = _copy.deepcopy(game)                        # the in-memory game, continued without saving
    game.last = last
    twin.m, twin.keep_last = Metrics(replay_rate=0, seed=game.seed), False
    loaded = Game.from_json(game.to_json(), metrics=Metrics(replay_rate=0, seed=game.seed), keep_last=False)
    twin.play(turns)
    loaded.play(turns)
    return twin.state_hash() == loaded.state_hash()


def run(seed, turns, every, saveload_every, trace_mem, out, learn=True):
    if trace_mem:
        tracemalloc.start()
    t_start = time.time()
    game = Game(seed=seed, learn=learn)
    recs, exceptions, errors = [], 0, []
    alive_min, negative = 99, 0
    meta = {"seed": seed, "turns": turns, "every": every, "tracemalloc": trace_mem, "python": sys.version.split()[0],
            "learn": learn,
            "memory": "tracemalloc current/peak" if trace_mem else "process RSS current/peak (/proc)"}
    while game.turn < turns:
        t0 = time.perf_counter()
        try:
            game.play_turn()
        except Exception:  # noqa: BLE001 — counted, the run continues
            exceptions += 1
            if len(errors) < 5:
                errors.append(traceback.format_exc())
        game.m.turn_ms.append((time.perf_counter() - t0) * 1000)
        alive_min = min(alive_min, len(game.alive()))
        negative += sum(1 for f in game.alive() if game.factions[f]["gold"] < 0)
        if game.turn % every == 0:
            sl = None
            if saveload_every and game.turn % saveload_every == 0:
                sl = saveload_check(game)
            mem = tracemalloc.get_traced_memory() if trace_mem else rss()
            r = health.record(game, alive_min, exceptions=exceptions, negative_treasury=negative, mem=mem,
                              saveload_ok=sl)
            r["wall_s"] = round(time.time() - t_start, 1)
            recs.append(r)
            exceptions, alive_min, negative = 0, 99, 0
            game.m.reset()
            inv = health.check(recs)
            meta.update(elapsed_s=round(time.time() - t_start, 1), final_turn=game.turn, errors=errors,
                        learning_curve=list(game.learner.curve), counters=game.counters,
                        invariants=[list(x) for x in inv], log_tail=list(game.log)[-30:])
            with open(out + ".tmp", "w") as fh:
                json.dump({"meta": meta, "records": recs}, fh)
            os.replace(out + ".tmp", out)
            print(f"seed {seed} turn {game.turn:6d}  {r['wall_s']:7.0f}s  turn p50 {r['turn_ms_med']:6.2f} ms "
                  f"p99 {r['turn_ms_p99']:6.1f}  dec p50 {r['dec_ms_med']:.3f} p99 {r['dec_ms_p99']:.2f} ms  "
                  f"alive {r['alive']} (min {r['alive_min']})  pop {r['pop']}  state {r['state_bytes'] // 1024} KB  "
                  f"mem {(r['mem_cur'] or 0) / 1e6:.1f} MB  vetoes {r['vetoes_per_100']}/100  "
                  f"replay {r['replay_ok']}/{r['replay_n']}  share {r['adaptive_share']}", flush=True)
    fails = [x for x in health.check(recs) if x[2] is False]
    print(f"seed {seed}: {turns} turns in {time.time() - t_start:.0f} s; invariants failed: "
          f"{[x[0] for x in fails] or 'none'}", flush=True)
    return recs


# ------------------------------------------------------------------------------------------------------------ charts
PANELS = [("dec_ms_p99", "decision time p99 (ms)"), ("dec_ms_med", "decision time median (ms)"),
          ("turn_ms_p99", "turn time p99 (ms)"), ("state_bytes", "saved state (KB)"),
          ("mem_cur", "memory MB (tracemalloc, or RSS)"), ("alive", "factions alive (min over window: dots)"),
          ("pop", "total population"), ("gold", "gold / wood / stone (all factions)"),
          ("deposit_amt", "deposit richness (sum)"), ("vetoes_per_100", "hard-check vetoes per 100 decisions"),
          ("adaptive_share", "adaptive faction score / mean of others (log)"), ("spawned", "factions spawned (cumulative)")]


def charts():
    files = sorted(glob.glob(os.path.join(RESULTS, "endless_*.json")))
    runs = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        runs[d["meta"]["seed"]] = d
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: run with  uv run --with matplotlib python sim.py --charts")
        return
    colors = ["#3c7dd9", "#e6194b", "#2ca02c", "#f58231", "#9b59b6"]
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    for ax, (key, title) in zip(axes.flat, PANELS):
        for k, (seed, d) in enumerate(sorted(runs.items())):
            rs = d["records"]
            x = [r["turn"] for r in rs]
            c = colors[k % len(colors)]
            if key == "state_bytes":
                y = [r[key] / 1024 for r in rs]
            elif key == "mem_cur":
                y = [(r[key] or 0) / 1e6 for r in rs]
            elif key == "adaptive_share":
                y = [max(r[key], 0.05) for r in rs]
            else:
                y = [r[key] for r in rs]
            ax.plot(x, y, color=c, lw=1.2, label=f"seed {seed}")
            if key == "alive":
                ax.plot(x, [r["alive_min"] for r in rs], ".", color=c, ms=3)
            if key == "gold":
                ax.plot(x, [r["wood"] for r in rs], color=c, lw=0.8, ls="--")
                ax.plot(x, [r["stone"] for r in rs], color=c, lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        if key == "alive":
            ax.set_ylim(0, 6)
            ax.axhline(2, color="#999", lw=0.8, ls="--")
        if key == "adaptive_share":
            ax.axhline(1, color="#999", lw=0.8, ls="--")
            ax.set_yscale("log")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("solvi realms: endless-game health (one point per 500 turns)", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"health.{ext}"), dpi=90)
    # learning curve of the adaptive faction: its share, against the ablation (heads frozen after the bootstrap) and the
    # earlier learner (fit_fast build head only), plus the learned heads' reward and the amount of judged decisions
    def load(pattern):
        out = {}
        for f in sorted(glob.glob(os.path.join(RESULTS, pattern))):
            with open(f) as fh:
                d = json.load(fh)
            out[d["meta"]["seed"]] = d
        return out
    frozen, old = load("ablation/frozen_*.json"), load("baseline_fitfast/endless_*.json")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for k, (seed, d) in enumerate(sorted(runs.items())):
        c = colors[k % len(colors)]
        if seed not in (1, 2, 3, 4):                   # the pre-registered evaluation seeds only
            continue
        rs = [r for r in d["records"] if r["turn"] <= 20000]
        axes[0].plot([r["turn"] for r in rs], [max(r["adaptive_share"], 0.05) for r in rs], color=c, lw=1.4,
                     label=f"seed {seed}")
        if seed in frozen:
            fr = frozen[seed]["records"]
            axes[0].plot([r["turn"] for r in fr], [max(r["adaptive_share"], 0.05) for r in fr], color=c, lw=1.0, ls="--")
        if seed in old:
            orr = [r for r in old[seed]["records"] if r["turn"] <= 20000]
            axes[0].plot([r["turn"] for r in orr], [max(r["adaptive_share"], 0.05) for r in orr], color=c, lw=0.8, ls=":")
        rr = [r for r in d["records"] if r["adaptive_reward"] is not None and r["others_reward"] is not None]
        axes[1].plot([r["turn"] for r in rr], [r["adaptive_reward"] - r["others_reward"] for r in rr], color=c, lw=1.2,
                     label=f"seed {seed}")
        axes[2].plot([r["turn"] for r in d["records"]], [r["teaches"] for r in d["records"]], color=c, lw=1.2,
                     label=f"seed {seed}")
    axes[0].axhline(1, color="#999", lw=0.8, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_title("adaptive share (log): solid learner, dashed frozen,\ndotted previous fit_fast learner", fontsize=9)
    axes[1].axhline(0, color="#999", lw=0.8, ls="--")
    axes[1].set_title("build reward, adaptive minus others\n(log-score gain 20 turns later, rolling 250 turns)", fontsize=9)
    axes[2].set_title("decisions judged and learned from\n(cumulative, own + observed)", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"learning.{ext}"), dpi=90)
    print("charts written to", RESULTS)


def share_fl(rs, turns=20000):
    """Adaptive share averaged over the first and the last 20% of the checkpoints up to `turns`."""
    rs = [r for r in rs if r["turn"] <= turns]
    k = max(1, len(rs) // 5)
    return sum(r["adaptive_share"] for r in rs[:k]) / k, sum(r["adaptive_share"] for r in rs[-k:]) / k


def criteria():
    """The pre-registered criteria A1-A3 (realms/adaptive.py) from results/endless_<1..4>.json, plus the ablation."""
    rows, a1, a2, a3 = [], 0, 0, 0
    for seed in (1, 2, 3, 4):
        f = os.path.join(RESULTS, f"endless_{seed}.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            d = json.load(fh)
        rs = [r for r in d["records"] if r["turn"] <= 20000]
        first, last = share_fl(rs)
        inv = health.check(rs)
        fails = [x[0] for x in inv if x[2] is False]
        fz = os.path.join(RESULTS, "ablation", f"frozen_{seed}.json")
        fzs = "—"
        if os.path.exists(fz):
            with open(fz) as fh:
                ff, fl = share_fl(json.load(fh)["records"])
            fzs = f"{ff:.2f} → {fl:.2f}"
        old = os.path.join(RESULTS, "baseline_fitfast", f"endless_{seed}.json")
        ols = "—"
        if os.path.exists(old):
            with open(old) as fh:
                of, ol = share_fl(json.load(fh)["records"])
            ols = f"{of:.2f} → {ol:.2f}"
        a1 += last >= 1.0
        a2 += last > first
        a3 += not fails
        rows.append(f"| {seed} | {first:.2f} | **{last:.2f}** | {'yes' if last >= 1 else 'no'} | {'yes' if last > first else 'no'} | "
                    f"{', '.join(fails) or 'all 11 held'} | {fzs} | {ols} |")
    print("| seed | share first 20% | share last 20% | A1 ≥ 1.0 | A2 last > first | A3 invariants | frozen after bootstrap "
          "(first → last 20%) | previous fit_fast learner (first → last 20%) |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print(f"\nA1: {a1}/4 seeds (need >= 3) -> {'PASS' if a1 >= 3 else 'FAIL'};  A2: {a2}/4 (need 4) -> "
          f"{'PASS' if a2 == 4 else 'FAIL'};  A3: {a3}/4 runs with every invariant held -> {'PASS' if a3 == 4 else 'FAIL'}")


def table():
    """Markdown summary table of all runs (for the README)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "endless_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        rs, m = d["records"], d["meta"]
        inv = health.check(rs)
        k = max(1, len(rs) // 10)
        rows.append(
            f"| {m['seed']} | {rs[-1]['turn']:,} | {m['elapsed_s'] / 60:.0f} min | "
            f"{sum(r['dec_ms_med'] for r in rs[:k]) / k:.2f} / {sum(r['dec_ms_med'] for r in rs[-k:]) / k:.2f} | "
            f"{sum(r['dec_ms_p99'] for r in rs[:k]) / k:.2f} / {sum(r['dec_ms_p99'] for r in rs[-k:]) / k:.2f} | "
            f"{max(r['state_bytes'] for r in rs) // 1024} KB | {min(r['alive_min'] for r in rs)} | {rs[-1]['spawned']} | "
            f"{sum(r['decisions'] for r in rs):,} | {sum(r['replay_ok'] for r in rs)}/{sum(r['replay_n'] for r in rs)} | "
            f"{', '.join(x[0] for x in inv if x[2] is True)} | {', '.join(x[0] for x in inv if x[2] is False) or '—'} |")
    print("| seed | turns | wall time | decision ms p50 first/last 10% | p99 first/last | max state | min alive | "
          "factions spawned | decisions | replay OK | held | failed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--turns", type=int, default=20000)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--saveload", type=int, default=5000, help="save/load round-trip every N turns (0: never)")
    ap.add_argument("--no-tracemalloc", action="store_true")
    ap.add_argument("--charts", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--frozen", action="store_true", help="ablation: the adaptive learner's heads stay at the bootstrap")
    ap.add_argument("--out", default="", help="output JSON (default results/endless_<seed>.json)")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.charts:
        charts()
    elif a.table:
        table()
        print()
        criteria()
    else:
        run(a.seed, a.turns, a.every, a.saveload, not a.no_tracemalloc,
            a.out or os.path.join(RESULTS, f"endless_{a.seed}.json"), learn=not a.frozen)
