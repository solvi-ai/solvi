"""Health invariants of the endless game, shared by the headless simulator (sim.py) and the browser app.

A run is a list of checkpoint records (dicts, one every K turns; see sim.py for the fields). The invariants:

  I1  no exceptions            engine exceptions == 0 and no solvi part raised an error
  I2  decision time flat       p99 decision time, mean over the last 10% of checkpoints <= 2 x the first 10%
  I3  turn time flat           p99 turn time, same test
  I4  state bounded            serialized state: max over the last half <= 1.5 x max over the first half, and < 1 MB
  I5  memory bounded           tracemalloc current (or process RSS when tracemalloc is off): max over the last half <= 1.5 x max over the first half
  I6  >= 2 factions alive      at every turn (the minimum over each window is recorded)
  I7  economy not degenerate   population, gold, wood, stone, deposit richness: over the last half the minimum stays
                               above 5% of the run's mean (no collapse to 0) and the linear trend moves the value by less
                               than 50% of its mean (no runaway to infinity)
  I8  treasury never negative  no faction ends a turn with gold < 0 after upkeep
  I9  trace replay OK          every sampled decision's hash-chained trace replays with no mismatch
  I10 no abstentions           every question got an answer (ok or forced by a hard check)
  I11 save/load exact          a saved and reloaded game continues bit-for-bit (same state hash N turns later)
"""
from __future__ import annotations

ECON_KEYS = ["pop", "gold", "wood", "stone", "deposit_amt"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _slope(ys):
    n = len(ys)
    if n < 2:
        return 0.0
    mx, my = (n - 1) / 2, _mean(ys)
    num = sum((i - mx) * (y - my) for i, y in enumerate(ys))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den


def check(recs):
    """→ [(id, name, ok, detail)]; ok is None when there is not enough data yet."""
    out = []
    n = len(recs)
    if n == 0:
        return out
    k = max(1, n // 10)
    first, last = recs[:k], recs[-k:]
    half_a, half_b = recs[: max(1, n // 2)], recs[n // 2:] or recs

    exc = sum(r.get("exceptions", 0) for r in recs)
    perr = sum(r.get("part_errors", 0) for r in recs)
    out.append(("I1", "no exceptions", exc == 0 and perr == 0, f"engine exceptions {exc}, solvi part errors {perr}"))

    for iid, key, name in (("I2", "dec_ms_p99", "decision time flat"), ("I3", "turn_ms_p99", "turn time flat")):
        a, b = _mean(r[key] for r in first), _mean(r[key] for r in last)
        ok = None if n < 4 else b <= 2 * a
        out.append((iid, name, ok, f"p99 first 10% {a:.2f} ms, last 10% {b:.2f} ms (x{b / a if a else 0:.2f}, limit x2)"))

    sa, sb = max(r["state_bytes"] for r in half_a), max(r["state_bytes"] for r in half_b)
    ok = None if n < 4 else (sb <= 1.5 * sa and sb < 1_000_000)
    out.append(("I4", "state bounded", ok, f"max state first half {sa / 1024:.0f} KB, last half {sb / 1024:.0f} KB"))

    if all(r.get("mem_cur") is not None for r in recs):
        ma, mb = max(r["mem_cur"] for r in half_a), max(r["mem_cur"] for r in half_b)
        ok = None if n < 4 else mb <= 1.5 * ma
        out.append(("I5", "memory bounded", ok, f"max memory first half {ma / 1e6:.1f} MB, last half {mb / 1e6:.1f} MB"))
    else:
        out.append(("I5", "memory bounded", None, "tracemalloc not enabled"))

    ma = min(r["alive_min"] for r in recs)
    out.append(("I6", ">= 2 factions alive", ma >= 2, f"fewest factions alive at any turn: {ma}"))

    bad, det = [], []
    for key in ECON_KEYS:
        ys = [r[key] for r in half_b]
        mean_all, mean_b = _mean(r[key] for r in recs), _mean(ys)
        drift = abs(_slope(ys)) * len(ys)
        collapse = min(ys) <= 0.05 * mean_all
        runaway = mean_b > 0 and drift >= 0.5 * mean_b
        if n >= 4 and (collapse or runaway):
            bad.append(key)
        det.append(f"{key} {mean_b:.0f} (drift {100 * drift / mean_b if mean_b else 0:.0f}%)")
    out.append(("I7", "economy not degenerate", None if n < 4 else not bad,
                ("degenerate: " + ", ".join(bad) + "; " if bad else "") + "; ".join(det)))

    neg = sum(r.get("negative_treasury", 0) for r in recs)
    bail = recs[-1].get("bailouts", 0)
    out.append(("I8", "treasury never negative", neg == 0,
                f"negative treasuries {neg}; debts forgiven (engine fallback) {bail}"))

    rn, ro = sum(r["replay_n"] for r in recs), sum(r["replay_ok"] for r in recs)
    out.append(("I9", "trace replay OK", rn > 0 and ro == rn, f"{ro}/{rn} sampled decisions replayed OK"))

    ab = sum(r["abstain"] for r in recs)
    dec = sum(r["decisions"] for r in recs)
    out.append(("I10", "no abstentions", ab == 0, f"{ab} abstentions in {dec} decisions"))

    sl = [r["saveload_ok"] for r in recs if r.get("saveload_ok") is not None]
    out.append(("I11", "save/load exact", (all(sl) if sl else None), f"{sum(sl)}/{len(sl)} round-trips identical"))
    return out


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def record(game, alive_min, exceptions=0, negative_treasury=0, mem=None, saveload_ok=None, state_bytes=None):
    """One checkpoint from the game's current state and its Metrics window (the caller resets the window after)."""
    m = game.m
    fs = game.alive()
    L = game.learner
    curve = [c for c in L.curve if c[1] is not None]
    last = curve[-1] if curve else [game.turn, None, None, 0, 0.0]
    return {
        "turn": game.turn,
        "turn_ms_med": round(_pct(m.turn_ms, 0.5), 3), "turn_ms_p99": round(_pct(m.turn_ms, 0.99), 3),
        "dec_ms_med": round(_pct(m.dec_ms, 0.5), 4), "dec_ms_p99": round(_pct(m.dec_ms, 0.99), 4),
        "decisions": m.n, "dec_per_turn": round(m.n / max(1, len(m.turn_ms)), 2),
        "vetoes_per_100": round(100 * m.forced / max(1, m.n), 2), "forced_by": dict(m.forced_by),
        "early_exit": m.early, "skipped_steps": m.skipped_steps, "abstain": m.abstain, "part_errors": m.part_errors,
        "masked": m.masked, "fallback": m.fallback, "by_q": dict(m.by_q),
        "replay_n": m.replay_n, "replay_ok": m.replay_ok, "exceptions": exceptions,
        "state_bytes": state_bytes if state_bytes is not None else len(game.to_json()),
        "mem_cur": mem[0] if mem else None, "mem_peak": mem[1] if mem else None,
        "alive": len(fs), "alive_min": alive_min, "spawned": game.spawned, "eliminated": game.eliminated,
        "pop": sum(c["pop"] for c in game.cities.values()), "cities": len(game.cities), "units": len(game.units),
        "gold": sum(game.factions[f]["gold"] for f in fs), "wood": sum(game.factions[f]["wood"] for f in fs),
        "stone": sum(game.factions[f]["stone"] for f in fs),
        "food_store": sum(c["food"] for c in game.cities.values()),
        "deposit_amt": sum(game.amt[i] for i in range(len(game.amt)) if game.dep[i]),
        "deposits": sum(1 for d in game.dep if d), "wars": len(game.wars),
        "negative_treasury": negative_treasury, "bailouts": game.counters["bailouts"],
        "counters": dict(game.counters), "log_len": len(game.log), "cards": len(game.last),
        "adaptive_share": round(game._score_share(), 3),
        "adaptive_reward": last[1], "others_reward": last[2], "teaches": L.teaches, "refits": L.refits,
        "explored": L.explored, "head_n": getattr(game.systems["adaptive"][1].heads.get("build"), "n", None),
        "saveload_ok": saveload_ok,
        "personalities": sorted(game.factions[f]["pers"] for f in fs),
    }
