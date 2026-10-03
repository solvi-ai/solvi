"""NAB with solvi: alerts on numeric series (server load, taxi demand, traffic, mentions), where every point after the
warm-up is one decision of a solvi System.

How it is solved:
  - the input of a decision is plain data built by the stream loop: the point, the 576 points before it, the same time
    of day on the previous 7 days, the trailing residuals against that daily profile, and the series' last 2000 scores
    (its memory, given like an agent's episode) — so a stored alert replays from its own record;
  - the catalog: a hard check (enough history) → the trailing level and robust scale → the daily profile → the score
    (against the daily profile where it explains the series, else against the level) → the threshold, the conformal
    quantile of the series' own earlier scores (`solvi.core.calibration.conformal_quantile`: at most `alpha` of points like
    the earlier ones exceed it) → a floor check → the rule `alert`;
  - only the alerts are stored (SQLiteStorage); the chain is verified, every alert replayed, and `solvi.core.store.diff` says
    which alerts a three times stricter alpha would drop.
The settings (score, horizon, alpha, floor) are the ones chosen on the dev series. The arithmetic is facts.py.

    uv run python nab/solution.py [--split eval] [--out runs/solvi_eval.jsonl]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import facts as F  # noqa: E402
from common.llm import DATA, read_jsonl  # noqa: E402
from score import load, score  # noqa: E402

from solvi import Answer, Catalog, Question, System  # noqa: E402
from solvi.core.store import SQLiteStorage  # noqa: E402
from solvi.core.calibration import conformal_quantile  # noqa: E402
from solvi.core.store.diff import diff  # noqa: E402

PICKED = {"horizon": 2000, "alpha": 0.0015, "floor": 5.0}     # chosen on dev


def build(alpha=PICKED["alpha"], floor=PICKED["floor"], storage=None):
    cat = Catalog()

    @cat.check(hard=True, then={"alert": "no"})
    def enough_history(recent):
        """without 48 earlier points nothing is unusual yet"""
        return len(recent) >= F.MIN_HIST

    @cat.fn
    def level(value, recent):
        """the trailing level: median and robust scale of the points before this one"""
        r = np.asarray(recent, float)
        med = float(np.median(r))
        scale = float(F.floor_scale(float(F.robust_scale(r)), med))
        return {"median": med, "scale": scale, "z": (value - med) / scale}

    @cat.fn
    def day_profile(value, same_time_days, day_residuals, level):
        """the same time of day over the previous days, and how far this point is from it; `explains`: the share of the
        trailing scale the daily profile removes"""
        lags = [v for v in same_time_days if v is not None]
        res = np.array([np.nan if v is None else v for v in day_residuals], float)
        if len(lags) < 2 or int((~np.isnan(res)).sum()) < F.MIN_HIST:
            return {"available": False}
        scale = float(F.robust_scale(res))
        return {"available": True, "z": (value - float(np.median(lags))) / float(F.floor_scale(scale, level["median"])),
                "explains": 1.0 - scale / level["scale"]}

    @cat.fn
    def score(level, day_profile):
        """how unusual the point is: against the daily profile where it explains the series, else against the level"""
        if day_profile["available"] and day_profile["explains"] > F.EXPLAINS:
            return min(abs(day_profile["z"]), 1e9)
        return min(abs(level["z"]), 1e9)

    @cat.fn
    def threshold(past_scores):
        """the score that at most alpha of points like the earlier ones exceed (infinite while the series is short)"""
        return min(float(conformal_quantile(past_scores, alpha)), 1e308)

    @cat.check
    def above_floor(score):
        return score > floor

    @cat.rule("alert")
    def alert(score, threshold, above_floor):
        return bool(above_floor and score > threshold)

    q = Question("alert", "Raise an alert at this point?", Answer.yes_no(), requires=["enough_history"])
    return System(cat, [q], storage=storage)


def stream(name):
    """One series as a stream: → (timestamps, the input of the decision at point i)."""
    ts, xs = load(name)
    x = np.asarray(xs, float)
    g, day, dense = F.grid(ts, xs)
    expected, scores = F.series_scores(ts, xs)
    resid = x - expected

    def state(i):
        lo = max(0, i - F.W)
        lags = [None if g[i] - day * k < 0 or np.isnan(dense[g[i] - day * k]) else float(dense[g[i] - day * k])
                for k in range(1, F.K_DAYS + 1)]
        return {"series": name, "time": ts[i], "value": float(x[i]), "recent": [float(v) for v in x[lo:i]],
                "same_time_days": lags, "day_residuals": [None if np.isnan(v) else float(v) for v in resid[lo:i]],
                "past_scores": [float(v) for v in scores[max(0, i - PICKED["horizon"]):i]]}
    return ts, state


def main(split, out):
    out = Path(out)
    db = out.with_suffix(".db")
    for p in (out, db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
        p.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteStorage(db)
    system = build(storage=store)
    n_asked, t0 = 0, time.perf_counter()
    for m in read_jsonl(DATA / "nab/prepared/series.jsonl"):
        if m["split"] != split:
            continue
        ts, state = stream(m["series"])
        alerts = []
        for i in range(int(0.15 * len(ts)), len(ts)):                  # the scorer ignores the warm-up: not asked
            res = system.ask(state(i), store=False)
            n_asked += 1
            if res["alert"].answer == "yes":
                alerts.append(ts[i])
                store.save(res, meta={"series": m["series"], "index": i})
        with open(out, "a") as f:
            f.write(json.dumps({"series": m["series"], "alerts": alerts}) + "\n")
        print(f"  {m['series']:58s} {len(alerts):4d} alert points", flush=True)
    ms = 1000 * (time.perf_counter() - t0) / max(n_asked, 1)
    stricter = diff(store, build(alpha=PICKED["alpha"] / 3))
    print(json.dumps({"decisions": n_asked, "ms_per_decision": round(ms, 2), "stored_alerts": store.head()["count"],
                      "chain_verified": store.verify()["ok"], "replay_failed": len(store.replay_all(system)),
                      "alerts_a_3x_stricter_alpha_drops": len(stricter.changed)}))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="eval", choices=("dev", "eval"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = main(a.split, a.out or HERE / "runs" / f"solvi_{a.split}.jsonl")
    print(json.dumps(score(out, a.split)["total"], indent=1))
