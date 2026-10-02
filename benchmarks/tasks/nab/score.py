"""NAB: alerts on numeric series.

    python nab/score.py runs/<arm>.jsonl [split=eval]

A prediction per series: {"series": "realKnownCause/nyc_taxi.csv", "alerts": [timestamps as in the csv]}.
Alerts in the first 15% of a series are ignored (the detector's warm-up, as in NAB). Alerts closer than 1% of the series
are one event. A window is found when an alert falls in it; an event with no alert in any window is a false alarm."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402


def load(series):
    with open(DATA / "nab/repo/data" / series) as f:
        rows = list(csv.reader(f))[1:]
    return [r[0] for r in rows], [float(r[1]) for r in rows]


def score(pred_file, split="eval"):
    meta = [r for r in read_jsonl(DATA / "nab/prepared/series.jsonl") if r["split"] == split]
    pred = {r["series"]: r for r in read_jsonl(pred_file)}
    tot = {"series": len(meta), "windows": 0, "found": 0, "events": 0, "false_alarms": 0, "points": 0, "missing": 0}
    per = {}
    for m in meta:
        ts, _ = load(m["series"])
        idx = {t: i for i, t in enumerate(ts)}
        warm, merge = int(0.15 * len(ts)), max(1, len(ts) // 100)
        wins = [(idx.get(a[:19], 0), idx.get(b[:19], len(ts) - 1)) for a, b in m["windows"]]
        wins = [w for w in wins if w[1] >= warm]
        p = pred.get(m["series"])
        tot["missing"] += p is None
        al = sorted({idx[str(t)[:19]] for t in (p or {}).get("alerts", []) if str(t)[:19] in idx})
        al = [i for i in al if i >= warm]
        events = []
        for i in al:
            if events and i - events[-1][-1] <= merge:
                events[-1].append(i)
            else:
                events.append([i])
        found = sum(any(a <= i <= b for i in al) for a, b in wins)
        false = sum(not any(a <= i <= b for i in e for a, b in wins) for e in events)
        per[m["series"]] = {"windows": len(wins), "found": found, "events": len(events), "false_alarms": false}
        for k, v in (("windows", len(wins)), ("found", found), ("events", len(events)), ("false_alarms", false), ("points", len(ts))):
            tot[k] += v
    rc = tot["found"] / max(tot["windows"], 1)
    pr = (tot["events"] - tot["false_alarms"]) / max(tot["events"], 1)
    tot.update(recall=round(rc, 3), precision=round(pr, 3), f1=round(2 * pr * rc / max(pr + rc, 1e-9), 3),
               false_alarms_per_10k_points=round(1e4 * tot["false_alarms"] / max(tot["points"], 1), 2))
    return {"total": tot, "per_series": per}


if __name__ == "__main__":
    print(json.dumps(score(*sys.argv[1:])["total"], indent=1))
