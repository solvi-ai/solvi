"""NAB baseline, no solvi: a point is an alert when it is more than k standard deviations from the mean of the trailing
window. k and the window are picked on the dev series (best F1), then applied to eval.

    python nab/baseline.py [--out runs/baseline_eval.jsonl]"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import DATA, read_jsonl, write_jsonl  # noqa: E402
from score import load, score  # noqa: E402


def alerts(ts, xs, w, k):
    out, win, s, s2 = [], deque(), 0.0, 0.0
    for t, x in zip(ts, xs):
        if len(win) >= w // 2:
            mean = s / len(win)
            sd = max((s2 / len(win) - mean * mean), 0.0) ** 0.5
            if sd > 0 and abs(x - mean) > k * sd:
                out.append(t)
        win.append(x)
        s += x
        s2 += x * x
        if len(win) > w:
            y = win.popleft()
            s -= y
            s2 -= y * y
    return out


def run(split, w, k, out):
    rows = []
    for m in read_jsonl(DATA / "nab/prepared/series.jsonl"):
        if m["split"] == split:
            ts, xs = load(m["series"])
            rows.append({"series": m["series"], "alerts": alerts(ts, xs, w, k)})
    return write_jsonl(out, rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(Path(__file__).parent / "runs" / "baseline_eval.jsonl"))
    a = p.parse_args()
    tune = Path(a.out).with_name("baseline_tune_dev.jsonl")
    best = max((score(run("dev", w, k, tune), "dev")["total"]["f1"], w, k) for w in (144, 288, 576, 1152) for k in (3, 4, 5, 6, 8))
    print("picked on dev: f1 %.3f, window %d, k %d" % best)
    print(json.dumps(score(run("eval", best[1], best[2], a.out))["total"], indent=1))
