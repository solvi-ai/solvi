"""What solvi.drift.DriftMonitor flags on simulated streams of decisions (the window tests and the sequential test):
  stationary  1,152 unchanged streams of 1,000 decisions (four parts of 288): answers 3 / 12 / 57, reference 1,500 calibration decisions or
              the stream's first 100, three shapes of confidence (beta(5,2), beta(2,2), beta(20,2)), act beta(8,2);
              every flag is false (the promise: at most alpha = 1% of streams flagged within the horizon)
  answered    the share answered alone 73% → 13% at decision 500 (window 50 / 100 / 200), 50 streams each
  mix         the mix of three answers 1:1:1 → 1:8:1 at decision 500, 50 streams each

    uv run python benchmarks/drift_simulation.py delays                 # ~5 minutes
    uv run python benchmarks/drift_simulation.py stationary 0           # one of four parts (300 streams), ~10 minutes
    uv run python benchmarks/drift_simulation.py stationary 0 --json out.json

The guide's drift section and best_practices.md quote these numbers."""
import json
import sys

import numpy as np

from solvi.drift import DriftMonitor

SHAPES = ((5, 2), (2, 2), (20, 2))


def iid(rng, n, k, shape, alone_at=0.5, weights=None):
    p = None if weights is None else np.array(weights, float) / sum(weights)
    out = []
    for _ in range(n):
        conf = float(rng.beta(*shape))
        out.append({"value": int(rng.choice(k, p=p)), "confidence": conf, "alone": conf > alone_at,
                    "act": float(rng.beta(8, 2))})
    return out


def stationary(part, parts=4):
    rng = np.random.default_rng(1000 + part)
    cases = [(k, ref, sh) for k in (3, 12, 57) for ref in (1500, 0) for sh in SHAPES]
    flagged, total, signals, cases_hit = 0, 0, {}, []
    per = 1200 // len(cases) // parts                           # 16 per case and part: 1,152 streams in four parts
    for k, ref, sh in cases:
        for _ in range(per):
            mon = DriftMonitor()
            if ref:
                mon.set_reference(iid(rng, ref, k, sh))
            reps = [mon.observe(d) for d in iid(rng, 1000 + (0 if ref else 100), k, sh)]
            hit = next((r for r in reps if r["drift"]), None)
            total += 1
            if hit:
                flagged += 1
                cases_hit.append([k, ref, list(sh), hit["seen"], hit["why"]])
                for f in hit["flags"]:
                    signals[f] = signals.get(f, 0) + 1
    out = {"streams": total, "flagged": flagged, "by_signal": signals, "cases": cases_hit}
    print(json.dumps(out), flush=True)
    return out


def delays():
    rng = np.random.default_rng(7)
    out = {}
    for window in (50, 100, 200):
        for kind in ("answered", "mix"):
            d = []
            for _ in range(50):
                mon = DriftMonitor(window=window)
                if kind == "answered":       # beta(5,2) > 0.5 answers ~89%; these thresholds make it 73% → 13%
                    pre = iid(rng, 500 + window, 3, (5, 2), alone_at=0.62)
                    post = iid(rng, 500, 3, (5, 2), alone_at=0.89)
                else:
                    pre = iid(rng, 500 + window, 3, (5, 2))
                    post = iid(rng, 500, 3, (5, 2), weights=(1, 8, 1))
                reps = [mon.observe(x) for x in pre + post]
                flags = [i for i, r in enumerate(reps) if r["drift"]]
                first = next((i for i in flags if i >= len(pre)), None)
                d.append(None if first is None else first - len(pre) + 1)
                if flags and flags[0] < len(pre):
                    d[-1] = "false"
            ok = sorted(x for x in d if isinstance(x, int))
            out[f"{kind} window={window}"] = {"median": ok[len(ok) // 2] if ok else None, "missed": d.count(None),
                                              "false": d.count("false")}
            print(kind, window, out[f"{kind} window={window}"], flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[args.index("--json") + 1] if "--json" in args else None
    res = stationary(int(args[1])) if args and args[0] == "stationary" else delays()
    if path:
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
