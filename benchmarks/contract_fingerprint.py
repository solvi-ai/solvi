"""What reading a contract's `applicable` source once per code object saves on `ask`.

    uv run python benchmarks/contract_fingerprint.py              # 2,000 asks per arm, 5 rounds
    uv run python benchmarks/contract_fingerprint.py --asks 500 --rounds 3 --json out.json

A System of one rule with a contract (`cat.contract(q, applicable=...)`): every `ask` builds the questions fingerprint,
which takes each contract's fingerprint. Two arms on the same inputs, alternated per round:
- `memo` — the source of `applicable` read once per code object (the default);
- `no memo` — the memo emptied before every ask, so the source is read on every ask (what `ask` did before).
Prints the median per-ask wall time of each arm and the change. Numbers vary with the machine and with the size of the
module `applicable` lives in (inspect.getsource scans it); compare the arms within one run."""
from __future__ import annotations

import argparse
import json
import statistics
import time


def _system():
    from solvi import Answer, Catalog, Question, System

    def applicable(text):
        return len(text) < 1000
    cat = Catalog()

    @cat.rule("short")
    def short(text: str) -> str:
        return "yes" if len(text) < 20 else "no"
    cat.contract("short", applicable=applicable)
    return System(cat, [Question("short", "short?", Answer.yes_no())])


def _run(s, states, clear):
    from solvi.core import contract
    out = []
    for st in states:
        if clear:
            contract._CODE.clear()
        t = time.perf_counter()
        s.ask(st, store=False)
        out.append(time.perf_counter() - t)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--asks", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    s = _system()
    states = [{"text": f"hello world {i}"} for i in range(a.asks)]
    _run(s, states[:100], False)                       # warm-up
    times = {"memo": [], "no memo": []}
    answers = {}
    for r in range(a.rounds):
        for arm in (("memo", "no memo") if r % 2 == 0 else ("no memo", "memo")):
            times[arm] += _run(s, states, arm == "no memo")
            answers[arm] = [s.ask(st, store=False)["short"].answer for st in states[:50]]
    assert answers["memo"] == answers["no memo"]       # the same answers
    med = {k: statistics.median(v) * 1e6 for k, v in times.items()}
    change = (med["memo"] - med["no memo"]) / med["no memo"]
    print(f"asks per arm: {a.asks * a.rounds}")
    for k, v in med.items():
        print(f"  {k:8s} median ask {v:8.1f} µs")
    print(f"  change of the median ask time with the memo: {change:+.1%}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"asks": a.asks * a.rounds, "median_us": med, "change": change}, f, indent=2)


if __name__ == "__main__":
    main()
