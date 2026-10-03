"""Banking77 with solvi: a stream of customer requests answered alone under a promised error rate (at most 5% wrong
among the answers given alone), where from request 1,000 on a share of the requests is about intents the decider was
never shown.

How it is solved:
  - the decider is the baseline's classifier with an act head (classifier.py), as a solvi decision part;
  - `solvi.core.guarantees.openset.leave_out` simulates new intents on calib only: three times a third of the known intents is taken
    away from the decider, and its signals on their requests stand in for requests it has no answer for;
  - `OpenSetGate.calibrate` sizes the threshold for a share of such requests and follows the share as the stream goes
    (it raises the threshold before any flag, and flags the change with its CUSUM);
  - `System.guarantee(promise=gate, signal="act")` puts the gate on the question; every decision is stored
    (JSONLStorage), the chain verified and a sample of the decisions replayed.
`--plain` swaps the gate for the plain promise, `System.guarantee(calib, max_error=0.05, signal="act")`, which knows
nothing of new intents. Stream labels are never read: the stream is fed as text.

    uv run --with scikit-learn --with scipy python banking77/solution.py [--plain] [--out runs/solvi.jsonl]"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from classifier import decider, label  # noqa: E402
from common.llm import DATA, read_jsonl, write_jsonl  # noqa: E402
from score import score  # noqa: E402

from solvi import Catalog, JSONLStorage, System  # noqa: E402
from solvi.core.guarantees.openset import OpenSetGate, leave_out  # noqa: E402

MAX_ERROR = 0.05
SEED = 100                     # which intents are left out together in the simulation (fixed before the stream was run)


def main(out, plain=False):
    d = DATA / "banking77/prepared"
    fit, calib = read_jsonl(d / "fit.jsonl"), read_jsonl(d / "calib.jsonl")
    known = json.loads((d / "intents.json").read_text())["known"]
    part = decider(fit, known)
    examples = [(r["text"], label(r["intent"])) for r in calib]

    store = Path(out).with_suffix(".decisions.jsonl")
    for f in (store, Path(str(store) + ".head")):
        f.unlink(missing_ok=True)
    cat = Catalog()
    system = System(cat, [part.question(cat)], storage=JSONLStorage(store))
    if plain:
        gate = None
        system.guarantee("intent", [({"text": x}, y) for x, y in examples], max_error=MAX_ERROR, signal="act")
    else:
        ds = part.decide([x for x, _ in examples])                  # the deployed decider's signals on calib
        signals = [d.extra["act"] for d in ds]
        right = [d.value == y for d, (_, y) in zip(ds, examples)]
        sim = leave_out(examples, lambda kept: decider(fit, [i for i in known if label(i) in set(kept)]), seed=SEED)
        gate = OpenSetGate.calibrate(signals, right, sim["novel"], max_error=MAX_ERROR)
        system.guarantee("intent", promise=gate, signal="act")

    back = {label(i): i for i in known}
    rows, flag_n = [], None
    for x in read_jsonl(d / "stream.jsonl"):
        r = system.ask({"text": x["text"]})["intent"]
        rows.append({"n": x["n"], "intent": back.get(r.answer) if r.status == "ok" else None, "escalate": r.status != "ok"})
        if gate is not None and gate.flag_at is not None and flag_n is None:
            flag_n = x["n"]
    if flag_n is not None:
        rows.append({"drift_flag_at": flag_n, "why": gate.why})
    write_jsonl(out, rows)

    ids = [rec.id for rec in system.storage.query()]
    random.Random(0).shuffle(ids)
    failed = sum(not system.storage.get(i).trace.replay(system)["ok"] for i in ids[:100])
    print(json.dumps({"store_verified": system.storage.verify()["ok"], "replayed": min(100, len(ids)), "replay_failed": failed}))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--plain", action="store_true", help="the plain promise instead of the open-set gate")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or str(HERE / "runs" / ("solvi_plain.jsonl" if a.plain else "solvi.jsonl"))
    print(json.dumps(score(main(out, a.plain)), indent=1))
