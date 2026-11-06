"""One isolated actor; source paths and job come from the frozen host launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path[:0] = [str(source / "src"), str(source / "benchmarks")]
    job = json.loads(args.job.read_text())
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    def emit(kind, value):
        with args.journal.open("a") as stream:
            stream.write(json.dumps(dict(kind=kind, payload=value), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    import solvi.experimental.experience.refinement as implementation
    expected = source / "src/solvi/experimental/experience/refinement.py"
    if Path(implementation.__file__).resolve() != expected:
        raise ValueError("foreign actor module origin")
    if job["tier"] == "program":
        from refinement_bench import trial
        from refinement_tasks import tasks
        problem = next(p for p in tasks() if p.task_id == job["task"])
        payload = trial(problem, job["seed"], "ACTIVE")
    elif job["tier"] == "real":
        from active_real_trial import real_trial
        payload = real_trial(job["dataset"], args.data, job["seed"], job["split"], job["arm"],
                             job["price"], cap=job["cap"], emit=emit)
    else:
        raise ValueError("unknown actor tier")
    result = dict(job=job, payload=payload, origin=str(expected),
                  actor_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
