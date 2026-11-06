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
    parser.add_argument("--roles", type=Path, required=True)
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

    import solvi.experimental.experience.packed as implementation
    import packed_spam_trial as actor
    import packed_spam_data as data_actor
    import active_real_models as model_actor
    expected = source / "src/solvi/experimental/experience/packed.py"
    for module, path in ((implementation, expected), (actor, source / "benchmarks/packed_spam_trial.py"),
                         (data_actor, source / "benchmarks/packed_spam_data.py"),
                         (model_actor, source / "benchmarks/active_real_models.py")):
        if Path(module.__file__).resolve() != path:
            raise ValueError("foreign actor module origin")
    if job["tier"] != "packed_real":
        raise ValueError("unknown actor tier")
    payload = actor.real_trial(job["dataset"], args.data, job["seed"], job["split"], job["arm"],
                               job["price"], fold=job["fold"], roles_path=args.roles, emit=emit)
    result = dict(job=job, payload=payload, origin=str(expected),
                  actor_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
