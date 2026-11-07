"""Resumable finite four-direction benchmark. See the host-frozen protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import time

from benchmarks import decision_value, decision_cooperation, decision_preferences, decision_teaching

MODULES = {'28':decision_value,'6':decision_cooperation,'15':decision_teaching,'25':decision_preferences}
RUNTIME = ('decision_learning','cooperation','teaching','preferences','_common','programs')


def manifest(seeds,limits,protocol):
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__),*(Path(m.__file__) for m in MODULES.values()),
             *(root/'src/solvi/experimental/experience'/f'{n}.py' for n in RUNTIME)]
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {'version':1,'seeds':seeds,'limits':limits,'hashes':hashes,
            'python':sys.version, 'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'arms':{k:list(m.ARMS) for k,m in MODULES.items()},'units':'declared work; includes fixed independent scoring; not time'}


def _completed_trials(raw):
    """A newline terminates a committed record; never replay a partial append."""
    seen = set()
    if not raw.exists():
        return seen
    valid_bytes = 0
    # Binary file iteration uses LF only; bytes.splitlines also treats CR as a
    # separator and could mistake a corrupt committed row for a partial tail.
    with raw.open('r+b') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                stream.truncate(valid_bytes)
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                raise ValueError('corrupt completed log record') from None
            key = tuple(record[k] for k in ('direction', 'seed', 'arm', 'limit'))
            if key in seen:
                raise ValueError('duplicate trial in raw')
            seen.add(key)
            valid_bytes += len(line)
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(2000,2012)))
    parser.add_argument('--limits',type=int,nargs='+',default=[20000,500000])
    parser.add_argument('--max-trials',type=int)
    parser.add_argument('--protocol',type=Path,required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s<0 for s in args.seeds) or any(limit<0 for limit in args.limits):
        parser.error('distinct nonnegative seeds and nonnegative limits required')
    args.output.mkdir(parents=True,exist_ok=True)
    path = args.output/'manifest.json'
    config = manifest(args.seeds,args.limits,args.protocol)
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('frozen manifest changed; do not overwrite the previous run')
    else:
        path.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
    raw = args.output/'raw.jsonl'
    seen = _completed_trials(raw)
    total = sum(len(m.ARMS) for m in MODULES.values())*len(args.seeds)*len(args.limits)
    done = len(seen)
    start = time.monotonic()
    performed = 0
    with raw.open('a',buffering=1) as stream:
        for direction,module in MODULES.items():
            for seed in args.seeds:
                for arm in module.ARMS:
                    for limit in args.limits:
                        key = (direction,seed,arm,limit)
                        if key in seen:
                            continue
                        before = time.monotonic()
                        try:
                            record = module.run_trial(seed,arm,limit)
                        except Exception as error:  # noqa: BLE001 - save failure before investigation
                            record = {'seed':seed,'arm':arm,'status':'error','error':f'{type(error).__name__}: {error}'}
                        record.update(direction=direction,limit=limit,seconds=time.monotonic()-before)
                        stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                        done += 1
                        performed += 1
                        print(f'{done}/{total} direction={direction} seed={seed} arm={arm} limit={limit} status={record["status"]} elapsed={time.monotonic()-start:.2f}s',flush=True)
                        if args.max_trials is not None and performed >= args.max_trials:
                            return
    print(f'complete {done}/{total}',flush=True)


if __name__ == '__main__':
    main()
