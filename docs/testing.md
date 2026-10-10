# Decision regression tests: `solvi test`

A catalog changes over time: a rule is edited, a threshold moves, a model is swapped. The cases you have already decided
should keep their answers. solvi runs them from `cases.json` files. This is the format every
[gallery](../gallery) entry uses.

## The format

A `cases.json` sits next to the `task.py` it tests. The task module defines `cat` and `QUESTIONS` (or `system()`), and
optionally `prepare(state)`. The file is a list of cases:

```json
[
  {"name": "legal threat",
   "state": {"message": "... my lawyer takes this to small claims court.", "tier": "standard"},
   "expected": {"priority": "high", "tags": ["refund_request", "legal_threat"], "route": "legal"},
   "status": {"priority": "forced"},
   "safeguards": {"priority": ["hard_check"]},
   "ask": ["priority", "tags", "route"]}
]
```

- `expected`: the answer per question, written as in solvi's JSON: an option, a list for multi-label, a number, or
  `"<not stated>"` for `solvi.Unknown`. `null` or `"abstain"` means the question must abstain.
- `status` (optional): `ok`, `forced` or `abstain` per question.
- `safeguards` (optional): the exact safeguard kinds that fire for a question (`hard_check`, `grounding`,
  `outside_options`, `low_confidence`, `escalated`, …). A plain list checks the whole response.
- `ask` (optional): ask only these questions. By default every question is asked.

Every response's trace must also replay (`res.trace.replay(system)`).

The file can also be an object: `{"task": "path/to/task.py", "system": "run.py:system", "cases": [...]}`. Paths are
relative to the file. `system` names a function that builds the System, for when the catalog alone is not enough, for
example when a head is fitted or a rule list is learned from examples (gallery 09 and 12). Files named `*.cases.json`
are collected too.

## Running

```bash
solvi test gallery/                       # every cases.json under a directory; exits 1 on a mismatch
solvi test gallery/07_kyc_aml/cases.json -q
solvi test gallery/ --json                # results as data
solvi test gallery/ --fuzz 30             # also 30 mutated inputs per case (see below)
solvi test . --store                      # also save the cases' decisions to the system's own store
```

From Python, use `solvi.testing.run_path(paths)` to get a `FileResult` per file and a `CaseResult` per case (`ok`,
`problems`, `answers`). To run the cases inside your own pytest suite, one test is enough:

```python
from solvi.testing import run_path

def test_decision_cases():
    files = run_path(["cases/"])
    bad = [(str(f.path), f.error, [c.problems for c in f.cases if not c.ok]) for f in files if not f.ok]
    assert files and not bad, bad
```

(solvi had a pytest plugin that collected every case as a test item; it was removed in 1.0 because it loaded in every
pytest session wherever solvi was installed.)

A test input is not a decision: when the System has a store (`System(storage=...)`), `solvi test`,
`--fuzz` and `solvi honesty` write nothing to it. `solvi test --store` (`run_path(paths, store=True)`) saves the cases'
decisions — never the fuzz mutations.

## Fuzzing

`--fuzz N` asks N mutated copies of each case's input (after `prepare`). A mutation removes a field, sets it to `None`,
gives it another type (text for a number, a list, a dict, NaN, inf, a huge number), cuts or garbles text, adds an unknown
field, or does one of these one level deep inside a dict or a list. Abstaining on such input is fine. The case fails
when:

- an exception escapes `System.ask` (a crash), or
- an answer is given with a confidence that is not a number in [0, 1] (NaN, inf).

`solvi.testing.mutations(state, n, seed)` and `solvi.testing.fuzz(system, state, n, seed)` are deterministic for a seed.

## Corruption tests for specialists

A specialist (`solvi.core.specialist`: a model proposes a typed spec, code checks it against the source, code
renders) is only as good as its check. `solvi.testing.corrupt.corrupt` measures the check: it corrupts recorded
proposals as if the proposer had misread its source, runs the specialist again on the same source with each corrupted
proposal, and counts how many corruptions the check catches.

```python
from solvi.testing.corrupt import change_digit, corrupt, drop, swap

records = [(rid, source, run.proposal) for rid, source, run in my_runs]      # the proposals as recorded
rep = corrupt(specialist, records, {
    "digit": change_digit(paths=r"^/series/\d+/points/\d+/value$"),     # one digit of a value
    "swap": swap(paths=r"^/series/\d+/points/\d+/value$"),              # two values of one series exchanged
    "drop": drop(paths=r"^/series/\d+/points/\d+$"),                    # a point left out
}, n=300, seed=0)
print(rep)                    # corruptions: caught of n (rate), skipped draws, per kind; rep.rows
```

A mutator is `(proposal_dict, rng) → (mutated_dict, path, value)` or `None` (no target in that proposal: the draw is
skipped and counted); paths are JSON pointers into the proposal's JSON form. A digit change or a swap is caught when
the run has no output or the mutated value is not accepted at its place (`accepted(run)`: by default the leaves of the
verified spec). An omission has no place of its own to be refused at — it is caught only through a relation the check
holds (a total that no longer adds up, shares that no longer make a whole), so `drop` counts a mutation caught when the
run reports an issue the unmutated run did not. Draws are deterministic: records sorted by id, kinds in order, one
`random.Random(seed)`. On the charts specialist (`tests/test_corrupt.py`) a changed digit is caught every time (the
quote no longer states the number), and a dropped bar of a chart without a total is never caught: nothing relates the
bars.

`solvi.testing.conformance.check_specialist(specialist, sources, inapplicable=())` checks the contract itself: two
runs give identical bytes, a run replays from its record, the record round-trips through JSON, an invalid proposal is
blocked with no output, an output is a text and never the source itself, and on each source in `inapplicable` the run
is blocked as `not_applicable` without a call to the proposer (`Specialist.applicable`).
