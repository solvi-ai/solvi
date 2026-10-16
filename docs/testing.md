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
solvi test . --timeout 5 --reruns 2       # stop a case after 5 s; ask a failing case twice more
solvi test . --isolate --max-mem 2G       # each case in its own process, at most 2 GB of memory
```

From Python, use `solvi.testing.run_path(paths)` to get a `FileResult` per file and a `CaseResult` per case (`ok`,
`problems`, `answers`, `status`). A failing case's status says what kind of failure it is: `regression` (an answer,
status, safeguard or the replay is not what the case expects), `nondeterministic` (a step of the fresh trace does not
recompute on replay, or — with `--reruns N` / `run_path(paths, reruns=N)` — asking the case again gave other answers),
`timeout` (longer than `--timeout S` / `timeout=S` seconds; stopped with SIGALRM, so POSIX and the main thread only),
`memory` (more than `--max-mem`, see below), `error` (ask raised) or `invalid` (the case cannot fail: an unknown key,
nothing expected). `FileResult.counts()` gives the numbers per status, and the summary line of `solvi test` lists them.

`--isolate` (`run_path(paths, isolate=True)`) runs each case in a fresh Python process: the task is imported and the
System built there, never in the runner. No case sees module state another case left behind, and a case that ends the
interpreter (a crash in an extension, `os._exit`) fails alone with status `error` while the next case runs. With
`--timeout` the case's process is killed when the time is up, so a case stuck inside a C call stops too, and it works
where SIGALRM does not. The price is a process start, the imports and the System's build per case (about 90 ms per case
on a small task). `--max-mem SIZE` (`max_mem="2G"`, or bytes; needs `--isolate`) caps the case's process: an
allocation past it raises MemoryError and the case has status `memory`. The cap is the address space (RLIMIT_AS) of
the whole process — Python, solvi, the task and the case — and numpy with a threaded BLAS reserves about 1 GB of it at
import, so leave room. macOS and Windows do not enforce RLIMIT_AS: there the process's peak resident memory is compared
with the cap after the case instead (a case over it still fails with `memory`, but it was not stopped), and a warning
says so. To run the cases inside your own pytest suite, one test is enough:

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

## Cases from recorded decisions and boundary inputs

Cases do not have to be written by hand. `testing.cases_from_store(storage, n=None)` turns the decisions a store holds
(full or compact records) into cases: the recorded input as `state`, the answers given as `expected`, each question's
status, the questions asked. The same input twice is one case; with `n`, every distinct outcome comes first. Running them
after a change is a regression test against what the System actually decided.

`testing.boundary(cases, n=50, system=None, thresholds=None)` makes inputs at the edges of what the cases show: per
field, a number just past the largest and smallest seen, 0, -1, a huge positive and a huge negative one; empty,
one-character, long, non-ASCII or padded text, the same accents decomposed (NFD) and a compatibility ligature ("ﬁß");
None where the field has shown None and a value of another type the field has shown (7 → "7": the str / int confusion
of states loaded from JSON); an empty, one-element or long list, a long list or set of distinct values (a repeated
element does not reach code that dedupes or indexes), a dict with many keys; the same one level into dicts and lists.
With `system=`, a number is also cut at each constant a rule compares the field with: `amount >= 100` gives 99, 100 and
101, `score > 0.5` the nearest floats on either side of 0.5 and 0.5 itself. The constants are read from the parts'
source where an argument is compared with a number or a numeric constant of the module or closure (`amount > LIMIT`);
an expression (`x * 2 > 100`), an item (`row["x"] > 1`) and the helpers a part calls are not read. `thresholds=` gives
them by hand (`{"amount": [100]}`) or turns them off (`False`). The changes are taken in turn from each kind and, within a kind, from the
first case of each input shape (field → type) first, so many cases of one shape do not crowd out the others. With `system=`, each input is asked (never stored)
and its current answers are pinned as `expected`, so a later change that moves an answer at an edge — a threshold
`amount < 100` becoming `<= 100` — fails a case even when the hand-written cases all pass. Inputs whose ask raises are
left out (that is what `--fuzz` is for), and so are inputs whose answers are nondeterministic.

```python
import json
from solvi import testing

cases = testing.cases_from_store(system.storage, n=200)
cases += testing.boundary(cases, n=100, system=system)
open("recorded.cases.json", "w").write(json.dumps({"task": "task.py", "cases": cases}, indent=1))
```

### Metamorphic cases

Some answers have no fixed expected value at a new input, but must keep a relation to the answer at a known one: the
same dose whether the weight is given in kilograms or grams, the same route whatever the order of the items. A
`testing.Relation(name, transform, expect=None)` says this: `transform(state)` gives the changed state (or None when it
does not apply), `expect(answers)` the answers the changed state must give, from the source state's answers (by
default the same answers). `testing.metamorphic(cases, relations, system=None)` makes the cases: each case's state
transformed, with `expected` from the System's current answers on the source state (with `system=`, never stored) or
from the case's own `expected`. A System that breaks the relation fails them as regressions, though no hand-written case
pins the converted input.

```python
to_grams = testing.Relation("kg to g",
                            lambda s: {**s, "weight": s["weight"] * 1000, "unit": "g"} if s["unit"] == "kg" else None)
heavier = testing.Relation("much heavier", lambda s: {**s, "weight": s["weight"] + 500},
                           expect=lambda a: {"heavy": "yes"})          # only "heavy" is checked
cases += testing.metamorphic(cases, [to_grams, heavier], system=system)
```

Numbers are compared up to rounding (1e-9), so a relation that scales a numeric answer gives it scaled in `expect`; a
rounded answer is rounded there too.

## Verdicts across runs, known bugs, a degraded environment

On a live environment one pass or fail says little. `solvi.testing.history` keeps the outcomes of the same cases over
runs (in memory, or in any store: one chained record per run):

```python
from solvi.testing import run_path
from solvi.testing.history import KnownBugs, VerdictHistory, run_status

results = run_path("cases")
run_status(results)          # {"status": "degraded", ...} when many cases are blocked at once (errors, timeouts)
history = VerdictHistory("verdicts.jsonl", window=10)
history.gate(results)        # the failures that count; flaky cases are quarantined
history.add(results, skip_degraded=True)
history.report()             # per case: stable / flaky / regression / fixed, passes on a later attempt
history.stable_share()       # the share of cases with the same outcome in each of their last runs
bugs = KnownBugs("verdicts.jsonl", quiet=5)
bugs.register("BUG-12", r"expected 'high', got 'low'")
bugs.triage(results)         # known bug per failing case, new failures, a fixed bug seen again, open bugs not seen lately
```

A case is "flaky" when its outcome changed more than once in the window, it was "nondeterministic", or it passed on a
later attempt (`add({key: {"status": "ok", "attempt": 2}})`); "regression" / "fixed" when it changed once. Nothing is
re-run by these: the runs, and any retries, are yours.

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
runs give identical bytes (skipped with `deterministic=False`, for a model proposer that does not repeat itself), a run
replays from its record — to the same bytes and without calling the proposer (a replay that calls it fails the
check) —, the record round-trips through JSON, an invalid proposal is blocked with no output, an output is a text and
never the source itself, and on each source in `inapplicable` the run is blocked as `not_applicable` without a call
to the proposer (`Specialist.applicable`).

`solvi.testing.conformance.check_pipeline(make_pipeline, make_change)` runs the usual attacks of an adapting proposer
on a change pipeline (`solvi.experimental.changes`), each family on a fresh pipeline: the same harmful change replayed
under new seeds until noise lets one through (`pool_replay`), a chain of small changes (`salami`), harm in one slice
under an overall gain (`slice_harm`), a change that answers what the system hands over (`refusal_bypass`), a rejected
change with a nonce added (`nonce`), and a change that makes a stage raise (`crash`), plus a benign `control` that must
be promoted. `make_change(family, i)` builds the i-th change of a family in your schema (None: the family ends, or is
skipped). The report names the stage that caught each proposal, or "passed"; with `strict=True` (default) a passed
attack or a refused control fails the check. On the toy in `tests/test_conformance_pipeline.py` an aggregate gate
passes `salami` and `slice_harm`; `SliceGuard` catches both, `RefusalRegression` the bypass, a failure memory keyed on
what the change does the nonce and every replay after the first rejection, and a raising stage ends as "error" both
times. A lucky first draw of a replay is still promoted there: no stage corrects for the number of candidates tried.
