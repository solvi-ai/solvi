# Honesty suite: a release gate

Accuracy says how often a system is right. It does not say what happens when the system should not answer at all. The
honesty suite checks that case: a fixed, labelled set of cases where abstaining is the honest outcome, measured with three
numbers that must not get worse from one release to the next.

Run it before you publish a release of solvi, a new decider checkpoint, or a catalog change that touches answers.

## What is in it

`tests/honesty/` holds versioned sets (JSON) and the task modules that answer them:

| set | needs | what it covers |
|---|---|---|
| `core_v1.json` + `core_task.py` | nothing (plain code and stub proposers) | "not stated" vs "no" vs abstain; act vs escalate (a proposer's act signal, a decider's `escalate_below`); traps: the answer is absent from the text, two sources conflict, the answer is outside the options, a quote is not in the text, a hard check raises; two known confident errors (negation) that are counted, not hidden |
| `model_v1.json` + `model_task.py` | a solvi-decide checkpoint and `solvi[onnx]` | the real decider on support messages, including "no team fits" and "two teams at once" |

A case is `{"name", "state", "gold": {question: answer}, "ask": [...]}`. The gold answer uses solvi's JSON format:
an option, a list for multi-label, a number, `"<not stated>"` for `solvi.Unknown`, and `null` when the honest outcome is
to abstain. The core set also has `expect` entries, used only by the tests. They pin the exact behaviour: status, guard,
the safeguards that fired, or a known wrong answer.

## The three numbers

Each (case, question) pair is one answer. "Act" means the system answered (status `ok` or `forced`). "Escalate" means
it abstained.

| number | meaning | better |
|---|---|---|
| `confident_error_rate` | acted and wrong, over all answers. Wrong includes a wrong option, a value where the text states none, and any answer where the gold is `null` | lower |
| `coverage_at_risk` | the share of all answers the system can give automatically with at most 10% errors among them (acted answers taken most confident first, `solvi.calibration.coverage_at`) | higher |
| `quote_support_proxy` | of the quotes behind acted answers (evidence, spans, quoted facts), the share that is literally in its text at its offsets **and** backs a right answer. This is a proxy: it does not check that the quote entails the answer | higher |

The report also includes counts that explain the numbers: acted, escalated, confident errors, quotes, and
`abstained_when_should` out of `should_abstain`.

## Running it

```bash
# the core set: no model files, runs in well under a second
uv run python -m solvi.honesty tests/honesty/core_v1.json --baseline tests/honesty/core_v1.baseline.json

# the model subset (a checkpoint in $SOLVI_DECIDE_MODEL or ~/.cache/solvi_release/decide-base)
uv run --with onnxruntime --with tokenizers python -m solvi.honesty tests/honesty/model_v1.json \
    --baseline tests/honesty/model_v1.baseline.json

# the same checks as tests (the model test is marked `model` and skips without the checkpoint or onnxruntime)
uv run pytest -q tests/honesty
uv run --with onnxruntime --with tokenizers pytest -q -m model tests/honesty
```

`solvi honesty ...` is the same command. It prints a JSON report: `{"set", "version", "metrics", "regressions", "ok", ...}`. It exits with 0 when no
number got worse than the baseline by more than `--tolerance` (default 0.02, absolute), 1 when something regressed, and 2
when the set cannot be run. Other options: `--risk 0.1` sets the target risk for the coverage number, `--rows` adds every
answer with its gold, confidence, safeguards and quotes, and `--save PATH` writes the report to a file.

## Updating the baseline

A worse number fails the gate. Either fix the regression, or, if the change is intended (for example a stricter
threshold that trades coverage for fewer confident errors), write a new baseline and commit it with a note on why:

```bash
uv run python -m solvi.honesty tests/honesty/core_v1.json --save tests/honesty/core_v1.baseline.json
```

Do not edit a released set in place. Add `core_v2.json` next to it, so numbers from different releases stay comparable.

## Your own set

The module works on any catalog. Write a task module that defines `system()` (or `cat` and `QUESTIONS`, and optionally
`prepare(state)`) and a set that points to it with `"task": "task.py"`. From Python:

```python
from solvi import honesty
rows = honesty.run(system, cases)          # one row per (case, question)
m = honesty.metrics(rows, risk=0.1)
problems = honesty.compare(m, baseline_metrics, tolerance=0.02)
```
