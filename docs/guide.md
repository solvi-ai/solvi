# solvi guide

This guide walks through the whole API. For a two-minute overview, see the [README](../README.md).

Contents:

1. [Concepts](#concepts)
2. [Installation](#installation)
3. [The catalog](#the-catalog)
4. [Questions and answer types](#questions-and-answer-types)
5. [Asking: System and Response](#asking-system-and-response)
6. [How the strategist plans a flow](#how-the-strategist-plans-a-flow)
7. [Questions without a rule: fit, learn_rule, teach](#questions-without-a-rule-fit-learn_rule-teach)
8. [Confidence, calibration and abstention](#confidence-calibration-and-abstention)
9. [The trace and verification](#the-trace-and-verification)
10. [Printing results: solvi.show](#printing-results-solvishow)
11. [Extracting fields from documents](#extracting-fields-from-documents)
12. [Guarantees and limitations](#guarantees-and-limitations)

## Concepts

A solvi task has two ingredients:

- a **catalog** of Python functions: computations, checks, extractors and answer rules;
- a list of **questions** with typed answers: yes/no, or a choice from a fixed list.

A request is a plain dict called `init_state` (for example `{"doc": text, "today": date(...)}`). Each key and each catalog
function's output is a **fact**. For every request, the **strategist** picks from the catalog only the parts needed for
the asked questions and orders them into a **flow**. Execution produces `computed_state` (every fact with its origin, and
a quote for extracted values) and a hash-chained **trace**. Each question gets an answer, a confidence, a reason and a
status.

## Installation

```bash
pip install solvi              # core: numpy, scipy
pip install "solvi[model]"     # + torch, transformers, for the ModernBERT extractors (solvi.extract_*)
```

For development, from a clone: `uv sync`, then `uv run pytest`.

## The catalog

```python
from solvi import Catalog, Quote

cat = Catalog()
```

| Decorator | Kind | Returns | Sets fact |
|---|---|---|---|
| `@cat.fn` | computation | any value | function name |
| `@cat.check` | soft check | `bool` | function name |
| `@cat.check(hard=True, then={"question": "answer"})` | hard check | `bool` | function name |
| `@cat.extract` | extraction from text | `Quote(value, start, end, source="doc", confidence=1.0)` | function name |
| `@cat.rule("question")` | answer rule | one of the question's options (a `bool` is accepted for yes/no) | the answer |

**A part's contract is its signature.** You do not write a separate schema:

- argument names are the inputs: keys of `init_state` or facts produced by other parts;
- the function name is the output fact.

Part names must be unique in a catalog (a duplicate raises `ValueError`). A rule is registered per question; registering
a second rule for the same question replaces the first. The decorators return the original function, so parts stay
ordinary, testable Python.

```python
@cat.fn
def total(items):                           # reads "items", sets "total"
    return sum(q * p for q, p in items)

@cat.check
def big_order(total):                       # soft check: a fact like any other
    return total > 1000

@cat.check(hard=True, then={"ship": "no"})  # hard check: if False, the answer to "ship" is forced to "no"
def paid(payment_status):
    return payment_status == "paid"

@cat.rule("ship")
def ship_rule(big_order):
    return not big_order                    # bool is normalized to "yes"/"no" for yes/no questions
```

### Soft and hard checks

- A **soft check** is a boolean fact. Rules can use it, answer heads can learn from it, and failed soft checks are listed
  in the reason of a learned answer.
- A **hard check** decides the answers it governs whenever it is in the question's flow and evaluates to `False`:
  - if `then` names the question, the answer is set to that value with `status="forced"` and confidence 1.0;
  - if the question lists the check in its `checkpoints` (or the check has no `then` at all) but `then` does not name it,
    the question abstains;
  - otherwise — the check has a `then` for other questions and only reads this question's facts — it is an ordinary failed
    check for this question and is listed in the reason.

  No rule and no model confidence can override a failed hard check. To make sure a hard check is always in a question's
  flow, list it in the question's `checkpoints`. When several hard checks fail, the first one declared in the catalog decides:
  declare the most important first. A question's flow and answer do not depend on which other questions are asked in the
  same request.

### Extractors and Quote

An extractor reads text from `init_state` and returns a `Quote`:

```python
import re

@cat.extract
def amount(doc):
    "the total amount to pay"
    m = re.search(r"Total due:\s*([0-9,]+\.[0-9]{2})", doc)
    if not m:
        raise ValueError("no total")        # a failed part becomes a missing fact; dependent answers abstain
    return Quote(float(m.group(1).replace(",", "")), m.start(1), m.end(1))
```

- `doc[start:end]` is the supporting quote. `source` names the `init_state` key holding the text (default `"doc"`).
- A quote whose offsets fall outside the source text is recorded as an error.
- `confidence` (default 1.0) is propagated to every answer that depends on this value (see
  [Confidence](#confidence-calibration-and-abstention)).
- Downstream parts receive the plain `value`, not the `Quote`.

You can write extractors by hand (regular expressions, parsers) or get them from a trained model
([Extracting fields from documents](#extracting-fields-from-documents)).

## Questions and answer types

```python
from solvi import Answer, Question

Question("ship", "Ship now?", Answer.yes_no(), checkpoints=["paid"])
Question("risk", "Risk level", Answer.choice(["low", "medium", "high"]))
Question("route", "Which team?", Answer.choice(["a", "b"]), uses=["country", "total"])
```

`Question(name, text, answer, checkpoints=[], uses=None)`:

- `name`: the key used for rules, `fit`, and `res[name]`;
- `text`: human-readable wording;
- `answer`: `Answer.yes_no()` (options `["yes", "no"]`) or `Answer.choice(options)`;
- `checkpoints`: parts that must be in this question's flow in every request (a missing name raises
  `solvi.strategist.PlanError`);
- `uses`: a hint for the strategist, the facts that matter when the question has neither a rule nor a trained head.

An answer outside the options is never returned: a rule that produces one makes the question abstain.

## Asking: System and Response

```python
from solvi import System

system = System(cat, questions, journal=None)
res = system.ask(init_state)                   # all questions
res = system.ask(init_state, ["ship"])         # a subset
```

`System(catalog, questions, journal=None)`. If `journal` is a file path, every `ask` appends one JSON line with the hash
of `init_state`, the answers, the flow and the hash of every trace record.

### Response

| Field | Content |
|---|---|
| `res["q"]` | the `Result` for question `q` |
| `res.results` | dict of all results |
| `res.flow` | the planned flow; `print(res.flow)` lists the steps and why each was taken |
| `res.flow.per_question` | question -> names of the parts in its flow |
| `res.flow.skipped` | catalog part -> why it was not executed (inputs unavailable, or not needed) |
| `res.flow.unresolved` | question -> facts that nothing can compute from this `init_state` |
| `res.computed_state` | a printable listing: each computed fact, its value, quote offsets, extraction confidence, errors |
| `res.values` | dict of all fact values (inputs and computed) |
| `res.trace` | the hash-chained trace (see [The trace](#the-trace-and-verification)) |
| `res.ms` | decision time in milliseconds |

### Result

| Field | Content |
|---|---|
| `.answer` | one of the options, or `None` when abstaining |
| `.confidence` | float in [0, 1] |
| `.why` | the reason: rule inputs and their values, or the top feature contributions of a learned head, or why it abstained or was forced |
| `.status` | `"ok"`, `"forced"` (a hard check decided) or `"abstain"` |
| `.probs` | class probabilities for answers from a learned head (empty for rules) |

A question abstains when:

- a fact it needs cannot be computed from `init_state` (nothing in the catalog produces it);
- a part it depends on raised an exception (the error is kept in the trace);
- its rule returned a value outside the options;
- a hard check failed and has no `then` entry for it;
- it has no rule and no trained head.

## How the strategist plans a flow

For each asked question the strategist picks targets:

1. **there is a rule**: the rule's arguments;
2. **a head was trained with `fit`**: the features the head selected;
3. **`uses` is set**: those facts;
4. **otherwise**: everything computable from `init_state` (the flow is marked as not narrowed).

It then walks backwards from the targets through the catalog signatures to the keys of `init_state`, and always adds:

- the question's checkpoints;
- every check whose inputs are already available in the flow and which touches at least one computed (not input) fact,
  a "check of what was computed".

Each part runs once per request, even if several questions need it. Steps are executed in topological order, rules
last; hard checks and the facts they read are ordered first. Catalog parts that are not needed, or cannot run because their inputs are missing, are not executed, and
`res.flow.skipped` says which case applies. A cycle in the catalog raises `PlanError`.

The flow depends only on which keys `init_state` has, the catalog, the questions and the trained heads. It is the same for
every request with the same keys.

### Early exit and parallel execution

At run time the executor first computes the hard checks and what they depend on. If a hard check fails, every question whose
flow contains it is settled (forced by `then`, or abstained), and the steps that only those questions needed are not run.
They are listed in `res.trace.skipped` with the check that made them unnecessary. Pass `early_exit=False` to
`solvi.runtime.execute` to compute the whole flow anyway (`System.facts_for` does this for training).

`System(catalog, questions, workers=8)` (or `system.ask(state, workers=8)`) runs independent steps in parallel threads: a step
starts as soon as the steps it reads have finished. This pays off when parts wait on I/O — HTTP APIs, databases, model
inference that releases the GIL. Answers, records and hashes are identical to a sequential run, because records are written in
flow order after execution.

```python
system = System(cat, QUESTIONS, workers=8)
res = system.ask(claim)
print(res.trace.skipped)      # e.g. [("fraud_score", "not needed: hard check policy_in_force failed"), ...]
```

## Questions without a rule: fit, learn_rule, teach

Some answers are hard to write as a rule (a risk level, a region from a messy address). solvi offers two ways to learn
them from labeled examples. Both use the facts that your catalog computes, not raw text.

### fit: a learned answer head

```python
history = [(init_state_1, "low"), (init_state_2, "high"), ...]   # 100-300 labeled examples
head = system.fit("risk", history)
print(head.features, head.cv_acc)          # selected facts and their cross-validated accuracy
```

- Features are all facts computable from the examples' `init_state` keys. Numbers are encoded as a value plus thresholds
  at training quantiles, booleans as +/-1, and strings with at most 20 distinct values as categories.
- The model is a multinomial logistic regression with L2 regularization. Training takes milliseconds to a few seconds.
- Features are selected greedily by 5-fold cross-validated accuracy (a feature is kept if it adds at least 1 point).
  **The selected facts become the question's flow**, so later requests compute only what the head uses.
- The answer's `why` lists the largest feature contributions, `probs` gives all class probabilities.

### fit_fast: learn in milliseconds, correct instantly

`system.fit_fast(question, examples, features=None)` fits a closed-form ridge head: one matrix decomposition, so it takes
milliseconds instead of seconds, and the ridge strength is chosen by exact leave-one-out accuracy (`head.loo_acc`). Features are
the given facts (numbers, booleans, categories, and numeric vectors such as a document embedding from
`LongSpanExtractor.embedder()`), by default every computed fact; when there are few of them, their pairwise products are added
so middle classes and interactions can be expressed.

Its main property is online learning: `system.teach(question, init_state, correct)` updates a fast head immediately with a
rank-one Sherman–Morrison step (about 0.1–0.2 ms) and returns the time in ms. Nothing is retrained, other questions, rules and
hard checks do not change, and the example still goes to the journal.

```python
head = system.fit_fast("suspicious", history[:10])     # start small
for state, label in reviewer_corrections:
    system.teach("suspicious", state, label)            # each one is absorbed at once
```

On the example tasks (`benchmarks/fast_head.py`) `fit_fast` trains 15–70× faster than `fit` and is 1–4 points less accurate at
200 examples; learning online from 10 to 200 corrections ends within a few points of fitting on all 200 at once. Use `fit` when accuracy on a fixed
dataset matters most, `fit_fast` when labels arrive one by one or you need to retrain on every request.

### learn_rule: a readable rule list

```python
rules = system.learn_rule("zone", examples, facts=["address_upper"], min_support=3, min_precision=0.8, max_rules=40)
print(rules)
```

This learns an ordered decision list ("if feature then answer", with a default at the end) and installs it in the
catalog as the question's rule. From then on it behaves like a hand-written rule: deterministic, explained by its inputs,
and re-checkable by `replay`.

- `facts`: the computed facts to build literals from. Booleans give `fact is True/False`, numbers `fact ≈ rounded value`,
  strings give one literal per upper-cased word or number, plus `has number starting 'NN'` for numbers of 5 or more
  digits (postcodes, codes). Other values are compared for equality.
- Each step adds the literal with the best smoothed precision on still-uncovered examples, with at least `min_support`
  examples, while precision stays at or above `min_precision`; at most `max_rules` rules.
- `system.learned_rules[question]` keeps the `RuleList`; `print` shows each rule with its support.

A learned rule replaces any rule previously registered for that question, and takes precedence over a trained head.
Because the list is readable, you can spot rules that only memorized a few examples, or that reproduce a labeling error,
and fix the labels or the catalog.

### teach: record corrections

```python
system = System(cat, questions, journal="decisions.jsonl")
system.teach("risk", init_state, "high")
```

`teach` appends the correction to the journal (it does nothing without a journal). It does not retrain by itself: read
the `{"teach": ..., "init": ..., "answer": ...}` lines back and include them in the next `fit` or `learn_rule` call.
Note that non-JSON values in `init_state` (dates, custom objects) are stored as their `repr`.

## Confidence, calibration and abstention

How confidence is computed:

- **Rule answers**: the minimum extraction confidence among all extracted values the rule's inputs depend on. A rule
  over dict inputs and computations only has confidence 1.0.
- **Learned heads**: the head's probability of the chosen answer, multiplied by the same extraction confidence.
- **Forced answers** (failed hard checks): 1.0.

Calibrate a question on held-out examples (Platt scaling on the confidence logit):

```python
system.calibrate("total_band", heldout_states, heldout_answers)   # lists of init_state and correct answers
```

After calibration, `ok` answers of that question carry the calibrated confidence. With fewer than 10 usable examples, or
when all held-out answers are right (or all wrong), only a constant shift is fitted.

solvi does not pick an abstention threshold for you. With calibrated confidence you choose one per question from
held-out data, for example "answer automatically at confidence >= 0.95, send the rest to a person":

```python
r = system.ask(state)["total_band"]
if r.status == "abstain" or r.confidence < 0.95:
    send_to_review(state, r.why)
```

## The trace and verification

Every step of the flow is a `Record` in `res.trace.records`:

| Field | Content |
|---|---|
| `step`, `kind`, `name` | step number, part kind (`fn`, `check`, `extract`, `rule`), fact name (`answer:<question>` for rules) |
| `inputs` | input name -> hash of the input value |
| `value` | the computed value |
| `quote` | `(start, end, source)` for extracted values |
| `confidence`, `error` | extraction confidence; error text if the part failed or had missing inputs |
| `prev`, `hash` | hash of the previous record (or of `init_state` for the first) and of this record |

`res.trace.init` keeps `init_state` and `res.trace.init_hash` its hash. `res.trace.value(name)` returns a recorded value.

`res.trace.replay(catalog)` independently re-executes every step from the recorded `init_state`, and checks:

- that each record still hashes to its stored hash and links to the previous one;
- that each step's inputs match the recorded input hashes;
- that recomputing the part gives the recorded value;
- that quotes lie within the source text.

It returns `{"ok": bool, "steps": int, "mismatches": [(step, name, reason), ...]}`.

Continuing the README quickstart:

```python
rep = res.trace.replay(cat)
assert rep["ok"]

res.trace.records[0].value = 6               # tamper with a recorded value
print(res.trace.replay(cat)["mismatches"][0][:2])   # (1, 'days_requested'): the altered step is named
```

Replay also catches consistent tampering, where the value is changed and all hashes of the chain are recomputed: the
recomputation from `init_state` no longer matches. In our tests, 500 of 500 naive and 500 of 500 consistent substitutions
were caught, with the exact step identified every time, and there were no false alarms on 1000 untouched traces.

Replay needs the same catalog code. Parts that call external systems (databases, APIs) must return the same values on
replay, or their steps will be reported as mismatches.

## Printing results: solvi.show

```python
from solvi.show import show

show(res, cat)                              # answers, flow, computed_state, replay result, time
show(res, cat, flow=False, state=False)     # answers, replay, time only
show(res)                                   # without the catalog: no replay
```

## Extracting fields from documents

With `pip install "solvi[model]"`, solvi provides three ModernBERT extractors. All of them predict a start and an end
position in the text, so the extracted value is always a substring of the document with exact offsets. Each gives you
plain functions `doc -> Quote` to register with `cat.extract`.

Labels are character spans: for each training document and field, `(start, end)` of the value in the text, or `None` if
the field is absent. Around 100 labeled documents per task was enough in our benchmarks (see
[benchmarks](benchmarks.md)). A GPU is recommended for training and for fast inference.

### MultiSpanExtractor: all fields in one pass

`solvi.extract_multi.MultiSpanExtractor` reads a document once and has a start/end head pair per field. Use it for
documents that fit into one window (receipts, invoices, forms).

```python
from solvi.extract_multi import MultiSpanExtractor

ex = MultiSpanExtractor(["company", "date", "total"],
                        model_name="answerdotai/ModernBERT-large", max_len=1024)
ex.fit(train_docs, train_spans, epochs=4, lr=3e-5, bs=8)
# train_docs: [text]; train_spans: [{"company": (s, e), "date": (s, e), "total": (s, e) or None}]

ex.fit_temperature(calib_docs, lambda field, i, span: span == calib_spans[i][field])   # optional, per-field temperature

for name in ["company", "date", "total"]:
    cat.extract(ex.field(name))            # registers the fact "company", ... read from init_state["doc"]

@cat.fn
def total_value(total):
    return float(total.replace(",", ""))
```

- `ex.field(name)` returns a function named `name` with one argument `doc`, which returns
  `Quote(doc[s:e], s, e, confidence=c)`.
- `ex.predict_doc(text)` returns `{field: (start, end, confidence)}`. Results are cached per text, so all fields of one
  document cost a single forward pass.
- `fit_temperature(docs, gold_ok)` picks a softmax temperature per field that minimizes log loss of "confidence vs.
  correct" on held-out documents; `gold_ok(field, doc_index, (start, end))` tells whether a prediction is correct.
- The extractor always returns a span; this one does not model "field absent". Use `LongSpanExtractor` when fields may
  be missing.

### LongSpanExtractor: long documents, fields by description, "no answer"

`solvi.extract_long.LongSpanExtractor` is for long documents (contracts) and optional fields. The input is
`[CLS] field description [SEP] window of the document [SEP]`, windows overlap, and position 0 means "no answer in this
window".

```python
from solvi.extract_long import LongSpanExtractor

GOV_LAW = "the clause that says which state's or country's law governs the contract"

lx = LongSpanExtractor(max_len=1024, stride=128, max_span=96)
lx.fit([(text, GOV_LAW, span_or_none) for text, span_or_none in train], epochs=3, neg_per_item=3)
lx.tune_threshold("governing_law", [(text, GOV_LAW, span_or_none) for text, span_or_none in heldout])

cat.extract(lx.field("governing_law", GOV_LAW))

@cat.fn
def governing_state(governing_law):
    for s in ["Delaware", "New York", "California"]:
        if s.lower() in governing_law.lower():
            return s
    return "other"
```

- `stride` is the overlap between windows in tokens; `max_span` caps answer length in tokens.
- Training uses every window that contains the answer plus up to `neg_per_item` windows without it. One extractor can
  learn several fields: pass items with different descriptions.
- `predict(text, desc)` returns `(start, end, span_score, no_answer_score)`, the best span over all windows.
- `tune_threshold(name, items)` picks the score threshold for "the field is present" that maximizes present/absent
  accuracy on held-out items.
- `field(name, desc)` returns a function `doc -> Quote`. **When the score is below the threshold, it returns
  `Quote("", 0, 0)`**, an empty value, so downstream functions should treat `""` as "not found" (e.g.
  `has_tax = tax != ""`).
- Cost grows with length: one pass per field per window. On CUAD contracts (median 33k characters) five fields took about
  1 s per contract on an A100.
- Fields are specified by description, but a field that was never labeled in training is **not** extracted reliably from
  its description alone (14% and 66% on two held-out fields in our tests). Label examples for every field you need. A
  universal extractor that handles new fields is in progress.

### SpanExtractor: one field per pass

`solvi.extract_model.SpanExtractor` is the simplest variant: description plus text in, one span out, one forward pass
per field. `fit([(text, description, (s, e) or None), ...])` trains it (examples without a span are skipped), and
`predict([(text, description), ...])` returns `[(start, end, confidence)]`. It has no `field()` helper; wrap it yourself:

```python
from solvi.extract_model import SpanExtractor

sx = SpanExtractor()
sx.fit(train_items, epochs=4)

@cat.extract
def total(doc):
    "the total amount paid"
    s, e, c = sx.predict([(doc, total.__doc__)])[0]
    return Quote(doc[s:e], s, e, confidence=c)
```

It is about 3.6x slower than `MultiSpanExtractor` for four fields with similar accuracy; prefer the one-pass extractor.

### Hardware notes

- Measured on an A100: about 39 ms per receipt for four fields with `MultiSpanExtractor`.
- On CPU, fp32 ONNX keeps accuracy but took about 0.7 s per receipt on 2 cores. Dynamic int8 quantization of
  ModernBERT-large lost up to 12 points on amounts, company names and addresses, so we do not recommend it yet.

## Guarantees and limitations

What solvi guarantees:

- Every value in `computed_state` was produced by your code; every extracted value carries its quote and offsets.
- A failed hard check always decides the answer, above any model confidence.
- When a needed fact cannot be computed, a part fails, or a rule returns an invalid option, the question abstains
  instead of guessing.
- `replay` recomputes the trace and names the step where anything was changed, including changes with recomputed hashes.

What it does not guarantee:

- The quality of learned answers depends on your examples, and the quality of extraction depends on the model and the
  labels. Learned rules reproduce labeling errors (which is also what makes those errors visible).
- The strategist plans from signatures. For a question with no rule, no trained head and no `uses`, it computes
  everything reachable.
- solvi answers yes/no and choice questions. It does not generate free text.
- New fields need labeled examples, roughly 100 documents per task.

Research note: in our experiments, an LLM could write a working catalog from a plain-language task description when every
draft was executed against examples with known answers and errors were fed back (see [benchmarks](benchmarks.md#writing-catalogs-with-an-llm)).
This is not part of the library.
