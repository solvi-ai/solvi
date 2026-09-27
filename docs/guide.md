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
10. [Grounded decisions: provenance, audit and safeguards](#grounded-decisions-provenance-audit-and-safeguards)
11. [Decisions with a model](#decisions-with-a-model)
12. [Printing results: solvi.show](#printing-results-solvishow)
13. [Extracting fields from documents](#extracting-fields-from-documents)
14. [Guarantees and limitations](#guarantees-and-limitations)

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
pip install "solvi[model]"     # + torch, transformers, for the ModernBERT extractors (solvi.extract_*) and the decider
pip install "solvi[onnx]"      # + onnxruntime, tokenizers: the decider (solvi.decide) on CPU without torch
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
- A quote whose offsets fall outside the source text is rejected: the fact is missing (dependent answers abstain) and the
  error is recorded. A quote from a model-backed part must also be literally the text at its offsets — see
  [Grounded decisions](#grounded-decisions-provenance-audit-and-safeguards). Hand-written code may return a value derived
  from the quoted text (a label, a parsed number, a date); `@cat.extract(exact=True)` makes it literal too.
- `confidence` (default 1.0) is propagated to every answer that depends on this value (see
  [Confidence](#confidence-calibration-and-abstention)).
- Downstream parts receive the plain `value`, not the `Quote`.

You can write extractors by hand (regular expressions, parsers) or get them from a trained model
([Extracting fields from documents](#extracting-fields-from-documents)).

### Several producers of one fact (fallback chain)

A fact can have alternative producers — a cheap regular expression and a slower model, say. Give each its own function name
and say which fact it `provides`:

```python
@cat.fn(provides="total", cost=0.1, validate=lambda v: v > 0)
def total_regex(doc):
    m = re.search(r"TOTAL\s*:?\s*(\d+\.\d\d)", doc)
    return None if m is None else float(m.group(1))           # None = "not found": rejected, the next producer runs

@cat.extract(provides="total", cost=50, min_confidence=0.8)
def total_model(doc):
    return extractor.field("total", "the total amount paid")(doc)

@cat.features("total")                                      # optional: cheap features for the producer policy
def total_features(doc):
    return {"lines": doc.count("\n"), "has_total": "TOTAL" in doc.upper()}
```

- The producers form a group in the catalog under the fact's name; the strategist plans it like one part (its inputs are
  the union of the producers' inputs). Catalogs without `provides` behave exactly as before.
- A producer's output is **accepted** only if it is not `None`, a `Quote` lies inside its text (literally, for a model-backed
  producer), a `Decision` is among its options, it reaches `min_confidence`,
  and `validate(value, [any of the producer's inputs by name])` returns true. An exception counts as a rejection. The
  first accepted output is used; if none is accepted the fact is missing and dependent answers abstain.
- Order: declaration order by default. `System(cat, questions, producers="learned")` lets a policy choose the order per
  input: producers predicted to agree with the reference producer (the last declared, normally the most trusted) at least
  `system.producer_policy.quality` of the time (0.9) go first, cheapest expected cost first (cost ÷ P(accepted)). The policy
  learns from every run: acceptance of each producer that ran, and — when two producers ran on the same input — whether the
  cheaper one agreed with the reference. With probability `producer_policy.explore` (0.1) the remaining producers also run
  as a *shadow* (never used, only compared) so the policy keeps getting agreement labels. The policy only orders; acceptance is
  always the producer's own deterministic check.
- The record of the fact says which producer was used (`record.producer`) and every producer that ran with its outcome
  (`record.tried`, e.g. `[["total_regex", "no value"], ["total_model", "accepted"]]`). Both are hashed into the chain.
  `replay` recomputes the value with the producer that was used, checks it still passes its validator, and checks that the
  producers tried before it are still rejected (shadow runs are not re-checked).
- `cost=` (ms) is a prior; measured run times replace it (`system.costs`).

## Questions and answer types

```python
from solvi import Answer, Question

Question("ship", "Ship now?", Answer.yes_no(), checkpoints=["paid"])
Question("risk", "Risk level", Answer.choice(["low", "medium", "high"]))
Question("route", "Which team?", Answer.choice(["a", "b"]), uses=["country", "total"])
```

`Question(name, text, answer, checkpoints=[], uses=None, min_confidence=None)`:

- `name`: the key used for rules, `fit`, and `res[name]`;
- `text`: human-readable wording;
- `answer`: `Answer.yes_no()` (options `["yes", "no"]`) or `Answer.choice(options)`;
- `checkpoints`: parts that must be in this question's flow in every request (a missing name raises
  `solvi.strategist.PlanError`);
- `uses`: a hint for the strategist, the facts that matter when the question has neither a rule nor a trained head;
- `min_confidence`: an answer below this confidence abstains (status `abstain`, the reason says what it would have answered).

An answer outside the options is never returned: a rule that produces one makes the question abstain.


### Ordinal and multi-label answers, option descriptions

- `Answer.ordinal(["low", "medium", "high"])` — ordered levels, lowest first. A learned head answers with the median of its
  distribution rather than the most likely level, so a split between "low" and "high" gives "medium", not a jump.
  `answer_type.rank(v)` gives the position.
- `Answer.multi(["pii", "abuse", "prompt_injection"])` — any subset, returned as a tuple in option order (empty tuple for none).
  A rule may return a list or set. `fit` / `fit_fast` learn one yes/no head per option; `teach` updates all of them.
- Any option list may be a dict `{option: description}`; descriptions are kept in `answer_type.descriptions`.

### Constraints between answers

```python
@cat.constraint
def unsafe_if_harm(verdict, harm):          # argument names are question names
    return harm == "none" or verdict == "unsafe"
```

After all questions are answered, solvi checks every constraint whose questions were asked. If learned answers break one, it
searches for the most probable combination of learned answers (from their distributions; multi-label answers per option) that
satisfies every constraint, and appends "changed from … to satisfy …" to the reason. Answers from rules, hard checks and
abstentions never change. `res.feasible` says whether the final answers satisfy every constraint; if fixed answers conflict,
it is `False` and `res.violations` names the constraints.

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
| `res.audit(q=None)` | what each answer rests on and which safeguards fired (see [Grounded decisions](#grounded-decisions-provenance-audit-and-safeguards)) |
| `res.safeguards` | the safeguard events of this response |
| `res.ms` | decision time in milliseconds |

### Result

| Field | Content |
|---|---|
| `.answer` | one of the options, or `None` when abstaining |
| `.confidence` | float in [0, 1] |
| `.why` | the reason: rule inputs and their values, or the top feature contributions of a learned head, or why it abstained or was forced |
| `.status` | `"ok"`, `"forced"` (a hard check decided) or `"abstain"` |
| `.probs` | class probabilities for answers from a learned head or a model decision (empty for plain rules) |
| `.provenance`, `.source` | where the answer came from: `computed` (a rule, a hard check), `learned` (a head, a learned rule), `decided` (a model-backed rule); and which rule / check / head |
| `.guard`, `.repaired` | the safeguard that settled it (`hard_check`, `outside_options`, `low_confidence`), and `(previous answer, constraints)` if joint decoding changed it |

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

### Learned order of hard checks

Every `ask` measures the run time of each part: `system.costs` keeps a moving average (ms) per part (`cost=` on a decorator
is the prior until a part has run). It also records which hard checks failed on which input.

`System(cat, questions, order="learned")` — or `system.learn_order(examples)` on a list of `init_state`s, which runs only the
hard checks and what they read and then switches the order — makes the executor evaluate hard checks **one at a time**, the
one with the highest expected saving first:

    score = P(check fails | cheap facts) × cost of the steps its failure would skip ÷ cost of evaluating it

and stop as soon as the failed checks settle every question they govern. P(fail) comes from a small online model per hard
check (`system.order_model`: Laplace counts, then a FastHead refitted every 50 rows and updated in between) on the scalar
values of `init_state` (and one level of dicts, e.g. `invoice.currency`). `learn_order(features=[...])` adds cheap computed
facts; they are computed before the hard checks.

Answers are identical to the default order. When several hard checks fail, the first one declared in the catalog decides;
so a question is settled by a failed check only after every hard check that governs it and is declared earlier has been
evaluated. Those earlier checks are scheduled next, since evaluating them settles the question whatever they return. Records
stay in flow order. What changes is only which steps run: `res.trace.skipped` lists the rest, and
`res.trace.explain_order()` (also printed by `show`) says why each hard check ran when it did:

```
1. policy_in_force: P(fail) 0.09 × saves 222.6 ms ÷ costs 0.0 ms = 712 → passed
2. fraud_ok: P(fail) 0.81 × saves 64.0 ms ÷ costs 170.2 ms = 0.305 → failed
3. no_litigation: ... [unblocks decision (already decided by a failed check)] → passed; settles decision, fast_track
```

`ask(state, order=...)` overrides the order for one request: `"default"`, `"learned"`, or any object with
`p_fail(check, row)` and `row(vals, init_keys)` (e.g. an oracle for experiments).

The learned order saves time only when hard checks fail often enough and their failure skips expensive work; when nothing
fails every hard check still runs. With `workers > 1` the default order runs all hard checks at once, while the learned order
runs them one after another (their inputs still run in parallel) — measure before choosing it for parallel execution.

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

- **Rule answers**: the minimum confidence among all extracted values and model decisions the rule's inputs depend on
  (and the rule's own, for a model-backed rule). A rule over dict inputs and computations only has confidence 1.0.
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
| `producer`, `tried` | for a fact with several producers: the one used, and every producer that ran with its outcome |
| `provenance`, `model`, `probs` | the value's provenance kind (`record.origin` gives the default when not stored), the model that produced it (`{"type", "id", "fp"}`), a decision's probabilities |

Answers from a learned head are records too (`kind="head"`, after the flow's steps): the answer, its probabilities and the
head's fingerprint.

`res.trace.init` keeps `init_state` and `res.trace.init_hash` its hash. `res.trace.value(name)` returns a recorded value.

`res.trace.replay(catalog)` independently re-executes every step from the recorded `init_state`, and checks:

- that each record still hashes to its stored hash and links to the previous one;
- that each step's inputs match the recorded input hashes;
- that recomputing the part gives the recorded value (and the same quote offsets);
- that quotes lie within the source text (and, for model-backed parts, are literally the quoted text);
- for model-backed steps, that the recorded model fingerprint matches the catalog's current model (see below).

It returns `{"ok": bool, "steps": int, "mismatches": [(step, name, reason), ...], "models": [(step, name, verdict), ...]}`.

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

## Grounded decisions: provenance, audit and safeguards

The principle: **fuzzy proposes, deterministic decides, everything is in the trace.** A model may extract a value, pick a
category or learn an answer, but its output is checked by deterministic code before anything uses it, and every step records
where its value came from. A model's hallucination is either caught or visible in the audit — never silently an answer.
A decision without any model and one with models are the same system; they differ only in the provenance of the facts.

### Provenance

Every fact and answer has a provenance kind (`record.origin`, `result.provenance`, `solvi.provenance.KINDS`):

| Kind | Where the value comes from | How it is kept honest |
|---|---|---|
| `given` | a key of `init_state` | hashed into `init_hash`; the chain starts from it |
| `computed` | a plain function: `fn`, `check`, a hand-written rule | replay re-runs it and compares |
| `quoted` | an `extract` part returning a `Quote` | the offsets must lie in the source text; for a model, `doc[start:end]` must be the value |
| `decided` | a model's choice among declared options, with probabilities (`Decision`) | the value must be one of the options; probabilities recorded |
| `learned` | a `fit` / `fit_fast` head, a `learn_rule` list, another trained function | the head type and a fingerprint of its parameters are recorded |
| `proposed` | reserved for a strategist model | the deterministic layer verifies what it proposes |

The default comes from what a part returns (a `Quote` → `quoted`, a `Decision` → `decided`) and whether a model is behind
it. Declare it explicitly with `provenance=` on any decorator. A part is model-backed when you pass `model=`:

```python
cat.extract(extractor.field("total", "the total amount paid"))    # field() functions bring their model along

@cat.extract(model=span_extractor)                                  # any function that calls a model
def vendor(doc): ...

@cat.fn(model=classifier, options=["travel", "meals", "equipment"])
def category(doc):
    p = classifier.predict(doc)                                    # {option: probability}
    return Decision(max(p, key=p.get), p)                           # downstream parts get the plain value

@cat.rule("risk", model=risk_model)                                 # a model answers the question directly
def risk(amount, country): ...
```

`LongSpanExtractor.field`, `MultiSpanExtractor.field` and `LongSpanExtractor.embedder` mark their functions (the attributes
`__solvi_model__` and `__solvi_provenance__`), so registering them is enough. For `SpanExtractor`, or any other model,
pass `model=`.

### Model identity in the trace

A model-backed record stores `record.model = {"type", "id", "fp"}`: the class, the model id (the Hugging Face id or path it
was loaded from, `model.model_id`), and a fingerprint (`solvi.provenance.fingerprint`):

- extractors: settings, thresholds / temperatures, the span head, evenly sampled encoder weights, and the names and sizes of
  the weight files — about 10 ms once, then cached until `fit` / `save`;
- `Head`, `FastHead` (fit, fit_fast): a hash of their parameters — it changes with every `teach`;
- `RuleList` (learn_rule): a hash of its rules;
- any other object: its own `fingerprint()` method, or a `version` attribute, or `"unversioned:<type>"` (then a changed model
  cannot be detected — give your models a version).

`res.trace.replay(catalog)` handles model-backed steps as follows:

- the fingerprint differs from the catalog's current model → a mismatch, *"model changed since this decision"*; the recorded
  output is still checked for grounding;
- same model, deterministic (the default; set `model.deterministic = False` otherwise) → the step is re-run and compared;
- `replay(catalog, trust_models=True)`, or the model is not available (no model on the part, `model.available = False`) →
  the model is not re-run; the recorded output is verified instead: the quote is literally at its offsets in the recorded
  input, the decision is among the options.

Pass the `System` instead of the catalog (`res.trace.replay(system)`) to verify answer-head records too: their fingerprint,
and (unless trusted) their probabilities recomputed from the recorded facts. `rep["models"]` lists every model-backed step
with its verdict: `recomputed`, `trusted`, `unavailable` or `changed`.

Hashes: a record hashes its provenance only when it differs from the default (`quoted` for a record with a quote, else
`computed`), and its model and probabilities only when present — so traces of catalogs without models hash exactly as before.

### Safeguards

| Safeguard | Fires when | Effect |
|---|---|---|
| grounding | a quote lies outside its text, or a model's quote is not literally `doc[start:end]` (strings up to whitespace, numbers as written, e.g. `1250.0` ↔ `"1,250.00"`) | the output is rejected: the fact is missing, the claim stays in the error; the next alternative producer runs, else dependent answers abstain |
| closed set | a `Decision` (or a value of a part with `options=`) is not one of the options; a rule's answer is not one of the question's options | rejected / the question abstains |
| low confidence | a `Quote` / `Decision` is below the part's `min_confidence`; an answer is below the question's `min_confidence` | rejected / the question abstains, saying what it would have answered |
| validate | a producer's `validate(value, ...)` returns false | rejected, next producer |
| hard check | a hard check governing the question is false | the answer is forced by `then`, or the question abstains |
| constraint repair | learned or model answers break a constraint between answers | the most probable consistent combination is chosen |
| fallback | an alternative producer was rejected and a later one was used | recorded in `tried` |

Hand-written extractors (no model) may return a value derived from the quoted text; the audit then shows the value next to
the text it was derived from. Numbers, dates and other non-string values from a model are checked when they can be compared
(numbers) and shown next to the quoted text otherwise.

### The audit

```python
print(res.audit())            # every answer
a = res.audit("approve")      # one answer: an AnswerAudit
a.to_dict()                   # the same as data
```

For each answer: the given inputs → computed facts → quotes (offsets, the quoted text, whether the value is literally that
text, the model) → model decisions (the model, probabilities) → learned parts (head type and fingerprint) → checks (hard or
soft, which one decided) → the rule → constraints → the answer; the parts skipped at run time; the safeguards that fired;
and a summary of the support: how many items are deterministic (given, computed, quoted by plain code) and how many come
from models (`a.share_deterministic`, `a.counts`).

```
approve = 'yes'  [ok]  confidence 0.60  ← computed by approve
  given       doc = 'Expense claim #2291\nVendor: C…; limit = {'travel': 100, 'meals': 60, 'e…
  computed    amount = 48.6
  quoted      total = '48.60'  doc[100:105] literal '48.60'
  decided     category = 'travel'  (travel 0.60, meals 0.20, equipment 0.20)  [StandInClassifier demo/expense-category #bb3352d4]
  check       amount_positive = True (hard)
  rule        approve (computed)
  → answer    'yes' — amount = 48.6; category = 'travel'; limit = {...}
  support     7 items (2 given, 3 computed, 1 quoted, 1 decided): 86% deterministic, 1 from models
  safeguards  grounding rejected ×1, fallback producer ×1
              · grounding rejected: total — total_model: not grounded: '488.60' is not the text at [100:105] ('48.60')
              · fallback producer: total — total_regex used after total_model rejected
```

### Lifetime stats

`system.stats` counts, over the system's lifetime: `asks`, `answers`, `abstained`, `model_outputs` (outputs of model-backed
parts, answer heads and learned rules), `grounding_rejected`, `outside_options`, `low_confidence`, `validator_rejected`,
`forced_by_hard_check`, `constraint_repairs` and `fallbacks`. `system.safeguard_report()` prints them. Counting costs about
1% of a decision. [examples/12_grounded_audit.py](../examples/12_grounded_audit.py) runs one catalog with and without models,
with a hallucinating extractor and a classifier answering outside its options.

## Decisions with a model

A **decider** picks among options described in words: "which team handles this email?", "which clauses apply?". solvi's
decider (solvi-decide, a ModernBERT cross-encoder) reads `[mode] task [opt] option 1 [opt] option 2 … [SEP] text` and scores
every option in one pass: a softmax over the options for a single choice, a sigmoid per option for multi-label. In solvi it is
a catalog part like any other, so everything in [Grounded decisions](#grounded-decisions-provenance-audit-and-safeguards)
applies unchanged: the closed set, `min_confidence`, constraints with joint decoding, hard checks, the audit, the stats.

```python
from solvi.decide import DecideModel

model = DecideModel.load("~/models/decide-base")          # a checkpoint folder or a Hugging Face id
team = model.decision("team", "Which team should handle this support email?", text_fact="email",
                      options={"billing": "payments, invoices, refunds", "technical": "bugs, errors, crashes",
                               "shipping": "delivery, tracking, parcels", "other": "none of the above"})

cat.fn(team)                              # a fact other parts read (returns Decision(value, probs); they get the value)
q = team.question(cat, "team", min_confidence=0.6)        # or: the answer of a question (registers cat.rule("team")(team))
```

### Loading

`DecideModel.load(path_or_hf_id, device=None, backend="auto")` reads a folder with `solvi_decide.json`, `config.json`,
`tokenizer.json` and the weights (`model.safetensors` and/or `onnx/model_fp16.onnx`), or downloads a Hugging Face id once.
`backend="onnx"` needs `solvi[onnx]` (onnxruntime + tokenizers, no torch; ~50 ms per decision on a CPU); `backend="torch"`
needs `solvi[model]` (CUDA when available); `"auto"` takes ONNX when the file and onnxruntime are there. Both give the same
probabilities to about three decimals. Any retrained checkpoint in the same folder format loads the same way; its
`solvi_decide.json` may set `temperature` (the default is the value fitted on the training pool, 1.45 for
`l14b_decider v1`), `temperature_multi`, `other_threshold` and `multi_threshold`.

`model.model_id` is the path or id it was loaded from; `model.fingerprint()` hashes the checkpoint files (names, sizes and
sampled bytes), the backend, the default calibration and every adaptation; `model.metadata()` lists them. Any object with
`logits(items)` (one array of logits per `solvi.decide.Item`) can stand in for the network: `DecideModel(scorer, meta)`.

### Scoring

```python
model.score(text, task, options, descriptions=None, multi=False)   # → {option: probability}; a list of texts → a list
model.decide(text, task, options, ...)                             # → Decision(value, probs)
model.logits(text, task, options)                                  # raw logits
```

Texts are batched (sorted by length, 16 per pass) and raw logits are cached per (text, task, options), so a decision used
both as a fact and as an answer, or replayed, runs the network once. Only the text is truncated (to `max_len`, 512 tokens).

### The decision part

`model.decision(name, task, text_fact="doc", options=..., descriptions=None, multi=False, other=None)` returns a callable
catalog function named `name` that reads `text_fact` (a fact name, or a list of them joined by new lines; a `Quote` is read
as its value) and returns `Decision(value, probs)`:

- the value is one of the options **by construction** — the network only scores the options it is given — and the options are
  the part's closed set (`part.options`), so the safeguard would reject anything else;
- the provenance is `decided`; the trace records `{"type": "DecisionPart", "id": model_id, "fp": ...}`, where the fingerprint
  covers the checkpoint and **this decision's** adaptation (teaching one decision does not mark the others as changed);
- `cat.fn(min_confidence=0.7)(part)` rejects unsure decisions (the fact is missing, dependent answers abstain);
  `Question(min_confidence=...)` abstains on unsure answers;
- as a question's answer (`part.question(cat, name, text=None, min_confidence=None, checkpoints=None)` or
  `cat.rule("q")(part)`), the answer keeps the probabilities, so constraints can repair it by joint decoding, and a failed hard
  check still forces the answer without asking the model.

`multi=True`: the value is a tuple of the options at probability ≥ 0.5, in option order; confidence is the least certain
option's `max(p, 1 − p)`.

### Label-bias correction without labels

A decider likes some labels whatever the text. Estimate that preference on unlabelled texts of your domain and subtract it:

```python
team.adapt(unlabelled_emails)            # or model.adapt(texts, task, options)
```

For each option, the mean logit over the texts (centered over the options) is subtracted before the softmax / sigmoid.
No labels are needed; in research (L14b) this gave +7 points on unseen domains. The correction is stored per
(task, options, descriptions, mode) in `model.adaptations` and is part of the fingerprint, so a replay of a decision made
before it reports "model changed since this decision".

### Few-shot adaptation ("S") and teach

```python
team.fit(labelled)                       # [(text, correct)], e.g. 16–64 examples
system.teach("team", {"email": text}, "billing")   # one correction, absorbed at once
```

`fit` learns a shift per option and one shared scale on the (bias-corrected) logits by L-BFGS
(`(a·z + b) / temperature`, regularized towards the model's defaults), then fits the temperature on out-of-fold
predictions (4 folds), so confidences are calibrated (ECE 0.055 at 32 examples in research; the accuracy gain over the bias
correction alone is small — S mostly calibrates). `teach(text, correct)` adds one example and refits the shift and scale from
the kept examples, warm-started (K + 1 parameters: ~1 ms, plus the forward pass if the text was not scored before); the
temperature and the threshold stay until the next `fit`. `System.teach(question, ...)` routes to it when the question's
answer is a decision part (or a rule that only passes a decided fact on), and returns the time in ms.

`model.save_adaptations(path)` / `model.load_adaptations(path)` keep adaptations with the checkpoint's fingerprint (loading
onto a different checkpoint is refused unless `strict=False`); `part.reset()` forgets one.

### "Other" as an abstain threshold

If the options include `"other"` or `"none"` (also "none of the above", "none of these"; or name it with `other="misc"`;
`other=False` turns this off), that option is **not scored** by the network. It is chosen when the best real option's
calibrated probability `m` is below a threshold: fitted by `fit` on out-of-fold predictions when the examples include at
least three labelled "other" (and three others), else `other_threshold` from the checkpoint's metadata (0.5). Its confidence
is `1 − m`; in `probs` it gets `π = g / (1 + g)` with `g = thr·(1 − m)/(1 − thr)` and the real options share `1 − π`, so it is
the most probable option exactly when `m < thr` (joint decoding sees consistent probabilities). For multi-label, "none" is
chosen when no option reaches 0.5.

### Calibration utilities

`solvi.calibration` works for any model or question:

```python
from solvi.calibration import coverage_at, ece, evaluate, reliability, threshold_for

coverage_at(conf, correct, 0.9)       # share of cases answerable automatically at ≥ 90% accuracy
threshold_for(conf, correct, 0.9)     # the confidence threshold that gives it (e.g. for Question(min_confidence=...))
ece(conf, correct)                    # expected calibration error; reliability(conf, correct, bins) for the diagram
evaluate(system, "team", examples)    # ask on [(init_state, answer)] → accuracy, ece, coverage_at, answered, ...
```

[examples/13_decide_model.py](../examples/13_decide_model.py) routes support emails with a decision part: bias correction on
60 unlabelled emails, S on 16 labelled ones, abstention through `min_confidence`, a constraint with a rule-based question, a
hard check, the audit and `System.teach`. It runs the real decider when `SOLVI_DECIDE_MODEL` points to a checkpoint and a
keyword stand-in otherwise.

## Printing results: solvi.show

```python
from solvi.show import show

show(res, cat)                              # answers, flow, computed_state, audit summary, replay result, time
show(res, cat, flow=False, state=False)     # answers, audit summary, replay, time
show(res, cat, audit=False)                 # without the audit summary
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
`predict([(text, description), ...])` returns `[(start, end, confidence)]`. It has no `field()` helper; wrap it yourself and
pass `model=` so the trace records the model:

```python
from solvi.extract_model import SpanExtractor

sx = SpanExtractor()
sx.fit(train_items, epochs=4)

@cat.extract(model=sx)
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

- Every value in `computed_state` was produced by your code; every extracted value carries its quote and offsets, and
  its provenance (and the model's identity, for a model-backed part).
- A model's quote that is not literally the text at its offsets, or a model decision outside its options, is rejected
  and counted; it never becomes an answer.
- A failed hard check always decides the answer, above any model confidence.
- When a needed fact cannot be computed, a part fails, or a rule returns an invalid option, the question abstains
  instead of guessing.
- `replay` recomputes the trace and names the step where anything was changed, including changes with recomputed hashes.
- Scheduling never changes answers: the learned order of hard checks gives the same answers as the default order, and a
  learned producer policy only chooses which producer to try first — every output is still accepted by its own check and
  the producer used is recorded and replayed.

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
