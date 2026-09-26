# solvi

Build decision systems from a catalog of Python functions and checks plus typed questions, and get answers you can verify.

## Why

You describe a task with plain Python functions (computations, checks, answer rules) and questions with typed answers
(yes/no, or a choice from a list). For each request, a strategist plans which functions and checks to run for the asked
questions. Every answer comes with:

- a **confidence** (calibratable per question);
- a **reason you can check**: the rule inputs, a formula over computed facts, or a quote with character offsets in the
  source document;
- a **hash-chained trace** of every step, which can be re-executed later to confirm the answer or pinpoint the step that
  was altered.

When something cannot be computed, a function fails, or a rule returns an answer outside the allowed options, solvi
abstains instead of guessing. A failed hard check always overrides any model confidence.

And it is fast. The strategist plans a flow over a 10 000-part catalog in about 6 ms and runs only the parts the questions
need (2.4% of that catalog). Hard checks run first, so a failing one skips the expensive rest; independent slow parts (API
calls, model inference) run in parallel. On an insurance-claim desk with slow services
([examples/09_strategy_at_scale.py](examples/09_strategy_at_scale.py)) a full decision takes 463 ms instead of 1 122 ms for a
script that computes everything, and 152 ms when an expired policy settles the claim first.

## Try it

- [solvi playground](https://huggingface.co/spaces/solvi-ai/playground): write a decision task in Python and run it, watch the
  strategist's plan, tamper with a trace and see the replay catch it, learn rules from examples.
- [solvi documents](https://huggingface.co/spaces/solvi-ai/documents): cited, typed answers from contracts, invoices, receipts,
  leases and more — the ModernBERT extractor (ONNX) and the decisions both run in your browser; add a field by describing it.
- [solvi arcade](https://huggingface.co/spaces/solvi-ai/arcade): game agents that explain every move — tic-tac-toe, maze,
  minesweeper, 20 questions, Mafia detective, a bot arena, and "hack the trace".
- [solvi realms](https://huggingface.co/spaces/solvi-ai/realms): an endless strategy game whose factions are solvi systems —
  tested for 100 000 turns: flat decision time (~0.3–0.6 ms), bounded memory and state, every sampled trace replay OK.
- All run **entirely in your browser** (Pyodide): no server, no GPU, nothing you type leaves the page.
- Models: [solvi-ai/extract-base](https://huggingface.co/solvi-ai/extract-base) (fields by description) and
  [solvi-ai/extract-receipts](https://huggingface.co/solvi-ai/extract-receipts).

## Gallery

[gallery/](gallery) — twelve decision tasks across directions (support triage, email routing, content guard, security alerts,
AI-agent audit, release rollout, KYC/AML, clinical screening, credit with adverse-action reasons, procurement 3-way match,
double-charge refunds, predictive maintenance), each with scenarios, a runner and a side-by-side against an answer-only model.

## Install

```bash
pip install solvi              # core: rules, checks, learned answer heads (numpy, scipy)
pip install "solvi[model]"     # + torch, transformers: ModernBERT field extractors for documents
```

Requires Python 3.10+.

## Quickstart (core only, no model)

```python
from datetime import date
from solvi import Answer, Catalog, Question, System

cat = Catalog()

@cat.fn                                            # argument names = facts it reads; function name = fact it sets
def days_requested(start, end):
    return (end - start).days + 1

@cat.fn
def remaining_after(balance, days_requested):
    return balance - days_requested

@cat.check(hard=True, then={"approve": "reject"})  # if this check is False, "approve" is forced to "reject"
def enough_balance(remaining_after):
    return remaining_after >= 0

@cat.check
def enough_notice(start, today, days_requested):
    return days_requested < 5 or (start - today).days >= 14

@cat.rule("approve")
def approve(enough_notice):
    return "approve" if enough_notice else "needs_manager"

system = System(cat, [Question("approve", "Approve the leave?",
                               Answer.choice(["approve", "needs_manager", "reject"]),
                               checkpoints=["enough_balance"])])
res = system.ask({"start": date(2026, 10, 19), "end": date(2026, 10, 23),
                  "today": date(2026, 9, 25), "balance": 14})
print(res["approve"].answer, res["approve"].confidence, res["approve"].why)
print(res.computed_state)
print(res.trace.replay(cat))
```

Output:

```
approve 1.0 enough_notice = True
days_requested           = 5
remaining_after          = 9
enough_balance           = True
enough_notice            = True
{'ok': True, 'steps': 5, 'mismatches': []}
```

With `"balance": 3` the hard check fails and the answer is `reject` with `status == "forced"`, whatever the rule says.
`solvi.show.show(res, cat)` prints answers, the planned flow, the computed state and the replay result in one go.

## How it works

- **Catalog.** `@cat.fn` (computation), `@cat.check` (bool), `@cat.extract` (value from text, returned as a `Quote` with
  offsets) and `@cat.rule(question)` (answer rule). A part's contract is its signature: argument names are the facts it
  reads, the function name is the fact it sets.
- **Questions.** `Question(name, text, Answer.yes_no() | Answer.choice([...]), checkpoints=[...])`. Questions without a
  rule get a small answer head trained from labeled examples (`system.fit`) or a readable learned rule list
  (`system.learn_rule`).
- **Strategist.** For each question it walks backwards from the rule's arguments (or the learned features) through the
  catalog signatures to the keys of `init_state`, adds the question's checkpoints and every check that touches a computed
  fact. Everything else in the catalog is not executed; the flow records why each part was taken or skipped.
- **Execution.** Each part runs once, even if several questions need it. Hard checks and their inputs run first; when one
  fails, the steps only the settled questions needed are skipped (`res.trace.skipped`). With `System(..., workers=8)`
  independent steps run in parallel threads as soon as their inputs are ready. Results go into `computed_state` with their
  provenance; extracted values keep their quote. Each step record is hashed and chained to the previous one in flow order,
  so the trace does not depend on scheduling.
- **Answers and trace.** `res[q].answer / .confidence / .why / .status` (`ok`, `forced`, `abstain`), plus
  `res.trace.replay(catalog)`, which recomputes every step from recorded inputs and reports mismatches, broken hash links
  and quotes outside the text.

## Extract from documents

With `solvi[model]`, fields are found by a fine-tuned ModernBERT extractor. The extracted value is always a span of the
document, so every answer built on it can be cited.

```python
from solvi.extract_multi import MultiSpanExtractor

ex = MultiSpanExtractor(["total", "date"])          # ModernBERT-large, one pass per document for all fields
ex.fit(train_docs, train_spans, epochs=4)           # train_spans: [{"total": (start, end), "date": (start, end) | None}]
cat.extract(ex.field("total"))                      # doc -> Quote(text, start, end, confidence)
cat.extract(ex.field("date"))

@cat.fn
def amount(total):                                  # extracted values are strings; parse them in ordinary functions
    return float(total.replace(",", ""))
```

For long documents (contracts), `solvi.extract_long.LongSpanExtractor` reads the whole text in overlapping 1024-token
windows, takes a field description instead of a fixed field list, and supports "no answer" with a per-field threshold.
See [docs/guide.md](docs/guide.md#extracting-fields-from-documents).

## Results

Same training documents for both sides. The baseline, Laya, is a ModernBERT-large model that answers the typed questions
directly, fine-tuned with its authors' recipe. Details and caveats: [docs/benchmarks.md](docs/benchmarks.md).

| Task (test set) | solvi | Baseline |
|---|---|---|
| SROIE receipts, 6 questions (361 receipts) | **97.8%** | 92.6% |
| SROIE, only 100 labeled training receipts | **95.9%** | 80.0% |
| CORD receipts, 4 questions (100 receipts) | **98.5%** | 96.0% |
| CORD, share of questions answered at >= 99% precision | **99.7%** | 14.2% |
| CUAD contracts, 5 questions (102 contracts, median 33k chars) | **94.8%** | 77.5% (sees first 1024 tokens only) |

- Calibrated confidence: ECE 0.008 (SROIE), 0.011 (CORD), 0.027 (CUAD).
- On the document benchmarks, 100% of answers are backed by a quote at stated offsets, or the system abstains.
- Speed: about 0.4 ms per decision when no model is involved; about 39 ms per receipt with the one-pass extractor on an
  A100 GPU.

## Speed

Strategist on random layered catalogs ([benchmarks/strategist_scale.py](benchmarks/strategist_scale.py), one CPU core):

| catalog parts | plan | plan + run + trace | parts run | share of catalog |
|---:|---:|---:|---:|---:|
| 100 | 0.2 ms | 1.3 ms | 31 | 31% |
| 1 000 | 0.7 ms | 2.6 ms | 115 | 12% |
| 10 000 | 6 ms | 10 ms | 236 | 2.4% |

Insurance claim desk with six slow services of 100-300 ms ([examples/09_strategy_at_scale.py](examples/09_strategy_at_scale.py)):

| questions asked | script computing everything | solvi, one by one | solvi, `workers=8` |
|---|---:|---:|---:|
| fast track? | 1 122 ms | 503 ms | 313 ms |
| full decision + payout | 1 122 ms | 773 ms | 461 ms |
| all five questions | 1 122 ms | 1 025 ms | 463 ms |
| expired policy (hard check settles it) | 1 122 ms | 152 ms | 153 ms |

A rule-only decision on a small catalog takes well under a millisecond; on documents the extractor dominates (about 39 ms per
receipt with the one-pass extractor on an A100).

## When to use it

- The answer is a computation or a rule over a few values found in a document or a record: receipts, invoices,
  contracts, requests, orders.
- You need to show why: auditors, compliance, or a human reviewing low-confidence cases.
- Some rules are non-negotiable (hard checks), and the rest can be learned from about 100 labeled examples.

## When not to use it

- Open-ended free-text questions or generated answers. solvi answers yes/no and choice questions only.
- New fields with no labeled examples. Extracting a field from its description alone does not work yet (14% and 66% on
  two held-out fields); a universal extractor is coming.
- No labels at all. Plan on roughly 100 labeled documents (field positions) per task.
- CPU-only deployment with a quantized model: the int8 ONNX extractor loses up to 12 points on amounts, company
  names and addresses. fp32 on CPU keeps accuracy but takes about 0.7 s per receipt on 2 cores.

## Examples

| File | What it shows |
|---|---|
| [examples/01_leave_request.py](examples/01_leave_request.py) | Leave request from a plain dict: rules, hard checks, parts the strategist skips |
| [examples/02_shop_order.py](examples/02_shop_order.py) | Shop order: two rule-based questions plus a "suspicious?" question learned from history with `fit` |
| [examples/03_invoices.py](examples/03_invoices.py) | Invoice approval from text: regex extractors with quotes, four questions, one learned |
| [examples/04_refunds.py](examples/04_refunds.py) | Refund e-mails: yes/no learned from examples under a hard "within 30 days" check |
| [examples/05_tic_tac_toe.py](examples/05_tic_tac_toe.py) | Tic-tac-toe agent: each move is an answer with its reason (win, block, fork, ...); a hard check rejects invalid boards |
| [examples/06_learned_rules.py](examples/06_learned_rules.py) | `learn_rule`: route parcels to delivery zones from free-form addresses with a readable if-then list learned from labels |
| [examples/09_strategy_at_scale.py](examples/09_strategy_at_scale.py) | Insurance claim desk: the strategist generates a different plan per question set, hard checks first with early exit, slow services in parallel; timed |
| [examples/10_learn_in_milliseconds.py](examples/10_learn_in_milliseconds.py) | `fit_fast`: a new question learned in milliseconds, then corrected one example at a time (each correction ~0.2 ms, nothing retrained) |
| [examples/11_answer_types_and_constraints.py](examples/11_answer_types_and_constraints.py) | Multi-label and ordinal answers tied by constraints between answers; contradictions in learned answers are repaired by joint decoding |
| [examples/07_receipts_model.py](examples/07_receipts_model.py) | Expense check on a scanned receipt: a receipts-tuned extractor cites each field, rules and a hard check decide (needs `solvi[model]`) |
| [examples/08_contracts_by_description.py](examples/08_contracts_by_description.py) | Contract review with fields defined only in words: the general extractor reads the whole contract, cites clauses or says "absent" (needs `solvi[model]`) |

Run them from a clone: `python examples/01_leave_request.py`.

## More

- [docs/guide.md](docs/guide.md): full API walkthrough.
- [docs/benchmarks.md](docs/benchmarks.md): setups, per-question numbers, caveats.
- [benchmarks/](benchmarks/): dataset loaders and benchmark scripts (SROIE, CORD, CUAD, Kleister-NDA).
- Tests: `pytest`.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

A paper is in preparation. Until then, please cite the repository:

```bibtex
@software{solvi,
  title  = {solvi: verifiable decision systems from catalogs of functions and checks},
  author = {mxkuzn and solvi contributors},
  year   = {2026},
  url    = {https://github.com/solvi-ai/solvi}
}
```
