# Benchmarks

This page gives the setups, per-question numbers and caveats behind the results in the [README](../README.md). All
numbers come from pre-registered experiments: success criteria were written down before each run, and results are
reported whether or not the criteria were met. Dataset loaders are in [benchmarks/datasets/](../benchmarks/datasets/).

## Setup

- **solvi**: ModernBERT-large field extractors (start/end pointer heads) plus ordinary functions, checks and rules; rules
  that are hard to write by hand are learned with `learn_rule`; confidence calibrated per question with `calibrate` on
  held-out training documents.
- **Baseline (Laya)**: a ModernBERT-large model that answers the typed questions directly from the text, fine-tuned with
  its authors' recipe (4 epochs, 1024-token context) on the same training documents and question labels.
- Accuracy counts abstentions as errors.
- ECE is the expected calibration error of answer confidence.
- "Answered at >= 99% precision" is the largest share of questions that can be answered, by thresholding confidence,
  while keeping accuracy on the answered part at or above 99%.
- Hardware: one A100 GPU (Colab), batch size 1 for timing.

## SROIE receipts

Real scanned receipts with OCR text (ICDAR 2019). Train 626, test 361. Six questions, with ground truth derived from the
reference fields (company, date, address, total):

| Question | Type | solvi | Laya |
|---|---|---|---|
| total band (under 10 / 10-30 / 30-100 / over 100) | choice | 99.4% | 95.3% |
| bought on a weekend | yes/no | 100% | 73.6% |
| year | choice | 99.7% | 99.7% |
| quarter | choice | 100% | 99.7% |
| state from the address (learned rule) | choice | 89.2% | 88.4% |
| seller is a "Sdn Bhd" company | yes/no | 98.6% | 99.2% |
| **mean** | | **97.8%** | 92.6% |
| ECE | | 0.008 | 0.037 |
| answered at >= 99% precision | | 88% | not measured |
| time per receipt | | 39 ms | 42 ms |

solvi used 563 receipts for training and 63 for calibration; Laya used all 626. The state question uses a rule list
learned from examples (21 one-literal rules such as "address has a number starting with 43 -> Selangor"). A hand-written
rule reached only 60.9% on the same question, because OCR text often lacks the state name.

Field extraction accuracy (one pass, four fields): total 95.6%, date 97.0%, company 84.8% (string similarity >= 0.8),
address 88.6% (word F1 >= 0.5). Every extracted value is a substring at the reported offsets.

An earlier version with a hand-written state rule and four separate extractor passes scored 93.0% (140 ms per receipt).

### Learning curve

| Training receipts | solvi | Laya |
|---|---|---|
| 50 | 91.2% | - |
| 100 | **95.9%** | 80.0% |
| 300 | 97.6% | 86.6% |
| 563 (solvi) / 626 (Laya) | 97.7% | 92.6% |

The computed questions (weekend, year, quarter) reach 99-100% with 50 receipts. Only the learned state question keeps
improving with data (61% -> 84% -> 90%). With 100 receipts Laya gets the total band right 39% of the time and the weekend
question 73%.

## CORD receipts

Indonesian receipts (CORD v2). Train 800, thresholds and calibration 100, test 100. Five fields found by description
with "no answer" support (`LongSpanExtractor`), four questions:

| Question | solvi | Laya |
|---|---|---|
| total band (1 of 4) | 100% | 97.9% |
| has tax | 99% | 98% |
| paid in cash | 99% | 94% |
| change is correct (change = cash - total, computed) | 96.1% | 94.1% |
| **mean** | **98.5%** | 96.0% |
| ECE | 0.011 | 0.009 |
| answered at >= 99% precision | **99.7%** | 14.2% |
| time per receipt | 125 ms (one pass per field) | 37 ms |

The accuracy gap is small (2.5 points, below the 3-point target we set in advance). The larger difference is in how
useful the confidence is: Laya's confidence is calibrated on average but does not separate right from wrong answers.

## CUAD contracts

Real commercial contracts annotated by lawyers (CUAD v1). Train 408 (40 of them for thresholds and calibration), test
102 (official test split). Median length about 33,000 characters. Five questions:

| Question | solvi | Laya |
|---|---|---|
| governing law (Delaware / New York / California / other) | 100% | 66.3% |
| signing era | 95.5% | 96.6% |
| non-compete clause present | 95.1% | 76.5% |
| exclusivity clause present | 91.2% | 79.4% |
| cap on liability present | 92.2% | 68.6% |
| **mean** | **94.8%** | 77.5% |
| ECE | 0.027 | 0.040 |
| answered at >= 99% precision | 23% | 7% |
| time per contract | 1.03 s | 0.10 s |

Laya's input is limited to the first 1024 tokens, so clauses later in the contract are invisible to it; for the clause
questions it is close to always giving the majority answer. solvi reads the whole contract in overlapping windows, which
makes it about 10x slower here (five fields times all windows).

## Rule-only decisions

Synthetic invoice approval and refund tasks (1000 documents each, regex extractors with quotes):

- questions with a rule: 100% correct, and no wrong answer ever had status `ok`;
- learned risk level (no rule, 200 examples): 87.8% (label noise ceiling about 95%), versus 55.5% for a bag-of-words
  model on the text;
- refunds learned from examples: 96.9%, equal to the label-noise ceiling; the hard "within 30 days" check forced "no" on
  343 of 343 late requests;
- catalog parts not needed for the questions were executed 0 times in 1000 decisions;
- trace replay: 0 false alarms on 1000 clean traces; 500/500 naive and 500/500 hash-consistent substitutions caught, with
  the altered step identified in every case;
- 0.38 ms per decision (95th percentile 0.51 ms); training a head took 0.5-2.5 s.

## Overhead per ask

`benchmarks/ask_overhead.py` times `ask` without a real model: the twelve gallery entries with their own cases (catalog
code only) and the project `solvi init --template support --with-model` writes (a decision answered by a keyword
stand-in decider, so what is timed is solvi's own work around a model call). Each setting runs on the same states, in
turn, 10 passes; medians of the per-ask wall time, on a shared 20-core Linux machine under load (differences under about
10% are noise there):

| Setting | gallery, ms | decider, ms |
|---|---|---|
| 0.5.0-style: no trace fingerprint, options in the caller's order, no store | 1.15 | 0.57 |
| 0.7 defaults: `trace.fingerprint`, canonical option order | 1.17 | 0.59 |
| + a calibrated guarantee on the decision (`act_guard`) | — | 0.61 |
| + every response stored, JSON lines | 2.40 | 1.24 |
| + every response stored, SQLite | 2.45 | 1.42 |

The 0.7 features cost a few percent each (the fingerprint about 3%, 40-100 µs; the guarantee record about 4%); no
setting is more than 10% slower than the 0.5.0-style one. Storing a response with its whole trace doubles the time of a
small ask (serializing the trace to JSON and appending a hash-chained record: about 1 ms). The released 0.5.0 (PyPI) on
the 0.5.0 gallery: 1.09-1.14 ms per ask; this version on the same gallery: 0.99-1.22 ms.

```
uv run python benchmarks/ask_overhead.py [--quick] [--only gallery|decider] [--json out.json]
```

## Known negative results

- **New field from its description only.** An extractor trained on seven fields of two datasets, then asked for a field
  it never saw labeled: tax 14%, change 66% (with labels: 95% and 98%). A universal extractor for this is in progress.
- **int8 on CPU.** ModernBERT-large exported to ONNX and dynamically quantized to int8: total 83.1%, company 73.7%,
  address 76.2% (down 11-12 points), date unchanged, about 320-410 ms per receipt. fp32 ONNX on CPU keeps accuracy at
  about 0.7 s per receipt (2 cores).

## Writing catalogs with an LLM

Not a library feature, but a workflow we tested: an LLM receives a plain-language task description and writes the solvi
module (`@cat.fn`, `@cat.check`, `@cat.extract`, `@cat.rule`). The module is executed on 40 examples with known answers,
and errors are returned for up to two fix rounds. Agreement with our reference catalog on 300 new examples:

| Domain | Qwen3-32B | Grok 4.7 |
|---|---|---|
| leave request (date rules, 2 questions) | 100% (after 1 fix) | 100% |
| shop order (2 questions) | 100% | 100% |
| invoice (regex parsing + 3 questions) | 58.7% (fixes did not help) | 100% |
| receipt (5 questions on given fields) | 98.3% (after fixes) | 98.7% (after fixes) |

No runtime exceptions in the final catalogs; compiling a domain cost under $0.01 with Qwen3-32B. Executing drafts
against labeled examples is essential: Qwen's first leave-request module loaded fine and scored 0%. The reference is our
own catalogs, so this measures agreement with them, not the correctness of the policy.

## Caveats

- One seed per configuration.
- Question labels on SROIE were derived by rules from the reference fields. This favors computed questions for solvi;
  learned rules also reproduce the labeling rule's mistakes (for example, a district of Kuala Lumpur labeled "other").
- Laya numbers on SROIE come from a separate run on the same test receipts and hardware. On CUAD, only solvi was rerun
  after a windowing bug was fixed; the bug did not affect Laya.
