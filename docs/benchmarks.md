# Benchmarks

This page gives the setups and numbers behind the results in the [README](../README.md). Every number on it names its
source: a script in [benchmarks/](../benchmarks/) that re-runs it, or a published model card. Results without such a
source are described in plain words. Dataset loaders are in [benchmarks/datasets/](../benchmarks/datasets/).

How solvi compares to giving the same rules to an LLM (Grok 4.7, gpt-oss-120b, Qwen3-235B, DeepSeek-V3.2), answering
directly or inside solvi, is on its own page: [solvi vs asking an LLM](vs_llm.md).

## Extraction on public datasets

The loaders in [benchmarks/datasets/](../benchmarks/datasets/) read SROIE receipts, CORD v2 receipts, CUAD contracts and
Kleister-NDA agreements into documents, reference fields or clause spans, and typed questions with ground truth. The
setup they were written for: ModernBERT-large field extractors (start/end pointer heads) plus ordinary functions, checks
and rules, with confidence calibrated per question on held-out training documents, against a model of the same size
fine-tuned to answer the typed questions directly from the text.

The published extractors report their own numbers on these datasets: the
[extract-base](https://huggingface.co/solvi-ai/extract-base) card (fields never trained on CORD, typed questions over
CUAD clause types it never saw, Kleister-NDA and SROIE without labels, and with a few labelled documents) and the
[extract-receipts](https://huggingface.co/solvi-ai/extract-receipts) card (typed questions over CORD receipts through
solvi rules, SROIE fields).

What the setup showed in plain words: questions computed from extracted fields need few training documents, while a
learned rule keeps improving with more; a model that answers from the first part of a long contract misses clauses
later in it, where solvi reads the whole contract in windows and is slower for it; and a confidence that is calibrated
on average can still fail to separate right from wrong answers, which decides how much can be answered at high
precision.

## Overhead per ask

`benchmarks/ask_overhead.py` times `ask` without a real model: the fifteen gallery entries with their own cases (catalog
code only) and the project `solvi init --template support --with-model` writes (a decision answered by a keyword
stand-in decider, so what is timed is solvi's own work around a model call). Each setting runs on the same states, in
turn, 10 passes; medians of the per-ask wall time, on a shared 20-core Linux machine under load (differences under about
10% are noise there). The script puts its stores in the system's temporary folder, which was a RAM disk (tmpfs) on
that machine, so the two "stored" rows are without a disk write:

| Setting | gallery, ms | decider, ms |
|---|---|---|
| 0.5.0-style: no trace fingerprint, options in the caller's order, no store | 1.15 | 0.57 |
| 0.7 defaults: `trace.fingerprint`, canonical option order | 1.17 | 0.59 |
| + a calibrated guarantee on the decision (`act_guard`) | — | 0.61 |
| + every response stored, JSON lines | 2.40 | 1.24 |
| + every response stored, SQLite | 2.45 | 1.42 |

The 0.7 features cost a few percent each (the fingerprint about 3%, 40-100 µs; the guarantee record about 4%); no
setting is more than 10% slower than the 0.5.0-style one. Storing a response with its whole trace doubles the time of a
small ask (serializing the trace to JSON and appending a hash-chained record: about 1 ms) — in memory. On a disk the
SQLite store commits every response (one sync per ask): with the temporary folder on an NVMe disk
(`TMPDIR=<a folder on the disk> ... --quick`, 5 passes, 0.7.1) the SQLite row measured 19.1 ms (gallery) and 17.2 ms
(decider), 28 and 51 times a default ask, while JSON lines stayed at 1.4 and 0.8 ms (about 2×). Where every response is
stored on a disk, the store, not solvi, sets the time of an ask. The released 0.5.0 (PyPI) on
the 0.5.0 gallery: 1.09-1.14 ms per ask; this version on the same gallery: 0.99-1.22 ms.

```
uv run python benchmarks/ask_overhead.py [--quick] [--only gallery|decider] [--json out.json]
```

## solvi vs asking an LLM

The gallery, generated refund and 3-way-match sets with values exactly at limits, conflicts, missing facts and injected
instructions, and Banking77 routing, given to solvi and to four LLMs (directly, and inside solvi under `act_guard`).
Strong reasoning models followed the short written rules as accurately as solvi. solvi's advantage there is cost,
latency, repeatability, audit and hard checks that hold whatever the model says. On free text the LLMs were clearly
better. A hosted decision model (Jev by TypeSafe) was close to the best LLM on free text and near the cheap LLM on
the written rules. An open decision model that reasons first (Jeeves by PostHog) did better on the rules (0.863 and
0.912), still below the strong LLMs and the catalog. Tables, caveats and the reproduction are on [solvi vs asking an LLM](vs_llm.md). Data, runner and raw answers
are in [benchmarks/vs_llm/](../benchmarks/vs_llm/).

## Nine public tasks

[benchmarks/tasks/](../benchmarks/tasks/) is a stand of nine real tasks on public data, each with a scorer, a
baseline that does not use solvi and one solution with solvi: the variant chosen on dev, on the library's current
pieces. Five are the kind of task solvi is built for; four were picked because they fit it badly. `fetch.sh` downloads
the data from its sources, `prepare.py` makes the fixed splits, and every LLM answer is cached by its request, so a
rerun of the published numbers costs nothing once the cache is filled. All on eval, solvi 0.8.0, `openai/gpt-oss-120b`
where an LLM is used:

| Task | Metric | Baseline | solvi | Script |
|---|---|---|---|---|
| τ-bench retail, 30 tasks | solved; changes not in gold; calls the environment refused | 18; 14; 10 | 14; 13; 0 | `taubench/solution.py` |
| CUAD, 1,025 questions | accuracy; quotes in the contract; answered alone, wrong among them | 0.882; 260 of 398; 100%, 11.8% | 0.899; 263 of 263; 67.6%, 4.0% | `cuad/solution.py` |
| RAGTruth, 600 responses | F1; answered alone, wrong among them | 0.784; 100%, 22.0% | 0.766; 19.0%, 12.3% | `ragtruth/solution.py` |
| Banking77 stream, 2,000 | before / after the shift: answered alone, wrong among them (promise 5%) | 83.7%, 3.9% / 65.0%, 22.5% | 57.6%, 0.7% / 13.7%, 0.7% | `banking77/solution.py` |
| German Credit, 1,000 × 2 versions | decisions equal to the policy | 2,000 of 2,000 | 2,000 of 2,000, with the audit | `credit/solution.py` |
| BIRD mini-dev, 150 | right; answered; wrong among answered | 78; 100%; 48.0% | 73; 74.7%; 34.8% | `bird/solution.py` |
| Abt-Buy, 1,916 pairs | F1; offers with two counterparts | 0.872; 19 | 0.933; 0 | `abtbuy/solution.py` |
| NATURAL PLAN, 3 × 100 | right: calendar / meetings / trips | 92 / 75 / 43 | 95 / 100 / 98 | `naturalplan/solution.py` |
| NAB, 33 series | F1; false alarms | 0.391; 115 | 0.361; 119 | `nab/solution.py` |

What it says: solvi did not make a model more accurate — on RAGTruth, BIRD, τ-bench and NAB the solution is at or
below the baseline. Where it won, something other than the model did the work: comparison code and a fitted head
(Abt-Buy, supervised on 5,743 labelled pairs against a zero-shot baseline), a search through checks (NATURAL PLAN), an
open-set threshold that keeps its promise when new intents arrive (Banking77), a trust signal under a guarantee with
every quote checked (CUAD). On every task each decision is stored and replays. Caveats per task (one run of τ-bench with
a simulated customer, the seed of Banking77's simulated new intents, the supervision on Abt-Buy) and the cost of a run
from an empty cache are in [benchmarks/tasks/README.md](../benchmarks/tasks/README.md).

## Known negative results

- **New field from its description only.** An extractor trained on a few fields does not find a field it never saw
  labelled; one trained on a broad mix does better. The [extract-base](https://huggingface.co/solvi-ai/extract-base)
  card gives both on CORD's tax and change (a model trained on 7 fields only: 14% / 66%; extract-base: 46.5% / 67.9%).
  For production, label some documents and fine-tune.
- **int8 on CPU.** Dynamic int8 quantization changed the spans of the published extractor (the
  [extract-base](https://huggingface.co/solvi-ai/extract-base) card), so no int8 export is provided: use fp32 ONNX on
  a CPU, and compare a quantized export of your own with it on your fields before using it.

## Writing catalogs with an LLM

Not a library feature, and no script for it is in this repository — a workflow, as advice: an LLM receives a
plain-language task description and writes the solvi module (`@cat.fn`, `@cat.check`, `@cat.extract`, `@cat.rule`).
Execute the module on examples with known answers and return the errors for a couple of fix rounds, then compare the
result with a catalog you trust on new examples. Executing drafts against labelled examples is essential: a first
module can load fine and answer nothing right. Text parsing (regular expressions) is where a draft most needs those
examples. Agreement with a reference catalog says the draft matches it, not that the policy is right.

## Caveats

- The extraction runs used one seed per configuration.
- Question labels on SROIE were derived by rules from the reference fields. This favors computed questions for solvi;
  learned rules also reproduce the labeling rule's mistakes (for example, a district of Kuala Lumpur labeled "other").
