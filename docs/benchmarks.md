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

## Known negative results

- **New field from its description only.** An extractor trained on a few fields does not find a field it never saw
  labelled; one trained on a broad mix does better. The [extract-base](https://huggingface.co/solvi-ai/extract-base)
  card gives both on CORD's tax and change (a model trained on 7 fields only: 14% / 66%; extract-base: 46.5% / 67.9%).
  For production, label some documents and fine-tune.
- **int8 on CPU.** A ModernBERT-large extractor exported to ONNX and dynamically quantized to int8 lost accuracy on
  most fields; fp32 ONNX on CPU keeps it, at a slower speed.

## Writing catalogs with an LLM

Not a library feature, but a workflow we tried: an LLM receives a plain-language task description and writes the solvi
module (`@cat.fn`, `@cat.check`, `@cat.extract`, `@cat.rule`). The module is executed on examples with known answers,
and errors are returned for a couple of fix rounds; the result is compared with a reference catalog on new examples, in
domains such as leave requests, shop orders, invoices and receipts. A strong model wrote catalogs that agreed with the
reference almost everywhere; a smaller one did on simple domains and failed on regex parsing. Executing drafts against
labelled examples is essential: a first module can load fine and answer nothing right. The reference is our own
catalogs, so this measures agreement with them, not the correctness of the policy.

## Caveats

- The extraction runs used one seed per configuration.
- Question labels on SROIE were derived by rules from the reference fields. This favors computed questions for solvi;
  learned rules also reproduce the labeling rule's mistakes (for example, a district of Kuala Lumpur labeled "other").
