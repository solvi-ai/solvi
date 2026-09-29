# solvi vs asking an LLM: data, runner, raw answers

This folder reproduces [docs/vs_llm.md](../../docs/vs_llm.md). It holds the four sets, the written policies the LLMs
get, a runner for the three arms (solvi alone, a model answering directly, the same model inside solvi), a scorer, and
every raw answer of the published run. A model is an LLM over chat completions or a decision model over the System One
API (Jev by TypeSafe, hosted; Jeeves by PostHog, open weights on your own GPU). The scorer recomputes every table from those answers without an API key or a
model.

| File | What |
|---|---|
| `bench.py` | the runner (`solvi`, `direct`, `inside`), the scorer (`score`), the task 11 claim check (`claims`) |
| `models.json` | the model registry: id, label, API (chat or System One), extra request fields, endpoint and key variable per model |
| `policies.py` | the written policy of each task, what the LLM gets instead of the catalog; question paraphrases |
| `build_data.py` | how the sets were built (generators with their seeds, the Banking77 mapping); checks the shipped files |
| `data/{G,R,P,T}.jsonl.gz` | the sets: gallery (117 cases), refunds (240), 3-way match (240), bank messages (520) |
| `data/blind_claims.json` | 80 messages about double charges written blind, for the task 11 claim reader |
| `raw/` | the raw answers of the published run: `a_solvi_*`, `b_<model>_*` (directly), `c_<model>_*` (inside solvi), `requests.jsonl.gz` (latency and cost of every request); Jev's are `b_jev-1.13_*` and `c_jev-1.13_*`, Jeeves' `b_jeeves_*`, `c_jeeves_*` (reasoning on) and `b_jeeves-nothink_*`, `c_jeeves-nothink_*` |
| `expected.json` | every number of the published tables, and the task 11 numbers before and in 0.7.0 |
| `make_playground_bundle.py` | builds `spaces/playground/vs_llm.json`, the data of the playground's "solvi vs LLM" tab: curated refund and 3-way-match cases, every arm's saved answers on them, the summary table (`--check`: is it up to date) |

The sets take 83 KB and the raw answers 1.5 MB, all compressed except the 11 KB of blind messages.

## Setup

From a clone of the repository:

```
uv sync                       # solvi 0.7.0 from the checkout (numpy, scipy, pydantic): enough for everything but the two below
uv sync --extra model         # torch + transformers: only for the bank messages with solvi-large
uv run --with datasets ...    # only to rebuild the bank set from Banking77 (build_data.py --with-T)
```

The runner uses the standard library for HTTP. Nothing else is needed.

## Check the published tables (free, a few seconds)

```
uv run python benchmarks/vs_llm/bench.py score --check
```

This prints the per-set tables from `raw/` and compares every number with `expected.json`. Expected output ends with:

```
spent: $39.50 {"grok-4.7": 34.32, "gpt-oss-120b": 2.07, "qwen3-235b-2507": 0.62, "deepseek-v3.2": 2.21, "jev-1.13": 0.27, "jeeves": 0.0, "jeeves-nothink": 0.0}

check against expected.json: every number matches
```

`--json results.json` writes the scored results, with tags and per-question accuracy, in the layout of `expected.json`.

## Rerun the solvi arm (free, offline)

```
uv run python benchmarks/vs_llm/bench.py solvi --sets G,R,P          # about 3 seconds on a CPU
uv run python benchmarks/vs_llm/bench.py score --raw benchmarks/vs_llm/raw --raw benchmarks/vs_llm/out --check
```

The first `--raw` gives the LLM answers from the published run, and the second overrides the solvi answers with yours.
The LLM arms' violation counts use the solvi arm's hard-check statuses, so they are recomputed with yours too.
Latency and cost of the new runs are not compared. Expected:

```
check against expected.json: every number matches (latency and cost of the new runs not compared)
  note: R, solvi arm: matches gallery task 11 as released in 0.7.0 (the published tables used the 0.6.1 catalog of task 11; see task11_claim_reader in expected.json)
```

The published run used gallery task 11 as released in 0.6.1; 0.7.0 rewrote how it reads the customer's claim. With the
0.7.0 gallery, the refund set gives accuracy 1.000, answered alone 99.7% with 0% error. To get the published 0.894 run
the arm on the 0.6.1 gallery (the other eleven tasks are the same in both):

```
git archive v0.6.1 gallery | tar -x -C /tmp/solvi-0.6.1
VS_LLM_GALLERY=/tmp/solvi-0.6.1/gallery uv run python benchmarks/vs_llm/bench.py solvi --sets G,R,P --out /tmp/vs_061
uv run python benchmarks/vs_llm/bench.py score --raw benchmarks/vs_llm/raw --raw /tmp/vs_061 --check
```

Bank messages with [solvi-large](https://huggingface.co/solvi-ai/solvi-large) (396M). On a laptop GPU the decisions
take about 15 seconds after the model loads; a CPU works but takes much longer. The published calibration comes back
exactly (`act_guard` threshold 0.522, answering 78.8% of the calibration messages), and the test answers match the
published ones message for message:

```
uv sync --extra model --extra onnx       # torch for the decider; huggingface_hub for the download
uv run solvi models pull solvi-ai/solvi-large --backend torch
uv run python benchmarks/vs_llm/bench.py solvi --sets T              # or --decider /path/to/checkpoint, --device cpu
```

### The task 11 claim reader on the blind messages

```
uv run python benchmarks/vs_llm/bench.py claims                                   # 0.7.0: acc 0.8625, alone 0.8375, error 0.149
uv run python benchmarks/vs_llm/bench.py claims --gallery /tmp/solvi-0.6.1/gallery   # before: acc 0.425, alone 1.0, error 0.575
```

## Run the model arms (costs money)

Any OpenAI-compatible chat-completions endpoint works. By default the runner uses OpenRouter with the key in
`$OPENROUTER_API_KEY` (`llm` still works as another name for `direct`):

```
export OPENROUTER_API_KEY=...
uv run python benchmarks/vs_llm/bench.py direct gpt-oss-120b                # directly, all four sets
uv run python benchmarks/vs_llm/bench.py inside gpt-oss-120b --sets G,R,P   # inside solvi
uv run python benchmarks/vs_llm/bench.py score --raw benchmarks/vs_llm/raw --raw benchmarks/vs_llm/out
```

The registered models are in `models.json`: `grok-4.7`, `gpt-oss-120b`, `qwen3-235b-2507`, `deepseek-v3.2` (with
reasoning on) and the decision models `jev-1.13`, `jeeves` and `jeeves-nothink`. To add a model for good, add an entry there (id, label, extra request fields, and optionally `base_url` and
`key_env` for another endpoint). `score` finds every raw file named `<kind>_<model>_<set>.jsonl.gz` in the folders it
reads and prints it as another row: `b` is the model answering directly, `c` is the model inside solvi. A new kind of arm
is one line in `ARM_KINDS` in `bench.py` (its label and whether it is scored like an LLM with a stated confidence or like
solvi, which answers or escalates), plus the code that writes its raw files. `--check` notes arms that are not in the
published run and leaves them unchecked. For a one-off model or endpoint:

```
uv run python benchmarks/vs_llm/bench.py direct my-model --model-id vendor/model-name \
    --base-url http://127.0.0.1:8000/v1 --key-env MY_KEY [--extra-body '{"reasoning": {"effort": "high"}}']
```

### A decision model (Jev)

`jev-1.13` has `"api": "systemone"` in `models.json`: `typesafe/jev-1.13` on OpenRouter (base URL
`https://openrouter.ai/api`, the same `$OPENROUTER_API_KEY`), with `{"provider": {"only": ["TypeSafe"],
"allow_fallbacks": false}}` merged into every request. It costs $0.042 per million input tokens; output is free.

```
uv run python benchmarks/vs_llm/bench.py direct jev-1.13            # about $0.11 for all four sets with the stability variants
uv run python benchmarks/vs_llm/bench.py inside jev-1.13            # about $0.14
```

A decision model writes no text: it picks one of the given options and returns a probability for each. So the questions
are put to it differently from an LLM:

- Directly, one request per case goes to OpenRouter's Decisions API (`direct_path` in `models.json`; it takes the same
  request as the System One API at `/v1/systemone`). The state is `{"policy": ..., "input": ...}`. Every single-answer
  question is a choice among its options plus an `abstain` option with a description, because the API has no "not
  stated". A multi-label question becomes one yes/no question per option (an option applies at probability ≥ 0.5). The
  confidence is the probability of the chosen option.
- Inside solvi, each question is a `solvi.systemone` decision over the same text as for an LLM, with the same
  `abstain` option; choosing it escalates. Multi-label questions always abstain (the API has none). The runner is the
  `opener` of `solvi.systemone`: it adds the provider pin, retries, caches and logs cost and latency.

Jev is not deterministic: the same request sent again changed 1-3% of its answers on the rule sets, all near ties. The
published run marked its repeat requests with another `user` value than the runner's, so a rerun draws new repeats, and
the scores can differ by a few answers.

### An open decision model on your own GPU (Jeeves)

`jeeves` and `jeeves-nothink` are PostHog's [Jeeves](https://github.com/PostHog/jeeves) (weights
[`PostHog/jeeves`](https://huggingface.co/PostHog/jeeves), Apache-2.0), served by its own `python -m inference.serve` at
`http://127.0.0.1:8009` (no key, no price). It speaks the same System One API plus an `options` object: `jeeves` sends
the authors' fast setting `{"max_think": 768, "nothink_threshold": 0.9}` (reasoning on), `jeeves-nothink` sends
`{"think": false}`. It needs a CUDA GPU with about 21 GB for the bf16 weights.

```
uv run python benchmarks/vs_llm/bench.py direct jeeves --budget     # about 1,800 requests; ~2.5 h on one A100, 6 at a time
uv run python benchmarks/vs_llm/bench.py inside jeeves --budget     # ~1 h
uv run python benchmarks/vs_llm/bench.py direct jeeves-nothink      # every set in full, ~20 min; inside ~5 min
uv run python benchmarks/vs_llm/bench.py inside jeeves-nothink
```

The published run used one Colab A100 40 GB (bf16; no FP8 on that GPU) and a small batching front-end over their
engine, so that several requests share a forward pass; its answers matched their own server on 195 of 198 probe
questions with reasoning and 198 of 198 without. Keep the number of concurrent requests at the server's batch size:
with 32 requests against 6 rows, requests waited so long in the queue that they timed out and were sent again. The
latency in `raw/` includes that queueing; the page quotes the latency of their own server, one request at a time.

Every request is sent at temperature 0 with seed 0 and JSON mode. Responses are cached on disk in
`out/cache_<model>.jsonl`, keyed by the request, so an interrupted run resumes without paying twice. `out/requests.jsonl`
logs the latency and cost of each request. `--budget` repeats the published run's cuts for the expensive model: stability
on half the subsample; inside solvi, refunds on half the test cases and 3-way match on half the calibration and a quarter
of the test cases.

Cost of the published run, per model, from `raw/requests.jsonl.gz` (OpenRouter's reported cost, 2026-09-28/29;
"directly" includes the 15-case pilot, the stability variants and replies that came back empty or cut off):

| Model | Directly, all four sets | Inside solvi | Total |
|---|---|---|---|
| Grok 4.7 | $10.69 | $23.64 (budget subsets, no bank messages) | $34.32 |
| gpt-oss-120b | $0.99 | $1.08 | $2.07 |
| Qwen3-235B-2507 | $0.23 | $0.40 | $0.62 |
| DeepSeek-V3.2 | $2.21 | not run | $2.21 |
| Jev 1.13 (2026-09-29) | $0.13 | $0.14 | $0.27 |
| Jeeves, both settings (2026-09-29) | own GPU | own GPU | about 4.5 A100-hours (24 Colab compute units) |

Grok inside solvi on the full sets, bank messages included, is about 2,550 questions: at $18.51 per 1,000 questions,
roughly $47 instead of $23.64. Rerunning everything as published costs about $40, most of it Grok. Jev's numbers
exclude a pilot of about $0.002.

Rerunning the inside-solvi arm on the released 0.7.0 may not give exactly the published numbers. The prompts of
`solvi.llm` are the same, but a quote that is not in the text no longer escalates a question that asks for no
evidence. Models behind a name also change over time.

## The sets

`build_data.py` documents how each set was built and checks the shipped files:

```
uv run --with datasets python benchmarks/vs_llm/build_data.py --gallery /tmp/solvi-0.6.1/gallery --with-T
# G: built 117 rows, shipped 117: identical
# R: built 240 rows, shipped 240: identical
# P: built 240 rows, shipped 240: identical
# T: built 520 rows, shipped 520: identical
```

- **R and P** are ours: one generator run with seed 3737 (240 cases each). The right answers come from the scenario's
  parameters and decimal arithmetic, not from the catalog. Split: cases with `i % 5 < 2` are calibration, the rest test.
- **T** comes from the [Banking77](https://huggingface.co/datasets/legacy-datasets/banking77) test split by PolyAI (CC BY
  4.0; Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", 2020). 380 messages of 34 intents
  are mapped to 6 queues by `MAP` in `build_data.py`, fixed before any model was run. 20 hand-written messages with no
  request (right answer: abstain) are added. Then there are 120 adversarial variants of test messages: 60 with an
  instruction appended, 60 with a distracting first sentence. Seed 37. `data/T.jsonl.gz` contains Banking77 messages,
  redistributed under CC BY 4.0 with this attribution.
- **G** is every labelled case of the gallery as released in 0.6.1 (0.7.0 added cases to task 11 only).
- **blind_claims.json**: 80 messages written by a separate agent that had seen neither the task 11 catalog nor the
  refund set. There are 20 claims, 20 denials, 15 unclear and 25 about something else, including traps like
  "double-check" and "declined twice".

Each row: `{"set", "id", "task", "state", "gold": {question: answer or "abstain"}, "tags", "split"}`.

## Raw answer format

One JSON object per line, gzipped. Each has `id`, `variant` (`base`, `rep`, `order`, `para`), and `answers`
(`{question: [answer, confidence]}`, or null when there was no usable reply). LLM records also carry `content` (the
reply text), `finish`, `ms`, `cost`, `usage` and `provider`; Jev's direct records carry `raw` (the API's answers, with
every option's probability) instead of `content`. solvi records carry `status` and `forced` per question and
`replay_ok`. Inside-solvi files start with a `{"guards": ...}` line holding the calibrated thresholds, and each record
keeps the LLM's raw decision (`raw`) and why an answer was escalated (`why`).

The stability variants of the published run reordered options and keys with a per-process random seed. The runner now
seeds the reordering from the case id, so a rerun is repeatable but draws different orders than the published run.
Jev's run already seeded it from the case id.
