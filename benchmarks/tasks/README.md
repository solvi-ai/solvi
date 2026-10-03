# The task stand: nine public tasks, solved with solvi and without

Nine real tasks on public data, each with a scorer, a baseline that does not use solvi, and one solution with solvi. Five
are the kind of task solvi is built for (decisions under rules, a promise on the error, a record of every answer); four
were picked because they fit it badly — a query to write, thousands of decisions that must agree, a plan to find, a
numeric series — to show where the library stops. The numbers below are what these scripts print; where solvi did not
help, the table says so.

Every split is fixed (`prepare.py`, seed 20261002). `dev` is for fitting, choosing and calibrating; `eval` is only
scored. Every choice in a solution was made on dev.

## Results

All numbers on `eval`, measured with solvi 0.8.0 and `openai/gpt-oss-120b` (through OpenRouter) wherever an LLM is used.
"Answered alone" is the share a solution returns without a person; "wrong among answered" is the error on that share.

| Task | Metric | Baseline (no solvi) | solvi solution |
|---|---|---|---|
| τ-bench retail, 30 tasks | tasks solved; changes not in gold; changing calls the environment refused | 18; 14; 10 | 14; 13; 0 |
| CUAD, 1,025 contract questions | accuracy; found of 304; false claims of 721; answered alone, wrong among answered | 0.882; 243; 60; 100%, 11.8% | 0.899; 227; 27; 67.6%, 4.0% |
| RAGTruth, 600 responses | F1; answered alone, wrong among answered | 0.784; 100%, 22.0% | 0.766; 19.0%, 12.3% |
| Banking77 stream, 2,000 requests (promise: ≤ 5% wrong) | answered alone, wrong among answered — before the shift / after | 83.7%, 3.9% / 65.0%, 22.5% (broken) | 57.6%, 0.7% / 13.7%, 0.7% (kept) |
| German Credit, 1,000 applications × 2 rule versions | decisions equal to the policy; of the audit checks plain code cannot do (a stored record edited without changing its decision, rules changed vs data damaged, a rule reading a sensitive field) | 2,000 of 2,000; none | 2,000 of 2,000; all three |
| BIRD mini-dev, 150 questions | right; answered; wrong among answered | 78; 100%; 48.0% | 73; 74.7%; 34.8% |
| Abt-Buy, 1,916 pairs | F1; offers with two counterparts | 0.872; 19 | 0.933; 0 |
| NATURAL PLAN, 3 × 100 problems | right: calendar / meetings / trips | 92 / 75 / 43 | 95 / 100 / 98 |
| NAB, 33 series | F1; windows found of 71; false alarms | 0.391; 44; 115 | 0.361; 40; 119 |

What each row says, in a sentence:

- **τ-bench.** The guard did not solve more tasks (14 against 18; one run, a simulated customer — a few tasks are
  noise). It stopped every call the environment would have refused before it was made, and recorded 437 decisions that
  all replay. Asking for the customer's explicit yes costs turns, and the simulated customer sometimes ends the
  conversation instead.
- **CUAD.** Every quote is literally in the contract (263 of 263; the baseline's 260 of 398), fewer false claims, and
  `System.guarantee` on a trust score answers two thirds alone at 4.0% wrong, where the baseline cannot say "not sure".
  It finds fewer clauses (227 against 243).
- **RAGTruth.** The judge inside solvi is not better than the plain call (F1 0.766 against 0.784). Under a 10% risk
  promise only 19% is answered alone, with 2.3% of all responses answered alone and wrong — the promise holds, at a
  high price: the judge's confidence carries little information.
- **Banking77.** From request 1,000 on, 40% of the requests are about 20 intents the classifier never saw. A plain
  threshold (the baseline, or solvi's `System.guarantee` without new intents: 58.5% answered, 16.4% wrong after the
  shift) breaks the promise; `OpenSetGate` keeps it and flags the shift 68 requests after it starts, at the price of
  answering less.
- **Credit.** Same decisions as plain code; what solvi adds is the record, replay, a diff that names the changed rule or
  threshold per decision, tamper detection, telling a rule change from damaged data, and refusing a rule that reads a
  field the input does not declare.
- **BIRD.** solvi does not write better queries (73 against 78 right). It decides which ones can go out alone: wrong
  among those 34.8% instead of 48.0%. Answering only when all three sampled queries return the same rows: 63.3%
  answered, 28.4% wrong (same script). About a quarter stays wrong even when all checks pass — readings of the
  question that differ from the reference's.
- **Abt-Buy.** The offers are read by comparison code (`pairfacts.py`), not by solvi; a head fitted on the labelled dev
  pairs (`System.fit`) and "one counterpart per offer" (`solvi.core.sets.decide_set`) give 0.933. It is supervised (5,743
  labelled pairs), the baseline is zero-shot. Under a 1% risk promise 96.9% of pairs are answered alone, 0.59% wrong.
- **NATURAL PLAN.** No model proposes a plan: the problem is read into typed facts by rules and `solvi.core.slow.search` walks the
  candidates through the same hard checks (the largest search, a 10-city trip, took 11,414 asks; 10 cities have 3.6
  million orders).
- **NAB.** The detector picked on dev is below the z-score baseline on eval. solvi adds nothing to the alerts; it
  stores, replays and diffs them (2.7 ms a decision on an input of 3.7k floats).

Further numbers the same scripts print (all eval unless said):

| Script | What it shows |
|---|---|
| `abtbuy/solution.py --repair runs/baseline.jsonl` | "one counterpart" on the plain LLM's answers: F1 0.872 → 0.909; on the head's: 0.931 → 0.933 |
| `abtbuy/llm_pair.py` | the same LLM asked each pair through `solvi.core.deciders.llm`: F1 0.860 with the reply contract in the prompt (the default with reasoning), 0.837 with `response_format="json_schema"` (plain call 0.872); "one counterpart" on it 0.860 → 0.870; at `max_tokens=400`, 26 of 1,916 replies were cut off |
| `ragtruth/reply_format.py` | the judge through `solvi.core.deciders.llm` with the format enforced / with the contract in the prompt / the plain call: dev 0.744 / 0.791 / 0.802, eval 0.733 / 0.766 / 0.784; under the enforced format about a fifth of the replies had no reasoning (119 of 600 on dev) |
| `cuad/solution.py` | on dev, the trust score separates right from wrong answers with AUROC 0.83, the LLM's own confidence with 0.63 (14% answered alone at the same risk) |
| `banking77/solution.py --plain` | the plain promise without new intents: 80.8% answered, 2.5% wrong before the shift; 58.5%, 16.4% after |
| `bird/solution.py` | `runs/solution_unanimous.jsonl`: answer only when all three queries agree — 63.3% answered, 28.4% wrong |

## Data and licences

`fetch.sh` downloads every set from its own source. The prepared splits of the sets whose licence allows passing them
on are also in this repository, in `packed/prepared.tar.xz` (1.9 MB; `replies.py unpack-data`), with a NOTICE that
names each set's source and licence; the files of a set stay under its licence, not solvi's, and BIRD's under CC BY-SA
4.0. Abt-Buy states no licence: it is only ever downloaded, and neither its data nor the model's replies about it (they
quote the offers) are in the repository.

| Task | Data | Licence | eval / dev |
|---|---|---|---|
| `taubench` | [τ-bench](https://github.com/sierra-research/tau-bench) retail: policy, 16 tools, database, tasks | MIT | 30 test tasks (stratified by the number of changes) / 20 |
| `cuad` | [CUAD v1](https://github.com/TheAtticusProject/cuad) commercial contracts, 41 clause kinds | CC BY 4.0 | 25 contracts × 41 = 1,025 questions / the same from other contracts |
| `ragtruth` | [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (the processed copy `wandb/RAGTruth-processed`) | MIT | 600 responses: 100 clean + 100 with unsupported content per kind / 600 |
| `banking77` | [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) | CC BY 4.0 | a stream of 2,000 (20 intents hidden until request 1,000) / 5,967 fit + 1,492 calib |
| `credit` | [Statlog German Credit](https://archive.ics.uci.edu/dataset/144) (UCI) | CC BY 4.0 | 400 new / 600 history applications |
| `bird` | [BIRD mini-dev](https://github.com/bird-bench/mini_dev), SQLite | CC BY-SA 4.0 | 150 questions (50 per difficulty) / 100 |
| `abtbuy` | Abt-Buy product offers (the copy `matchbench/Abt-Buy` on Hugging Face) | **none stated**: the stand downloads it and never redistributes it; only numbers are published | 1,916 pairs / 5,743 + 1,916 |
| `naturalplan` | [NATURAL PLAN](https://github.com/google-deepmind/natural-plan) | Apache-2.0 | 3 × 100 problems / 3 × 50 |
| `nab` | [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) | MIT | 33 real series / 25 (11 artificial + every fourth real one) |

## Run it

From the repository root, with solvi installed (`uv sync`):

```
benchmarks/tasks/fetch.sh                                         # about 0.9 GB of downloads, 1.7 GB on disk
uv run --no-project --with pyarrow python benchmarks/tasks/prepare.py   # data/<task>/prepared/ and data/manifest.json
```

The data goes to `benchmarks/tasks/data/` (or `$STAND_DATA`). The prepared files are the same on every machine:
`data/manifest.json` lists their sha256.

Each task folder has `score.py` (its first lines give the prediction format), `baseline.py` and `solution.py`; every
script writes `runs/<name>.jsonl` and prints the scorer's output. LLM calls go to OpenRouter with the key from
`OPENROUTER_API_KEY`, and every answer is cached by its request in `--cache` (default `benchmarks/tasks/cache/`), so a
rerun costs nothing. A baseline calls the model directly; a solvi solution reaches it through `common/proxy.py`, a local
OpenAI-compatible endpoint over the same cache:

```
export OPENROUTER_API_KEY=...
uv run python benchmarks/tasks/common/proxy.py --cache benchmarks/tasks/cache &     # http://127.0.0.1:8765/<tag>/v1
uv run python benchmarks/tasks/ragtruth/baseline.py
uv run python benchmarks/tasks/ragtruth/solution.py
uv run python benchmarks/tasks/common/llm.py                     # what the cache's ledger says was spent, by task
```

Every script takes `--cache`, `--budget` (dollars; calls stop when the ledger reaches it) and `--offline` (a request
that is not cached is an error, never sent — start the proxy with `--offline` too, and it lists such requests in
`<cache>/misses.jsonl`). Extra packages: `banking77` needs `--with scikit-learn --with scipy`; `prepare.py` needs
`pyarrow`; τ-bench's environment is imported from the downloaded repository.

What a full run from an empty cache costs on OpenRouter (prices of 2026-10-02; gpt-oss-120b $0.037 / $0.17 per million
tokens in / out):

| Task | Baseline | Solution |
|---|---|---|
| taubench | about $0.40 (three quarters of it the simulated customer, `qwen/qwen3.7-plus`) | about $0.40 |
| cuad | $0.10 | about $0.25 (dev for the calibration and eval; one request per question) |
| ragtruth | $0.07 | about $0.25 (two judges on dev and eval) |
| bird | $0.02 | about $0.20 (three candidates per question on dev and eval, retries) |
| abtbuy | $0.05 | $0 (`llm_pair.py`, the LLM comparison arm: about $0.10) |
| naturalplan | $0.29 | $0 |
| banking77, credit, nab | $0 | $0 |

Models answer differently from run to run and providers change, so a run from an empty cache will not give exactly the
numbers above. The replies that produced them are published instead.

## Check the numbers without a key

`packed/replies.jsonl.xz` (2.5 MB) holds every model reply the published run read — 9,458 of them, each once, keyed by
the hash of its request; no request, key or account detail is in it — for every task but Abt-Buy. From it every script
reruns offline, and `results.json` is the one place the published numbers are kept: `stand.py check` fails when a
number in this README, the main README, `docs/benchmarks.md` or `docs/best_practices.md` is not the one in
`results.json`, or (with `--measured`) when a run printed another one.

```
benchmarks/tasks/ci.sh data     # the packed splits; τ-bench, NATURAL PLAN, NAB and BIRD downloads; Abt-Buy downloaded
benchmarks/tasks/ci.sh run      # replies.py unpack, stand.py run (offline), stand.py check: 19 min on 20 cores
uv run python benchmarks/tasks/stand.py check                     # the docs against results.json only: a second
```

The workflow `.github/workflows/stand.yml` does this every week and on demand. Without Abt-Buy's replies its four
steps that read a model are skipped (the baseline, `--repair`, `llm_pair.py` with both reply formats) and their 12
numbers are not compared; its solution, which reads no model, still runs and is. A changed number goes into
`results.json` and the doc together, from a run.

## How each task is solved

- `taubench/solution.py` — the baseline's agent with an `agents.Guard` between the model and the tools: grounding of
  every value in the conversation, policies over what the tools returned, `guard.require_confirmation` on every change,
  `once=True`; a refused call goes back to the model with the reasons (`feedback()`). `harness.py` runs the benchmark's
  environment and simulated customer.
- `cuad/solution.py` — a `Maybe[Span[str]]` question per clause kind through `solvi.core.deciders.llm` with `long="retrieve"`,
  `retrieve_query` (`queries.py`) and `max_len=3000`; a yes / no check of each quoted passage; a trust fact (dev
  reliability × confidence × the check); `System.guarantee(signal="trust", max_risk=0.03)`. `repair.py` cuts an
  almost-literal quote to its longest literal piece before solvi checks it.
- `ragtruth/solution.py` — two wordings of the judge through `solvi.core.deciders.llm`; the answer from the one better on dev, the
  escalation from `Vote(rule="all")` under `act_guard(max_risk=0.10)`; every decision stored.
- `banking77/solution.py` — the baseline's classifier with an act head as a decision part (`classifier.py`),
  `openset.leave_out` on calib, `OpenSetGate.calibrate(max_error=0.05)`, `System.guarantee(promise=gate)`, a stored
  stream. The share answered alone depends on which intents the simulation leaves out together (`SEED`, fixed on calib
  before the stream was run): with the default seed the gate answers 64.0% / 23.2% with 0.9% / 3.0% wrong, still
  within the promise.
- `credit/solution.py` — the policy of `POLICY.md` as one function per rule, `SQLiteStorage`, `replay_all`, `diff`,
  `verify`, `Shadow`, `solvi.check.lint`.
- `bird/solution.py` — `solvi.core.slow.generate` (three queries), `solvi.core.slow.agree` (key: the digest of the rows), hard checks that
  return `Fail` with the reason, one `solvi.core.slow.refine` round, `System.fit` over agreement and the query's shape, and
  `System.guarantee(method="empirical")` — an empirical target, not a promise.
- `abtbuy/solution.py` — typed facts from `pairfacts.py`, `System.fit(select=False)`, `System.guarantee(max_risk=0.01)`,
  `sets.decide_set` with `AtMostOne` per offer.
- `naturalplan/solution.py` — typed facts by rules (`plans.py`), one System of hard checks per kind, `solvi.core.slow.search`
  over a dict of domains (calendar) or a `Tree` with `prune=` (meetings, trips).
- `nab/solution.py` — every point after the warm-up is one `ask` over the trailing window (`facts.py`); the threshold is
  `calibration.conformal_quantile` of the series' own earlier scores; alerts stored, replayed, diffed.

A test runs every scorer on a tiny fixture (`pytest -m stand`, not part of the default run).
